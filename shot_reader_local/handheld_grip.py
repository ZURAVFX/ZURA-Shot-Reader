"""Fit reversible arm corrections to plausible contact with a tracked prop."""

import json

import numpy as np
from mathutils.bvhtree import BVHTree

from .arm_alignment import ARMS, create_offsets, dispose, smooth_offsets, solve_offsets
from .digit_alignment import DIGITS


CONSTRAINT = 'Shot Reader Handheld Contact'


def fit_grip(context, rig, prop, side, frames):
    scene = context.scene
    graph = context.evaluated_depsgraph_get()
    old, subframe = scene.frame_current, scene.frame_subframe
    bones = rig.pose.bones
    indices = ARMS[side]
    bias = 0 if side == 'Right' else 36
    digits = [(f'bone_{first + bias + 2:03}', f'bone_{first + bias + 3:03}')
              for _channel, first in DIGITS.values()]
    faces = [tuple(poly.vertices) for poly in prop.data.polygons]
    prior_distances = []
    edits = []
    accepted = False
    train = np.flatnonzero(np.arange(len(frames)) % 4 != 2)
    test = np.flatnonzero(np.arange(len(frames)) % 4 == 2)

    def vertices():
        return BVHTree.FromPolygons([prop.matrix_world @ vertex.co for vertex in prop.data.vertices], faces)

    def positions(names):
        evaluated = rig.evaluated_get(graph)
        return [evaluated.matrix_world @ evaluated.pose.bones[name].head for name in names]

    def distances(surface, names):
        return np.array([surface.find_nearest(point)[3] for point in positions(names)])

    def penetration(surface, names):
        depths = []
        for point in positions(names):
            nearest, normal, _, _ = surface.find_nearest(point)
            depths.append(max(0., -(point - nearest).dot(normal)))
        return max(depths)

    try:
        for index in train[::max(1, len(train) // 8)]:
            scene.frame_set(frames[index])
            prior_distances.append(distances(vertices(), [tip for _joint, tip in digits]))
        median = np.median(prior_distances, axis=0)
        order = np.argsort(median)[:2]
        if median[order[0]] > .10 or median[order[1]] > .13 or median[order[0]] < .009:
            return None
        joints, tips = zip(*(digits[i] for i in order))
        elbow_name = f'bone_{indices[1]:03}'
        controls = [bones[f'bone_{i:03}'] for i in indices[:3]]
        edits = create_offsets(scene, side, controls, name=CONSTRAINT)
        offsets = np.zeros((len(frames), 9))
        before, after = [], []

        def apply(value):
            for index, (_bone, _constraint, control) in enumerate(edits):
                control.rotation_euler = value[index * 3:index * 3 + 3]
            context.view_layer.update()

        for index in train:
            scene.frame_set(frames[index])
            apply(np.zeros(9))
            tree = vertices()
            inverse = prop.matrix_world.inverted()
            targets = []
            for point in positions(joints):
                nearest, normal, _, _ = tree.find_nearest(point)
                goal = inverse @ (nearest + normal * .011)
                # The fitted tube outline represents the visible surface. Keep the inferred grip behind it.
                goal.y = -abs(goal.y)
                targets.append(prop.matrix_world @ goal)
            elbow = positions([elbow_name])[0]
            offsets[index] = solve_offsets(apply, lambda: np.asarray(positions((*joints, elbow_name))).ravel() * 1000.,
                                           np.asarray([*targets, elbow]).ravel() * 1000.,
                                           np.repeat([1., .85, .42], 3), np.repeat([72., 58., 30.], 3),
                                           [.24, .35, .42])
        smooth_offsets(offsets, train)
        offsets *= .75
        # Use the fitted solid to push penetrating fingertips out of the tube.
        # This step constrains geometry only. It does not fit the withheld plate observations.
        limited = 0
        for index, frame in enumerate(frames):
            scene.frame_set(frame)
            tree = vertices()
            base = offsets[index].copy()
            apply(base)
            if penetration(tree, tips) <= .004:
                continue
            goal = []
            for point in positions(tips):
                nearest, normal, _, _ = tree.find_nearest(point)
                goal.append(nearest + normal * .003 if (point - nearest).dot(normal) < 0. else point)
            elbow = positions([elbow_name])[0]
            change = solve_offsets(lambda value: apply(base + value),
                                   lambda: np.asarray(positions((*tips, elbow_name))).ravel() * 1000.,
                                   np.asarray([*goal, elbow]).ravel() * 1000.,
                                   np.repeat([1., 1., .65], 3), np.repeat([45., 38., 25.], 3),
                                   [.07, .10, .14])
            offsets[index] += change
            limited += 1
        elbow_shifts, penetrations = [], []
        for index in test:
            scene.frame_set(frames[index])
            tree = vertices()
            apply(np.zeros(9))
            before.append(distances(tree, tips))
            elbow = positions([elbow_name])[0]
            apply(offsets[index])
            after.append(distances(tree, tips))
            elbow_shifts.append((positions([elbow_name])[0] - elbow).length)
            penetrations.append(penetration(tree, tips))
        error0, error1 = float(np.median(before)), float(np.median(after))
        if error1 >= min(.05, error0 * .77) or np.median(elbow_shifts) > .055 or max(penetrations) > .005:
            return None
        for index, frame in enumerate(frames):
            scene.frame_set(frame)
            apply(offsets[index])
            penetrations.append(penetration(vertices(), tips))
        if max(penetrations) > .005:
            return None
        for frame, value in zip(frames, offsets):
            for index, (_bone, _constraint, control) in enumerate(edits):
                control.rotation_euler = value[index * 3:index * 3 + 3]
                control.keyframe_insert('rotation_euler', frame=frame, group='Handheld Contact')
        report = {'held_out_frames': len(test), 'fingertip_gap_before_mm': error0 * 1000,
                  'fingertip_gap_after_mm': error1 * 1000, 'median_elbow_shift_mm': float(np.median(elbow_shifts)) * 1000,
                  'hand': side, 'fingers': [list(DIGITS)[index] for index in order],
                  'maximum_fingertip_penetration_mm': max(penetrations) * 1000,
                  'collision_limited_frames': limited}
        prop['shot_reader_grip_report'] = json.dumps(report)
        accepted = True
        return report
    finally:
        if edits and not accepted:
            dispose(edits)
        scene.frame_set(old, subframe=subframe)
