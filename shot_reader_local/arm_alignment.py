"""Register the visible arms to the plate with reversible local rotation offsets."""

import json
from pathlib import Path
import tempfile

import bpy
import numpy as np

from .calibration import camera_intrinsics
from .colour_bake import performer
from .mirror_refine import mirror_basis, reflection
from .reflected_room import mirror_polygon


ARMS = {'Right': (39, 40, 41, 48, 52), 'Left': (75, 76, 77, 84, 88)}
CONSTRAINT = 'Shot Reader Arm Alignment'
job = None
status = ''


def projection(scene, rig, mirror):
    camera = scene.camera
    view = camera.matrix_world.inverted()
    if mirror:
        front = min(point[1] for point in mirror.bound_box)
        view @= reflection(*mirror_basis(mirror.matrix_world, front))
    return view @ rig.matrix_world, camera_intrinsics(scene, camera.data)


def points(view, lens, bones):
    xyz = np.array([view @ bone.head for bone in bones])
    fx, fy, cx, cy = lens
    return xyz[:, :2] / np.maximum(-xyz[:, 2:], .01) * [fx, -fy] + [cx, cy]


def contained(measured, aperture):
    directions = np.roll(aperture, -1, axis=0) - aperture
    delta = measured[:, None, :] - aperture[None]
    cross = directions[:, 0] * delta[:, :, 1] - directions[:, 1] * delta[:, :, 0]
    return (cross >= 0).all(axis=1) | (cross <= 0).all(axis=1)


def create_offsets(scene, side, bones, labels=('Upper Arm', 'Forearm', 'Wrist'), name=CONSTRAINT):
    edits = []
    for label, bone in zip(labels, bones):
        control = bpy.data.objects.new(f'{side} {label} Plate Offset', None)
        scene.collection.objects.link(control)
        control.hide_set(True)
        control.hide_render = True
        rotation = bone.constraints.new('COPY_ROTATION')
        rotation.name = name
        rotation.target = control
        rotation.owner_space = 'LOCAL'
        # The unparented helper's world axes carry the numerical local rotation offset.
        rotation.target_space = 'WORLD'
        rotation.mix_mode = 'AFTER'
        edits.append((bone, rotation, control))
    return edits


def dispose(edits):
    for bone, constraint, control in edits:
        bone.constraints.remove(constraint)
        bpy.data.objects.remove(control, do_unlink=True)


def solve_offsets(apply, project, target, weight, penalty, limits):
    value = np.zeros(len(penalty))
    for _ in range(2):
        base = project()
        jacobian = []
        for axis in range(len(value)):
            sample = value.copy()
            sample[axis] += .012
            apply(sample)
            jacobian.append((project() - base) / .012)
        weighted = np.asarray(jacobian).T * weight[:, None]
        gradient = np.r_[(target - base) * weight, -penalty * value]
        value += np.linalg.lstsq(np.vstack([weighted, np.diag(penalty)]), gradient, rcond=None)[0]
        for bone, limit in enumerate(limits):
            segment = value[bone * 3:bone * 3 + 3]
            segment *= min(1., limit / max(np.linalg.norm(segment), 1e-6))
        apply(value)
    return value


def smooth_offsets(offsets, known):
    frames = np.arange(len(offsets))
    for axis in range(offsets.shape[1]):
        series = np.interp(frames, known, offsets[known, axis])
        offsets[:, axis] = np.convolve(np.pad(series, 2, mode='edge'), np.array([1, 2, 3, 2, 1]) / 9, mode='valid')
    insertion = np.searchsorted(known, frames)
    left = known[np.maximum(0, insertion - 1)]
    right = known[np.minimum(len(known) - 1, insertion)]
    distance = np.minimum(abs(frames - left), abs(frames - right))
    offsets *= np.clip((9. - distance) / 7., 0., 1.)[:, None]


def observations_by_side(context, rig, mirror, values, shoulder_indices):
    scene = context.scene
    frames = np.linspace(0, len(values) - 1, min(9, len(values)), dtype=int)
    shoulders = [rig.pose.bones[f'bone_{indices[0]:03}'] for indices in ARMS.values()]
    guess = []
    for index in frames:
        scene.frame_set(scene.frame_start + int(index))
        view, lens = projection(scene, rig, mirror)
        guess.append(points(view, lens, shoulders))
    guess, wanted = np.asarray(guess), values[frames][:, shoulder_indices, :2]
    return (1, 0) if np.median(np.linalg.norm(guess - wanted[:, ::-1], axis=2)) < np.median(np.linalg.norm(guess - wanted, axis=2)) else (0, 1)


def fit_side(context, rig, mirror, bones, observations, side):
    scene = context.scene
    edits = create_offsets(scene, side, bones)
    count = len(observations)
    offsets = np.zeros((count, 9))
    checks = np.zeros((count, 5), dtype=bool)
    original = np.zeros((count, 5, 2))
    corrected = original.copy()
    penalty = np.repeat([70., 60., 28.], 3)
    limits = [.30, .38, .45]
    width, height = scene.render.resolution_x, scene.render.resolution_y
    depsgraph = context.evaluated_depsgraph_get()
    names = [bone.name for bone in bones]

    def project_rig(view, lens):
        evaluated = rig.evaluated_get(depsgraph).pose.bones
        return points(view, lens, [evaluated[name] for name in names])

    def apply(rotation):
        for index, (_bone, _constraint, control) in enumerate(edits):
            control.rotation_euler = rotation[index * 3:index * 3 + 3]
        context.view_layer.update()

    try:
        for index, track in enumerate(observations):
            scene.frame_set(scene.frame_start + index)
            apply(np.zeros(9))
            view, lens = projection(scene, rig, mirror)
            original[index] = baseline = project_rig(view, lens)
            if mirror:
                aperture = np.array(mirror_polygon(scene, mirror)) * [width, height]
            else:
                aperture = np.array([[0, 0], [width, 0], [width, height], [0, height]])
            checks[index] = (track[:, 2] > .43) & contained(track[:, :2], aperture)
            visible = checks[index]
            if not visible[2] or np.linalg.norm(track[2, :2] - baseline[2]) > height * .115:
                visible[:] = False
                continue
            fit = [2, 3] if visible[3] else [2]
            weight = np.repeat([1., .64][:len(fit)], 2)
            target = track[fit, :2].ravel()
            offsets[index] = solve_offsets(apply, lambda: project_rig(view, lens)[fit].ravel(), target, weight, penalty, limits)
        known = np.flatnonzero(checks[:, 2])
        if len(known) < max(5, count * .12):
            dispose(edits)
            edits.clear()
            return fit_occluded_wrist(context, rig, mirror, bones, observations, side)
        smooth_offsets(offsets, known)
        for index in known:
            scene.frame_set(scene.frame_start + int(index))
            apply(offsets[index])
            view, lens = projection(scene, rig, mirror)
            corrected[index] = project_rig(view, lens)
        before = np.linalg.norm(original - observations[:, :, :2], axis=2)
        after = np.linalg.norm(corrected - observations[:, :, :2], axis=2)
        wrists = checks[:, 2]
        palm = checks[:, 4]
        elbow = checks[:, 1]
        heldout = palm if palm.sum() >= 5 else elbow
        landmark = 'middle knuckle' if palm.sum() >= 5 else 'elbow'
        k = 4 if palm.sum() >= 5 else 1
        if heldout.sum() < 5:
            dispose(edits)
            return None
        pre, post = float(np.median(before[heldout, k])), float(np.median(after[heldout, k]))
        w0, w1 = float(np.median(before[wrists, 2])), float(np.median(after[wrists, 2]))
        if post >= pre * .99 or w1 >= w0 * .97:
            dispose(edits)
            return None
        for index, value in enumerate(offsets):
            for joint, (_bone, _constraint, control) in enumerate(edits):
                control.rotation_euler = value[joint * 3:joint * 3 + 3]
                control.keyframe_insert('rotation_euler', frame=scene.frame_start + index, group='Arm Plate Offset')
        return {'wrist_frames': int(wrists.sum()), 'held_out_frames': int(heldout.sum()), 'held_out': landmark,
                'wrist_before_px': w0, 'wrist_after_px': w1, 'holdout_before_px': pre, 'holdout_after_px': post}
    except Exception:
        dispose(edits)
        raise


def fit_occluded_wrist(context, rig, mirror, bones, observations, side):
    """Use the two visible knuckles when the plate crops out the wrist."""
    scene = context.scene
    count = len(observations)
    graph = context.evaluated_depsgraph_get()
    names = [bone.name for bone in bones[3:5]]
    size = np.array([scene.render.resolution_x, scene.render.resolution_y])
    viewport = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    targets = observations[:, 3:5, :2]
    valid = np.zeros(count, dtype=bool)
    reference = np.zeros_like(targets)

    def projected(view, lens):
        pose = rig.evaluated_get(graph).pose.bones
        return points(view, lens, [pose[name] for name in names])

    for index, observation in enumerate(observations):
        scene.frame_set(scene.frame_start + index)
        rim = np.asarray(mirror_polygon(scene, mirror)) if mirror else viewport
        reference[index] = projected(*projection(scene, rig, mirror))
        valid[index] = bool((observation[3:5, 2] > .43).all() and contained(targets[index], rim * size).all()
                            and np.linalg.norm(targets[index] - reference[index], axis=1).max() < size[1] * .09)
    held_out = np.arange(count) % 5 == 2
    training = np.flatnonzero(valid & ~held_out)
    checking = np.flatnonzero(valid & held_out)
    if len(training) < 8 or len(checking) < 4:
        return None
    edits = create_offsets(scene, side, bones)
    accepted = False
    try:
        offsets = np.zeros((count, 9))

        def apply(value):
            for index, (_, _, handle) in enumerate(edits):
                handle.rotation_euler = value[index * 3:index * 3 + 3]
            context.view_layer.update()

        for index in training:
            scene.frame_set(scene.frame_start + int(index))
            apply(np.zeros(9))
            view, lens = projection(scene, rig, mirror)
            offsets[index] = solve_offsets(apply, lambda: projected(view, lens).ravel(), targets[index].ravel(),
                                           np.repeat([.8, 1.], 2), np.repeat([70., 60., 28.], 3),
                                           [.30, .38, .45])
        smooth_offsets(offsets, training)
        checked = []
        for index in checking:
            scene.frame_set(scene.frame_start + int(index))
            apply(offsets[index])
            checked.append(projected(*projection(scene, rig, mirror)))
        original = np.linalg.norm(reference[checking] - targets[checking], axis=2)
        corrected = np.linalg.norm(np.asarray(checked) - targets[checking], axis=2)
        before, after = [np.median(error, axis=0) for error in (original, corrected)]
        if np.any(after > before * 1.03) or np.mean(after) >= np.mean(before) * .88:
            return None
        for index, value in enumerate(offsets):
            for joint, (_, _, handle) in enumerate(edits):
                handle.rotation_euler = value[joint * 3:joint * 3 + 3]
                handle.keyframe_insert('rotation_euler', frame=scene.frame_start + index, group='Arm Plate Offset')
        accepted = True
        return {'held_out': 'index and middle knuckles', 'held_out_frames': len(checking),
                'wrist_frames': 0, 'holdout_before_px': float(np.mean(before)),
                'holdout_after_px': float(np.mean(after)), 'knuckles_before_px': before.tolist(),
                'knuckles_after_px': after.tolist()}
    finally:
        if not accepted:
            dispose(edits)


def register_arms(context, rig, mirror, observations):
    scene = context.scene
    before_frame, subframe = scene.frame_current, scene.frame_subframe
    values = np.asarray(observations, dtype=np.float64).reshape(-1, 2, 5, 3)
    if len(values) != scene.frame_end - scene.frame_start + 1 or not np.isfinite(values).all():
        raise ValueError('The source and fitted arm observations do not cover the same frames.')
    groups = [[rig.pose.bones[f'bone_{index:03}'] for index in indices] for indices in ARMS.values()]
    try:
        order = observations_by_side(context, rig, mirror, values.reshape(-1, 10, 3), (0, 5))
        report = {}
        for i, (side, bones) in enumerate(zip(ARMS, groups)):
            result = fit_side(context, rig, mirror, bones, values[:, order[i]], side)
            if result:
                report[side] = result
        if not report:
            raise ValueError('Visible arm observations did not support a better checked placement.')
        rig['shot_reader_arm_alignment_report'] = json.dumps(report)
        return report
    finally:
        scene.frame_set(before_frame, subframe=subframe)


def finish():
    global job, status
    from . import refresh

    process, temp, scene, window, camera, mirror, rig = job
    if process.poll() is None:
        return .35
    try:
        path = Path(temp.name)
        if process.returncode:
            log = (path / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Local arm tracking failed.')
        if scene.camera != camera:
            raise ValueError('The active camera changed during arm tracking.')
        values = json.loads((path / 'arms.json').read_text(encoding='utf-8'))
        with bpy.context.temp_override(window=window, scene=scene, view_layer=scene.view_layers[0]):
            result = register_arms(bpy.context, rig, mirror, values)
        status = f'Checked arm alignment applied · {len(result)} side(s)'
    except (OSError, ValueError, RuntimeError, ReferenceError, KeyError) as error:
        status = str(error)
    finally:
        temp.cleanup()
        job = None
        refresh()
    return None


def stop_job():
    global job, status
    if job:
        process, temp, *_ = job
        if process.poll() is None:
            process.kill()
            process.wait()
        temp.cleanup()
        job = None
    if bpy.app.timers.is_registered(finish):
        bpy.app.timers.unregister(finish)
    status = ''


class ShotReaderAlignArms(bpy.types.Operator):
    bl_idname = 'shot_reader.align_arms'
    bl_label = 'Refine Arms from Plate'
    bl_description = 'Align visible wrists or knuckles with reversible rig rotations'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        rig, mesh = performer(context)
        return bool(job is None and context.mode == 'OBJECT' and reference_background(context.scene) and mesh
                    and not rig.get('shot_reader_arm_alignment_report'))

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene = context.scene
        rig, _mesh = performer(context)
        clip = reference_background(scene).clip
        prefs = context.preferences.addons[ADDON].preferences
        model = Path(bpy.path.abspath(prefs.pose_weights))
        mirror = scene.shot_reader.body_mirror if rig['shot_reader_alignment'].startswith('Mirror') else None
        if not model.is_file() or (rig['shot_reader_alignment'].startswith('Mirror') and not mirror):
            self.report({'ERROR'}, 'Set a local ViTPose WholeBody model and select the mirror used to fit the performer.')
            return {'CANCELLED'}
        fps = scene.render.fps / scene.render.fps_base
        origin = clip.frame_start - clip.frame_offset
        request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'model': str(model),
                   'size': [scene.render.resolution_x, scene.render.resolution_y], 'region': list(scene.shot_reader.body_region),
                   'times': [(frame - origin) / fps for frame in range(scene.frame_start, scene.frame_end + 1)]}
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-arms-')
        try:
            (Path(temp.name) / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('arm_observations.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except OSError as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        job = process, temp, scene, context.window, scene.camera, mirror, rig
        status = 'Tracking arms and hands locally…'
        bpy.app.timers.register(finish, first_interval=.35)
        return {'FINISHED'}


def active_constraints(rig):
    return [offset for bone in rig.pose.bones if (offset := bone.constraints.get(CONSTRAINT))]


class ShotReaderToggleArmAlignment(bpy.types.Operator):
    bl_idname = 'shot_reader.toggle_arm_alignment'
    bl_label = 'Toggle Arm Plate Alignment'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        rig, _mesh = performer(context)
        return bool(rig and active_constraints(rig))

    def execute(self, context):
        rig, _mesh = performer(context)
        constraints = active_constraints(rig)
        enabled = not any(item.enabled for item in constraints)
        for item in constraints:
            item.enabled = enabled
        return {'FINISHED'}
