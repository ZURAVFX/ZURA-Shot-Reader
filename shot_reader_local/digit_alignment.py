"""Refine visible fingers against the plate without overwriting the captured hand pose."""

import json
from pathlib import Path
import tempfile

import bpy
import numpy as np

from .arm_alignment import contained, create_offsets, dispose, observations_by_side, points, projection, smooth_offsets, solve_offsets
from .colour_bake import performer
from .reflected_room import mirror_polygon


DIGITS = {'Thumb': (1, 43), 'Index': (5, 48), 'Middle': (9, 52), 'Ring': (13, 56), 'Little': (17, 61)}
CONSTRAINT = 'Shot Reader Finger Alignment'
job = None
status = ''


def fit_digit(context, rig, mirror, side, label, bones, frames, tracks):
    scene = context.scene
    controls = bones[:3]
    labels = ('Base', 'Middle', 'Distal')
    first = 1
    penalty = [18., 43., 18., 14., 38., 14., 8., 28., 8.]
    limits = [.66, .78, .64]
    if label == 'Little':
        controls = [bones[0].parent, *controls]
        labels = ('Metacarpal', *labels)
        first = 0
        penalty = [22., 45., 22., *penalty]
        limits = [.65, *limits]
    edits = create_offsets(scene, f'{side} {label}', controls, labels, CONSTRAINT)
    names = [bone.name for bone in bones]
    depsgraph = context.evaluated_depsgraph_get()
    penalty = np.asarray(penalty)
    neutral = np.zeros(len(penalty))
    offsets = np.zeros((len(tracks), len(penalty)))
    measured = tracks[:, first:, :2]
    original = np.zeros_like(measured)
    corrected = np.zeros_like(measured)
    visible = np.zeros(len(tracks), dtype=bool)
    checked = visible.copy()
    tip_fit = visible.copy()
    image_limit = scene.render.resolution_y * .045

    def apply(value):
        for joint, (_bone, _detail, control) in enumerate(edits):
            control.rotation_euler = value[joint * 3:joint * 3 + 3]
        context.view_layer.update()

    def predict(view, lens):
        pose = rig.evaluated_get(depsgraph).pose.bones
        return points(view, lens, [pose[name] for name in names[first:]])

    try:
        for index, track in enumerate(tracks):
            if min(track[first:3, 2]) < .55 or not frames[index]['wrist']:
                continue
            if not contained(track[first:3, :2], frames[index]['aperture']).all():
                continue
            scene.frame_set(scene.frame_start + index)
            apply(neutral)
            view, lens = projection(scene, rig, mirror)
            original[index] = predicted = predict(view, lens)
            if np.linalg.norm(predicted[:-1] - track[first:3, :2], axis=1).max() > image_limit:
                continue
            visible[index] = True
            reliable_tip = track[3, 2] > .55 and contained(track[3:, :2], frames[index]['aperture'])[0]
            checked[index] = reliable_tip and index % 3 == 1
            tip_fit[index] = reliable_tip and not checked[index]
            end = 4 if tip_fit[index] else 3
            joints = end - first
            weight = np.repeat([.54, .75, 1., .82][first:end], 2)
            offsets[index] = solve_offsets(apply, lambda: predict(view, lens)[:joints].ravel(),
                                           track[first:end, :2].ravel(), weight, penalty, limits)
        known = np.flatnonzero(visible)
        if len(known) < 8 or checked.sum() < 5:
            dispose(edits)
            return None
        fitted_tips = np.flatnonzero(tip_fit)
        smooth_offsets(offsets[:, :-3], known)
        if len(fitted_tips):
            # Preserve distal motion on the observed frames and interpolate it into the withheld frames.
            smooth_offsets(offsets[:, -3:], fitted_tips)
        for index in known:
            scene.frame_set(scene.frame_start + int(index))
            apply(offsets[index])
            view, lens = projection(scene, rig, mirror)
            corrected[index] = predict(view, lens)
        error0 = np.linalg.norm(original - measured, axis=2)
        error1 = np.linalg.norm(corrected - measured, axis=2)
        p0, p1 = float(np.median(error0[visible, :-1])), float(np.median(error1[visible, :-1]))
        tip0, tip1 = float(np.median(error0[checked, -1])), float(np.median(error1[checked, -1]))
        if p1 >= p0 * .97 or tip1 >= tip0 * .97:
            dispose(edits)
            return None
        for frame, value in enumerate(offsets, scene.frame_start):
            for joint, (_bone, _detail, control) in enumerate(edits):
                control.rotation_euler = value[joint * 3:joint * 3 + 3]
                control.keyframe_insert('rotation_euler', frame=frame, group='Finger Plate Alignment')
        return {'fit_frames': len(known), 'fitted_fingertips': len(fitted_tips), 'held_out_frames': int(checked.sum()), 'joints_before_px': p0,
                'joints_after_px': p1, 'tip_before_px': tip0, 'tip_after_px': tip1}
    except Exception:
        dispose(edits)
        raise


def register_digits(context, rig, mirror, values):
    scene = context.scene
    current, subframe = scene.frame_current, scene.frame_subframe
    values = np.asarray(values, dtype=np.float64)
    count = scene.frame_end - scene.frame_start + 1
    if values.shape != (count, 44, 3) or not np.isfinite(values).all():
        raise ValueError('The hand observations do not match the scene frame range.')
    width, height = scene.render.resolution_x, scene.render.resolution_y
    report = {}
    try:
        order = observations_by_side(context, rig, mirror, values, (0, 1))
        for side, wrist, bias, group in zip(('Right', 'Left'), (41, 77), (0, 36), order):
            hand = values[:, 2 + group * 21:2 + (group + 1) * 21]
            frames = []
            wrist_bone = rig.pose.bones[f'bone_{wrist:03}']
            for index, observed in enumerate(hand[:, 0]):
                scene.frame_set(scene.frame_start + index)
                view, lens = projection(scene, rig, mirror)
                aperture = (np.array(mirror_polygon(scene, mirror)) * [width, height] if mirror else
                            np.array([[0, 0], [width, 0], [width, height], [0, height]]))
                close = np.linalg.norm(points(view, lens, [wrist_bone])[0] - observed[:2]) < height * .045
                frames.append({'wrist': observed[2] > .48 and close and contained(observed[None, :2], aperture)[0],
                               'aperture': aperture})
            for label, (channel, first) in DIGITS.items():
                bones = [rig.pose.bones[f'bone_{first + bias + i:03}'] for i in range(4)]
                result = fit_digit(context, rig, mirror, side, label, bones, frames, hand[:, channel:channel + 4])
                if result:
                    report.setdefault(side, {})[label] = result
        if not report:
            raise ValueError('The available finger observations did not improve the held-out fingertips.')
        rig['shot_reader_digit_alignment_report'] = json.dumps(report)
        return report
    finally:
        scene.frame_set(current, subframe=subframe)


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
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Local finger tracking failed.')
        if scene.camera != camera:
            raise ValueError('The active camera changed during finger tracking.')
        values = json.loads((path / 'arms.json').read_text(encoding='utf-8'))
        with bpy.context.temp_override(window=window, scene=scene, view_layer=scene.view_layers[0]):
            report = register_digits(bpy.context, rig, mirror, values)
        status = f"Checked hand alignment applied · {sum(map(len, report.values()))} fingers"
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


class ShotReaderAlignFingers(bpy.types.Operator):
    bl_idname = 'shot_reader.align_fingers'
    bl_label = 'Refine Fingers from Plate'
    bl_description = 'Refine the captured hand pose and check it against withheld fingertip observations'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        rig, mesh = performer(context)
        return bool(job is None and context.mode == 'OBJECT' and reference_background(context.scene) and mesh
                    and not rig.get('shot_reader_digit_alignment_report'))

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
        request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'model': str(model), 'fingers': True,
                   'size': [scene.render.resolution_x, scene.render.resolution_y], 'region': list(scene.shot_reader.body_region),
                   'times': [(frame - origin) / fps for frame in range(scene.frame_start, scene.frame_end + 1)]}
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-fingers-')
        try:
            (Path(temp.name) / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('arm_observations.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except OSError as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        job = process, temp, scene, context.window, scene.camera, mirror, rig
        status = 'Tracking fingers locally…'
        bpy.app.timers.register(finish, first_interval=.35)
        return {'FINISHED'}


def active_constraints(rig):
    return [offset for bone in rig.pose.bones if (offset := bone.constraints.get(CONSTRAINT))]


class ShotReaderToggleDigitAlignment(bpy.types.Operator):
    bl_idname = 'shot_reader.toggle_digit_alignment'
    bl_label = 'Toggle Finger Plate Alignment'
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
