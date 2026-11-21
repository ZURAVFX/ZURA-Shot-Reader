"""Register the deforming head to local plate observations without replacing the pose."""

import json
from pathlib import Path
import tempfile

import bpy
from mathutils import Vector
import numpy as np

from .calibration import camera_intrinsics
from .colour_bake import performer
from .mirror_refine import mirror_basis, reflection


CONSTRAINT = 'Shot Reader Head Alignment'
job = None
status = ''


def face_vertices(rig, body):
    saved = rig.get('shot_reader_face_vertices')
    if saved is not None:
        ids = list(saved)
        if len(ids) == 5 and len(set(ids)) == 5 and 0 <= min(ids) and max(ids) < len(body.data.vertices):
            return ids
    # This is the verified vertex ordering for older SAM 3D Body exports with the full MHR mesh.
    if len(body.data.vertices) == 18439 and str(rig.get('shot_reader_source', '')).startswith('SAM 3D Body'):
        return [2767, 1057, 3943, 1734, 4620]
    return None


def project(scene, mesh, ids, mirror):
    camera = scene.camera
    view = camera.matrix_world.inverted()
    if mirror:
        front = min(point[1] for point in mirror.bound_box)
        view @= reflection(*mirror_basis(mirror.matrix_world, front))
    matrix = view @ mesh.matrix_world
    xyz = np.array([matrix @ mesh.data.vertices[index].co for index in ids])
    fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
    if np.any(xyz[:, 2] >= -.01):
        raise ValueError('The fitted head falls behind the plate camera.')
    return xyz[:, :2] / -xyz[:, 2:] * [fx, -fy] + [cx, cy]


def register_head(context, rig, body, mirror, observations):
    scene = context.scene
    ids = face_vertices(rig, body)
    if ids is None:
        raise ValueError('This mesh has no known face landmark mapping. Refit the SAM 3D Body performance.')
    count = len(observations)
    valid = np.array([item is not None for item in observations])
    if valid.sum() < max(5, count * .25):
        raise ValueError('Too few usable facial observations for a checked correction.')
    if count != scene.frame_end - scene.frame_start + 1:
        raise ValueError('The scene duration changed during face tracking.')
    measured = np.array([item if item is not None else np.zeros((5, 2)) for item in observations])
    head = rig.pose.bones['bone_113']
    handle = bpy.data.objects.new('Head Plate Offset', None)
    scene.collection.objects.link(handle)
    handle.empty_display_size = .025
    handle.hide_set(True)
    handle.hide_render = True
    align = head.constraints.new('COPY_LOCATION')
    align.name = CONSTRAINT
    align.target = handle
    align.owner_space = align.target_space = 'LOCAL'
    align.use_offset = True
    old_frame, subframe = scene.frame_current, scene.frame_subframe
    depsgraph = context.evaluated_depsgraph_get()
    offsets = np.zeros((count, 3))
    before = np.zeros((count, 5, 2))
    after = before.copy()
    order = {}
    weights = np.array([.68, .16, .16])
    try:
        for index in np.flatnonzero(valid):
            scene.frame_set(scene.frame_start + int(index))
            handle.location = (0., 0., 0.)
            context.view_layer.update()
            points = project(scene, body.evaluated_get(depsgraph), ids, mirror)
            current = [0, *(1 + np.argsort(points[1:3, 0])), *(3 + np.argsort(points[3:, 0]))]
            order[index] = current
            before[index] = base = points[current]
            delta = (measured[index, :3] - base[:3]).T @ weights
            jacobian = []
            for axis in range(3):
                trial = Vector((0., 0., 0.))
                trial[axis] = .005
                handle.location = trial
                context.view_layer.update()
                sample = project(scene, body.evaluated_get(depsgraph), ids, mirror)[current]
                jacobian.append(((sample[:3] - base[:3]).T @ weights) / .005)
            jacobian = np.asarray(jacobian).T
            move = np.linalg.lstsq(jacobian, delta, rcond=.05)[0]
            offsets[index] = move * min(1., .045 / max(1e-8, np.linalg.norm(move)))
        known = np.flatnonzero(valid)
        for axis in range(3):
            series = np.interp(np.arange(count), known, offsets[valid, axis])
            offsets[:, axis] = np.convolve(np.pad(series, 2, mode='edge'), [1/9, 2/9, 3/9, 2/9, 1/9], mode='valid')
        distance = np.min(abs(np.arange(count)[:, None] - known), axis=1)
        offsets *= np.clip((10. - distance) / 8., 0., 1.)[:, None]
        for index in known:
            scene.frame_set(scene.frame_start + int(index))
            handle.location = offsets[index]
            context.view_layer.update()
            after[index] = project(scene, body.evaluated_get(depsgraph), ids, mirror)[order[index]]

        original_error = np.linalg.norm(before[valid] - measured[valid], axis=2)
        corrected_error = np.linalg.norm(after[valid] - measured[valid], axis=2)
        initial = float(np.median(original_error[:, 3:].mean(axis=1)))
        final = float(np.median(corrected_error[:, 3:].mean(axis=1)))
        if not np.isfinite(final) or final >= initial * .97:
            raise ValueError(f'The head correction did not improve the held-out eyes ({initial:.1f} to {final:.1f} px).')
        for index, location in enumerate(offsets):
            handle.location = location
            handle.keyframe_insert('location', frame=scene.frame_start + index, group='Head Plate Offset')
        report = {'observed_frames': len(known), 'fit_landmarks': ['nose', 'brows'], 'held_out': ['both eye centres'],
                  'eyes_before_median_px': initial, 'eyes_after_median_px': final,
                  'nose_before_median_px': float(np.median(original_error[:, 0])),
                  'nose_after_median_px': float(np.median(corrected_error[:, 0]))}
        rig['shot_reader_head_alignment_report'] = json.dumps(report)
    except Exception:
        head.constraints.remove(align)
        bpy.data.objects.remove(handle, do_unlink=True)
        raise
    finally:
        scene.frame_set(old_frame, subframe=subframe)
    return report


def finish():
    global job, status
    from . import refresh

    process, temp, scene, window, camera, mirror, rig, mesh = job
    if process.poll() is None:
        return .35
    try:
        directory = Path(temp.name)
        if process.returncode:
            log = (directory / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Head tracking failed.')
        if scene.camera != camera:
            raise ValueError('The active camera changed during face tracking.')
        observations = json.loads((directory / 'face.json').read_text(encoding='utf-8'))
        with bpy.context.temp_override(window=window, scene=scene, view_layer=scene.view_layers[0]):
            report = register_head(bpy.context, rig, mesh, mirror, observations)
        status = f"Held-out eye error · {report['eyes_before_median_px']:.1f} → {report['eyes_after_median_px']:.1f} px"
    except (OSError, KeyError, ValueError, RuntimeError, ReferenceError) as error:
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


class ShotReaderAlignHead(bpy.types.Operator):
    bl_idname = 'shot_reader.align_head'
    bl_label = 'Refine Head Position from Plate'
    bl_description = 'Register head position from the nose and brows in the plate while checking both eyes'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        rig, mesh = performer(context)
        return bool(job is None and context.mode == 'OBJECT' and reference_background(context.scene) and mesh
                    and CONSTRAINT not in rig.pose.bones['bone_113'].constraints)

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene = context.scene
        rig, mesh = performer(context)
        camera = scene.camera
        clip = reference_background(scene).clip
        prefs = context.preferences.addons[ADDON].preferences
        checkpoint = Path(bpy.path.abspath(prefs.face_weights))
        if not checkpoint.is_file():
            self.report({'ERROR'}, 'Set the local Face Landmarker checkpoint in Shot Reader preferences.')
            return {'CANCELLED'}
        mirror = scene.shot_reader.body_mirror if rig['shot_reader_alignment'].startswith('Mirror') else None
        if rig['shot_reader_alignment'].startswith('Mirror') and not mirror:
            self.report({'ERROR'}, 'Select the mirror used to fit this performer.')
            return {'CANCELLED'}
        fps = scene.render.fps / scene.render.fps_base
        origin = clip.frame_start - clip.frame_offset
        request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library), 'model': str(checkpoint),
                   'size': [scene.render.resolution_x, scene.render.resolution_y], 'region': list(scene.shot_reader.body_region),
                   'times': [(frame - origin) / fps for frame in range(scene.frame_start, scene.frame_end + 1)]}
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-head-')
        try:
            (Path(temp.name) / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('head_observations.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except OSError as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        status = 'Tracking face landmarks locally…'
        job = process, temp, scene, context.window, camera, mirror, rig, mesh
        bpy.app.timers.register(finish, first_interval=.35)
        return {'FINISHED'}
