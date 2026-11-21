"""Blender integration for tracking the plate camera against a reconstructed set."""

import json
from pathlib import Path
import tempfile

import bpy
from mathutils import Matrix
import numpy as np

from .calibration import camera_intrinsics


job = None
status = ""


def apply_camera_path(scene, camera, origin, lens, result):
    state = scene.shot_reader_calibration
    if result['status'] == 'tracked':
        tracked = bpy.data.objects.new(f"{camera.name} · Plate Track", camera.data.copy())
        tracked.data.animation_data_clear()
        for name, value in lens.items():
            setattr(tracked.data, name, value)
        scene.collection.objects.link(tracked)
        tracked.rotation_mode = 'QUATERNION'
        previous = None
        for frame, pose in enumerate(result['frames'], scene.frame_start):
            matrix = Matrix(pose['rotation']).to_4x4()
            matrix.translation = pose['location']
            world = origin @ matrix
            orientation = world.to_quaternion()
            if previous is not None:
                orientation.make_compatible(previous)
            previous = orientation
            tracked.location = world.translation
            tracked.rotation_quaternion = orientation
            tracked.keyframe_insert('location', frame=frame, group='Plate Camera')
            tracked.keyframe_insert('rotation_quaternion', frame=frame, group='Plate Camera')
        animation = tracked.animation_data
        for layer in animation.action.layers:
            for strip in layer.strips:
                channels = strip.channelbag(animation.action_slot)
                if channels:
                    for curve in channels.fcurves:
                        for point in curve.keyframe_points:
                            point.interpolation = 'LINEAR'
        state.track_previous_camera = camera
        scene.camera = tracked
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
    report = bpy.data.texts.new(f"{scene.name} · Camera Track.json")
    report.write(json.dumps(result, indent=2))
    state.track_report = report
    if result['status'] == 'near-static':
        state.track_summary = f"Locked camera · {result['motion_p90_px']:.2f} px set motion (90th percentile)"
    else:
        state.track_summary = (f"Held-out set tracks · {result['fixed_median_px']:.1f} → "
                               f"{result['heldout_median_px']:.2f} px")


def finish():
    global job, status
    from . import refresh

    process, temp, scene, camera, anchor, origin, lens = job
    if process.poll() is None:
        return .35
    try:
        path = Path(temp.name)
        if process.returncode:
            log = (path / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
            raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Camera tracking failed.')
        if scene.camera != camera or scene.shot_reader_room.room != anchor:
            raise ValueError('The camera or reference set changed during tracking.')
        tracked = json.loads((path / 'camera.json').read_text(encoding='utf-8'))
        if len(tracked['frames']) != scene.frame_end - scene.frame_start + 1:
            raise ValueError('The scene frame range changed during tracking.')
        apply_camera_path(scene, camera, origin, lens, tracked)
        status = ''
    except (OSError, RuntimeError, ValueError, ReferenceError, KeyError) as error:
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


class ShotReaderTrackCamera(bpy.types.Operator):
    bl_idname = 'shot_reader.track_camera'
    bl_label = 'Track Camera from Visible Set'
    bl_description = 'Track static plate texture on the reconstructed set to animate a copy of the active camera'

    @classmethod
    def poll(cls, context):
        from . import reference_background

        scene = context.scene
        return bool(job is None and context.mode == 'OBJECT' and scene.camera and scene.camera.data.type == 'PERSP'
                    and scene.shot_reader_room.room and scene.shot_reader_room.room.type == 'MESH'
                    and reference_background(scene))

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment

        global job, status
        scene, prefs = context.scene, context.preferences.addons[ADDON].preferences
        camera, room = scene.camera, scene.shot_reader_room.room
        clip = reference_background(scene).clip
        seed = room.get('shot_reader_plate_frame', scene.frame_start)
        if not scene.frame_start <= seed <= scene.frame_end or scene.frame_end <= scene.frame_start:
            self.report({'ERROR'}, 'Include the set reconstruction frame in a scene range of at least two frames.')
            return {'CANCELLED'}
        if scene.render.pixel_aspect_x != scene.render.pixel_aspect_y:
            self.report({'ERROR'}, 'Camera tracking requires square pixels.')
            return {'CANCELLED'}
        old_frame, old_subframe = scene.frame_current, scene.frame_subframe
        temp = tempfile.TemporaryDirectory(prefix='shot-reader-camera-')
        try:
            scene.frame_set(seed)
            origin = camera.matrix_world.normalized()
            lens = {name: getattr(camera.data, name) for name in
                    ('sensor_fit', 'sensor_width', 'sensor_height', 'lens', 'shift_x', 'shift_y')}
            room_to_camera = np.asarray(origin.inverted() @ room.matrix_world)
            vertices = np.empty(3 * len(room.data.vertices), dtype=np.float32)
            room.data.vertices.foreach_get('co', vertices)
            vertices = vertices.reshape(-1, 3) @ room_to_camera[:3, :3].T + room_to_camera[:3, 3]
            fps = scene.render.fps / scene.render.fps_base
            request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library),
                       'frame_start': scene.frame_start, 'seed': seed - scene.frame_start,
                       'size': [scene.render.resolution_x, scene.render.resolution_y],
                       'times': [max(0., (frame - clip.frame_start + clip.frame_offset) / fps)
                                 for frame in range(scene.frame_start, scene.frame_end + 1)],
                       'intrinsics': camera_intrinsics(scene, camera.data), 'threshold': scene.shot_reader_calibration.threshold}
            directory = Path(temp.name)
            np.save(directory / 'set.npy', vertices * [1., -1., -1.], allow_pickle=False)
            (directory / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            command = [bpy.path.abspath(prefs.python), str(Path(__file__).with_name('camera_track.py')), temp.name]
            process = start_worker(command, worker_environment(prefs), temp)
        except (OSError, RuntimeError, ValueError) as error:
            temp.cleanup()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            scene.frame_set(old_frame, subframe=old_subframe)
        job = process, temp, scene, camera, room, origin, lens
        status = 'Tracking static set features…'
        bpy.app.timers.register(finish, first_interval=.35)
        return {'FINISHED'}


class ShotReaderRestoreTrackedCamera(bpy.types.Operator):
    bl_idname = 'shot_reader.restore_tracked_camera'
    bl_label = 'Use Camera Before Tracking'
    bl_options = {'UNDO'}

    def execute(self, context):
        scene = context.scene
        camera = scene.shot_reader_calibration.track_previous_camera
        if not camera or camera.name not in scene.objects:
            return {'CANCELLED'}
        scene.camera = camera
        return {'FINISHED'}


CLASSES = (ShotReaderTrackCamera, ShotReaderRestoreTrackedCamera)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
