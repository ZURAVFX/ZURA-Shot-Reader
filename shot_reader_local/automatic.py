"""Coordinate the local reconstruction workers without an external agent."""

import json
from math import atan, degrees, isclose
from pathlib import Path
import tempfile

import bpy
from mathutils import Matrix, Vector

from . import arm_alignment, camera_motion, colour_bake, digit_alignment, hair_reconstruction, handheld, head_alignment, mirror_refine, performance, reflected_room, room
from .calibration import camera_intrinsics


run = None
status = ''


def visible_set_operator(scene):
    track = scene.shot_reader_calibration.track_report
    if track:
        result = json.loads(track.as_string())
        if result.get('status') == 'tracked':
            locations = [Vector(pose['location']) for pose in result['frames']]
            if locations and max((position - locations[0]).length for position in locations) >= .005:
                return bpy.ops.shot_reader.build_parallax_room
    return bpy.ops.shot_reader.build_clean_room


def steps(scene, *, performer):
    from . import ADDON

    state, settings = scene.shot_reader_room, scene.shot_reader
    prior_track = scene.shot_reader_calibration.track_report
    prior_refinement, prior_room, prior_wall = settings.mirror_refinement_report, state.room, state.reflected_wall
    result = []
    if not state.room:
        result.append(('Reconstructing the reference depth', bpy.ops.shot_reader.build_room, room,
                       lambda: state.room is not None, False))
    if state.room or result:
        result.append(('Tracking the camera', bpy.ops.shot_reader.track_camera, camera_motion,
                       lambda: scene.shot_reader_calibration.track_report != prior_track, False))
    visible_set = ('Recovering the visible set', visible_set_operator, room,
                   lambda: state.room and state.room != prior_room and
                   (state.room.get('shot_reader_clean_set') or state.room.get('shot_reader_parallax_set')), False)
    if not performer:
        return result + [visible_set]
    result.append(('Fitting the animated performer', bpy.ops.shot_reader.fit_performance, performance,
                   lambda: any(obj.get('shot_reader_performance_rig') for obj in scene.objects), True))
    if settings.body_mirror:
        result.append(('Refining the mirror', bpy.ops.shot_reader.track_mirror_foreground, mirror_refine,
                       lambda: settings.mirror_refinement_report != prior_refinement, False))
    result.append(visible_set)
    prefs = bpy.context.preferences.addons[ADDON].preferences
    if Path(bpy.path.abspath(prefs.face_weights)).is_file():
        result.append(('Refining the head from the plate', bpy.ops.shot_reader.align_head, head_alignment,
                       lambda: any(obj.get('shot_reader_head_alignment_report') for obj in scene.objects), False))
    if Path(bpy.path.abspath(prefs.pose_weights)).is_file():
        result.append(('Refining the arms from the plate', bpy.ops.shot_reader.align_arms, arm_alignment,
                       lambda: any(obj.get('shot_reader_arm_alignment_report') for obj in scene.objects), False))
        result.append(('Refining the fingers from the plate', bpy.ops.shot_reader.align_fingers, digit_alignment,
                       lambda: any(obj.get('shot_reader_digit_alignment_report') for obj in scene.objects), False))
    if Path(bpy.path.abspath(prefs.hair_weights)).is_file():
        result.append(('Refining the silhouette and reconstructing hair', bpy.ops.shot_reader.reconstruct_hair, hair_reconstruction,
                       lambda: any(obj.get('shot_reader_hair_report') for obj in scene.objects), False))
    if scene.frame_end - scene.frame_start >= 12:
        result.append(('Checking for a handheld item', bpy.ops.shot_reader.track_handheld, handheld,
                       lambda: handheld.status == handheld.NO_ITEM or
                       any(obj.get('shot_reader_handheld_report') for obj in scene.objects), False))
    result.append(('Baking the performer colour', bpy.ops.shot_reader.bake_performer_colour, colour_bake,
                   lambda: any(obj.get('shot_reader_colour_report') for obj in scene.objects), False))
    if settings.body_mirror:
        result.append(('Recovering the reflected set', bpy.ops.shot_reader.build_reflected_room, reflected_room,
                       lambda: state.reflected_wall and state.reflected_wall != prior_wall, False))
    return result


def launch_detector(scene, prefs, calibrate):
    from . import reference_background, start_worker, worker_environment

    settings = scene.shot_reader
    reference = reference_background(scene)
    if reference is None:
        raise ValueError('No shot with a linked plate was produced by the local blockout.')
    clip = reference.clip
    previous, subframe = scene.frame_current, scene.frame_subframe
    temp = tempfile.TemporaryDirectory(prefix='shot-reader-auto-')
    try:
        scene.frame_set(scene.frame_start)
        mirrors = [obj for obj in scene.objects if obj.type == 'MESH' and obj.get('shot_reader_class') == 'mirror']
        if not settings.body_mirror and len(mirrors) == 1:
            settings.body_mirror = mirrors[0]
        mirror = settings.body_mirror
        fps = scene.render.fps / scene.render.fps_base
        origin = clip.frame_start - clip.frame_offset
        interval = max(1, (scene.frame_end - scene.frame_start + 5) // 6)
        frames = list(range(scene.frame_start, scene.frame_end + 1, interval))
        request = {'clip': bpy.path.abspath(clip.filepath, library=clip.library),
                   'weights': bpy.path.abspath(prefs.person_weights),
                   'size': [scene.render.resolution_x, scene.render.resolution_y],
                   'frames': frames, 'times': [(frame - origin) / fps for frame in frames],
                   'mirror': reflected_room.mirror_polygon(scene, mirror, required=False) if mirror else None}
        if mirror and calibrate:
            low, high = tuple(map(min, zip(*mirror.bound_box))), tuple(map(max, zip(*mirror.bound_box)))
            world = [list(mirror.matrix_world @ Vector((x, low[1], z)))
                     for x, z in ((low[0], high[2]), (high[0], high[2]), (high[0], low[2]), (low[0], low[2]))]
            request['calibration'] = {'world': world, 'intrinsics': camera_intrinsics(scene, scene.camera.data),
                                      'refine_focal': settings.match_lens,
                                      'matte': bpy.path.abspath(prefs.matte_weights)}
        (Path(temp.name) / 'request.json').write_text(json.dumps(request), encoding='utf-8')
        process = start_worker([bpy.path.abspath(prefs.body_python), str(Path(__file__).with_name('auto_subject.py')),
                                temp.name], worker_environment(prefs), temp)
        return process, temp
    except (OSError, ValueError, RuntimeError):
        temp.cleanup()
        raise
    finally:
        scene.frame_set(previous, subframe=subframe)


def apply_initial_alignment(scene, found):
    from . import update_lens_sheet

    settings = scene.shot_reader
    calibrated = found.get('calibration')
    exclusions = []
    if calibrated:
        scene.frame_set(calibrated.get('frame', scene.frame_start))
        camera = scene.camera
        camera.animation_data_clear()
        matrix = Matrix(calibrated['rotation']).to_4x4()
        matrix.translation = Vector(calibrated['position'])
        camera.matrix_world = matrix
        focal_scale = calibrated.get('focal_scale', 1.)
        calibrated['starting_lens_mm'] = camera.data.lens
        camera.data.lens *= focal_scale
        calibrated['matched_lens_mm'] = camera.data.lens
        if not isclose(focal_scale, 1.):
            normalized = camera_intrinsics(scene, camera.data)[0] / scene.render.resolution_x
            camera.data['shot_reader_lens_method'] = 'Physical mirror rim'
            update_lens_sheet(scene, {'method': 'Physical mirror rim',
                                      'fx_normalized': normalized,
                                      'horizontal_fov_deg': degrees(2 * atan(.5 / normalized)),
                                      'rim_heldout_median_px': calibrated['edge_holdout_median_px']})
        settings.body_mirror.scale.z *= calibrated['height_scale']
        exclusions.append(calibrated['corners'])
        report = bpy.data.texts.new(f'{scene.name} · Initial Camera Alignment.json')
        report.write(json.dumps(calibrated, indent=2))
        state = scene.shot_reader_calibration
        state.report = report
        label = 'Physical mirror rim' if 'rim_heldout_after_px' in calibrated else 'Local mirror fit'
        state.summary = f"{label} · {camera.data.lens:.1f} mm"
        state.check = f"Held-out boundary samples · {calibrated['edge_holdout_median_px']:.2f} px median"
    elif settings.body_mirror:
        polygon = reflected_room.mirror_polygon(scene, settings.body_mirror, required=False)
        if polygon:
            exclusions.append(polygon)
    if found.get('direct'):
        left, top, right, bottom = found['direct']['region']
        exclusions.append([[left, top], [right, top], [right, bottom], [left, bottom]])
    if exclusions and not scene.shot_reader_room.exclusions:
        text = bpy.data.texts.new(f'{scene.name} · Detected Set Occlusions.json')
        text.write(json.dumps(exclusions))
        scene.shot_reader_room.exclusions = text


def advance():
    global run, status
    from . import ADDON, job as blockout_job, refresh, status as blockout_status

    current = run
    scene = current['scene']
    try:
        if current['initial']:
            if blockout_job is not None:
                return .5
            if not current['pending_shots']:
                raise ValueError(blockout_status or 'Local blockout did not create a linked shot.')
            current['initial'] = False
            return start_next_shot(current, bpy.context.preferences.addons[ADDON].preferences)
        if current['detector'] is not None:
            process, temp = current['detector']
            if process.poll() is None:
                return .5
            path = Path(temp.name)
            if process.returncode:
                log = (path / 'run.log').read_text(encoding='utf-8', errors='replace').strip()
                raise ValueError(log.splitlines()[-1].removeprefix('ValueError: ') if log else 'Local subject detection failed.')
            found = json.loads((path / 'subject.json').read_text(encoding='utf-8'))
            apply_initial_alignment(scene, found)
            if found.get('calibration'):
                current['completed'].append('Aligning the starting camera')
            settings = scene.shot_reader
            performer = bool(found.get('direct'))
            current['performer'] = performer
            if performer:
                settings.body_region = found.get('reflected', found['direct'])['region']
                settings.foreground_region = found['direct']['region']
                settings.body_direct_head = found['direct']['head']
                settings.foreground_region_frame = settings.body_anchor_frame = found['seed']
            if found.get('warning'):
                current['warnings'].append(found['warning'])
            temp.cleanup()
            current['detector'] = None
            current['queue'] = steps(scene, performer=performer)
        if current['active']:
            label, _operator, worker, success, necessary = current['active']
            if worker.job is not None:
                return .5
            if not success():
                failure = f'{label}: {worker.status or "No output was produced"}'
                if necessary:
                    raise ValueError(failure)
                current['warnings'].append(failure)
            else:
                current['completed'].append(label)
                if worker is room and label == 'Recovering the visible set':
                    room_object = scene.shot_reader_room.room
                    room_object.hide_render = False
                    room_object.display_type = 'TEXTURED'
            current['active'] = None
        while current['queue']:
            step = current['queue'].pop(0)
            label, operator, _worker, _success, necessary = step
            number = current['shots_processed'] + 1
            prefix = f'Shot {number} of {current["total_shots"]} · ' if current['total_shots'] > 1 else ''
            status = f'{prefix}{label}…'
            if operator is visible_set_operator:
                operator = operator(scene)
                if not current['performer'] and operator.idname() == bpy.ops.shot_reader.build_clean_room.idname():
                    surface = scene.shot_reader_room.room
                    if surface:
                        surface.hide_render = False
                        surface.display_type = 'TEXTURED'
                    continue
            with bpy.context.temp_override(window=current['window'], scene=scene, view_layer=scene.view_layers[0]):
                try:
                    started = operator.poll() and operator() == {'FINISHED'}
                except RuntimeError as error:
                    started = False
                    detail = str(error).strip()
                else:
                    detail = 'stage prerequisites were unavailable'
            if started:
                current['active'] = step
                refresh()
                return .5
            message = f'{label}: {detail}'
            if necessary:
                raise ValueError(message)
            current['warnings'].append(message)
        if current['pending_shots'] or current['first_shot']:
            save_report(current)
            return start_next_shot(current, bpy.context.preferences.addons[ADDON].preferences)
        save_report(current)
        status = 'Local reconstruction ready' if not current['warnings'] else 'Local reconstruction finished · see report'
    except (ValueError, ReferenceError, RuntimeError, OSError, KeyError) as error:
        if not current['initial'] and current['first_shot']:
            current['warnings'].append(str(error))
            release_stage(current)
            save_report(current)
            return start_next_shot(current, bpy.context.preferences.addons[ADDON].preferences)
        status = str(error)
    finally:
        refresh()
    stop(clear=False, unregister=False)
    return None


def queue_blockouts(source, scenes):
    if run and run['initial'] and run['scene'] == source:
        run['pending_shots'] = list(scenes)
        run['first_shot'] = scenes[0] if scenes else None
        run['total_shots'] = len(scenes)


def save_report(current):
    scene = current['scene']
    report = bpy.data.texts.new(f'{scene.name} · Local Reconstruction.json')
    report.write(json.dumps({'completed': current['completed'], 'warnings': current['warnings']}, indent=2))
    scene.shot_reader.local_reconstruction = report
    current['warnings_total'] += len(current['warnings'])
    current['shots_processed'] += 1


def start_next_shot(current, prefs):
    global status
    while current['pending_shots']:
        scene = current['pending_shots'].pop(0)
        current.update(scene=scene, completed=['Building the starting blockout'], warnings=[], queue=[])
        current['window'].scene = scene
        try:
            current['detector'] = launch_detector(scene, prefs, True)
        except (ValueError, ReferenceError, RuntimeError, OSError) as error:
            current['warnings'].append(str(error))
            save_report(current)
            continue
        if current['total_shots'] > 1:
            number = current['shots_processed'] + 1
            status = f'Shot {number} of {current["total_shots"]} · finding the performer and camera alignment…'
        else:
            status = 'Finding the performer and camera alignment locally…'
        return .5
    current['window'].scene = current['first_shot']
    if current['warnings_total']:
        status = 'Local reconstruction finished · see shot reports'
    else:
        status = 'Local reconstruction ready'
    stop(clear=False, unregister=False)
    return None


def release_stage(current):
    if current['detector']:
        process, temp = current['detector']
        if process.poll() is None:
            process.kill()
            process.wait()
        temp.cleanup()
        current['detector'] = None
    if current['active']:
        current['active'][2].stop_job()
        current['active'] = None


def stop(*, clear=True, unregister=True):
    global run, status
    if run:
        if run['initial'] and unregister and bpy.ops.shot_reader.cancel.poll():
            bpy.ops.shot_reader.cancel()
        release_stage(run)
        run = None
    if unregister and bpy.app.timers.is_registered(advance):
        bpy.app.timers.unregister(advance)
    if clear:
        status = ''


class ShotReaderReconstructLocally(bpy.types.Operator):
    bl_idname = 'shot_reader.reconstruct_locally'
    bl_label = 'Reconstruct Clip Locally'
    bl_description = 'Build and reconstruct the selected clip with local models'

    @classmethod
    def poll(cls, context):
        from . import job as blockout_job, reference_background

        scene = context.scene
        workers = (arm_alignment, camera_motion, colour_bake, digit_alignment, hair_reconstruction, handheld, head_alignment, mirror_refine,
                   performance, reflected_room, room)
        linked = bool(scene.camera and scene.camera.data.type == 'PERSP' and reference_background(scene))
        return bool(run is None and blockout_job is None and all(worker.job is None for worker in workers)
                    and context.mode == 'OBJECT' and (linked or scene.shot_reader.clip)
                    and not any(obj.get('shot_reader_performance_rig') for obj in scene.objects))

    def execute(self, context):
        from . import ADDON, reference_background

        global run, status
        scene = context.scene
        prefs = context.preferences.addons[ADDON].preferences
        weights = Path(bpy.path.abspath(prefs.person_weights))
        if not weights.is_file():
            self.report({'ERROR'}, 'Set the local person segmentation checkpoint in Shot Reader preferences.')
            return {'CANCELLED'}
        initial = not (scene.camera and scene.camera.data.type == 'PERSP' and reference_background(scene))
        try:
            if initial:
                if bpy.ops.shot_reader.run() != {'FINISHED'}:
                    return {'CANCELLED'}
                detector = None
            else:
                detector = launch_detector(scene, prefs, False)
        except (OSError, ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        run = {'scene': scene, 'window': context.window, 'detector': detector, 'queue': [], 'initial': initial,
               'active': None, 'warnings': [], 'completed': [], 'pending_shots': [], 'first_shot': None,
               'total_shots': 0, 'shots_processed': 0, 'warnings_total': 0}
        status = 'Building the starting blockout locally…' if initial else 'Finding the performer locally…'
        bpy.app.timers.register(advance, first_interval=.5)
        return {'FINISHED'}


class ShotReaderCancelLocal(bpy.types.Operator):
    bl_idname = 'shot_reader.cancel_local_reconstruction'
    bl_label = 'Cancel Local Reconstruction'

    def execute(self, _context):
        global status
        stop()
        status = 'Local reconstruction cancelled'
        return {'FINISHED'}
