"""Refine a fitted performance and its reflection using a foreground matte."""

import json
from pathlib import Path
import tempfile

import bpy
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector
import numpy as np

from .calibration import camera_intrinsics


job = None
status = ""


def fitted_rig(context):
    active = context.active_object
    if active and active.get("shot_reader_alignment") == "Mirror camera estimate":
        return active
    rigs = [obj for obj in context.scene.objects if obj.get("shot_reader_alignment") == "Mirror camera estimate"]
    return rigs[0] if len(rigs) == 1 else None


def mirror_basis(mirror_matrix, front):
    normal = (mirror_matrix.inverted().transposed().to_3x3() @ Vector((0., -1., 0.))).normalized()
    distance = normal.dot(mirror_matrix @ Vector((0., front, 0.)))
    return normal, distance


def export_geometry(context, rig, camera, mirror, directory):
    scene = context.scene
    duration = scene.frame_end - scene.frame_start + 1
    if duration < 5:
        raise ValueError("Use at least five frames so the silhouette fit can be checked independently.")
    source_frame, source_subframe = scene.frame_current, scene.frame_subframe
    frames = range(scene.frame_start, scene.frame_end + 1, max(1, (duration + 16) // 17))
    fps = scene.render.fps / scene.render.fps_base
    body = next((obj for obj in rig.children_recursive if obj.type == "MESH" and obj.vertex_groups
                 and not obj.get('shot_reader_hair_report')), None)
    if body is None or "bone_113" not in rig.data.bones:
        raise ValueError("Select a fitted SAM 3D Body character with its skinned mesh.")

    head = rig.data.bones["bone_113"]
    groups = {body.vertex_groups[bone.name].index for bone in (head, *head.children_recursive)
              if body.vertex_groups.get(bone.name)}
    is_head = np.array([sum(group.weight for group in vert.groups if group.group in groups) >= .6
                        for vert in body.data.vertices])
    if is_head.sum() < 100:
        raise ValueError("The selected rig does not expose a usable head mesh.")
    body.data.calc_loop_triangles()
    faces = np.array([tri.vertices[:] for tri in body.data.loop_triangles])
    rw, rh = scene.render.resolution_x, scene.render.resolution_y
    if scene.render.pixel_aspect_x != scene.render.pixel_aspect_y:
        raise ValueError("Mirror refinement requires square pixels.")
    scale = min(1., 768 / max(rw, rh))
    width, height = round(rw * scale), round(rh * scale)
    native_scale = 1 / scale
    virtual_points = []
    path = []
    intrinsics = []
    try:
        scene.frame_set(scene.frame_start)
        cam_world = camera.matrix_world.normalized()
        inverse = cam_world.inverted()
        mirror_world = mirror.matrix_world.copy()
        mirror_camera = inverse @ mirror_world
        low, high = np.array(mirror.bound_box).min(axis=0), np.array(mirror.bound_box).max(axis=0)
        normal, distance = mirror_basis(mirror_camera, low[1])
        if distance > -.01:
            raise ValueError("The camera must face the reflective side of the mirror.")
        corners = [list(mirror_camera @ Vector((x, low[1], z)))
                   for x, z in ((low[0], high[2]), (high[0], high[2]), (high[0], low[2]), (low[0], low[2]))]
        depsgraph = context.evaluated_depsgraph_get()
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            if not np.allclose(mirror.matrix_world, mirror_world, atol=1e-5):
                raise ValueError("The reflective surface must stay fixed over the scene range.")
            pose = inverse @ camera.matrix_world.normalized()
            if normal.dot(pose.translation) - distance < .01:
                raise ValueError("The tracked camera crosses the reflective surface.")
            path.append(pose)
            if frame not in frames:
                continue
            fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
            intrinsics.append([fx * width / rw, fy * height / rh, cx * width / rw, cy * height / rh])
            evaluated = body.evaluated_get(depsgraph)
            if len(evaluated.data.vertices) != len(is_head):
                raise ValueError("Temporarily disable modifiers that change the fitted mesh topology.")
            coords = np.empty(3 * len(is_head), dtype=np.float32)
            evaluated.data.vertices.foreach_get("co", coords)
            model_to_camera = np.asarray(inverse @ evaluated.matrix_world)
            direct = coords.reshape(-1, 3) @ model_to_camera[:3, :3].T + model_to_camera[:3, 3]
            n = np.array(normal)
            virtual_points.append(direct - 2 * (direct @ n - distance)[:, None] * n)
    finally:
        scene.frame_set(source_frame, subframe=source_subframe)
    start = scene.frame_start
    np.savez(directory / "geometry.npz", virtual=np.stack(virtual_points), head=is_head, faces=faces,
             normal=np.array(normal), distance=distance, corners=corners, size=[width, height], native_scale=native_scale,
             times=[(frame - start) / fps for frame in frames], intrinsics=intrinsics,
             cameras=np.array([path[frame - start] for frame in frames]),
             corner_tolerance=scene.shot_reader_calibration.threshold)
    return cam_world, mirror_world, float(low[1]), path, start, list(zip(frames, intrinsics))


def reflection(normal, distance):
    transform = Matrix.Identity(4)
    for row in range(3):
        for col in range(3):
            transform[row][col] -= 2 * normal[row] * normal[col]
    transform.translation = 2 * distance * normal
    return transform


def apply_refinement(scene, rig, mirror, world, mirror_world, front, path, start, _intrinsics, result):
    settings = scene.shot_reader
    camera_space = world.inverted()
    old_plane = camera_space @ mirror_world
    normal, distance = mirror_basis(old_plane, front)
    tuned = Vector(result["normal"]).normalized()
    multiplier = result["scale"]
    scale = Matrix.Scale(multiplier, 4)
    reflected_world = world @ reflection(tuned, distance)
    change = reflected_world @ scale @ reflection(normal, distance) @ camera_space
    # Apply a similarity transform above the performance so all pose keys and controls stay editable.
    parent = rig
    while parent.parent and parent.parent.get("shot_reader_mirror_refinement"):
        parent = parent.parent
    if parent.parent is not None:
        raise ValueError("The fitted character must be at the scene root or under a Shot Reader refinement.")

    # Keep each mirror surface vertex on its original camera ray. Rebuild the object's local axes
    # so a subsequent body fit uses the updated, physical mirror plane.
    x_axis = old_plane.to_3x3().col[0].copy()
    x_axis = (x_axis - x_axis.dot(tuned) * tuned).normalized()
    z_axis = tuned.cross(x_axis)
    new_plane = Matrix((x_axis, -tuned, z_axis)).transposed().to_4x4()
    origin = old_plane @ Vector((0., front, 0.))
    new_plane.translation = origin * (distance / origin.dot(tuned))
    destination = new_plane.inverted()
    warped = []
    for vertex in mirror.data.vertices:
        point = old_plane @ vertex.co
        depth = point.dot(normal) - distance
        ray = point - depth * normal
        if ray.dot(tuned) >= -1e-5:
            raise ValueError("The solved mirror plane does not intersect the original camera rays.")
        coord = distance / ray.dot(tuned) * ray + depth * tuned
        warped.append(destination @ coord)

    original = mirror.data
    data = original.copy()
    for vertex, coord in zip(data.vertices, warped):
        vertex.co = coord
    data.update()
    mirror.data = data
    mirror.matrix_world = world @ new_plane
    anchor = bpy.data.objects.new("Mirror Silhouette Alignment", None)
    scene.collection.objects.link(anchor)
    anchor.empty_display_size = .05
    anchor["shot_reader_mirror_refinement"] = True
    anchor["original_mirror"] = mirror
    anchor["original_mirror_mesh"] = original
    anchor["original_mirror_transform"] = [value for row in mirror_world for value in row]
    if settings.mirror_refinement_report:
        anchor["original_report"] = settings.mirror_refinement_report
    anchor.matrix_world = change
    offsets = [reflected_world.to_3x3() @ ((1. - multiplier) * pose.translation) for pose in path]
    if any(offset.length > 1e-6 for offset in offsets):
        location = change.translation
        for frame, offset in enumerate(offsets, start):
            anchor.location = location + offset
            anchor.keyframe_insert("location", frame=frame, group="Mirror Camera Compensation")
        animation = anchor.animation_data
        channels = animation.action.layers[0].strips[0].channelbag(animation.action_slot)
        for curve in channels.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'
    parent.parent = anchor
    parent.matrix_parent_inverse = Matrix.Identity(4)
    report = bpy.data.texts.new(f"{scene.name} · Mirror Silhouette Check.json")
    report.write(json.dumps(result, indent=2))
    settings.mirror_refinement_report = report
    settings.mirror_refinement_anchor = anchor
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)


def finish_refinement():
    global job, status
    from . import refresh

    process, temp, scene, rig, camera, mirror, bases = job
    if process.poll() is None:
        return .4
    try:
        directory = Path(temp.name)
        if process.returncode:
            log = (directory / "run.log").read_text(encoding="utf-8", errors="replace").strip()
            raise ValueError(log.splitlines()[-1].removeprefix("ValueError: ") if log else "Mirror refinement failed.")
        world, mirror_world, _front, path, start, lenses = bases
        if scene.camera != camera or scene.frame_start != start or scene.frame_end != start + len(path) - 1:
            raise ValueError("The camera or mirror changed during refinement. Run the solve again.")
        frame, subframe = scene.frame_current, scene.frame_subframe
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        factor = min(1., 768 / max(rw, rh))
        sx, sy = round(rw * factor) / rw, round(rh * factor) / rh
        try:
            for sample, expected in lenses:
                scene.frame_set(sample)
                lens = np.asarray(camera_intrinsics(scene, camera.data)) * [sx, sy, sx, sy]
                if not (np.allclose(camera.matrix_world.normalized(), world @ path[sample - start], atol=1e-5)
                        and np.allclose(mirror.matrix_world, mirror_world, atol=1e-5)
                        and np.allclose(lens, expected, atol=.01)):
                    raise ValueError("The camera, lens or mirror changed during refinement. Run the solve again.")
        finally:
            scene.frame_set(frame, subframe=subframe)
        result = json.loads((directory / "solve.json").read_text(encoding="utf-8"))
        apply_refinement(scene, rig, mirror, *bases, result)
        before, after = result["baseline"], result["refined"]
        status = f"Held-out silhouette overlap {before['foreground_iou']:.0%} → {after['foreground_iou']:.0%}"
    except (OSError, ReferenceError, RuntimeError, ValueError, KeyError) as error:
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
    if bpy.app.timers.is_registered(finish_refinement):
        bpy.app.timers.unregister(finish_refinement)
    status = ""


class ShotReaderRefineMirror(bpy.types.Operator, ImportHelper):
    bl_idname = "shot_reader.refine_mirror"
    bl_label = "Refine Mirror from Foreground Matte"
    bl_description = "Choose a video with the direct-view performer in white on black, starting at the scene's first frame"
    filter_glob: bpy.props.StringProperty(default="*.mov;*.mp4;*.mkv;*.avi;*.webm", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        from . import performance

        scene = context.scene
        mirror = scene.shot_reader.body_mirror
        return bool(job is None and performance.job is None and context.mode == "OBJECT" and scene.camera
                    and scene.camera.data.type == "PERSP" and mirror and mirror.name in scene.objects and fitted_rig(context))

    def execute(self, context):
        matte = Path(bpy.path.abspath(self.filepath))
        if not matte.is_file():
            self.report({"ERROR"}, "Choose a foreground matte video.")
            return {"CANCELLED"}
        return launch_refinement(self, context, matte=matte)


class ShotReaderTrackMirrorForeground(bpy.types.Operator):
    bl_idname = "shot_reader.track_mirror_foreground"
    bl_label = "Track Foreground and Refine Mirror"
    bl_description = "Track the selected foreground performer directly from the linked plate and refine the mirror fit"

    @classmethod
    def poll(cls, context):
        from . import reference_background

        return ShotReaderRefineMirror.poll(context) and bool(reference_background(context.scene))

    def execute(self, context):
        from . import ADDON, reference_background

        scene = context.scene
        settings = scene.shot_reader
        x0, y0, x1, y1 = settings.foreground_region
        u, v = settings.body_direct_head
        seed_frame = settings.foreground_region_frame
        if not (u >= 0. and v >= 0. and x0 < u < x1 and y0 < v < y1 and
                seed_frame == settings.body_anchor_frame and scene.frame_start <= seed_frame <= scene.frame_end):
            self.report({"ERROR"}, "On one frame, pin the foreground head and frame the entire direct-view performer.")
            return {"CANCELLED"}
        prefs = context.preferences.addons[ADDON].preferences
        weights = Path(bpy.path.abspath(prefs.matte_weights))
        if not weights.is_file():
            self.report({"ERROR"}, "Set a Segment Anything checkpoint in Shot Reader preferences.")
            return {"CANCELLED"}
        clip = reference_background(scene).clip
        fps = scene.render.fps / scene.render.fps_base
        start = max(0., (clip.frame_offset + scene.frame_start - clip.frame_start) / fps)
        request = {"clip": bpy.path.abspath(clip.filepath, library=clip.library), "start": start,
                   "time": start + (seed_frame - scene.frame_start) / fps, "head": [u, v],
                   "region": [x0, y0, x1, y1], "weights": str(weights)}
        return launch_refinement(self, context, tracking=request)


def launch_refinement(operator, context, *, matte=None, tracking=None):
    from . import ADDON, start_worker, worker_environment

    global job, status
    scene, rig = context.scene, fitted_rig(context)
    camera, mirror = scene.camera, scene.shot_reader.body_mirror
    prefs = context.preferences.addons[ADDON].preferences
    temp = tempfile.TemporaryDirectory(prefix="shot-reader-silhouette-")
    try:
        directory = Path(temp.name)
        bases = export_geometry(context, rig, camera, mirror, directory)
        script = str(Path(__file__).with_name("mirror_solve.py"))
        if tracking is not None:
            request = directory / "foreground.json"
            request.write_text(json.dumps(tracking), encoding="utf-8")
            command = [bpy.path.abspath(prefs.body_python), script, temp.name, str(request), "--track"]
        else:
            command = [bpy.path.abspath(prefs.python), script, temp.name, str(matte)]
        process = start_worker(command, worker_environment(prefs), temp)
    except (OSError, RuntimeError, ValueError) as error:
        temp.cleanup()
        operator.report({"ERROR"}, str(error))
        return {"CANCELLED"}
    job = process, temp, scene, rig, camera, mirror, bases
    status = "Tracking foreground and refining mirror…" if tracking else "Refining mirror from foreground silhouette…"
    bpy.app.timers.register(finish_refinement, first_interval=.4)
    return {"FINISHED"}


class ShotReaderRestoreMirrorRefinement(bpy.types.Operator):
    bl_idname = "shot_reader.restore_mirror_refinement"
    bl_label = "Restore Previous Mirror Fit"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return job is None and context.mode == "OBJECT" and bool(context.scene.shot_reader.mirror_refinement_anchor)

    def execute(self, context):
        global status
        settings = context.scene.shot_reader
        anchor = settings.mirror_refinement_anchor
        mirror = anchor.get("original_mirror")
        mesh = anchor.get("original_mirror_mesh")
        matrix = anchor.get("original_mirror_transform")
        if not (mirror and mesh and matrix and len(anchor.children) == 1 and anchor.parent is None):
            self.report({"ERROR"}, "The original mirror or the fitted character has changed.")
            return {"CANCELLED"}
        mirror.data = mesh
        mirror.matrix_world = Matrix(tuple(tuple(matrix[i:i + 4]) for i in range(0, 16, 4)))
        performer = anchor.children[0]
        performer.parent = None
        performer.matrix_parent_inverse = Matrix.Identity(4)
        settings.mirror_refinement_anchor = performer if performer.get("shot_reader_mirror_refinement") else None
        settings.mirror_refinement_report = anchor.get("original_report")
        bpy.data.objects.remove(anchor, do_unlink=True)
        status = "Previous mirror fit restored"
        return {"FINISHED"}


CLASSES = (ShotReaderRefineMirror, ShotReaderTrackMirrorForeground, ShotReaderRestoreMirrorRefinement)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
