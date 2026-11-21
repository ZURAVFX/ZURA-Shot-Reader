"""Import a fitted performance and expose correction controls for SAM 3D Body."""

import json
import math
import os
from pathlib import Path
import tempfile

import bpy
from bpy_extras.io_utils import ImportHelper
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Matrix, Vector


MHR_CONTROLS = {
    1: "Pelvis", 35: "Spine", 37: "Chest", 110: "Neck", 113: "Head",
    39: "Upper arm · Right", 40: "Forearm · Right", 41: "Wrist · Right",
    75: "Upper arm · Left", 76: "Forearm · Left", 77: "Wrist · Left",
    2: "Thigh · Left", 3: "Shin · Left", 4: "Ankle · Left",
    18: "Thigh · Right", 19: "Shin · Right", 20: "Ankle · Right",
}
job = None
status = ""


def add_mhr_controls(context, rig):
    """Add local rotation offsets on top of the original per-bone animation."""
    data = rig.data
    if not all(data.bones.get(f"bone_{index:03}") for index in MHR_CONTROLS):
        return False
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for index, label in MHR_CONTROLS.items():
        source = data.edit_bones[f"bone_{index:03}"]
        control = data.edit_bones.new(label)
        control.head, control.tail, control.roll = source.head, source.tail, source.roll
        control.parent = source.parent
        control.use_deform = False
    bpy.ops.object.mode_set(mode="OBJECT")

    control_collection = data.collections.new("Artist Corrections")
    source_collection = data.collections.new("MHR Deform Bones")
    source_collection.is_visible = False
    shape = bpy.data.objects.get("Shot Reader Rotation Ring")
    if shape is None:
        mesh = bpy.data.meshes.new("Shot Reader Rotation Ring")
        verts = [(math.cos(i * math.tau / 24), 0, math.sin(i * math.tau / 24)) for i in range(24)]
        mesh.from_pydata(verts, [(i, (i + 1) % 24) for i in range(24)], [])
        shape = bpy.data.objects.new(mesh.name, mesh)
    for bone in data.bones:
        is_control = bone.name in MHR_CONTROLS.values()
        for collection in tuple(bone.collections):
            collection.unassign(bone)
        (control_collection if is_control else source_collection).assign(bone)
    for index, label in MHR_CONTROLS.items():
        control = rig.pose.bones[label]
        control.custom_shape = shape
        control.use_custom_shape_bone_size = False
        control.custom_shape_scale_xyz = (.073,) * 3
        control.rotation_mode = "XYZ"
        control.lock_scale = (True,) * 3
        control.lock_location = (index != 1,) * 3
        data.bones[label].color.palette = "THEME04" if "Left" in label else (
            "THEME03" if "Right" in label else "THEME02"
        )
        driven = rig.pose.bones[f"bone_{index:03}"]
        offset = driven.constraints.new("COPY_ROTATION")
        offset.name = "Artist rotation offset"
        offset.target, offset.subtarget = rig, label
        offset.owner_space = offset.target_space = "LOCAL"
        offset.mix_mode = "AFTER"
        if index == 1:
            shift = driven.constraints.new("COPY_LOCATION")
            shift.name = "Artist root offset"
            shift.target, shift.subtarget = rig, label
            shift.owner_space = shift.target_space = "LOCAL"
            shift.use_offset = True
    return True


def import_performance(context, file):
    before = set(context.scene.objects)
    if file.suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(file))
    elif file.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(file), use_anim=True)
    elif file.suffix.lower() == ".bvh":
        bpy.ops.import_anim.bvh(filepath=str(file), use_fps_scale=True)
    else:
        raise ValueError("Choose a GLB, glTF, FBX or BVH performance export.")
    imported = set(context.scene.objects) - before
    rigs = sorted((obj for obj in imported if obj.type == "ARMATURE"), key=lambda obj: obj.name)
    if not rigs:
        raise ValueError("The imported file did not contain a skeletal rig.")
    for rig in rigs:
        rig["shot_reader_performance_rig"] = True
        rig.show_in_front = True
        if file.suffix.lower() == ".glb":
            add_mhr_controls(context, rig)
    return rigs


def place_capture(context, rig, capture, camera, mirror):
    """Apply the estimated translation in the evaluated plate camera on every sample."""
    from .motion_cleanup import stabilize

    scene = context.scene
    action = rig.animation_data.action
    action.name = "SAM 3D Body · Plate Motion"
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for curve in channelbag.fcurves:
                    for key in curve.keyframe_points:
                        key.co.x += scene.frame_start
                        key.handle_left.x += scene.frame_start
                        key.handle_right.x += scene.frame_start
                    curve.update()

    rig.name = "SAM 3D Body · Fitted Performance"
    rig.rotation_mode = "QUATERNION"
    interval = scene.render.fps / (scene.render.fps_base * capture["fps"])
    old_frame, old_subframe = scene.frame_current, scene.frame_subframe
    settings = scene.shot_reader
    pinned = all(0. <= value <= 1. for value in settings.body_direct_head)
    anchor_index = round((settings.body_anchor_frame - scene.frame_start) / interval)
    if mirror and pinned and not 0 <= anchor_index < len(capture["translations"]):
        raise ValueError("Pick the foreground head on a frame inside the scene range.")
    poses = []
    scale = 1.
    minimum_scale = 1.
    previous = None
    try:
        for index, (x, y, z) in enumerate(capture["translations"]):
            time = scene.frame_start + index * interval
            frame = math.floor(time)
            scene.frame_set(frame, subframe=time - frame)
            camera_matrix = camera.matrix_world
            camera_rotation = camera_matrix.to_3x3().normalized()
            axes = Matrix(((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)))
            if mirror:
                axes[0][0] = -1.
                x = -x
            rotation = camera_rotation @ axes
            origin = camera_matrix.translation
            location = origin + camera_rotation @ Vector((x, -y, -z))
            virtual = origin
            if mirror:
                # Reflected footage was unflipped for inference. Reflect the virtual camera pose back to the room.
                normal = (mirror.matrix_world.inverted_safe().transposed().to_3x3() @ Vector((0., -1., 0.))).normalized()
                face = min(point[1] for point in mirror.bound_box)
                point = mirror.matrix_world @ Vector((0., face, 0.))
                location -= 2 * (location - point).dot(normal) * normal
                distance = (origin - point).dot(normal)
                virtual = origin - 2 * distance * normal
                for col in range(3):
                    axis = rotation.col[col].copy()
                    rotation.col[col] = axis - 2 * axis.dot(normal) * normal
                # A mirror image constrains angles. Depth can slide along the mirror-view rays.
                head = rig.pose.bones["bone_113"].head + Vector((0., .01, .045))
                head_ray = location + rotation @ head - virtual
                approach = head_ray.dot(normal) * (1. if distance >= 0. else -1.)
                if approach <= 0.01:
                    raise ValueError("The fitted head is outside the usable mirror rays. Recheck the mirror surface.")
                minimum_scale = max(minimum_scale, (abs(distance) + .18) / approach)
                if pinned and index == anchor_index:
                    scale = depth_from_direct_head(scene, camera.data, camera_rotation, origin,
                                                   virtual, head_ray, settings.body_direct_head)

            poses.append((time, location, rotation, virtual))

        scale = max(scale, minimum_scale)
        rig.scale = (scale,) * 3
        for time, location, rotation, virtual in poses:
            orientation = rotation.to_quaternion()
            if previous is not None:
                orientation.make_compatible(previous)
            previous = orientation
            rig.location = virtual + scale * (location - virtual)
            rig.rotation_quaternion = orientation
            rig.keyframe_insert("location", frame=time, group="Camera-space placement")
            rig.keyframe_insert("rotation_quaternion", frame=time, group="Camera-space placement")
    finally:
        scene.frame_set(old_frame, subframe=old_subframe)
    rig["shot_reader_source"] = f"SAM 3D Body · {len(capture['translations'])} frames"
    rig["shot_reader_alignment"] = "Mirror camera estimate" if mirror else "Plate camera estimate"
    if capture.get('face_vertices'):
        rig['shot_reader_face_vertices'] = capture['face_vertices']
    if mirror:
        rig["shot_reader_mirror_depth_scale"] = scale
        rig["shot_reader_foreground_head_anchor"] = bool(pinned)
    stabilize(rig)
    for child in rig.children_recursive:
        for modifier in child.modifiers:
            if modifier.type == 'ARMATURE' and modifier.object == rig:
                modifier.use_deform_preserve_volume = True
    scene.frame_set(old_frame, subframe=old_subframe)


def depth_from_direct_head(scene, camera, rotation, origin, virtual, head_ray, uv):
    """Find the reflected depth that brings the rig head closest to the foreground ray."""
    from .calibration import camera_intrinsics

    fx, fy, cx, cy = camera_intrinsics(scene, camera)
    u, v = uv
    ray = (rotation @ Vector(((u * scene.render.resolution_x - cx) / fx,
                              (cy - v * scene.render.resolution_y) / fy, -1.))).normalized()
    delta = virtual - origin
    perpendicular = head_ray - ray * head_ray.dot(ray)
    if perpendicular.length_squared < 1e-6:
        raise ValueError("The foreground anchor cannot resolve depth here. Pick the head on another frame.")
    scale = -perpendicular.dot(delta) / perpendicular.length_squared
    if scale <= 0. or (delta + head_ray * scale).dot(ray) <= 0.:
        raise ValueError("The foreground anchor is inconsistent with the mirror. Pick the direct view of the head.")
    return scale


def finish_capture():
    global job, status
    from . import refresh

    process, temp, scene, window, camera, mirror = job
    if process.poll() is None:
        progress = Path(temp.name, "progress.txt")
        if progress.is_file():
            try:
                message = progress.read_text(encoding="utf-8", errors="replace")
                if message:
                    status = message
            except OSError:
                pass
        refresh()
        return 0.5
    try:
        if status == "Cancelling body fit…":
            status = "Body fit cancelled"
        elif process.returncode:
            path = Path(temp.name, "run.log")
            with path.open("rb") as log:
                log.seek(max(0, path.stat().st_size - 32768))
                details = log.read().decode("utf-8", errors="replace")
            bpy.data.texts.new("Shot Reader · Body Fit Log").write(details)
            status = "Body fit failed. See Body Fit Log in the Text Editor."
        else:
            capture = json.loads(Path(temp.name, "body.json").read_text(encoding="utf-8"))
            if camera != scene.camera or camera.type != "CAMERA" or (mirror and mirror.name not in scene.objects):
                raise ValueError("Camera or mirror changed. Run the fit with the new setup.")
            with bpy.context.temp_override(window=window, scene=scene, view_layer=scene.view_layers[0]):
                before = set(scene.objects)
                try:
                    rigs = import_performance(bpy.context, Path(temp.name, "body.glb"))
                    for rig in rigs:
                        place_capture(bpy.context, rig, capture, camera, mirror)
                except Exception:
                    for obj in set(scene.objects) - before:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    raise
                proxies = [obj for obj in before if obj.get("shot_reader_class") == "person"]
                if len(proxies) == 1:
                    for obj in (proxies[0], *proxies[0].children_recursive):
                        obj.hide_set(True)
                        obj.hide_render = True
            status = f"Fitted performance · {len(capture['translations'])} frames"
    except (ValueError, OSError, ReferenceError, RuntimeError, TypeError, KeyError) as error:
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
    if bpy.app.timers.is_registered(finish_capture):
        bpy.app.timers.unregister(finish_capture)
    status = ""


class ShotReaderFitPerformance(bpy.types.Operator):
    bl_idname = "shot_reader.fit_performance"
    bl_label = "Fit Character from Plate"
    bl_description = "Run SAM 3D Body locally and create a skinned character with editable motion"

    @classmethod
    def poll(cls, context):
        from . import job as blockout_job, mirror_refine, reference_background

        camera = context.scene.camera
        return (job is None and blockout_job is None and mirror_refine.job is None and context.mode == "OBJECT" and camera is not None
                and camera.data.type == "PERSP" and reference_background(context.scene) is not None)

    def execute(self, context):
        from . import ADDON, reference_background, start_worker, worker_environment
        from .calibration import camera_intrinsics

        global job, status
        scene = context.scene
        settings = scene.shot_reader
        camera = scene.camera
        prefs = context.preferences.addons[ADDON].preferences
        comfy, python, weights = (Path(bpy.path.abspath(getattr(prefs, key)))
                                  for key in ("body_comfy", "body_python", "body_weights"))
        if not (comfy / "comfy_extras" / "nodes_sam3d_body.py").is_file() or not python.is_file() or not weights.is_file():
            self.report({"ERROR"}, "Set a SAM-capable ComfyUI folder, its Python executable and a body checkpoint in Shot Reader preferences.")
            return {"CANCELLED"}
        reference = reference_background(scene).clip
        clip = Path(bpy.path.abspath(reference.filepath, library=reference.library))
        if not clip.is_file():
            self.report({"ERROR"}, "Relink the source clip before fitting a character.")
            return {"CANCELLED"}
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        cw, ch = reference.size
        if not ch or abs(cw / ch - rw / rh) > .005 or scene.render.pixel_aspect_x != scene.render.pixel_aspect_y:
            self.report({"ERROR"}, "Match the scene aspect ratio to the plate and use square pixels.")
            return {"CANCELLED"}
        mirror = settings.body_mirror
        if mirror and mirror.name not in scene.objects:
            self.report({"ERROR"}, "Select a mirror surface in this scene.")
            return {"CANCELLED"}
        fps = scene.render.fps / scene.render.fps_base
        start = (reference.frame_offset + scene.frame_start - reference.frame_start) / fps
        fx, fy, cx, cy = camera_intrinsics(scene, camera.data)
        request = {
            "comfy": str(comfy), "weights": str(weights), "clip": str(clip),
            "start": max(0., start), "end": start + (scene.frame_end - scene.frame_start + 1) / fps,
            "region": list(settings.body_region), "reflected": bool(mirror),
            "intrinsics": [fx / rw, fy / rh, cx / rw, cy / rh],
        }
        temp = tempfile.TemporaryDirectory(prefix="shot-reader-body-")
        try:
            path = Path(temp.name, "request.json")
            path.write_text(json.dumps(request), encoding="utf-8")
            command = [str(python), str(Path(__file__).with_name("body_worker.py")), str(path)]
            process = start_worker(command, worker_environment(prefs), temp)
        except OSError as error:
            temp.cleanup()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        job = process, temp, scene, context.window, camera, mirror
        status = "Loading SAM 3D Body…"
        bpy.app.timers.register(finish_capture, first_interval=0.5)
        return {"FINISHED"}


class ShotReaderCancelFit(bpy.types.Operator):
    bl_idname = "shot_reader.cancel_fit"
    bl_label = "Cancel Body Fit"

    def execute(self, _context):
        global status
        if job and job[0].poll() is None:
            job[0].terminate()
            status = "Cancelling body fit…"
        return {"FINISHED"}


class ShotReaderPickBodyRegion(bpy.types.Operator):
    bl_idname = "shot_reader.pick_body_region"
    bl_label = "Frame Performer"
    bl_description = "Click two corners around the performer, allowing space for motion throughout the shot"
    bl_options = {"UNDO"}
    pin_head: bpy.props.BoolProperty(options={"HIDDEN", "SKIP_SAVE"})
    track_foreground: bpy.props.BoolProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == "VIEW_3D" and ShotReaderFitPerformance.poll(context)

    def invoke(self, context, _event):
        from . import reference_background

        self.camera = context.scene.camera
        self.background = reference_background(context.scene)
        self.previous = (self.background.show_background_image, self.background.alpha,
                         self.camera.data.show_background_images, context.space_data.use_local_camera)
        self.first = None
        self.background.show_background_image = True
        self.background.alpha = 1.
        self.camera.data.show_background_images = True
        context.space_data.use_local_camera = False
        context.space_data.region_3d.view_perspective = "CAMERA"
        if self.pin_head:
            header = "Click the centre of the foreground head · Esc cancels"
        elif self.track_foreground:
            header = "Frame the direct view of the performer · Click the first corner"
        else:
            header = "Click the first corner around the performer · Esc cancels"
        context.area.header_text_set(header)
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def finish(self, context):
        (self.background.show_background_image, self.background.alpha,
         self.camera.data.show_background_images, context.space_data.use_local_camera) = self.previous
        context.area.header_text_set(None)
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self.finish(context)
            return {"CANCELLED"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"RUNNING_MODAL"}
        region = next(region for region in context.area.regions if region.type == "WINDOW")
        view = context.space_data.region_3d
        corners = [location_3d_to_region_2d(region, view, self.camera.matrix_world @ point)
                   for point in self.camera.data.view_frame(scene=context.scene)]
        if any(point is None for point in corners):
            return {"RUNNING_MODAL"}
        xs, ys = zip(*corners)
        u = (event.mouse_x - region.x - min(xs)) / (max(xs) - min(xs))
        v = (max(ys) - event.mouse_y + region.y) / (max(ys) - min(ys))
        if not (0. <= u <= 1. and 0. <= v <= 1.):
            return {"RUNNING_MODAL"}
        settings = context.scene.shot_reader
        if self.pin_head:
            settings.body_direct_head = (u, v)
            settings.body_anchor_frame = context.scene.frame_current
            self.finish(context)
            return {"FINISHED"}
        if self.first is None:
            self.first = u, v
            context.area.header_text_set("Click the opposite corner · Esc cancels")
            return {"RUNNING_MODAL"}
        left, right = sorted((self.first[0], u))
        top, bottom = sorted((self.first[1], v))
        if min(right - left, bottom - top) < .03:
            self.report({"WARNING"}, "Use a larger region.")
            return {"RUNNING_MODAL"}
        if self.track_foreground:
            settings.foreground_region = (left, top, right, bottom)
            settings.foreground_region_frame = context.scene.frame_current
        else:
            settings.body_region = (left, top, right, bottom)
        self.finish(context)
        return {"FINISHED"}


class ShotReaderResetBodyRegion(bpy.types.Operator):
    bl_idname = "shot_reader.reset_body_region"
    bl_label = "Use Full Frame"
    bl_options = {"UNDO"}
    clear_head: bpy.props.BoolProperty(options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        context.scene.shot_reader.property_unset("body_direct_head" if self.clear_head else "body_region")
        return {"FINISHED"}


class ShotReaderImportPerformance(bpy.types.Operator, ImportHelper):
    bl_idname = "shot_reader.import_performance"
    bl_label = "Import Fitted Performance"
    bl_options = {"UNDO"}
    filter_glob: bpy.props.StringProperty(default="*.glb;*.gltf;*.fbx;*.bvh", options={"HIDDEN"})

    def execute(self, context):
        try:
            rigs = import_performance(context, Path(self.filepath))
        except (ValueError, RuntimeError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Imported {len(rigs)} performance rig(s). Place them against the source clip.")
        return {"FINISHED"}


class ShotReaderOpenRGBMocap(bpy.types.Operator):
    bl_idname = "shot_reader.open_rgb_mocap"
    bl_label = "Open RGB MoCap"

    def execute(self, context):
        from . import ADDON

        path = Path(bpy.path.abspath(context.preferences.addons[ADDON].preferences.rgb_mocap))
        if os.name != "nt" or not path.is_file() or path.suffix.lower() not in {".exe", ".cmd", ".bat"}:
            self.report({"ERROR"}, "Point Shot Reader preferences to your RGB MoCap launcher on Windows.")
            return {"CANCELLED"}
        try:
            os.startfile(str(path), cwd=str(path.parent))
        except OSError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class SHOT_READER_PT_performance(bpy.types.Panel):
    bl_label = "Performance"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "SHOT_READER_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from . import arm_alignment, colour_bake, digit_alignment, hair_reconstruction, handheld, head_alignment, mirror_refine, motion_cleanup

        settings = context.scene.shot_reader
        options = self.layout.column(align=True)
        options.enabled = job is None
        options.prop(settings, "body_mirror")
        row = options.row(align=True)
        row.operator("shot_reader.pick_body_region", icon="SELECT_SET")
        if tuple(settings.body_region) != (0., 0., 1., 1.):
            row.operator("shot_reader.reset_body_region", text="", icon="LOOP_BACK")
        if settings.body_mirror:
            row = options.row(align=True)
            text = "Pin Foreground Head" if settings.body_direct_head[0] < 0. else "Repin Foreground Head"
            row.operator("shot_reader.pick_body_region", text=text, icon="PIVOT_CURSOR").pin_head = True
            if settings.body_direct_head[0] >= 0.:
                row.operator("shot_reader.reset_body_region", text="", icon="X").clear_head = True
        options.operator("shot_reader.fit_performance", icon="OUTLINER_OB_ARMATURE")
        if settings.body_mirror and mirror_refine.fitted_rig(context):
            refine = options.box()
            frame_text = "Reframe Foreground" if settings.foreground_region_frame else "Frame Foreground"
            refine.operator("shot_reader.pick_body_region", text=frame_text, icon="SELECT_SET").track_foreground = True
            refine.operator("shot_reader.track_mirror_foreground", icon="IMAGE_ALPHA")
            refine.operator("shot_reader.refine_mirror", text="Use Matte File", icon="FILE_MOVIE")
            if mirror_refine.status:
                self.layout.label(text=mirror_refine.status)
        if settings.mirror_refinement_anchor:
            options.operator("shot_reader.restore_mirror_refinement", icon="LOOP_BACK")
        rig, mesh = colour_bake.performer(context)
        if mesh:
            head_offset = rig.pose.bones['bone_113'].constraints.get(head_alignment.CONSTRAINT)
            if head_offset:
                options.prop(head_offset, 'enabled', text='Head Plate Alignment')
            else:
                options.operator('shot_reader.align_head', icon='CON_LOCLIKE')
            if head_alignment.status:
                options.label(text=head_alignment.status)
            arm_offsets = arm_alignment.active_constraints(rig)
            if arm_offsets:
                text = 'Disable Arm Plate Alignment' if any(item.enabled for item in arm_offsets) else 'Enable Arm Plate Alignment'
                options.operator('shot_reader.toggle_arm_alignment', text=text, icon='CON_ROTLIKE')
            else:
                options.operator('shot_reader.align_arms', icon='CON_ROTLIKE')
            if arm_alignment.status:
                options.label(text=arm_alignment.status)
            stable = rig.get(motion_cleanup.SMOOTH)
            action = rig.animation_data.action if rig.animation_data else None
            text = ('Use Captured Arm Motion' if action == stable else 'Use Stabilized Arm Motion') if stable else 'Stabilize Arm Motion'
            options.operator('shot_reader.stabilize_arm_motion', text=text)
            skin = next((item for item in mesh.modifiers if item.type == 'ARMATURE' and item.object == rig), None)
            if skin:
                options.prop(skin, 'use_deform_preserve_volume', text='Preserve Skin Volume')
            fingers = digit_alignment.active_constraints(rig)
            if fingers:
                text = 'Disable Finger Plate Alignment' if any(item.enabled for item in fingers) else 'Enable Finger Plate Alignment'
                options.operator('shot_reader.toggle_digit_alignment', text=text, icon='CON_ROTLIKE')
            else:
                options.operator('shot_reader.align_fingers', icon='CON_ROTLIKE')
            if digit_alignment.status:
                options.label(text=digit_alignment.status)
            hair = rig.get('shot_reader_hair')
            if hair:
                scalp = hair.get('shot_reader_scalp_hair')
                label = 'Build Scalp Hair' if not scalp else (
                    'Show Plate Hair in Viewport' if hair.hide_viewport else 'Show Scalp Hair in Viewport'
                )
                options.operator('shot_reader.toggle_viewport_hair', text=label)
                options.prop(hair, 'hide_render', text='Hide Plate Hair in Render')
            else:
                options.operator('shot_reader.reconstruct_hair', icon='MESH_DATA')
            if hair_reconstruction.status:
                options.label(text=hair_reconstruction.status)
            keys = mesh.data.shape_keys
            shape = keys.key_blocks.get(hair_reconstruction.BODY_SHAPE) if keys else None
            if shape:
                options.prop(shape, 'value', text='Plate Silhouette')
            options.operator('shot_reader.bake_performer_colour', icon='MATERIAL')
            layer = mesh.modifiers.get(colour_bake.MODIFIER)
            if layer:
                row = options.row(align=True)
                row.prop(layer, 'show_viewport', text='Plate Colour in Viewport')
                row.prop(layer, 'show_render', text='Render')
            if colour_bake.status:
                options.label(text=colour_bake.status)
            options.operator('shot_reader.track_handheld')
            grip = handheld.grip_constraints(rig)
            if grip:
                text = 'Disable Handheld Grip' if any(item.enabled for item in grip) else 'Enable Handheld Grip'
                options.operator('shot_reader.toggle_handheld_grip', text=text)
            if handheld.status:
                options.label(text=handheld.status)
        if status:
            self.layout.label(text=status)
        if job:
            self.layout.operator("shot_reader.cancel_fit", icon="CANCEL")
        self.layout.operator("shot_reader.import_performance", icon="ARMATURE_DATA")
        if os.name == "nt":
            self.layout.operator("shot_reader.open_rgb_mocap", icon="PLAY")


CLASSES = (ShotReaderFitPerformance, ShotReaderCancelFit, ShotReaderPickBodyRegion, ShotReaderResetBodyRegion,
           ShotReaderImportPerformance, ShotReaderOpenRGBMocap, SHOT_READER_PT_performance)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
