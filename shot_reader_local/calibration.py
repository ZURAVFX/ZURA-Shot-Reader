"""Blender tools for picking corresponding set points and verifying a camera solve."""

import json
from pathlib import Path
import subprocess

import bmesh
import bpy
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Matrix, Vector


class ShotReaderLandmark(bpy.types.PropertyGroup):
    object: bpy.props.PointerProperty(type=bpy.types.Object)
    local: bpy.props.FloatVectorProperty(size=3)
    uv: bpy.props.FloatVectorProperty(size=2, precision=5)
    holdout: bpy.props.BoolProperty(name="Holdout", description="Measure error without using this point to fit")


class ShotReaderCalibration(bpy.types.PropertyGroup):
    points: bpy.props.CollectionProperty(type=ShotReaderLandmark)
    active: bpy.props.IntProperty()
    frame: bpy.props.IntProperty()
    refine_focal: bpy.props.BoolProperty(name="Refine Focal Length")
    threshold: bpy.props.FloatProperty(name="Pixel Tolerance", default=4., min=0.5, max=100.)
    original_camera: bpy.props.PointerProperty(type=bpy.types.Object)
    report: bpy.props.PointerProperty(type=bpy.types.Text)
    summary: bpy.props.StringProperty()
    check: bpy.props.StringProperty()
    track_previous_camera: bpy.props.PointerProperty(type=bpy.types.Object)
    track_report: bpy.props.PointerProperty(type=bpy.types.Text)
    track_summary: bpy.props.StringProperty()


class SHOT_READER_UL_landmarks(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False)
        row.prop(item, "holdout", text="Check")


class ShotReaderPickLandmark(bpy.types.Operator):
    bl_idname = "shot_reader.pick_landmark"
    bl_label = "Pick Plate Landmark"
    bl_description = "Select a single set vertex or an Empty, then click its counterpart in the source clip"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D" and context.scene.camera is not None

    def invoke(self, context, _event):
        from . import reference_background

        scene = context.scene
        state = scene.shot_reader_calibration
        reference = reference_background(scene)
        if reference is None or len(state.points) and state.frame != scene.frame_current:
            self.report({"ERROR"}, "Attach a source clip and pick all landmarks on the same frame.")
            return {"CANCELLED"}
        if scene.camera.data.type != "PERSP":
            self.report({"ERROR"}, "A perspective camera is required.")
            return {"CANCELLED"}
        rw, rh, clip_size = scene.render.resolution_x, scene.render.resolution_y, reference.clip.size
        if (not clip_size[1] or abs(clip_size[0] / clip_size[1] - rw / rh) > .005 or
                scene.render.pixel_aspect_x != scene.render.pixel_aspect_y):
            self.report({"ERROR"}, "Match the scene aspect ratio to the source and use square pixels.")
            return {"CANCELLED"}
        obj = context.active_object
        if obj and context.mode == "EDIT_MESH":
            vertices = [vertex for vertex in bmesh.from_edit_mesh(obj.data).verts if vertex.select and not vertex.hide]
            if len(vertices) != 1:
                self.report({"ERROR"}, "Select exactly one set vertex in Edit Mode.")
                return {"CANCELLED"}
            point = vertices[0].co.copy()
        elif obj and obj.type == "EMPTY" and context.mode == "OBJECT":
            point = (0., 0., 0.)
        else:
            self.report({"ERROR"}, "Select one set vertex in Edit Mode, or select an Empty.")
            return {"CANCELLED"}
        self.anchor, self.local = obj, point
        self.background = reference
        self.previous_display = reference.show_background_image, reference.alpha, scene.camera.data.show_background_images
        self.previous_local_camera = context.space_data.use_local_camera
        context.space_data.use_local_camera = False
        context.space_data.region_3d.view_perspective = "CAMERA"
        reference.show_background_image = True
        reference.alpha = 1.
        scene.camera.data.show_background_images = True
        self.camera = scene.camera
        context.area.header_text_set("Click the matching point on the plate. Esc cancels.")
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def finish(self, context):
        self.background.show_background_image, self.background.alpha, self.camera.data.show_background_images = self.previous_display
        context.space_data.use_local_camera = self.previous_local_camera
        context.area.header_text_set(None)
        context.area.tag_redraw()

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self.finish(context)
            return {"CANCELLED"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"RUNNING_MODAL"}
        scene = context.scene
        region = next(region for region in context.area.regions if region.type == "WINDOW")
        view = context.space_data.region_3d
        corners = [location_3d_to_region_2d(region, view, self.camera.matrix_world @ point)
                   for point in self.camera.data.view_frame(scene=scene)]
        if any(point is None for point in corners):
            self.finish(context)
            self.report({"ERROR"}, "Could not locate the camera frame.")
            return {"CANCELLED"}
        xs, ys = zip(*corners)
        x = event.mouse_x - region.x
        y = event.mouse_y - region.y
        u = (x - min(xs)) / (max(xs) - min(xs))
        v = (max(ys) - y) / (max(ys) - min(ys))
        if not 0. <= u <= 1. or not 0. <= v <= 1.:
            return {"RUNNING_MODAL"}
        state = scene.shot_reader_calibration
        item = state.points.add()
        item.name = f"{self.anchor.name} · {len(state.points)}"
        item.object, item.local, item.uv = self.anchor, self.local, (u, v)
        state.active = len(state.points) - 1
        state.frame = scene.frame_current
        state.summary = ""
        state.check = ""
        self.finish(context)
        return {"FINISHED"}


class ShotReaderRemoveLandmark(bpy.types.Operator):
    bl_idname = "shot_reader.remove_landmark"
    bl_label = "Remove Landmark"
    bl_options = {"UNDO"}

    def execute(self, context):
        state = context.scene.shot_reader_calibration
        if len(state.points):
            state.points.remove(state.active)
            state.active = max(0, min(state.active, len(state.points) - 1))
            state.summary = state.check = ""
        return {"FINISHED"}


def camera_intrinsics(scene, camera):
    width, height = scene.render.resolution_x, scene.render.resolution_y
    corners = camera.view_frame(scene=scene)
    x = [point.x / -point.z for point in corners]
    y = [point.y / -point.z for point in corners]
    fx, fy = width / (max(x) - min(x)), height / (max(y) - min(y))
    return [fx, fy, -min(x) * fx, max(y) * fy]


class ShotReaderSolveLandmarks(bpy.types.Operator):
    bl_idname = "shot_reader.solve_landmarks"
    bl_label = "Solve Camera from Landmarks"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (scene.camera is not None and scene.camera.data.type == "PERSP"
                and len(scene.shot_reader_calibration.points) >= 4)

    def execute(self, context):
        from . import ADDON, worker_environment

        scene = context.scene
        state = scene.shot_reader_calibration
        camera = scene.camera
        if scene.frame_current != state.frame:
            self.report({"ERROR"}, f"Return to landmark frame {state.frame} before solving.")
            return {"CANCELLED"}
        if scene.render.pixel_aspect_x != scene.render.pixel_aspect_y:
            self.report({"ERROR"}, "The landmark solve requires square pixels.")
            return {"CANCELLED"}
        context.view_layer.update()
        if any(not item.object or item.object.name not in scene.objects for item in state.points):
            self.report({"ERROR"}, "Remove landmarks whose source objects were deleted.")
            return {"CANCELLED"}
        width, height = scene.render.resolution_x, scene.render.resolution_y
        request = {
            "size": [width, height], "intrinsics": camera_intrinsics(scene, camera.data),
            "threshold": state.threshold, "refine_focal": state.refine_focal,
            "points": [{"name": item.name, "xyz": list(item.object.matrix_world @ Vector(item.local)),
                        "uv": list(item.uv), "holdout": item.holdout} for item in state.points],
        }
        prefs = context.preferences.addons[ADDON].preferences
        try:
            worker = subprocess.run([
                bpy.path.abspath(prefs.python), str(Path(__file__).with_name("camera_solve.py")),
            ], input=json.dumps(request), capture_output=True, encoding="utf-8", timeout=20,
                env=worker_environment(prefs), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if worker.returncode:
                raise ValueError(worker.stderr.strip() or "The camera solve failed.")
            solved = json.loads(worker.stdout)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        if not camera.get("shot_reader_landmark_camera"):
            state.original_camera = camera
            output = bpy.data.objects.new(f"{camera.name} · Landmark Match", camera.data.copy())
            output.data.animation_data_clear()
            scene.collection.objects.link(output)
            output["shot_reader_landmark_camera"] = True
        else:
            output = camera
        matrix = Matrix(solved["rotation"]).to_4x4()
        matrix.translation = solved["position"]
        output.matrix_world = matrix
        output.data.sensor_fit = "HORIZONTAL"
        output.data.lens = output.data.sensor_width * solved["focal_px"] / width
        cx, cy = solved["principal_px"]
        output.data.shift_x = .5 - cx / width
        output.data.shift_y = (cy - height / 2) / width
        scene.camera = output
        state.summary = f"Fit {solved['inlier_rms_px']:.2f} px · {solved['inlier_count']} inliers"
        holdout = solved["holdout_rms_px"]
        state.check = f"Holdout {holdout:.2f} px" if holdout is not None else "Mark an independent landmark as Check."
        report = state.report or bpy.data.texts.new(f"{scene.name} · Camera Solve.json")
        report.clear()
        report.write(json.dumps(solved, indent=2))
        state.report = report
        return {"FINISHED"}


class ShotReaderRestoreCamera(bpy.types.Operator):
    bl_idname = "shot_reader.restore_camera"
    bl_label = "Use Original Camera"
    bl_options = {"UNDO"}

    def execute(self, context):
        scene = context.scene
        source = scene.shot_reader_calibration.original_camera
        if source and source.name in scene.objects:
            scene.camera = source
            return {"FINISHED"}
        return {"CANCELLED"}


class SHOT_READER_PT_calibration(bpy.types.Panel):
    bl_label = "Camera Alignment"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "SHOT_READER_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from . import camera_motion

        state = context.scene.shot_reader_calibration
        layout = self.layout
        layout.operator("shot_reader.pick_landmark", icon="EYEDROPPER")
        if len(state.points):
            layout.label(text=f"Plate frame {state.frame}")
            row = layout.row()
            row.template_list("SHOT_READER_UL_landmarks", "", state, "points", state, "active", rows=3)
            row.operator("shot_reader.remove_landmark", text="", icon="REMOVE")
        layout.prop(state, "refine_focal")
        layout.prop(state, "threshold")
        layout.operator("shot_reader.solve_landmarks")
        if state.summary:
            layout.label(text=state.summary)
            layout.label(text=state.check)
            layout.prop(state, "report", text="Solve Report")
        if state.original_camera and context.scene.camera != state.original_camera:
            layout.operator("shot_reader.restore_camera")
        layout.separator()
        layout.operator("shot_reader.track_camera", icon="CAMERA_DATA")
        if camera_motion.status:
            layout.label(text=camera_motion.status)
        if state.track_summary:
            layout.label(text=state.track_summary)
            layout.prop(state, "track_report", text="Track Report")
        if state.track_previous_camera and context.scene.camera != state.track_previous_camera:
            layout.operator("shot_reader.restore_tracked_camera", icon="LOOP_BACK")


CLASSES = (ShotReaderLandmark, ShotReaderCalibration, SHOT_READER_UL_landmarks,
           ShotReaderPickLandmark, ShotReaderRemoveLandmark, ShotReaderSolveLandmarks,
           ShotReaderRestoreCamera, SHOT_READER_PT_calibration)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.shot_reader_calibration = bpy.props.PointerProperty(type=ShotReaderCalibration)


def unregister():
    del bpy.types.Scene.shot_reader_calibration
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
