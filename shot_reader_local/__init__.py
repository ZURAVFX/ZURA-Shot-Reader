bl_info = {
    "name": "Shot Reader Local",
    "author": "ZURA VFX",
    "version": (0, 46, 0),
    "blender": (5, 2, 0),
    "location": "3D View > Sidebar > Shot Reader",
    "category": "3D View",
}

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import bpy

from . import arm_alignment, automatic, calibration, camera_motion, colour_bake, digit_alignment, hair_reconstruction, handheld, head_alignment, mirror_refine, motion_cleanup, performance, reflected_room, room
from .blockout import apply_plate_lens, build_blockout
from .geometry_export import ShotReaderExportGeometry


ADDON = __package__
MACOS = platform.system() == "Darwin"
if MACOS:
    RUNTIME = Path.home() / "Library" / "Application Support" / "ZURAVFX" / "ShotReader"
    PYTHON_PATH = RUNTIME / "runtime" / "bin" / "python"
    MODEL_PATH = RUNTIME / "model-mlx"
else:
    RUNTIME = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ZURAVFX" / "ShotReader"
    PYTHON_PATH = RUNTIME / "runtime" / "Scripts" / "python.exe"
    MODEL_PATH = RUNTIME / "model"
local_paths = Path(__file__).with_name("local_paths.json")
LOCAL = json.loads(local_paths.read_text(encoding="utf-8")) if local_paths.is_file() else {}
job = None
status = ""


class ShotReaderPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON

    python: bpy.props.StringProperty(name="Python executable", subtype="FILE_PATH",
                                     default=LOCAL.get("python", str(PYTHON_PATH)))
    script: bpy.props.StringProperty(name="Shot Reader script", subtype="FILE_PATH",
                                     default=LOCAL.get("script", str(RUNTIME / "shot_reader.py")))
    model: bpy.props.StringProperty(name="Model folder", subtype="DIR_PATH", default=LOCAL.get("model", str(MODEL_PATH)))
    ffmpeg: bpy.props.StringProperty(name="FFmpeg folder", subtype="DIR_PATH", default=LOCAL.get("ffmpeg", ""))
    lens_model: bpy.props.StringProperty(name="MoGe-2 ONNX model", subtype="FILE_PATH",
                                          default=LOCAL.get("lens_model", str(RUNTIME / "moge-2-vitb.onnx")))
    character_library: bpy.props.StringProperty(
        name="Custom character library", subtype="FILE_PATH",
        description="Optional Blender library containing a Shot Reader Character collection",
    )
    rgb_mocap: bpy.props.StringProperty(name="RGB MoCap launcher", subtype="FILE_PATH")
    body_comfy: bpy.props.StringProperty(name="SAM · ComfyUI folder", subtype="DIR_PATH",
                                         default=LOCAL.get("body_comfy", ""))
    body_python: bpy.props.StringProperty(name="SAM · ComfyUI Python", subtype="FILE_PATH",
                                          default=LOCAL.get("body_python", ""))
    body_weights: bpy.props.StringProperty(name="SAM · Body checkpoint", subtype="FILE_PATH",
                                           default=LOCAL.get("body_weights", ""))
    matte_weights: bpy.props.StringProperty(name="SAM · Foreground matte checkpoint", subtype="FILE_PATH",
                                            default=LOCAL.get("matte_weights", ""))
    person_weights: bpy.props.StringProperty(name='Local person segmentation checkpoint', subtype='FILE_PATH',
                                             default=LOCAL.get('person_weights', ''))
    face_weights: bpy.props.StringProperty(name='Local Face Landmarker checkpoint', subtype='FILE_PATH',
                                           default=LOCAL.get('face_weights', ''))
    pose_weights: bpy.props.StringProperty(name='Local ViTPose WholeBody model', subtype='FILE_PATH',
                                           default=LOCAL.get('pose_weights', ''))
    hair_weights: bpy.props.StringProperty(name='Local hair segmentation model', subtype='FILE_PATH',
                                           default=LOCAL.get('hair_weights', ''))

    def draw(self, context):
        for field in ("python", "script", "model", "ffmpeg", "lens_model", "character_library",
                      "body_comfy", "body_python", "body_weights", "matte_weights", "person_weights", "face_weights", "pose_weights", "hair_weights", "rgb_mocap"):
            self.layout.prop(self, field)


class ShotReaderSettings(bpy.types.PropertyGroup):
    clip: bpy.props.StringProperty(name="Video clip", subtype="FILE_PATH")
    detect_cuts: bpy.props.BoolProperty(
        name="Detect hard cuts",
        description="Make a separate scene for each detected shot",
        default=True,
    )
    subjects: bpy.props.StringProperty(
        name="Subject hints",
        description="Optional labels for people or objects to track. Separate with |, e.g. P01: person | O01: chair",
    )
    sheet: bpy.props.PointerProperty(
        name="Shot sheet (JSON)", type=bpy.types.Text,
        description="The model's estimates for camera, subject positions and paths, and set pieces",
    )
    reference: bpy.props.PointerProperty(type=bpy.types.MovieClip)
    local_reconstruction: bpy.props.PointerProperty(type=bpy.types.Text)
    match_lens: bpy.props.BoolProperty(name="Estimate lens from plate", default=True)
    reuse_analysis: bpy.props.BoolProperty(name="Reuse Local Shot Analysis", default=True)
    body_region: bpy.props.FloatVectorProperty(size=4, min=0., max=1., default=(0., 0., 1., 1.))
    foreground_region: bpy.props.FloatVectorProperty(size=4, min=0., max=1.)
    foreground_region_frame: bpy.props.IntProperty()
    body_direct_head: bpy.props.FloatVectorProperty(size=2, default=(-1., -1.))
    body_anchor_frame: bpy.props.IntProperty()
    mirror_refinement_report: bpy.props.PointerProperty(type=bpy.types.Text)
    mirror_refinement_anchor: bpy.props.PointerProperty(type=bpy.types.Object)
    body_mirror: bpy.props.PointerProperty(
        name="Mirror surface", type=bpy.types.Object,
        description="Optional mirror mesh with its reflective face on local negative Y",
        poll=lambda _settings, obj: obj.type == "MESH",
    )


def reference_background(scene):
    clip = scene.shot_reader.reference
    if clip and scene.camera:
        return next((item for item in scene.camera.data.background_images if item.clip == clip), None)
    return None


def attach_reference(scene, path, sheet):
    """Attach the original movie as a viewport-only camera overlay."""
    try:
        clip = bpy.data.movieclips.load(str(path))
    except RuntimeError:
        return False
    source_range = sheet.get("source_range_seconds")
    if isinstance(source_range, list) and source_range and isinstance(source_range[0], (int, float)):
        clip.frame_offset = round(source_range[0] * scene.render.fps / scene.render.fps_base)
    clip.frame_start = scene.frame_start
    camera = scene.camera.data
    background = camera.background_images.new()
    background.source = "MOVIE_CLIP"
    background.clip = clip
    background.display_depth = "FRONT"
    background.frame_method = "FIT"
    background.alpha = 0.5
    background.show_background_image = False
    camera.show_background_images = True
    scene.shot_reader.reference = clip
    return True


def add_sheet(scene, label, raw):
    sheet = json.loads(raw)
    if not isinstance(sheet, dict) or not isinstance(sheet.get("camera"), dict):
        raise ValueError("Expected a Shot Reader JSON object with a camera section.")
    text = bpy.data.texts.new(f"{label} · Shot Sheet.json")
    text.write(json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")
    scene.shot_reader.sheet = text


def refresh():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def worker_environment(prefs):
    env = os.environ.copy()
    paths = [bpy.path.abspath(prefs.ffmpeg)] if prefs.ffmpeg else []
    paths.append(env.get("PATH") or os.defpath)
    if MACOS:
        paths.extend(("/opt/homebrew/bin", "/usr/local/bin"))
    env.update(PATH=os.pathsep.join(paths), HF_HUB_OFFLINE="1", PYTHONUTF8="1")
    return env


def start_worker(command, env, temp):
    with Path(temp.name, "run.log").open("wb") as log:
        return subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def update_lens_sheet(scene, fit):
    text = scene.shot_reader.sheet
    if not text:
        return
    try:
        sheet = json.loads(text.as_string())
    except ValueError:
        return
    if fit is None:
        sheet.pop("plate_lens", None)
    else:
        sheet["plate_lens"] = fit
    if text.users > 1:
        text = text.copy()
        scene.shot_reader.sheet = text
    text.clear()
    text.write(json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")


def poll_job():
    global job, status
    process, temp, scene, window, clip, character_library, camera = job
    label = clip.stem
    code = process.poll()
    if code is None:
        progress = Path(temp.name, 'analysis-progress.txt')
        try:
            if not camera and (message := progress.read_text(encoding='utf-8').strip()) and status != message:
                status = message
                refresh()
        except OSError:
            pass
        return 0.4
    try:
        if status == "Cancelling…":
            status = "Cancelled"
        elif code:
            log_path = Path(temp.name, "run.log")
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 65536))
                detail = log.read().decode("utf-8", errors="replace")
            error_log = bpy.data.texts.new(f"{label} · Shot Reader Log")
            error_log.write(detail)
            status = "Lens estimate failed. See Shot Reader Log." if camera else "Analysis failed. See Shot Reader Log."
        elif camera:
            fit = json.loads(Path(temp.name, "sheet.json").read_text(encoding="utf-8"))
            apply_plate_lens(camera, fit)
            update_lens_sheet(scene, fit)
            status = f"Plate lens applied · {camera.lens:.1f} mm"
        else:
            sheets = json.loads(Path(temp.name, "sheet.json").read_text(encoding="utf-8"))
            blockouts = []
            missing_reference = False
            for index, sheet in enumerate(sheets, start=1):
                name = f"{label} · Shot {index:02}" if len(sheets) > 1 else label
                add_sheet(scene, name, json.dumps(sheet))
                text = scene.shot_reader.sheet
                blockout = build_blockout(sheet, text.name, scene.render, character_library)
                blockout.shot_reader.clip = str(clip)
                blockout.shot_reader.sheet = text
                blockout.shot_reader.match_lens = scene.shot_reader.match_lens
                missing_reference |= not attach_reference(blockout, clip, sheet)
                blockouts.append(blockout)
            automatic.queue_blockouts(scene, blockouts)
            if any(candidate == window for candidate in bpy.context.window_manager.windows) and window.scene == scene:
                window.scene = blockouts[0]
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.spaces.active.region_3d.view_perspective = "CAMERA"
                        area.spaces.active.shading.color_type = "MATERIAL"
                        area.spaces.active.overlay.show_overlays = False
            status = "Blockout ready" if len(blockouts) == 1 else f"{len(blockouts)} blockout scenes ready"
            if missing_reference:
                status += " (Blender could not load the reference video)"
    except (OSError, ValueError, TypeError, ReferenceError, RuntimeError, KeyError) as error:
        status = str(error)
    finally:
        temp.cleanup()
        job = None
        refresh()
    return None


class ShotReaderRun(bpy.types.Operator):
    bl_idname = "shot_reader.run"
    bl_label = "Create Blockout from Clip"
    bl_description = "Analyse the clip locally and open an editable blockout with a camera"

    @classmethod
    def poll(cls, context):
        return job is None

    def execute(self, context):
        global job, status
        if MACOS and platform.machine() != "arm64":
            self.report({"ERROR"}, "Use the Apple silicon version of Blender with this runtime.")
            return {"CANCELLED"}
        prefs = context.preferences.addons[ADDON].preferences
        settings = context.scene.shot_reader
        clip = Path(bpy.path.abspath(settings.clip))
        python = Path(bpy.path.abspath(prefs.python))
        script = Path(bpy.path.abspath(prefs.script))
        model = Path(bpy.path.abspath(prefs.model))
        character_library = bpy.path.abspath(prefs.character_library) if prefs.character_library else None
        required = {
            "Video clip": clip,
            "Python executable": python,
            "Shot Reader script": script,
            "Shot Reader schema": script.with_name("schema.md"),
            "Model configuration": model / "config.json",
        }
        for label, path in required.items():
            if not path.is_file():
                self.report({"ERROR"}, f"{label} missing: {path}")
                return {"CANCELLED"}
        if not any(path.is_file() for path in model.glob("model*.safetensors")):
            self.report({"ERROR"}, f"Model weights missing: {model}")
            return {"CANCELLED"}
        env = worker_environment(prefs)
        if not all(shutil.which(tool, path=env["PATH"]) for tool in ("ffmpeg", "ffprobe")):
            self.report({"ERROR"}, "Set the FFmpeg folder in the add-on preferences.")
            return {"CANCELLED"}
        temp = tempfile.TemporaryDirectory(prefix="shot-reader-")
        command = [
            str(python), str(Path(__file__).with_name("infer.py")), str(script), str(clip), "--model", str(model),
            "--out", str(Path(temp.name, "sheet.json")),
        ]
        if MACOS:
            command.extend(("--runtime", "mlx"))
        lens_model = Path(bpy.path.abspath(prefs.lens_model))
        if settings.match_lens and lens_model.is_file():
            command.extend(("--lens-model", str(lens_model)))
        for subject in settings.subjects.split("|"):
            if subject.strip():
                command.append("--subject=" + subject.strip())
        if settings.detect_cuts:
            command.append("--detect-cuts")
        if settings.reuse_analysis:
            command.extend(("--cache-directory", str(RUNTIME / "analysis-cache")))
        try:
            process = start_worker(command, env, temp)
        except OSError as error:
            temp.cleanup()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        job = process, temp, context.scene, context.window, clip, character_library, None
        status = "Reading shot on the GPU…"
        bpy.app.timers.register(poll_job, first_interval=0.4, persistent=False)
        return {"FINISHED"}


class ShotReaderMatchLens(bpy.types.Operator):
    bl_idname = "shot_reader.match_lens"
    bl_label = "Match Lens to Source"
    bl_description = "Estimate the focal length from the source footage and apply it to the scene camera"

    @classmethod
    def poll(cls, context):
        camera = context.scene.camera
        return job is None and camera is not None and camera.data.type == "PERSP" and reference_background(context.scene) is not None

    def execute(self, context):
        global job, status
        scene = context.scene
        prefs = context.preferences.addons[ADDON].preferences
        reference = reference_background(scene).clip
        clip = Path(bpy.path.abspath(reference.filepath, library=reference.library))
        python = Path(bpy.path.abspath(prefs.python))
        model = Path(bpy.path.abspath(prefs.lens_model))
        if not model.is_file():
            self.report({"ERROR"}, "Set the MoGe-2 ONNX model in Shot Reader preferences.")
            return {"CANCELLED"}
        env = worker_environment(prefs)
        if not clip.is_file() or not python.is_file() or not shutil.which("ffmpeg", path=env["PATH"]):
            self.report({"ERROR"}, "Check that the source clip, Python and FFmpeg are available.")
            return {"CANCELLED"}
        fps = scene.render.fps / scene.render.fps_base
        start = max(0, reference.frame_offset) / fps
        end = start + (scene.frame_end - scene.frame_start + 1) / fps
        if scene.shot_reader.sheet:
            try:
                sheet = json.loads(scene.shot_reader.sheet.as_string())
                start, end = map(float, sheet.get("source_range_seconds", (start, end)))
            except (ValueError, TypeError):
                pass
        temp = tempfile.TemporaryDirectory(prefix="shot-reader-lens-")
        command = [
            str(python), str(Path(__file__).with_name("lens_fit.py")), str(model), str(clip),
            str(start), str(end), str(Path(temp.name, "sheet.json")),
        ]
        try:
            process = start_worker(command, env, temp)
        except OSError as error:
            temp.cleanup()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        job = process, temp, scene, context.window, clip, None, scene.camera.data
        status = "Estimating the lens from the source clip…"
        bpy.app.timers.register(poll_job, first_interval=0.4, persistent=False)
        return {"FINISHED"}


class ShotReaderRestoreLens(bpy.types.Operator):
    bl_idname = "shot_reader.restore_lens"
    bl_label = "Restore Previous Lens"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        camera = context.scene.camera
        return job is None and camera is not None and "shot_reader_original_focal" in camera.data

    def execute(self, context):
        global status
        camera = context.scene.camera.data
        camera.lens = camera["shot_reader_original_focal"]
        camera.sensor_fit = camera.get("shot_reader_original_sensor_fit", camera.sensor_fit)
        for key in ("shot_reader_original_focal", "shot_reader_original_sensor_fit", "shot_reader_lens_method"):
            if key in camera:
                del camera[key]
        update_lens_sheet(context.scene, None)
        status = "Previous lens restored"
        return {"FINISHED"}


class ShotReaderCancel(bpy.types.Operator):
    bl_idname = "shot_reader.cancel"
    bl_label = "Cancel Analysis"

    @classmethod
    def poll(cls, context):
        return job is not None and status != "Cancelling…"

    def execute(self, context):
        global status
        status = "Cancelling…"
        job[0].terminate()
        return {"FINISHED"}


class ShotReaderBlockout(bpy.types.Operator):
    bl_idname = "shot_reader.blockout"
    bl_label = "Rebuild from Shot Sheet"
    bl_description = "Create a separate scene with an initial camera and animated subject proxies"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.shot_reader.sheet is not None and context.window is not None

    def execute(self, context):
        source = context.scene
        prefs = context.preferences.addons[ADDON].preferences
        library = bpy.path.abspath(prefs.character_library) if prefs.character_library else None
        text = source.shot_reader.sheet
        try:
            sheet = json.loads(text.as_string())
            scene = build_blockout(sheet, text.name, source.render, library)
        except (TypeError, ValueError, KeyError, RuntimeError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        scene.shot_reader.sheet = text
        background = reference_background(source)
        if background:
            if attach_reference(scene, bpy.path.abspath(background.clip.filepath), sheet):
                new_background = reference_background(scene)
                new_background.alpha = background.alpha
                new_background.show_background_image = background.show_background_image
            else:
                self.report({"WARNING"}, "Blender could not load the reference video.")
        context.window.scene = scene
        return {"FINISHED"}


def selected_character(context):
    rigs = [obj for obj in context.scene.objects if obj.type == "ARMATURE"
            and (obj.get("shot_reader_height_m") or obj.get("shot_reader_performance_rig")) and not obj.hide_get()]
    selection = context.active_object
    while selection:
        match = next((rig for rig in rigs if rig == selection or rig.parent == selection), None)
        if match:
            return match
        selection = selection.parent
    return rigs[0] if len(rigs) == 1 else None


class ShotReaderPoseCharacter(bpy.types.Operator):
    bl_idname = "shot_reader.pose_character"
    bl_label = "Pose Character"
    bl_description = "Select the character rig and enter Pose Mode"

    @classmethod
    def poll(cls, context):
        return context.mode in {"OBJECT", "POSE"} and selected_character(context) is not None

    def execute(self, context):
        rig = selected_character(context)
        if context.mode == "POSE":
            bpy.ops.object.mode_set(mode="OBJECT")
            if context.active_object == rig:
                return {"FINISHED"}
        bpy.ops.object.select_all(action="DESELECT")
        rig.select_set(True)
        context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="POSE")
        if context.area and context.area.type == "VIEW_3D":
            context.space_data.overlay.show_overlays = True
        return {"FINISHED"}


class SHOT_READER_PT_panel(bpy.types.Panel):
    bl_label = "Shot Reader"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Shot Reader"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.shot_reader
        layout.prop(settings, "clip")
        layout.prop(settings, "detect_cuts")
        if automatic.run:
            layout.operator('shot_reader.cancel_local_reconstruction', icon='CANCEL')
        elif job:
            layout.operator("shot_reader.cancel", icon="CANCEL")
        else:
            layout.operator('shot_reader.reconstruct_locally', icon='PLAY')
            layout.operator("shot_reader.run", icon="PLAY")
        if automatic.status:
            layout.label(text=automatic.status)
        if settings.local_reconstruction:
            layout.prop(settings, 'local_reconstruction', text='Reconstruction Report')
        if status:
            layout.label(text=status)
        background = reference_background(context.scene)
        if background:
            layout.separator()
            layout.prop(background, "show_background_image", text="Show Source Clip", toggle=True, icon="IMAGE_REFERENCE")
            if background.show_background_image:
                layout.prop(background, "alpha", text="Source Opacity", slider=True)
            layout.operator("shot_reader.match_lens", icon="CAMERA_DATA")
            layout.operator("shot_reader.export_geometry", icon="RENDER_ANIMATION")
            camera = context.scene.camera.data
            if "shot_reader_original_focal" in camera:
                layout.label(text=f"{camera.lens:.1f} mm · {camera['shot_reader_lens_method']}")
                layout.operator("shot_reader.restore_lens", icon="LOOP_BACK")
        if selected_character(context):
            layout.operator(
                "shot_reader.pose_character",
                text="Exit Pose Mode" if context.mode == "POSE" else "Pose Character",
                icon="ARMATURE_DATA",
            )
        header, body = layout.panel("shot_reader_analysis_details", default_closed=True)
        header.label(text="Analysis Details")
        if body:
            body.prop(settings, "match_lens")
            body.prop(settings, "reuse_analysis")
            body.prop(settings, "subjects")
            body.prop(settings, "sheet")
            if settings.sheet:
                body.operator("shot_reader.blockout", icon="SCENE_DATA")


@bpy.app.handlers.persistent
def stop_job(_unused=None):
    global job, status
    automatic.stop()
    room.stop_job()
    colour_bake.stop_job()
    handheld.stop_job()
    head_alignment.stop_job()
    arm_alignment.stop_job()
    digit_alignment.stop_job()
    hair_reconstruction.stop_job()
    reflected_room.stop_job()
    camera_motion.stop_job()
    mirror_refine.stop_job()
    performance.stop_job()
    if job:
        process, temp, _scene, _window, _clip, _library, _camera = job
        if process.poll() is None:
            process.kill()
            process.wait()
        temp.cleanup()
        job = None
    if bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.unregister(poll_job)
    status = ""


CLASSES = (
    ShotReaderPreferences, ShotReaderSettings, ShotReaderRun,
    ShotReaderMatchLens, ShotReaderRestoreLens, ShotReaderCancel,
    ShotReaderBlockout, ShotReaderPoseCharacter, SHOT_READER_PT_panel,
    automatic.ShotReaderReconstructLocally, automatic.ShotReaderCancelLocal,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.utils.register_class(ShotReaderExportGeometry)
    bpy.types.Scene.shot_reader = bpy.props.PointerProperty(type=ShotReaderSettings)
    bpy.app.handlers.load_pre.append(stop_job)
    calibration.register()
    camera_motion.register()
    room.register()
    bpy.utils.register_class(reflected_room.ShotReaderBuildReflectedRoom)
    performance.register()
    bpy.utils.register_class(colour_bake.ShotReaderBakePerformerColour)
    bpy.utils.register_class(handheld.ShotReaderTrackHandheld)
    bpy.utils.register_class(handheld.ShotReaderToggleHandheldGrip)
    bpy.utils.register_class(head_alignment.ShotReaderAlignHead)
    bpy.utils.register_class(arm_alignment.ShotReaderAlignArms)
    bpy.utils.register_class(arm_alignment.ShotReaderToggleArmAlignment)
    bpy.utils.register_class(digit_alignment.ShotReaderAlignFingers)
    bpy.utils.register_class(digit_alignment.ShotReaderToggleDigitAlignment)
    bpy.utils.register_class(hair_reconstruction.ShotReaderReconstructHair)
    bpy.utils.register_class(hair_reconstruction.ShotReaderToggleViewportHair)
    bpy.utils.register_class(motion_cleanup.ShotReaderStabilizeArmMotion)
    mirror_refine.register()


def unregister():
    stop_job()
    mirror_refine.unregister()
    bpy.utils.unregister_class(handheld.ShotReaderToggleHandheldGrip)
    bpy.utils.unregister_class(handheld.ShotReaderTrackHandheld)
    bpy.utils.unregister_class(colour_bake.ShotReaderBakePerformerColour)
    bpy.utils.unregister_class(head_alignment.ShotReaderAlignHead)
    bpy.utils.unregister_class(arm_alignment.ShotReaderAlignArms)
    bpy.utils.unregister_class(arm_alignment.ShotReaderToggleArmAlignment)
    bpy.utils.unregister_class(digit_alignment.ShotReaderAlignFingers)
    bpy.utils.unregister_class(digit_alignment.ShotReaderToggleDigitAlignment)
    bpy.utils.unregister_class(hair_reconstruction.ShotReaderReconstructHair)
    bpy.utils.unregister_class(hair_reconstruction.ShotReaderToggleViewportHair)
    bpy.utils.unregister_class(motion_cleanup.ShotReaderStabilizeArmMotion)
    performance.unregister()
    bpy.utils.unregister_class(reflected_room.ShotReaderBuildReflectedRoom)
    room.unregister()
    camera_motion.unregister()
    calibration.unregister()
    bpy.app.handlers.load_pre.remove(stop_job)
    del bpy.types.Scene.shot_reader
    bpy.utils.unregister_class(ShotReaderExportGeometry)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
