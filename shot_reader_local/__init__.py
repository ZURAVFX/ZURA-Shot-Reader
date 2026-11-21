bl_info = {
    "name": "Shot Reader Local",
    "author": "ZURA VFX",
    "version": (0, 4, 0),
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

from .blockout import build_blockout


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
job = None
status = ""


class ShotReaderPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON

    python: bpy.props.StringProperty(name="Python executable", subtype="FILE_PATH",
                                     default=str(PYTHON_PATH))
    script: bpy.props.StringProperty(name="Shot Reader script", subtype="FILE_PATH",
                                     default=str(RUNTIME / "shot_reader.py"))
    model: bpy.props.StringProperty(name="Model folder", subtype="DIR_PATH", default=str(MODEL_PATH))
    ffmpeg: bpy.props.StringProperty(name="FFmpeg folder", subtype="DIR_PATH")
    character_library: bpy.props.StringProperty(
        name="Custom character library", subtype="FILE_PATH",
        description="Optional Blender library containing a Shot Reader Character collection",
    )

    def draw(self, context):
        for field in ("python", "script", "model", "ffmpeg", "character_library"):
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


def poll_job():
    global job, status
    process, temp, scene, window, clip, character_library = job
    label = clip.stem
    code = process.poll()
    if code is None:
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
            status = "Analysis failed. See the Shot Reader Log in the Text Editor."
        else:
            sheets = json.loads(Path(temp.name, "sheet.json").read_text(encoding="utf-8"))
            blockouts = []
            missing_reference = False
            for index, sheet in enumerate(sheets, start=1):
                name = f"{label} · Shot {index:02}" if len(sheets) > 1 else label
                add_sheet(scene, name, json.dumps(sheet))
                text = scene.shot_reader.sheet
                blockout = build_blockout(sheet, text.name, scene.render, character_library)
                blockout.shot_reader.sheet = text
                missing_reference |= not attach_reference(blockout, clip, sheet)
                blockouts.append(blockout)
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
        env = os.environ.copy()
        search_paths = []
        if prefs.ffmpeg:
            search_paths.append(bpy.path.abspath(prefs.ffmpeg))
        search_paths.append(env.get("PATH") or os.defpath)
        if MACOS:
            search_paths.extend(("/opt/homebrew/bin", "/usr/local/bin"))
        env["PATH"] = os.pathsep.join(search_paths)
        if not all(shutil.which(tool, path=env["PATH"]) for tool in ("ffmpeg", "ffprobe")):
            self.report({"ERROR"}, "Set the FFmpeg folder in the add-on preferences.")
            return {"CANCELLED"}
        env.update(HF_HUB_OFFLINE="1", PYTHONUTF8="1")
        temp = tempfile.TemporaryDirectory(prefix="shot-reader-")
        command = [
            str(python), str(Path(__file__).with_name("infer.py")), str(script), str(clip), "--model", str(model),
            "--out", str(Path(temp.name, "sheet.json")),
        ]
        if MACOS:
            command.extend(("--runtime", "mlx"))
        for subject in settings.subjects.split("|"):
            if subject.strip():
                command.append("--subject=" + subject.strip())
        if settings.detect_cuts:
            command.append("--detect-cuts")
        try:
            with Path(temp.name, "run.log").open("wb") as log:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as error:
            temp.cleanup()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        job = process, temp, context.scene, context.window, clip, character_library
        status = "Reading shot on the GPU…"
        bpy.app.timers.register(poll_job, first_interval=0.4, persistent=False)
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
    rigs = [obj for obj in context.scene.objects if obj.type == "ARMATURE" and obj.get("shot_reader_height_m")]
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
        if job:
            layout.operator("shot_reader.cancel", icon="CANCEL")
        else:
            layout.operator("shot_reader.run", icon="PLAY")
        if status:
            layout.label(text=status)
        background = reference_background(context.scene)
        if background:
            layout.separator()
            layout.prop(background, "show_background_image", text="Show Source Clip", toggle=True, icon="IMAGE_REFERENCE")
            if background.show_background_image:
                layout.prop(background, "alpha", text="Source Opacity", slider=True)
        if selected_character(context):
            layout.operator(
                "shot_reader.pose_character",
                text="Exit Pose Mode" if context.mode == "POSE" else "Pose Character",
                icon="ARMATURE_DATA",
            )
        header, body = layout.panel("shot_reader_analysis_details", default_closed=True)
        header.label(text="Analysis Details")
        if body:
            body.prop(settings, "subjects")
            body.prop(settings, "sheet")
            if settings.sheet:
                body.operator("shot_reader.blockout", icon="SCENE_DATA")


@bpy.app.handlers.persistent
def stop_job(_unused=None):
    global job, status
    if job:
        process, temp, _scene, _window, _clip, _library = job
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
    ShotReaderCancel, ShotReaderBlockout, ShotReaderPoseCharacter, SHOT_READER_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.shot_reader = bpy.props.PointerProperty(type=ShotReaderSettings)
    bpy.app.handlers.load_pre.append(stop_job)


def unregister():
    stop_job()
    bpy.app.handlers.load_pre.remove(stop_job)
    del bpy.types.Scene.shot_reader
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
