# ZURA Shot Reader

Turn a reference shot into an editable Blender blockout. The add-on reads video locally with [Voyager Shot Reader 8B](https://huggingface.co/voyager-so/shot-reader-8b), then builds a camera, placeable set geometry and poseable characters from the shot sheet.

## Install on Windows

1. Install Blender 5.2, [64-bit Python 3.11 to 3.13](https://www.python.org/downloads/windows/) and [FFmpeg](https://ffmpeg.org/download.html). Make `python`, `ffmpeg` and `ffprobe` available on PATH. Use a current NVIDIA driver and allow at least 25 GB of free disk space. Local inference was tested on an RTX 4080 with 16 GB of VRAM.
2. Download `setup-runtime.ps1` from the [release](https://github.com/ZURAVFX/ZURA-Shot-Reader/releases/latest) and run it in PowerShell. It creates a dedicated environment and downloads the model weights (about 17.5 GB). If PowerShell blocks the downloaded script, inspect it, then run `Unblock-File .\setup-runtime.ps1` before running `.\setup-runtime.ps1`.
3. Download the add-on ZIP from the same release. In Blender Preferences, choose **Add-ons → Install from Disk**. Select the ZIP and enable **Shot Reader Local**.
4. In the 3D View, press **N** and open **Shot Reader**. Choose a video and click **Create Blockout from Clip**.

The default runtime location is `%LOCALAPPDATA%\ZURAVFX\ShotReader`. If you set up the runtime elsewhere, point the add-on preferences to its Python executable, `shot_reader.py` and the model folder. Restart Blender if it was open when you changed PATH.

## Work with the blockout

- **Show Source Clip** and **Source Opacity** blend the original video over the camera view as you scrub. The reference video remains linked by path.
- **Export Geometry for AI Render** exports the current frame or the whole shot as PNGs from that camera, with clay materials and the physical mirror. The editable scene keeps its original materials.
- **Pose Character** selects the included mannequin's rig. Move its hand and foot IK targets, adjust the elbow and knee poles, or rotate the hips, chest and head.
- **Analysis Details** holds optional subject hints and the output JSON. Use **Rebuild from Shot Sheet** to regenerate the scene from an edited sheet.
- **Reuse Local Shot Analysis** lets repeated reconstruction attempts reuse a completed shot sheet for the same unchanged clip, subject hints, models and analysis settings. Turn it off under **Analysis Details** to run fresh analysis.
- **Detect hard cuts** creates a scene per detected shot. **Reconstruct Clip Locally** runs each scene through the local pipeline and writes a reconstruction report to each scene.

## Local reconstruction

The Windows installer sets up the Voyager runtime for the basic blockout. The full automatic workflow also uses a SAM 3D Body build of ComfyUI. In the add-on preferences, point **SAM · ComfyUI folder** and **SAM · ComfyUI Python** to that installation, and select the local body, foreground matte and person segmentation checkpoints. The lens and visible-set stages use **MoGe-2 ONNX model**. The face, pose and hair checkpoints enable their corresponding refinement stages.

The resulting rig keeps the captured animation and the **Artist Corrections** controls for keying pose adjustments. Use **Pose Character** to access the controls and the linked plate to check alignment.

## Bring your own character

Set **Custom character library** in the add-on preferences to a Blender file containing a `Shot Reader Character` collection. Include one armature and its mesh objects, set the armature's `shot_reader_height_m` and `shot_reader_floor_m` custom properties in metres, and leave its object transforms applied. Set `shot_reader_forward` to `1` for +Y forward or `-1` for -Y forward. Optional `shot_reader_rig_name` names the rig in the Outliner. The collection is appended once per detected person, so each instance is poseable on its own. Adobe Mixamo rigs can be prepared locally using [Adobe's generator](https://www.adobe.com/products/substance3d/plugins/mixamo-in-blender.html).

## Source

`shot_reader_local/` contains the add-on. Run Blender in background with `--python scripts/build_mannequin.py` to regenerate the original mannequin library. The setup script retrieves Voyager's inference code and model from the pinned upstream revision under the model's Apache-2.0 licence.
