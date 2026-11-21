# ZURA Shot Reader preview for Apple silicon

This build uses the 4-bit MLX conversion of the same Voyager Shot Reader 8B model as the Windows build. It retains the editable camera and set, poseable mannequin, shot sheet and source overlay.

## Requirements

- Apple silicon, macOS 14 or later and Blender 5.2 for Apple silicon.
- Native arm64 Python 3.12, FFmpeg and FFprobe. With Homebrew, run `brew install python@3.12 ffmpeg`.
- Allow about 35 GB of free disk space for the environment, original weights and converted model. The original download is about 17.5 GB. Allow for extra unified memory during conversion. A 32 GB model is a sensible starting point for the first test. A minimum supported memory configuration has yet to be measured on Mac.

## Install

1. Open Terminal in the extracted preview folder and run `bash setup-runtime-macos.sh`. This installs the local MLX environment and converts the pinned Voyager checkpoint once.
2. Open Blender Preferences, choose **Add-ons → Install from Disk**, and select the versioned `ZURA-Shot-Reader` ZIP extracted alongside this guide. Enable **Shot Reader Local**.
3. In Blender's 3D View, press **N** and open **Shot Reader**. Choose a short video and click **Create Blockout from Clip**. Use **Pose Character** and **Show Source Clip** to check the result.

The default runtime is `~/Library/Application Support/ZURAVFX/ShotReader`. If Blender cannot find FFmpeg, set **FFmpeg folder** in the add-on preferences, typically to `/opt/homebrew/bin`. If inference fails, open the generated **Shot Reader Log** in Blender's Text Editor.

The Blender add-on and installer are a preview. Apple Metal acceleration, conversion memory and end-to-end behaviour on real Apple hardware still need a Mac test.

The preview installer configures the local shot reader and basic blockout. The separate SAM 3D Body performance runtime, room reconstruction and full automatic pipeline have been validated on Windows with NVIDIA. Check the clip with **Show Source Clip**, pose the included rig and export scene geometry from Blender with **Export Geometry for AI Render**.
