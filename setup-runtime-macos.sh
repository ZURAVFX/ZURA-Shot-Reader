#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Run setup in a native Apple silicon terminal on macOS." >&2
    exit 1
fi
macos_version="$(sw_vers -productVersion)"
if (( ${macos_version%%.*} < 14 )); then
    echo "macOS 14 or later is required." >&2
    exit 1
fi
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
for program in ffmpeg ffprobe; do
    if ! command -v "$program" >/dev/null; then
        echo "Install FFmpeg first, for example with: brew install ffmpeg" >&2
        exit 1
    fi
done

root="$HOME/Library/Application Support/ZURAVFX/ShotReader"
installer="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${SHOT_READER_PYTHON:-python3.12}"
runtime="$root/runtime/bin/python"
if [[ ! -x "$runtime" ]]; then
    if ! command -v "$python" >/dev/null; then
        echo "Install native Python 3.12 first, for example with: brew install python@3.12" >&2
        exit 1
    fi
    "$python" -c 'import platform, sys; sys.exit(0 if platform.machine() == "arm64" and sys.version_info[:2] == (3, 12) else "Native arm64 Python 3.12 is required.")'
    "$python" -m venv "$root/runtime"
fi
"$runtime" -m pip install --disable-pip-version-check \
    'mlx==0.32.4' 'mlx-vlm==0.8.0' 'transformers==5.23.0'
"$runtime" -c 'import mlx.core as mx; from mlx_vlm import generate, load; assert mx.metal.is_available(), "Metal is unavailable"; print("MLX and Metal are ready.")'

export HF_HUB_DISABLE_XET=1
export PYTHONUTF8=1
"$runtime" "$installer/prepare-runtime-macos.py" "$root"
