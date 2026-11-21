param(
    [string]$Python = "python",
    [string]$Root = (Join-Path $env:LOCALAPPDATA 'ZURAVFX\ShotReader')
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-Checked([scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "A setup command exited with code $LASTEXITCODE."
    }
}

foreach ($program in @('ffmpeg', 'ffprobe')) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
        throw "Install FFmpeg and put ffmpeg and ffprobe on PATH, then run setup again."
    }
}

$runtime = Join-Path $Root 'runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    Invoke-Checked { & $Python -c "import struct,sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12),(3,13)) and struct.calcsize('P')==8 else '64-bit Python 3.11 to 3.13 required')" }
    Invoke-Checked { & $Python -m venv (Join-Path $Root 'runtime') }
}
Invoke-Checked { & $runtime -m pip install 'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128 }
Invoke-Checked { & $runtime -m pip install 'transformers==5.8.0' 'accelerate==1.14.0' 'bitsandbytes==0.50.2' pillow 'huggingface-hub==1.30.0' }
Invoke-Checked { & $runtime -m pip install opencv-python scipy onnxruntime }
Invoke-Checked { & $runtime -c "import torch, bitsandbytes; from transformers import AutoModelForImageTextToText, AutoProcessor; assert torch.cuda.is_available(), 'CUDA unavailable. Check your NVIDIA driver.'" }

Write-Host 'Downloading Voyager Shot Reader 8B (about 17.5 GB).'
$env:HF_HUB_DISABLE_XET = '1'
$download = @'
from pathlib import Path
import shutil
import sys
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
model = root / "model"
snapshot_download(
    "voyager-so/shot-reader-8b",
    revision="5632e6b2de74b33d4b83c53e9c543702ea236f39",
    local_dir=model,
    allow_patterns=["model.safetensors", "*.json", "*.jinja", "schema.md", "shot_reader.py"],
    ignore_patterns=["lora/*"],
)
for name in ("shot_reader.py", "schema.md"):
    shutil.copyfile(model / name, root / name)
'@
$downloaderPath = Join-Path $Root 'fetch-model.py'
Set-Content -LiteralPath $downloaderPath -Value $download -Encoding utf8
Invoke-Checked { & $runtime $downloaderPath $Root }
Write-Host "Shot Reader runtime ready at $Root. Open the Shot Reader tab in Blender."
