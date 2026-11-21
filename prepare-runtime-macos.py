import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download


REPO = "voyager-so/shot-reader-8b"
REVISION = "5632e6b2de74b33d4b83c53e9c543702ea236f39"
root = Path(sys.argv[1])
source = root / "model-source"
converted = root / "model-mlx"

if not converted.exists():
    print("Downloading the original weights (about 17.5 GB).", flush=True)
    snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=source,
        allow_patterns=["model.safetensors", "*.json", "*.jinja", "schema.md", "shot_reader.py"],
        ignore_patterns=["lora/*"],
    )
    print("Converting the model to 4-bit MLX. This only runs once.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".convert-", dir=root) as staging:
        scratch = Path(staging)
        compatible_source = scratch / "source"
        compatible_source.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name != "config.json":
                (compatible_source / path.name).symlink_to(path.resolve())
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        text = config["text_config"]
        # MLX-VLM 0.8 reads the legacy aliases for Qwen3-VL's nested RoPE settings.
        rope = text["rope_parameters"]
        text.setdefault("rope_theta", rope["rope_theta"])
        text.setdefault("rope_scaling", {
            "rope_type": rope["rope_type"],
            "mrope_section": rope["mrope_section"],
            "mrope_interleaved": rope.get("mrope_interleaved", True),
        })
        config["vision_config"]["model_type"] = config["model_type"]
        config.setdefault("vocab_size", text["vocab_size"])
        (compatible_source / "config.json").write_text(json.dumps(config), encoding="utf-8")

        output = scratch / "model"
        subprocess.run([
            sys.executable, "-m", "mlx_vlm", "convert", "--hf-path", str(compatible_source),
            "--mlx-path", str(output), "--quantize", "--q-bits", "4", "--q-group-size", "64",
        ], check=True)
        verify_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if verify_config.get("quantization", {}).get("bits") != 4:
            raise SystemExit("Conversion did not produce a 4-bit MLX model.")
        output.rename(converted)

config_path = converted / "config.json"
if not config_path.is_file() or not any(converted.glob("model*.safetensors")):
    raise SystemExit(f"Invalid model folder: {converted}")
config = json.loads(config_path.read_text(encoding="utf-8"))
if config.get("model_type") != "qwen3_vl" or config.get("quantization", {}).get("bits") != 4:
    raise SystemExit(f"Expected a 4-bit Qwen3-VL model in {converted}")

for name in ("shot_reader.py", "schema.md"):
    original = source / name
    if not original.is_file():
        original = Path(hf_hub_download(REPO, name, revision=REVISION))
    shutil.copyfile(original, root / name)
print("Shot Reader runtime ready. Install the add-on ZIP in Blender and open the Shot Reader tab.")
