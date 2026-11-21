"""Run Voyager's local CLI and preserve the source framing for Blender."""

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def video_format(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", path],
        check=True, capture_output=True, encoding="utf-8",
    )
    stream = json.loads(result.stdout)["streams"][0]
    width, height = stream["width"], stream["height"]
    sar = stream.get("sample_aspect_ratio", "1:1")
    pixel_aspect = Fraction(sar.replace(":", "/")) if sar not in {"N/A", "0:1"} else Fraction(1)
    for side_data in stream.get("side_data_list", []):
        if round(side_data.get("rotation", 0)) % 180:
            width, height = height, width
            pixel_aspect = 1 / pixel_aspect
            break
    rate = Fraction(stream.get("avg_frame_rate", "0/1")) if stream.get("avg_frame_rate") != "0/0" else Fraction(0)
    if rate <= 0:
        rate = Fraction(stream["r_frame_rate"])
    return {
        "width": width, "height": height,
        "pixel_aspect": float(pixel_aspect), "fps": str(rate),
    }


def sample_frames(path, duration, size, count):
    from PIL import Image

    times = [index * duration / count for index in range(count)]
    frames = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, time in enumerate(times):
            target = Path(scratch, f"{index}.jpg")
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{time:.6f}", "-i", path,
                "-map", "0:v:0", "-frames:v", "1", "-q:v", "2", str(target),
            ], check=True)
            if target.is_file():
                with Image.open(target) as image:
                    frames.append(image.convert("RGB").resize(size, Image.Resampling.BICUBIC))
            elif frames:
                frames.append(frames[-1])
            else:
                raise ValueError("FFmpeg could not decode the first frame.")
    return times, frames


def mlx_runner():
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    loaded = None

    def run(model_path, prompt, frames, max_tokens, _four_bit):
        nonlocal loaded
        if loaded is None:
            loaded = load(model_path, trust_remote_code=False)
        model, processor = loaded
        formatted = apply_chat_template(processor, model.config, prompt, num_images=len(frames))
        with tempfile.TemporaryDirectory(prefix="shot-reader-mlx-") as directory:
            images = []
            for index, frame in enumerate(frames):
                path = Path(directory, f"{index:02}.png")
                frame.save(path, format="PNG")
                images.append(str(path))
            result = generate(
                model, processor, formatted, image=images, max_tokens=max_tokens,
                temperature=0.0, verbose=False,
            )
        return result.text

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("clip")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--detect-cuts", action="store_true")
    parser.add_argument("--runtime", choices=("cuda", "mlx"), default="cuda")
    options = parser.parse_args()
    metadata = video_format(options.clip)
    module_spec = importlib.util.spec_from_file_location("voyager_shot_reader", options.script)
    reader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reader)
    if options.runtime == "mlx":
        reader.run_transformers = mlx_runner()
    if hasattr(reader, "PROMPT"):
        reader.PROMPT += (
            "\nIf a person appears both directly and in a mirror, list them only once, at the "
            "physical position. Describe the mirror as a set piece, with its width, thickness, "
            "height, base_m (height of the lower edge), ground position, and yaw_deg. Include "
            "base_m on any elevated set pieces. Keep this rule even if no subjects were supplied."
        )
    aspect = metadata["width"] / metadata["height"] * metadata["pixel_aspect"]
    # Stay close to the model card's image budget, while preserving portrait geometry.
    scale = (448 * 256 / aspect) ** 0.5
    reader.SIZE = (max(28, round(scale * aspect)), max(28, round(scale)))
    reader.sample_frames = lambda path, duration: sample_frames(path, duration, reader.SIZE, reader.N_FRAMES)
    duration = reader.duration(options.clip)
    cuts = [0.0]
    if options.detect_cuts:
        scan = subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", options.clip,
            "-map", "0:v:0", "-vf", "scale=320:-2,scdet=t=14", "-an", "-f", "null", "-",
        ], check=True, capture_output=True, encoding="utf-8", errors="replace")
        for match in re.findall(r"lavfi\.scd\.time:\s*([\d.]+)", scan.stderr):
            cut = float(match)
            if cuts[-1] + 0.45 <= cut <= duration - 0.45:
                cuts.append(cut)
    cuts.append(duration)
    sheets = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        clip = options.clip
        if len(cuts) > 2:
            clip = str(options.out.with_name(f"take-{index + 1}.mp4"))
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", options.clip,
                "-t", str(end - start), "-map", "0:v:0", "-an", "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "16", "-y", clip,
            ], check=True)
        sys.argv = [
            str(options.script), clip, "--model", options.model,
            "--out", str(options.out),
        ]
        if options.runtime == "cuda":
            sys.argv.append("--load-in-4bit")
        sys.argv.extend("--subject=" + subject for subject in options.subject)
        print(f"Analysing shot {index + 1} of {len(cuts) - 1}", flush=True)
        reader.main()
        sheet = json.loads(options.out.read_text(encoding="utf-8"))
        sheet["source_format"] = metadata
        sheet["source_range_seconds"] = [start, end]
        sheets.append(sheet)
    options.out.write_text(json.dumps(sheets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
