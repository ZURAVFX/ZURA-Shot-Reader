"""Run Voyager's local CLI and preserve the source framing for Blender."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def analysis_cache_key(options, metadata, duration):
    """Build a cache identity from the plate, model, code and settings."""
    def file_identity(path):
        path = Path(path).resolve()
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]

    model = Path(options.model)
    files = [Path(options.clip), options.script, options.script.with_name('schema.md'), Path(__file__)]
    files.extend(sorted(path for path in model.iterdir() if path.suffix in {'.json', '.jinja', '.safetensors'}))
    if options.lens_model:
        files.extend([options.lens_model, Path(__file__).with_name('lens_fit.py')])
    identity = {'inputs': [file_identity(path) for path in files], 'metadata': metadata,
                'duration': duration, 'runtime': options.runtime, 'subjects': options.subject,
                'detect_cuts': options.detect_cuts}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()


def write_progress(output, message):
    output.with_name('analysis-progress.txt').write_text(message, encoding='utf-8')


def cache_entry(directory, identity, start, end):
    if not directory:
        return None
    key = hashlib.sha256(f'{identity}:{start:.9f}:{end:.9f}'.encode('ascii')).hexdigest()
    return directory / f'{key}.json'


def read_cached_sheet(path, source_range, metadata):
    if not path:
        return None
    try:
        sheet = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if (isinstance(sheet, dict) and isinstance(sheet.get('camera'), dict)
            and sheet.get('source_format') == metadata and sheet.get('source_range_seconds') == source_range):
        return sheet
    return None


def save_cached_sheet(path, sheet):
    if not path:
        return
    scratch = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.tmp',
                                         dir=path.parent, delete=False) as handle:
            scratch = Path(handle.name)
            json.dump(sheet, handle, ensure_ascii=False)
        scratch.replace(path)
    except (OSError, ValueError, TypeError) as error:
        print(f'Could not save local shot cache: {error}', file=sys.stderr, flush=True)
    finally:
        if scratch:
            try:
                scratch.unlink(missing_ok=True)
            except OSError:
                pass


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


def cuda_runner(reader, output):
    """Keep the quantised model on one GPU and reuse it across shots."""
    loaded = None

    def run(model_path, prompt, frames, max_tokens, four_bit):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        nonlocal loaded
        try:
            if loaded is None:
                if not torch.cuda.is_available():
                    raise ValueError('The Windows Shot Reader runtime needs an available CUDA GPU.')
                available = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
                device = max(range(len(available)), key=available.__getitem__)
                with torch.cuda.device(device):
                    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                free = available[device] / (1024 ** 3)
                if four_bit and free < 4.:
                    raise ValueError(f'Only {free:.1f} GiB of GPU memory is free. Free GPU memory before starting shot analysis.')
                write_progress(output, f'Loading the local model on CUDA · {free:.1f} GiB free')
                processor = AutoProcessor.from_pretrained(model_path)
                options = {'dtype': precision, 'device_map': {'': f'cuda:{device}'}}
                if four_bit:
                    options['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=precision)
                model = AutoModelForImageTextToText.from_pretrained(model_path, **options).eval()
                print(f'Local model device: {model.device}', flush=True)
                loaded = model, processor
            model, processor = loaded
            inputs = processor(text=[reader.chat(processor, prompt)], images=frames, return_tensors='pt').to(model.device)
            tokens = inputs['input_ids'].shape[1]
            write_progress(output, f'Analysing {len(frames)} frames on CUDA · {tokens:,} prompt tokens')
            print(f'Starting generation from {tokens:,} prompt tokens on {model.device}', flush=True)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            return processor.tokenizer.decode(generated[0, tokens:], skip_special_tokens=True)
        except torch.OutOfMemoryError as error:
            raise ValueError('The local shot model ran out of GPU memory. Free GPU memory and try again.') from error

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
    parser.add_argument("--lens-model", type=Path)
    parser.add_argument("--cache-directory", type=Path)
    options = parser.parse_args()
    metadata = video_format(options.clip)
    module_spec = importlib.util.spec_from_file_location("voyager_shot_reader", options.script)
    reader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reader)
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
    identity = None
    if options.cache_directory:
        try:
            identity = analysis_cache_key(options, metadata, duration)
        except OSError as error:
            print(f'Local shot cache unavailable: {error}', file=sys.stderr, flush=True)
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
    lens = None
    if options.lens_model:
        try:
            from lens_fit import PlateLens

            lens = PlateLens(options.lens_model)
        except Exception as error:
            print(f"Plate lens unavailable: {error}", file=sys.stderr, flush=True)
    sheets = []
    runner_ready = False
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        source_range = [start, end]
        cache = cache_entry(options.cache_directory, identity, start, end) if identity else None
        cached = read_cached_sheet(cache, source_range, metadata)
        if cached is not None:
            write_progress(options.out, f'Reusing local analysis for shot {index + 1} of {len(cuts) - 1}')
            sheets.append(cached)
            continue
        if not runner_ready:
            reader.run_transformers = mlx_runner() if options.runtime == 'mlx' else cuda_runner(reader, options.out)
            runner_ready = True
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
        message = f'Analysing shot {index + 1} of {len(cuts) - 1} on the local model'
        write_progress(options.out, message)
        print(message, flush=True)
        reader.main()
        sheet = json.loads(options.out.read_text(encoding="utf-8"))
        sheet["source_format"] = metadata
        sheet["source_range_seconds"] = source_range
        if lens:
            try:
                sheet["plate_lens"] = lens.estimate(options.clip, start, end)
            except Exception as error:
                print(f"Plate lens estimate skipped for shot {index + 1}: {error}", file=sys.stderr, flush=True)
        sheets.append(sheet)
        if not options.lens_model or 'plate_lens' in sheet:
            save_cached_sheet(cache, sheet)
    options.out.write_text(json.dumps(sheets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
