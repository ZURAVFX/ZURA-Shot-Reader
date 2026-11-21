"""Estimate a centred pinhole lens from MoGe-2 affine point maps."""

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import median
import subprocess


def recover_projection(points, mask):
    import numpy as np
    from scipy.optimize import least_squares

    height, width = mask.shape
    rows = np.linspace(0, height - 1, min(height, 56)).round().astype(int)
    cols = np.linspace(0, width - 1, min(width, 56)).round().astype(int)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    xyz = points[yy, xx]
    valid = (mask[yy, xx] > 0.5) & np.isfinite(xyz).all(axis=-1)
    if valid.sum() < 128 or valid.mean() < 0.2:
        raise ValueError("Too few reliable image points")
    x, y, z = xyz[valid].T.astype(float)
    uv = np.stack(((xx[valid] + 0.5 - width / 2) / width,
                   (yy[valid] + 0.5 - height / 2) / width), axis=-1)
    xy = np.stack((x, y), axis=-1)
    scale = max(float(np.median(np.abs(z))), 1e-4)
    lower = -z.min() + 1e-5 * scale

    def projection(shift):
        ray = xy / (z + shift[0])[:, None]
        focal = np.sum(ray * uv) / max(np.sum(ray * ray), 1e-12)
        return focal, ray * focal - uv

    fit = least_squares(lambda shift: projection(shift)[1].ravel(),
                       [max(0, lower + 0.05 * scale)], bounds=(lower, np.inf),
                       loss="soft_l1", f_scale=0.04, max_nfev=80)
    focal, error = projection(fit.x)
    residual = float(np.sqrt(np.mean(error ** 2)))
    if not fit.success or not 0.20 <= focal <= 8 or residual > 0.10:
        raise ValueError("Inconsistent camera projection")
    return float(focal), residual, float(fit.x[0])


class PlateLens:
    def __init__(self, model):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.model_name = "MoGe-2 ViT-S" if "vits" in Path(model).name else "MoGe-2 ViT-B"

    def frame_maps(self, video, time):
        import numpy as np
        from PIL import Image

        frame = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{time:.6f}", "-i", str(video),
            "-map", "0:v:0", "-frames:v", "1",
            "-vf", "scale='min(768,iw)':'min(768,ih)':force_original_aspect_ratio=decrease",
            "-f", "image2pipe", "-c:v", "png", "-",
        ], check=True, capture_output=True).stdout
        if not frame:
            raise ValueError("Could not read the reference frame")
        with Image.open(BytesIO(frame)) as source:
            sample = np.asarray(source.convert("RGB"))
        return self.image_maps(sample)

    def image_maps(self, sample):
        import numpy as np

        sample = np.asarray(sample, dtype=np.float32)
        image = np.ascontiguousarray(sample.transpose(2, 0, 1)[None] / 255.0)
        points, mask, scale = self.session.run(["points", "mask", "metric_scale"], {
            "image": image, "num_tokens": np.asarray(1400, dtype=np.int64),
        })
        return sample.astype(np.uint8), points[0], mask[0], float(scale.reshape(-1)[0])

    def frame_focal(self, video, time):
        _, points, mask, _ = self.frame_maps(video, time)
        focal, residual, _ = recover_projection(points, mask)
        return focal, residual

    def estimate(self, video, start, end):
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Invalid reference time range")
        samples = []
        for fraction in (0.15, 0.45, 0.75):
            try:
                samples.append(self.frame_focal(video, start + fraction * (end - start)))
            except ValueError:
                continue
        if len(samples) < 2:
            raise ValueError("The reference did not produce enough reliable lens estimates")
        focals = [focal for focal, _ in samples]
        if max(focals) / min(focals) > 1.20:
            raise ValueError("Frame estimates disagree. Use a single shot with a fixed lens")
        focal = median(focals)
        return {
            "method": self.model_name, "fx_normalized": focal,
            "horizontal_fov_deg": math.degrees(2 * math.atan(0.5 / focal)),
            "sample_count": len(samples), "frame_fx_normalized": focals,
            "fit_error": median(error for _, error in samples),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("start", type=float)
    parser.add_argument("end", type=float)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    fit = PlateLens(args.model).estimate(args.video, args.start, args.end)
    args.out.write_text(json.dumps(fit), encoding="utf-8")


if __name__ == "__main__":
    main()
