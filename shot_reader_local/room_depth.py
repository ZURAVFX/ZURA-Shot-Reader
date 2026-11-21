"""Recover visible room surfaces in calibrated Blender camera coordinates."""

import json
import sys
from pathlib import Path


def anchor_to_wall(depth, valid, intrinsics, wall):
    """Use a calibrated mirror plane to register the observed wall and scale adjacent depth."""
    import cv2
    import numpy as np

    height, width = depth.shape
    fx, fy, cx, cy = intrinsics
    y, x = np.mgrid[:height, :width]
    rays = np.stack(((x + .5 - cx * width) / (fx * width),
                     (cy * height - y - .5) / (fy * height), -np.ones_like(x)), axis=-1)
    plane = rays @ np.asarray(wall['normal'])
    expected = wall['distance'] / np.where(abs(plane) > .001, plane, np.nan)
    quad = np.asarray(wall['rim'])
    stencil = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(stencil, np.rint(quad * [width, height]).astype(np.int32), 1)
    border = cv2.distanceTransform(1 - stencil, cv2.DIST_L2, 5)
    plane_valid = (expected > .05) & np.isfinite(expected)
    band = valid & plane_valid & (border > 5) & (border < max(16, height * .090))
    holdout = (x // 4 + y // 4) % 4 == 1
    sample = band & ~holdout
    if sample.sum() < 250 or (band & holdout).sum() < 50:
        return depth, None
    ratio = expected / np.maximum(depth, .05)
    scale = float(np.median(ratio[sample]))
    if not .30 < scale < 5. or np.median(abs(ratio[sample] / scale - 1.)) > .13:
        return depth, None
    band_check = band & holdout
    before = np.median(abs(depth - expected)[band_check])
    calibrated = depth * scale
    after_scale = np.median(abs(calibrated - expected)[band_check])
    if after_scale > max(.08, before * .72):
        return depth, None
    # Snap coherent wall samples only. Retain the sink, ledges and other depths off the plane.
    relative = abs(ratio / scale - 1.)
    limit = quad[:, 1].max() + .15
    wall_weight = np.clip((.11 - relative) / .055, 0., 1.) * np.clip((limit - (y + .5) / height) / .05, 0., 1.)
    wall_weight = np.where(valid & plane_valid, wall_weight, 0.)
    calibrated += wall_weight * np.where(plane_valid, expected - calibrated, 0.)
    return calibrated, {'scale': scale, 'held_out_wall_pixels': int(band_check.sum()),
                        'wall_depth_before': float(np.median(depth[band_check])),
                        'wall_depth_after': float(np.median(calibrated[band_check])),
                        'wall_error_before': float(before), 'wall_error_after': float(np.median(abs(calibrated - expected)[band_check]))}


def geometry(request, directory):
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_dilation

    from lens_fit import PlateLens, recover_projection

    model = PlateLens(request["model"])
    if request.get("clean_plate"):
        texture = Image.open(directory / "depth-input.png").convert("RGB")
        texture.thumbnail((768, 768), Image.Resampling.LANCZOS)
        known = np.asarray(Image.open(directory / "coverage.png").convert("L").resize(texture.size), dtype=np.uint8) == 255
        rgb, points, confidence, metric_scale = model.image_maps(np.asarray(texture))
    else:
        rgb, points, confidence, metric_scale = model.frame_maps(request["video"], request["time"])
        known = np.ones(confidence.shape, dtype=bool)
    _, projection_error, shift = recover_projection(points, confidence)
    if not np.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError("MoGe did not return a usable metric scale.")
    depth = (points[..., 2] + shift) * metric_scale
    valid = (confidence > .5) & np.isfinite(depth) & (depth > .05) & (depth < 150)
    height, width = valid.shape
    excluded = Image.new("L", (width, height))
    draw = ImageDraw.Draw(excluded)
    for polygon in request["exclude"]:
        if len(polygon) < 3 or any(len(p) != 2 or not np.isfinite(p).all() or min(p) < 0 or max(p) > 1 for p in polygon):
            raise ValueError("Each exclusion area needs at least three normalised x/y points.")
        draw.polygon([(u * (width - 1), v * (height - 1)) for u, v in polygon], fill=255)
    valid &= ~binary_dilation((np.asarray(excluded) > 0) | ~known, iterations=2)
    fx, fy, cx, cy = request["intrinsics"]
    if min(fx, fy) <= 0:
        raise ValueError("The room estimate needs a calibrated perspective camera.")
    ratios = []
    for u, v, camera_depth in request.get("scale_anchors", []):
        x, y = min(width - 1, round(u * width)), min(height - 1, round(v * height))
        if 0 <= x < width and 0 <= y < height and camera_depth > 0 and valid[y, x]:
            ratios.append(camera_depth / float(depth[y, x]))
    if ratios:
        candidate = float(np.median(ratios))
        if max(ratios) / min(ratios) > 1.3 or not .1 <= candidate <= 10:
            raise ValueError("Room scale anchors disagree. Keep marks on measured static surfaces.")
        depth *= candidate

    wall_report = None
    if request.get('wall_anchor') and not ratios:
        depth, wall_report = anchor_to_wall(depth, valid, (fx, fy, cx, cy), request['wall_anchor'])
    geometry_info = depth_mesh(depth, valid, (fx, fy, cx, cy), directory)
    if wall_report:
        geometry_info['wall_registration'] = wall_report
    if request.get("clean_plate"):
        from shutil import copyfile

        copyfile(directory / "clean-plate.png", directory / "plate.png")
    else:
        Image.fromarray(rgb).save(directory / "plate.png")
    return {**geometry_info, "model_projection_error": projection_error, "scale_anchors": len(ratios)}


def depth_mesh(depth, valid, intrinsics, directory):
    """Triangulate supported depth samples and retain their source raster coordinates."""
    import numpy as np

    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    step = 2
    y, x = np.mgrid[0:height:step, 0:width:step]
    z = depth[y, x]
    u, v = (x + .5) / width, (y + .5) / height
    xyz = np.stack(((u - cx) * z / fx, -(v - cy) * z / fy, -z), axis=-1)
    indices = np.arange(z.size).reshape(z.shape)
    a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]
    candidates = np.concatenate([np.stack((a, c, b), -1).reshape(-1, 3),
                                 np.stack((b, c, d), -1).reshape(-1, 3)])
    selected = valid[y, x].ravel()[candidates].all(axis=1)
    safe = candidates[selected]
    face_z = z.ravel()[safe]
    continuous = (face_z.max(axis=1) - face_z.min(axis=1)) <= .08 * face_z.min(axis=1)
    faces = safe[continuous]
    if len(faces) < 100:
        raise ValueError("The mask left too little usable room geometry.")
    used, compact = np.unique(faces, return_inverse=True)
    uv = np.stack((u, 1 - v), axis=-1).reshape(-1, 2)
    np.savez_compressed(directory / "room.npz", xyz=xyz.reshape(-1, 3)[used].astype(np.float32),
                        faces=compact.reshape(-1, 3).astype(np.int32), uv=uv[used].astype(np.float32))
    return {"vertices": len(used), "triangles": len(faces), "valid_fraction": float(valid.mean())}


if __name__ == "__main__":
    try:
        result = geometry(json.load(sys.stdin), Path(sys.argv[1]))
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
