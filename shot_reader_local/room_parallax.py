"""Estimate observed static room depth from calibrated camera translation."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from room_clean import raster_exclusions, read_frames
from room_depth import depth_mesh


def normalized(image):
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    mean = cv2.blur(luminance, (9, 9), borderType=cv2.BORDER_REFLECT101)
    variance = np.maximum(cv2.blur(luminance * luminance, (9, 9)) - mean * mean, 0.)
    sigma = np.sqrt(variance)
    return (luminance - mean) / np.maximum(.025, sigma), sigma


def solve(request, directory):
    width, height = request['size']
    camera = np.asarray(request['relative'], dtype=np.float32)
    lenses = np.asarray(request['lenses'], dtype=np.float32)
    seed = int(request['seed'])
    lengths = np.linalg.norm(camera[:, :3, 3], axis=1)
    baseline = float(lengths.max())
    fx, fy, cx, cy = lenses[seed]
    if baseline < .005:
        raise ValueError('Camera translation is too small to measure room depth from this shot.')
    hint = request.get('depth_hint')
    near, far = (hint if hint else [max(.05, baseline * 5.), min(250., baseline * 44.)])
    typical_depth = (near + far) * .5
    if not (0 < near < far and baseline * fx / typical_depth > 2.3):
        raise ValueError('Camera parallax is too small relative to the estimated set distance.')
    if far / near > 60:
        raise ValueError('The reference set depth range is too wide. Use a closer reference or restrict the visible set.')

    ranked = [int(i) for i in np.argsort(lengths)[::-1] if i != seed and lengths[i] >= baseline * .30]
    if len(ranked) < 4:
        raise ValueError('At least four translated source frames are needed to check depth independently.')
    checks = ranked[1:4:2]
    fit = [index for index in ranked[:9] if index not in checks]
    rgb = read_frames(request['clip'], request['times'], (width, height))
    frames = [normalized(frame) for frame in rgb]
    kernel = np.ones((7, 7), np.uint8)
    allowed = [1 - cv2.dilate(raster_exclusions((width, height), polygons).astype(np.uint8), kernel)
               for polygons in request['exclusions']]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    reference_rays = np.stack(((x - cx) / fx, -(y - cy) / fy, -np.ones_like(x)), axis=-1)
    reference, texture = frames[seed]
    rotations = reference_rays @ camera[:, :3, :3].transpose(0, 2, 1)[:, None]
    translation = camera[:, None, None, :3, 3]

    def compare(index, inverse_depth):
        xyz = rotations[index] + translation[index] * np.asarray(inverse_depth, dtype=np.float32)[..., None]
        z = -xyz[..., 2]
        f_x, f_y, c_x, c_y = lenses[index]
        x_map = xyz[..., 0] / np.maximum(z, 1.e-6) * f_x + c_x
        y_map = -xyz[..., 1] / np.maximum(z, 1.e-6) * f_y + c_y
        valid = (z > 0.) & (x_map >= 3.) & (y_map >= 3.) & (x_map < width - 4.) & (y_map < height - 4.)
        valid &= cv2.remap(allowed[index], x_map, y_map, cv2.INTER_NEAREST) > 0
        value = cv2.remap(frames[index][0], x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        error = np.minimum(2.75, abs(value - reference))
        return cv2.blur(np.where(valid, error, 2.75).astype(np.float32), (5, 5)), valid

    planes = np.linspace(1 / far, 1 / near, 124, dtype=np.float32)
    cost = []
    for depth in planes:
        estimates = np.stack([compare(index, depth)[0] for index in fit])
        take = min(3, len(fit))
        cost.append(np.partition(estimates, take - 1, axis=0)[:take].mean(axis=0))
    cost = np.stack(cost)
    best = cost.argmin(axis=0)
    yi, xi = np.indices(best.shape)
    least = cost[best, yi, xi]
    alternative = np.where(abs(np.arange(len(planes))[:, None, None] - best) > 4, cost, 5.).min(axis=0)
    before = cost[np.maximum(0, best - 1), yi, xi]
    after = cost[np.minimum(len(planes) - 1, best + 1), yi, xi]
    fractional = .5 * (before - after) / np.maximum(1.e-6, before - 2 * least + after)
    inverse = planes[best] + np.clip(fractional, -.5, .5) * (planes[1] - planes[0])
    depth = cv2.medianBlur(1. / inverse, 3)
    valid = (allowed[seed] > 0) & (texture > .024) & (least < .65) & ((alternative - least) > .028)
    valid &= (best > 0) & (best < len(planes) - 1)
    if valid.mean() < .045:
        raise ValueError('There is too little consistent textured depth in the moving plate.')

    heldout, comparison, verified = [], [], 0
    flat = float(np.median(1. / depth[valid]))
    for index in checks:
        fit_cost, inside = compare(index, 1. / depth)
        base_cost, flat_inside = compare(index, flat)
        mask = valid & inside & flat_inside
        if mask.sum() > width * height * .025:
            heldout.append(float(np.median(fit_cost[mask])))
            comparison.append(float(np.median(base_cost[mask])))
            verified += int(mask.sum())
    if not heldout or np.median(heldout) >= np.median(comparison) * .99:
        raise ValueError('Recovered depths did not improve held-out view reprojection compared with a flat surface.')
    # Retain only surfaces that also reproject into at least one independent view with consistent texture.
    confirmed = np.zeros(valid.shape, dtype=bool)
    for index in checks:
        error, inside = compare(index, 1. / depth)
        confirmed |= inside & (error < .63)
    info = depth_mesh(depth, valid & confirmed, (fx / width, fy / height, cx / width, cy / height), directory)
    cv2.imwrite(str(directory / 'plate.png'), rgb[seed])
    report = {'reference_frame': request['frame'], 'baseline_scene_units': baseline, 'fit_views': len(fit),
              'held_out_views': len(heldout), 'held_out_pixel_samples': verified,
              'held_out_photo_cost': float(np.median(heldout)), 'flat_plane_photo_cost': float(np.median(comparison)),
              'depth_range': [float(near), float(far)]}
    return {**info, 'parallax': report}


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    result = solve(request, folder)
    (folder / 'room.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
