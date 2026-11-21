"""Recover a locked mirror's static background after masking both performer views."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from foreground_matte import track_mattes
from mirror_solve import check_plate_motion
from room_clean import median_plate, raster_exclusions, read_frames


def recover(request, directory):
    width, height = request['size']
    scale = min(1., 768 / max(width, height))
    size = round(width * scale), round(height * scale)
    times, video = request['times'], request['clip']
    weights = Path(request['weights'])
    direct, images = track_mattes(video, times, size, request['views'][0], weights)
    reflected, _ = track_mattes(video, times, size, request['views'][1], weights)
    mirror = raster_exclusions(size, [request['mirror']])
    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    motion = check_plate_motion({'size': size, 'intrinsics': [[1, 1, 0, 0]], 'corners': [], 'native_scale': 1 / scale},
                                gray, direct | mirror)
    if motion.get('peak_frame_median_px', np.inf) > 1.3:
        raise ValueError('A reflected background texture requires a locked source shot with trackable set detail.')

    factor = min(1., 1920 / max(width, height))
    w, h = round(width * factor), round(height * factor)
    frames = read_frames(video, times, (w, h))
    mask = raster_exclusions((w, h), [request['mirror']])
    mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    py, px = np.where(mask)
    if not len(px):
        raise ValueError('The reflective surface falls outside the plate.')
    x0, x1 = max(0, int(px.min()) - 8), min(w, int(px.max()) + 9)
    y0, y1 = max(0, int(py.min()) - 8), min(h, int(py.max()) + 9)
    bounds = np.s_[y0:y1, x0:x1]
    aperture = mask[bounds]
    frames = frames[:, y0:y1, x0:x1]
    buffer = max(7, round(max(w, h) * .006))
    kernel = np.ones((buffer * 2 + 1, buffer * 2 + 1), np.uint8)
    visible = []
    for occluded in direct | reflected:
        scaled = cv2.resize(occluded.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        visible.append(aperture & ~cv2.dilate(scaled, kernel)[bounds].astype(bool))
    visible = np.stack(visible)
    held_out = np.arange(len(frames)) % 4 == 2
    observed = visible[~held_out].sum(axis=0) >= 2
    if observed.sum() < aperture.sum() * .20:
        raise ValueError('Too little of the reflective background becomes visible in this shot.')
    texture = median_plate(frames[~held_out], visible[~held_out])
    checks = observed & visible[held_out]
    differences = np.abs(frames[held_out].astype(np.int16) - texture.astype(np.int16)).mean(axis=3)
    # Fill holes on the preview wall with neighbouring observed rows. Keep a separate evidence matte.
    nearest = distance_transform_edt(~observed, sampling=(3., 1.), return_distances=False, return_indices=True)
    filled = texture[nearest[0], nearest[1]]
    cv2.imwrite(str(directory / 'wall.png'), filled)
    cv2.imwrite(str(directory / 'observed.png'), observed.astype(np.uint8) * 255)
    return {'uv_bounds': [x0 / w, y0 / h, x1 / w, y1 / h], 'frames': len(frames),
            'held_out_frames': int(held_out.sum()), 'held_out_samples': int(checks.sum()),
            'held_out_rgb_mae': float(differences[checks].mean()) if checks.any() else None,
            'observed_fraction': float(observed.sum() / aperture.sum()), 'observed_pixels': int(observed.sum()),
            'plate_motion': motion}


if __name__ == '__main__':
    path = Path(sys.argv[1])
    request = json.loads((path / 'request.json').read_text(encoding='utf-8'))
    result = recover(request, path)
    (path / 'reflection.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
