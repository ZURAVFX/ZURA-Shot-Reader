"""Build a clean set texture and depth mesh from a locked plate."""

import json
from pathlib import Path
import sys
import warnings

import cv2
import numpy as np

from foreground_matte import track_mattes
from fixture_fit import measure_fixtures
from mirror_solve import check_plate_motion
from room_depth import geometry
from set_continuation import continue_texture


def read_frames(video, times, size):
    capture = cv2.VideoCapture(video)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Could not open the linked source clip.')
        frames = []
        for time in times:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(fps * time))
            ok, frame = capture.read()
            if not ok:
                raise ValueError('The source clip does not cover the reconstruction range.')
            frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        return np.stack(frames)
    finally:
        capture.release()


def raster_exclusions(size, polygons):
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        polygon = np.asarray(polygon, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not ((polygon >= 0) & (polygon <= 1)).all():
            raise ValueError('Each fixed exclusion needs at least three normalised points.')
        cv2.fillPoly(mask, [np.rint(polygon * [width - 1, height - 1]).astype(np.int32)], 1)
    return mask.astype(bool)


def median_plate(frames, usable):
    clean = np.zeros_like(frames[0])
    for row in range(0, clean.shape[0], 64):
        rows = slice(row, row + 64)
        samples = np.where(usable[:, rows, :, None], frames[:, rows], np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered')
            clean[rows] = np.nan_to_num(np.nanmedian(samples, axis=0)).astype(np.uint8)
    return clean


def observed_projection(directory, plate, coverage, intrinsics):
    """Preserve measured plate colour where the depth mesh has no usable triangle."""
    height, width = coverage.shape
    with np.load(directory / 'room.npz', allow_pickle=False) as arrays:
        uv = arrays['uv']
        faces = arrays['faces']
        depth = float(-arrays['xyz'][:, 2].min())
    scale = 4
    w, h = (width + scale - 1) // scale, (height + scale - 1) // scale
    xy = np.rint(uv * [w, -h] + [0, h]).astype(np.int32)
    raster = np.zeros((h, w), np.uint8)
    for tri in xy[faces]:
        cv2.fillConvexPoly(raster, tri, 1)
    polygon_coverage = cv2.resize(raster, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    if (coverage & ~polygon_coverage).sum() < 250:
        return None
    # Cover measured set pixels beneath the mesh as well to avoid seams between triangles.
    projection = np.dstack((plate, coverage.astype(np.uint8) * 255))
    if not cv2.imwrite(str(directory / 'observed-set.png'), projection):
        raise ValueError('Could not save the observed set projection.')
    return {'normalized_intrinsics': list(intrinsics), 'depth_scene_units': depth + max(1., depth * .15),
            'measured_pixels': int(coverage.sum()),
            'measured_pixels_outside_depth_mesh': int((coverage & ~polygon_coverage).sum())}


def reconstruct(request, directory):
    rw, rh = request['size']
    factor = min(1., 768 / max(rw, rh))
    size = (round(rw * factor), round(rh * factor))
    mask, images = track_mattes(request['video'], request['times'], size, request['selection'], Path(request['weights']))
    fixed = raster_exclusions(size, request['exclude'])
    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    fx, fy, cx, cy = request['intrinsics']
    motion = check_plate_motion({'size': size, 'intrinsics': [[fx * size[0], fy * size[1], cx * size[0], cy * size[1]]],
                                 'corners': [], 'native_scale': 1 / factor}, gray, mask | fixed)
    displacement = motion.get('peak_frame_median_px')
    if displacement is None or displacement > 1.3:
        raise ValueError('Clean visible set requires a locked shot with trackable room texture. Use a single-frame set for this plate.')

    texture_factor = min(1., 1920 / max(rw, rh))
    texture_size = round(rw * texture_factor), round(rh * texture_factor)
    frames = read_frames(request['video'], request['times'], texture_size)
    fixed = raster_exclusions(texture_size, request['exclude'])
    margin = max(3, round(max(texture_size) * .004))
    expanded = np.stack([cv2.dilate(cv2.resize(item.astype(np.uint8), texture_size, interpolation=cv2.INTER_NEAREST),
                                    np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)) > 0 for item in mask])
    usable = ~(expanded | fixed)
    reference = int(np.argmin(abs(np.asarray(request['times']) - request['time'])))
    train = np.arange(len(frames)) % 4 != 2
    train[reference] = True
    observations = usable[train].sum(axis=0)
    coverage = observations >= 2
    revealed = coverage & ~usable[reference] & ~fixed
    clean = median_plate(frames[train], usable[train])
    clean[~coverage] = 0
    source = frames[reference].copy()
    source[coverage] = clean[coverage]
    cv2.imwrite(str(directory / 'clean-plate.png'), clean)
    cv2.imwrite(str(directory / 'depth-input.png'), source)
    cv2.imwrite(str(directory / 'coverage.png'), coverage.astype(np.uint8) * 255)

    differences, comparison_pixels = [], 0
    for image, observed in zip(frames[~train], usable[~train]):
        check = observed & revealed
        if check.any():
            differences.append(np.abs(image[check].astype(np.int16) - clean[check]).mean(axis=1))
            comparison_pixels += int(check.sum())
    info = geometry({**request, 'clean_plate': True}, directory)
    if info.get('wall_registration') and request.get('fixtures'):
        with np.load(directory / 'room.npz', allow_pickle=False) as arrays:
            groups, fixtures = measure_fixtures(arrays['xyz'], arrays['uv'], arrays['faces'], clean, request['fixtures'])
        if fixtures:
            np.save(directory / 'fixture-groups.npy', groups, allow_pickle=False)
            info['fixture_geometry'] = fixtures
    info['clean_set'] = {'sample_count': len(frames), 'held_out_frames': int((~train).sum()),
                         'coverage_fraction': float(coverage.mean()), 'newly_visible_fraction': float(revealed.mean()),
                         'newly_visible_pixels': int(revealed.sum()), 'checks': comparison_pixels,
                         'held_out_revealed_rgb_mae': float(np.concatenate(differences).mean()) if differences else None,
                         'plate_motion': motion}
    projection = observed_projection(directory, clean, coverage, request['intrinsics'])
    if projection:
        info['clean_set']['observed_projection'] = projection
        estimated = continue_texture(clean, coverage, fixed)
        if estimated:
            texture, measurements = estimated
            checks = []
            rejected = np.zeros(coverage.shape, np.uint8)
            for frame, visible in zip(frames[~train], usable[~train]):
                check = (texture[..., 3] > 0) & visible
                yy, xx = np.where(check)
                error = abs(frame[check].astype(np.int16) - texture[check, :3]).mean(axis=1)
                contradicted = error > 36.
                rejected[yy[contradicted], xx[contradicted]] = 1
                checks.append((yy, xx, error))
            exclude = cv2.dilate(rejected, np.ones((9, 9), np.uint8)) > 0
            rejected_pixels = int(np.count_nonzero(exclude & (texture[..., 3] > 0)))
            texture[exclude, 3] = 0
            errors = np.concatenate([error[~exclude[y, x]] for y, x, error in checks]) if checks else np.array([])
            measurements.update(estimated_pixels=int(np.count_nonzero(texture[..., 3])),
                                removed_by_plate_check=rejected_pixels,
                                held_out_plate_samples=int(sum(len(item[2]) for item in checks)),
                                retained_plate_samples=len(errors),
                                retained_plate_rgb_mae=float(errors.mean()) if len(errors) else None)
            if cv2.imwrite(str(directory / 'estimated-set.png'), texture):
                info['clean_set']['estimated_projection'] = {
                    **measurements, 'normalized_intrinsics': projection['normalized_intrinsics'],
                    'depth_scene_units': projection['depth_scene_units'] * 1.03,
                }
    return info


if __name__ == '__main__':
    path = Path(sys.argv[1])
    settings = json.loads((path / 'request.json').read_text(encoding='utf-8'))
    result = reconstruct(settings, path)
    (path / 'room.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
