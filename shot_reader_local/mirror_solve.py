"""Fit a mirror plane against a direct-view foreground matte, outside Blender."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation


def read_mattes(path, times, width, height):
    capture = cv2.VideoCapture(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Could not open the foreground matte video.')
        frames = []
        for time in times:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(time * fps))
            ok, frame = capture.read()
            if not ok:
                raise ValueError('The foreground matte must cover the entire scene range.')
            h, w = frame.shape[:2]
            if abs(w / h - width / height) > .008:
                raise ValueError('The matte must have the same aspect ratio as the plate.')
            gray = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            frames.append(gray >= 128)
        return np.stack(frames)
    finally:
        capture.release()


def check_plate_motion(data, gray, mattes):
    """Measure textured set points outside the performer and the reflecting surface."""
    width, height = map(int, data['size'])
    fx, fy, cx, cy = data['intrinsics'][0]
    rays = data['corners']
    blocked = mattes.any(axis=0).astype(np.uint8)
    if len(rays):
        mirror = np.rint(rays[:, :2] / -rays[:, 2:3] * [fx, -fy] + [cx, cy]).astype(np.int32)
        cv2.fillConvexPoly(blocked, mirror, 1)
    border = max(5, round(min(width, height) * .025))
    blocked = cv2.dilate(blocked, np.ones((2 * border + 1, 2 * border + 1), np.uint8))
    points = cv2.goodFeaturesToTrack(gray[0], 300, .012, 5, mask=(1 - blocked) * 255)
    shifts = []
    track_counts = []
    if points is not None and len(points) >= 12:
        for frame in gray[1:]:
            next_points, valid, _ = cv2.calcOpticalFlowPyrLK(gray[0], frame, points, None)
            if next_points is None:
                continue
            recovered, reverse, _ = cv2.calcOpticalFlowPyrLK(frame, gray[0], next_points, None)
            if recovered is None:
                continue
            roundtrip = np.linalg.norm((points - recovered)[:, 0], axis=1)
            good = (valid.ravel() != 0) & (reverse.ravel() != 0) & (roundtrip < .8)
            a, b = points[good, 0], next_points[good, 0]
            if len(a) < 12:
                continue
            homography, inliers = cv2.findHomography(a, b, cv2.RANSAC, 1.25)
            if homography is None or inliers.sum() < 12 or inliers.mean() < .65:
                continue
            selected = inliers.ravel().astype(bool)
            shifts.append(float(np.median(np.linalg.norm(b[selected] - a[selected], axis=1))) * float(data['native_scale']))
            track_counts.append(int(selected.sum()))
    if len(shifts) < max(2, (len(gray) - 1) // 2):
        return {'status': 'inconclusive', 'checked_frames': len(shifts)}
    return {'status': 'moving' if np.median(shifts) > 3. else 'near-static', 'checked_frames': len(shifts),
            'median_set_displacement_px': float(np.median(shifts)), 'peak_frame_median_px': max(shifts),
            'median_set_tracks': int(np.median(track_counts))}


def camera_varies(data):
    poses, intrinsics = data['cameras'], data['intrinsics']
    return bool(np.max(np.abs(poses - poses[0])) > 1e-5 or np.max(np.abs(intrinsics - intrinsics[0])) > .01)


def solve(data, mask):
    width, height = map(int, data['size'])
    points = data['virtual'].astype(np.float64)
    arrays = data['faces']
    if np.any(mask.sum(axis=(1, 2)) < width * height * .002) or np.any(mask.mean(axis=(1, 2)) > .9):
        raise ValueError('Use a matte with the foreground performer in white on a black background.')
    head = points[:, data['head']]
    normal0, corner_rays = data['normal'], data['corners']
    distance = float(data['distance'])
    cameras = data['cameras']
    focal = data['intrinsics'][:, :2] * [1., -1.]
    centre = data['intrinsics'][:, 2:]
    origins, axes = cameras[:, None, :3, 3], cameras[:, :3, :3]
    corner_tolerance = float(data['corner_tolerance'])
    pixel_scale = max(width, height) / 768
    expected_top = mask.any(axis=2).argmax(axis=1)
    offsets = np.arange(30, 156, 25) * pixel_scale
    rows = np.clip(np.rint(expected_top[:, None] + offsets), 0, height - 1).astype(int)
    samples = mask[np.arange(len(mask))[:, None], rows]
    expected_left = samples.argmax(axis=2)
    expected_right = width - 1 - samples[:, :, ::-1].argmax(axis=2)
    active_left = samples.any(axis=2) & (expected_left > 2)
    active_right = samples.any(axis=2) & (expected_right < width - 3)
    train = np.arange(len(points)) % 2 == 0

    def parameters(values):
        return Rotation.from_rotvec(np.radians((values[0], values[1], 0.))).apply(normal0), float(values[2])

    def pixels(vertices, indices):
        camera_points = (vertices - origins[indices]) @ axes[indices]
        z = -camera_points[..., 2:3]
        return camera_points[..., :2] / np.maximum(z, .01) * focal[indices, None, :] + centre[indices, None, :]

    def project(vertices, normal, multiplier, indices):
        virtual = (vertices - origins[indices]) * multiplier + origins[indices]
        direct = virtual - 2 * (virtual @ normal - distance)[..., None] * normal
        return pixels(direct, indices)

    all_frames = np.ones(len(points), dtype=bool)
    prior_corners = pixels(corner_rays[None], all_frames)

    def mirror_corners(normal):
        return distance * corner_rays / (corner_rays @ normal)[:, None]

    def corner_drift(normal, indices):
        candidate = pixels(mirror_corners(normal)[None], indices)
        return float(np.percentile(np.linalg.norm(candidate - prior_corners[indices], axis=2), 90) * float(data['native_scale']))

    def contour(projected, indices):
        column, row = np.moveaxis(projected, -1, 0)
        vertical = np.maximum(0., abs(row[:, None, :] - rows[indices, :, None]) - 3 * pixel_scale) * 1.7
        left = (column[:, None, :] + vertical).min(axis=2)
        right = (column[:, None, :] - vertical).max(axis=2)
        offscreen = np.maximum(0., np.maximum(-column, column - width + 1.))
        return left, right, (row + offscreen * 1.7).min(axis=1)

    def mirror_shape(normal):
        dots = corner_rays @ normal
        if np.any(dots >= -1e-5):
            return 1., 1.
        corners = mirror_corners(normal)
        edges = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(edges, axis=1)
        unit = edges / lengths[:, None]
        orthogonal = (unit * np.roll(unit, -1, axis=0)).sum(axis=1)
        opposite = (edges + np.roll(edges, -2, axis=0)) / lengths.mean()
        return float(np.sqrt(np.mean(orthogonal ** 2))), float(np.linalg.norm(opposite, axis=1).mean())

    def errors(values, indices):
        normal, multiplier = parameters(values)
        left, right, top = contour(project(head[indices], normal, multiplier, indices), indices)
        return np.r_[(left - expected_left[indices])[active_left[indices]],
                     (right - expected_right[indices])[active_right[indices]]], top - expected_top[indices]

    def objective(values):
        side, top = errors(values, train)
        residuals = np.r_[side, top * 1.2] / pixel_scale
        normal = parameters(values)[0]
        orthogonal, opposite = mirror_shape(normal)
        drift = max(0., corner_drift(normal, train) - corner_tolerance * .5)
        return (np.mean(200 * (np.sqrt(1 + (residuals / 10) ** 2) - 1)) + (orthogonal * 140) ** 2
                + (opposite * 120) ** 2 + (drift * 7) ** 2)

    def rasterise(vertices):
        result = np.zeros((height, width), dtype=np.uint8)
        triangles = np.clip(np.rint(vertices), -10000, 10000).astype(np.int32)[arrays]
        triangles = triangles[(triangles[:, :, 0].max(axis=1) >= 0) & (triangles[:, :, 0].min(axis=1) < width)
                              & (triangles[:, :, 1].max(axis=1) >= 0) & (triangles[:, :, 1].min(axis=1) < height)]
        for triangle in triangles:
            cv2.fillConvexPoly(result, triangle, 1)
        return result.astype(bool)

    def evaluate(values, indices):
        normal, multiplier = parameters(values)
        side, top = errors(values, indices)
        actual = mask[indices]
        rendered = np.stack([rasterise(frame) for frame in project(points[indices], normal, multiplier, indices)])
        iou = (actual & rendered).sum(axis=(1, 2)) / (actual | rendered).sum(axis=(1, 2))
        upscale = float(data['native_scale'])
        return {'head_side_median_px': float(np.median(abs(side))) * upscale if side.size else None,
                'head_top_median_px': float(np.median(abs(top))) * upscale,
                'foreground_iou': float(np.median(iou)), 'frames': int(np.count_nonzero(indices))}

    base = (0., 0., 1.)
    bounds = [(-10., 10.), (-10., 10.), (.84, 1.20)]
    coarse = differential_evolution(objective, bounds, x0=base, seed=7, popsize=8, maxiter=18, tol=.01, polish=False)
    fit = minimize(objective, coarse.x, method='Powell', bounds=bounds, options={'xtol': .0002})
    values = fit.x if fit.fun < coarse.fun else coarse.x
    normal, multiplier = parameters(values)
    baseline_check, candidate_check = evaluate(base, ~train), evaluate(values, ~train)
    # Apply only when independent frames agree with the fit.
    if baseline_check['foreground_iou'] >= candidate_check['foreground_iou']:
        raise ValueError('The matte refinement did not improve silhouette overlap on held-out frames.')
    drift = corner_drift(normal, ~train)
    if drift > corner_tolerance:
        raise ValueError('The foreground fit shifts the tracked mirror corners beyond the pixel tolerance.')
    return {'normal': normal.tolist(), 'scale': multiplier, 'adjustment_degrees': values[:2].tolist(),
            'corner_shape_error': mirror_shape(normal), 'mirror_corner_shift_p90_px': drift,
            'camera_motion_live': camera_varies(data),
            'baseline': baseline_check, 'refined': candidate_check}


if __name__ == '__main__':
    scratch, input_path = Path(sys.argv[1]), Path(sys.argv[2])
    with np.load(scratch / 'geometry.npz', allow_pickle=False) as archive:
        size = tuple(map(int, archive['size']))
        if '--track' in sys.argv:
            from foreground_matte import track_mattes

            request = json.loads(input_path.read_text(encoding='utf-8'))
            mask, images = track_mattes(request['clip'], archive['times'] + request.pop('start'), size,
                                      request, Path(request['weights']))
            gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
            motion = check_plate_motion(archive, gray, mask)
            if motion['status'] == 'moving' and not camera_varies(archive):
                raise ValueError('The plate moves while the Blender camera is locked. Track the plate camera before refining the mirror.')
            source = 'Local foreground tracking'
        else:
            mask = read_mattes(str(input_path), archive['times'], *size)
            motion = None
            source = 'Matte video'
        result = solve(archive, mask)
        result['matte_source'] = source
        if motion:
            result['plate_motion'] = motion
    (scratch / 'solve.json').write_text(json.dumps(result), encoding='utf-8')
