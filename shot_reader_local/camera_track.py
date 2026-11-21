"""Recover a fixed-lens camera path from a reference frame and static set depth."""

import json
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree


def anchor_features(image, vertices, intrinsics):
    height, width = image.shape
    fx, fy, cx, cy = intrinsics
    vertices = vertices[np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > .05)]
    if len(vertices) < 12:
        raise ValueError('Too little visible set geometry in the reference frame.')
    projected = vertices[:, :2] / vertices[:, 2:] * [fx, fy] + [cx, cy]
    valid = ((projected >= 0) & (projected < [width - 1, height - 1])).all(axis=1)
    projected, depth = projected[valid], vertices[valid, 2]
    if len(depth) < 12:
        raise ValueError('The reconstructed set is outside the reference camera.')
    coverage = np.zeros(image.shape, np.uint8)
    x, y = np.rint(projected).astype(int).T
    coverage[y, x] = 255
    coverage = cv.dilate(coverage, np.ones((9, 9), np.uint8))
    coverage = cv.erode(coverage, np.ones((7, 7), np.uint8))
    selected = []
    # Take features from each tile to avoid fitting one textured prop or one plane edge.
    for y0 in range(0, height, 96):
        for x0 in range(0, width, 96):
            tile = np.s_[y0:y0 + 96, x0:x0 + 96]
            marks = cv.goodFeaturesToTrack(image[tile], 25, .015, 7, mask=coverage[tile], blockSize=7)
            if marks is not None:
                selected.extend(marks[:, 0] + [x0, y0])
    if len(selected) < 12:
        raise ValueError('Too few textured points overlap the reconstructed set.')
    selected = np.asarray(selected, dtype=np.float32)
    distances, neighbors = cKDTree(projected).query(selected, k=5)
    nearby = depth[neighbors]
    good = (distances[:, -1] < 8.) & (nearby.max(axis=1) / nearby.min(axis=1) < 1.09)
    selected, z = selected[good], np.median(nearby[good], axis=1)
    uv = (selected - [cx, cy]) / [fx, fy]
    return selected, np.column_stack((uv * z[:, None], z)).astype(np.float64)


def solve(request, vertices):
    capture = cv.VideoCapture(request['clip'])
    try:
        if not capture.isOpened():
            raise ValueError('Could not open the linked plate.')
        fps = capture.get(cv.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('The linked plate does not report a valid frame rate.')
        source_frames = np.rint(np.asarray(request['times']) * fps).astype(int)
        seed = request['seed']
        rw, rh = request['size']
        scale = min(1., 960 / max(rw, rh))
        width, height = round(rw * scale), round(rh * scale)
        fx, fy, cx, cy = np.asarray(request['intrinsics']) * [width / rw, height / rh, width / rw, height / rh]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])

        def read(index):
            source = source_frames[index]
            if abs(capture.get(cv.CAP_PROP_POS_FRAMES) - source) > .25:
                capture.set(cv.CAP_PROP_POS_FRAMES, int(source))
            ok, image = capture.read()
            if not ok:
                raise ValueError('The linked plate does not cover the Blender scene range.')
            if abs(image.shape[1] / image.shape[0] - rw / rh) > .008:
                raise ValueError('Match the render aspect ratio to the plate before tracking.')
            return cv.cvtColor(cv.resize(image, (width, height), interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)

        first = read(seed)
        anchor, xyz = anchor_features(first, vertices, (fx, fy, cx, cy))
        if len(anchor) < 25:
            raise ValueError('Too few canvas features have reliable set depth. Reconstruct more visible set.')
        holdout = np.arange(len(anchor)) % 5 == 2
        tolerance = max(.85, request['threshold'] * scale)
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 35, .003)
        flip = np.diag([1., -1., -1.])
        results = [None] * len(source_frames)
        results[seed] = {'rotation': np.eye(3).tolist(), 'location': [0., 0., 0.], 'checks': int(holdout.sum()),
                         'check_px': 0., 'fixed_check_px': 0., 'motion_px': 0., 'inliers': int((~holdout).sum())}

        for direction in (range(seed + 1, len(results)), range(seed - 1, -1, -1)):
            indices = np.arange(len(anchor))
            pixels, previous = anchor, first
            rotation, translation = np.zeros((3, 1)), np.zeros((3, 1))
            for index in direction:
                image = read(index)
                next_pixels, valid, _ = cv.calcOpticalFlowPyrLK(previous, image, pixels, None, winSize=(23, 23),
                                                               maxLevel=3, criteria=criteria)
                if next_pixels is None or valid is None:
                    raise ValueError(f'Lost the set tracks at scene frame {request["frame_start"] + index}.')
                recovered, back, _ = cv.calcOpticalFlowPyrLK(image, previous, next_pixels, None, winSize=(23, 23),
                                                            maxLevel=3, criteria=criteria)
                if recovered is None or back is None:
                    raise ValueError(f'Could not verify the set tracks at scene frame {request["frame_start"] + index}.')
                good = valid.ravel().astype(bool) & back.ravel().astype(bool)
                good &= np.linalg.norm(recovered - pixels, axis=1) < .7
                good &= ((next_pixels >= [3, 3]) & (next_pixels < [width - 3, height - 3])).all(axis=1)
                indices, pixels, previous = indices[good], next_pixels[good], image
                if abs(index - seed) % 8 == 0 and len(pixels) >= 15:
                    # Anchor the surviving paths back to the original texture periodically to limit drift.
                    direct, seen, _ = cv.calcOpticalFlowPyrLK(first, image, anchor[indices], pixels.copy(),
                                                            winSize=(23, 23), maxLevel=2, criteria=criteria,
                                                            flags=cv.OPTFLOW_USE_INITIAL_FLOW)
                    if direct is not None and seen is not None:
                        close = seen.ravel().astype(bool) & (np.linalg.norm(direct - pixels, axis=1) < 1.2)
                        pixels[close] = direct[close]
                fitting = ~holdout[indices]
                checks = ~fitting
                if fitting.sum() < 10 or checks.sum() < 3:
                    raise ValueError(f'Set tracks ran out at scene frame {request["frame_start"] + index}. Pick a more central reference frame or expand the reconstructed set.')

                ok, guess_r, guess_t, selected = cv.solvePnPRansac(
                    xyz[indices[fitting]], pixels[fitting], K, None, rotation.copy(), translation.copy(), True,
                    iterationsCount=180, reprojectionError=tolerance, confidence=.998, flags=cv.SOLVEPNP_ITERATIVE,
                )
                if not ok or selected is None or len(selected) < max(9, int(.5 * fitting.sum())):
                    raise ValueError(f'The set depth and plate tracks disagree at scene frame {request["frame_start"] + index}.')
                inliers = np.flatnonzero(fitting)[selected.ravel()]
                rotation, translation = cv.solvePnPRefineLM(xyz[indices[inliers]], pixels[inliers], K, None, guess_r, guess_t)
                projection, _ = cv.projectPoints(xyz[indices[checks]], rotation, translation, K, None)
                fixed = np.linalg.norm(anchor[indices[checks]] - pixels[checks], axis=1) / scale
                check = np.linalg.norm(projection[:, 0] - pixels[checks], axis=1) / scale
                r = cv.Rodrigues(rotation)[0]
                if np.any(xyz[indices[inliers]] @ r[2] + translation[2, 0] <= 0):
                    raise ValueError('The tracked camera flipped behind the set.')
                results[index] = {'rotation': (flip @ r.T @ flip).tolist(), 'location': (-flip @ r.T @ translation).ravel().tolist(),
                                  'check_px': float(np.median(check)), 'fixed_check_px': float(np.median(fixed)),
                                  'checks': int(checks.sum()), 'inliers': len(inliers),
                                  'motion_px': float(np.median(np.linalg.norm(anchor[indices] - pixels, axis=1))) / scale}
    finally:
        capture.release()

    heldout = np.array([frame['check_px'] for frame in results])
    fixed = np.array([frame['fixed_check_px'] for frame in results])
    movement = float(np.percentile([frame['motion_px'] for frame in results], 90))
    status = 'near-static' if movement < .8 else 'tracked'
    if status == 'tracked' and (np.median(heldout) >= np.median(fixed) or np.percentile(heldout, 90) > request['threshold'] * 1.5):
        raise ValueError('The moving camera did not agree with independent set tracks. Improve the reference geometry or tolerance.')
    return {'status': status, 'frames': results, 'anchors': len(anchor), 'motion_p90_px': movement,
            'heldout_median_px': float(np.median(heldout)), 'heldout_p90_px': float(np.percentile(heldout, 90)),
            'fixed_median_px': float(np.median(fixed))}


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    report = solve(request, np.load(directory / 'set.npy', allow_pickle=False))
    (directory / 'camera.json').write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
