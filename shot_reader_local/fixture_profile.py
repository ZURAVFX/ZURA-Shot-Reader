"""Fit a gooseneck profile where the observed plate supports a tubular faucet."""

import cv2
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree


def profile(values):
    x, y, span, rise, tilt, short, long = values
    angle = np.linspace(0., np.pi, 55)
    arc_y = y - rise * np.sin(angle)
    arc_x = x + span * np.cos(angle) + tilt * (arc_y - y)
    left_y = y + np.linspace(0., short, 31)
    right_y = y + np.linspace(long, 0., 39)
    left_x = x - span + tilt * (left_y - y)
    right_x = x + span + tilt * (right_y - y)
    return np.column_stack((np.r_[right_x[:-1], arc_x, left_x[1:]], np.r_[right_y[:-1], arc_y, left_y[1:]]))


def sample(field, line):
    pixels = line.astype(np.float32)
    return cv2.remap(field, pixels[:, :1], pixels[:, 1:], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0]


def chrome_mask(plate):
    values = plate.astype(np.float32)
    light = values.max(axis=2)
    silver = (np.ptp(values, axis=2) <= .15 * light + 7.) & (light > 96.)
    return cv2.morphologyEx(silver.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def fit_profile(plate, screen, faces, part, measured, above, config):
    width = max(80, round(960. * plate.shape[1] / plate.shape[0]))
    height = 960
    raster = cv2.resize(plate, (width, height), interpolation=cv2.INTER_AREA)
    pixels = np.rint(screen * [width, height]).astype(np.int32)
    stencil = np.zeros((height, width), dtype=np.uint8)
    for face in faces[part]:
        cv2.fillConvexPoly(stencil, pixels[face], 1)
    silver = chrome_mask(raster) & cv2.dilate(stencil, np.ones((7, 7), np.uint8))
    distances = cv2.distanceTransform(1 - silver, cv2.DIST_L2, 5)
    interior = cv2.distanceTransform(silver, cv2.DIST_L2, 5)
    ys, xs = np.where(silver)
    if len(xs) < 100:
        return None
    left, right = np.percentile(xs, [4, 96])
    top, bottom = np.percentile(ys, [1, 97])
    span, rise = right - left, bottom - top
    if min(span, rise) < 35 or max(span, rise) > 470:
        return None

    def objective(values):
        line = profile(values)
        short, long = values[-2:]
        if line[:, 0].min() < 2. or line[:, 0].max() > right or line[:, 1].max() > bottom or short > long * .85:
            return 30.
        missing = sample(distances, line)
        groups = np.split(np.minimum(missing, 13.), [38, 93])
        error = [group.mean() for group in groups]
        return max(error) * .55 + np.mean(error) - .20 * np.median(sample(interior, line))

    bounds = [(max(5., left), left + .66*span), (top + .13*rise, top + .44*rise), (.11*span, .33*span),
              (.12*rise, .42*rise), (-.23, .23), (.24*rise, .57*rise), (.39*rise, .70*rise)]
    result = differential_evolution(objective, bounds, maxiter=95, popsize=11, polish=False, seed=873)
    line = profile(result.x)
    if result.fun >= 30. or np.mean(sample(distances, line) < 1.) < .90:
        return None
    centres = screen[faces].mean(axis=1) * [width, height]
    distance, closest = cKDTree(line).query(centres)
    x, y = np.clip(np.rint(centres), 0, [width-1, height-1]).astype(int).T
    evidence = part & (distance < 5.) & (silver[y, x] > 0)
    withheld = np.arange(len(line)) // 4 % 4 == 1
    withheld_faces = withheld[closest]
    anchors = []
    for index in range(len(line)):
        near = evidence & (abs(closest - index) < 4) & (withheld_faces == withheld[index])
        anchors.append(np.median(measured[near], axis=0) if near.any() else [np.nan]*3)
    anchors = np.asarray(anchors)
    known = np.isfinite(anchors).all(axis=1)
    if known.sum() < len(line) * .75:
        return None
    # Solve the orientation from spatially distributed points and check other points from the relief.
    checking = known & withheld
    if checking.sum() < 18:
        return None
    training = known & ~checking
    midpoint = np.median(anchors[training, :2], axis=0)
    _, _, axes = np.linalg.svd(anchors[training, :2] - midpoint, full_matrices=False)
    plane = axes[1]
    fx, fy, cx, cy = config['intrinsics']
    u, v = (line / [width, height]).T
    camera = np.asarray(config['world'])
    rays = np.column_stack(((u - cx) / fx, (cy - v) / fy, -np.ones(len(u)))) @ camera[:3, :3].T
    incidence = rays[:, :2] @ plane
    if np.min(abs(incidence)) < .12:
        return None
    travel = ((midpoint - camera[:2, 3]) @ plane) / incidence
    if np.min(travel) < .20 or np.max(travel) > 25:
        return None
    path = camera[:3, 3] + rays * travel[:, None]
    error = np.linalg.norm(path[checking] - anchors[checking], axis=1)
    if np.median(error) > .045 or np.percentile(error, 90) > .095:
        return None
    radius_pixels = np.clip(np.percentile(sample(interior, line), 55), 3.5, 7.5)
    radius = float(np.clip(radius_pixels * np.median(travel) / (fx * width), .006, .028))
    if path[-1, 2] - path[0, 2] < .025 or np.ptp(path[:, 2]) < 8 * radius:
        return None
    extent = np.min(line, axis=0) - 6., np.max(line, axis=0) + 6.
    within = ((centres >= extent[0]) & (centres <= extent[1])).all(axis=1)
    wall_clearance = (measured - config['wall_point']) @ np.asarray(config['wall_toward_camera'])
    original_curve = within & (wall_clearance > .13) & (above > .027)
    return original_curve, {
        'path': path[::3].tolist(),
        'radius': radius,
        'plate_centreline_support': float(np.mean(sample(distances, line) < 1.)),
        'held_out_depth_probes': int(checking.sum()),
        'median_depth_residual': float(np.median(error)),
    }
