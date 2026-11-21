"""Locate a visible rectangular mirror and solve the starting camera locally."""

import cv2
import numpy as np
from scipy.optimize import least_squares


def line(points, scale):
    if len(points) < 16:
        raise ValueError('A mirror edge has too little visible support.')
    x, y = np.asarray(points, dtype=np.float64).T
    design = np.column_stack((x, np.ones(len(x))))
    # A deterministic consensus fit tolerates corners, chrome highlights and foreground spill.
    seeds = np.linspace(0, len(x) - 1, min(22, len(x)), dtype=int)
    candidates = []
    for i in seeds:
        for j in seeds:
            if x[j] - x[i] < max(12, np.ptp(x) * .25):
                continue
            slope = (y[j] - y[i]) / (x[j] - x[i])
            if abs(slope) > .20:
                continue
            offset = y[i] - slope * x[i]
            error = abs(y - slope * x - offset)
            ok = error < scale
            candidates.append((int(ok.sum()), -float(np.median(error)), ok))
    if not candidates:
        raise ValueError('Could not find a straight mirror edge.')
    count, _, ok = max(candidates, key=lambda item: item[:2])
    if count < max(13, len(x) * .38):
        raise ValueError('A detected mirror boundary is not straight enough to calibrate.')
    params = np.linalg.lstsq(design[ok], y[ok], rcond=None)[0]
    # Alternating boundary samples provide a local check independent of the line fit.
    support = np.flatnonzero(ok)
    check = support[::4]
    train = np.setdiff1d(support, check)
    if len(train) >= 6:
        params = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
    return params, count / len(x), float(np.median(abs(design[check] @ params - y[check])))


def rim_samples(image, quad, foreground):
    """Follow the strongest consistent physical edge near each segmented side."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), .9)
    height, width = gray.shape
    derivative_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    derivative_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    radius = max(5., height * .012)
    offsets = np.arange(-radius, radius + .25, .5, dtype=np.float32)
    groups = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        axis = b - a
        distance = np.linalg.norm(axis)
        tangent = axis / distance
        normal = np.array([-tangent[1], tangent[0]])
        longitudinal = np.arange(.075 * distance, .925 * distance, 2.)
        centres = a + longitudinal[:, None] * tangent
        strip = centres[:, None, :] + offsets[:, None] * normal
        x, y = (strip[..., index].astype(np.float32) for index in range(2))
        gx = cv2.remap(derivative_x, x, y, cv2.INTER_LINEAR)
        gy = cv2.remap(derivative_y, x, y, cv2.INTER_LINEAR)
        normal_strength = gx * normal[0] + gy * normal[1]
        tangent_strength = gx * tangent[0] + gy * tangent[1]
        strength = np.maximum(0., abs(normal_strength) - abs(tangent_strength) * .60)
        blocked = cv2.remap(foreground, x, y, cv2.INTER_NEAREST) > 0
        strength[blocked | (x < 2) | (y < 2) | (x > width - 3) | (y > height - 3)] = 0.
        # Pick a coherent ridge across the side instead of chasing isolated strong pixels.
        peaks = (strength[:, 1:-1] >= strength[:, :-2]) & (strength[:, 1:-1] >= strength[:, 2:])
        peak_strength = np.where(peaks, strength[:, 1:-1], 0.)
        ridge = np.argmax(np.minimum(peak_strength, 14.).sum(axis=0))
        allowed = abs(offsets[1:-1] - offsets[ridge + 1]) <= 2.25
        choices = np.where(allowed, peak_strength, 0.)
        columns = choices.argmax(axis=1) + 1
        rows = np.arange(len(centres))
        top_strength = strength[rows, columns]
        valid = (top_strength >= 3.2) & allowed[columns - 1]
        if valid.sum() < 12 or np.ptp(longitudinal[valid]) < distance * .095:
            raise ValueError('The mirror rim has too little unobstructed contrast.')
        before = strength[rows, columns - 1]
        after = strength[rows, columns + 1]
        fraction = .5 * (after - before) / np.maximum(2 * top_strength - before - after, 1.e-4)
        points = centres + (offsets[columns] + .5 * np.clip(fraction, -.5, .5))[:, None] * normal
        groups.append(points[valid])
    return groups


def fit_rim(groups, quad, intrinsics, image_height):
    """Jointly compare a pinhole fit and a radial fit on withheld edge samples."""
    size = image_height / 2
    centre = np.asarray(intrinsics[2:])
    groups = [(points - centre) / size for points in groups]
    tangents = np.roll(quad, -1, axis=0) - quad
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normal = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)
    train = [np.arange(len(points)) % 4 != 1 for points in groups]

    def transform(points, radial):
        return points / (1 + radial * (points * points).sum(axis=1, keepdims=True))

    def terms(parameters, indexes):
        angle, bias = parameters[:4], parameters[4:8]
        radial = parameters[8] if len(parameters) > 8 else 0.
        directions = normal * np.cos(angle[:, None]) + tangents * np.sin(angle[:, None])
        error = [(transform(points[selected], radial) @ direction - offset) * size
                 for points, selected, direction, offset in zip(groups, indexes, directions, bias)]
        return error, directions

    start = np.r_[np.zeros(4), ((quad - centre) / size * normal).sum(axis=1)]
    straight = least_squares(lambda p: np.concatenate(terms(p, train)[0]), start, loss='soft_l1', f_scale=.8)
    curved = least_squares(lambda p: np.concatenate(terms(p, train)[0]), np.r_[straight.x, 0.],
                           bounds=([-np.inf] * 8 + [-.095], [np.inf] * 8 + [.095]),
                           loss='soft_l1', f_scale=.8)
    check = [~indices for indices in train]
    base, directions = terms(straight.x, check)
    alternative, _ = terms(curved.x, check)
    intersections = np.array([np.linalg.solve(directions[[i - 1, i]], straight.x[4:8][[i - 1, i]])
                              for i in range(4)])
    heldout = [float(np.median(abs(item))) for item in base]
    distorted = [float(np.median(abs(item))) for item in alternative]
    report = {'rim_edge_errors_px': heldout, 'radial_edge_errors_px': distorted,
              'radial_division_candidate': float(curved.x[-1]), 'rim_samples': [len(item) for item in groups]}
    return intersections * size + centre, report


def rectangle_focal(quad, centre):
    """Recover square-pixel focal length from the two orthogonal vanishing directions."""
    points = np.column_stack((quad - centre, np.ones(4)))
    lines = np.cross(points, np.roll(points, -1, axis=0))
    horizontal = np.cross(lines[0], lines[2])
    vertical = np.cross(lines[1], lines[3])
    product = horizontal[2] * vertical[2]
    if abs(product) < 1.e-8:
        return None
    squared = -(horizontal[:2] @ vertical[:2]) / product
    return float(np.sqrt(squared)) if squared > 0 and np.isfinite(squared) else None


def projected_rim_error(quad, groups, *, holdout):
    errors = []
    for start, end, samples in zip(quad, np.roll(quad, -1, axis=0), groups):
        tangent = end - start
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        selected = np.arange(len(samples)) % 4 == 1
        errors.append(float(np.median(abs((samples[selected == holdout] - start) @ normal))))
    return errors


def calibrate_rim(image, quad, foreground, world, intrinsics, *, refine_focal=True):
    groups = rim_samples(image, quad, foreground)
    precise, checks = fit_rim(groups, quad, intrinsics, image.shape[0])
    if not cv2.isContourConvex(precise.astype(np.float32)) or np.max(np.linalg.norm(precise - quad, axis=1)) > image.shape[0] * .038:
        raise ValueError('The measured rim is inconsistent with the detected mirror.')
    default = camera_from_corners(world, quad, intrinsics)
    base_errors = projected_rim_error(np.array(default['projected_corners']), groups, holdout=True)
    centre = np.asarray(intrinsics[2:])
    focal = rectangle_focal(precise, centre)
    candidates = [(precise, list(intrinsics))]
    if refine_focal and focal and .76 * intrinsics[0] < focal < intrinsics[0] * 1.30:
        alternatives = []
        for parity in (0, 2, 3):
            # Omit the same check samples as the primary fit, and also one third of its training samples.
            corners = []
            for points in groups:
                order = np.arange(len(points)) % 4
                samples = points[(order != 1) & (order != parity)]
                _, axis = np.linalg.eigh(np.cov(samples.T))
                perpendicular = axis[:, 0]
                corners.append((perpendicular, perpendicular @ samples.mean(axis=0)))
            reduced = np.array([np.linalg.solve(np.array([corners[i - 1][0], corners[i][0]]),
                                               np.array([corners[i - 1][1], corners[i][1]]))
                                for i in range(4)])
            alternatives.append(rectangle_focal(reduced, centre))
        if all(item and abs(item / focal - 1.) <= .13 for item in alternatives):
            candidates.append((precise, [focal, focal * intrinsics[1] / intrinsics[0], *centre]))

    best = None
    for corners, lens in candidates:
        try:
            pose = camera_from_corners(world, corners, lens)
        except ValueError:
            continue
        error = np.mean(projected_rim_error(np.array(pose['projected_corners']), groups, holdout=False))
        if best is None or error < best[2] * .93:
            best = pose, lens, error
    if best is None:
        raise ValueError('The measured rim did not yield a camera pose.')
    pose, lens, _ = best
    errors = projected_rim_error(np.array(pose['projected_corners']), groups, holdout=True)
    if np.mean(errors) >= np.mean(base_errors) * .85:
        raise ValueError('The physical rim did not improve the held-out camera projection.')
    pose.update(checks, rim_heldout_before_px=base_errors, rim_heldout_after_px=errors,
                focal_scale=float(lens[0] / intrinsics[0]))
    return precise, pose


def find_corners(image, direct_bounds, reflected_bounds, reflected_mask, weights):
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    height, width = image.shape[:2]
    l, t, r, b = reflected_bounds
    w, h = r - l, b - t
    mid = (l + r) / 2
    box = np.clip([l - .4 * w, t - .87 * h, r + .55 * w, b + .15 * h], 0, [width - 1, height - 1] * 2)
    points = np.clip([[mid, t - .39 * h], [l + .03 * w, t - .35 * h], [r + .03 * w, t - .52 * h],
                      [mid, b + .27 * h], [l - .38 * w, t + .12 * h]], 0, [width - 1, height - 1])
    name = weights.name.lower()
    variant = next((item for item in ('vit_b', 'vit_l', 'vit_h') if item in name), 'vit_b')
    model = sam_model_registry[variant]()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    predictor = SamPredictor(model.eval().to('cuda' if torch.cuda.is_available() else 'cpu'))
    predictor.set_image(image, image_format='BGR')
    masks, likelihoods, _ = predictor.predict(point_coords=points, point_labels=np.array([1, 1, 1, 0, 0]),
                                               box=box, multimask_output=True)
    candidates = []
    for mask, probability in zip(masks, likelihoods):
        mask = mask | reflected_mask
        ys, xs = np.where(mask)
        if len(xs) < width * height * .10:
            continue
        left, right = np.percentile(xs, [1, 99])
        top, bottom = np.percentile(ys, [1, 99])
        if bottom - top < h * 1.15 or right - left < w * 1.06:
            continue
        rim_tolerance = max(2.1, height * .004)
        try:
            top_points, bottom_points, left_points, right_points = [], [], [], []
            for x in range(round(left + (right-left)*.10), round(right - (right-left)*.09)):
                rows = np.flatnonzero(mask[:, x])
                if len(rows) < h * .1:
                    continue
                if 1 < rows[0] < t - h * .11:
                    top_points.append((x, rows[0]))
                if x < direct_bounds[0] - 4 and abs(rows[-1] - b) < h * .2:
                    bottom_points.append((x, rows[-1]))
            for y in range(round(top + 8), round(bottom - 8)):
                cols = np.flatnonzero(mask[y])
                if len(cols) < w * .16:
                    continue
                if y > top + (bottom-top)*.09 and 1 < cols[0] < l - w*.10:
                    left_points.append((y, cols[0]))
                if y < direct_bounds[1] - 6 and r + w * .10 < cols[-1] < width - 2:
                    right_points.append((y, cols[-1]))
            fits = [line(item, rim_tolerance) for item in (top_points, bottom_points, left_points, right_points)]
        except ValueError:
            continue
        top_fit, bottom_fit, left_fit, right_fit = [item[0] for item in fits]

        def corner(horizontal, vertical):
            a, b_ = horizontal
            c, d = vertical
            y = (a * d + b_) / (1 - a * c)
            return [c * y + d, y]

        quad = np.array([corner(top_fit, left_fit), corner(top_fit, right_fit), corner(bottom_fit, right_fit),
                         corner(bottom_fit, left_fit)])
        if not ((quad >= [-2, -2]) & (quad <= [width + 2, height + 2])).all():
            continue
        inside = cv2.isContourConvex(quad.astype(np.float32))
        if not inside or cv2.contourArea(quad.astype(np.float32)) < width * height * .12:
            continue
        score = sum(item[1] for item in fits) + float(probability) * .7 - sum(item[2] for item in fits) * .06
        candidates.append((score, quad, np.mean([item[2] for item in fits])))
    if not candidates:
        raise ValueError('Could not establish all four edges of the main mirror.')
    _score, quad, edge_error = max(candidates, key=lambda item: item[0])
    return quad, float(edge_error)


def camera_from_corners(world, pixels, intrinsics):
    fx, fy, cx, cy = intrinsics
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    base = np.array(world, dtype=np.float64)
    up = base[0] - base[3]

    def points(scale):
        result = base.copy()
        result[:2] += up * (scale - 1.)
        return result

    ok, r, t = cv2.solvePnP(points(1.), pixels, matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('The detected mirror did not yield a camera pose.')

    def residual(parameters):
        projected, _ = cv2.projectPoints(points(parameters[-1]), parameters[:3], parameters[3:6], matrix, None)
        difference = projected[:, 0] - pixels
        # The lower corner under the foreground actor is extrapolated from the visible edges.
        difference[2] *= .65
        return np.r_[difference.ravel(), (parameters[-1] - 1.) * 1.2]

    result = least_squares(residual, np.r_[r.ravel(), t.ravel(), 1.],
                           bounds=([-np.inf] * 6 + [.55], [np.inf] * 6 + [1.75]))
    rotation = cv2.Rodrigues(result.x[:3])[0]
    locations = (points(result.x[-1]) @ rotation.T + result.x[3:6])
    rms = float(np.sqrt(np.mean(residual(result.x)[:8] ** 2)))
    if np.any(locations[:, 2] <= .05) or rms > 3.:
        raise ValueError('The detected mirror edges disagree with the estimated lens.')
    pixels, _ = cv2.projectPoints(points(result.x[-1]), result.x[:3], result.x[3:6], matrix, None)
    return {'rotation': (rotation.T @ np.diag([1., -1., -1.])).tolist(),
            'position': (-rotation.T @ result.x[3:6]).tolist(), 'height_scale': float(result.x[-1]), 'corner_rms': rms,
            'projected_corners': pixels[:, 0].tolist()}
