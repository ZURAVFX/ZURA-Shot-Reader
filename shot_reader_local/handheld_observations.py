"""Track consistently bright, elongated handheld objects near the fitted hands."""

import json
from pathlib import Path
import sys
import warnings

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from room_clean import raster_exclusions, read_frames


STATIONS = np.linspace(-.50, .50, 15)


def cross_sections(points):
    bounds = []
    for station in STATIONS:
        sample = points[abs(points[:, 0] - station) < .065, 1]
        bounds.append(np.percentile(sample, [4, 96]) if len(sample) >= 3 else [np.nan, np.nan])
    return bounds


def score_profile(profile, observations, predictions, reference_width):
    profile = np.asarray(profile)
    contour = np.concatenate((profile[:, :2], profile[::-1, [0, 2]]))
    scores = []
    for points, (x, y, angle, length, _width) in zip(observations, predictions):
        axis = np.array([np.cos(angle), np.sin(angle)])
        side = np.array([-axis[1], axis[0]])
        shape = np.array([x, y]) + contour[:, :1] * length * axis + contour[:, 1:] * reference_width * side
        origin = np.floor(np.minimum(shape.min(axis=0), points.min(axis=0))).astype(int)
        maximum = np.ceil(np.maximum(shape.max(axis=0), points.max(axis=0))).astype(int)
        raster = np.zeros(tuple((maximum - origin + 1)[::-1]), dtype=np.uint8)
        cv2.fillPoly(raster, [np.rint(shape - origin).astype(np.int32)], 1)
        truth = np.zeros_like(raster)
        px, py = (points - origin).T
        truth[py, px] = 1
        scores.append((raster & truth).sum() / (raster | truth).sum())
    return float(np.mean(scores))


def fit_appearance(request, report, observations, directory):
    if directory is None or 'profile' not in report:
        return
    width, height = request['size']
    native = request.get('native_size', request['size'])
    limit = min(1., 1440 / max(native))
    raster = tuple(round(axis * limit) for axis in native)
    factor = np.array(raster) / [width, height]
    training = [item for item in observations if item[0] % 4 != 2]
    checking = [item for item in observations if item[0] % 4 == 2]
    if len(training) > 24:
        training = [training[i] for i in np.linspace(0, len(training) - 1, 24, dtype=int)]
    if len(checking) > 18:
        checking = [checking[i] for i in np.linspace(0, len(checking) - 1, 18, dtype=int)]
    samples = training + checking
    frames = read_frames(request['clip'], [request['times'][item[0]] for item in samples], raster)
    profile = np.asarray(report['profile'])
    motion = np.asarray(report['motion'])
    length0, width0 = np.median(motion[np.asarray([item[0] for item in training]), 3:], axis=0)
    station = np.linspace(-.5, .5, 96)
    cross = np.linspace(0., 1., 32)[:, None]
    low, high = (np.interp(station, profile[:, 0], profile[:, column]) for column in (1, 2))
    sides = low + cross * (high - low)
    rgb, masks = [], []
    for sample, frame in zip(samples, frames):
        index = sample[0]
        x, y, angle, length, _ = motion[index]
        axis = np.array([np.cos(angle), np.sin(angle)])
        transverse = np.array([-axis[1], axis[0]])
        xy = (np.array([x, y]) + station[None, :, None] * np.clip(length, length0 * .82, length0 * 1.18) * axis
              + sides[..., None] * width0 * transverse)
        source = np.zeros((height, width), dtype=np.uint8)
        observed_x, observed_y = sample[4].T
        source[observed_y, observed_x] = 1
        source = cv2.morphologyEx(source, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        sample_x, sample_y = np.moveaxis(xy.astype(np.float32), -1, 0)
        covered = cv2.remap(source, sample_x, sample_y, cv2.INTER_NEAREST) > 0
        # An inset leaves uncertain skin and background boundaries to the uniform material.
        covered &= (cross > .055) & (cross < .945)
        px, py = (sample_x * np.float32(factor[0]), sample_y * np.float32(factor[1]))
        colour = cv2.remap(frame, px, py, cv2.INTER_LINEAR)[..., ::-1].astype(np.float32)
        covered &= (colour.max(axis=2) - colour.min(axis=2)) < 55.
        rgb.append(colour)
        masks.append(covered)
    rgb, masks = np.asarray(rgb), np.asarray(masks)
    count = len(training)
    qualified = masks[:count].sum(axis=0) >= max(3, count // 5)
    if qualified.mean() < .35:
        return
    uniform = np.asarray(report['colour']) * 255.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        estimate = np.nanmedian(np.where(masks[:count, ..., None], rgb[:count], np.nan), axis=0)
    estimate = np.where(qualified[..., None], estimate, uniform)
    alpha = cv2.GaussianBlur(qualified.astype(np.float32), (5, 5), .8)
    palette = estimate * alpha[..., None] + uniform * (1. - alpha[..., None])
    held = masks[count:]
    if held.sum() < 300:
        return
    before = float(np.abs(rgb[count:] - uniform)[held].mean())
    after = float(np.abs(rgb[count:] - palette)[held].mean())
    if after >= before - .45:
        return
    path = directory / 'handheld-colour.png'
    if cv2.imwrite(str(path), np.rint(palette[..., ::-1]).clip(0., 255.).astype(np.uint8)):
        report['appearance'] = {'training_frames': len(training), 'held_out_frames': len(checking),
                                'sampled_render_texels': int(held.sum()), 'uniform_rgb_mae': before,
                                'texture_rgb_mae': after, 'verified_visible_fraction': float(qualified.mean())}


def candidates(image, eligible, hands, height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = ((hsv[..., 1] <= 45) & (hsv[..., 2] >= 185) & eligible).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    result = {name: [] for name in hands}
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not 75 * (height / 768) ** 2 <= area <= 1200 * (height / 768) ** 2:
            continue
        iy, ix = np.where(labels[y:y+h, x:x+w] == index)
        points = np.column_stack((ix + x, iy + y))
        centre = points.mean(axis=0)
        _, axes = np.linalg.eigh(np.cov(points.T))
        long = axes[:, 1]
        if long[0] < 0.:
            long *= -1.
        coords = (points - centre) @ np.column_stack((long, [-long[1], long[0]]))
        limits = np.percentile(coords, [2, 98], axis=0)
        length, width = limits[1] - limits[0]
        if not (.023 * height <= length <= .093 * height and 1.65 < length / max(1., width) < 5.5):
            continue
        centre += limits.mean(axis=0)[0] * long
        coords[:, 0] -= limits.mean(axis=0)[0]
        sections = cross_sections(coords / [length, width])
        colour = np.median(image[points[:, 1], points[:, 0]], axis=0)[::-1] / 255.
        # A small object must remain near an anatomically plausible hand, even on frames withheld from the fit.
        for name, point in hands.items():
            distance = np.linalg.norm(centre - point)
            if distance < height * .103:
                score = area ** .45 / (1. + distance / (height * .055))
                result[name].append((score, np.r_[centre, np.arctan2(long[1], long[0]), length, width], colour, sections, points))
    return result


def analyse(request, directory=None):
    width, height = request['size']
    hands = request['hands']
    times = np.asarray(request['times'])
    images = read_frames(request['clip'], times, (width, height))
    paths = {}
    for index, (image, views) in enumerate(zip(images, hands)):
        whole = np.ones((height, width), dtype=bool)
        apertures = [whole] if len(views) == 1 else [
            ~raster_exclusions(request['size'], [request['mirror'][index]]),
            raster_exclusions(request['size'], [request['mirror'][index]])]
        for view, hand_views in enumerate(views):
            anchors = {name: np.asarray(data['xy']) for name, data in hand_views.items() if data['depth'] > .05}
            if not anchors:
                continue
            zone = np.zeros((height, width), np.uint8)
            for point in anchors.values():
                cv2.circle(zone, tuple(np.rint(point).astype(int)), round(height * .105), 1, -1)
            for name, choices in candidates(image, zone.astype(bool) & apertures[view], anchors, height).items():
                if not choices:
                    continue
                winner = max(choices, key=lambda item: item[0])
                paths.setdefault((view, name), []).append((index, *winner[1:]))

    accepted = []
    for (view, side), observed in paths.items():
        index = np.asarray([item[0] for item in observed], dtype=int)
        values = np.array([item[1] for item in observed])
        train = index % 4 != 2
        check = ~train
        if len(observed) < len(times) * .57 or train.sum() < 5 or check.sum() < 4:
            continue
        # Fit only the training observations. The held-out detections check the entire projected track.
        fitted = values[train].copy()
        fitted[:, 2] = np.unwrap(fitted[:, 2] * 2) / 2
        sizes = np.median(fitted[:, 3:], axis=0)
        variation = np.median(abs(fitted[:, 3:] - sizes), axis=0) / sizes
        if variation.max() > .29:
            continue
        fit = PchipInterpolator(times[index[train]], fitted, extrapolate=False)
        prediction = gaussian_filter1d(fit(np.clip(times, times[index[train][0]], times[index[train][-1]])),
                                       .9, axis=0, mode='nearest')
        errors = np.linalg.norm(prediction[index[check], :2] - values[check, :2], axis=1)
        angle = (np.rad2deg(prediction[index[check], 2] - values[check, 2]) + 90.) % 180. - 90.
        if np.median(errors) > 2.6 * height / 768 or np.percentile(errors, 90) > 7 * height / 768:
            continue
        if np.median(abs(angle)) > 8.:
            continue
        clip = [item[4] for item, held in zip(observed, check) if held]
        profiles = np.array([item[3] for item, used in zip(observed, train) if used])
        # The profile is signed in the same axis direction as the unwrapped training track.
        flipped = np.cos(fitted[:, 2] - values[train, 2]) < 0.
        profiles[flipped] = -profiles[flipped, ::-1, ::-1]
        best = np.zeros((len(STATIONS), 2))
        usable = np.isfinite(profiles[..., 0]).sum(axis=0) >= max(3, int(train.sum() * .25))
        if usable.sum() >= 8 and 'default_profile' in request:
            best[usable] = np.nanmedian(profiles[:, usable], axis=0)
            for channel in range(2):
                best[:, channel] = np.interp(STATIONS, STATIONS[usable], best[usable, channel])
                best[:, channel] = gaussian_filter1d(best[:, channel], .55)
            best = np.clip(best, -.85, .85)
            fitted_profile = np.column_stack((STATIONS, best))
            checked = prediction[index[check]].copy()
            checked[:, 3] = np.clip(checked[:, 3], sizes[0] * .82, sizes[0] * 1.18)
            prior = score_profile(request['default_profile'], clip, checked, sizes[1])
            improved = score_profile(fitted_profile, clip, checked, sizes[1])
            profile_report = {'template_overlap': prior, 'fitted_overlap': improved}
            if improved >= prior + .035:
                profile_report['profile'] = fitted_profile.tolist()
        else:
            profile_report = {}
        accepted.append({'view': view, 'hand': side, 'motion': prediction.tolist(),
                         'colour': np.median(np.array([item[2] for item in observed])[train], axis=0).tolist(),
                         'observed_frames': len(observed), 'held_out_frames': int(check.sum()),
                         'median_projected_error_px': float(np.median(errors)),
                         'median_axis_error_degrees': float(np.median(abs(angle))), **profile_report})
    best = max(accepted, key=lambda item: (item['observed_frames'], -item['median_projected_error_px']), default=None)
    if best:
        fit_appearance(request, best, paths[(best['view'], best['hand'])], directory)
    return best


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    (directory / 'handheld.json').write_text(json.dumps(analyse(request, directory), allow_nan=False), encoding='utf-8')
