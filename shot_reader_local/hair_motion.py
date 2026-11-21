"""Estimate small pose-dependent shifts of hair against both plate views."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


NAMES = ('Ends Sideways', 'Ends Depth', 'Ends Height', 'Ends Spread', 'Crown Sideways', 'Crown Depth')
REGIONAL_NAMES = tuple(f'{side} Lock {direction}' for side in ('Left', 'Right')
                       for direction in ('Sideways', 'Depth', 'Height'))
REGIONAL_NAMES += ('Nape Spread', 'Nape Depth', 'Crown Height', 'Crown Spread')


def modes_for(vertices, head):
    x, _, z = (vertices - head).T
    tail = np.clip((.02 - z) / .23, 0., 1.)
    side = np.clip(x / .08, -1., 1.)
    modes = np.zeros((len(NAMES), len(vertices), 3), np.float32)
    modes[0, :, 0] = .055 * tail
    modes[1, :, 1] = .064 * tail
    modes[2, :, 2] = .061 * tail
    modes[3, :, 0] = .030 * tail * side
    modes[4, :, 0] = .023 * (1. - tail)
    modes[5, :, 1] = .025 * (1. - tail)
    return modes


def transform_deltas(modes, poses, weights):
    # The same rest-space shape keys are applied before the armature in Blender.
    return np.einsum('mvj,fkij,vk->fmvi', modes, poses[:, :, :3, :3], weights).astype(np.float32)


def regional_modes(vertices, head):
    x, y, z = (vertices - head).T
    lower = np.clip((-z + .025) / .13, 0., 1.) * np.clip((z + .56) / .22, 0., 1.)
    side = np.clip(.5 + x / .13, 0., 1.)
    neck = np.exp(-((z + .07) / .10) ** 2)
    cap = np.clip((z + .055) / .12, 0., 1.)
    lateral = np.clip(x / .08, -1., 1.)
    front = np.clip((y + .05) / .11, 0., 1.)
    scales = np.array([neck * lateral * .032, neck * front * .045, cap * .035, cap * lateral * .028])
    local = scales[..., None] * np.eye(3)[[0, 1, 2, 0], None]
    return np.concatenate([weight[None, :, None] * np.diag([.062, .074, .074])[:, None]
                           for weight in (lower * side, lower * (1. - side))] + [local]).astype(np.float32)


def fit_motion(vertices, faces, head, posed, poses, weights, observations, holdout, project, depth_buffer, times):
    if len(vertices) < 100 or len(posed) < 8 or len(observations[0][0]) != 2:
        return None
    width, height = observations[0][3][0].shape[::-1]
    modes = np.concatenate((modes_for(vertices, head), regional_modes(vertices, head)))
    delta = transform_deltas(modes, poses, weights)
    scores = []
    checking = np.flatnonzero(holdout)
    training = np.flatnonzero(~holdout)
    paths = np.zeros((len(posed), len(modes)))
    if len(checking) < 2:
        return None

    for stride in (2, 1):
        w, h = width // stride, height // stride
        crop = np.s_[:h * stride:stride, :w * stride:stride]
        scores.append([
            [(view, lens / stride, region[crop], mask[crop], body[crop], w, h)
             for view, region, mask, body in zip(views, regions, masks, body_z)]
            for views, lens, regions, masks, body_z in observations
        ])

    def evaluate(frame, value, full=False):
        world = posed[frame] + np.einsum('m,mvc->vc', value, delta[frame, :len(value)])
        overlap = []
        for view, lens, region, mask, body, w, h in scores[int(full)][frame]:
            image, z = project(world, view, lens)
            depth = depth_buffer(image, z, faces, w, h)
            rendered = np.isfinite(depth) & (depth <= body + .013) & region
            target = (mask == 2) & region
            hair = np.count_nonzero(rendered & target) / max(1, np.count_nonzero(rendered | target))
            if full:
                performer = (mask > 0) & region
                silhouette = rendered | (np.isfinite(body) & region)
                overlap.append((hair, np.count_nonzero(silhouette & performer) /
                                max(1, np.count_nonzero(silhouette | performer))))
            else:
                overlap.append(hair)
        return np.array(overlap)

    count = len(NAMES)
    previous = np.zeros(count)
    for i in training:
        def objective(value):
            direct, reflected = evaluate(i, value)
            return -(.40 * direct + .60 * reflected) + .004 * np.dot(value, value)

        result = minimize(objective, previous, method='Powell', bounds=[(-1., 1.)] * count,
                          options={'maxfev': 100, 'ftol': .001, 'xtol': .07})
        paths[i, :count] = result.x
        previous = result.x

    def interpolate(path):
        return np.column_stack([gaussian_filter1d(np.interp(times, times[training], track[training]), .65, mode='nearest')
                                for track in path.T])

    baseline = interpolate(paths[:, :count])
    previous = np.zeros(len(REGIONAL_NAMES))
    for i in training:
        def objective(value):
            direct, reflected = evaluate(i, np.r_[paths[i, :count], value], full=True)[:, 0]
            return -(.26 * direct + .74 * reflected) + .0018 * np.dot(value, value)

        result = minimize(objective, previous, method='Powell', bounds=[(-1., 1.)] * len(REGIONAL_NAMES),
                          options={'maxfev': 250, 'ftol': .0005, 'xtol': .065})
        paths[i, count:] = result.x
        previous = result.x
    paths = interpolate(paths)
    before = np.array([evaluate(i, np.zeros(count), full=True) for i in checking])
    base_checked = np.array([evaluate(i, baseline[i], full=True) for i in checking])
    after = np.array([evaluate(i, paths[i], full=True) for i in checking])
    regional_gain = (after - base_checked).mean(axis=0)
    if regional_gain[1, 0] < .008 or regional_gain[0, 0] < -.006 or regional_gain[:, 1].mean() < -.003:
        paths, modes, delta, after = baseline, modes[:count], delta[:, :count], base_checked
        labels = NAMES
    else:
        labels = NAMES + REGIONAL_NAMES
    gains = (after - before).mean(axis=0)
    if gains[1, 0] < .012 or gains[0, 0] < -.008 or (after[:, 1, 0] - before[:, 1, 0]).min() < -.03:
        return None
    if gains[:, 1].mean() < -.003:
        return None
    report = {'held_out_frames': len(checking), 'direct_before': float(before[:, 0, 0].mean()),
              'direct_after': float(after[:, 0, 0].mean()), 'reflected_before': float(before[:, 1, 0].mean()),
              'reflected_after': float(after[:, 1, 0].mean()), 'controls': list(labels)}
    if len(labels) > count:
        report['regional_reflected_gain'] = float(regional_gain[1, 0])
    return modes, paths, delta, report
