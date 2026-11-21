"""Fit a small, reversible body shape correction to multi-view plate silhouettes."""

import numpy as np


def limit_upper_arm_deformation(modes, vertices, arms):
    """Bound cross-section changes for any combination of the fitted modes."""
    for joint, elbow in arms:
        length = np.linalg.norm(elbow - joint)
        if length < 1.e-6:
            continue
        direction = (elbow - joint) / length
        along = (vertices - joint) @ direction
        radius = np.linalg.norm(vertices - joint - along[:, None] * direction, axis=1)
        fraction = along / length
        end_blend = np.clip(np.minimum(fraction, 1. - fraction) / .22, 0., 1.)
        outer_blend = np.clip((.105 - radius) / .025, 0., 1.)
        blend = end_blend * outer_blend
        radial = modes - (modes @ direction)[..., None] * direction
        maximum = np.linalg.norm(radial, axis=2).sum(axis=0)
        limit = .24 * radius
        scale = 1. - blend * (1. - np.minimum(1., limit / np.maximum(maximum, 1.e-8)))
        modes *= scale[None, :, None]
    return modes


def body_modes(vertices, torso, shoulder, arms):
    x, y, z = vertices.T
    span = max(.15, shoulder[2] - torso[2])
    across = np.clip((.31 - abs(x)) / .11, 0., 1.)
    taper = across * np.clip((shoulder[2] + .11 - z) / .12, 0., 1.)
    upper = taper * np.exp(-((z - shoulder[2] + .055) / (span * .63)) ** 2)
    lower = taper * np.exp(-((z - torso[2] + .055) / (span * .63)) ** 2)
    width = np.clip(x / .15, -1., 1.)
    depth = np.clip((y - torso[1]) / .09, -1., 1.)
    modes = np.zeros((12, len(vertices), 3), np.float32)
    modes[0, :, 0] = .040 * width * upper
    modes[1, :, 0] = .047 * width * lower
    modes[2, :, 1] = .045 * depth * upper
    modes[3, :, 1] = .047 * depth * lower
    modes[4, :, 1] = .032 * upper
    modes[5, :, 2] = .030 * upper
    for index, (joint, elbow) in enumerate(arms):
        axis = elbow - joint
        place = np.clip((vertices - joint) @ axis / (axis @ axis), 0., 1.)
        radius = vertices - (joint + place[:, None] * axis)
        distance = np.linalg.norm(radius, axis=1)
        support = np.clip(1. - distance / .11, 0., 1.) * np.maximum(0., np.sin(np.pi * place)) ** .7
        modes[6 + index, :, :] = .065 * radius / np.maximum(distance[:, None], .032) * support[:, None]
        cap = np.exp(-((vertices - joint) ** 2).sum(axis=1) / .065 ** 2)
        modes[8 + index, :, 0] = .072 * np.sign(joint[0]) * cap
        modes[10 + index, :, 2] = .060 * cap
    return limit_upper_arm_deformation(modes, vertices, arms)


def fit_body(request, data, masks, allowed, holdout, project, depth_buffer):
    from scipy.ndimage import binary_dilation
    from scipy.optimize import minimize

    deltas = data['body_modes']
    if not np.isfinite(deltas).all():
        return None
    body = data['body']
    faces = data['faces']
    views, lenses = data['views'], data['lenses'] * .5
    width, height = request['size']
    w, h = width // 2, height // 2
    valid = []
    for frame in range(len(body)):
        for view, (mask, region) in enumerate(zip(masks[frame], allowed[frame])):
            observed = region & ~binary_dilation(mask == 2, iterations=4)
            truth = ((mask > 0) & observed)[:2*h:2, :2*w:2]
            if truth.sum() >= .02 * (w * h):
                valid.append((frame, view, observed[:2*h:2, :2*w:2], truth))
    fitting = [item for item in valid if not holdout[item[0]]]
    checking = [item for item in valid if holdout[item[0]]]
    if len(fitting) < 6 or len(checking) < 2:
        return None
    trials = []
    for view in range(views.shape[1]):
        samples = [item for item in fitting if item[1] == view]
        trials.extend(samples[i] for i in np.linspace(0, len(samples) - 1, min(5, len(samples)), dtype=int))

    def evaluate(params, selection):
        scores = []
        posed = body + np.einsum('d,dfvc->fvc', params.astype(np.float32), deltas)
        for frame, view, region, truth in selection:
            xy, depth = project(posed[frame], views[frame, view], lenses[frame])
            rendered = np.isfinite(depth_buffer(xy, depth, faces, w, h)) & region
            scores.append(np.count_nonzero(rendered & truth) / max(1, np.count_nonzero(rendered | truth)))
        return np.asarray(scores)

    neutral = np.zeros(len(deltas))
    base_fit = evaluate(neutral, fitting)
    base_check = evaluate(neutral, checking)

    def objective(params):
        return -evaluate(params, trials).mean() + .00060 * (params @ params)

    result = minimize(objective, neutral, method='Powell', bounds=[(-1., 1.)] * len(deltas),
                      options={'maxfev': 290, 'xtol': .08, 'ftol': .00075})
    fitted = evaluate(result.x, fitting)
    checked = evaluate(result.x, checking)
    if fitted.mean() < base_fit.mean() + .003 or checked.mean() < base_check.mean() + .004:
        return None
    if min(checked - base_check) < -.035:
        return None
    corrected = body + np.einsum('d,dfvc->fvc', result.x.astype(np.float32), deltas)
    delta = np.einsum('d,dvc->vc', result.x.astype(np.float32), data['rest_modes'])
    report = {'torso_before_iou': float(base_check.mean()), 'torso_after_iou': float(checked.mean()),
              'held_out_views': len(checking), 'max_vertex_displacement_units': float(np.linalg.norm(delta, axis=1).max())}
    return corrected, delta, report
