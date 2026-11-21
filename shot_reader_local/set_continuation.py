"""Continue observed set texture into short, unobserved borders of a locked plate."""

import cv2
import numpy as np


def extend_rows(image, known, fixed, limit):
    height, width = known.shape
    columns = np.arange(width)
    anchor = np.maximum.accumulate(np.where(known | fixed, columns, -1), axis=1)
    rows = np.arange(height)[:, None]
    supported = (anchor >= 0) & known[rows, np.maximum(anchor, 0)]
    distance = columns - anchor
    eligible = ~known & ~fixed & supported & (distance <= limit)
    if not eligible.any():
        return np.zeros_like(image), np.zeros(known.shape, np.float32), []

    colour = image.astype(np.float32)
    accum = np.zeros_like(colour)
    weight = np.zeros((height, width), np.float32)
    diagnostics = []
    band = min(height, max(30, round(min(height, width) * .135)))
    stride = max(1, band // 2)
    shifts = np.arange(max(10, round(limit * .60)), min(width // 2, limit * 3), 2)
    for start in range(-stride, height, stride):
        top, bottom = max(0, start), min(height, start + band)
        if bottom <= top:
            continue
        y, x = np.where(eligible[top:bottom])
        if len(x) < 25:
            continue
        y += top
        boundaries = (anchor[y, x] == x - 1)
        ys, xs = y[boundaries], x[boundaries] - 1
        if len(xs) < 12:
            continue

        # The first few seen pixels are reserved for the spatial validation below.
        fit_x = xs[:, None] - np.arange(10, 31, 3)
        checked_x = xs[:, None] - np.arange(1, 9, 2)
        py = ys[:, None]
        fit_source = (fit_x >= 0) & known[py, fit_x.clip(0, width - 1)]
        choices = []
        for offset in shifts:
            donor = (fit_x - offset).clip(0, width - 1)
            fit = fit_source & (fit_x >= offset) & known[py, donor]
            if fit.sum() < max(45, fit_source.sum() * .76):
                continue
            error = abs(colour[py, fit_x.clip(0, width - 1)] - colour[py, donor]).mean(axis=2)
            filled = (x >= offset) & (distance[y, x] < offset - 1) & known[y, (x - offset).clip(0, width - 1)]
            if filled.mean() >= .52:
                choices.append((float(np.median(error[fit]) + .2 * np.mean(error[fit]) + 7 * (1. - filled.mean())), offset, filled))
        if not choices:
            continue
        _, shift, fill = min(choices, key=lambda item: item[0])
        donor = (checked_x - shift).clip(0, width - 1)
        check = (checked_x >= shift) & known[py, checked_x.clip(0, width - 1)] & known[py, donor]
        if check.sum() < 36:
            continue
        error = abs(colour[py, checked_x.clip(0, width - 1)] - colour[py, donor]).mean(axis=2)[check]
        if np.mean(error) > 36. or np.median(error) > 24.:
            continue

        y, x = y[fill], x[fill]
        taper = np.maximum(.025, np.sin(np.pi * ((y - start + .5) / band)) ** 2)
        donor = colour[y, x - shift]
        # Match low frequency colour at the seam while keeping the donor's detail.
        boundary = anchor[y, x]
        edge_donor = (boundary - shift).clip(0, width - 1)
        balance = (colour[y, boundary] - colour[y, edge_donor]).clip(-26, 26)
        balance *= (known[y, edge_donor] * np.exp(-distance[y, x] / 13.))[:, None]
        accum[y, x] += (donor + balance) * taper[:, None]
        weight[y, x] += taper
        diagnostics.append((float(np.mean(error)), int(check.sum())))

    out = np.clip(accum / np.maximum(weight[..., None], 1.e-6), 0., 255.).astype(np.uint8)
    confidence = weight * (1. - (distance - 1) / (limit + 1))
    return out, confidence, diagnostics


def continue_texture(plate, known, fixed):
    """Return a separate estimated patch and spatial validation measurements."""
    h, w = known.shape
    height, width = max(1, h // 2), max(1, w // 2)
    size = (width, height)
    colour = cv2.resize(plate, size, interpolation=cv2.INTER_AREA)
    measured = cv2.resize(known.astype(np.float32), size, interpolation=cv2.INTER_AREA) > .98
    protected = cv2.resize(fixed.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    extent = max(8, round(min(width, height) * .225))
    forward, score_a, check_a = extend_rows(colour, measured, protected, extent)
    backward, score_b, check_b = extend_rows(colour[:, ::-1], measured[:, ::-1], protected[:, ::-1], extent)
    score_b, backward = score_b[:, ::-1], backward[:, ::-1]
    use_a = score_a >= score_b
    scores = np.maximum(score_a, score_b)
    estimate = np.where(use_a[..., None], forward, backward)
    usable = scores > 0.
    if usable.sum() < 65:
        return None
    # Upsample premultiplied colour so interpolation at sparse boundaries does not introduce dark fringes.
    mix = cv2.resize(usable.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    premul = cv2.resize(estimate.astype(np.float32) * usable[..., None], (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(premul / np.maximum(mix[..., None], 1.e-6), 0, 255).astype(np.uint8)
    available = (mix > .22) & ~fixed & ~known
    image = np.dstack((rgb, available.astype(np.uint8) * 255))
    checks = check_a + check_b
    errors, counts = zip(*checks)
    return image, {
        'estimated_pixels': int(available.sum()),
        'held_out_boundary_samples': int(sum(counts)),
        'held_out_boundary_rgb_mae': float(np.average(errors, weights=counts)),
    }
