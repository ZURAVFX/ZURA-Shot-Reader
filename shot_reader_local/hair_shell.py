"""Recover thin hair lying against the fitted neck and shoulders."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def fit_shell(rest, points, faces, head, frames, holdout, project, depth_buffer):
    relative = rest[faces].mean(axis=1) - head
    near = (relative[:, 2] > -.44) & (relative[:, 2] < -.058) & (abs(relative[:, 0]) < .30)
    candidates = faces[near]
    if not len(candidates):
        return None
    positive = np.zeros(len(candidates), dtype=int)
    negative = positive.copy()
    missing = positive.copy()
    for index, (views, lens, regions, masks, body_z, hair_z) in enumerate(frames):
        if holdout[index]:
            continue
        samples = points[index, candidates].mean(axis=1)
        height, width = masks.shape[-2:]
        for view, region, mask, surface, volume in zip(views, regions, masks, body_z, hair_z):
            xy, depth = project(samples, view, lens)
            x, y = np.rint(xy).clip([0, 0], [width - 1, height - 1]).astype(int).T
            inside = ((xy >= 0) & (xy < [width - 1, height - 1])).all(axis=1)
            visible = inside & region[y, x] & (depth > .02) & (depth <= surface[y, x] + .006)
            hair = mask[y, x] == 2
            positive += visible & hair
            negative += visible & ~hair
            missing += visible & hair & (volume[y, x] > depth + .003)
    select = (positive >= 4) & (positive >= 1.65 * negative) & (missing >= 3)
    subset = candidates[select]
    if len(subset) < 12:
        return None

    # Retain connected locks of hair and drop isolated triangles caused by mask noise.
    n = len(rest)
    pairs = np.concatenate((subset[:, :2], subset[:, 1:], subset[:, ::2]))
    edges = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(n, n)).tocsr()
    _, labels = connected_components(edges, directed=False)
    labels = labels[subset[:, 0]]
    subset = subset[np.bincount(labels)[labels] >= 11]
    if len(subset) < 12:
        return None

    before, after, patch_pixels = [], [], []
    for index in np.flatnonzero(holdout):
        views, lens, regions, masks, body_z, hair_z = frames[index]
        height, width = masks.shape[-2:]
        for view, region, mask, surface, volume in zip(views, regions, masks, body_z, hair_z):
            xy, depth = project(points[index], view, lens)
            skin = depth_buffer(xy, depth - .003, subset, width, height)
            patch = np.isfinite(skin) & (skin <= surface + .002) & region
            original = np.isfinite(volume) & (volume <= surface + .013) & region
            extended = original | patch
            target = (mask == 2) & region
            before.append(np.count_nonzero(original & target) / max(1, np.count_nonzero(original | target)))
            after.append(np.count_nonzero(extended & target) / max(1, np.count_nonzero(extended | target)))
            patch_pixels.append((int(np.count_nonzero((patch & ~original) & target)),
                                 int(np.count_nonzero((patch & ~original) & ~target))))
    new_true, new_false = np.sum(patch_pixels, axis=0)
    first, second = float(np.mean(before)), float(np.mean(after))
    if second < first + .004 or new_true < 1.45 * new_false:
        return None
    report = {'hair_overlap_before': first, 'hair_overlap_after': second, 'held_out_views': len(after),
              'correct_added_samples': int(new_true), 'extra_samples': int(new_false), 'faces': len(subset)}
    count = len(frames[0][0])
    report['views'] = [
        {'overlap_before': float(np.mean(before[index::count])), 'overlap_after': float(np.mean(after[index::count])),
         'correct_added_samples': int(sum(good for good, _ in patch_pixels[index::count]))}
        for index in range(count)
    ]
    return subset, report
