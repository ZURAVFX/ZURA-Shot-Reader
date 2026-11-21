"""Bake visible plate colours onto consistent vertices of the animated performer."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from foreground_matte import track_mattes


def weighted_colour(samples, confidence):
    strength = confidence.sum(axis=0)
    pixels = np.zeros((len(strength), 3), dtype=np.float32)
    for channel in range(3):
        values = samples[:, :, channel]
        order = values.argsort(axis=0)
        score = np.take_along_axis(confidence, order, axis=0).cumsum(axis=0)
        median = (score >= strength / 2).argmax(axis=0)
        columns = np.arange(len(strength))
        pixels[:, channel] = values[order[median, columns], columns]
    return pixels, strength


def robust_colour(samples, confidence):
    centre, strength = weighted_colour(samples, confidence)
    # Average nearby measurements while limiting the influence of colour outliers.
    distance = np.abs(samples - centre).mean(axis=2)
    weight = confidence * np.minimum(1., 16. / np.maximum(distance, .001))
    colour = (weight[..., None] * samples).sum(axis=0)
    colour /= np.maximum(weight.sum(axis=0)[:, None], 1.e-8)
    return colour, strength


def plate_samples(data, request):
    views = request['views']
    width, height = request['size']
    times = data['times']
    tracked = [track_mattes(request['clip'], times, (width, height), view, Path(request['weights'])) for view in views]
    # Each camera projection has already been depth tested against the posed mesh in Blender.
    coords, weights = data['uv'], data['weights'].copy()
    excluded = data['hair_occluded'].copy() if 'hair_occluded' in data else np.zeros(weights.shape, dtype=bool)
    geometric = excluded.copy()
    masks = None
    if request.get('hair_model'):
        from hair_volume import categories

        masks = categories({**request, 'model': request['hair_model']}, tracked[0][1])
    stream = []
    for view_index, (mattes, images) in enumerate(tracked):
        pixels = coords[view_index]
        colours = []
        for index, (mask, image) in enumerate(zip(mattes, images)):
            if view_index == 1:
                mask = mask & ~tracked[0][0][index]
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            x = np.clip(pixels[index, :, 0], 0, width - 1).astype(np.float32)
            y = np.clip(pixels[index, :, 1], 0, height - 1).astype(np.float32)
            pixel_x, pixel_y = x.round().astype(int), y.round().astype(int)
            # Keep the foreground edge out of the bake even where the tracked mask is imperfect.
            weights[view_index, index] *= np.clip((distance[pixel_y, pixel_x] - 3.) / 9., 0., 1.)
            if masks is not None:
                hair = cv2.dilate((masks[index, view_index] == 2).astype(np.uint8), np.ones((5, 5), np.uint8))
                excluded[view_index, index] |= hair[pixel_y, pixel_x] > 0
            rgb = cv2.remap(image, x[:, None], y[:, None], cv2.INTER_LINEAR)[:, 0, ::-1]
            colours.append(rgb.astype(np.float32))
        stream.append(np.stack(colours))
    return np.stack(stream), weights, geometric, excluded


def bake(data, request):
    samples, weights, geometric, excluded = plate_samples(data, request)
    return fit_samples(data, samples, weights, geometric, excluded)


def fit_samples(data, samples, weights, geometric, excluded):
    times = data['times']
    holdout = np.arange(len(times)) % 4 == 2
    colours = samples[:, ~holdout].reshape(-1, samples.shape[2], 3)
    # If the plate consistently shows hair on a part of the body missed by the hair mesh,
    # retain its plate colour there until there is enough clear evidence for underlying skin.
    primary = np.where(geometric, 0., weights)[:, ~holdout]
    semantic = excluded & ~geometric
    sampled_hair = (primary * semantic[:, ~holdout]).sum(axis=(0, 1))
    sampled_total = primary.sum(axis=(0, 1))
    preserve = sampled_hair / np.maximum(sampled_total, 1.e-6) >= .33
    excluded = geometric | (semantic & ~preserve)
    clear_weights = np.where(excluded, 0., weights)
    confidence = clear_weights[:, ~holdout].reshape(-1, weights.shape[2])
    pixels, strength = robust_colour(colours, confidence)
    direct = strength >= .20
    if direct.sum() < max(200, len(direct) * .04):
        raise ValueError('Too little of the fitted performer is visible. Adjust the performer regions and refit before baking.')
    # Score the fit on reference frames that did not contribute to the colours.
    check_weights = clear_weights[:, holdout, :] * direct
    error = np.abs(samples[:, holdout] - pixels[None, None]).mean(axis=3)
    baseline = np.abs(samples[:, holdout] - 188.).mean(axis=3)
    total = check_weights.sum()
    report = {'views': samples.shape[0], 'sampled_frames': len(times), 'held_out_frames': int(holdout.sum()),
              'observed_vertices': int(direct.sum()), 'vertices': len(direct),
              'heldout_vertex_samples': int(np.count_nonzero(check_weights)),
              'plate_colour_error': float((error * check_weights).sum() / total) if total else None,
              'uniform_grey_error': float((baseline * check_weights).sum() / total) if total else None}
    removed = (weights > 0) & excluded
    if removed.any() and total > 0:
        unfiltered = weights[:, ~holdout].reshape(-1, weights.shape[2])
        original, original_strength = weighted_colour(colours, unfiltered)
        comparison = check_weights * (original_strength >= .20)
        count = comparison.sum()
        original_error = np.abs(samples[:, holdout] - original[None, None]).mean(axis=3)
        report.update(hair_samples_rejected=int(removed.sum()),
                      geometry_samples_rejected=int((removed & geometric).sum()),
                      comparable_vertex_samples=int(np.count_nonzero(comparison)),
                      unfiltered_colour_error=float((comparison * original_error).sum() / count) if count else None,
                      filtered_colour_error=float((comparison * error).sum() / count) if count else None)
        affected = comparison * removed[:, ~holdout].any(axis=(0, 1))
        affected_weight = affected.sum()
        previous, _ = weighted_colour(colours, confidence)
        previous_error = np.abs(samples[:, holdout] - previous[None, None]).mean(axis=3)
        report['previous_filtered_colour_error'] = float((previous_error * check_weights).sum() / total)
        if affected_weight:
            report.update(affected_vertex_samples=int(np.count_nonzero(affected)),
                          affected_unfiltered_error=float((affected * original_error).sum() / affected_weight),
                          affected_previous_error=float((affected * previous_error).sum() / affected_weight),
                          affected_filtered_error=float((affected * error).sum() / affected_weight))
    edges, positions = data['edges'], data['rest']
    a, b = edges.T
    length = np.maximum(1e-6, np.linalg.norm(positions[a] - positions[b], axis=1))
    graph = coo_matrix((length, (a, b)), shape=(len(direct), len(direct))).tocsr()
    distance, _previous, sources = dijkstra(graph, directed=False, indices=np.flatnonzero(direct), min_only=True,
                                            return_predecessors=True)
    neighbours = ~direct & np.isfinite(distance)
    pixels[neighbours] = pixels[sources[neighbours]]
    pixels[~direct & ~neighbours] = np.median(pixels[direct], axis=0)
    return pixels / 255., np.minimum(strength, 1.) * direct, report


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    with np.load(directory / 'surface.npz', allow_pickle=False) as data:
        colour, confidence, report = bake(data, request)
    np.savez(directory / 'colour.npz', colour=colour, confidence=confidence)
    (directory / 'colour.json').write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
