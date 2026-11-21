"""Separate coherent counter and fixture surfaces from a registered depth mesh."""

import cv2
import numpy as np

from fixture_profile import fit_profile


def connected_faces(screen, triangles, faces, eligible, *, close):
    """Find the largest supported image component without assigning unseen faces."""
    width, height = 320, 480
    pixels = np.rint(screen * [width - 1, height - 1]).astype(np.int32)
    stencil = np.zeros((height, width), dtype=np.uint8)
    for face in faces[eligible]:
        cv2.fillConvexPoly(stencil, pixels[face], 1)
    if close:
        stencil = cv2.morphologyEx(stencil, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(stencil, connectivity=8)
    if count < 2:
        return np.zeros(len(faces), dtype=bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    indices = np.clip(np.rint(triangles * [width - 1, height - 1]), 0, [width - 1, height - 1]).astype(int)
    return eligible & (labels[indices[:, 1], indices[:, 0]] == component)


def fit_counter(world, faces, screen, centres, region, normals):
    xyz = world[faces].mean(axis=1)
    flat = region & (abs(normals[:, 2]) >= .92)
    tiles = np.floor(centres * [48, 64]).astype(int)
    holdout = (tiles[:, 0] + tiles[:, 1]) % 4 == 1
    training = flat & ~holdout
    if training.sum() < 220:
        return None
    z = xyz[:, 2]
    levels = np.floor(z[training] / .018).astype(np.int64)
    if levels.max() - levels.min() > 2000:
        return None
    keys, counts = np.unique(levels, return_counts=True)
    height = (keys[counts.argmax()] + .5) * .018
    training &= abs(z - height) < .028
    if training.sum() < 180:
        return None
    pivot = np.median(xyz[training, :2], axis=0)
    design = np.column_stack((xyz[:, :2] - pivot, np.ones(len(xyz))))
    plane, _, rank, _ = np.linalg.lstsq(design[training], z[training], rcond=None)
    if rank < 3 or np.linalg.norm(plane[:2]) > .14:
        return None
    residual = abs(design @ plane - z)
    training &= residual < .014
    if training.sum() < 150:
        return None
    plane = np.linalg.lstsq(design[training], z[training], rcond=None)[0]
    if np.linalg.norm(plane[:2]) > .14:
        return None
    residual = abs(design @ plane - z)
    support = region & (abs(normals[:, 2]) > .70) & (residual < .021)
    counter = connected_faces(screen, centres, faces, support, close=5)
    checking = counter & holdout & flat
    if counter.sum() < 200 or checking.sum() < 40:
        return None
    error = float(np.median(residual[checking]))
    extent = np.ptp(xyz[counter, :2], axis=0)
    if error > .010 or min(extent) < .06 or max(extent) < .15:
        return None
    return counter, plane, pivot, {
        'faces': int(counter.sum()),
        'held_out_faces': int(checking.sum()),
        'median_plane_residual': error,
        'estimate_world_z': float(plane[2]),
        'visible_extent_xy': extent.tolist(),
    }


def fit_backsplash(measured, centres, normals, available, above, counter_plane, pivot, config):
    """Fit the exposed wall between the calibrated mirror and the counter."""
    origin = np.asarray(config['wall_point'])
    normal = np.asarray(config['wall_toward_camera'])
    horizontal = np.r_[normal[:2], 0.]
    length = np.linalg.norm(horizontal)
    if length < .8:
        return None
    horizontal /= length
    tangent = np.cross([0., 0., 1.], horizontal)
    relative = measured - origin
    across, altitude = relative @ tangent, relative[:, 2]
    distance = relative @ normal
    support = (available & (above > .095) & (above < .72) & (abs(normals @ normal) > .86)
               & (distance > -.045) & (distance < .16))
    blocks = np.floor(centres * [48, 64]).astype(int)
    withheld = (blocks[:, 0] + blocks[:, 1]) % 4 == 1
    train, check = support & ~withheld, support & withheld
    if train.sum() < 250 or check.sum() < 100:
        return None
    lo, hi = np.percentile(across[train], [1, 99])
    bottom, top = np.percentile(altitude[train], [3, 97])
    if hi - lo < .26 or top - bottom < .24:
        return None
    design = np.column_stack((across, altitude, np.ones(len(measured))))
    coefficients = np.linalg.lstsq(design[train], distance[train], rcond=None)[0]
    for _ in range(3):
        residual = design @ coefficients - distance
        keep = train & (abs(residual) < .033)
        if keep.sum() < 180:
            return None
        coefficients = np.linalg.lstsq(design[keep], distance[keep], rcond=None)[0]
    if np.linalg.norm(coefficients[:2]) > .26:
        return None
    residual = abs(design @ coefficients - distance)
    error = float(np.median(residual[check]))
    baseline = float(np.median(abs(distance[check])))
    if error > .019 or np.percentile(residual[check], 90) > .053 or (baseline > .019 and error > .65 * baseline):
        return None

    def point(s, z):
        facing = (coefficients @ [s, z, 1.] - normal[2] * z) / length
        return origin + tangent * s + np.array([0., 0., z]) + horizontal * facing

    corners = []
    for s in (lo, hi):
        root = point(s, 0.)
        up = point(s, 1.) - root
        slope = counter_plane[:2]
        height = (counter_plane[2] + (root[:2] - pivot) @ slope - root[2]) / (up[2] - up[:2] @ slope)
        if not -.22 <= bottom - height <= .35 or top - height < .18:
            return None
        corners.extend([point(s, height - .007), point(s, top + .025)])
    correction = distance - design @ coefficients
    removed = (available & (across >= lo) & (across <= hi) & (altitude < top + .025)
               & (above > -.008) & (correction > -.075) & (correction < .105))
    if removed.sum() < 250:
        return None
    return removed, {'source_faces': int(removed.sum()), 'corners': np.asarray(corners)[[0, 2, 3, 1]].tolist(),
                     'held_out_faces': int(check.sum()), 'median_plane_residual': error,
                     'median_mirror_plane_residual': baseline}


def support_blockout(counter_points, counter_plane, pivot, backsplash_corners, base):
    """Continue the measured counter down to the existing vanity datum."""
    rear = np.asarray(backsplash_corners[:2])[:, :2]
    direction = rear[1] - rear[0]
    width = np.linalg.norm(direction)
    if width < .26:
        return None
    right = direction / width
    forward = np.array([-right[1], right[0]])
    measured = counter_points[:, :2] - rear[0]
    if np.median(measured @ forward) < 0.:
        forward *= -1.
    lo, hi = np.percentile(measured @ right, [4, 96])
    lo, hi = max(-.04, lo + .025), min(width + .04, hi - .025)
    front = np.percentile(measured @ forward, 82) - .025
    if not (.22 <= hi - lo <= 3. and .14 <= front <= .95):
        return None
    panel = np.array([rear[0] + right * u + forward * v for u, v in
                      ((lo, .015), (hi, .015), (hi, front), (lo, front))])
    z = (panel - pivot) @ counter_plane[:2] + counter_plane[2] - .016
    if np.min(z) - base < .12 or np.max(z) - base > 1.6:
        return None
    if np.cross(np.r_[panel[1] - panel[0], 0.], np.r_[panel[2] - panel[0], 0.])[2] < 0.:
        panel, z = panel[::-1], z[::-1]
    return {'corners': np.vstack((np.column_stack((panel, np.full(4, base))),
                                  np.column_stack((panel, z)))).tolist(),
            'base_source': 'existing vanity blockout', 'width': float(hi - lo), 'depth': float(front - .015)}


def fit_faucet(world, faces, screen, centres, eligible, above, config, plate):
    xyz = world[faces].mean(axis=1)
    away_from_wall = (xyz - config['wall_point']) @ np.asarray(config['wall_toward_camera'])
    low, high = config['faucet_u']
    margin = max(.11, min(.26, (high - low) * 1.55))
    nearby = (centres[:, 0] > low - margin) & (centres[:, 0] < high + margin)
    height, width = plate.shape[:2]
    x, y = np.clip(np.rint(centres * [width - 1, height - 1]), 0, [width-1, height-1]).astype(int).T
    rgb = plate[y, x].astype(float)
    light = rgb.max(axis=1)
    metal = (light > 93.) & (np.ptp(rgb, axis=1) < np.minimum(46., 12. + .21 * light))
    possible = eligible & nearby & (above > .012) & (above < .52) & (away_from_wall > .13)
    bright = connected_faces(screen, centres, faces, possible & metal, close=9)
    if bright.sum() < 110:
        return None
    # Recover adjacent dark reflections without filling the gaps in a tap's silhouette.
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    _, occurrences = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    occurrences = occurrences.reshape(len(faces), 3)
    supported = np.zeros(int(occurrences.max()) + 1, dtype=bool)
    supported[occurrences[bright]] = True
    part = bright | (possible & supported[occurrences].any(axis=1))
    if (above[part] < .105).sum() < 30 or np.ptp(above[part]) < .10:
        return None
    return part, {'faces': int(part.sum()), 'chrome_seed_faces': int(bright.sum()),
                  'visible_height': float(np.ptp(above[part])),
                  'median_clearance_from_wall': float(np.median(away_from_wall[part]))}


def measure_fixtures(xyz, uv, faces, plate, config):
    matrix = np.asarray(config['world'])
    world = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    screen = uv * [1., -1.] + [0., 1.]
    centres = screen[faces].mean(axis=1)
    low, high = config['counter_u']
    region = ((centres[:, 0] > low - .045) & (centres[:, 0] < high + .045)
              & (centres[:, 1] > config['reference_bottom'] + .025))
    triangles = world[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1.e-9)
    result = fit_counter(world, faces, screen, centres, region, normals)
    labels = np.zeros(len(faces), dtype=np.uint8)
    report = {}
    if result is None:
        return labels, report
    counter, plane, pivot, report['counter'] = result
    labels[counter] = 1
    measured = triangles.mean(axis=1)
    above = measured[:, 2] - ((measured[:, :2] - pivot) @ plane[:2] + plane[2])
    if 'faucet_u' in config:
        faucet = fit_faucet(world, faces, screen, centres, region & ~counter, above, config, plate)
        if faucet:
            subset, report['faucet'] = faucet
            labels[subset] = 2
            if 'intrinsics' in config:
                arch = fit_profile(plate, screen, faces, subset, measured, above, config)
                if arch:
                    relief, report['faucet_profile'] = arch
                    labels[relief & ~counter] = 3
                    # Depth at a foreground boundary can bridge the fixture and wall.
                    # Keep those pixels in the plate render and out of the orbit geometry.
                    observer = matrix[:3, 3] - measured
                    observer /= np.maximum(np.linalg.norm(observer, axis=1)[:, None], 1.e-9)
                    facing = abs(np.sum(normals * observer, axis=1))
                    depth_span = np.ptp(xyz[faces, 2], axis=1)
                    bridges = (region & ~counter & (above > .018) & (facing < .35)
                               & (depth_span > .022) & (labels != 3))
                    labels[bridges] = 4
                    report['faucet']['faces'] = int((labels == 2).sum())
                    report['faucet_profile']['source_faces'] = int((labels == 3).sum())
                    if bridges.any():
                        report['fixture_depth_bridges'] = {'source_faces': int(bridges.sum())}
    backsplash = fit_backsplash(measured, centres, normals, region & (labels == 0), above, plane, pivot, config)
    if backsplash:
        wall, report['backsplash'] = backsplash
        labels[wall] = 5
        if 'support_base' in config:
            support = support_blockout(measured[counter], plane, pivot, report['backsplash']['corners'],
                                       config['support_base'])
            if support:
                report['counter_support'] = support
    return labels, report
