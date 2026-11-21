"""Carve a deforming hair volume using local foreground and mirror masks."""

import json
from pathlib import Path
import sys

import cv2
import mediapipe as mp
from numba import njit
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from room_clean import raster_exclusions, read_frames
from silhouette_fit import fit_body
from hair_shell import fit_shell
from hair_motion import fit_motion


@njit(cache=True)
def depth_buffer(pixels, depth, faces, width, height):
    result = np.full((height, width), np.inf, np.float32)
    for i, j, k in faces:
        za, zb, zc = depth[i], depth[j], depth[k]
        if min(za, zb, zc) <= .02:
            continue
        ax, ay = pixels[i]
        bx, by = pixels[j]
        cx, cy = pixels[k]
        area = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(area) < .001:
            continue
        for y in range(max(0, int(np.floor(min(ay, by, cy)))), min(height - 1, int(np.ceil(max(ay, by, cy)))) + 1):
            for x in range(max(0, int(np.floor(min(ax, bx, cx)))), min(width - 1, int(np.ceil(max(ax, bx, cx)))) + 1):
                u = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / area
                v = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / area
                w = 1. - u - v
                if min(u, v, w) >= 0.:
                    z = 1. / (u / za + v / zb + w / zc)
                    if z < result[y, x]:
                        result[y, x] = z
    return result


def project(points, matrix, lens):
    xyz = points @ matrix[:3, :3].T + matrix[:3, 3]
    depth = -xyz[:, 2]
    fx, fy, cx, cy = lens
    return xyz[:, :2] / np.maximum(depth[:, None], .01) * [fx, -fy] + [cx, cy], depth


def skin(points, joints, head_height):
    z = points[:, 2]
    head = np.clip((z - head_height + .22) / .18, 0., 1.)
    torso = np.clip((z - head_height + .48) / .27, 0., 1.)
    weights = np.column_stack((head, torso * (1 - head), (1 - torso) * (1 - head)))
    transformed = np.stack([points @ matrix[:3, :3].T + matrix[:3, 3] for matrix in joints], axis=1)
    return (transformed * weights[:, :, None]).sum(axis=1), weights


def categories(request, frames):
    config = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=request['model']),
        output_category_mask=True,
    )
    width, height = request['size']
    regions = [np.rint(np.array(view['region']) * [width, height, width, height]).astype(int) for view in request['views']]
    out = []
    with mp.tasks.vision.ImageSegmenter.create_from_options(config) as model:
        if 'hair' not in model.labels:
            raise ValueError('Use a local person segmentation model with a hair category.')
        hair = model.labels.index('hair')
        for frame in frames:
            planes = []
            for x0, y0, x1, y1 in regions:
                rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                data = model.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).category_mask.numpy_view()[..., 0]
                group = np.zeros((height, width), np.uint8)
                group[y0:y1, x0:x1] = np.where(data == hair, 2, data > 0)
                planes.append(group)
            out.append(planes)
    return np.asarray(out)


def reconstruct(request, data):
    width, height = request['size']
    images = read_frames(request['clip'], request['times'], (width, height))
    masks = categories(request, images)
    holdout = np.arange(len(images)) % 4 == 2
    mesh = data['body']
    faces = data['faces']
    poses = data['hair_pose']
    matrices = data['views']
    lenses = data['lenses']
    head = data['head']
    rest = data['rest']
    pitch = .009
    axes = [np.arange(head[0] - .26, head[0] + .266, pitch),
            np.arange(head[1] - .13, head[1] + .235, pitch),
            np.arange(head[2] - .67, head[2] + .235, pitch)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    candidates = grid.reshape(-1, 3)
    distance, near = cKDTree(rest).query(candidates)
    signed = ((candidates - rest[near]) * data['normals'][near]).sum(axis=1)
    eligible = (distance < .14) & (signed > -.015)
    candidates = candidates[eligible]
    positives = np.zeros(len(candidates), dtype=np.int32)
    negatives = positives.copy()
    allowed = []
    for frame in range(len(images)):
        regions = []
        for view in range(matrices.shape[1]):
            region = raster_exclusions((width, height), [[
                [request['views'][view]['region'][a], request['views'][view]['region'][b]]
                for a, b in ((0, 1), (2, 1), (2, 3), (0, 3))
            ]])
            if view == 1:
                aperture = raster_exclusions((width, height), [request['mirror'][frame]])
                region &= aperture & (masks[frame, 0] == 0)
            regions.append(region)
        allowed.append(regions)
    fit = fit_body(request, data, masks, allowed, holdout, project, depth_buffer) if 'body_modes' in data else None
    original_mesh = mesh
    shape_delta, shape_report = None, None
    if fit:
        mesh, shape_delta, shape_report = fit
    camera_z = []
    for frame in range(len(images)):
        buckets, regions = [], allowed[frame]
        for view in range(matrices.shape[1]):
            image, z = project(mesh[frame], matrices[frame, view], lenses[frame])
            buckets.append(depth_buffer(image, z, faces, width, height))
        camera_z.append(buckets)
        if holdout[frame]:
            continue
        world, _ = skin(candidates, poses[frame], head[2])
        for view in range(matrices.shape[1]):
            xy, z = project(world, matrices[frame, view], lenses[frame])
            inside = ((xy >= 0) & (xy < [width - 1, height - 1])).all(axis=1) & (z > .02)
            px = np.clip(np.rint(xy), [0, 0], [width - 1, height - 1]).astype(int)
            x, y = px.T
            visible = inside & regions[view][y, x] & (z <= buckets[view][y, x] + .027)
            is_hair = masks[frame, view, y, x] == 2
            positives += visible & is_hair
            negatives += visible & ~is_hair

    accepted = (positives >= max(3, int((~holdout).sum() * .23))) & (positives >= 1.5 * negatives)
    volume = np.zeros(len(eligible), dtype=bool)
    volume[eligible] = accepted
    volume = volume.reshape(grid.shape[:3])
    groups, total = label(volume)
    if total == 0:
        raise ValueError('The hair masks did not support a continuous volume.')
    sizes = np.bincount(groups.ravel())
    sizes[0] = 0
    volume &= sizes[groups] >= max(30, int(sizes.max() * .035))
    padding = 2
    field = gaussian_filter(np.pad(volume.astype(np.float32), padding), .9)
    if field.max() <= .47:
        raise ValueError('The observed hair was too thin to build a stable volume.')
    vertices, hair_faces, _, _ = marching_cubes(field, level=.47, spacing=(pitch,) * 3)
    vertices += [axis[0] - padding * pitch for axis in axes]
    hair_faces = np.ascontiguousarray(hair_faces[:, ::-1])
    _, skin_weights = skin(vertices, poses[0], head[2])
    posed = np.asarray([skin(vertices, pose, head[2])[0] for pose in poses])
    motion = fit_motion(vertices, hair_faces, head, posed, poses, skin_weights,
                        list(zip(matrices, lenses, allowed, masks, camera_z)),
                        holdout, project, depth_buffer, np.asarray(request['times']))
    if motion:
        modes, path, world_deltas, motion_report = motion
        posed += np.einsum('fm,fmvc->fvc', path, world_deltas)
    neutral, raw, composed, hair_quality, palette, colour_samples = [], [], [], [], [], []
    hair_buffers = []
    for frame in range(len(images)):
        world = posed[frame]
        buffers = []
        for view in range(matrices.shape[1]):
            image, z = project(world, matrices[frame, view], lenses[frame])
            depth = depth_buffer(image, z, hair_faces, width, height)
            buffers.append(depth)
            body_mask = np.isfinite(camera_z[frame][view])
            hair_mask = np.isfinite(depth) & (depth <= camera_z[frame][view] + .013)
            region = allowed[frame][view]
            target = (masks[frame, view] > 0) & region
            target_hair = (masks[frame, view] == 2) & region
            if holdout[frame]:
                before, after = body_mask & region, (body_mask | hair_mask) & region
                raw.append((target & before).sum() / max(1, (target | before).sum()))
                composed.append((target & after).sum() / max(1, (target | after).sum()))
                hair_quality.append((hair_mask & target_hair).sum() / max(1, ((hair_mask & region) | target_hair).sum()))
                if fit:
                    image, z = project(original_mesh[frame], matrices[frame, view], lenses[frame])
                    unchanged = np.isfinite(depth_buffer(image, z, faces, width, height)) & region
                    neutral.append((target & unchanged).sum() / max(1, (target | unchanged).sum()))
            else:
                pixels = images[frame][hair_mask & target_hair]
                if len(pixels):
                    palette.append(np.median(pixels[:, ::-1], axis=0))
                px = np.rint(image).astype(int)
                in_frame = ((px >= 0) & (px < [width, height])).all(axis=1)
                x, y = np.clip(px, [0, 0], [width - 1, height - 1]).T
                seen = (in_frame & target_hair[y, x] & (z > .02) &
                        (z <= np.minimum(depth, camera_z[frame][view])[y, x] + .018))
                colour_samples.append(np.where(seen[:, None], images[frame][y, x, ::-1], np.nan))
        hair_buffers.append(buffers)
    baseline, final = float(np.mean(raw)), float(np.mean(composed))
    if not palette or final < baseline + .012:
        raise ValueError(f'Added hair did not sufficiently improve the held-out performer silhouette ({baseline:.1%} to {final:.1%}).')
    rgb = np.median(palette, axis=0) / 255.
    samples = np.stack(colour_samples)
    observed = (np.isfinite(samples[:, :, 0]).sum(axis=0) >= 2)
    colours = np.tile(rgb, (len(vertices), 1))
    if observed.any():
        colours[observed] = np.nanmedian(samples[:, observed], axis=0) / 255.
        _, closest = cKDTree(vertices[observed]).query(vertices[~observed])
        colours[~observed] = colours[observed][closest]
    report = {'views': int(matrices.shape[1]), 'fit_frames': int((~holdout).sum()), 'held_out_frames': int(holdout.sum()),
              'silhouette_before_iou': float(np.mean(neutral)) if fit else baseline,
              'silhouette_after_iou': final, 'hair_iou': float(np.mean(hair_quality)),
              'vertices': len(vertices), 'faces': len(hair_faces), 'rgb': rgb.tolist(), 'observed_colour_vertices': int(observed.sum())}
    if shape_report:
        report['body_correction'] = shape_report
        report['silhouette_with_body_only_iou'] = baseline
    shell = fit_shell(rest, mesh, faces, head, list(zip(matrices, lenses, allowed, masks, camera_z, hair_buffers)),
                      holdout, project, depth_buffer)
    shell_faces = None
    if shell:
        shell_faces, report['draped_surface'] = shell
        report['hair_iou'] = report['draped_surface']['hair_overlap_after']
    motion_arrays = {}
    if motion:
        report['hair_motion'] = motion_report
        motion_arrays = {'hair_modes': modes, 'hair_path': path}
    return vertices, hair_faces, skin_weights, colours, shape_delta, shell_faces, motion_arrays, report


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    with np.load(folder / 'input.npz', allow_pickle=False) as archive:
        vertices, faces, weights, colours, delta, shell_faces, motion_arrays, info = reconstruct(request, archive)
    arrays = {'vertices': vertices, 'faces': faces, 'weights': weights, 'colours': colours, **motion_arrays}
    if delta is not None:
        arrays['body_delta'] = delta
    if shell_faces is not None:
        arrays['shell_faces'] = shell_faces
    np.savez(folder / 'hair.npz', **arrays)
    (folder / 'hair.json').write_text(json.dumps(info, allow_nan=False), encoding='utf-8')
