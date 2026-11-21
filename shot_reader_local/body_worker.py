"""Run native ComfyUI SAM 3D Body in its own Python environment."""

import json
from pathlib import Path
import sys


def fit(config, output):
    import cv2
    import numpy as np
    import torch

    comfy = Path(config['comfy'])
    weights = Path(config['weights'])
    sys.path.insert(0, str(comfy))
    import folder_paths
    from comfy.cli_args import args
    from comfy.model_management import load_models_gpu
    from comfy_extras.nodes_sam3d_body import SAM3DBody_Loader
    from comfy_extras.sam3d_body.export.glb_skeletal import build_glb_skeletal
    from comfy_extras.sam3d_body.utils import (
        compute_canonical_colors, compute_hand_vert_mask, run_batched_frames,
    )

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')
    args.force_fp32 = args.fp32_unet = True
    folder_paths.add_model_folder_path('detection', str(weights.parent), is_default=True)

    video = cv2.VideoCapture(config['clip'])
    try:
        if not video.isOpened():
            raise ValueError('Could not open the reference video in ComfyUI Python.')
        fps = video.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('The reference video does not report a frame rate.')
        first = max(0, round(config['start'] * fps))
        count = round(config['end'] * fps) - first
        if count < 1:
            raise ValueError('The scene frame range contains no video frames.')
        video.set(cv2.CAP_PROP_POS_FRAMES, first)
        ok, frame = video.read()
        if not ok:
            raise ValueError('The requested scene range starts outside the video.')

        height, width = frame.shape[:2]
        scale = min(1., 768 / max(width, height))
        w, h = round(width * scale), round(height * scale)
        u0, v0, u1, v1 = config['region']
        left, top, right, bottom = round(w * u0), round(h * v0), round(w * u1), round(h * v1)
        if min(right - left, bottom - top) < 32:
            raise ValueError('Choose a larger region that contains the person throughout the shot.')
        fx, fy, cx, cy = config['intrinsics']
        cx = right - 1 - cx * w if config['reflected'] else cx * w - left
        intrinsics = torch.tensor([[[fx * w, 0., cx], [0., fy * h, cy * h - top], [0., 0., 1.]]],
                                  dtype=torch.float32)
        box = torch.tensor([[0., 0., right - left, bottom - top]], dtype=torch.float32)

        print('Loading SAM 3D Body…', flush=True)
        model = SAM3DBody_Loader.execute(weights.name)[0]
        inner = model.model
        batch_size = 2
        load_models_gpu([model], memory_required=inner.memory_used_forward(batch_size, True))

        outputs = []
        batch = []
        for index in range(count):
            if index:
                ok, frame = video.read()
                if not ok:
                    break
            image = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)[top:bottom, left:right]
            if config['reflected']:
                image = cv2.flip(image, 1)
            batch.append(torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            if len(batch) == batch_size or index == count - 1:
                outputs.extend(run_batched_frames(inner, batch, [box] * len(batch), None, inner.image_size,
                                                  'full', cam_int=intrinsics, crops_per_chunk=batch_size))
                batch.clear()
                print(f'Fitting body · {len(outputs)} / {count} frames', flush=True)
                (output / 'progress.txt').write_text(f'Fitting body · {len(outputs)} / {count} frames', encoding='utf-8')
        if len(outputs) != count or any(len(people) != 1 for people in outputs):
            raise ValueError(f'Expected one fitted body on all {count} frames of the scene. Check the range and region.')
    finally:
        video.release()

    translations = np.stack([people[0]['pred_cam_t'] for people in outputs])
    if not np.isfinite(translations).all() or np.any(translations[:, 2] <= 0):
        raise ValueError('The body model did not produce valid camera-space positions.')
    pose_data = {
        'frames': outputs, 'faces': inner.head_pose.faces_np(), 'image_size': (bottom - top, right - left),
        'canonical_colors': compute_canonical_colors(inner), 'hand_vert_mask': compute_hand_vert_mask(inner),
    }
    (output / 'progress.txt').write_text('Building the skinned character…', encoding='utf-8')
    (output / 'body.glb').write_bytes(build_glb_skeletal(
        pose_data, model, fps=fps, track_index=0, camera_translation='off', bone_smooth_window=3,
    ))
    face_vertices = None
    try:
        from comfy_extras.sam3d_body.export.glb_shared import FACE_LANDMARK_TARGETS, select_face_landmark_vert_ids

        canonical = pose_data['canonical_colors']
        landmarks = select_face_landmark_vert_ids(canonical['positions'], canonical['face_mask'])
        named = dict(zip((item[0] for item in FACE_LANDMARK_TARGETS), landmarks.tolist()))
        first = outputs[0][0]
        landmark_ids = [int(np.argmin(((first['pred_vertices'] - anchor) ** 2).sum(axis=1)))
                        for anchor in first['pred_keypoints_3d'][:3]]
        face_vertices = [landmark_ids[0], named['l_brow_mid'], named['r_brow_mid'], *landmark_ids[1:]]
    except (ImportError, KeyError, ValueError):
        print('The model did not expose a face landmark map.', flush=True)
    (output / 'body.json').write_text(json.dumps({'fps': fps, 'translations': translations.tolist(),
                                                 'face_vertices': face_vertices}), encoding='utf-8')


if __name__ == '__main__':
    request = Path(sys.argv[1])
    sys.argv[:] = [sys.argv[0]]
    fit(json.loads(request.read_text(encoding='utf-8')), request.parent)
