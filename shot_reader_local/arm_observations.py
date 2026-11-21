"""Measure arm and hand landmarks in a local ViTPose whole-body model."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort


def observe(request):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        try:
            import torch

            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
        except ImportError:
            pass
    model = ort.InferenceSession(request['model'], sess_options=options, providers=providers)
    input_data = model.get_inputs()[0]
    h, w = map(int, input_data.shape[-2:])
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    native_w, native_h = request['size']
    x0, y0, x1, y1 = np.array(request['region']) * [native_w, native_h, native_w, native_h]
    crop_w = 1.12 * max(x1 - x0, (y1 - y0) * w / h)
    crop_h = crop_w * h / w
    left, top = (x0 + x1 - crop_w) / 2, (y0 + y1 - crop_h) / 2
    transform = np.array([[w / crop_w, 0., -left * w / crop_w], [0., h / crop_h, -top * h / crop_h]])
    cap = cv2.VideoCapture(request['clip'])
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Could not open the linked plate for arm tracking.')
        result = []
        for stamp in request['times']:
            frame = max(0, round(stamp * fps))
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - frame) > .1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError('The linked plate does not cover the whole scene.')
            matrix = transform.copy()
            matrix[0, 0] *= native_w / image.shape[1]
            matrix[1, 1] *= native_h / image.shape[0]
            rgb = cv2.warpAffine(image, matrix, (w, h))[:, :, ::-1].astype(np.float32) / 255.
            tensor = ((rgb - mean) / std).transpose(2, 0, 1)[None].copy()
            heatmap = model.run(None, {input_data.name: tensor})[0][0]
            if len(heatmap) < 133:
                raise ValueError('Use a COCO WholeBody ViTPose model with hand landmarks.')
            indexes = ([5, 6, *range(91, 133)] if request.get('fingers') else
                       [5, 7, 9, 96, 100, 6, 8, 10, 117, 121])
            peaks = []
            for channel in heatmap[indexes]:
                row, col = np.unravel_index(channel.argmax(), channel.shape)
                x, y = float(col), float(row)
                if 0 < col < channel.shape[1] - 1:
                    x += .25 * np.sign(channel[row, col + 1] - channel[row, col - 1])
                if 0 < row < channel.shape[0] - 1:
                    y += .25 * np.sign(channel[row + 1, col] - channel[row - 1, col])
                peaks.append([left + (x + .5) / channel.shape[1] * crop_w, top + (y + .5) / channel.shape[0] * crop_h,
                              float(channel[row, col])])
            result.append(peaks)
    finally:
        cap.release()
    return result


if __name__ == '__main__':
    path = Path(sys.argv[1])
    request = json.loads((path / 'request.json').read_text(encoding='utf-8'))
    (path / 'arms.json').write_text(json.dumps(observe(request), allow_nan=False), encoding='utf-8')
