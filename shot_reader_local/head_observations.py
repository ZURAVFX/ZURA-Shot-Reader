"""Measure face anchors in the performer region of the linked plate."""

import json
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np


def observe(request):
    detector = mp.tasks.vision.FaceLandmarker
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=request['model']),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    movie = cv2.VideoCapture(request['clip'])
    try:
        fps = movie.get(cv2.CAP_PROP_FPS)
        if not movie.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Could not open the linked plate for face tracking.')
        timeline = request['times']
        width, height = request['size']
        factor = min(1., 960 / max(width, height))
        w, h = round(width * factor), round(height * factor)
        x0, y0, x1, y1 = np.rint(np.asarray(request['region']) * [w, h, w, h]).astype(int)
        if min(x1 - x0, y1 - y0) < 32:
            raise ValueError('Frame a larger performer region including the face.')
        observations = []
        with detector.create_from_options(options) as model:
            for time in timeline:
                frame = max(0, round(time * fps))
                if abs(movie.get(cv2.CAP_PROP_POS_FRAMES) - frame) > .1:
                    movie.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = movie.read()
                if not ok:
                    raise ValueError('The source clip does not cover the scene frame range.')
                crop = cv2.cvtColor(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                result = model.detect_for_video(image, round(1000 * (time - timeline[0])))
                if not result.face_landmarks:
                    observations.append(None)
                    continue
                landmarks = result.face_landmarks[0]

                def xy(index):
                    point = landmarks[index]
                    return np.array([(x0 + point.x * (x1 - x0)) * width / w,
                                     (y0 + point.y * (y1 - y0)) * height / h])

                brows = sorted([xy(105), xy(334)], key=lambda point: point[0])
                eyes = sorted([(xy(33) + xy(133)) * .5, (xy(263) + xy(362)) * .5], key=lambda point: point[0])
                observations.append(np.stack([xy(1), *brows, *eyes]).tolist())
    finally:
        movie.release()
    return observations


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    result = observe(request)
    (directory / 'face.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
