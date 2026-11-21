"""Find performer regions and an anchor frame with a local person detector."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from room_clean import raster_exclusions, read_frames


def analyse(settings):
    rw, rh = settings['size']
    scale = min(1., 768 / max(rw, rh))
    width, height = round(rw * scale), round(rh * scale)
    size = width, height
    frames = read_frames(settings['clip'], settings['times'], size)
    mirror = raster_exclusions(size, [settings['mirror']]) if settings['mirror'] else None
    model = YOLO(settings['weights'])
    detections = model.predict(list(frames), imgsz=768, classes=[0], conf=.28, verbose=False)
    calibration, warning = None, None
    if settings.get('calibration'):
        from mirror_detect import calibrate_rim, camera_from_corners, find_corners

        request = settings['calibration']
        for index, (image, result) in enumerate(zip(frames[:3], detections[:3])):
            if result.masks is None or len(result.boxes) < 2:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            direct, reflected = np.argsort(areas)[-2:][::-1]
            if areas[reflected] / areas[direct] > .82 or areas[reflected] < width * height * .035:
                continue
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [np.rint(result.masks.xy[reflected]).astype(np.int32)], 1)
            try:
                quad, error = find_corners(image, boxes[direct], boxes[reflected], mask.astype(bool), Path(request['matte']))
                intrinsics = np.asarray(request['intrinsics']) * [width / rw, height / rh, width / rw, height / rh]
                calibration = camera_from_corners(request['world'], quad, intrinsics)
            except ValueError:
                continue
            foreground = np.zeros_like(mask)
            cv2.fillPoly(foreground, [np.rint(result.masks.xy[direct]).astype(np.int32)], 1)
            foreground = cv2.dilate(foreground, np.ones((7, 7), np.uint8))
            try:
                quad, calibration = calibrate_rim(image, quad, foreground, request['world'], intrinsics,
                                                  refine_focal=request.get('refine_focal', True))
            except (ValueError, np.linalg.LinAlgError):
                pass
            else:
                error = float(np.median(calibration['rim_heldout_after_px']))
                for key in ('rim_edge_errors_px', 'radial_edge_errors_px', 'rim_heldout_before_px', 'rim_heldout_after_px'):
                    calibration[key] = [item / scale for item in calibration[key]]
            calibration['edge_holdout_median_px'] = error / scale
            calibration['corner_rms'] /= scale
            calibration['projected_corners'] = (np.asarray(calibration['projected_corners']) / [width, height]).tolist()
            calibration['corners'] = (quad / [width, height]).tolist()
            calibration['frame'] = settings['frames'][index]
            mirror = raster_exclusions(size, [calibration['corners']])
            break
        if calibration is None:
            warning = 'The local detector could not align the camera to the mirror edges. The starting lens was retained.'
    pairs = []
    for index, frame in enumerate(detections):
        seen = {}
        if frame.masks is None:
            continue
        for polygon, bounds, score in zip(frame.masks.xy, frame.boxes.xyxy.cpu().numpy(), frame.boxes.conf.cpu().numpy()):
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
            area = int(mask.sum())
            if area < width * height * .008:
                continue
            reflected = bool(mirror is not None and (mask & mirror).sum() > area * .68)
            if reflected in seen:
                continue
            x0, y0, x1, y1 = bounds
            rows = np.mgrid[:height, :width][0]
            y = y0 + min((y1-y0) * .16, (x1-x0) * .47)
            band = (abs(rows - y) <= max(5., (x1-x0) * .06)) & mask.astype(bool)
            iy, ix = np.where(band)
            if len(ix):
                seen[reflected] = {'bounds': bounds, 'head': [float(np.median(ix) / width), float(np.median(iy) / height)], 'score': float(score)}
        if False in seen and (mirror is None or True in seen):
            pairs.append((index, seen))
    if not pairs:
        if mirror is not None or settings.get('calibration'):
            warning = 'The local detector could not establish the mirror and performer views. The visible set was processed independently.'
        return {'calibration': calibration, 'warning': warning}
    seed, chosen = pairs[0]
    labels = {}
    for reflected in chosen:
        boxes = np.stack([entry[reflected]['bounds'] for _, entry in pairs])
        l, t = boxes[:, :2].min(axis=0)
        r, b = boxes[:, 2:].max(axis=0)
        region = [l - width * .024, t - height * .033, r + width * .025, b + height * .01]
        labels['reflected' if reflected else 'direct'] = {
            'head': chosen[reflected]['head'], 'region': np.clip(np.asarray(region) / [width, height, width, height], 0, 1).tolist(),
        }
    return {'seed': settings['frames'][seed], 'calibration': calibration, 'warning': warning, **labels}


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    (directory / 'subject.json').write_text(json.dumps(analyse(request), allow_nan=False), encoding='utf-8')
