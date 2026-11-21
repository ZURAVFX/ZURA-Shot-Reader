"""Track an artist-selected foreground performer with a local SAM checkpoint."""

import cv2
import numpy as np


def person_component(mask, anchor):
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count == 1:
        return mask
    x, y = anchor
    component = labels[y, x] or 1 + statistics[1:, cv2.CC_STAT_AREA].argmax()
    return labels == component


def track_mattes(path, times, size, selection, checkpoint):
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = checkpoint.name.lower()
    variant = next((v for v in ('vit_b', 'vit_l', 'vit_h') if v in name), 'vit_b')
    model = sam_model_registry[variant]()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    predictor = SamPredictor(model.eval().to(device))
    width, height = size
    capture = cv2.VideoCapture(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Could not open the linked plate for matte tracking.')
        timeline = np.r_[times, selection['time']]
        frames = []
        for seconds in timeline:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
            ok, image = capture.read()
            if not ok:
                raise ValueError('The linked plate is shorter than the scene range.')
            frames.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    finally:
        capture.release()

    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    x0, y0, x1, y1 = np.rint(np.array(selection['region']) * ([width - 1, height - 1] * 2)).astype(int)
    anchor = tuple(np.rint(np.array(selection['head']) * [width - 1, height - 1]).astype(int))
    if min(x1 - x0, y1 - y0) < 30 or not (x0 <= anchor[0] <= x1 and y0 <= anchor[1] <= y1):
        raise ValueError('Frame the whole foreground performer and pin the head inside that region.')
    height_span, width_span = y1 - y0, x1 - x0
    margin = max(8, round(min(size) * .045))
    box = (x0, y0, x1, y1)
    points = np.array([anchor, (round(x0 + .32 * width_span), round(y0 + .65 * height_span)),
                       (round(x0 + .36 * width_span), round(y0 + .86 * height_span))])
    ys, xs = np.mgrid[:height, :width].astype(np.float32)
    prior_box = (xs >= x0 - margin) & (xs <= x1 + margin) & (ys >= y0 - margin) & (ys <= y1 + margin)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def segment(index, prompts, region, prior=None):
        predictor.set_image(frames[index], image_format='BGR')
        segments, quality, _ = predictor.predict(point_coords=prompts, point_labels=np.ones(len(prompts)),
                                                  box=np.array(region), multimask_output=True)
        guide = prior_box if prior is None else prior
        intersection = (segments & guide).sum(axis=(1, 2))
        area = np.maximum(segments.sum(axis=(1, 2)), 1)
        if prior is None:
            ranking = quality + .6 * intersection / area + .22 * intersection / guide.sum()
            allowed = prior_box
        else:
            ranking = quality * .35 + intersection / (area + guide.sum() - intersection)
            allowed = cv2.dilate(prior.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
        mask = person_component(segments[ranking.argmax()] & allowed, tuple(prompts[0]))
        if mask.sum() < width * height * .002:
            raise ValueError('Foreground tracking lost the performer. Frame the performer more closely or supply a matte.')
        return mask

    seed_index = len(frames) - 1
    seed = segment(seed_index, points, box)
    masks = [None] * len(times)
    split = int(np.searchsorted(times, selection['time']))
    for direction in (range(split, len(times)), range(split - 1, -1, -1)):
        previous, last_index = seed, seed_index
        for index in direction:
            displacement = flow.calc(gray[index], gray[last_index], None)
            prior = cv2.remap(previous.astype(np.uint8), xs + displacement[..., 0], ys + displacement[..., 1],
                              cv2.INTER_NEAREST).astype(bool)
            y, x = np.where(prior)
            if not x.size:
                raise ValueError('Foreground tracking lost the performer. Pick a clearer reference frame.')
            region = (max(0, x.min() - margin), max(0, y.min() - margin),
                      min(width - 1, x.max() + margin), min(height - 1, y.max() + margin))
            distance = cv2.distanceTransform(prior.astype(np.uint8), cv2.DIST_L2, 3)
            bands = np.linspace(y.min(), y.max() + 1, 4).astype(int)
            tracked = []
            for start, end in zip(bands[:-1], bands[1:]):
                band = distance[start:end]
                if band.size and band.max() > 2:
                    py, px = np.unravel_index(band.argmax(), band.shape)
                    tracked.append((px, py + start))
            if not tracked:
                raise ValueError('The foreground matte became too thin to track.')
            previous = masks[index] = segment(index, np.array(tracked), region, prior)
            last_index = index
    return np.stack(masks), frames[:-1]
