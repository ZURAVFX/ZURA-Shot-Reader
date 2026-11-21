"""Solve a fixed-frame Blender camera from measured 3D-to-plate correspondences."""

import json
import sys


def solve(request):
    import cv2 as cv
    import numpy as np
    from scipy.optimize import least_squares

    width, height = request["size"]
    fx, fy, cx, cy = request["intrinsics"]
    threshold = float(request["threshold"])
    points = request["points"]
    xyz = np.array([point["xyz"] for point in points], dtype=np.float64)
    pixels = np.array([point["uv"] for point in points], dtype=np.float64) * [width, height]
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(pixels)):
        raise ValueError("The landmark coordinates must be finite.")
    fit = np.array([not point["holdout"] for point in points])
    eligible = np.flatnonzero(fit)
    if len(eligible) < 4:
        raise ValueError("Mark at least four fitting landmarks.")
    centred = xyz[eligible] - xyz[eligible].mean(axis=0)
    _, singular, basis = np.linalg.svd(centred, full_matrices=False)
    if singular[0] == 0 or singular[1] / singular[0] < 0.01:
        raise ValueError("Spread the fitting landmarks across the frame, away from a single line.")
    planar = singular[2] / singular[0] < 1e-4
    refine = bool(request["refine_focal"])
    if refine and (len(eligible) < 6 or singular[2] / singular[0] < .03):
        raise ValueError("Refining focal length needs six or more landmarks spread across different depths.")

    def intrinsics(scale):
        return np.array([[fx * scale, 0, cx], [0, fy * scale, cy], [0, 0, 1.]])

    def project(parameters, indices):
        scale = np.exp(parameters[6]) if refine else 1.
        xy, _ = cv.projectPoints(xyz[indices], parameters[:3], parameters[3:6], intrinsics(scale), None)
        return xy[:, 0]

    candidates = []
    tolerance = max(threshold * 2, width * .012) if refine else threshold
    if planar:
        active = eligible
        if len(eligible) > 4:
            _, mask = cv.findHomography(centred @ basis[:2].T, pixels[eligible], cv.RANSAC, threshold)
            if mask is not None:
                active = eligible[mask.ravel().astype(bool)]
        if len(active) >= 4:
            _, rotations, translations, _ = cv.solvePnPGeneric(
                xyz[active], pixels[active], intrinsics(1.), None, flags=cv.SOLVEPNP_IPPE,
            )
            for rotation, translation in zip(rotations, translations):
                candidates.append((np.r_[rotation.ravel(), translation.ravel()], active))
    else:
        for scale in ((.55, .75, 1., 1.25, 1.65, 2.) if refine else (1.,)):
            ok, rotation, translation, inliers = cv.solvePnPRansac(
                xyz[eligible], pixels[eligible], intrinsics(scale), None, flags=cv.SOLVEPNP_SQPNP,
                iterationsCount=400, reprojectionError=tolerance, confidence=.999,
            )
            if ok and inliers is not None:
                parameters = np.r_[rotation.ravel(), translation.ravel()]
                if refine:
                    parameters = np.r_[parameters, np.log(scale)]
                candidates.append((parameters, eligible[inliers.ravel()]))

    minimum = 6 if refine else 4
    solutions = []
    for start, active in candidates:
        if len(active) < minimum or not np.isfinite(start).all():
            continue
        if refine:
            spread = np.linalg.svd(xyz[active] - xyz[active].mean(axis=0), compute_uv=False)
            if spread[0] == 0 or spread[-1] / spread[0] < .03:
                continue

        def estimate(parameters, selected):
            bounds = ([-np.inf] * 6 + [np.log(.35)], [np.inf] * 6 + [np.log(3.)]) if refine else (-np.inf, np.inf)
            result = least_squares(lambda p: (project(p, selected) - pixels[selected]).ravel(), parameters,
                                   bounds=bounds, loss="soft_l1", f_scale=threshold, max_nfev=120)
            return result.x

        model = estimate(start, active)
        errors = np.linalg.norm(project(model, eligible) - pixels[eligible], axis=1)
        active = eligible[errors <= threshold]
        if len(active) < minimum:
            continue
        model = estimate(model, active)
        errors = np.linalg.norm(project(model, np.arange(len(points))) - pixels, axis=1)
        active = eligible[errors[eligible] <= threshold]
        rotation = cv.Rodrigues(model[:3])[0]
        depth = (xyz @ rotation[2] + model[5])[active]
        if len(active) >= minimum and np.all(depth > 0):
            solutions.append((len(active), float(np.mean(errors[active] ** 2)), model, errors, active))
    if not solutions:
        raise ValueError("The landmarks do not agree within the pixel tolerance. Check mismatched points and scene geometry.")
    _, mse, model, errors, active = min(solutions, key=lambda item: (-item[0], item[1]))
    rotation = cv.Rodrigues(model[:3])[0]
    scale = float(np.exp(model[6])) if refine else 1.
    return {
        "rotation": (rotation.T @ np.diag([1, -1, -1])).tolist(),
        "position": (-rotation.T @ model[3:6]).tolist(),
        "focal_px": fx * scale, "principal_px": [cx, cy],
        "inlier_rms_px": mse ** .5, "inlier_count": len(active),
        "holdout_rms_px": float(np.mean(errors[~fit] ** 2) ** .5) if np.any(~fit) else None,
        "planar": bool(planar),
        "landmarks": [
            {"name": point["name"], "error_px": float(err), "holdout": point["holdout"], "inlier": i in active}
            for i, (point, err) in enumerate(zip(points, errors))
        ],
    }


if __name__ == "__main__":
    try:
        print(json.dumps(solve(json.load(sys.stdin)), allow_nan=False))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
