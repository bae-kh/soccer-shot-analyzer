"""Camera intrinsics and approximate metric reconstruction from ball size."""
import json
import math
from pathlib import Path

import cv2
import numpy as np


class CameraModel:
    def __init__(self, width, height, matrix, distortion=None, source='assumed_fov', warnings=None):
        self.width, self.height = int(width), int(height)
        self.matrix = np.asarray(matrix, dtype=float)
        self.distortion = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float).reshape(-1)
        self.source = source
        self.warnings = list(warnings or [])
        self.calibrated = source == 'calibration_json'
        self.calibration_path = None
        self.gravity = np.array([0.0, 9.81, 0.0])
        self.gravity_source = 'assumed_camera_down'

    @classmethod
    def assumed(cls, width, height, horizontal_fov_deg=60.0, warnings=None):
        focal = width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
        return cls(width, height, [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                   warnings=[f'ASSUMED_FOV_{horizontal_fov_deg:g}_DEG'] + list(warnings or []))

    def rays(self, pixels):
        points = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(points, self.matrix, self.distortion).reshape(-1, 2)
        rays = np.column_stack((normalized, np.ones(len(normalized))))
        return rays / np.linalg.norm(rays, axis=1)[:, None]

    def sphere_position(self, bbox, radius_m=0.11):
        """Infer sphere depth from silhouette bounds; boxes approximate silhouettes."""
        x1, y1, x2, y2 = map(float, bbox)
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        normalized = cv2.undistortPoints(np.array([[x1, v], [x2, v], [u, y1], [u, y2]], dtype=float).reshape(-1, 1, 2),
                                        self.matrix, self.distortion).reshape(-1, 2)
        mid_x, half_x = (normalized[0, 0] + normalized[1, 0]) / 2, (normalized[1, 0] - normalized[0, 0]) / 2
        mid_y, half_y = (normalized[2, 1] + normalized[3, 1]) / 2, (normalized[3, 1] - normalized[2, 1]) / 2
        depths = []
        for mid, half in [(mid_x, half_x), (mid_y, half_y)]:
            if half <= 1e-6:
                raise ValueError('Invalid ball size')
            # Tangent bounds give s²q² - (s²+m²+1)q + m² = 0, q=Z²/R².
            a = half * half
            b = a + mid * mid + 1
            q = (b + math.sqrt(max(0.0, b * b - 4 * a * mid * mid))) / (2 * a)
            depths.append(radius_m * math.sqrt(q))
        depth = max(depths)  # Blur enlarges an axis; favor its less enlarged counterpart.
        factor = (depth * depth - radius_m * radius_m) / depth
        position = np.array([mid_x * factor, mid_y * factor, depth])
        angle = math.asin(radius_m / np.linalg.norm(position))
        return position, angle

    def project(self, positions):
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        points, _ = cv2.projectPoints(positions, np.zeros(3), np.zeros(3), self.matrix, self.distortion)
        return points.reshape(-1, 2)

    def describe(self):
        return {'source': self.source, 'intrinsics_calibrated': self.calibrated,
                'camera_matrix': self.matrix.tolist(), 'dist_coeffs': self.distortion.tolist(),
                'resolution': [self.width, self.height], 'warnings': self.warnings,
                'coordinate_system_3d': 'camera: X right, Y down, Z forward; metres',
                'calibration_path': self.calibration_path,
                'gravity_camera_mps2': self.gravity.tolist(), 'gravity_source': self.gravity_source,
                'assumptions': ['fixed_camera', 'same_lens_zoom_and_uncropped_resize', 'ball_radius_0.11_m']}


def load_camera(width, height, calibration_path=None, backend_dir=None):
    backend = Path(backend_dir) if backend_dir is not None else Path(__file__).resolve().parent
    candidates = [Path(calibration_path)] if calibration_path is not None else [backend / 'calibration.json', backend.parent / 'calibration.json']
    selected = next((path for path in candidates if path.is_file()), None)
    if selected is None:
        return CameraModel.assumed(width, height, warnings=['CALIBRATION_MISSING'])
    try:
        data = json.loads(selected.read_text(encoding='utf-8'))
        matrix = np.asarray(data['camera_matrix'], dtype=float)
        distortion = np.asarray(data['dist_coeffs'], dtype=float).reshape(-1)
        resolution = data.get('resolution') or [data.get('img_width'), data.get('img_height')]
        original_w, original_h = map(float, resolution)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError('Invalid camera matrix')
        if not np.allclose(matrix[2], [0, 0, 1]) or distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
            raise ValueError('Invalid distortion or matrix bottom row')
        if not np.isfinite([original_w, original_h]).all() or min(original_w, original_h) <= 0:
            raise ValueError('Calibration resolution is required')
        rms = data.get('rms_error')
        if rms is not None and (not np.isfinite(rms) or rms < 0 or rms > 2.0):
            raise ValueError('Calibration RMS exceeds 2 pixels or is invalid')
        matrix = matrix.copy()
        matrix[0] *= width / original_w
        matrix[1] *= height / original_h
        camera = CameraModel(width, height, matrix, distortion, 'calibration_json')
        if 'gravity_camera_mps2' in data:
            gravity = np.asarray(data['gravity_camera_mps2'], dtype=float)
            if gravity.shape != (3,) or not np.isfinite(gravity).all() or abs(np.linalg.norm(gravity) - 9.81) > .1:
                raise ValueError('Invalid camera gravity vector')
            camera.gravity = gravity
            camera.gravity_source = 'provided_camera_orientation'
        camera.calibration_path = str(selected.resolve())
        return camera
    except (ValueError, TypeError, KeyError, OSError) as exc:
        camera = CameraModel.assumed(width, height, warnings=['CALIBRATION_INVALID', str(exc)])
        camera.calibration_path = str(selected.resolve())
        return camera


def _background_features(gray):
    h, w = gray.shape
    mask = np.zeros_like(gray)
    # Exclude common player/grass and static playback overlay areas. Remaining
    # foreground matches must still agree with spatially distributed background.
    mask[int(h * .08):int(h * .7), int(w * .05):int(w * .95)] = 255
    points, descriptors = cv2.SIFT_create(nfeatures=1600, contrastThreshold=.025).detectAndCompute(gray, mask)
    return points, descriptors


def _background_pair(left, right, shape):
    """Mutual descriptor matches and broad support, rather than one-way flow."""
    kl, dl = left
    kr, dr = right
    result = {'usable': False, 'reason': 'INSUFFICIENT_BACKGROUND_FEATURES'}
    if dl is None or dr is None or min(len(dl), len(dr)) < 20:
        return result
    matcher = cv2.BFMatcher()

    def matches(a, b):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2) if len(pair) == 2
                for m, n in [pair] if m.distance < .7 * n.distance}

    forward, backward = matches(dl, dr), matches(dr, dl)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    result.update(match_count=len(pairs), reason='INSUFFICIENT_MUTUAL_MATCHES')
    if len(pairs) < 20:
        return result
    old = np.float32([kl[i].pt for i, j in pairs])
    new = np.float32([kr[j].pt for i, j in pairs])
    matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    result['reason'] = 'BACKGROUND_TRANSFORM_UNVERIFIED'
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return result
    selected = inliers.reshape(-1).astype(bool)
    points = old[selected]
    if len(points) < 20:
        return result
    h, w = shape
    cells = {(min(3, int(x / w * 4)), min(2, int(y / h * 3))) for x, y in points}
    residual = np.linalg.norm(cv2.transform(old[:, None, :], matrix).reshape(-1, 2) - new, axis=1)
    span_x, span_y = float(np.ptp(points[:, 0]) / w), float(np.ptp(points[:, 1]) / h)
    error = float(np.median(residual[selected]))
    result.update(inlier_count=len(points), inlier_fraction=float(selected.mean()),
                  occupied_cells=len(cells), span_x=span_x, span_y=span_y, median_error_px=error)
    if selected.mean() < .6 or len(cells) < 4 or span_x < .35 or span_y < .2 or error > 1.:
        result['reason'] = 'INSUFFICIENT_SPATIAL_BACKGROUND_SUPPORT'
        return result
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
    shift = np.median(np.linalg.norm(corners @ matrix.T - corners[:, :2], axis=1)) / math.hypot(w, h)
    result.update(usable=True, reason=None, transform=matrix.tolist(), shift_fraction=float(shift),
                  scale_change=abs(math.hypot(matrix[0, 0], matrix[0, 1]) - 1),
                  rotation_deg=abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))))
    return result


def assess_camera_motion(video_path, start_frame, end_frame):
    """Assess sampled background changes, including cumulative start-to-end drift."""
    cap = cv2.VideoCapture(str(video_path))
    numbers = sorted(set([int(start_frame), int((start_frame + end_frame) / 2), int(end_frame)]))
    frames = {}
    try:
        for frame_number in numbers:
            if frame_number < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if ok:
                scale = min(1.0, 960 / frame.shape[1])
                frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
                frames[frame_number] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()
    features = {n: _background_features(gray) for n, gray in frames.items()}
    pairs = list(zip(numbers, numbers[1:]))
    if len(numbers) > 2:
        pairs.append((numbers[0], numbers[-1]))
    metrics, rejected = [], []
    for a, b in pairs:
        if a not in frames or b not in frames or frames[a].shape != frames[b].shape:
            rejected.append({'frames': [a, b], 'reason': 'SAMPLE_FRAME_UNREADABLE_OR_RESOLUTION_CHANGED'})
            continue
        metric = {**_background_pair(features[a], features[b], frames[a].shape), 'frames': [a, b]}
        (metrics if metric['usable'] else rejected).append(metric)
    moving = any(m['shift_fraction'] > .003 or m['scale_change'] > .005 or m['rotation_deg'] > .25 for m in metrics)
    complete = bool(pairs) and len(frames) == len(numbers) and not rejected
    return {'status': 'MOVING' if moving else ('FIXED_CANDIDATE' if complete else 'UNVERIFIED'),
            'samples': metrics, 'rejected_pairs': rejected, 'sample_frames': numbers,
            'method': 'mutual_sift_spatial_background_affine',
            'thresholds': {'shift_fraction': .003, 'scale_change': .005, 'rotation_deg': .25},
            'note': 'Sampled background transform heuristic; stationary status is not ground truth or all-frame verification.'}
