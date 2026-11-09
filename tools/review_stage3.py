"""Independent synthetic calibration and speed-assumption checks (not field accuracy)."""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.calibrate import calibrate_camera
from backend.camera_geometry import CameraModel, load_camera
from backend.shot_physics import estimate_shot


def checkerboard_video(path, matrix=None):
    matrix = np.array([[1000., 0, 640], [0, 1050, 360], [0, 0, 1]]) if matrix is None else np.asarray(matrix, dtype=float)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError('Cannot write synthetic video')
    try:
        for index in range(24):
            phase = index * 2 * math.pi / 24
            rvec = np.array([.35 * math.sin(phase), .45 * math.cos(phase), .12 * math.sin(phase * 2)])
            tvec = np.array([-.12 + .025 * math.sin(phase), -.08 + .025 * math.cos(phase), .60 + .08 * math.sin(phase * 2)])
            frame = np.full((720, 1280, 3), 175, np.uint8)
            for y in range(7):
                for x in range(10):
                    square = np.array([[x, y, 0], [x + 1, y, 0], [x + 1, y + 1, 0], [x, y + 1, 0]], dtype=float) * .025
                    pixels, _ = cv2.projectPoints(square, rvec, tvec, matrix, np.zeros(5))
                    shade = 245 if (x + y) % 2 else 10
                    cv2.fillConvexPoly(frame, np.round(pixels.reshape(-1, 2)).astype(np.int32), (shade,) * 3, cv2.LINE_AA)
            writer.write(cv2.GaussianBlur(frame, (3, 3), .5))
    finally:
        writer.release()
    return matrix


def observations(camera, size_multiplier=1):
    """Forward tangent bounds, independently from the inverse implementation."""
    rows = []
    for frame in range(28):
        time = frame / 60
        x, y, z = np.array([0, 0, 8]) + np.array([2, -5, 20]) * time + .5 * camera.gravity * time ** 2
        bounds = []
        for axis, focal, principal in [(x, camera.matrix[0, 0], camera.matrix[0, 2]), (y, camera.matrix[1, 1], camera.matrix[1, 2])]:
            tangent = .11 * math.sqrt(axis * axis + z * z - .11 ** 2)
            bounds.append([(axis * z - tangent) / (z * z - .11 ** 2) * focal + principal,
                           (axis * z + tangent) / (z * z - .11 ** 2) * focal + principal])
        center = [(bounds[0][0] + bounds[0][1]) / 2, (bounds[1][0] + bounds[1][1]) / 2]
        half = [(bounds[0][1] - bounds[0][0]) * size_multiplier / 2,
                (bounds[1][1] - bounds[1][0]) * size_multiplier / 2]
        rows.append({'frame_number': frame, 'time_sec': time, 'source': 'yolo', 'observed': True,
                     'confidence': .9, 'ball_bbox': [center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1]],
                     'raw_center': center, 'x': center[0], 'y': center[1], 'radius_px': min(half)})
    return rows


def speed_case(name, actual, used, multiplier=1, fps_fallback=False):
    rows = observations(actual, multiplier)
    result = estimate_shot(rows, {'fps': 60, 'width': 1920, 'height': 1080, 'fps_fallback': fps_fallback}, used, {'status': 'FIXED_CANDIDATE'})
    first = result['flight_segment']['start_frame'] / 60
    last = result['flight_segment']['end_frame'] / 60
    known = float(np.mean(np.linalg.norm(np.array([2, -5, 20]) + np.linspace(first, last, 31)[:, None] * actual.gravity, axis=1)) * 3.6)
    return {'case': name, 'known_speed_kmh': round(known, 3), 'status': result['measurement_status'],
            'estimated_speed_kmh': result['speed'], 'reason': result['failure_reason'],
            'relative_error': None if result['speed'] is None else round(abs(result['speed'] / known - 1), 4),
            'median_projection_error_px': result['physics_diagnostics'].get('median_reprojection_error_px'),
            'warnings': result['physics_diagnostics']['warnings']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = checkerboard_video(args.output / 'synthetic_checkerboard.mp4')
    path = args.output / 'synthetic_calibration.json'
    assert calibrate_camera(str(args.output / 'synthetic_checkerboard.mp4'), str(path)), 'Synthetic calibration failed'
    loaded = load_camera(1280, 720, calibration_path=path)
    assert loaded.calibrated
    focal_error = np.abs(np.diag(loaded.matrix)[:2] / np.diag(expected)[:2] - 1)
    principal_error = np.abs(loaded.matrix[:2, 2] - expected[:2, 2])
    assert max(focal_error) < .02 and max(principal_error) < 15, (focal_error, principal_error)
    actual = CameraModel(1920, 1080, CameraModel.assumed(1920, 1080, 90).matrix, source='calibration_json')
    cases = [speed_case('known_intrinsics_ideal_bbox', actual, actual),
             speed_case('actual_fov90_assumed_fov60', actual, CameraModel.assumed(1920, 1080)),
             speed_case('known_intrinsics_bbox20percent_too_large', actual, actual, 1.2),
             speed_case('unknown_fps_metadata', actual, actual, fps_fallback=True)]
    report = {'synthetic_calibration': {'status': 'PASS', 'known_matrix': expected.tolist(),
              'recovered_matrix': loaded.matrix.tolist(), 'focal_relative_errors': focal_error.tolist(),
              'principal_errors_px': principal_error.tolist(),
              'rms_error_px': json.loads(path.read_text(encoding='utf-8'))['rms_error']},
              'speed_assumption_cases': cases, 'field_accuracy_verified': False}
    (args.output / 'synthetic_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
