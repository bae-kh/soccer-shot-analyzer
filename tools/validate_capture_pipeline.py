"""Execute real calibration routes with known synthetic cameras and physical trajectories.

Detector accuracy and field speed accuracy are outside this synthetic verification.
All videos, profiles, uploads and reports stay inside the requested output directory.
"""
import argparse
import asyncio
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
from tools.review_stage3 import checkerboard_video
import capture_context
from camera_geometry import CameraModel
from shot_physics import estimate_shot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def observation_rows(matrix, capture_fps, file_fps, width=1280, height=720,
                     size_multiplier=1, velocity=(8, -5, 8)):
    """Forward analytic sphere silhouette, independent of sphere_position inversion."""
    rows = []
    radius = .11
    for frame in range(round(.45 * capture_fps) + 1):
        time = frame / capture_fps
        position = np.array([0., 0., 8.]) + np.array(velocity) * time + np.array([0., .5 * 9.81 * time*time, 0.])
        x, y, z = position
        denominator = z*z - radius*radius
        bounds = []
        for axis, focal, principal in ((x, matrix[0, 0], matrix[0, 2]), (y, matrix[1, 1], matrix[1, 2])):
            tangent = radius * math.sqrt(axis*axis + z*z - radius*radius)
            bounds.append([(axis*z - tangent) / denominator * focal + principal,
                           (axis*z + tangent) / denominator * focal + principal])
        center = [(b[0] + b[1])/2 for b in bounds]
        half = [(b[1]-b[0])*size_multiplier/2 for b in bounds]
        box = [center[0]-half[0], center[1]-half[1], center[0]+half[0], center[1]+half[1]]
        assert 1 < box[0] < box[2] < width-1 and 1 < box[1] < box[3] < height-1, box
        rows.append({'frame_number': frame, 'time_sec': frame/file_fps, 'source': 'yolo',
                     'confidence': .9, 'observed': True, 'ball_bbox': box,
                     'raw_center': center, 'x': center[0], 'y': center[1], 'radius_px': min(half)})
    return rows


def known_average_speed(segment, capture_fps, velocity):
    if segment['start_frame'] is None:
        return None
    times = np.linspace(segment['start_frame']/capture_fps, segment['end_frame']/capture_fps, 2001)
    vectors = np.array(velocity) + times[:, None]*np.array([0., 9.81, 0.])
    return float(np.mean(np.linalg.norm(vectors, axis=1))*3.6)


async def calibrate_route(service, video, label):
    from fastapi import UploadFile
    with video.open('rb') as stream:
        upload = UploadFile(filename=video.name, file=stream)
        response = await service.calibrate_stream(upload, profile_label=label)
        events = [json.loads(event.removeprefix('data: ').strip()) async for event in response.body_iterator]
    if any(event['type'] == 'error' for event in events):
        raise AssertionError(events)
    return next(event['data'] for event in events if event['type'] == 'result')


def run_case(output, name, known_matrix, options, capture_fps=60, file_fps=60,
             width=1280, height=720, expected='estimate', motion='FIXED_CANDIDATE',
             size_multiplier=1, velocity=(8, -5, 8)):
    camera, context = capture_context.build_context(output / (name + '.mp4'), width, height,
                                                   file_fps, overrides=options)
    rows = observation_rows(known_matrix, capture_fps, file_fps, width, height, size_multiplier, velocity)
    original_time = [row['time_sec'] for row in rows]
    metadata = {'width': width, 'height': height, 'fps': file_fps, 'measurement_context': context}
    result = estimate_shot(rows, metadata, camera, {'status': motion})
    assert [row['time_sec'] for row in rows] == original_time
    truth = known_average_speed(result['flight_segment'], capture_fps, velocity)
    error = abs(result['speed'] / truth - 1) if result['speed'] is not None and truth else None
    errors = []
    if expected == 'estimate':
        if result['measurement_status'] != 'ESTIMATED':
            errors.append('Expected speed estimate, got ' + str(result['failure_reason']))
        if error is None or error >= .03:
            errors.append('Known speed error exceeds 3% or no speed')
        if result['flight_segment']['end_frame'] is None or result['flight_segment']['end_frame']/capture_fps > .45 + 1e-8:
            errors.append('Physical window exceeds 0.45 seconds')
        if not all(abs(point['time_sec'] - point['frame_number']/capture_fps) < 1e-10 for point in result['trajectory_3d']):
            errors.append('Incorrect physical timestamps')
    elif expected != 'limitation':
        if result['speed'] is not None or result['measurement_status'] != 'UNAVAILABLE' or result['failure_reason'] != expected:
            errors.append('Incorrect unavailable result or reason')
    save(output / (name + '.json'), {'known_matrix': known_matrix.tolist(),
         'known_capture_fps': capture_fps, 'known_velocity_at_zero_mps': list(velocity),
         'input_observations': rows, 'video_metadata': metadata, 'result': result})
    return {'case': name, 'expected': expected, 'status': result['measurement_status'],
            'failure_reason': result['failure_reason'], 'known_interval_average_kmh': truth,
            'estimated_speed_kmh': result['speed'], 'relative_error': error,
            'selected_points': result['flight_segment']['point_count'],
            'effective_capture_fps': context['timing']['effective_capture_fps'],
            'validation_passed': not errors, 'validation_errors': errors,
            'tracking_timestamps_preserved': True}


def verification_preview(output, entries):
    """Two synthetic camera views and one known slow-motion trajectory; no raster AI."""
    tiles = []
    for entry in entries:
        cap = cv2.VideoCapture(str(output / entry['video']))
        ok, image = cap.read()
        cap.release()
        assert ok
        image = cv2.resize(image, (640, 360))
        canvas = np.full((425, 640, 3), 245, np.uint8)
        canvas[65:] = image
        cv2.putText(canvas, entry['label'], (16, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 30, 30), 2)
        cv2.putText(canvas, f"Known fx={entry['known_matrix'][0][0]:.0f} | recovered={entry['recovered_matrix'][0][0]:.2f}",
                    (16, 51), cv2.FONT_HERSHEY_SIMPLEX, .53, (30, 30, 30), 1)
        tiles.append(canvas)
    path = output / 'slow_motion_4x.json'
    case = json.loads(path.read_text(encoding='utf-8'))
    result = case['result']
    canvas = np.full((330, 1280, 3), 245, np.uint8)
    camera = CameraModel(1280, 720, result['camera_model']['camera_matrix'], result['camera_model']['dist_coeffs'], source='calibration_json')
    # Plot image-space observations and fitted projection in the original camera frame.
    points = np.array([row['raw_center'] for row in case['input_observations']])
    base = np.array([points[0, 0]-60, points[:, 1].min()-45])
    for row in case['input_observations']:
        pixel = np.array(row['raw_center'])-base + np.array([70, 100])
        cv2.circle(canvas, tuple(pixel.astype(int)), 3, (0, 150, 0), -1)
    fit = np.array([[p['X_m'], p['Y_m'], p['Z_m']] for p in result['fitted_trajectory']])
    projected = camera.project(fit)-base + np.array([70, 100])
    cv2.polylines(canvas, [projected.astype(np.int32)], False, (220, 130, 0), 2)
    cv2.putText(canvas, 'Synthetic verification only - not soccer video accuracy', (24, 34), cv2.FONT_HERSHEY_SIMPLEX, .72, (30, 30, 30), 2)
    cv2.putText(canvas, f"120 capture FPS / 30 playback FPS / factor 4 | estimate {result['speed']:.1f} km/h", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, .66, (30, 30, 30), 1)
    cv2.putText(canvas, 'Green: known projected observations | blue: fitted projection', (590, 150), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 30, 30), 1)
    preview = np.vstack((np.hstack(tiles), canvas))
    assert cv2.imwrite(str(output / 'verification_preview.jpg'), preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profiles, uploads = output/'profiles', output/'uploads'
    uploads.mkdir()
    (output/'runtime/yolo').mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(output/'runtime/yolo'), YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    service = importlib.import_module('main')
    protected = [ROOT/'docs/video_capture_conditions.json', ROOT/'backend/calibration.json', ROOT/'calibration.json']
    protected += sorted((ROOT/'backend/camera_profiles').glob('*.json'))
    hashes = {str(path): digest(path) for path in protected}
    sources = [ROOT/'backend'/name for name in ('analyze_shot.py', 'main.py', 'calibrate.py', 'camera_geometry.py', 'capture_context.py', 'shot_physics.py')]
    sources += [Path(__file__).resolve(), ROOT/'tools/review_stage3.py']
    metadata = {'synthetic_only': True, 'python': sys.executable,
                'source_hashes': {str(path): digest(path) for path in sources},
                'service_config_hashes_before': hashes}
    save(output/'run_metadata.json', metadata)
    cameras = []
    with patch.object(service, 'UPLOAD_DIR', str(uploads)), patch.object(service, 'PROFILE_DIR', profiles), patch.object(capture_context, 'PROFILE_DIR', profiles):
        for name, matrix in [('normal', [[1000., 0, 640], [0, 1050, 360], [0, 0, 1]]),
                             ('wide', [[500., 0, 640], [0, 525, 360], [0, 0, 1]])]:
            video = output / (name + '_checkerboard.mp4')
            known = checkerboard_video(video, matrix)
            route_result = asyncio.run(calibrate_route(service, video, 'SYNTHETIC_' + name))
            options = {'timing_mode': 'normal', 'intrinsics_profile_id': route_result['profile_id'],
                       'same_camera_settings': True, 'uncropped': True}
            camera, context = capture_context.build_context(video, 1280, 720, 60, overrides=options)
            assert context['ready_for_metric_estimate']
            focal_errors = np.abs(np.diag(camera.matrix)[:2]/np.diag(known)[:2]-1)
            principal_errors = np.abs(camera.matrix[:2, 2]-known[:2, 2])
            assert max(focal_errors) < .02 and max(principal_errors) < 15
            cameras.append({'label': 'SYNTHETIC_' + name, 'video': video.name, 'profile_id': route_result['profile_id'],
                            'known_matrix': known.tolist(), 'recovered_matrix': camera.matrix.tolist(),
                            'focal_relative_errors': focal_errors.tolist(), 'principal_errors_px': principal_errors.tolist()})
            save(output/'calibrations.json', cameras)
        assert len(service.camera_profiles()['profiles']) == 2
        normal, wide = (np.array(c['known_matrix']) for c in cameras)
        options = {'timing_mode': 'normal', 'intrinsics_profile_id': cameras[0]['profile_id'], 'same_camera_settings': True, 'uncropped': True}
        results = []
        specs = [
            ('normal_60fps', normal, options, {}),
            ('normal_30fps', normal, options, {'capture_fps': 30, 'file_fps': 30}),
            ('wide_60fps', wide, {**options, 'intrinsics_profile_id': cameras[1]['profile_id']}, {}),
            ('resized_half', np.diag([.5, .5, 1]) @ normal, options, {'width': 640, 'height': 360}),
            ('slow_motion_4x', normal, {**options, 'timing_mode': 'slow_motion', 'slowdown_factor': 4}, {'capture_fps': 120, 'file_fps': 30}),
            ('slow_motion_capture_fps', normal, {**options, 'timing_mode': 'slow_motion', 'capture_fps': 240, 'one_frame_per_capture': True}, {'capture_fps': 240, 'file_fps': 30}),
            ('approaching_camera', normal, options, {'velocity': (4, -2, -4)}),
            ('unknown_time', normal, {**options, 'timing_mode': 'unknown'}, {'expected': 'CAPTURE_TIMING_UNVERIFIED'}),
            ('capture_fps_without_mapping', normal, {**options, 'timing_mode': 'slow_motion', 'capture_fps': 240}, {'expected': 'CAPTURE_TIMING_UNVERIFIED'}),
            ('profile_required', normal, {'timing_mode': 'normal'}, {'expected': 'CAMERA_PROFILE_REQUIRED'}),
            ('profile_missing', normal, {**options, 'intrinsics_profile_id': 'missing'}, {'expected': 'CAMERA_PROFILE_NOT_FOUND'}),
            ('settings_unconfirmed', normal, {**options, 'same_camera_settings': False}, {'expected': 'CAMERA_SETTINGS_UNCONFIRMED'}),
            ('crop_unconfirmed', normal, {**options, 'uncropped': False}, {'expected': 'UNCROPPED_IMAGE_UNCONFIRMED'}),
            ('aspect_mismatch', normal, options, {'width': 1280, 'height': 960, 'expected': 'CAMERA_PROFILE_ASPECT_MISMATCH'}),
            ('camera_moving', normal, options, {'motion': 'MOVING', 'expected': 'CAMERA_MOVING'}),
            ('wrong_profile_falsely_confirmed', wide, options, {'expected': 'limitation'}),
            ('bbox_20percent_enlarged', normal, options, {'size_multiplier': 1.2, 'expected': 'limitation'}),
        ]
        for name, matrix, settings, extra in specs:
            results.append(run_case(output, name, matrix, settings, **extra))
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        verification_preview(output, cameras)
    assert {str(path): digest(path) for path in protected} == hashes
    assert sorted((ROOT/'backend/camera_profiles').glob('*.json')) == [p for p in protected if p.parent == ROOT/'backend/camera_profiles']
    report = {'synthetic_calibration_routes': 2, 'calculation_backend_mocked': False,
              'calibrations': cameras, 'known_speed_cases_passed': sum(r['expected'] == 'estimate' and r['validation_passed'] for r in results),
              'guard_cases_passed': sum(r['expected'] not in {'estimate', 'limitation'} and r['validation_passed'] for r in results),
              'validation_errors': [r for r in results if not r['validation_passed']],
              'limitation_demonstrations': sum(r['expected'] == 'limitation' for r in results),
              'cases': results, 'service_config_and_profiles_unchanged': True,
              'detector_ground_truth_accuracy_evaluated': False, 'field_speed_accuracy_evaluated': False}
    save(output/'validation.json', report)
    if report['validation_errors']:
        print('FAIL: inspect validation.json. Actual camera/video accuracy remains unverified.', flush=True)
        return 1
    print('PASS: synthetic capture pipeline; actual camera/video accuracy remains unverified.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
