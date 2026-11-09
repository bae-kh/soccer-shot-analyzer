import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import cv2

from backend.camera_geometry import CameraModel, load_camera, assess_camera_motion
from backend.shot_physics import estimate_shot, select_flight
from tools.run_ai_baseline import classify_result

ROOT = Path(__file__).resolve().parents[1]


def sphere_bbox(position, camera, radius=.11):
    """Independent analytic tangent bounds of a sphere under a pinhole camera."""
    x, y, z = position
    denominator = z * z - radius * radius
    bounds = []
    for axis, focal, principal in [(x, camera.matrix[0, 0], camera.matrix[0, 2]),
                                    (y, camera.matrix[1, 1], camera.matrix[1, 2])]:
        tangent = radius * math.sqrt(axis * axis + z * z - radius * radius)
        bounds.append([(axis * z - tangent) / denominator * focal + principal,
                       (axis * z + tangent) / denominator * focal + principal])
    return [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]


def synthetic_shot(camera, velocity=(2, -5, 20), position=(0, 0, 8), fps=60, frames=28):
    rows = []
    for frame in range(frames):
        t = frame / fps
        point = np.array(position) + np.array(velocity) * t + .5 * camera.gravity * t * t
        bbox = sphere_bbox(point, camera)
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        rows.append({'frame_number': frame, 'time_sec': t, 'source': 'yolo', 'observed': True,
                     'confidence': .9, 'ball_bbox': bbox, 'raw_center': center,
                     'x': center[0], 'y': center[1], 'radius_px': min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2})
    return rows, {'fps': fps, 'width': camera.width, 'height': camera.height}


class CameraTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / 'output/ai_baseline/_tests'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.directory = Path(self.temp.name)
        self.backend = self.directory / 'backend'
        self.backend.mkdir()
        self.payload = {'camera_matrix': [[1400, 0, 800], [0, 1500, 500], [0, 0, 1]],
                        'dist_coeffs': [0, 0, 0, 0, 0], 'img_width': 1920, 'img_height': 1080, 'rms_error': .3}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, payload, path=None):
        path = path or self.backend / 'calibration.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        return path

    def test_writer_schema_resolution_and_full_matrix_scale(self):
        self.save(self.payload)
        camera = load_camera(960, 720, backend_dir=self.backend)
        self.assertTrue(camera.calibrated)
        np.testing.assert_allclose(camera.matrix, [[700, 0, 400], [0, 1000, 500 * 2 / 3], [0, 0, 1]])

    def test_backend_path_has_priority_and_root_legacy_is_supported(self):
        root = self.directory / 'calibration.json'
        self.save(self.payload, root)
        camera = load_camera(1920, 1080, backend_dir=self.backend)
        self.assertEqual(camera.calibration_path, str(root.resolve()))
        other = {**self.payload, 'resolution': [1920, 1080]}
        other['camera_matrix'] = [[1600, 0, 900], [0, 1600, 520], [0, 0, 1]]
        self.save(other)
        self.assertEqual(load_camera(1920, 1080, backend_dir=self.backend).matrix[0, 0], 1600)

    def test_invalid_calibration_does_not_look_calibrated(self):
        self.save({**self.payload, 'img_width': None})
        camera = load_camera(1920, 1080, backend_dir=self.backend)
        self.assertFalse(camera.calibrated)
        self.assertIn('CALIBRATION_INVALID', camera.warnings)

    def test_zero_rms_is_valid_and_large_rms_is_rejected(self):
        self.save({**self.payload, 'rms_error': 0})
        self.assertTrue(load_camera(1920, 1080, backend_dir=self.backend).calibrated)
        self.save({**self.payload, 'rms_error': 3})
        self.assertFalse(load_camera(1920, 1080, backend_dir=self.backend).calibrated)

    def test_distorted_projection_returns_same_ray(self):
        camera = CameraModel(1920, 1080, self.payload['camera_matrix'], [.1, -.04, .001, -.002, 0])
        point = np.array([[1.2, -.7, 6]])
        ray = camera.rays(camera.project(point))[0]
        np.testing.assert_allclose(ray, point[0] / np.linalg.norm(point), atol=1e-7)

    def test_off_axis_sphere_recovers_xyz(self):
        camera = CameraModel(1920, 1080, self.payload['camera_matrix'])
        point = np.array([2, 1.5, 8])
        reconstructed, _ = camera.sphere_position(sphere_bbox(point, camera))
        np.testing.assert_allclose(reconstructed, point, atol=1e-8)

    def test_background_motion_check_static_and_translated_video(self):
        texture = np.random.RandomState(42).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        for moving in (False, True):
            path = self.directory / ('moving.mp4' if moving else 'static.mp4')
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            for frame in range(9):
                transform = np.array([[1, 0, frame * .8 if moving else 0], [0, 1, 0]], dtype=float)
                writer.write(cv2.warpAffine(texture, transform, (320, 240)))
            writer.release()
            result = assess_camera_motion(path, 0, 8)
            self.assertEqual(result['status'], 'MOVING' if moving else 'FIXED_CANDIDATE', result)

    def motion_video(self, frames, name='motion_cases.mp4'):
        path = self.directory / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
        self.assertTrue(writer.isOpened())
        for frame in frames:
            writer.write(frame)
        writer.release()
        return path

    def test_moving_foreground_does_not_make_distributed_static_background_move(self):
        background = np.random.RandomState(8).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        frames = []
        for x in (100, 130, 160):
            image = background.copy()
            cv2.rectangle(image, (x, 50), (x + 35, 210), (250, 30, 30), -1)
            frames.append(image)
        result = assess_camera_motion(self.motion_video(frames), 0, 2)
        self.assertEqual(result['status'], 'FIXED_CANDIDATE', result)
        self.assertEqual(len(result['samples']), 3)

    def test_cumulative_drift_is_checked_even_when_each_half_is_under_threshold(self):
        background = np.random.RandomState(8).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        frames = [cv2.warpAffine(background, np.float32([[1, 0, dx], [0, 1, 0]]), (320, 240)) for dx in (0, .8, 1.6)]
        result = assess_camera_motion(self.motion_video(frames), 0, 2)
        self.assertEqual(result['status'], 'MOVING', result)
        cumulative = next(m for m in result['samples'] if m['frames'] == [0, 2])
        self.assertGreater(cumulative['shift_fraction'], .003)

    def test_zoom_with_distributed_background_is_moving(self):
        background = np.random.RandomState(8).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        frames = [cv2.warpAffine(background, cv2.getRotationMatrix2D((160, 120), 0, scale), (320, 240))
                  for scale in (1, 1.02, 1.04)]
        result = assess_camera_motion(self.motion_video(frames), 0, 2)
        self.assertEqual(result['status'], 'MOVING', result)

    def test_foreground_only_texture_is_unverified_without_broad_support(self):
        patch = np.random.RandomState(8).randint(0, 256, (60, 60, 3), dtype=np.uint8)
        frames = []
        for x in (50, 60, 70):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[50:110, x:x+60] = patch
            frames.append(frame)
        result = assess_camera_motion(self.motion_video(frames), 0, 2)
        self.assertEqual(result['status'], 'UNVERIFIED', result)

    def test_unreadable_endpoint_and_single_frame_are_not_fixed(self):
        texture = np.random.RandomState(8).randint(0, 256, (240, 320, 3), dtype=np.uint8)
        path = self.motion_video([texture] * 3)
        for start, end in [(0, 6), (1, 1)]:
            self.assertEqual(assess_camera_motion(path, start, end)['status'], 'UNVERIFIED')

    def test_calibration_gravity_is_explicit_and_invalid_vector_is_rejected(self):
        self.save({**self.payload, 'gravity_camera_mps2': [0, 0, 9.81]})
        camera = load_camera(1920, 1080, backend_dir=self.backend)
        np.testing.assert_allclose(camera.gravity, [0, 0, 9.81])
        self.assertEqual(camera.gravity_source, 'provided_camera_orientation')
        self.save({**self.payload, 'gravity_camera_mps2': [0, 0, 0]})
        self.assertIn('CALIBRATION_INVALID', load_camera(1920, 1080, backend_dir=self.backend).warnings)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraModel(1920, 1080, [[1400, 0, 880], [0, 1300, 490], [0, 0, 1]], source='calibration_json')

    def test_known_ballistic_speed_and_no_future_points(self):
        rows, metadata = synthetic_shot(self.camera)
        result = estimate_shot(rows, metadata, self.camera, {'status': 'FIXED_CANDIDATE'})
        self.assertEqual(result['measurement_status'], 'ESTIMATED', result['physics_diagnostics'])
        first, last = result['fitted_trajectory'][0]['time_sec'], result['fitted_trajectory'][-1]['time_sec']
        expected = np.mean(np.linalg.norm(np.array([2, -5, 20]) + np.linspace(first, last, 31)[:, None] * self.camera.gravity, axis=1)) * 3.6
        self.assertAlmostEqual(result['speed'], expected, delta=.5)
        self.assertLess(result['physics_diagnostics']['median_reprojection_error_px'], 1)
        self.assertLessEqual(last, result['flight_segment']['end_frame'] / metadata['fps'])

    def test_high_speed_is_not_clipped_to_150(self):
        rows, metadata = synthetic_shot(self.camera, velocity=(2, -4, 50))
        result = estimate_shot(rows, metadata, self.camera)
        self.assertEqual(result['measurement_status'], 'ESTIMATED', result['failure_reason'])
        self.assertGreater(result['speed'], 175)

    def test_motion_toward_camera_is_allowed(self):
        rows, metadata = synthetic_shot(self.camera, position=(0, 0, 15), velocity=(3, -5, -15))
        result = estimate_shot(rows, metadata, self.camera)
        self.assertEqual(result['measurement_status'], 'ESTIMATED', result['failure_reason'])
        self.assertLess(result['physics_diagnostics']['velocity_at_interval_start_mps'][2], 0)

    def test_predictions_and_csrt_cannot_supply_metric_ball_sizes(self):
        rows, metadata = synthetic_shot(self.camera)
        for row in rows:
            row['source'] = 'csrt' if row['frame_number'] % 2 else 'kalman_prediction'
            row['confidence'] = None
        result = estimate_shot(rows, metadata, self.camera)
        self.assertIsNone(result['speed'])
        self.assertEqual(result['failure_reason'], 'INSUFFICIENT_FLIGHT_OBSERVATIONS')

    def test_moving_camera_returns_null_speed(self):
        rows, metadata = synthetic_shot(self.camera)
        result = estimate_shot(rows, metadata, self.camera, {'status': 'MOVING'})
        self.assertIsNone(result['speed'])
        self.assertEqual(result['failure_reason'], 'CAMERA_MOVING')

    def test_direction_change_ends_candidate_flight(self):
        rows, metadata = synthetic_shot(self.camera)
        for i, row in enumerate(rows):
            x = 400 + 10 * i if i < 10 else 490 - 10 * (i - 9)
            row.update(raw_center=[x, 700], ball_bbox=[x - 10, 690, x + 10, 710])
        selected, segment = select_flight(rows, metadata['fps'], 1920, 1080)
        self.assertEqual(segment['reason'], 'POSSIBLE_CONTACT_OR_DIRECTION_CHANGE')
        self.assertLessEqual(selected[-1]['frame_number'], 9)

    def test_unavailable_is_preserved_and_not_success(self):
        self.assertEqual(classify_result({'measurement_status': 'UNAVAILABLE', 'speed': None,
                                         'failure_reason': 'CAMERA_MOVING'}, True), ('UNAVAILABLE', 'CAMERA_MOVING'))
        self.assertEqual(classify_result({'measurement_status': 'ESTIMATED', 'speed': 80}, True), ('ESTIMATED', None))
        self.assertEqual(classify_result({'measurement_status': 'ESTIMATED', 'speed': None}, True), ('FAILED', 'INVALID_SPEED'))

    def test_small_pre_kick_drift_does_not_become_flight(self):
        rows, metadata = synthetic_shot(self.camera)
        for i, row in enumerate(rows):
            x = 400+1.3*i
            row.update(raw_center=[x, 700], ball_bbox=[x-40, 660, x+40, 740])
        selected, segment = select_flight(rows, metadata['fps'], 1920, 1080)
        self.assertEqual(selected, [])
        self.assertFalse(segment['motion_confirmed'])

    def test_pre_kick_frames_removed_before_sustained_departure(self):
        rows, metadata = synthetic_shot(self.camera)
        for i, row in enumerate(rows):
            x = 400+(.8*i if i < 8 else 20*(i-7))
            row.update(raw_center=[x, 700], ball_bbox=[x-20, 680, x+20, 720])
        selected, segment = select_flight(rows, metadata['fps'], 1920, 1080)
        self.assertTrue(segment['motion_confirmed'])
        self.assertGreaterEqual(selected[0]['frame_number'], 9)
        self.assertTrue(all(r['frame_number'] >= 9 for r in selected))

    def test_single_jump_with_duplicate_frames_does_not_confirm_flight(self):
        rows, metadata = synthetic_shot(self.camera)
        for i, row in enumerate(rows):
            x = 400 if i < 8 else 500
            row.update(raw_center=[x, 700], ball_bbox=[x-20, 680, x+20, 720])
        selected, _ = select_flight(rows, metadata['fps'], 1920, 1080)
        self.assertEqual(selected, [])

    def test_size_rejection_does_not_change_tracked_observation(self):
        rows, metadata = synthetic_shot(self.camera)
        rows[8]['ball_size_quality'] = {'usable': False, 'reason': 'PARTIAL_FOREGROUND_OR_BOX_CONTAMINATION'}
        selected, segment = select_flight(rows, metadata['fps'], 1920, 1080)
        self.assertNotIn(8, [r['frame_number'] for r in selected])
        self.assertTrue(rows[8]['observed'])
        self.assertIn({'frame_number': 8, 'reason': 'PARTIAL_FOREGROUND_OR_BOX_CONTAMINATION'}, segment['excluded_observations'])

    def test_axis_aligned_depth_motion_can_confirm_flight(self):
        rows, metadata = synthetic_shot(self.camera, velocity=(0, 0, 20))
        result = estimate_shot(rows, metadata, self.camera)
        self.assertEqual(result['measurement_status'], 'ESTIMATED', result['failure_reason'])
        self.assertGreater(len(result['fitted_trajectory']), 5)

    def test_insufficient_size_evidence_cannot_create_speed(self):
        rows, metadata = synthetic_shot(self.camera)
        for row in rows:
            row['ball_size_quality'] = {'usable': False, 'reason': 'ELONGATED_BALL_BOX'}
        result = estimate_shot(rows, metadata, self.camera)
        self.assertIsNone(result['speed'])
        self.assertEqual(result['failure_reason'], 'INSUFFICIENT_FLIGHT_OBSERVATIONS')


if __name__ == '__main__':
    unittest.main()
