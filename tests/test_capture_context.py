import json
from pathlib import Path
import tempfile
import unittest

from backend.capture_context import (ROOT, validate_options, options_for_video, resolve_timing,
                                     camera_for_options, list_profiles)
from backend.shot_physics import estimate_shot, select_flight
from backend.camera_geometry import CameraModel
from test_shot_physics import synthetic_shot


class CaptureContextTests(unittest.TestCase):
    def test_report_matches_full_sample_path_not_uploaded_filename(self):
        options, source = options_for_video(ROOT / 'soccer_videos/test_videos/video1.mp4')
        self.assertEqual(options['timing_mode'], 'slow_motion')
        self.assertEqual(source['reported_condition']['certainty'], 'user_confirmed')
        self.assertIsNone(resolve_timing(30, options)['effective_capture_fps'])
        options, source = options_for_video(ROOT / 'backend/uploads/video1.mp4')
        self.assertEqual(options, {})
        self.assertIsNone(source['reported_condition'])
        options, _ = options_for_video(ROOT / 'soccer_videos/real_test_videos/video1.mp4')
        self.assertEqual(options, {})  # Unreported does not imply normal speed.

    def test_invalid_values_and_profile_path_traversal_are_rejected(self):
        for options in ({'timing_mode': 'normal', 'slowdown_factor': 4},
                        {'timing_mode': 'slow_motion', 'capture_fps': float('nan')},
                        {'timing_mode': 'slow_motion', 'capture_fps': True},
                        {'timing_mode': 'slow_motion', 'slowdown_factor': .5},
                        {'same_camera_settings': 'true'}, {'intrinsics_profile_id': '../calibration'},
                        {'intrinsics_profile_id': 'x/y'}, {'calibration_path': '/arbitrary/path'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options(options)

    def test_slow_motion_factor_and_capture_frame_mapping(self):
        result = resolve_timing(30, {'timing_mode': 'slow_motion', 'slowdown_factor': 4})
        self.assertEqual(result['effective_capture_fps'], 120)
        options = {'timing_mode': 'slow_motion', 'capture_fps': 240}
        self.assertIsNone(resolve_timing(30, options)['effective_capture_fps'])
        options['one_frame_per_capture'] = True
        self.assertEqual(resolve_timing(30, options)['effective_capture_fps'], 240)
        options['slowdown_factor'] = 4
        with self.assertRaises(ValueError):
            resolve_timing(30, options)
        options['slowdown_factor'] = 8
        self.assertEqual(resolve_timing(30, options)['effective_capture_fps'], 240)

    def test_normal_timing_requires_valid_file_fps_and_explicit_confirmation(self):
        self.assertIsNone(resolve_timing(30, {})['effective_capture_fps'])
        self.assertIsNone(resolve_timing(30, {'timing_mode': 'normal'}, fps_fallback=True)['effective_capture_fps'])
        self.assertEqual(resolve_timing(59.94, {'timing_mode': 'normal'})['effective_capture_fps'], 59.94)

    def test_unknown_timing_does_not_emit_a_physical_speed(self):
        camera = CameraModel.assumed(1920, 1080)
        rows, metadata = synthetic_shot(camera)
        metadata['measurement_context'] = {'timing': resolve_timing(60, {}),
                                           'problems': ['CAPTURE_TIMING_UNVERIFIED', 'CAMERA_PROFILE_REQUIRED']}
        result = estimate_shot(rows, metadata, camera, {'status': 'FIXED_CANDIDATE'})
        self.assertEqual(result['failure_reason'], 'CAPTURE_TIMING_UNVERIFIED')
        self.assertIsNone(result['speed'])
        self.assertFalse(result['trajectory_3d'])
        self.assertTrue(result['flight_segment']['selected_frame_numbers'])  # Tracking diagnostic retained.

    def test_slow_motion_recomputes_window_timestamps_and_ballistic_fit(self):
        camera = CameraModel(1920, 1080, [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]], source='calibration_json')
        rows, metadata = synthetic_shot(camera, velocity=(8, -5, 8), fps=120, frames=55)
        reference = estimate_shot(rows, metadata, camera, {'status': 'FIXED_CANDIDATE'})
        playback = [{**row, 'time_sec': row['frame_number']/30} for row in rows]
        metadata = {**metadata, 'fps': 30, 'measurement_context': {
            'timing': resolve_timing(30, {'timing_mode': 'slow_motion', 'slowdown_factor': 4}), 'problems': []}}
        result = estimate_shot(playback, metadata, camera, {'status': 'FIXED_CANDIDATE'})
        self.assertEqual(result['measurement_status'], 'ESTIMATED', result['failure_reason'])
        self.assertEqual(result['speed'], reference['speed'])
        self.assertEqual(result['flight_segment']['selected_frame_numbers'], reference['flight_segment']['selected_frame_numbers'])
        self.assertGreater(result['flight_segment']['end_frame'], 30)  # Not a 0.45 second playback window.
        for point in result['trajectory_3d']:
            self.assertAlmostEqual(point['time_sec'], point['frame_number']/120)
        self.assertEqual(playback[20]['time_sec'], 20/30)  # File timestamps retained in tracking.
        self.assertLess(result['physics_diagnostics']['duration_sec'], .46)

    def test_physical_window_never_rounds_up_past_limit(self):
        camera = CameraModel(1920, 1080, [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]], source='calibration_json')
        for fps in (30, 59.94, 120):
            with self.subTest(fps=fps):
                rows, _ = synthetic_shot(camera, velocity=(8, -5, 8), fps=fps, frames=80)
                selected, segment = select_flight(rows, fps, camera.width, camera.height)
                self.assertTrue(selected)
                self.assertLessEqual((segment['end_frame']-segment['tracking_start_frame'])/fps, .45 + 1e-9)


class CameraProfileTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / 'output/ai_baseline/_tests'
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.directory = Path(self.temp.name)
        self.payload = {'camera_matrix': [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]],
                        'dist_coeffs': [0, 0, 0, 0, 0], 'resolution': [1920, 1080], 'rms_error': .3}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, data):
        (self.directory / (name + '.json')).write_text(json.dumps(data), encoding='utf-8')

    def test_lens_zoom_profiles_are_independent_and_resize_intrinsics(self):
        self.save('normal', self.payload)
        wide = {**self.payload, 'camera_matrix': [[700, 0, 960], [0, 700, 540], [0, 0, 1]]}
        self.save('wide', wide)
        options = {'same_camera_settings': True, 'uncropped': True}
        camera, status = camera_for_options(960, 540, {**options, 'intrinsics_profile_id': 'normal'}, self.directory)
        self.assertEqual(status['problems'], [])
        self.assertEqual(camera.matrix[0, 0], 700)
        camera, status = camera_for_options(960, 540, {**options, 'intrinsics_profile_id': 'wide'}, self.directory)
        self.assertEqual(camera.matrix[0, 0], 350)
        self.assertEqual(len(list_profiles(self.directory)), 2)

    def test_no_global_fallback_and_no_unconfirmed_or_cropped_matching(self):
        self.save('normal', self.payload)
        camera, status = camera_for_options(1920, 1080, {}, self.directory)
        self.assertFalse(camera.calibrated)
        self.assertEqual(status['problems'], ['CAMERA_PROFILE_REQUIRED'])
        _, status = camera_for_options(1920, 1080, {'intrinsics_profile_id': 'normal'}, self.directory)
        self.assertIn('CAMERA_SETTINGS_UNCONFIRMED', status['problems'])
        self.assertIn('UNCROPPED_IMAGE_UNCONFIRMED', status['problems'])
        _, status = camera_for_options(1440, 1080, {'intrinsics_profile_id': 'normal', 'same_camera_settings': True, 'uncropped': True}, self.directory)
        self.assertIn('CAMERA_PROFILE_ASPECT_MISMATCH', status['problems'])

    def test_missing_and_invalid_profiles_block_metric_estimate(self):
        _, status = camera_for_options(1920, 1080, {'intrinsics_profile_id': 'missing'}, self.directory)
        self.assertIn('CAMERA_PROFILE_NOT_FOUND', status['problems'])
        self.save('invalid', {**self.payload, 'rms_error': 4})
        _, status = camera_for_options(1920, 1080, {'intrinsics_profile_id': 'invalid'}, self.directory)
        self.assertIn('CALIBRATION_INVALID', status['problems'])
        self.assertEqual(list_profiles(self.directory), [])
