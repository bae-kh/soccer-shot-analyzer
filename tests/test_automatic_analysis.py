import copy
import unittest

from backend.automatic_analysis import predict_goal, automatic_comment
from backend.capture_context import build_context, DEFAULT_AUTOMATIC_OPTIONS, ROOT, validate_options
from backend.goal_entry import assess_goal_entry
from backend.goal_measurement import assess_physical_goal, score_week9
from test_goal_measurement import fixture
from backend.shot_physics import estimate_shot
from test_goal_entry import point, GOAL, META
from test_shot_physics import synthetic_shot


class AutomaticContextTests(unittest.TestCase):
    def test_default_automatic_estimate_requires_no_profile_or_confirmation(self):
        camera, context = build_context(ROOT/'backend/uploads/shot.mp4', 1920, 1080, 60,
                                        overrides=DEFAULT_AUTOMATIC_OPTIONS)
        self.assertFalse(camera.calibrated)
        self.assertEqual(context['problems'], [])
        self.assertEqual(context['timing']['status'], 'ASSUMED_NORMAL')
        self.assertEqual(context['timing']['effective_capture_fps'], 60)
        self.assertIn('NO_ZOOM_ASSUMED', context['assumptions'])
        rows, meta = synthetic_shot(camera, velocity=(8, -5, 8))
        meta['measurement_context'] = context
        result = estimate_shot(rows, meta, camera, {'status': 'FIXED_CANDIDATE'})
        self.assertEqual(result['measurement_status'], 'ESTIMATED', result['failure_reason'])
        self.assertGreater(result['speed'], 0)
        self.assertEqual(result['speed_basis'], 'ASSUMPTION_BASED')
        self.assertEqual(result['flight_segment']['time_basis'], 'normal_speed_assumed_from_file_fps')

    def test_strict_mode_and_explicit_bad_profile_are_not_silently_bypassed(self):
        for options in [{'analysis_mode': 'calibrated', 'timing_mode': 'normal'},
                        {**DEFAULT_AUTOMATIC_OPTIONS, 'intrinsics_profile_id': 'not_present'}]:
            camera, context = build_context(ROOT/'backend/uploads/x.mp4', 1920, 1080, 60, overrides=options)
            self.assertTrue(context['problems'])
            rows, meta = synthetic_shot(camera)
            meta['measurement_context'] = context
            self.assertIsNone(estimate_shot(rows, meta, camera, {'status': 'FIXED_CANDIDATE'})['speed'])

    def test_known_sample_conditions_remain_in_provenance_despite_requested_default(self):
        _, context = build_context(ROOT/'soccer_videos/test_videos/video1.mp4', 1920, 1080, 30,
                                   overrides=DEFAULT_AUTOMATIC_OPTIONS)
        self.assertEqual(context['timing']['effective_capture_fps'], 30)
        self.assertEqual(context['source']['reported_condition']['condition'], 'slow_motion')
        self.assertIn('SAMPLE_CAPTURE_REPORT_DIFFERS_FROM_AUTOMATIC_DEFAULTS', context['assumptions'])

    def test_missing_file_fps_and_moving_camera_still_prevent_speed(self):
        camera, context = build_context(ROOT/'backend/uploads/x.mp4', 1920, 1080, 30, True,
                                        DEFAULT_AUTOMATIC_OPTIONS)
        self.assertIn('CAPTURE_TIMING_UNVERIFIED', context['problems'])
        rows, meta = synthetic_shot(camera)
        meta['measurement_context'] = context
        self.assertIsNone(estimate_shot(rows, meta, camera, {'status': 'FIXED_CANDIDATE'})['speed'])
        _, context = build_context(ROOT/'backend/uploads/x.mp4', 1920, 1080, 60, overrides=DEFAULT_AUTOMATIC_OPTIONS)
        meta['measurement_context'] = context
        self.assertEqual(estimate_shot(rows, meta, camera, {'status': 'MOVING'})['failure_reason'], 'CAMERA_MOVING')

    def test_invalid_analysis_mode_is_rejected(self):
        for mode in ['force', True, None]:
            with self.assertRaises(ValueError):
                validate_options({'analysis_mode': mode})


class AutomaticGoalTests(unittest.TestCase):
    def test_image_candidate_and_speed_remain_visible_without_goal_or_score(self):
        rows = [point(1, 80), point(2, 130), point(3, 160)]
        entry = assess_goal_entry(rows, GOAL, META, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'})
        self.assertEqual(entry['status'], 'CANDIDATE_OBSERVED')
        physical = {'status': 'UNDETERMINED', 'reason': 'GOAL_PLANE_REQUIRED', 'events': []}
        physics = {'speed': 60., 'measurement_status': 'ESTIMATED', 'speed_basis': 'ASSUMPTION_BASED'}
        result = predict_goal(physical)
        self.assertEqual(result['status'], 'UNDETERMINED')
        self.assertEqual(result['reason'], 'GOAL_PLANE_REQUIRED')
        self.assertEqual(result['evidence_frames'], [])
        score = score_week9(physical, physics)
        self.assertIsNone(score['value'])
        comment = automatic_comment(result, physics, score)
        self.assertIn('60.0 km/h', comment)
        self.assertNotIn('점', comment)

    def test_measured_inside_and_outside_provide_distinct_evidence(self):
        for x, expected in [(1.5, 'GOAL_LIKELY'), (-.5, 'OUTSIDE_LIKELY')]:
            rows, metadata, camera, plane = fixture(x=x)
            physical = assess_physical_goal(rows, metadata, camera, {'kick_detected': True},
                                            {'status': 'FIXED_CANDIDATE'}, plane)
            original = copy.deepcopy(physical)
            result = predict_goal(physical)
            self.assertEqual(result['status'], expected)
            self.assertEqual(result['basis'], 'measured_goal_plane')
            self.assertEqual(result['evidence_frames'], [0, 4])
            self.assertFalse(result['physical_goal_confirmed'])
            self.assertEqual(physical, original)

    def test_missing_crossing_cannot_promote_a_status(self):
        for status in ('ESTIMATED_GOAL', 'ESTIMATED_OUTSIDE'):
            prediction = predict_goal({'status': status, 'events': []})
            self.assertEqual(prediction['status'], 'UNDETERMINED')
            self.assertEqual(prediction['reason'], 'GOAL_CROSSING_EVIDENCE_REQUIRED')


if __name__ == '__main__':
    unittest.main()
