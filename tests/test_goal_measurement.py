"""Independent sphere projections test full crossing, depth and refusal cases."""
import copy
import unittest

import cv2
import numpy as np

from backend.camera_geometry import CameraModel
from backend.capture_context import validate_options
from backend.goal_measurement import assess_physical_goal, score_shot, score_week9, validate_goal_plane, validate_scoring_rules
from test_shot_physics import sphere_bbox


def fixture(depths=(-.5, -.1, 0, .1, .5), x=1.5, y=1., tilted=False):
    camera = CameraModel(1920, 1080, [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], source='calibration_json')
    rotation = cv2.Rodrigues(np.array([.05, .2, -.03]) if tilted else np.zeros(3))[0]
    translation = np.array([-1.5, -1., 6.])
    world = np.array([[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0]], dtype=float)
    config = {'width_m': 3., 'height_m': 2., 'corners_px': camera.project(world @ rotation.T + translation).tolist(),
              'reference_frame': 0, 'plane_reference': 'rear_edge_goal_line',
              'confirmed_by_user': True, 'pixel_tolerance_px': .2, 'ball_radius_m': .11}
    rows = []
    for frame, depth in enumerate(depths):
        position = rotation @ np.array([x[frame] if isinstance(x, (tuple, list)) else x, y, depth]) + translation
        box = [float(v) for v in sphere_bbox(position, camera)]
        rows.append({'frame_number': frame, 'ball_bbox': box, 'source': 'yolo', 'observed': True,
                     'confidence': .9, 'raw_center': [(box[0]+box[2])/2, (box[1]+box[3])/2]})
    meta = {'width': 1920, 'height': 1080, 'frame_count': len(rows),
            'measurement_context': {'calibration': {'problems': [], 'same_camera_settings': True, 'uncropped': True}}}
    return rows, meta, camera, config


RULES = {'id': 'synthetic_example_only', 'confirmed_by_user': True, 'accept_conditional_estimates': True,
         'speed_min_kmh': 20, 'speed_max_kmh': 100, 'speed_weight': .4, 'accuracy_weight': .6,
         'target_normalized': [.5, .5], 'accuracy_radius_m': 1., 'outside_score': 0}


class PhysicalGoalTests(unittest.TestCase):
    def assess(self, case=None, motion='FIXED_CANDIDATE', kick=True):
        rows, meta, camera, config = case or fixture()
        return assess_physical_goal(rows, meta, camera, {'kick_detected': kick}, {'status': motion}, config)

    def test_slow_full_sphere_crossing_is_conditional_not_confirmed(self):
        result = self.assess()
        self.assertEqual(result['status'], 'ESTIMATED_GOAL')
        self.assertFalse(result['accuracy_validated'])
        event = result['events'][0]
        self.assertEqual(event['continuity_frames'], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(event['crossing_normalized'], [.5, .5], atol=1e-8)

    def test_oblique_plane_and_two_frame_crossing(self):
        result = self.assess(fixture((-.8, .8), tilted=True))
        self.assertEqual(result['status'], 'ESTIMATED_GOAL')
        np.testing.assert_allclose(result['events'][0]['crossing_normalized'], [.5, .5], atol=1e-7)

    def test_image_overlap_in_front_and_center_only_crossing_are_held(self):
        for depths in [(-1, -.8, -.5), (-.5, -.1, 0, .05)]:
            self.assertEqual(self.assess(fixture(depths))['status'], 'UNDETERMINED')

    def test_reverse_crossing_is_not_a_goal(self):
        self.assertEqual(self.assess(fixture((.5, 0, -.5)))['status'], 'UNDETERMINED')

    def test_outside_post_and_above_crossbar(self):
        for x, y in [(-.5, 1.), (3.5, 1.), (1.5, -.5)]:
            self.assertEqual(self.assess(fixture((-.5, .5), x=x, y=y))['status'], 'ESTIMATED_OUTSIDE')

    def test_border_near_post_is_uncertain(self):
        self.assertEqual(self.assess(fixture((-.5, .5), x=.12))['status'], 'UNDETERMINED')

    def test_bending_shot_can_approach_from_outside_and_cross_inside(self):
        depths = (-.5, -.15, 0., .15, .5)
        result = self.assess(fixture(depths, x=[-.55, .65, 1.3, 1.5, 1.5]))
        self.assertEqual(result['status'], 'ESTIMATED_GOAL')

    def test_center_crossing_outside_does_not_override_post_overlap_in_crossing_interval(self):
        depths = (-.5, -.15, 0., .15, .5)
        result = self.assess(fixture(depths, x=[-.65, -.55, -.45, .65, .85]))
        self.assertEqual(result['status'], 'UNDETERMINED')
        self.assertEqual(result['reason'], 'GOAL_BOUNDARY_OR_DEPTH_UNCERTAIN')

    def test_missing_weak_prediction_csrt_or_rejected_breaks_chain(self):
        for change in [{'source': 'csrt'}, {'source': 'kalman_prediction', 'observed': False},
                       {'confidence': .1}, {'confidence': .1199999}, {'observation_validation': {'accepted': False}},
                       {'ball_size_quality': {'usable': False}}]:
            rows, meta, camera, config = fixture()
            rows[2].update(change)
            self.assertEqual(self.assess((rows, meta, camera, config))['status'], 'UNDETERMINED')

    def test_camera_profile_and_motion_gates(self):
        for motion in ['MOVING', 'UNVERIFIED']:
            self.assertEqual(self.assess(motion=motion)['status'], 'UNDETERMINED')
        rows, meta, camera, config = fixture()
        meta['measurement_context']['calibration']['same_camera_settings'] = False
        self.assertEqual(self.assess((rows, meta, camera, config))['reason'], 'GOAL_CALIBRATION_REQUIRED')

    def test_no_plane_and_no_kick_are_held(self):
        rows, meta, camera, _ = fixture()
        self.assertEqual(assess_physical_goal(rows, meta, camera, {}, {}, None)['reason'], 'GOAL_PLANE_REQUIRED')
        self.assertEqual(self.assess(kick=False)['reason'], 'KICK_NOT_DETECTED')

    def test_invalid_plane_frame_order_and_input_immutability(self):
        case = fixture()
        original = copy.deepcopy(case[:2])
        self.assess(case)
        self.assertEqual(case[:2], original)
        case[0][2]['frame_number'] = 1
        self.assertEqual(self.assess(case)['reason'], 'INVALID_OBSERVATION_SEQUENCE')
        rows, meta, camera, config = fixture()
        config['reference_frame'] = 99
        self.assertEqual(self.assess((rows, meta, camera, config))['reason'], 'INVALID_GOAL_PLANE_GEOMETRY')


class ScoringTests(unittest.TestCase):
    def goal(self):
        rows, meta, camera, config = fixture()
        return assess_physical_goal(rows, meta, camera, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'}, config)

    def test_known_arithmetic_and_speed_clamping(self):
        for speed, expected in [(60, 80), (10, 60), (120, 100)]:
            result = score_shot(self.goal(), {'speed': speed, 'measurement_status': 'ESTIMATED'}, RULES)
            self.assertEqual(result['status'], 'PROVISIONAL')
            self.assertEqual(result['value'], expected)
            self.assertFalse(result['candidate_used_as_goal'])

    def test_nonzero_target_distance_is_metric(self):
        rules = {**RULES, 'target_normalized': [.5, .25]}
        self.assertEqual(score_shot(self.goal(), {'speed': 60, 'measurement_status': 'ESTIMATED'}, rules)['value'], 50)

    def test_no_rules_no_speed_or_only_image_candidate_does_not_score(self):
        self.assertIsNone(score_shot(self.goal(), {'speed': 60}, None)['value'])
        self.assertIsNone(score_shot(self.goal(), {'speed': None, 'measurement_status': 'UNAVAILABLE'}, RULES)['value'])
        self.assertIsNone(score_shot({'status': 'UNDETERMINED'}, {'speed': 60}, RULES)['value'])

    def test_outside_score_does_not_require_speed(self):
        rows, meta, camera, config = fixture((-.5, .5), x=-.5)
        result = assess_physical_goal(rows, meta, camera, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'}, config)
        self.assertEqual(score_shot(result, {'speed': None}, RULES)['value'], 0)

    def test_accuracy_only_rule_does_not_require_speed_or_timing(self):
        rules = {**RULES, 'speed_weight': 0, 'accuracy_weight': 1}
        result = score_shot(self.goal(), {'speed': None, 'measurement_status': 'UNAVAILABLE'}, rules)
        self.assertEqual(result['value'], 100)
        self.assertIsNone(result['components']['speed_normalized'])

    def test_speed_and_goal_ball_size_assumptions_must_match(self):
        goal = self.goal()
        goal['plane']['config']['ball_radius_m'] = .1
        result = score_shot(goal, {'speed': 60, 'measurement_status': 'ESTIMATED',
                                 'physics_diagnostics': {'ball_radius_m': .11}}, RULES)
        self.assertIsNone(result['value'])
        self.assertEqual(result['reason'], 'BALL_RADIUS_ASSUMPTION_MISMATCH')

    def test_invalid_or_unagreed_configuration_rejected(self):
        _, _, _, config = fixture()
        for changes in [{'confirmed_by_user': False}, {'width_m': float('nan')}, {'reference_frame': True},
                        {'plane_reference': 'detected_rectangle'}, {'pixel_tolerance_px': 0}, {'extra': 1}]:
            with self.assertRaises(ValueError):
                validate_goal_plane({**config, **changes})
        for changes in [{'speed_weight': .8}, {'confirmed_by_user': False}, {'accept_conditional_estimates': False},
                        {'speed_max_kmh': 20}, {'accuracy_radius_m': 0}, {'target_normalized': [2, .5]}]:
            with self.assertRaises(ValueError):
                validate_scoring_rules({**RULES, **changes})
        self.assertEqual(validate_options({'goal_plane': config, 'scoring_rules': RULES})['scoring_rules'], RULES)


class Week9ScoringTests(unittest.TestCase):
    def crossing(self, x=1.5, y=1.):
        rows, meta, camera, config = fixture(x=x, y=y)
        # Exact synthetic projections can use a smaller perturbation at the corners.
        config['pixel_tolerance_px'] = .05
        return assess_physical_goal(rows, meta, camera, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'}, config)

    def physics(self, speed=60):
        return {'speed': speed, 'measurement_status': 'ESTIMATED', 'speed_basis': 'CALIBRATED_INPUTS',
                'measurement_context': {'timing': {'status': 'CONFIRMED_BY_USER'}},
                'physics_diagnostics': {'ball_radius_m': .11}}

    def test_valid_measured_crossings_reward_outer_locations_at_the_same_speed(self):
        for (x, y), value in [((1.5, 1.), 20.), ((2.7, 1.), 53.9), ((2.7, 1.8), 68.)]:
            with self.subTest(x=x, y=y):
                result = score_week9(self.crossing(x, y), self.physics())
                self.assertEqual(result['status'], 'PROVISIONAL')
                self.assertEqual(result['value'], value)
                self.assertEqual(result['rule_set_id'], 'week9_outer_v1')

    def test_outside_zero_and_center_zero_have_different_reasons(self):
        outside = score_week9(self.crossing(-.5), {})
        center = score_week9(self.crossing(), self.physics(20))
        self.assertEqual((outside['value'], outside['basis']), (0., 'MEASURED_PLANE_OUTSIDE'))
        self.assertEqual((center['value'], center['basis']), (0., 'MEASURED_PLANE_OUTER_DISTANCE'))

    def test_missing_or_incompatible_inputs_never_produce_partial_scores(self):
        goal, physics = self.crossing(), self.physics()
        no_position = copy.deepcopy(goal)
        no_position['events'][0]['crossing_normalized'] = [1.1, .5]
        cases = [(no_position, physics, 'GOAL_CROSSING_EVIDENCE_REQUIRED'),
                 (goal, {**physics, 'speed': None}, 'SPEED_ESTIMATE_REQUIRED'),
                 (goal, {**physics, 'speed_basis': 'ASSUMPTION_BASED'}, 'SPEED_BASIS_UNVERIFIED'),
                 (goal, {**physics, 'measurement_context': {}}, 'SPEED_BASIS_UNVERIFIED'),
                 (goal, {**physics, 'physics_diagnostics': {'ball_radius_m': .10}}, 'BALL_RADIUS_ASSUMPTION_MISMATCH')]
        for physical, estimate, reason in cases:
            with self.subTest(reason=reason):
                score = score_week9(physical, estimate)
                self.assertEqual((score['status'], score['value'], score['reason']), ('NOT_EVALUATED', None, reason))
                self.assertEqual(score['components'], {})


if __name__ == '__main__':
    unittest.main()
