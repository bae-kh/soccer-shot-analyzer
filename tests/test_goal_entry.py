import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from goal_entry import assess_goal_entry, evaluate_score, goal_camera_interval

GOAL = {'bbox': [100, 100, 300, 200], 'confidence': .8,
        'coordinate_space': 'original_pixels', 'origin': 'top_left'}
META = {'width': 640, 'height': 360, 'fps': 30}


def point(frame, x, y=150, source='yolo', confidence=.8):
    return {'frame_number': frame, 'raw_center': [x, y],
            'ball_bbox': [x - 5, y - 5, x + 5, y + 5], 'source': source,
            'confidence': confidence if source == 'yolo' else None,
            'observed': source in {'yolo', 'csrt'}, 'x': x, 'y': y}


class GoalEntryTests(unittest.TestCase):
    def test_camera_interval_covers_ball_before_goal_reference_frame(self):
        self.assertEqual(goal_camera_interval([point(14, 80)], {'frame_number': 17}, 100), (14, 100))

    def test_camera_interval_covers_later_reference_and_empty_track(self):
        self.assertEqual(goal_camera_interval([point(2, 80)], {'frame_number': 17}, 10), (2, 17))
        self.assertEqual(goal_camera_interval([], None, 10), (0, 10))

    def assess(self, rows=None, goal=GOAL, motion='FIXED_CANDIDATE', kick=True):
        return assess_goal_entry(rows if rows is not None else [point(10, 80), point(11, 130), point(12, 160)],
                                 goal, META, {'kick_detected': kick}, {'status': motion})

    def test_observed_entry_is_only_image_candidate_and_no_score(self):
        result = self.assess()
        self.assertEqual(result['status'], 'CANDIDATE_OBSERVED')
        self.assertEqual(result['physical_goal_status'], 'UNDETERMINED')
        self.assertEqual(result['events'][0]['inside_yolo_frames'], [11, 12])
        self.assertEqual(result['events'][0]['entry_image_position_normalized'], [.15, .5])
        score = evaluate_score(result)
        self.assertIsNone(score['value'])
        self.assertFalse(score['candidate_used_as_goal'])
        self.assertFalse(score['prerequisites']['physical_goal_confirmed'])

    def test_candidate_after_early_physics_window_and_unknown_timing(self):
        rows = [point(100, 80), point(101, 130), point(102, 160)]
        metadata = {**META, 'measurement_context': {'timing': {'effective_capture_fps': None}}}
        result = assess_goal_entry(rows, GOAL, metadata, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'})
        self.assertEqual(result['status'], 'CANDIDATE_OBSERVED')

    def test_center_inside_but_ball_crosses_boundary_is_deferred(self):
        result = self.assess([point(1, 80), point(2, 102), point(3, 103)])
        self.assertEqual(result['reason'], 'BORDER_OR_BRIEF_OVERLAP')
        self.assertEqual(result['events'], [])

    def test_single_inside_detection_is_deferred(self):
        result = self.assess([point(1, 70), point(2, 80), point(3, 140)])
        self.assertEqual(result['status'], 'UNAVAILABLE')

    def test_predictions_and_missing_frames_cannot_bridge_entry(self):
        for source in ('kalman_prediction', 'missing'):
            with self.subTest(source=source):
                result = self.assess([point(1, 80), point(2, 100, source=source), point(3, 140), point(4, 160)])
                self.assertEqual(result['reason'], 'ENTRY_CONTINUITY_UNVERIFIED')

    def test_gap_between_saved_frames_cannot_bridge_entry(self):
        self.assertEqual(self.assess([point(1, 80), point(3, 140), point(4, 160)])['status'], 'UNAVAILABLE')

    def test_csrt_inside_alone_cannot_create_candidate(self):
        result = self.assess([point(1, 80), point(2, 130, source='csrt'), point(3, 160, source='csrt')])
        self.assertEqual(result['events'], [])

    def test_csrt_can_support_continuity_between_trusted_yolo(self):
        result = self.assess([point(1, 80), point(2, 130), point(3, 140, source='csrt'), point(4, 160)])
        self.assertEqual(result['events'][0]['inside_yolo_frames'], [2, 4])

    def test_weak_or_rejected_yolo_cannot_create_candidate(self):
        for bad in (point(3, 160, confidence=.1), {**point(3, 160), 'observation_validation': {'accepted': False}}):
            self.assertEqual(self.assess([point(1, 80), point(2, 130), bad])['events'], [])

    def test_starting_inside_and_static_edge_jitter_do_not_establish_entry(self):
        self.assertEqual(self.assess([point(1, 130), point(2, 130), point(3, 130)])['reason'], 'ENTRY_CONTINUITY_UNVERIFIED')

    def test_camera_movement_and_unverified_camera_defer(self):
        for motion, reason in [('MOVING', 'CAMERA_MOVING'), ('UNVERIFIED', 'CAMERA_MOTION_UNVERIFIED')]:
            self.assertEqual(self.assess(motion=motion)['reason'], reason)

    def test_missing_goal_and_no_kick_defer(self):
        self.assertEqual(self.assess(goal=None)['reason'], 'GOAL_NOT_FOUND')
        self.assertEqual(self.assess(kick=False)['reason'], 'KICK_NOT_DETECTED')

    def test_invalid_goal_bounds_or_coordinate_space_defer(self):
        for changes in ({'bbox': [100, 100, 700, 200]}, {'bbox': [100, 200, 300, 100]},
                        {'coordinate_space': 'normalized'}, {'bbox': [100, float('nan'), 300, 200]}):
            self.assertEqual(self.assess(goal={**GOAL, **changes})['reason'], 'INVALID_GOAL_GEOMETRY')

    def test_invalid_ball_and_order_cannot_create_candidate(self):
        self.assertEqual(self.assess([point(1, 80), point(2, 130), point(2, 160)])['reason'], 'INVALID_OBSERVATION_SEQUENCE')
        self.assertEqual(self.assess([point(1, 80), point(2, 130), {**point(3, 160), 'raw_center': [float('nan'), 150]}])['events'], [])

    def test_no_candidate_is_not_a_miss(self):
        result = self.assess([point(1, 60), point(2, 70), point(3, 80)])
        self.assertEqual(result['status'], 'NO_CANDIDATE_OBSERVED')
        self.assertEqual(result['physical_goal_status'], 'UNDETERMINED')

    def test_multiple_entries_and_input_immutability(self):
        rows = [point(1, 80), point(2, 130), point(3, 160), point(4, 80), point(5, 140), point(6, 150)]
        original = copy.deepcopy(rows)
        self.assertEqual(len(self.assess(rows)['events']), 2)
        self.assertEqual(rows, original)


if __name__ == '__main__':
    unittest.main()
