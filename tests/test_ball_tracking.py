import unittest
from backend.ball_tracking import BallTracker, select_initial_ball
from backend.goal_geometry import assemble_goal


def detection(x, y, radius=10, confidence=0.8):
    return {"bbox": [x, y, radius * 2, radius * 2], "confidence": confidence}


class TrackingTests(unittest.TestCase):
    def make_tracker(self):
        initial = detection(300, 700)
        tracker = BallTracker(initial, 1920, 1080, 30)
        tracker.step(0, [initial])
        return tracker

    def test_stationary_ball_does_not_start_a_shot(self):
        tracker = self.make_tracker()
        for frame in range(1, 20):
            tracker.step(frame, [detection(300 + frame % 2, 700)])
        self.assertFalse(tracker.active)
        self.assertEqual(tracker.trajectory(), [])

    def test_fast_kick_above_old_120_pixel_limit(self):
        tracker = self.make_tracker()
        row, _ = tracker.step(1, [detection(450, 500)])
        self.assertTrue(tracker.active)
        self.assertTrue(row["observed"])
        self.assertEqual(tracker.shot_start_frame, 0)
        self.assertAlmostEqual(row["time_sec"], 1 / 30)

    def test_weak_motion_requires_confirmation(self):
        tracker = self.make_tracker()
        tracker.step(1, [detection(300, 680, confidence=0.06)])
        self.assertFalse(tracker.active)
        tracker.step(2, [detection(300, 660, confidence=0.08)])
        self.assertTrue(tracker.active)

    def test_prediction_is_bounded_and_not_counted_as_observation(self):
        tracker = self.make_tracker()
        tracker.step(1, [detection(312, 700)])
        for frame in range(2, tracker.prediction_limit + 2):
            row, _ = tracker.step(frame, [])
            self.assertEqual(row["source"], "kalman_prediction")
            self.assertFalse(row["observed"])
        row, _ = tracker.step(tracker.prediction_limit + 2, [])
        self.assertEqual(row["source"], "missing")
        self.assertIsNone(row["x"])
        self.assertEqual(tracker.summary()["observed_points"], 2)

    def test_reacquires_after_occlusion(self):
        tracker = self.make_tracker()
        tracker.step(1, [detection(312, 700)])
        tracker.step(2, [])
        tracker.step(3, [])
        row, _ = tracker.step(4, [detection(348, 700)])
        self.assertEqual(row["source"], "yolo")
        self.assertTrue(row["reacquired"])

    def gap_tracker(self):
        tracker = self.make_tracker()
        tracker.step(1, [detection(312, 700)])
        tracker.step(2, [])
        tracker.step(3, [])
        return tracker

    def test_one_weak_distant_reacquisition_cannot_seed_observed_track(self):
        tracker = self.gap_tracker()
        row, selected = tracker.step(4, [detection(420, 700, confidence=.07)])
        self.assertIsNone(selected)
        self.assertFalse(row['observed'])
        self.assertEqual(row['rejected_observations'][0]['reason'], 'WEAK_REACQUISITION_UNCONFIRMED')
        row, _ = tracker.step(5, [])
        self.assertFalse(row['observed'])

    def test_two_consecutive_weak_detector_confirmations_can_reacquire(self):
        tracker = self.gap_tracker()
        tracker.step(4, [detection(420, 700, confidence=.07)])
        row, selected = tracker.step(5, [detection(430, 700, confidence=.07)])
        self.assertIsNotNone(selected)
        self.assertTrue(row['observed'])
        self.assertTrue(row['reacquired'])

    def test_weak_confirmation_does_not_transfer_to_another_distant_object(self):
        tracker = self.gap_tracker()
        tracker.step(4, [detection(420, 700, confidence=.07)])
        row, _ = tracker.step(5, [detection(500, 700, confidence=.07)])
        self.assertFalse(row['observed'])

    def test_roi_candidate_probe_is_not_a_temporal_confirmation(self):
        tracker = self.gap_tracker()
        tracker.current_frame = 4
        self.assertIsNone(tracker._associate([detection(420, 700, confidence=.07)]))
        self.assertIsNone(tracker.pending_reacquisition)
        row, _ = tracker.step(4, [detection(420, 700, confidence=.07)])
        self.assertFalse(row['observed'])

    def test_strong_and_nearby_weak_reacquisition_are_preserved(self):
        for x, confidence in [(420, .8), (350, .07)]:
            with self.subTest(x=x):
                row, selected = self.gap_tracker().step(4, [detection(x, 700, confidence=confidence)])
                self.assertIsNotNone(selected)
                self.assertTrue(row['observed'])

    def test_distant_high_confidence_outlier_is_rejected(self):
        tracker = self.make_tracker()
        tracker.step(1, [detection(312, 700)])
        row, selected = tracker.step(2, [detection(1850, 50, confidence=0.99)])
        self.assertIsNone(selected)
        self.assertEqual(row["source"], "kalman_prediction")

    def test_drifting_fallback_is_rejected(self):
        tracker = self.make_tracker()
        tracker.step(1, [detection(312, 700)])
        row, _ = tracker.step(2, [], [1500, 100, 20, 20])
        self.assertEqual(row["source"], "kalman_prediction")

    def test_long_gap_finishes_instead_of_inventing_track(self):
        tracker = self.make_tracker()
        tracker.step(1, [detection(312, 700)])
        for frame in range(2, tracker.recovery_limit + 3):
            tracker.step(frame, [])
        self.assertTrue(tracker.finished)
        row, selected = tracker.step(50, [detection(350, 700)])
        self.assertIsNone(selected)
        self.assertEqual(row["source"], "missing")

    def test_initial_selection_prefers_shooting_ball_over_top_sign(self):
        selected = select_initial_ball([(0, [detection(1000, 850, confidence=0.22), detection(960, 180, confidence=0.35)])], 1920, 1080)
        self.assertEqual(selected["bbox"][:2], [1000, 850])

    def test_rapid_acceleration_after_duplicate_blurred_kick(self):
        initial = detection(590, 800, radius=38)
        tracker = BallTracker(initial, 1920, 1080, 30)
        tracker.step(0, [initial])
        tracker.step(1, [detection(621, 786, radius=38, confidence=0.14)])
        tracker.step(2, [detection(621, 786, radius=38, confidence=0.14)])
        row, selected = tracker.step(3, [detection(739, 587, radius=35, confidence=0.85),
                                        detection(620, 770, radius=20, confidence=0.08)])
        self.assertEqual(selected["bbox"][:2], [739, 587])
        self.assertTrue(row["observed"])

    def test_static_background_logo_does_not_start_a_shot(self):
        frames = [(i, [detection(951, 654, confidence=0.25),
                       detection(824, 346, confidence=0.85)]) for i in range(10)]
        initial = select_initial_ball(frames, 1920, 1080)
        tracker = BallTracker(initial, 1920, 1080, 30)
        tracker.step(0, frames[0][1])
        tracker.step(1, [detection(824, 346, confidence=0.9)])
        self.assertFalse(tracker.active)

    def test_later_airborne_detection_does_not_replace_initial_ball(self):
        initial = select_initial_ball([(0, [detection(1000, 850, confidence=0.2)]),
                                       (18, [detection(800, 450, confidence=0.95)])], 1920, 1080)
        self.assertEqual(initial["frame_number"], 0)

    def test_weak_detection_on_shoe_does_not_reverse_flight(self):
        tracker = BallTracker(detection(665, 950, radius=33), 1920, 1080, 30)
        tracker.step(0, [detection(665, 950, radius=33)])
        for frame, x, y, confidence in [(1, 655, 876, 0.063), (2, 639, 833, 0.084),
                                         (3, 610, 758, 0.60), (4, 596, 707, 0.178),
                                         (5, 576, 675, 0.172)]:
            tracker.step(frame, [detection(x, y, radius=28, confidence=confidence)])
        row, selected = tracker.step(6, [detection(520, 973, radius=22, confidence=0.091)])
        self.assertIsNone(selected)
        self.assertFalse(row["observed"])


class GoalTests(unittest.TestCase):
    def test_separate_posts_and_crossbar_form_goal_mouth(self):
        parts = [{"bbox": [480, 450, 505, 690], "confidence": 0.7},
                 {"bbox": [1230, 480, 1265, 700], "confidence": 0.8},
                 {"bbox": [485, 425, 1260, 460], "confidence": 0.8}]
        area = assemble_goal(parts, 1920, 1080, 0)
        self.assertEqual(area["method"], "paired_posts")
        self.assertEqual(area["bbox"], [480.0, 425.0, 1265.0, 700.0])
        self.assertEqual(area["coordinate_space"], "original_pixels")

    def test_single_post_is_not_a_complete_goal(self):
        self.assertIsNone(assemble_goal([{"bbox": [690, 110, 712, 360], "confidence": 0.9}], 1920, 1080, 0))

    def test_full_box_fallback(self):
        area = assemble_goal([{"bbox": [780, 415, 1128, 517], "confidence": 0.8}], 1920, 1080, 0)
        self.assertEqual(area["method"], "box")


if __name__ == "__main__":
    unittest.main()
