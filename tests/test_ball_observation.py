"""Guard against changing identity, static distractors, and CSRT drift."""
import unittest

import cv2
import numpy as np

from backend.ball_tracking import BallTracker
from backend.ball_observation import BallObservationGuard


def scene(ball_center=(80, 220), ball_color=(230, 230, 230), distractor=True):
    rng = np.random.default_rng(12)
    image = rng.integers(35, 75, (300, 400, 3), dtype=np.uint8)
    if distractor:
        cv2.circle(image, (200, 90), 13, (210, 210, 210), -1)
    if ball_center is not None:
        cv2.circle(image, ball_center, 12, ball_color, -1)
        cv2.circle(image, (ball_center[0]-3, ball_center[1]-2), 3, (30, 30, 30), -1)
    return image


def detection(center, confidence=.8):
    return {'bbox': [*center, 24, 24], 'confidence': confidence}


class ObservationTests(unittest.TestCase):
    def tracker(self, color=(230, 230, 230)):
        initial = detection((80, 220))
        frame = scene(ball_color=color)
        tracker = BallTracker(initial, 400, 300, 30, initial_frame=frame)
        tracker.step(0, [initial], frame=frame)
        return tracker

    def test_fast_true_ball_is_preserved(self):
        tracker = self.tracker()
        row, selected = tracker.step(1, [detection((120, 180))], frame=scene((120, 180)))
        self.assertIsNotNone(selected)
        self.assertTrue(row['observed'])
        self.assertEqual(row['observation_validation']['status'], 'IMAGE_CHECKED')

    def test_yellow_ball_does_not_require_white_pixels(self):
        color = (15, 215, 230)
        tracker = self.tracker(color)
        row, selected = tracker.step(1, [detection((120, 180))], frame=scene((120, 180), color))
        self.assertIsNotNone(selected)
        self.assertTrue(row['observed'])

    def test_stationary_letter_is_not_a_new_ball(self):
        guard = BallObservationGuard(scene(), detection((80, 220))['bbox'])
        current = scene(None)
        guard.begin_frame(1, current)
        result = guard.inspect(current, detection((200, 90), .95)['bbox'], previous_center=(80, 220), previous_radius=12)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'UNCHANGED_BACKGROUND_OBJECT')

    def test_dark_wrong_object_does_not_correct_kalman(self):
        tracker = self.tracker()
        tracker.step(1, [detection((120, 180))], frame=scene((120, 180)))
        row, selected = tracker.step(2, [detection((135, 165))], frame=scene((135, 165), (15, 15, 15)))
        self.assertIsNone(selected)
        self.assertEqual(row['source'], 'kalman_prediction')
        self.assertIsNone(row['ball_bbox'])
        self.assertFalse(row['observed'])
        self.assertEqual(row['rejected_observations'][0]['reason'], 'BALL_APPEARANCE_MISMATCH')

    def test_csrt_success_on_empty_background_is_rejected(self):
        tracker = self.tracker()
        tracker.step(1, [detection((120, 180))], frame=scene((120, 180)))
        row, _ = tracker.step(2, [], [140, 160, 24, 24], frame=scene(None))
        self.assertEqual(row['source'], 'kalman_prediction')
        self.assertFalse(row['observed'])
        self.assertEqual(row['rejected_observations'][0]['source'], 'csrt')

    def test_duplicate_frame_does_not_lose_a_ball_already_tracked(self):
        tracker = self.tracker()
        frame = scene((120, 180))
        tracker.step(1, [detection((120, 180))], frame=frame)
        row, selected = tracker.step(2, [detection((120, 180))], frame=frame)
        self.assertIsNotNone(selected)
        self.assertTrue(row['observed'])

    def test_matching_visible_ball_can_reacquire_with_weak_detector_after_gap(self):
        tracker = self.tracker()
        tracker.step(1, [detection((120, 180))], frame=scene((120, 180)))
        tracker.step(2, [], frame=scene(None))
        tracker.step(3, [], frame=scene(None))
        candidate = {**detection((220, 80)), 'confidence': .07}
        row, selected = tracker.step(4, [candidate], frame=scene((220, 80)))
        self.assertIsNotNone(selected)
        self.assertTrue(row['observed'])
        self.assertTrue(row['reacquired'])

    def test_ambiguous_colored_patch_cannot_reacquire_from_one_weak_detection(self):
        tracker = self.tracker()
        tracker.step(1, [detection((120, 180))], frame=scene((120, 180)))
        tracker.step(2, [], frame=scene(None))
        tracker.step(3, [], frame=scene(None))
        current = scene((220, 80))
        cv2.rectangle(current, (220, 68), (231, 92), (230, 30, 30), -1)
        row, selected = tracker.step(4, [{**detection((220, 80)), 'confidence': .07}], frame=current)
        self.assertIsNone(selected)
        self.assertFalse(row['observed'])
        self.assertEqual(row['rejected_observations'][0]['reason'], 'WEAK_REACQUISITION_UNCONFIRMED')

    def test_csrt_stuck_at_last_yolo_location_still_checks_background(self):
        seed = scene(ball_color=(210, 210, 210))
        guard = BallObservationGuard(seed, detection((80, 220))['bbox'])
        current = scene(None)
        guard.begin_frame(1, current)
        result = guard.inspect(current, detection((200, 90))['bbox'], source='csrt',
                               previous_center=(200, 90), previous_radius=12)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'CSRT_UNCHANGED_BACKGROUND')

    def test_csrt_stationary_arrived_ball_is_preserved(self):
        seed = scene(distractor=False)
        guard = BallObservationGuard(seed, detection((80, 220))['bbox'])
        current = scene((200, 90), distractor=False)
        for frame in (1, 2):
            guard.begin_frame(frame, current)
            result = guard.inspect(current, detection((200, 90))['bbox'], source='csrt',
                                   previous_center=(200, 90), previous_radius=12)
            self.assertTrue(result['accepted'])

    def test_csrt_at_seed_area_does_not_lose_stationary_ball(self):
        seed = scene()
        guard = BallObservationGuard(seed, detection((80, 220))['bbox'])
        guard.begin_frame(1, seed)
        result = guard.inspect(seed, detection((80, 220))['bbox'], source='csrt',
                               previous_center=(80, 220), previous_radius=12)
        self.assertTrue(result['accepted'])

    def test_csrt_dark_occluder_patch_needs_detector_confirmation(self):
        seed = scene()
        guard = BallObservationGuard(seed, detection((80, 220))['bbox'])
        current = scene(None)
        # Partial dark occluder with adjacent bright background. This patch
        # retains enough seed appearance to pass the broad YOLO guard.
        for left, right, value in [(108, 120, 40), (120, 126, 200), (126, 132, 230)]:
            cv2.rectangle(current, (left, 168), (right-1, 191), (value, value, value), -1)
        guard.begin_frame(1, current)
        csrt = guard.inspect(current, detection((120, 180))['bbox'], 'csrt', (120, 180), 12)
        yolo = guard.inspect(current, detection((120, 180))['bbox'], 'yolo', (120, 180), 12)
        self.assertFalse(csrt['accepted'])
        self.assertEqual(csrt['reason'], 'CSRT_OCCLUSION_OR_APPEARANCE_MISMATCH')
        self.assertTrue(yolo['accepted'])


if __name__ == '__main__':
    unittest.main()
