import unittest

import cv2
import numpy as np

from backend.ball_size_quality import BallSizeInspector


def ball(color=(240, 240, 240)):
    frame = np.full((160, 160, 3), (30, 70, 30), dtype=np.uint8)
    cv2.circle(frame, (80, 80), 30, color, -1)
    return frame


class SizeQualityTests(unittest.TestCase):
    def setUp(self):
        self.seed = ball()
        self.inspector = BallSizeInspector(self.seed, [80, 80, 60, 60])

    def test_clean_ball_has_supported_size(self):
        self.assertTrue(self.inspector.inspect(self.seed, [50, 50, 110, 110])['usable'])

    def test_black_panels_do_not_require_one_white_component(self):
        frame = self.seed.copy()
        cv2.line(frame, (50, 80), (110, 80), (0, 0, 0), 10)
        cv2.line(frame, (80, 50), (80, 110), (0, 0, 0), 10)
        self.assertTrue(self.inspector.inspect(frame, [50, 50, 110, 110])['usable'])

    def test_shoe_hides_lower_half_and_invalidates_size(self):
        frame = self.seed.copy()
        frame[80:111, 50:111] = (25, 25, 25)
        result = self.inspector.inspect(frame, [45, 50, 115, 110])
        self.assertFalse(result['usable'])
        self.assertEqual(result['reason'], 'PARTIAL_FOREGROUND_OR_BOX_CONTAMINATION')

    def test_elongated_ball_and_shoe_box_is_not_a_diameter(self):
        result = self.inspector.inspect(self.seed, [50, 40, 110, 140])
        self.assertFalse(result['usable'])
        self.assertEqual(result['reason'], 'ELONGATED_BALL_BOX')

    def test_colored_ball_keeps_geometric_size_check(self):
        yellow = ball((20, 210, 230))
        inspector = BallSizeInspector(yellow, [80, 80, 60, 60])
        result = inspector.inspect(yellow, [50, 50, 110, 110])
        self.assertTrue(result['usable'])
        self.assertFalse(result['foreground_checked'])

    def test_bright_background_does_not_claim_foreground_support(self):
        frame = np.full((160, 160, 3), 220, dtype=np.uint8)
        cv2.circle(frame, (80, 80), 30, (240, 240, 240), -1)
        inspector = BallSizeInspector(frame, [80, 80, 60, 60])
        result = inspector.inspect(frame, [50, 50, 110, 110])
        self.assertTrue(result['usable'])
        self.assertFalse(result['foreground_checked'])


if __name__ == '__main__':
    unittest.main()
