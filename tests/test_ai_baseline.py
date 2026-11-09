import tempfile
import unittest
from pathlib import Path

from tools.run_ai_baseline import ROOT, classify_result, diagnostics, video_key


class BaselineTests(unittest.TestCase):
    def test_current_analyzer_failure_is_not_success(self):
        result = {"speed": 0, "score": 0, "comment": "공을 찾지 못했습니다.", "trajectory_image": "missing.jpg"}
        self.assertEqual(classify_result(result, False), ("FAILED", "BALL_NOT_FOUND"))

    def test_result_requires_readable_image(self):
        self.assertEqual(classify_result({"speed": 90, "score": 69}, False), ("FAILED", "OUTPUT_IMAGE_MISSING"))
        self.assertEqual(classify_result({"speed": 90, "score": 69}, True), ("SUCCESS", None))

    def test_invalid_speeds_are_not_success(self):
        for speed in [0, -1, float("inf"), float("nan"), None, "90", True]:
            with self.subTest(speed=speed):
                self.assertEqual(classify_result({"speed": speed}, True)[0], "FAILED")

    def test_same_basename_in_two_groups_has_distinct_output_key(self):
        root = Path("soccer_videos")
        self.assertNotEqual(video_key(root / "test_videos/video1.mp4", root), video_key(root / "real_test_videos/video1.mp4", root))

    def test_diagnostics_preserve_optimizer_failure_and_point_count(self):
        temp_root = ROOT / "output" / "ai_baseline" / "_tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            log = Path(directory) / "sample.log"
            log.write_text("[V2.5] Phase 3: Trajectory Optimization with 12 points\n[Optimizer] Failed: Initial guess is outside of provided bounds\n", encoding="utf-8")
            result = diagnostics(log)
        self.assertEqual(result["tracked_point_count"], 12)
        self.assertIn("bounds", result["optimizer_error"])
        self.assertFalse(result["goal_scale_detected"])


if __name__ == "__main__":
    unittest.main()
