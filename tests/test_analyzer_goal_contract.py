"""Failure-path contract with an empty detector on a readable synthetic video."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from test_goal_measurement import RULES, fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))


class AnalyzerGoalContractTests(unittest.TestCase):
    def test_both_modes_use_the_new_default_and_keep_explicit_custom_rules(self):
        from analyze_shot import _goal_outputs
        from goal_measurement import assess_physical_goal
        rows, meta, camera, config = fixture()
        physical = assess_physical_goal(rows, meta, camera, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'}, config)
        physics = {'speed': 60, 'measurement_status': 'ESTIMATED', 'speed_basis': 'CALIBRATED_INPUTS',
                   'measurement_context': {'timing': {'status': 'CONFIRMED_BY_USER'}},
                   'physics_diagnostics': {'ball_radius_m': .11}}
        for mode in ('automatic', 'calibrated'):
            entry = {'status': 'CANDIDATE_OBSERVED'}
            prediction, default = _goal_outputs(entry, physical, physics, {'analysis_mode': mode, 'options': {}})
            self.assertEqual(entry['physical_goal_status'], 'ESTIMATED_GOAL')
            self.assertEqual((default['rule_set_id'], default['value']), ('week9_outer_v1', 20.))
            if mode == 'automatic':
                self.assertEqual(prediction['basis'], 'measured_goal_plane')
            else:
                self.assertIsNone(prediction)
            _, custom = _goal_outputs(entry, physical, physics, {'analysis_mode': mode, 'options': {'scoring_rules': RULES}})
            self.assertEqual((custom['rule_set_id'], custom['value'], custom['basis']), (RULES['id'], 80., 'USER_DEFINED_TARGET'))

    def test_no_initial_ball_retains_goal_and_score_deferred_contract(self):
        from analyze_shot import ProShootingAnalyzer
        runtime = ROOT/'output/ai_baseline/_tests'
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            video = str(Path(directory)/'blank.mp4')
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            analyzer = ProShootingAnalyzer.__new__(ProShootingAnalyzer)
            analyzer.callback, analyzer.goal_area = None, None
            analyzer.extract_balls = lambda frame: []
            analyzer.detect_goal_area = lambda *args: None
            result = analyzer.run(video, str(Path(directory)/'result.jpg'))
            self.assertEqual(result['failure_reason'], 'BALL_NOT_FOUND')
            self.assertIsNone(result['speed'])
            self.assertEqual(result['analysis_mode'], 'automatic')
            self.assertEqual(result['measurement_context']['timing']['mode'], 'normal')
            self.assertEqual(result['goal_prediction']['status'], 'UNDETERMINED')
            self.assertEqual(result['goal_entry']['reason'], 'GOAL_NOT_FOUND')
            self.assertEqual(result['goal_entry']['physical_goal_status'], 'UNDETERMINED')
            self.assertEqual(result['physical_goal']['reason'], 'GOAL_PLANE_REQUIRED')
            self.assertIsNone(result['score_evaluation']['value'])
            self.assertIsNotNone(cv2.imread(str(Path(directory)/result['trajectory_image'])))


if __name__ == '__main__':
    unittest.main()
