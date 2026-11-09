import unittest
import json
from pathlib import Path
import tempfile

from tools.evaluate_references import aggregate, compare_reference, digest, evaluate_ledger, make_ledger


RAW = {'speed': 60, 'measurement_status': 'ESTIMATED', 'flight_segment': {'start_frame': 10, 'end_frame': 20},
       'physical_goal': {'status': 'ESTIMATED_GOAL'}}
REFERENCE = {'speed': {'value_kmh': 50, 'method': 'radar_window_mean', 'quantity': 'initial_flight_mean_speed',
                       'independent': True, 'evidence': 'separate_radar_record.csv', 'frame_interval': [10, 20]},
             'goal': {'is_goal': True, 'method': 'independent_goal_line_view',
                      'independent': True, 'evidence': 'side_view.mp4#frame=123'}}


class ReferenceTests(unittest.TestCase):
    def test_ledger_uses_original_input_hash_and_checks_saved_result_against_summary(self):
        runtime = Path(__file__).resolve().parents[1]/'output/ai_baseline/_tests'
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            run = Path(directory)
            inputs = run/'inputs'
            inputs.mkdir()
            (run/'results').mkdir()
            source, record = inputs/'test.mp4', run/'results/test.json'
            source.write_bytes(b'original')
            item = {'video': 'test.mp4', 'raw_result': RAW}
            record.write_text(json.dumps(item), encoding='utf-8')
            summary = {'run': {'input_directory': str(inputs), 'inputs': [{'path': str(source), 'sha256': digest(source)}]},
                       'videos': [item]}
            (run/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            self.assertEqual(len(make_ledger(run)['shots']), 1)
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                make_ledger(run)
            source.write_bytes(b'original')
            record.write_text(json.dumps({'video': 'test.mp4', 'raw_result': {**RAW, 'speed': 61}}), encoding='utf-8')
            with self.assertRaises(ValueError):
                make_ledger(run)

    def test_domains_stay_separate_and_changed_input_is_rejected(self):
        runtime = Path(__file__).resolve().parents[1]/'output/ai_baseline/_tests'
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            source, record = Path(directory)/'input.mp4', Path(directory)/'result.json'
            source.write_bytes(b'synthetic-test-fixture')
            record.write_text(json.dumps({'video': 'test.mp4', 'raw_result': RAW}), encoding='utf-8')
            entry = {'video': 'test.mp4', 'domain': 'real', 'group': 'normal', 'input_path': str(source),
                     'input_sha256': digest(source), 'result_path': str(record), 'result_sha256': digest(record), 'reference': REFERENCE}
            ledger = {'shots': [entry, {**entry, 'domain': 'synthetic'}]}
            report = evaluate_ledger(ledger)
            self.assertEqual(report['domains']['real']['shot_count'], 1)
            self.assertEqual(report['domains']['synthetic']['shot_count'], 1)
            with self.assertRaises(ValueError):
                evaluate_ledger({'shots': [entry, entry]})
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                evaluate_ledger(ledger)

    def test_known_errors_and_goal_confusion(self):
        row = compare_reference(RAW, REFERENCE)
        report = aggregate([row])
        self.assertEqual(report['speed_mae_kmh'], 10)
        self.assertEqual(report['speed_mape_percent'], 20)
        self.assertEqual(report['goal_confusion']['TP'], 1)
        self.assertEqual(report['goal_accuracy'], 1)
        self.assertFalse(report['accuracy_validated'])

    def test_missing_reference_does_not_report_zero_error_or_perfect_accuracy(self):
        report = aggregate([compare_reference(RAW, {'speed': None, 'goal': None})])
        self.assertIsNone(report['speed_mae_kmh'])
        self.assertIsNone(report['goal_accuracy'])
        self.assertEqual(report['speed_comparison_coverage'], 0)

    def test_wrong_time_quantity_or_ai_label_is_not_comparable(self):
        for changes in [{'frame_interval': [11, 20]}, {'quantity': 'peak_speed'},
                        {'method': 'timed_distance'}, {'independent': False}, {'value_kmh': float('nan')}]:
            ref = {**REFERENCE, 'speed': {**REFERENCE['speed'], **changes}}
            self.assertNotEqual(compare_reference(RAW, ref)['speed']['status'], 'COMPARED')

    def test_unavailable_ai_and_image_candidate_are_excluded_but_reduce_coverage(self):
        raw = {**RAW, 'speed': None, 'measurement_status': 'UNAVAILABLE',
               'physical_goal': {'status': 'UNDETERMINED'}, 'goal_entry': {'status': 'CANDIDATE_OBSERVED'}}
        unavailable = compare_reference(raw, REFERENCE)
        report = aggregate([compare_reference(RAW, REFERENCE), unavailable])
        self.assertEqual(report['speed_comparison_coverage'], .5)
        self.assertEqual(report['goal_comparison_coverage'], .5)
        self.assertEqual(report['speed_mae_kmh'], 10)
        self.assertEqual(unavailable['goal']['status'], 'AI_UNAVAILABLE')

    def test_goal_labels_must_be_boolean_and_independent(self):
        for changes in [{'is_goal': 'yes'}, {'independent': False}, {'method': 'front_camera_guess'}]:
            self.assertEqual(compare_reference(RAW, {**REFERENCE, 'goal': {**REFERENCE['goal'], **changes}})['goal']['status'], 'INVALID_REFERENCE')


if __name__ == '__main__':
    unittest.main()
