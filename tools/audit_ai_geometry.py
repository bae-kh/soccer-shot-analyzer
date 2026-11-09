"""Validate stage 3 artifacts and interface invariants; not an accuracy metric."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.automatic_analysis import predict_goal
from backend.goal_measurement import score_shot, score_week9


def fingerprint(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
    metadata = summary['run']
    assert len(summary['videos']) == metadata['video_count']
    for entry in [metadata['analyzer']] + metadata['tracking_modules'] + metadata['models'] + metadata['inputs']:
        assert fingerprint(Path(entry['path'])) == entry['sha256'], entry['path']
    for entry in ([metadata['capture_conditions']] if metadata.get('capture_conditions') else []) + metadata.get('camera_profiles', []):
        assert fingerprint(Path(entry['path'])) == entry['sha256'], entry['path']
    for name, entry in metadata['calibration'].items():
        path = Path(metadata['working_directory']) / name
        assert path.is_file() == (entry is not None), f'Calibration presence changed: {path}'
        if entry is not None:
            assert fingerprint(path) == entry['sha256'], f'Calibration changed: {path}'
    conditional_errors = []
    source_counts = Counter()
    for row in summary['videos']:
        raw = row['raw_result']
        image_dir = args.run / 'images' / Path(row['video']).parent
        data = json.loads((image_dir / raw['tracking_data']).read_text(encoding='utf-8'))
        frames, trajectory = data['frames'], data['trajectory']
        vm = data['video_metadata']
        assert raw['trajectory'] == trajectory
        assert data['physics']['measurement_status'] == raw['measurement_status']
        assert all(r['frame_number'] == i and abs(r['time_sec'] - i / vm['fps']) < 1e-10 for i, r in enumerate(frames))
        assert all(not r['observed'] and r['confidence'] is None for r in frames if r['source'] == 'kalman_prediction')
        assert all(r['x'] is None and not r['observed'] for r in frames if r['source'] == 'missing')
        assert sum(r['observed'] for r in trajectory) == raw['tracking_summary']['observed_points']
        segment = raw['flight_segment']
        by_frame = {r['frame_number']: r for r in frames}
        for point in raw['trajectory_3d']:
            detection = by_frame[point['frame_number']]
            assert detection['source'] == 'yolo' and detection['confidence'] >= .12
            assert all(math.isfinite(point[k]) for k in ['X_m', 'Y_m', 'Z_m'])
            assert point['Z_m'] > 0
        assert all(segment['start_frame'] <= p['frame_number'] <= segment['end_frame'] for p in raw['fitted_trajectory'])
        automatic = raw.get('analysis_mode') == 'automatic'
        if 'physical_goal' in raw:
            assert data['physical_goal'] == raw['physical_goal']
            assert raw['physical_goal']['accuracy_validated'] is False
            assert raw['physical_goal']['predictions_used'] is False
            options = vm['measurement_context']['options']
            prediction = predict_goal(raw['physical_goal']) if automatic else None
            assert prediction == raw['goal_prediction'] == data['goal_prediction']
            custom_rules = options.get('scoring_rules')
            expected = (score_shot(raw['physical_goal'], data['physics'], custom_rules) if custom_rules is not None
                        else score_week9(raw['physical_goal'], data['physics']))
            assert raw['score_evaluation'] == expected
            assert raw['score'] == expected['value'] and raw['score_status'] == expected['status']
            assert expected.get('accuracy_validated', False) is False
            if not options.get('goal_plane'):
                assert raw['physical_goal']['status'] == 'UNDETERMINED'
                assert raw['physical_goal']['reason'] == 'GOAL_PLANE_REQUIRED'
                assert not raw['physical_goal']['events']
                assert raw['score'] is None
        if 'goal_entry' in raw:
            assessment = raw['goal_entry']
            assert data['goal_entry'] == assessment
            assert assessment['physical_goal_status'] == raw.get('physical_goal', {}).get('status', 'UNDETERMINED')
            assert assessment['rules']['predictions_used'] is False
            assert raw['score_evaluation'] == data['score_evaluation']
            assert raw['score_evaluation']['candidate_used_as_goal'] is False
            if assessment['status'] == 'CANDIDATE_OBSERVED':
                assert assessment['events'] and assessment['camera_motion']['status'] == 'FIXED_CANDIDATE'
            else:
                assert not assessment['events']
            bbox = assessment.get('goal_bbox')
            margin = assessment['rules'].get('inset_margin_px')
            for event in assessment['events']:
                outside = by_frame[event['outside_frame']]
                assert outside['observed'] and outside['source'] == 'yolo' and outside['confidence'] >= .12
                assert all(by_frame[n]['observed'] for n in event['continuity_frames'])
                for number in event['inside_yolo_frames']:
                    observation = by_frame[number]
                    assert observation['source'] == 'yolo' and observation['confidence'] >= .12
                    ball = observation['ball_bbox']
                    assert ball[0] >= bbox[0] + margin and ball[2] <= bbox[2] - margin
                    assert ball[1] >= bbox[1] + margin and ball[3] <= bbox[3] - margin
                assert by_frame[event['first_inside_frame']]['raw_center'] == event['entry_image_position']
            assert cv2.imread(str(image_dir / raw['goal_entry_image'])) is not None
        if raw['measurement_status'] == 'UNAVAILABLE':
            assert raw['speed'] is None and raw['failure_reason']
            assert row['status'] == 'UNAVAILABLE'
        else:
            assert row['status'] == 'ESTIMATED' and math.isfinite(raw['speed']) and raw['speed'] > 0
            assert raw['failure_reason'] is None and raw['camera_motion']['status'] != 'MOVING'
            assert raw['physics_diagnostics']['relative_speed_uncertainty_conditional'] <= .5
            if raw.get('measurement_context'):
                assert raw['measurement_context']['ready_for_metric_estimate']
                actual_fps = raw['measurement_context']['timing']['effective_capture_fps']
                assert actual_fps > 0
                if raw.get('speed_basis') == 'CALIBRATED_INPUTS':
                    assert raw['camera_model']['intrinsics_calibrated']
                assert all(abs(p['time_sec'] - p['frame_number']/actual_fps) < 1e-10 for p in raw['trajectory_3d'])
            conditional_errors.append(raw['physics_diagnostics']['median_reprojection_error_px'])
        for name in [raw['trajectory_image'], raw['observation_image']]:
            assert cv2.imread(str(image_dir / name)) is not None
        cap = cv2.VideoCapture(str(image_dir / raw['tracking_video']))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
        assert abs(cap.get(cv2.CAP_PROP_FPS) - vm['fps']) < .1
        cap.release()
        source_counts.update(r['source'] for r in trajectory)
    audit = {'videos': len(summary['videos']), 'statuses': summary['counts'], 'reasons': summary['failure_counts'],
             'source_counts': dict(source_counts),
             'camera_motion': dict(Counter(r['camera_motion'] for r in summary['videos'])),
             'calibrated_intrinsics_videos': sum(r['raw_result']['camera_model']['intrinsics_calibrated'] for r in summary['videos']),
             'mean_seconds': round(sum(r['elapsed_sec'] for r in summary['videos']) / len(summary['videos']), 3),
             'estimated_fit_median_error_range_px_against_detector': [min(conditional_errors), max(conditional_errors)] if conditional_errors else None,
             'all_file_hashes_match': True, 'geometry_and_media_invariants': 'PASS',
             'calibration_files_match': True,
             'ground_truth_accuracy_evaluated': False}
    if any(r['raw_result'].get('analysis_mode') == 'automatic' for r in summary['videos']):
        audit['automatic_goal_predictions'] = dict(Counter(r['raw_result']['goal_prediction']['status'] for r in summary['videos']))
        audit['automatic_scores_available'] = sum(r['raw_result']['score'] is not None for r in summary['videos'])
    (args.run / 'geometry_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
