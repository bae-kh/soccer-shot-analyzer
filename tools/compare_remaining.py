"""Assert additive measurement/scoring work preserves all sample observations."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--previous', type=Path, default=ROOT/'output/ai_baseline/followup_tracking_camera_20261011')
    args = parser.parse_args()
    current = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    previous = json.loads((args.previous/'summary.json').read_text(encoding='utf-8'))
    assert current['run']['inputs'] == previous['run']['inputs']
    assert current['run']['models'] == previous['run']['models']
    old = {r['video']: r for r in previous['videos']}
    assert len(current['videos']) == len(old) == 32
    checks = []
    for row in current['videos']:
        key, raw = row['video'], row['raw_result']
        before = old[key]['raw_result']
        for field in ['trajectory', 'tracking_summary', 'flight_segment', 'measurement_status', 'speed', 'failure_reason']:
            assert raw[field] == before[field], (key, field)
        assert raw['goal_entry'] == before['goal_entry'], (key, 'goal_entry')
        assert raw['score'] is None and raw['score_status'] == 'NOT_EVALUATED'
        assert raw['physical_goal']['status'] == 'UNDETERMINED'
        assert raw['physical_goal']['reason'] == 'GOAL_PLANE_REQUIRED'
        assert raw['physical_goal']['accuracy_validated'] is False
        for run, result in [(args.run, raw), (args.previous, before)]:
            track = json.loads((run/'images'/Path(key).parent/result['tracking_data']).read_text(encoding='utf-8'))
            if run == args.previous:
                old_frames = track['frames']
            else:
                new_frames = track['frames']
        assert new_frames == old_frames, (key, 'all_frame_records')
        checks.append({'video': key, 'all_frame_records_identical': True,
                       'initial_flight_identical': True, 'goal_candidate_identical': True,
                       'physical_goal': raw['physical_goal']['status'], 'score': raw['score']})
    report = {'videos_checked': len(checks), 'changed_observation_frames': 0,
              'goal_candidates': dict(Counter(r['raw_result']['goal_entry']['status'] for r in current['videos'])),
              'camera_motion': dict(Counter(r['raw_result']['goal_entry']['camera_motion']['status'] for r in current['videos'])),
              'speed_failures': current['failure_counts'], 'checks': checks}
    (args.run/'remaining_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
