"""Check rerun artifacts against the pre-week-9 batch and replay stored observations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.automatic_analysis import predict_goal
from backend.goal_entry import assess_goal_entry
from backend.goal_measurement import score_week9


GROUPS = ('real_test_videos', 'test_videos')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def batch(root, group):
    batch_root = root / group
    summary = read(batch_root / 'summary.json')
    rows = {r['video']: r for r in summary['videos']}
    assert len(rows) == summary['run']['video_count'], group
    hashes = {Path(item['path']).name: item['sha256'] for item in summary['run']['inputs']}
    return batch_root, rows, hashes, summary['run']['models']


def score_is_held(physical, prediction, score):
    return (physical['status'] == prediction['status'] == 'UNDETERMINED'
            and score['status'] == 'NOT_EVALUATED' and score['value'] is None
            and score['rule_set_id'] == 'week9_outer_v1' and score['reason'] == 'GOAL_PLANE_REQUIRED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    args = parser.parse_args()
    rerun, replay = [], []
    for group in GROUPS:
        old_dir, old_rows, old_hashes, old_models = batch(args.before, group)
        new_dir, new_rows, new_hashes, new_models = batch(args.after, group)
        assert old_models == new_models, group
        for name, row in new_rows.items():
            label = f'{group}/{name}'
            assert new_hashes[name] == old_hashes[name], label
            assert row['image_generated'] and row['status'] in {'ESTIMATED', 'UNAVAILABLE'}, label
            raw, before = row['raw_result'], old_rows[name]['raw_result']
            assert score_is_held(raw['physical_goal'], raw['goal_prediction'], raw['score_evaluation']), label
            assert raw['score'] is None and raw['score_status'] == 'NOT_EVALUATED', label
            for key in ('trajectory', 'speed', 'failure_reason', 'measurement_status', 'speed_basis',
                        'flight_segment', 'goal_entry'):
                assert raw[key] == before[key], (label, key)
            new_saved = read(new_dir/'images'/raw['tracking_data'])
            old_saved = read(old_dir/'images'/before['tracking_data'])
            assert new_saved['frames'] == old_saved['frames'], label
            for key in ('goal_entry', 'physical_goal', 'goal_prediction', 'score_evaluation'):
                assert new_saved[key] == raw[key], (label, key)
            rerun.append({'video': label, 'speed': raw['speed'], 'candidate': raw['goal_entry']['status'],
                          'previous_score': before['score'], 'score': raw['score']})
        for name in sorted(old_rows.keys() - new_rows.keys()):
            raw = old_rows[name]['raw_result']
            saved = read(old_dir/'images'/raw['tracking_data'])
            physical = saved['physical_goal']
            image = assess_goal_entry(saved['trajectory'], saved['goal_area'], saved['video_metadata'],
                                      saved['tracking_summary'], saved['goal_entry']['camera_motion'])
            assert image['status'] == saved['goal_entry']['status'] and image['events'] == saved['goal_entry']['events'], name
            assert score_is_held(physical, predict_goal(physical), score_week9(physical, saved['physics'])), name
            replay.append({'video': f'{group}/{name}', 'candidate': image['status'], 'score': None})

    test11 = next((row for row in rerun if row['video'] == 'test_videos/video11.mp4'), None)
    if test11 is not None:
        assert test11['candidate'] == 'CANDIDATE_OBSERVED' and test11['previous_score'] == 51.2 and test11['score'] is None
    report = {'rerun_video_count': len(rerun), 'stored_observation_replay_count': len(replay),
              'unchanged_tracking_and_speed': len(rerun), 'speed_estimates': sum(r['speed'] is not None for r in rerun),
              'image_entry_statuses': dict(Counter(r['candidate'] for r in rerun)),
              'previous_scores_available_for_reruns': sum(r['previous_score'] is not None for r in rerun),
              'current_scores_available_for_reruns': sum(r['score'] is not None for r in rerun),
              'accuracy_validated': False, 'rerun': rerun, 'replay': replay}
    (args.after/'regression.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in {'rerun', 'replay'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
