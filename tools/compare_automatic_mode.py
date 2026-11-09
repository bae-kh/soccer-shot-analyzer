"""Compare automatic estimates with prior strict-mode tracking, not ground truth."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--previous', type=Path, default=ROOT/'output/ai_baseline/remaining_goal_speed_final_20261011')
    args = parser.parse_args()
    current = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    previous = json.loads((args.previous/'summary.json').read_text(encoding='utf-8'))
    assert current['run']['inputs'] == previous['run']['inputs']
    assert current['run']['models'] == previous['run']['models']
    old = {r['video']:r for r in previous['videos']}
    assert len(current['videos']) == len(old) == 32
    checks = []
    for row in current['videos']:
        key, raw = row['video'], row['raw_result']
        before = old[key]['raw_result']
        assert raw['analysis_mode'] == 'automatic'
        assert raw['trajectory'] == before['trajectory'], (key, 'trajectory')
        for field in ['start_frame','end_frame','point_count','selected_frame_numbers']:
            assert raw['flight_segment'].get(field) == before['flight_segment'].get(field), (key,field)
        assert raw['goal_entry']['events'] == before['goal_entry']['events'], (key,'goal_events')
        assert raw['goal_entry']['status'] == before['goal_entry']['status'], (key,'goal_status')
        paths=[args.run/'images'/Path(key).parent/raw['tracking_data'],
               args.previous/'images'/Path(key).parent/before['tracking_data']]
        frames=[json.loads(p.read_text(encoding='utf-8'))['frames'] for p in paths]
        assert frames[0] == frames[1], (key,'frame_records')
        checks.append({'video':key,'all_observations_identical':True,'same_flight_selection':True,
                       'speed_kmh':raw['speed'],'goal_prediction':raw['goal_prediction']['status'],
                       'score':raw['score'],'score_basis':raw['score_evaluation'].get('basis')})
    report = {'videos_checked':len(checks),'changed_tracking_frames':0,'source_tracking_preserved':True,
              'speed_estimates':sum(c['speed_kmh'] is not None for c in checks),
              'goal_predictions':dict(Counter(c['goal_prediction'] for c in checks)),
              'scores_available':sum(c['score'] is not None for c in checks),
              'score_bases':dict(Counter(c['score_basis'] for c in checks)),
              'failure_reasons':current['failure_counts'],'accuracy_validated':False,'checks':checks}
    (args.run/'automatic_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'},indent=2))


if __name__=='__main__':
    main()
