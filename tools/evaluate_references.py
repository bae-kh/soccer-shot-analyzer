"""Create a reference ledger or evaluate independently measured, matched shots.

Unlabelled/deferred shots count in coverage, never as zero-error measurements.
Real and synthetic references are aggregated separately. No label enters AI.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def compare_reference(raw, reference):
    row = {'speed': {'status': 'UNLABELLED'}, 'goal': {'status': 'UNLABELLED'}}
    speed = reference.get('speed')
    if speed is not None:
        valid = (isinstance(speed, dict) and _positive(speed.get('value_kmh'))
                 and speed.get('method') in {'radar_window_mean', 'independent_3d_tracking'}
                 and speed.get('quantity') == 'initial_flight_mean_speed'
                 and speed.get('independent') is True and isinstance(speed.get('evidence'), str) and bool(speed['evidence'].strip())
                 and isinstance(speed.get('frame_interval'), list) and len(speed['frame_interval']) == 2
                 and all(type(n) is int and n >= 0 for n in speed['frame_interval'])
                 and speed['frame_interval'][0] < speed['frame_interval'][1])
        if not valid:
            row['speed'] = {'status': 'INVALID_REFERENCE'}
        elif raw.get('measurement_status') != 'ESTIMATED' or not _positive(raw.get('speed')):
            row['speed'] = {'status': 'AI_UNAVAILABLE', 'reason': raw.get('failure_reason')}
        elif speed['frame_interval'] != [raw.get('flight_segment', {}).get('start_frame'),
                                         raw.get('flight_segment', {}).get('end_frame')]:
            row['speed'] = {'status': 'WINDOW_MISMATCH'}
        else:
            error = raw['speed'] - speed['value_kmh']
            row['speed'] = {'status': 'COMPARED', 'signed_error_kmh': error,
                            'absolute_error_kmh': abs(error), 'relative_error_percent': abs(error)/speed['value_kmh']*100}
    goal = reference.get('goal')
    if goal is not None:
        valid = (isinstance(goal, dict) and type(goal.get('is_goal')) is bool
                 and goal.get('method') == 'independent_goal_line_view'
                 and goal.get('independent') is True and isinstance(goal.get('evidence'), str) and bool(goal['evidence'].strip()))
        status = raw.get('physical_goal', {}).get('status', 'UNDETERMINED')
        if not valid:
            row['goal'] = {'status': 'INVALID_REFERENCE'}
        elif status not in {'ESTIMATED_GOAL', 'ESTIMATED_OUTSIDE'}:
            row['goal'] = {'status': 'AI_UNAVAILABLE', 'reason': raw.get('physical_goal', {}).get('reason', 'NO_PHYSICAL_ESTIMATE')}
        else:
            predicted = status == 'ESTIMATED_GOAL'
            row['goal'] = {'status': 'COMPARED', 'truth': goal['is_goal'], 'predicted': predicted,
                           'correct': predicted == goal['is_goal']}
    return row


def aggregate(rows):
    speed = [r['speed'] for r in rows if r['speed']['status'] == 'COMPARED']
    goal = [r['goal'] for r in rows if r['goal']['status'] == 'COMPARED']
    count = len(rows)
    confusion = Counter(('TP' if r['truth'] else 'FP') if r['predicted'] else ('FN' if r['truth'] else 'TN') for r in goal)
    result = {'shot_count': count,
              'speed_statuses': dict(Counter(r['speed']['status'] for r in rows)),
              'goal_statuses': dict(Counter(r['goal']['status'] for r in rows)),
              'speed_comparison_coverage': len(speed)/count if count else None,
              'goal_comparison_coverage': len(goal)/count if count else None,
              'speed_mae_kmh': sum(r['absolute_error_kmh'] for r in speed)/len(speed) if speed else None,
              'speed_bias_kmh': sum(r['signed_error_kmh'] for r in speed)/len(speed) if speed else None,
              'speed_mape_percent': sum(r['relative_error_percent'] for r in speed)/len(speed) if speed else None,
              'goal_accuracy': sum(r['correct'] for r in goal)/len(goal) if goal else None,
              'goal_confusion': {k: confusion[k] for k in ['TP', 'FP', 'TN', 'FN']},
              'accuracy_validated': False}
    # A numeric metric alone does not establish a project's acceptance threshold.
    result['validation_status'] = 'REFERENCE_COMPARISONS_AVAILABLE' if speed or goal else 'NO_COMPARABLE_REFERENCES'
    return result


def make_ledger(run, domain='real'):
    run = Path(run).resolve()
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    conditions = json.loads((ROOT/'docs/video_capture_conditions.json').read_text(encoding='utf-8'))
    files = {r['video']: r for r in summary['videos']}
    if len(files) != len(summary['videos']):
        raise ValueError('Duplicate video in run summary')
    saved_records = {}
    for path in (run/'results').rglob('*.json'):
        saved_records[json.loads(path.read_text(encoding='utf-8')).get('video')] = path
    entries = []
    input_dir = Path(summary['run']['input_directory']).resolve()
    original_inputs = {str(Path(p['path']).resolve()): p['sha256'] for p in summary['run']['inputs']}
    for key, item in files.items():
        raw = item.get('raw_result', {})
        source = (input_dir/key).resolve()
        source.relative_to(input_dir)
        record = saved_records.get(key)
        if record is None or not source.is_file():
            raise ValueError('Missing saved result/input for ' + key)
        if original_inputs.get(str(source)) != digest(source):
            raise ValueError('Input changed since AI analysis for ' + key)
        if json.loads(record.read_text(encoding='utf-8')) != item:
            raise ValueError('Saved result does not match run summary for ' + key)
        group = next((r['condition'] for r in conditions['reports'] if key in r['videos']), 'unverified_capture')
        entries.append({'video': key, 'domain': domain, 'group': group,
                        'input_path': str(source.resolve()), 'input_sha256': digest(source),
                        'result_path': str(record), 'result_sha256': digest(record),
                        'ai_speed_window': [raw.get('flight_segment', {}).get('start_frame'),
                                            raw.get('flight_segment', {}).get('end_frame')],
                        'reference': {'speed': None, 'goal': None}})
    return {'schema_version': 1, 'notes': 'Null means no independent reference; do not copy AI outputs into reference values.',
            'run': str(run), 'shots': entries}


def evaluate_ledger(ledger):
    rows, seen = [], set()
    for entry in ledger['shots']:
        key = (entry['domain'], entry['video'])
        if key in seen or entry['domain'] not in {'real', 'synthetic'}:
            raise ValueError('Duplicate shot or unknown reference domain')
        seen.add(key)
        source, record = Path(entry['input_path']), Path(entry['result_path'])
        if digest(source) != entry['input_sha256'] or digest(record) != entry['result_sha256']:
            raise ValueError('Input/result hash mismatch for ' + entry['video'])
        saved = json.loads(record.read_text(encoding='utf-8'))
        if saved.get('video') != entry['video']:
            raise ValueError('Saved result video mismatch')
        raw = saved.get('raw_result', {})
        rows.append({'video': entry['video'], 'domain': entry['domain'], 'group': entry['group'],
                     **compare_reference(raw, entry['reference'])})
    domains, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        domains[row['domain']].append(row)
        groups[row['domain'] + '/' + row['group']].append(row)
    return {'schema_version': 1, 'method': 'independent_reference_and_saved_ai_result_comparison',
            'domains': {k: aggregate(v) for k, v in domains.items()},
            'groups': {k: aggregate(v) for k, v in groups.items()}, 'shots': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Create a ledger from a completed baseline run')
    parser.add_argument('--ledger', type=Path, help='Evaluate an existing reference ledger')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.run) == bool(args.ledger):
        parser.error('Choose exactly one of --run or --ledger')
    result = make_ledger(args.run) if args.run else evaluate_ledger(json.loads(args.ledger.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output), 'domains': result.get('domains'),
                      'shots': len(result['shots'])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
