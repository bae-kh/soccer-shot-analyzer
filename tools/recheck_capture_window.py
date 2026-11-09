"""Recompute current physics on saved real-video observations; does not rerun YOLO."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.capture_context import build_context
from backend.shot_physics import estimate_shot
from tools.run_ai_baseline import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.source_run/'summary.json').read_text(encoding='utf-8'))
    metadata = summary['run']
    assert len(summary['videos']) == 32
    for entry in [metadata['analyzer'], metadata['capture_conditions']] + metadata['inputs'] + metadata['models']:
        assert fingerprint(Path(entry['path']))['sha256'] == entry['sha256'], entry['path']
    changed = []
    for entry in metadata['tracking_modules']:
        if fingerprint(Path(entry['path']))['sha256'] != entry['sha256']:
            changed.append(Path(entry['path']).name)
    assert changed == ['shot_physics.py'], changed
    records = []
    for video in summary['videos']:
        raw = video['raw_result']
        file = args.source_run/'images'/Path(video['video']).parent/raw['tracking_data']
        data = json.loads(file.read_text(encoding='utf-8'))
        vm = data['video_metadata']
        camera, context = build_context(Path(metadata['input_directory'])/video['video'],
                                        vm['width'], vm['height'], vm['fps'], vm['fps_fallback'])
        assert context == raw['measurement_context']
        result = estimate_shot(data['trajectory'], {**vm, 'measurement_context': context}, camera, raw['camera_motion'])
        assert (result['measurement_status'], result['failure_reason'], result['speed']) == (raw['measurement_status'], raw['failure_reason'], raw['speed'])
        before = raw['flight_segment']['selected_frame_numbers']
        after = result['flight_segment']['selected_frame_numbers']
        assert set(after).issubset(before)
        start = result['flight_segment']['tracking_start_frame']
        assert all((frame-start)/vm['fps'] <= .45 + 1e-9 for frame in after)
        records.append({'video': video['video'], 'status': result['measurement_status'], 'speed': result['speed'],
                        'failure_reason': result['failure_reason'], 'before_selected_frames': before,
                        'after_selected_frames': after, 'removed_frames': sorted(set(before)-set(after)),
                        'source_tracking_sha256': fingerprint(file)['sha256']})
    output = {'source_run': str(args.source_run.resolve()), 'method': 'physics_recompute_on_saved_observations',
              'yolo_csrt_rerun': False, 'videos': len(records), 'measurement_decisions_unchanged': 32,
              'changed_modules_since_source': changed,
              'current_physics_sha256': fingerprint(ROOT/'backend/shot_physics.py')['sha256'],
              'selected_frames_changed_videos': sum(bool(row['removed_frames']) for row in records),
              'removed_metric_frames': sum(len(row['removed_frames']) for row in records),
              'failure_counts': dict(Counter(row['failure_reason'] for row in records)),
              'field_accuracy_verified': False, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'records'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
