"""Run a real multipart analyze/SSE request and fetch its result image locally."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
sys.path.insert(0, str(ROOT/'backend'))
from test_analysis_stream import request
import main as service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['automatic','calibrated'], default='automatic')
    parser.add_argument('--video', type=Path, default=ROOT/'soccer_videos/test_videos/video11.mp4',
                        help='A video with an image entry candidate and no goal-plane inputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.video
    with patch.object(service, 'UPLOAD_DIR', str(args.output.resolve())):
        data = None if args.mode == 'automatic' else {'capture_options':json.dumps({'analysis_mode':'calibrated','timing_mode':'unknown'})}
        status, body = request(service.app, '/analyze', 'POST', files={'file': (source.name, source.read_bytes(), 'video/mp4')}, data=data)
        assert status == 200
        events = [json.loads(chunk.removeprefix('data: ').strip()) for chunk in body.split('\n\n') if chunk and not chunk.startswith(':')]
        assert not any(e['type'] == 'error' for e in events), events
        result = next(e['data'] for e in events if e['type'] == 'result')
        if args.mode == 'automatic':
            assert result['analysis_mode'] == 'automatic' and result['measurement_status'] == 'ESTIMATED'
            assert result['speed'] > 0 and result['speed_basis'] == 'ASSUMPTION_BASED'
            assert result['goal_prediction']['status'] == 'UNDETERMINED'
            assert result['goal_prediction']['reason'] == 'GOAL_PLANE_REQUIRED'
        else:
            assert result['speed'] is None and result['measurement_status'] == 'UNAVAILABLE'
        assert result['score'] is None and result['score_status'] == 'NOT_EVALUATED'
        assert result['score_evaluation']['value'] is None
        assert result['score_evaluation']['rule_set_id'] == 'week9_outer_v1'
        assert result['score_evaluation']['reason'] == 'GOAL_PLANE_REQUIRED'
        assert result['goal_entry']['status'] == 'CANDIDATE_OBSERVED'
        assert result['goal_entry']['physical_goal_status'] == 'UNDETERMINED'
        assert result['physical_goal']['status'] == 'UNDETERMINED'
        assert result['physical_goal']['reason'] == 'GOAL_PLANE_REQUIRED'
        saved = json.loads((args.output/result['tracking_data']).read_text(encoding='utf-8'))
        for key in ('goal_entry', 'physical_goal', 'goal_prediction', 'score_evaluation'):
            assert result[key] == saved[key]
        status, image_body = request(service.app, '/results/'+result['goal_entry_image'], decode=False)
        assert status == 200 and len(image_body) > 1000 and image_body.startswith(b'\xff\xd8')
    artifact = {'method': 'actual_multipart_asgi_sse_and_yolo_csrt', 'models_mocked': False,
                'video': str(source.resolve()),
                'result_image_get_status': status, 'result': result,
                'progress_events': sum(e['type'] == 'progress' for e in events)}
    assert artifact['progress_events'] > 0
    (args.output/'api_validation.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in artifact.items() if k!='result'},indent=2))


if __name__ == '__main__':
    main()
