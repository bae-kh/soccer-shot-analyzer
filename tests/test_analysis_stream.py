"""Check the actual SSE route preserves measured/unavailable result contracts."""
import asyncio
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'output/ai_baseline/_tests'
RUNTIME.mkdir(parents=True, exist_ok=True)
(RUNTIME / 'yolo').mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(RUNTIME / 'yolo'))
os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
sys.path.insert(0, str(ROOT / 'backend'))


def request(app, path, method='GET', files=None, data=None, decode=True):
    """Send a real multipart ASGI request without an extra HTTP client dependency."""
    boundary = 'capture-options-test-boundary'
    parts = []
    for name, value in (data or {}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, entry in (files or {}).items():
        filename, content = entry[:2]
        mime = entry[2] if len(entry) > 2 else 'application/octet-stream'
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode() + content + b'\r\n')
    body = b''.join(parts) + (f'--{boundary}--\r\n'.encode() if parts else b'')

    async def run():
        messages, sent = [], False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
                 'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                 'query_string': b'', 'http_version': '1.1', 'client': ('local-test', 1), 'server': ('local-test', 80),
                 'headers': [(b'content-type', f'multipart/form-data; boundary={boundary}'.encode()),
                             (b'content-length', str(len(body)).encode())]}
        await app(scope, receive, send)
        response_body = b''.join(m.get('body', b'') for m in messages)
        return next(m['status'] for m in messages if m['type'] == 'http.response.start'), response_body.decode() if decode else response_body

    return asyncio.run(run())


class AnalysisStreamTests(unittest.TestCase):
    def test_automatic_result_and_numeric_zero_score_survive_default_sse_request(self):
        main = importlib.import_module('main')
        result = {'analysis_mode':'automatic','speed':50.4,'speed_basis':'ASSUMPTION_BASED',
                  'measurement_status':'ESTIMATED','score':0.,'score_status':'PROVISIONAL',
                  'goal_prediction':{'status':'OUTSIDE_LIKELY','physical_goal_confirmed':False},
                  'score_evaluation':{'status':'PROVISIONAL','value':0.,'basis':'OUTSIDE_ESTIMATE'}}
        received = []

        def analyze(video, output, callback, capture_options=None):
            received.append(capture_options)
            return result

        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory, patch.object(main,'UPLOAD_DIR',directory), patch.object(main,'process_video_with_progress',analyze):
            status, body = request(main.app,'/analyze','POST',files={'file':('synthetic-contract.mp4',b'contract-fixture')})
            self.assertEqual(status,200)
            self.assertEqual(received,[None])  # Analyzer applies the automatic default, not route-specific settings.
            events=[json.loads(c.removeprefix('data: ').strip()) for c in body.split('\n\n') if c]
            self.assertEqual(next(e['data'] for e in events if e['type']=='result'), result)

    def test_goal_measurements_and_scoring_are_forwarded_and_invalid_configuration_is_422(self):
        main = importlib.import_module('main')
        from test_goal_measurement import fixture, RULES
        options = {'goal_plane': fixture()[3], 'scoring_rules': RULES}
        received = []

        def analyze(video, output, callback, capture_options=None):
            received.append(capture_options)
            return {'physical_goal': {'status': 'ESTIMATED_GOAL'}, 'score': 80, 'score_status': 'PROVISIONAL'}

        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory, patch.object(main, 'UPLOAD_DIR', directory), patch.object(main, 'process_video_with_progress', analyze):
            status, body = request(main.app, '/analyze', 'POST', files={'file': ('shot.mp4', b'video')},
                                   data={'capture_options': json.dumps(options)})
            self.assertEqual(status, 200)
            self.assertEqual(received, [options])
            self.assertIn('PROVISIONAL', body)
            for changes in [{'confirmed_by_user': False}, {'speed_weight': .9}, {'speed_max_kmh': 0}]:
                bad = {**options, 'scoring_rules': {**RULES, **changes}}
                status, _ = request(main.app, '/analyze', 'POST', files={'file': ('shot.mp4', b'video')},
                                    data={'capture_options': json.dumps(bad)})
                self.assertEqual(status, 422)
            self.assertEqual(len(received), 1)

    def test_http_form_delivers_capture_options_and_rejects_invalid_requests(self):
        main = importlib.import_module('main')
        options = {'timing_mode': 'slow_motion', 'slowdown_factor': 4,
                   'intrinsics_profile_id': 'zoom_out', 'same_camera_settings': True, 'uncropped': True}
        received = []

        def analyze(video_path, image_path, callback, capture_options=None):
            received.append(capture_options)
            self.assertEqual(Path(video_path).read_bytes(), b'video')
            callback(100)
            return {'speed': None, 'measurement_status': 'UNAVAILABLE',
                    'failure_reason': 'CAMERA_PROFILE_NOT_FOUND', 'measurement_context': capture_options}

        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory, patch.object(main, 'UPLOAD_DIR', directory), patch.object(main, 'process_video_with_progress', analyze):
            status, body = request(main.app, '/analyze', 'POST', files={'file': ('../same.mp4', b'video', 'video/mp4')},
                                   data={'capture_options': json.dumps(options)})
            self.assertEqual(status, 200)
            events = [json.loads(chunk.removeprefix('data: ').strip()) for chunk in body.split('\n\n') if chunk]
            result = next(e['data'] for e in events if e['type'] == 'result')
            self.assertEqual(result['measurement_context'], options)
            self.assertEqual(received, [options])
            self.assertTrue(all(p.parent == Path(directory) for p in Path(directory).glob('*')))
            for bad in ('null', '{invalid', '{"intrinsics_profile_id":"../outside"}', '{"timing_mode":"slow_motion","slowdown_factor":0}'):
                status, _ = request(main.app, '/analyze', 'POST', files={'file': ('x.mp4', b'video')}, data={'capture_options': bad})
                self.assertEqual(status, 422)
            self.assertEqual(len(received), 1)

    def test_calibration_creates_independent_profiles_and_list_endpoint(self):
        main = importlib.import_module('main')
        import calibrate
        payload = {'camera_matrix': [[700, 0, 960], [0, 700, 540], [0, 0, 1]],
                   'dist_coeffs': [0, 0, 0, 0, 0], 'resolution': [1920, 1080], 'rms_error': .3}

        def calibration(video, output, *args):
            Path(output).write_text(json.dumps(payload), encoding='utf-8')
            return True

        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
            profiles = Path(directory) / 'profiles'
            with patch.object(main, 'UPLOAD_DIR', directory), patch.object(main, 'PROFILE_DIR', profiles), patch.object(calibrate, 'calibrate_camera', calibration):
                ids = []
                for label in ('normal 1x', 'wide 0.5x'):
                    status, body = request(main.app, '/calibrate', 'POST', files={'file': ('same.mp4', b'video')}, data={'profile_label': label})
                    self.assertEqual(status, 200)
                    events = [json.loads(chunk.removeprefix('data: ').strip()) for chunk in body.split('\n\n') if chunk]
                    result = next(e['data'] for e in events if e['type'] == 'result')
                    self.assertEqual(result['profile_label'], label)
                    ids.append(result['profile_id'])
                self.assertNotEqual(ids[0], ids[1])
                status, body = request(main.app, '/camera-profiles')
                self.assertEqual(status, 200)
                result = json.loads(body)['profiles']
                self.assertEqual({p['label'] for p in result}, {'normal 1x', 'wide 0.5x'})
                self.assertEqual({p['id'] for p in result}, set(ids))
                self.assertFalse((Path(directory) / 'calibration.json').exists())

    def test_sse_keeps_null_speed_and_new_diagnostics(self):
        main = importlib.import_module('main')
        from fastapi import UploadFile

        async def read_response(result, directory):
            def analyze(video_path, image_path, callback):
                self.assertEqual(Path(video_path).read_bytes(), b'uploaded-video')
                self.assertTrue(Path(image_path).name.endswith('_result.jpg'))
                self.assertEqual(Path(image_path).parent, Path(directory))
                callback(100)
                return result

            with patch.object(main, 'UPLOAD_DIR', directory), patch.object(main, 'process_video_with_progress', analyze):
                file = UploadFile(filename='sample.mp4', file=io.BytesIO(b'uploaded-video'))
                response = await main.analyze_stream(file)
                self.assertEqual(response.media_type, 'text/event-stream')
                events = [json.loads(chunk.removeprefix('data: ').strip()) async for chunk in response.body_iterator]
                await file.close()
            return next(event['data'] for event in events if event['type'] == 'result')

        for status, speed, reason in [('ESTIMATED', 81.8, None), ('UNAVAILABLE', None, 'CAMERA_MOVING')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
                result = {'speed': speed, 'score': None, 'score_status': 'NOT_EVALUATED',
                          'measurement_status': status, 'failure_reason': reason,
                          'goal_entry': {'status': 'CANDIDATE_OBSERVED', 'physical_goal_status': 'UNDETERMINED'},
                          'goal_entry_image': 'sample_result_goal_entry.jpg',
                          'score_evaluation': {'status': 'NOT_EVALUATED', 'value': None},
                          'camera_model': {'source': 'assumed_fov', 'is_calibrated': False},
                          'flight_segment': {'start_frame': 20, 'end_frame': 30},
                          'image': 'sample_result.jpg', 'observation_image': 'sample_result_observed.jpg'}
                self.assertEqual(asyncio.run(read_response(result, directory)), result)
