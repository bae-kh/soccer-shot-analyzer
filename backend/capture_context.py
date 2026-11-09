"""Explicit capture timing and lens/zoom profiles; never infer capture FPS from playback."""
import json
import math
from pathlib import Path
import re
try:
    from .goal_measurement import validate_goal_plane, validate_scoring_rules
except ImportError:
    from goal_measurement import validate_goal_plane, validate_scoring_rules

try:
    from .camera_geometry import CameraModel, load_camera
except ImportError:
    from camera_geometry import CameraModel, load_camera

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS_PATH = ROOT / 'docs/video_capture_conditions.json'
PROFILE_DIR = ROOT / 'backend/camera_profiles'
FIELDS = {'timing_mode', 'slowdown_factor', 'capture_fps', 'one_frame_per_capture',
          'intrinsics_profile_id', 'same_camera_settings', 'uncropped', 'goal_plane', 'scoring_rules', 'analysis_mode'}
DEFAULT_AUTOMATIC_OPTIONS = {'analysis_mode': 'automatic', 'timing_mode': 'normal'}


def validate_options(options):
    if not isinstance(options, dict) or set(options) - FIELDS:
        raise ValueError('Unknown capture option or non-object capture_options')
    result = dict(options)
    if result.get('analysis_mode', 'calibrated') not in {'automatic', 'calibrated'}:
        raise ValueError('analysis_mode must be automatic or calibrated')
    for name, validator in [('goal_plane', validate_goal_plane), ('scoring_rules', validate_scoring_rules)]:
        if name in result:
            result[name] = validator(result[name])
    mode = result.get('timing_mode', 'unknown')
    if mode not in {'unknown', 'normal', 'slow_motion'}:
        raise ValueError('timing_mode must be unknown, normal or slow_motion')
    for name in ('same_camera_settings', 'uncropped', 'one_frame_per_capture'):
        if name in result and type(result[name]) is not bool:
            raise ValueError(name + ' must be a boolean')
    for name in ('capture_fps', 'slowdown_factor'):
        value = result.get(name)
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive and finite')
            if name == 'slowdown_factor' and value < 1:
                raise ValueError('slowdown_factor must be at least 1')
    if mode != 'slow_motion' and any(result.get(k) is not None for k in ('capture_fps', 'slowdown_factor')):
        raise ValueError('Capture FPS/slowdown factor requires slow_motion mode')
    profile_id = result.get('intrinsics_profile_id')
    if profile_id is not None and not (isinstance(profile_id, str) and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', profile_id)):
        raise ValueError('Invalid intrinsics_profile_id')
    return result


def options_for_video(video_path, overrides=None, conditions_path=None):
    """Only repository-relative sample paths match reports, never an upload's basename."""
    options, report = {}, None
    try:
        key = Path(video_path).resolve().relative_to((ROOT / 'soccer_videos').resolve()).as_posix()
    except ValueError:
        key = None
    if key:
        data = json.loads(Path(conditions_path or CONDITIONS_PATH).read_text(encoding='utf-8'))
        report = next((r for r in data['reports'] if key in r['videos']), None)
        if report:
            options['timing_mode'] = 'slow_motion' if report['condition'] == 'slow_motion' else 'unknown'
            for field in FIELDS - {'timing_mode'}:
                if field in report:
                    options[field] = report[field]
        options.update(data.get('video_options', {}).get(key, {}))
    if overrides is not None:
        options.update(validate_options(overrides))
    return validate_options(options), {'sample_key': key, 'reported_condition': report,
                                       'explicit_overrides': overrides is not None}


def resolve_timing(file_fps, options, fps_fallback=False):
    """Uniform time only. Capture FPS is usable iff each output frame is a capture frame."""
    mode = options.get('timing_mode', 'unknown')
    actual_fps, basis = None, 'unverified'
    if mode == 'normal' and not fps_fallback:
        actual_fps, basis = file_fps, 'user_confirmed_normal_constant_frame_rate'
    if mode == 'slow_motion':
        factor, capture = options.get('slowdown_factor'), options.get('capture_fps')
        if factor is not None and not fps_fallback:
            actual_fps, basis = file_fps * factor, 'user_provided_uniform_slowdown_factor'
        if capture is not None and options.get('one_frame_per_capture'):
            if actual_fps is not None and not math.isclose(actual_fps, capture, rel_tol=.005):
                raise ValueError('Capture FPS and slowdown factor disagree')
            actual_fps, basis = capture, 'user_confirmed_one_output_frame_per_capture_frame'
    if actual_fps is not None and (not math.isfinite(actual_fps) or actual_fps <= 0):
        raise ValueError('Invalid effective capture FPS')
    return {'status': 'CONFIRMED_BY_USER' if actual_fps is not None else 'UNVERIFIED',
            'mode': mode, 'file_fps': file_fps, 'effective_capture_fps': actual_fps,
            'time_basis': basis, 'file_fps_fallback': fps_fallback,
            'assumption': 'uniform_frame_intervals; no_variable_speed_edits_or_duplicate_frames'}


def camera_for_options(width, height, options, profile_dir=None):
    profile_id = options.get('intrinsics_profile_id')
    problems = []
    camera = CameraModel.assumed(width, height, warnings=['CAMERA_PROFILE_REQUIRED'])
    if not profile_id:
        if options.get('analysis_mode') == 'automatic':
            camera = CameraModel.assumed(width, height, warnings=['NO_ZOOM_ASSUMED', 'UNCALIBRATED_AUTOMATIC_ESTIMATE'])
        else:
            problems.append('CAMERA_PROFILE_REQUIRED')
    else:
        path = Path(profile_dir or PROFILE_DIR) / (profile_id + '.json')
        if not path.is_file():
            problems.append('CAMERA_PROFILE_NOT_FOUND')
        else:
            camera = load_camera(width, height, path)
            if not camera.calibrated:
                problems.append('CALIBRATION_INVALID')
            else:
                original = json.loads(path.read_text(encoding='utf-8'))
                resolution = original.get('resolution') or [original['img_width'], original['img_height']]
                if abs(width / height / (resolution[0] / resolution[1]) - 1) > .005:
                    problems.append('CAMERA_PROFILE_ASPECT_MISMATCH')
        if not options.get('same_camera_settings'):
            problems.append('CAMERA_SETTINGS_UNCONFIRMED')
        if not options.get('uncropped'):
            problems.append('UNCROPPED_IMAGE_UNCONFIRMED')
    return camera, {'intrinsics_profile_id': profile_id, 'problems': problems,
                    'same_camera_settings': options.get('same_camera_settings', False),
                    'uncropped': options.get('uncropped', False)}


def build_context(video_path, width, height, file_fps, fps_fallback=False, overrides=None):
    options, source = options_for_video(video_path, overrides)
    timing = resolve_timing(file_fps, options, fps_fallback)
    automatic = options.get('analysis_mode') == 'automatic'
    assumptions = []
    if automatic and options.get('timing_mode') == 'normal' and timing['effective_capture_fps'] is not None:
        timing.update(status='ASSUMED_NORMAL', time_basis='normal_speed_assumed_from_file_fps')
        assumptions.append('NORMAL_SPEED_NO_SLOW_MOTION_ASSUMED')
    camera, calibration = camera_for_options(width, height, options)
    if automatic and not options.get('intrinsics_profile_id'):
        assumptions += ['NO_ZOOM_ASSUMED', 'HORIZONTAL_FOV_60_DEG_ASSUMED', 'BALL_RADIUS_0_11_M_ASSUMED']
    if automatic and source.get('reported_condition'):
        condition = source['reported_condition']['condition']
        if (condition == 'slow_motion' and options.get('timing_mode') == 'normal') or condition in {
                'possibly_different_camera_or_zoom', 'zoomed_out'}:
            assumptions.append('SAMPLE_CAPTURE_REPORT_DIFFERS_FROM_AUTOMATIC_DEFAULTS')
    problems = ([] if timing['effective_capture_fps'] is not None else ['CAPTURE_TIMING_UNVERIFIED']) + calibration['problems']
    return camera, {'timing': timing, 'calibration': calibration, 'source': source,
                    'options': options, 'problems': problems, 'ready_for_metric_estimate': not problems,
                    'analysis_mode': 'automatic' if automatic else 'calibrated', 'assumptions': assumptions,
                    'accuracy_validated': False}


def list_profiles(profile_dir=None):
    profiles = []
    for path in sorted(Path(profile_dir or PROFILE_DIR).glob('*.json')):
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', path.stem):
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            resolution = data.get('resolution') or [data['img_width'], data['img_height']]
            if load_camera(*resolution, path).calibrated:
                profiles.append({'id': path.stem, 'label': data.get('profile_label', path.stem),
                                 'resolution': resolution, 'rms_error': data.get('rms_error')})
        except (ValueError, KeyError, TypeError, OSError):
            continue
    return profiles
