"""Video-based reference points and goal presentation for automatic analysis."""
import math

try:
    from .goal_measurement import WEEK9_RULES, goal_crossing_event, outer_position_fraction, speed_score_fraction
except ImportError:
    from goal_measurement import WEEK9_RULES, goal_crossing_event, outer_position_fraction, speed_score_fraction


VIDEO_RULES = {**{key: WEEK9_RULES[key] for key in (
    'speed_min_kmh', 'speed_max_kmh', 'speed_weight', 'accuracy_weight')},
    'id': 'week10_video_v1', 'accuracy_basis': 'image_region_entry_outer_distance'}


def score_video(goal_entry, physics):
    """Score present observations at their original weights, without inferring goal or miss."""
    speed = physics.get('speed')
    has_speed = (physics.get('measurement_status') == 'ESTIMATED'
                 and type(speed) in (int, float) and math.isfinite(speed) and speed > 0)
    events = goal_entry.get('events') or []
    event = events[0] if goal_entry.get('status') == 'CANDIDATE_OBSERVED' and events else {}
    position = event.get('entry_image_position_normalized')
    has_position = (isinstance(position, (list, tuple)) and len(position) == 2
                    and all(type(p) in (float, int) and 0 <= p <= 1 for p in position))
    rules = VIDEO_RULES
    speed_max = 100 * rules['speed_weight']
    position_max = 100 * rules['accuracy_weight']
    speed_points = speed_max * speed_score_fraction(speed, rules) if has_speed else None
    position_points = position_max * outer_position_fraction(*position) if has_position else None
    max_points = speed_max * has_speed + position_max * has_position
    components = {
        'speed_points': round(speed_points, 1) if has_speed else None,
        'speed_kmh': speed if has_speed else None,
        'speed_basis': physics.get('speed_basis') if has_speed else None,
        'position_points': round(position_points, 1) if has_position else None,
        'entry_image_position_normalized': list(position) if has_position else None,
        'entry_frame': event.get('first_inside_frame') if has_position else None,
    }
    result = {'status': 'NOT_EVALUATED', 'value': None, 'rule_set_id': rules['id'],
              'reason': 'VIDEO_SCORE_EVIDENCE_REQUIRED', 'basis': None, 'max_points': max_points,
              'components': components, 'rules': dict(rules), 'partial': False,
              'accuracy_validated': False, 'candidate_used_as_goal': False}
    if not max_points:
        return result
    if has_speed and has_position:
        basis, label = 'VIDEO_SPEED_AND_POSITION', '속도와 화면상 진입 위치'
    elif has_speed:
        basis, label = 'VIDEO_SPEED_ONLY', '속도 항목'
    else:
        basis, label = 'VIDEO_POSITION_ONLY', '화면상 진입 위치 항목'
    comment = f'영상 참고 점수의 계산 근거는 {label}입니다.'
    if has_speed and 'SAMPLE_CAPTURE_REPORT_DIFFERS_FROM_AUTOMATIC_DEFAULTS' in (
            physics.get('measurement_context', {}).get('assumptions', [])):
        comment += ' 기록된 촬영 조건과 속도 계산의 기본 가정이 다릅니다.'
    return {**result, 'status': 'PROVISIONAL', 'reason': 'VIDEO_BASED_REFERENCE_SCORE',
            'value': round((components['speed_points'] or 0) + (components['position_points'] or 0), 1),
            'basis': basis, 'partial': not (has_speed and has_position), 'comment': comment}


def predict_goal(physical):
    """Expose only measured-plane estimates as goal or outside predictions."""
    event = goal_crossing_event(physical)
    result = {'status': 'UNDETERMINED', 'reason': physical.get('reason'), 'evidence_frames': [],
              'basis': None, 'physical_goal_confirmed': False, 'accuracy_validated': False,
              'position_normalized': None, 'warnings': ['NO_GOAL_GROUND_TRUTH']}
    if event is None:
        if physical.get('status') in {'ESTIMATED_GOAL', 'ESTIMATED_OUTSIDE'}:
            result['reason'] = 'GOAL_CROSSING_EVIDENCE_REQUIRED'
        return result
    result.update(status='GOAL_LIKELY' if event['outcome'] == 'ESTIMATED_GOAL' else 'OUTSIDE_LIKELY',
                  reason='MEASURED_PLANE_ESTIMATE', basis='measured_goal_plane',
                  position_normalized=list(event['crossing_normalized']),
                  evidence_frames=[event['before_frame'], event['behind_frame']])
    return result


def automatic_comment(prediction, physics, score):
    labels = {'GOAL_LIKELY': '측정한 골대 면 안으로 통과한 것으로 추정됩니다.',
              'OUTSIDE_LIKELY': '측정한 골대 면의 바깥으로 통과한 것으로 추정됩니다.',
              'UNDETERMINED': '골라인 통과 여부를 판단할 근거가 부족합니다.'}
    comment = labels[prediction['status']]
    if physics.get('speed') is not None:
        comment += f" 초기 비행 속도는 약 {physics['speed']:.1f} km/h입니다."
        if physics.get('speed_basis') == 'ASSUMPTION_BASED':
            comment += ' 촬영 조건 가정에 따른 속도 추정값입니다.'
    if score['value'] is not None:
        if score['rule_set_id'] == VIDEO_RULES['id']:
            comment += f" 영상 참고 점수는 {score['max_points']:g}점 만점에 {score['value']:.1f}점입니다."
        else:
            comment += f" 참고 점수는 {score['value']:.1f}점입니다."
    return comment
