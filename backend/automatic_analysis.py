"""Compatibility presentation for the automatic analysis mode."""
try:
    from .goal_measurement import goal_crossing_event
except ImportError:
    from goal_measurement import goal_crossing_event


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
        comment += f" 참고 점수는 {score['value']:.1f}점입니다."
    return comment
