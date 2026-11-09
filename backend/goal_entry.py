"""Observed image-region entry candidates, never a physical goal verdict."""
import math


def _finite(values, size):
    return isinstance(values, (list, tuple)) and len(values) == size and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def goal_camera_interval(trajectory, goal_area, tracked_end):
    """Cover both the reference goal frame and all assessed ball observations."""
    reference = goal_area['frame_number'] if goal_area else 0
    first = trajectory[0]['frame_number'] if trajectory else 0
    return min(reference, first), max(reference, tracked_end)


def assess_goal_entry(trajectory, goal_area, metadata, tracking_summary, camera_motion=None):
    """Use contiguous raw observations; predictions and fitted paths are not inputs.

    Two trusted YOLO observations inside the inset rectangle must follow an
    observed outside point in the same uninterrupted observation chain. CSRT
    can maintain continuity, but cannot establish outside/inside evidence alone.
    Frame continuity deliberately uses no assumed capture time or speed.
    """
    result = {'status': 'UNAVAILABLE', 'reason': None, 'physical_goal_status': 'UNDETERMINED',
              'scope': 'image_plane_detected_goal_rectangle', 'coordinate_space': 'original_pixels',
              'origin': 'top_left', 'events': [], 'evidence_frames': [],
              'camera_motion': camera_motion or {'status': 'UNVERIFIED'},
              'warnings': ['GOAL_RECTANGLE_IS_APPROXIMATE', 'DEPTH_AND_GOAL_PLANE_NOT_VERIFIED'],
              'rules': {'minimum_inside_yolo_observations': 2, 'minimum_yolo_confidence': .12,
                        'continuity': 'consecutive_observed_frames', 'predictions_used': False}}

    def unavailable(reason):
        result['reason'] = reason
        return result

    if not goal_area:
        return unavailable('GOAL_NOT_FOUND')
    bbox = goal_area.get('bbox')
    width, height = metadata.get('width', 0), metadata.get('height', 0)
    if (not _finite(bbox, 4) or not 0 <= bbox[0] < bbox[2] <= width
            or not 0 <= bbox[1] < bbox[3] <= height
            or goal_area.get('coordinate_space') != 'original_pixels'
            or goal_area.get('origin') != 'top_left'):
        return unavailable('INVALID_GOAL_GEOMETRY')
    if not _finite([goal_area.get('confidence')], 1) or goal_area['confidence'] < .1:
        return unavailable('GOAL_CONFIDENCE_LOW')
    result['goal_bbox'] = list(bbox)
    margin = max(2., (bbox[3] - bbox[1]) * .01)
    result['rules']['inset_margin_px'] = margin
    if not tracking_summary.get('kick_detected'):
        return unavailable('KICK_NOT_DETECTED')
    if result['camera_motion']['status'] == 'MOVING':
        return unavailable('CAMERA_MOVING')
    if result['camera_motion']['status'] != 'FIXED_CANDIDATE':
        return unavailable('CAMERA_MOTION_UNVERIFIED')
    numbers = [r.get('frame_number') for r in trajectory]
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in numbers) or any(
            a >= b for a, b in zip(numbers, numbers[1:])):
        return unavailable('INVALID_OBSERVATION_SEQUENCE')
    previous, outside, inside = None, None, []
    eligible_count, overlap_count, unanchored_inside = 0, 0, False
    for row in trajectory:
        frame = row['frame_number']
        ball, center = row.get('ball_bbox'), row.get('raw_center')
        yolo = row.get('source') == 'yolo' and _finite([row.get('confidence')], 1) and row['confidence'] >= .12
        validation = row.get('observation_validation')
        valid = (row.get('observed') is True and (yolo or row.get('source') == 'csrt')
                 and (validation is None or validation.get('accepted') is True)
                 and _finite(ball, 4) and _finite(center, 2)
                 and 0 <= ball[0] < ball[2] <= width and 0 <= ball[1] < ball[3] <= height
                 and ball[0] <= center[0] <= ball[2] and ball[1] <= center[1] <= ball[3])
        if not valid:
            previous, outside, inside = None, None, []
            continue
        if previous is None or frame != previous + 1:
            outside, inside = None, []
        previous = frame
        eligible_count += int(yolo)
        overlap = ball[2] >= bbox[0] and ball[0] <= bbox[2] and ball[3] >= bbox[1] and ball[1] <= bbox[3]
        fully_inside = (ball[0] >= bbox[0] + margin and ball[2] <= bbox[2] - margin
                        and ball[1] >= bbox[1] + margin and ball[3] <= bbox[3] - margin)
        overlap_count += int(overlap and yolo)
        if not overlap:
            if yolo:
                outside = row
            inside = []
            continue
        if not fully_inside:
            inside = []
            continue
        if outside is None:
            unanchored_inside = unanchored_inside or bool(yolo)
            continue
        inside.append(row)
        trusted = [r for r in inside if r['source'] == 'yolo' and r['confidence'] >= .12]
        if len(trusted) < 2:
            continue
        travel = math.dist(outside['raw_center'], trusted[-1]['raw_center'])
        if travel < max(3., (outside['ball_bbox'][2] - outside['ball_bbox'][0]) * .25):
            continue
        entry = trusted[0]
        result['events'].append({'outside_frame': outside['frame_number'],
                                 'outside_center': list(outside['raw_center']),
                                 'first_inside_frame': entry['frame_number'],
                                 'confirmation_frame': trusted[1]['frame_number'],
                                 'inside_yolo_frames': [r['frame_number'] for r in trusted[:2]],
                                 'continuity_frames': list(range(outside['frame_number'], trusted[1]['frame_number'] + 1)),
                                 'entry_image_position': list(entry['raw_center']),
                                 'entry_image_position_normalized': [(entry['raw_center'][0] - bbox[0]) / (bbox[2] - bbox[0]),
                                                                     (entry['raw_center'][1] - bbox[1]) / (bbox[3] - bbox[1])]})
        # One event per inside run. A new outside YOLO point can start another.
        outside, inside = None, []
    result['trusted_yolo_observations'] = eligible_count
    result['overlapping_yolo_observations'] = overlap_count
    if result['events']:
        result.update(status='CANDIDATE_OBSERVED', reason='OBSERVED_IMAGE_REGION_ENTRY')
        result['evidence_frames'] = sorted({n for e in result['events']
                                            for n in [e['outside_frame']] + e['inside_yolo_frames']})
    elif eligible_count < 3:
        return unavailable('INSUFFICIENT_GOAL_OBSERVATIONS')
    elif unanchored_inside:
        return unavailable('ENTRY_CONTINUITY_UNVERIFIED')
    elif overlap_count:
        return unavailable('BORDER_OR_BRIEF_OVERLAP')
    else:
        result.update(status='NO_CANDIDATE_OBSERVED', reason='NO_OBSERVED_REGION_ENTRY')
    return result


def evaluate_score(goal_entry):
    """Explicit prerequisites, without invented weights or candidate-as-goal scoring."""
    return {'status': 'NOT_EVALUATED', 'value': None, 'rule_set_id': None,
            'reason': 'SCORING_RULES_AND_CONFIRMED_GOAL_REQUIRED',
            'prerequisites': {'scoring_rules_defined': False,
                              'physical_goal_confirmed': goal_entry['physical_goal_status'] == 'CONFIRMED'},
            'candidate_used_as_goal': False}
