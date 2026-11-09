"""Conditional single-camera goal-plane estimates, never referee decisions.

The measured opening's four inner corners lie on the rear goal-line plane.
Raw observed ball silhouettes supply depth. Early ballistic fits are not used.
Pixel perturbations are sensitivity checks, not calibrated confidence bounds.
"""
import itertools
import math

import cv2
import numpy as np


WEEK9_RULES = {'id': 'week9_outer_v1', 'speed_min_kmh': 20., 'speed_max_kmh': 100.,
               'speed_weight': .4, 'accuracy_weight': .6,
               'accuracy_basis': 'goal_plane_outer_distance', 'outside_score': 0.}


def _number(value, minimum=0, maximum=float('inf')):
    return type(value) in (int, float) and math.isfinite(value) and minimum < value <= maximum


def validate_goal_plane(config):
    if config is None:
        return None
    fields = {'width_m', 'height_m', 'corners_px', 'reference_frame', 'plane_reference',
              'confirmed_by_user', 'pixel_tolerance_px', 'ball_radius_m'}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError('goal_plane requires measured dimensions, four corners and confirmations')
    if not all(_number(config[k], 0, limit) for k, limit in
               [('width_m', 20), ('height_m', 10), ('ball_radius_m', .2), ('pixel_tolerance_px', 5)]):
        raise ValueError('Invalid goal dimensions, ball radius or pixel tolerance')
    if min(config['width_m'], config['height_m']) <= 2 * config['ball_radius_m']:
        raise ValueError('Goal opening must be larger than the ball')
    corners = config['corners_px']
    if (not isinstance(corners, list) or len(corners) != 4 or any(
            not isinstance(p, list) or len(p) != 2 or any(not _number(v, -1) or v < 0 for v in p)
            for p in corners)):
        raise ValueError('corners_px must be TL, TR, BR, BL in original pixels')
    if type(config['reference_frame']) is not int or config['reference_frame'] < 0:
        raise ValueError('reference_frame must be a zero-based nonnegative integer')
    if config['plane_reference'] != 'rear_edge_goal_line' or config['confirmed_by_user'] is not True:
        raise ValueError('Confirm corners on the rear goal-line plane for this video')
    return dict(config)


def validate_scoring_rules(rules):
    if rules is None:
        return None
    fields = {'id', 'confirmed_by_user', 'accept_conditional_estimates', 'speed_min_kmh',
              'speed_max_kmh', 'speed_weight', 'accuracy_weight', 'target_normalized',
              'accuracy_radius_m', 'outside_score'}
    if not isinstance(rules, dict) or set(rules) != fields:
        raise ValueError('Incomplete scoring_rules')
    if not isinstance(rules['id'], str) or not 1 <= len(rules['id']) <= 64:
        raise ValueError('Scoring rule id must contain 1-64 characters')
    if rules['confirmed_by_user'] is not True or rules['accept_conditional_estimates'] is not True:
        raise ValueError('Scoring requires agreed rules and acceptance of conditional estimates')
    for k in ('speed_min_kmh', 'speed_max_kmh', 'speed_weight', 'accuracy_weight', 'outside_score'):
        if not _number(rules[k], -1) or rules[k] < 0:
            raise ValueError(k + ' must be finite and nonnegative')
    if (rules['speed_max_kmh'] <= rules['speed_min_kmh'] or rules['outside_score'] > 100
            or not math.isclose(rules['speed_weight'] + rules['accuracy_weight'], 1., abs_tol=1e-9)
            or not _number(rules['accuracy_radius_m'])):
        raise ValueError('Invalid score range, weights or accuracy radius')
    target = rules['target_normalized']
    if not isinstance(target, list) or len(target) != 2 or any(
            not _number(v, -1, 1) or v < 0 for v in target):
        raise ValueError('Target must be two normalized coordinates between 0 and 1')
    return dict(rules)


def _pose(corners, config, camera, alternatives=False):
    w, h = config['width_m'], config['height_m']
    objects = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=float)
    ok, rotations, translations, errors = cv2.solvePnPGeneric(
        objects, np.asarray(corners, dtype=float), camera.matrix, camera.distortion,
        flags=cv2.SOLVEPNP_IPPE)
    solutions = []
    if ok:
        for rv, tv, error in zip(rotations, translations, errors):
            rotation = cv2.Rodrigues(rv)[0]
            translation = tv.reshape(3)
            if np.min((objects @ rotation.T + translation)[:, 2]) > 0:
                solutions.append((float(error[0]), rotation, translation))
    if not solutions:
        raise ValueError('GOAL_POSE_UNAVAILABLE')
    solutions.sort(key=lambda s: s[0])
    error, rotation, translation = solutions[0]
    if not math.isfinite(error) or error > 2:
        raise ValueError('GOAL_POSE_REPROJECTION_HIGH')
    if alternatives:
        return [(r, t, 1 if np.dot(r[:, 2], t) > 0 else -1, e)
                for e, r, t in solutions if e <= 2 and e - error < .1]
    if len(solutions) > 1 and solutions[1][0] - error < .1:
        if np.linalg.norm(solutions[1][1] - rotation) > .05:
            raise ValueError('GOAL_POSE_AMBIGUOUS')
    sign = 1 if np.dot(rotation[:, 2], translation) > 0 else -1
    return rotation, translation, sign, error


def _local(position, pose):
    rotation, translation, sign, _ = pose
    point = rotation.T @ (position - translation)
    point[2] *= sign  # camera/field side negative; net side positive
    return point


def _usable(row, width, height):
    bbox = row.get('ball_bbox')
    if (row.get('source') != 'yolo' or row.get('observed') is not True
            or not _number(row.get('confidence'), 0, 1) or row['confidence'] < .12
            or not isinstance(bbox, list) or len(bbox) != 4
            or any(not _number(v, -1) for v in bbox)):
        return False
    x1, y1, x2, y2 = bbox
    validation, size = row.get('observation_validation'), row.get('ball_size_quality')
    return (1 < x1 < x2 < width - 1 and 1 < y1 < y2 < height - 1
            and min(x2-x1, y2-y1) >= 5 and max(x2-x1, y2-y1) / min(x2-x1, y2-y1) <= 1.4
            and (validation is None or validation.get('accepted') is True)
            and (size is None or size.get('usable') is True))


def assess_physical_goal(trajectory, metadata, camera, summary, motion, config=None):
    result = {'status': 'UNDETERMINED', 'reason': 'GOAL_PLANE_REQUIRED', 'events': [],
              'method': 'observed_sphere_depth_and_measured_goal_plane',
              'accuracy_validated': False, 'predictions_used': False,
              'warnings': ['SINGLE_CAMERA_DEPTH_IS_APPROXIMATE', 'PIXEL_SENSITIVITY_IS_NOT_A_CONFIDENCE_INTERVAL',
                           'ADJACENT_FRAME_LINEAR_PATH_ASSUMED', 'OFFENCES_NOT_ASSESSED']}
    if config is None:
        return result

    def held(reason):
        result['reason'] = reason
        return result

    validate_goal_plane(config)
    if not summary.get('kick_detected'):
        return held('KICK_NOT_DETECTED')
    calibration = metadata.get('measurement_context', {}).get('calibration', {})
    if not camera.calibrated or calibration.get('problems') or not (
            calibration.get('same_camera_settings') and calibration.get('uncropped')):
        return held('GOAL_CALIBRATION_REQUIRED')
    if motion.get('status') != 'FIXED_CANDIDATE':
        return held('CAMERA_MOVING' if motion.get('status') == 'MOVING' else 'CAMERA_MOTION_UNVERIFIED')
    width, height = metadata['width'], metadata['height']
    corners = np.asarray(config['corners_px'], dtype=float)
    if (config['reference_frame'] >= metadata['frame_count'] or np.any(corners < 0)
            or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height)
            or not cv2.isContourConvex(corners.astype(np.float32))
            or cv2.contourArea(corners.astype(np.float32), oriented=True) < 100):
        return held('INVALID_GOAL_PLANE_GEOMETRY')
    numbers = [r.get('frame_number') for r in trajectory]
    if any(type(n) is not int or n < 0 for n in numbers) or any(a >= b for a, b in zip(numbers, numbers[1:])):
        return held('INVALID_OBSERVATION_SEQUENCE')
    try:
        pose = _pose(corners, config, camera)
    except (ValueError, cv2.error) as exc:
        return held(str(exc) if isinstance(exc, ValueError) else 'GOAL_POSE_UNAVAILABLE')
    result['plane'] = {'rotation': pose[0].tolist(), 'translation_m': pose[1].tolist(),
                       'behind_sign': pose[2], 'reprojection_error_px': pose[3], 'config': config}
    radius, epsilon = config['ball_radius_m'], config['pixel_tolerance_px']
    # Test independent +/- pixel edge perturbations and each goal-corner perturbation.
    variants = [pose]
    try:
        for index in range(4):
            for axis in range(2):
                for delta in (-epsilon, epsilon):
                    perturbed = corners.copy()
                    perturbed[index, axis] += delta
                    variants.extend(_pose(perturbed, config, camera, alternatives=True))
    except (ValueError, cv2.error):
        return held('GOAL_POSE_SENSITIVITY_UNSTABLE')
    previous, chain = None, []
    eligible = 0
    uncertain_crossing = False
    for row in trajectory:
        if not _usable(row, width, height):
            previous, chain = None, []
            continue
        try:
            point = camera.sphere_position(row['ball_bbox'], radius)[0]
            local = _local(point, pose)
            sensitivity = max(np.linalg.norm(_local(point, variant) - local) for variant in variants)
            maximum_edge_error = 0.
            for shifts in itertools.product((-epsilon, epsilon), repeat=4):
                box = np.array(row['ball_bbox']) + shifts
                if min(box[2]-box[0], box[3]-box[1]) < 3:
                    raise ValueError('Small perturbed box')
                perturbed = camera.sphere_position(box, radius)[0]
                # Sum bounds to cover ball-edge and corner perturbations together.
                edge_error = np.linalg.norm(_local(perturbed, pose) - local)
                maximum_edge_error = max(maximum_edge_error, edge_error)
            sensitivity += maximum_edge_error
        except (ValueError, cv2.error, FloatingPointError):
            previous, chain = None, []
            continue
        eligible += 1
        if previous is None or row['frame_number'] != previous[0] + 1 or local[2] < previous[1][2]:
            chain = []
        if local[2] + sensitivity < -radius:
            chain = [(row['frame_number'], local, sensitivity)]
        elif chain:
            chain.append((row['frame_number'], local, sensitivity))
            if local[2] >= 0:
                uncertain_crossing = True
            # Every intervening frame must be a usable YOLO observation. Evaluate
            # each segment through the sphere/plane overlap slab, including slow crossings.
            if local[2] - sensitivity > radius:
                slab, crossing = [], None
                for left, right in zip(chain, chain[1:]):
                    before, after = left[1], right[1]
                    dz = after[2] - before[2]
                    if after[2] < -radius or before[2] > radius:
                        continue
                    if dz == 0:
                        slab.extend((before, after))
                        continue
                    for depth in (max(-radius, before[2]), min(radius, after[2])):
                        slab.append(before + ((depth-before[2])/dz)*(after-before))
                    if crossing is None and before[2] <= 0 <= after[2]:
                        crossing = before + (-before[2]/dz)*(after-before)
                tolerance = max(p[2] for p in chain)
                w, h = config['width_m'], config['height_m']
                margin = radius + tolerance
                inside = bool(slab) and all(margin < p[0] < w-margin and margin < p[1] < h-margin for p in slab)
                outside = bool(slab) and (all(p[0]+margin < 0 for p in slab)
                                          or all(p[0]-margin > w for p in slab)
                                          or all(p[1]+margin < 0 for p in slab))
                if crossing is not None and (inside or outside):
                    result['events'].append({'before_frame': chain[0][0], 'behind_frame': row['frame_number'],
                        'continuity_frames': [p[0] for p in chain],
                        'crossing_position_m': crossing.tolist(),
                        'crossing_normalized': [float(crossing[0]/w), float(crossing[1]/h)],
                        'before_signed_distance_m': float(chain[0][1][2]), 'behind_signed_distance_m': float(local[2]),
                        'sensitivity_m': float(tolerance), 'ball_radius_m': radius,
                        'outcome': 'ESTIMATED_GOAL' if inside else 'ESTIMATED_OUTSIDE'})
                    result.update(status=result['events'][0]['outcome'], reason='CONDITIONAL_OBSERVED_PLANE_CROSSING')
                    break
        previous = (row['frame_number'], local, sensitivity)
    result['eligible_observations'] = eligible
    if not result['events']:
        result['reason'] = 'GOAL_BOUNDARY_OR_DEPTH_UNCERTAIN' if uncertain_crossing else 'FULL_BALL_CROSSING_NOT_OBSERVED'
    return result


def goal_crossing_event(physical):
    """Return a usable crossing from the physical assessment, never a 2D candidate."""
    status = physical.get('status')
    if status not in {'ESTIMATED_GOAL', 'ESTIMATED_OUTSIDE'} or not physical.get('events'):
        return None
    event = physical['events'][0]
    if not isinstance(event, dict) or event.get('outcome') != status:
        return None
    before, behind = event.get('before_frame'), event.get('behind_frame')
    position = event.get('crossing_normalized')
    if (type(before) is not int or type(behind) is not int or not 0 <= before < behind
            or not isinstance(position, (list, tuple)) or len(position) != 2
            or not all(type(v) in (int, float) and math.isfinite(v) for v in position)):
        return None
    u, v = position
    in_opening = 0 < u < 1 and 0 < v < 1
    if (status == 'ESTIMATED_GOAL' and not in_opening
            or status == 'ESTIMATED_OUTSIDE' and not (u < 0 or u > 1 or v < 0)):
        return None
    return event


def score_week9(physical, physics):
    """Default score: a measured crossing and both required inputs, or no total."""
    result = {'status': 'NOT_EVALUATED', 'value': None, 'rule_set_id': WEEK9_RULES['id'],
              'reason': None, 'basis': None, 'components': {}, 'rules': dict(WEEK9_RULES),
              'accuracy_validated': False, 'candidate_used_as_goal': False,
              'prerequisites': {'scoring_rules_defined': True, 'physical_goal_confirmed': False}}

    def held(reason):
        return {**result, 'reason': reason}

    if physical.get('status') not in {'ESTIMATED_GOAL', 'ESTIMATED_OUTSIDE'}:
        return held('GOAL_PLANE_REQUIRED' if physical.get('reason') == 'GOAL_PLANE_REQUIRED'
                    else 'PHYSICAL_GOAL_ESTIMATE_REQUIRED')
    event = goal_crossing_event(physical)
    if event is None:
        return held('GOAL_CROSSING_EVIDENCE_REQUIRED')
    rules = WEEK9_RULES
    if physical['status'] == 'ESTIMATED_OUTSIDE':
        value, basis, components = rules['outside_score'], 'MEASURED_PLANE_OUTSIDE', {'outside_score': rules['outside_score']}
    else:
        speed = physics.get('speed')
        if physics.get('measurement_status') != 'ESTIMATED' or not _number(speed):
            return held('SPEED_ESTIMATE_REQUIRED')
        timing = physics.get('measurement_context', {}).get('timing', {})
        if physics.get('speed_basis') != 'CALIBRATED_INPUTS' or timing.get('status') != 'CONFIRMED_BY_USER':
            return held('SPEED_BASIS_UNVERIFIED')
        goal_radius = event.get('ball_radius_m')
        speed_radius = physics.get('physics_diagnostics', {}).get('ball_radius_m')
        if not (_number(goal_radius) and _number(speed_radius) and math.isclose(goal_radius, speed_radius, abs_tol=1e-9)):
            return held('BALL_RADIUS_ASSUMPTION_MISMATCH')
        u, v = event['crossing_normalized']
        speed01 = min(1., max(0., (speed-rules['speed_min_kmh']) / (rules['speed_max_kmh']-rules['speed_min_kmh'])))
        position01 = math.hypot(u-.5, v-.5)/math.sqrt(.5)
        value = 100*(rules['speed_weight']*speed01 + rules['accuracy_weight']*position01)
        basis = 'MEASURED_PLANE_OUTER_DISTANCE'
        components = {'speed_normalized': speed01, 'position_normalized': position01,
                      'crossing_normalized': [u, v], 'speed_kmh': speed}
    return {**result, 'status': 'PROVISIONAL', 'value': round(value, 1), 'reason': 'WEEK9_CONDITIONAL_SCORE',
            'basis': basis, 'components': components, 'partial': False,
            'comment': '측정한 골대 면의 통과 추정으로 계산한 참고 점수입니다. 실측 정확도는 미검증입니다.'}


def score_shot(physical, physics, rules=None):
    result = {'status': 'NOT_EVALUATED', 'value': None, 'rule_set_id': None,
              'reason': 'SCORING_RULES_AND_CONFIRMED_GOAL_REQUIRED', 'candidate_used_as_goal': False,
              'prerequisites': {'scoring_rules_defined': rules is not None, 'physical_goal_confirmed': False}}
    if rules is None:
        return result
    validate_scoring_rules(rules)
    result['rule_set_id'] = rules['id']
    if physical['status'] not in {'ESTIMATED_GOAL', 'ESTIMATED_OUTSIDE'}:
        result['reason'] = 'PHYSICAL_GOAL_ESTIMATE_REQUIRED'
        return result
    if not physical.get('events'):
        result['reason'] = 'GOAL_CROSSING_EVIDENCE_REQUIRED'
        return result
    speed = physics.get('speed')
    needs_speed = physical['status'] == 'ESTIMATED_GOAL' and rules['speed_weight'] > 0
    if needs_speed and (physics.get('measurement_status') != 'ESTIMATED' or not _number(speed)):
        result['reason'] = 'SPEED_ESTIMATE_REQUIRED'
        return result
    if needs_speed and not math.isclose(physical['plane']['config']['ball_radius_m'],
            physics.get('physics_diagnostics', {}).get('ball_radius_m', .11), abs_tol=1e-9):
        result['reason'] = 'BALL_RADIUS_ASSUMPTION_MISMATCH'
        return result
    if physical['status'] == 'ESTIMATED_OUTSIDE':
        value, components = rules['outside_score'], {'outside_score': rules['outside_score']}
    else:
        speed_score = (float(np.clip((speed-rules['speed_min_kmh']) / (rules['speed_max_kmh']-rules['speed_min_kmh']), 0, 1))
                       if needs_speed else None)
        position = physical['events'][0]['crossing_position_m']
        config = physical['plane']['config']
        target = [rules['target_normalized'][0]*config['width_m'], rules['target_normalized'][1]*config['height_m']]
        distance = math.dist(position[:2], target)
        accuracy_score = max(0., 1-distance/rules['accuracy_radius_m'])
        value = 100*(rules['speed_weight']*(speed_score or 0) + rules['accuracy_weight']*accuracy_score)
        components = {'speed_normalized': speed_score, 'accuracy_normalized': accuracy_score,
                      'target_distance_m': distance, 'speed_kmh': speed}
    result.update(status='PROVISIONAL', value=round(value, 1), reason='CONDITIONAL_ESTIMATES_WITH_USER_RULES',
                  components=components, rules=rules, accuracy_validated=False,
                  basis='USER_DEFINED_OUTSIDE' if physical['status'] == 'ESTIMATED_OUTSIDE' else 'USER_DEFINED_TARGET',
                  comment='골대 면 통과·속도 추정과 선택한 기준으로 계산한 참고 점수입니다. 실측 정확도는 미검증입니다.')
    return result
