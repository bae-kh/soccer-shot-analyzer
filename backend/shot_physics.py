"""Conservative early-flight metric estimates; no forced ballistic extrapolation."""
import math

import numpy as np
from scipy.optimize import least_squares

try:
    from .camera_geometry import CameraModel
except ImportError:
    from camera_geometry import CameraModel

BALL_RADIUS_M = 0.11


def _diameter(row):
    box = row['ball_bbox']
    return min(box[2]-box[0], box[3]-box[1])


def _metric_rejection(row, width, height):
    if row['source'] != 'yolo' or (row.get('confidence') or 0) < .12 or row.get('ball_bbox') is None:
        return 'NO_CONFIDENT_YOLO_SIZE'
    bbox = row['ball_bbox']
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    if min(w, h) < 5 or min(bbox[0], bbox[1]) <= 1 or bbox[2] >= width-1 or bbox[3] >= height-1:
        return 'SMALL_OR_CLIPPED_BALL_BOX'
    if max(w, h)/min(w, h) > 1.4:
        return 'ELONGATED_BALL_BOX'
    quality = row.get('ball_size_quality')
    if quality is not None and not quality['usable']:
        return quality['reason']
    return None


def _flight_onset(candidates, anchor, anchor_diameter, fps):
    """Require departure plus two consistent motion intervals, not one jitter."""
    previous, pending = None, None
    for row in candidates:
        if previous is None:
            previous = row
            continue
        gap = row['frame_number']-previous['frame_number']
        delta = np.array(row['raw_center'])-previous['raw_center']
        radial = math.log(_diameter(row)/_diameter(previous))
        distance = np.linalg.norm(np.array(row['raw_center'])-anchor)
        position_departed = distance > max(6, anchor_diameter*.6)
        radial_departed = distance < anchor_diameter*.2 and abs(math.log(_diameter(row)/anchor_diameter)) > .14
        positional = np.linalg.norm(delta)/gap > max(1.5, anchor_diameter*.06)
        position_signal = positional and position_departed
        depth_motion = abs(radial)/gap > .02 and radial_departed and not positional
        departed = position_departed or radial_departed
        if gap/fps > .13:
            pending = None
        elif position_signal or depth_motion:
            consistent = (pending is not None and (row['frame_number']-pending[0])/fps <= .13
                          and ((position_signal and pending[3] and np.dot(delta, pending[1]) > 0)
                               or (depth_motion and radial*pending[2] > 0)))
            if consistent:
                return pending[0]
            pending = (row['frame_number'], delta, radial, position_signal)
        elif not departed or (pending is not None and (row['frame_number']-pending[0])/fps > .13):
            pending = None
        previous = row
    return None


def select_flight(trajectory, fps, width, height):
    """Select conservative size-supported flight; retain tracking independently."""
    if not trajectory:
        return [], {'reason': 'NO_SHOT', 'start_frame': None, 'end_frame': None, 'point_count': 0,
                    'excluded_observations': [], 'motion_confirmed': False}
    start = trajectory[0]['frame_number']
    window = [r for r in trajectory if r['frame_number']-start <= math.floor(fps*.45 + 1e-9)]
    candidates, excluded = [], []
    for row in window:
        rejection = _metric_rejection(row, width, height)
        if rejection:
            excluded.append({'frame_number': row['frame_number'], 'reason': rejection})
        else:
            candidates.append(row)
    anchor = trajectory[0].get('raw_center') or [trajectory[0].get('x'), trajectory[0].get('y')]
    if anchor[0] is None and candidates:
        anchor = candidates[0]['raw_center']
    anchor_diameter = (_diameter(trajectory[0]) if trajectory[0].get('ball_bbox')
                       else (_diameter(candidates[0]) if candidates else 10))
    onset = _flight_onset(candidates, np.array(anchor, dtype=float), anchor_diameter, fps) if candidates else None
    selected, directions, last = [], [], None
    reason = 'EARLY_WINDOW_LIMIT' if onset is not None else 'NO_CONFIRMED_FLIGHT'
    candidate_numbers = {r['frame_number'] for r in candidates}
    for row in window:
        if row['frame_number'] not in candidate_numbers:
            if last is not None and (row['frame_number']-last['frame_number'])/fps > .13:
                reason = 'OBSERVATION_GAP'
                break
            continue
        if onset is None or row['frame_number'] < onset:
            excluded.append({'frame_number': row['frame_number'], 'reason': 'PRE_FLIGHT_OR_UNCONFIRMED_MOTION'})
            continue
        center = np.array(row['raw_center'], dtype=float)
        if last is not None:
            dt = (row['frame_number'] - last['frame_number']) / fps
            if dt > .13:
                reason = 'OBSERVATION_GAP'
                break
            velocity = (center - np.array(last['raw_center'])) / dt
            travel = np.linalg.norm(velocity) / fps
            if travel > math.hypot(width, height) * .15:
                reason = 'POSITION_JUMP'
                break
            if len(directions) >= 2 and travel > 3:
                reference = np.median(directions[-3:], axis=0)
                if np.dot(reference, velocity) < -.2 * np.linalg.norm(reference) * np.linalg.norm(velocity):
                    reason = 'POSSIBLE_CONTACT_OR_DIRECTION_CHANGE'
                    break
            if travel > 3:
                directions.append(velocity)
        selected.append(row)
        last = row
    return selected, {'reason': reason, 'start_frame': selected[0]['frame_number'] if selected else None,
                      'end_frame': selected[-1]['frame_number'] if selected else None,
                      'point_count': len(selected), 'scope': 'early_flight_candidate_not_confirmed_collision',
                      'max_window_sec': .45, 'tracking_start_frame': start, 'motion_confirmed': onset is not None,
                      'selected_frame_numbers': [r['frame_number'] for r in selected],
                      'excluded_observations': sorted(excluded, key=lambda r:r['frame_number']),
                      'size_quality_policy': 'elongation_and_optional_original_foreground; no_manual_video_exceptions'}


def _fit(points, times, confidence, radii, camera, measured, angles):
    times = times - times[0]
    design = np.column_stack([np.ones(len(times)), times])
    # Bounding-box size noise dominates range; this uncertainty is conditional,
    # not a calibrated confidence score or a ground-truth speed error.
    sigma = np.maximum(.03, np.linalg.norm(points, axis=1) * np.maximum(.5, radii * .08) / radii)
    weights = np.sqrt(confidence) / sigma
    gravity = camera.gravity
    initial = np.linalg.lstsq(design * weights[:, None],
                              (points - .5 * gravity * times[:, None] ** 2) * weights[:, None], rcond=None)[0].reshape(-1)
    lower = np.array([-120, -120, .2, -100, -100, -100], dtype=float)
    upper = np.array([120, 120, 120, 100, 100, 100], dtype=float)
    initial = np.clip(initial, lower + 1e-6, upper - 1e-6)
    pixel_sigma = np.maximum(1.5, radii * .08)
    size_sigma = .10 + .15 * (1 - confidence)
    def model(params):
        return params[:3] + times[:, None] * params[3:] + .5 * gravity * times[:, None] ** 2
    def residual(params):
        fitted = model(params)
        safe = fitted.copy()
        safe[:, 2] = np.maximum(.2, safe[:, 2])
        pixels = camera.project(safe)
        predicted_size = np.arcsin(np.clip(BALL_RADIUS_M / np.linalg.norm(safe, axis=1), 1e-8, .99))
        xy_error = ((pixels - measured) * (np.sqrt(confidence) / pixel_sigma)[:, None]).reshape(-1)
        size_error = np.log(predicted_size / angles) * np.sqrt(confidence) / size_sigma
        depth_penalty = np.maximum(0, .2 - fitted[:, 2]) * 10
        return np.concatenate((xy_error, size_error, depth_penalty))
    fit = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1', f_scale=1, max_nfev=300)
    fitted = model(fit.x)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    velocity = fit.x[3:] + gravity * times[-1] / 2
    speed = float(np.mean(np.linalg.norm(fit.x[3:] + np.linspace(0, times[-1], 31)[:, None] * gravity, axis=1)))
    direction = velocity / max(speed, 1e-8)
    sigma_speed = math.sqrt(max(0.0, direction @ covariance[3:, 3:] @ direction))
    return fit, fitted, speed * 3.6, sigma_speed * 3.6


def estimate_shot(trajectory, metadata, camera, camera_motion=None):
    context = metadata.get('measurement_context')
    physical_fps = context['timing']['effective_capture_fps'] if context else None
    if physical_fps is not None:
        trajectory = [{**row, 'file_time_sec': row['time_sec'],
                       'time_sec': row['frame_number'] / physical_fps} for row in trajectory]
    selected, segment = select_flight(trajectory, physical_fps or metadata['fps'], metadata['width'], metadata['height'])
    segment['time_basis'] = ('physical_time_from_capture_settings' if physical_fps is not None
                             else 'file_playback_time; physical_timing_unverified')
    if context and context.get('timing', {}).get('status') == 'ASSUMED_NORMAL':
        segment['time_basis'] = 'normal_speed_assumed_from_file_fps'
    motion = camera_motion or {'status': 'UNVERIFIED'}
    warnings = list(camera.warnings) + ['BALL_BBOX_SIZE_APPROXIMATION', 'FIXED_CAMERA_REQUIRED', 'NO_GROUND_TRUTH_SPEED']
    if camera.gravity_source == 'assumed_camera_down':
        warnings.append('GRAVITY_DIRECTION_ASSUMED_CAMERA_DOWN')
    if motion['status'] == 'UNVERIFIED':
        warnings.append('CAMERA_MOTION_UNVERIFIED')
    result = {'measurement_status': 'UNAVAILABLE', 'failure_reason': None, 'speed': None,
              'speed_definition': 'early_flight_window_average_3d_speed', 'flight_segment': segment,
              'trajectory_3d': [], 'fitted_trajectory': [], 'camera_model': camera.describe(),
              'camera_motion': motion, 'physics_diagnostics': {'warnings': warnings,
              'model': 'robust_short_window_ballistic_pixel_and_size_fit', 'ball_radius_m': BALL_RADIUS_M}}
    if context is not None:
        result['measurement_context'] = context
        warnings.extend(context['problems'])
        warnings.extend(context.get('assumptions', []))
    result['speed_basis'] = ('ASSUMPTION_BASED' if not camera.calibrated or (
        context and context.get('timing', {}).get('status') == 'ASSUMED_NORMAL') else 'CALIBRATED_INPUTS')
    if motion['status'] == 'MOVING':
        result['failure_reason'] = 'CAMERA_MOVING'
        return result
    if context is not None and context['problems']:
        result['failure_reason'] = context['problems'][0]
        return result
    if 'CALIBRATION_INVALID' in camera.warnings:
        result['failure_reason'] = 'CALIBRATION_INVALID'
        return result
    if len(selected) < 5:
        result['failure_reason'] = 'INSUFFICIENT_FLIGHT_OBSERVATIONS'
        return result
    times = np.array([r['time_sec'] for r in selected])
    duration = times[-1] - times[0]
    if duration < .1 or np.any(np.diff(times) <= 0):
        result['failure_reason'] = 'FLIGHT_INTERVAL_TOO_SHORT'
        return result
    points, angles = [], []
    for row in selected:
        point, angle = camera.sphere_position(row['ball_bbox'], BALL_RADIUS_M)
        points.append(point)
        angles.append(angle)
        result['trajectory_3d'].append({'frame_number': row['frame_number'], 'time_sec': row['time_sec'],
                                       'X_m': float(point[0]), 'Y_m': float(point[1]), 'Z_m': float(point[2]),
                                       'source': 'yolo_ball_size_reconstruction', 'estimated': True})
    points, angles = np.array(points), np.array(angles)
    radii = np.array([min(r['ball_bbox'][2] - r['ball_bbox'][0], r['ball_bbox'][3] - r['ball_bbox'][1]) / 2 for r in selected])
    confidence = np.array([r['confidence'] for r in selected])
    if not np.isfinite(points).all() or np.any(points[:, 2] < .2) or np.any(points[:, 2] > 120):
        result['failure_reason'] = 'INVALID_DEPTH_ESTIMATE'
        return result
    measured = np.array([r['raw_center'] for r in selected])
    fit, fitted, speed, sigma_speed = _fit(points, times, confidence, radii, camera, measured, angles)
    projected = camera.project(fitted)
    image_errors = np.linalg.norm(projected - measured, axis=1)
    predicted_angles = np.arcsin(np.clip(BALL_RADIUS_M / np.linalg.norm(fitted, axis=1), 0, .99))
    size_errors = np.abs(np.log(predicted_angles / angles))
    median_error, p90_error = float(np.median(image_errors)), float(np.percentile(image_errors, 90))
    relative_uncertainty = 2 * sigma_speed / max(speed, 1e-8)
    diagnostics = result['physics_diagnostics']
    diagnostics.update({'duration_sec': float(duration), 'flight_points': len(selected),
                        'median_reprojection_error_px': median_error, 'p90_reprojection_error_px': p90_error,
                        'median_log_size_error': float(np.median(size_errors)),
                        'relative_speed_uncertainty_conditional': relative_uncertainty,
                        'candidate_speed_kmh': float(speed), 'conditional_speed_sigma_kmh': float(sigma_speed),
                        'velocity_at_interval_start_mps': fit.x[3:].tolist(),
                        'position_at_interval_start_m': fit.x[:3].tolist(),
                        'gravity_camera_mps2': camera.gravity.tolist()})
    for row, point, pixel in zip(selected, fitted, projected):
        result['fitted_trajectory'].append({'frame_number': row['frame_number'], 'time_sec': row['time_sec'],
                                            'x': float(pixel[0]), 'y': float(pixel[1]),
                                            'X_m': float(point[0]), 'Y_m': float(point[1]), 'Z_m': float(point[2])})
    radius = float(np.median(radii))
    if not fit.success or np.any(fit.active_mask) or np.any(fitted[:, 2] <= .2) or not np.isfinite(speed) or speed <= 1:
        result['failure_reason'] = 'UNSTABLE_3D_FIT'
    elif median_error > max(3, radius * .4) or p90_error > max(6, radius * .9):
        result['failure_reason'] = 'REPROJECTION_MISMATCH'
    elif np.median(size_errors) > .15 or np.percentile(size_errors, 90) > .35:
        result['failure_reason'] = 'BALL_SIZE_INCONSISTENT'
    elif relative_uncertainty > .5:
        result['failure_reason'] = 'SPEED_UNCERTAINTY_HIGH'
    else:
        result['measurement_status'] = 'ESTIMATED'
        result['speed'] = round(speed, 1)
        if not camera.calibrated:
            bounds = []
            for fov in (50, 70):
                alternate = CameraModel.assumed(camera.width, camera.height, fov)
                alternate.gravity = camera.gravity.copy()
                reconstructed = [alternate.sphere_position(r['ball_bbox']) for r in selected]
                alternate_points = np.array([r[0] for r in reconstructed])
                alternate_angles = np.array([r[1] for r in reconstructed])
                bounds.append(_fit(alternate_points, times, confidence, radii, alternate, measured, alternate_angles)[2])
            diagnostics['fov_50_70_deg_speed_sensitivity_kmh'] = [round(min(bounds), 1), round(max(bounds), 1)]
            diagnostics['warnings'].append('INTRINSICS_ASSUMED_SPEED_NOT_CALIBRATED')
    return result
