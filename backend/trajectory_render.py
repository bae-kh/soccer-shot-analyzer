"""Draw actual observations and accepted model projections with separate colors."""
import cv2
import numpy as np


def draw_observations(image, trajectory, start_frame=None, end_frame=None):
    last = None
    colors = {'yolo': (0, 220, 0), 'csrt': (0, 180, 255), 'kalman_prediction': (230, 0, 230)}
    for row in trajectory:
        if start_frame is not None and row['frame_number'] < start_frame:
            continue
        if end_frame is not None and row['frame_number'] > end_frame:
            break
        center = row.get('raw_center') if row.get('observed') else None
        if center is None and row['source'] == 'kalman_prediction' and row['x'] is not None:
            center = [row['x'], row['y']]
        if center is None or row['source'] not in colors:
            last = None
            continue
        point = tuple(round(v) for v in center)
        color = colors[row['source']]
        if last is not None and row['frame_number'] == last[0] + 1:
            cv2.line(image, last[1], point, color, 3, cv2.LINE_AA)
        cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)
        last = row['frame_number'], point
    return image


def result_image(frame, trajectory, physics, goal_area=None, observations_only=False):
    image = frame.copy()
    segment = physics['flight_segment']
    draw_observations(image, trajectory, trajectory[0]['frame_number'] if trajectory else None, segment.get('end_frame'))
    if goal_area:
        x1, y1, x2, y2 = map(round, goal_area['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 170, 0), 2)
    if not observations_only and physics['measurement_status'] == 'ESTIMATED':
        fitted = physics['fitted_trajectory']
        for left, right in zip(fitted, fitted[1:]):
            cv2.line(image, (round(left['x']), round(left['y'])), (round(right['x']), round(right['y'])), (255, 255, 0), 2, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(.5, min(1.0, image.shape[1] / 1600))
    if observations_only:
        lines = ['OBSERVATIONS: YOLO green | CSRT orange | PRED magenta',
                 f"EARLY WINDOW: {segment.get('start_frame')} - {segment.get('end_frame')} frames"]
    else:
        if physics['speed'] is None:
            speed_text = f"SPEED UNAVAILABLE: {physics['failure_reason']}"
        else:
            calibration = 'calibrated intrinsics' if physics['camera_model']['intrinsics_calibrated'] else 'ASSUMED CAMERA'
            speed_text = f"EARLY SPEED ESTIMATE: {physics['speed']:.1f} km/h | {calibration}"
        lines = [speed_text, 'YOLO green | CSRT orange | MODEL cyan (observed interval only)',
                 'NO FUTURE EXTRAPOLATION | SPEED ACCURACY NOT VERIFIED']
    for index, line in enumerate(lines):
        location = (20, round((35 + index * 32) * scale))
        cv2.putText(image, line, location, font, .65 * scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, line, location, font, .65 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def goal_entry_image(frame, trajectory, assessment, physical=None, score=None):
    """Whole observed path and evidence on an original frame, with no extrapolation."""
    image = frame.copy()
    draw_observations(image, [r for r in trajectory if r.get('observed')])
    if assessment.get('goal_bbox'):
        x1, y1, x2, y2 = map(round, assessment['goal_bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 170, 0), 2)
    for event in assessment['events']:
        point = tuple(round(v) for v in event['entry_image_position'])
        cv2.circle(image, point, 12, (255, 80, 255), 3, cv2.LINE_AA)
        cv2.putText(image, f"candidate f{event['first_inside_frame']}", (point[0] + 15, point[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 80, 255), 2, cv2.LINE_AA)
    if physical and physical.get('plane'):
        corners = physical['plane']['config']['corners_px']
        cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, (255, 255, 0), 3)
        for event in physical['events']:
            row = next((r for r in trajectory if r['frame_number'] == event['behind_frame']), None)
            if row and row.get('raw_center'):
                point = tuple(round(v) for v in row['raw_center'])
                cv2.circle(image, point, 14, (255, 255, 0), 3, cv2.LINE_AA)
    verdict = physical['status'] if physical else 'UNDETERMINED'
    score_status = score['status'] if score else 'NOT_EVALUATED'
    lines = [f"GOAL REGION: {assessment['status']} | {assessment['reason']}",
             'WHOLE OBSERVED PATH: YOLO green | CSRT orange | NO PREDICTIONS',
             f'PHYSICAL GOAL: {verdict} | SCORE: {score_status} | ACCURACY UNVERIFIED']
    scale = max(.5, min(1., image.shape[1] / 1600))
    for i, line in enumerate(lines):
        location = (15, round((30 + i * 30) * scale))
        cv2.putText(image, line, location, cv2.FONT_HERSHEY_SIMPLEX, .6 * scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, line, location, cv2.FONT_HERSHEY_SIMPLEX, .6 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    return image
