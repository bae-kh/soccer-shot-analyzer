import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
from ball_tracking import BallTracker, select_initial_ball
from ball_size_quality import BallSizeInspector
from goal_geometry import assemble_goal
from camera_geometry import load_camera, assess_camera_motion
from shot_physics import estimate_shot
from capture_context import build_context, DEFAULT_AUTOMATIC_OPTIONS
from trajectory_render import result_image, goal_entry_image
from goal_entry import assess_goal_entry, goal_camera_interval
from goal_measurement import assess_physical_goal, score_shot, score_week9
from automatic_analysis import predict_goal, automatic_comment


def _goal_outputs(goal_entry, physical, physics, context):
    goal_entry['physical_goal_status'] = physical['status']
    prediction = predict_goal(physical) if context['analysis_mode'] == 'automatic' else None
    custom_rules = context['options'].get('scoring_rules')
    score = score_week9(physical, physics) if custom_rules is None else score_shot(physical, physics, custom_rules)
    return prediction, score

class HybridTracker:
    def __init__(self):
        self.csrt = None
        self.is_tracking = False
        self.missed_yolo = 0
        
    def init(self, frame, u, v, w, h):
        height, width = frame.shape[:2]
        x = int(max(0, min(width, u - w / 2)))
        y = int(max(0, min(height, v - h / 2)))
        right = int(max(0, min(width, u + w / 2)))
        bottom = int(max(0, min(height, v + h / 2)))
        w, h = right - x, bottom - y
        if w < 2 or h < 2:
            self.is_tracking = False
            return
        self.csrt = cv2.TrackerCSRT_create()
        self.csrt.init(frame, (x, y, w, h))
        self.is_tracking = True
        self.missed_yolo = 0
        
    def update(self, frame, yolo_bbox=None):
        if not self.is_tracking: return None
        
        if yolo_bbox is not None:
             u, v, w, h = yolo_bbox
             self.init(frame, u, v, w, h)
             self.missed_yolo = 0
             return yolo_bbox
             
        self.missed_yolo += 1
        if self.missed_yolo > 15:
             self.is_tracking = False
             return None
             
        success, bbox = self.csrt.update(frame)
        if success:
             x, y, w, h = [float(v) for v in bbox]
             return (x + w/2, y + h/2, w, h)
        else:
             self.is_tracking = False
             return None

class ProShootingAnalyzer:
    def __init__(self, callback=None):
        self.callback = callback
        print("[V3 Initializing] Loading YOLO Model...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_path = os.path.join(current_dir, 'yolov8s.pt')
        self.model = YOLO(yolo_path)
        self.cam_mtx = None
        self.cam_dist = None
        self.F_px = 640.0 
        self.calibrated_from_json = False
        self.goal_area = None
        
    def detect_goal_area(self, cap, width, height):
        print("[V3 Geometry] Scanning for goal area (not camera calibration)...")
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            goal_model_path = os.path.join(current_dir, 'goal_segment_best.pt')
            if not os.path.exists(goal_model_path): 
                print("[V2.5] Goal model not found.")
                return None
                
            goal_model = YOLO(goal_model_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            best_goal = None
            # Goal geometry is a detected area; it does not determine focal length.
            for frame_number in range(20):
                ret, frame = cap.read()
                if not ret: break
                results = goal_model(frame, imgsz=640, verbose=False, conf=0.04)
                result = results[0]
                parts = []
                for index, box in enumerate(result.boxes):
                    confidence = float(box.conf[0])
                    parts.append({'bbox': box.xyxy[0].cpu().tolist(), 'confidence': confidence, 'method': 'box'})
                    if result.masks is not None and index < len(result.masks.xy):
                        polygon = result.masks.xy[index]
                        if len(polygon) > 2:
                            x, y, w, h = cv2.boundingRect(np.int32(polygon))
                            parts.append({'bbox': [x, y, x + w, y + h], 'confidence': confidence, 'method': 'mask'})
                area = assemble_goal(parts, width, height, frame_number)
                if area and (best_goal is None or area['confidence'] > best_goal['confidence']):
                    best_goal = area
            if best_goal is not None:
                self.goal_area = best_goal
                print(f"[V3 Geometry] Goal area detected: {best_goal['bbox']}")
                return best_goal
        except Exception as e:
            print(f"[V3 Geometry] Goal area detection failed: {e}")
        return None
        
    def setup_calibration(self, width, height, calib_file=None):
        self.camera = load_camera(width, height, calibration_path=calib_file)
        self.cam_mtx = self.camera.matrix
        self.cam_dist = self.camera.distortion
        self.F_px = self.camera.matrix[0, 0]
        self.calibrated_from_json = self.camera.calibrated
        print(f"[V3 Geometry] Camera: {self.camera.source}; warnings={self.camera.warnings}")

    def extract_balls(self, frame):
        result = self.model(frame, imgsz=640, verbose=False, conf=0.04)[0]
        return [{"bbox": box.xywh[0].cpu().tolist(), "confidence": float(box.conf[0])}
                for box in result.boxes if int(box.cls[0]) == 32]

    def recover_in_roi(self, frame, tracker):
        # A crop gives a small distant ball more pixels at the same YOLO input size.
        center = tracker.filter.state[:2] + tracker.filter.state[3:5] / tracker.fps
        side = int(max(160, min(tracker.width, tracker.height) * 0.25,
                       tracker.filter.state[2] * 8, tracker._gate() * 2))
        side = min(side, int(min(tracker.width, tracker.height) * 0.65))
        x1, y1 = max(0, int(center[0] - side / 2)), max(0, int(center[1] - side / 2))
        x2, y2 = min(tracker.width, x1 + side), min(tracker.height, y1 + side)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return []
        result = self.model(frame[y1:y2, x1:x2], imgsz=640, verbose=False, conf=0.1)[0]
        detections = []
        for box in result.boxes:
            if int(box.cls[0]) == 32:
                u, v, w, h = box.xywh[0].cpu().tolist()
                detections.append({"bbox": [u + x1, v + y1, w, h], "confidence": float(box.conf[0]), "detector": "roi"})
        return detections

    def draw_tracking(self, frame, tracker, record):
        image = frame.copy()
        if self.goal_area:
            x1, y1, x2, y2 = map(int, self.goal_area["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 2)
        shot = [r for r in tracker.trajectory() if r["x"] is not None]
        for left, right in zip(shot, shot[1:]):
            # Do not draw a continuous observation across a long missing interval.
            if right["frame_number"] - left["frame_number"] > tracker.prediction_limit:
                continue
            color = (0, 200, 0) if right["observed"] else (255, 0, 255)
            cv2.line(image, (int(left["x"]), int(left["y"])), (int(right["x"]), int(right["y"])), color, 2)
        if record["x"] is not None:
            center = (int(record["x"]), int(record["y"]))
            cv2.circle(image, center, max(2, int(record["radius_px"])), (0, 255, 255), 2)
        if record["ball_bbox"] is not None:
            x1, y1, x2, y2 = map(int, record["ball_bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0) if record["source"] == "yolo" else (0, 180, 255), 1)
        text = f"frame {record['frame_number']} | {record['source']} | {'SHOT' if tracker.active else 'WAITING'}"
        cv2.putText(image, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image

    def run(self, video_path, output_path, capture_options=None):
        if not capture_options:
            capture_options = dict(DEFAULT_AUTOMATIC_OPTIONS)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError("Could not open video")
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = reported_fps if np.isfinite(reported_fps) and reported_fps > 0 else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            self.camera, measurement_context = build_context(video_path, width, height, fps,
                                                            fps != reported_fps, capture_options)
        except Exception:
            cap.release()
            raise
        self.detect_goal_area(cap, width, height)
        out_path = output_path
        metadata = {"width": width, "height": height, "fps": fps, "frame_count": total_frames,
                    "fps_fallback": fps != reported_fps, "coordinate_space": "original_pixels",
                    "origin": "top_left", "frame_index_base": 0}
        metadata['tracking_time_basis'] = 'file_playback_time'
        metadata['measurement_context'] = measurement_context
        print("[V3 Tracking] Phase 1: Initial Ball Sweep")
        preview = []
        first_preview_image = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_number in range(min(30, max(15, int(fps)))):
            ret, frame = cap.read()
            if not ret:
                break
            if first_preview_image is None:
                first_preview_image = frame.copy()
            preview.append((frame_number, self.extract_balls(frame)))
        initial = select_initial_ball(preview, width, height)
        if initial is None:
            cap.release()
            physics = estimate_shot([], metadata, self.camera)
            physics['failure_reason'] = 'BALL_NOT_FOUND'
            fallback = first_preview_image if first_preview_image is not None else np.zeros((height, width, 3), dtype=np.uint8)
            if not cv2.imwrite(out_path, result_image(fallback, [], physics, self.goal_area)):
                raise RuntimeError('Could not save failure image')
            summary = {"kick_detected": False, "observed_points": 0, "tracking_status": "INSUFFICIENT"}
            goal_entry = assess_goal_entry([], self.goal_area, metadata, summary)
            physical_goal = assess_physical_goal([], metadata, self.camera, summary, {},
                                                measurement_context['options'].get('goal_plane'))
            prediction, score_evaluation = _goal_outputs(goal_entry, physical_goal, physics, measurement_context)
            return {"score": score_evaluation['value'], "speed": None, "comment": "공을 찾지 못했습니다.",
                    "score_status": score_evaluation['status'],
                    "trajectory_image": os.path.basename(out_path), "trajectory": [],
                    "video_metadata": metadata, "goal_area": self.goal_area,
                    "tracking_summary": summary, "goal_entry": goal_entry,
                    "physical_goal": physical_goal, "goal_prediction": prediction,
                    "analysis_mode": measurement_context['analysis_mode'], "score_evaluation": score_evaluation,
                    **physics}
        print(f"[V3 Tracking] Initial ball: {initial['bbox'][:2]}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, initial['frame_number'])
        readable_seed, seed_frame = cap.read()
        if not readable_seed:
            cap.release()
            raise RuntimeError('Could not read initial ball appearance frame')
        tracker = BallTracker(initial, width, height, fps, initial_frame=seed_frame)
        size_inspector = BallSizeInspector(seed_frame, initial['bbox'])
        csrt = HybridTracker()
        debug_path = os.path.splitext(out_path)[0] + "_tracking.mp4"
        track_path = os.path.splitext(out_path)[0] + "_tracking.json"
        writer = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create tracking video")
        first_track_frame = None
        last_static_frame = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        preview_by_frame = dict(preview)
        frame_idx = 0
        print("[V3 Tracking] Phase 2: Kalman Tracking")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = preview_by_frame.get(frame_idx)
                if detections is None:
                    detections = self.extract_balls(frame)
                tracker.prepare_frame(frame_idx, frame)
                if tracker.active:
                    candidate = tracker._associate(detections, frame)
                    if candidate is None or candidate["confidence"] < 0.12:
                        detections = detections + self.recover_in_roi(frame, tracker)
                was_active = tracker.active
                record, detection = tracker.step(frame_idx, detections, lambda: csrt.update(frame), frame=frame)
                record['ball_size_quality'] = (size_inspector.inspect(frame, record['ball_bbox'])
                                               if record['source'] == 'yolo' and record['ball_bbox'] else None)
                if any(r['source'] == 'csrt' for r in record['rejected_observations']):
                    csrt.is_tracking = False
                if detection is not None and tracker.active:
                    csrt.init(frame, *detection["bbox"])
                if not tracker.active and tracker.last_static_frame == frame_idx and record["observed"]:
                    last_static_frame = frame.copy()
                if tracker.active and not was_active:
                    first_track_frame = last_static_frame if last_static_frame is not None else frame.copy()
                    print(f"[V3 Tracking] KICK! Initialized at frame {frame_idx}")
                writer.write(self.draw_tracking(frame, tracker, record))
                if self.callback and frame_idx % 5 == 0:
                    self.callback(15 + int(frame_idx / max(1, total_frames) * 75))
                frame_idx += 1
                if tracker.finished:
                    break
        finally:
            cap.release()
            writer.release()
        trajectory = tracker.trajectory()
        summary = tracker.summary()
        summary["tracking_status"] = "TRACKED" if summary["observed_points"] >= 5 else "INSUFFICIENT"
        motion_start = summary['shot_start_frame'] or 0
        physical_fps = measurement_context['timing']['effective_capture_fps']
        motion_end = min(len(tracker.records) - 1, motion_start + round((physical_fps or fps) * .45))
        camera_motion = assess_camera_motion(video_path, motion_start, motion_end)
        physics = estimate_shot(trajectory, metadata, self.camera, camera_motion)
        summary['flight_observations'] = physics['flight_segment'].get('point_count', 0)
        # The goal rectangle was detected near the start, so the early speed
        # window's camera check is insufficient for a later region entry.
        goal_motion_start, goal_motion_end = goal_camera_interval(trajectory, self.goal_area, len(tracker.records) - 1)
        goal_plane = measurement_context['options'].get('goal_plane')
        if goal_plane:
            goal_motion_start = min(goal_motion_start, goal_plane['reference_frame'])
            goal_motion_end = max(goal_motion_end, goal_plane['reference_frame'])
        goal_motion = assess_camera_motion(video_path, goal_motion_start, goal_motion_end)
        goal_motion['frame_interval'] = [goal_motion_start, goal_motion_end]
        goal_entry = assess_goal_entry(trajectory, self.goal_area, metadata, summary, goal_motion)
        physical_goal = assess_physical_goal(trajectory, metadata, self.camera, summary, goal_motion, goal_plane)
        prediction, score_evaluation = _goal_outputs(goal_entry, physical_goal, physics, measurement_context)
        print(f"[V3 Geometry] Measurement: {physics['measurement_status']}; reason={physics['failure_reason']}; speed={physics['speed']}")
        print(f"[V3 Geometry] Flight window has {summary['flight_observations']} trusted YOLO points")
        # A result image is still saved when metric estimation is unavailable.
        if first_track_frame is None:
            first_track_frame = first_preview_image
        observed_path = os.path.splitext(out_path)[0] + "_observed.jpg"
        for target, only_observations in [(out_path, False), (observed_path, True)]:
            rendered = result_image(first_track_frame, trajectory, physics, self.goal_area, only_observations)
            if not cv2.imwrite(target, rendered):
                raise RuntimeError("Could not save result image")
        goal_image_path = os.path.splitext(out_path)[0] + '_goal_entry.jpg'
        evidence_frame = (goal_entry['events'][0]['confirmation_frame'] if goal_entry['events'] else
                          next((r['frame_number'] for r in reversed(trajectory) if r['observed']), motion_start))
        if physical_goal['events']:
            evidence_frame = physical_goal['events'][0]['behind_frame']
        evidence_cap = cv2.VideoCapture(video_path)
        try:
            evidence_cap.set(cv2.CAP_PROP_POS_FRAMES, evidence_frame)
            readable, evidence_image = evidence_cap.read()
        finally:
            evidence_cap.release()
        if not readable:
            evidence_image = first_track_frame
            goal_entry['warnings'].append('EVIDENCE_FRAME_UNREADABLE_USING_START_FRAME')
        goal_entry['image_frame'] = evidence_frame if readable else None
        if not cv2.imwrite(goal_image_path, goal_entry_image(evidence_image, trajectory, goal_entry, physical_goal, score_evaluation)):
            raise RuntimeError('Could not save goal entry image')
        with open(track_path, "w", encoding="utf-8") as stream:
            json.dump({"video_metadata": metadata, "goal_area": self.goal_area, "tracking_summary": summary,
                       "initial_detection": initial, "frames": tracker.records, "trajectory": trajectory,
                       "physics": physics, "goal_entry": goal_entry,
                       "physical_goal": physical_goal, "goal_prediction": prediction,
                       "analysis_mode": measurement_context['analysis_mode'], "score_evaluation": score_evaluation}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        if physics['measurement_status'] == 'ESTIMATED':
            comment = "초기 비행 구간의 속도 추정입니다. 실측 정확도는 검증하지 않았습니다."
            if not self.camera.calibrated:
                comment += " 카메라 보정이 없어 기본 화각을 가정했습니다."
        else:
            comment = "신뢰할 수 있는 속도를 계산하지 못했습니다: " + physics['failure_reason']
        if score_evaluation['status'] == 'PROVISIONAL':
            comment = score_evaluation['comment']
        if prediction:
            comment = automatic_comment(prediction, physics, score_evaluation)
        if self.callback:
            self.callback(100)
        return {"score": score_evaluation['value'], "score_status": score_evaluation['status'], "comment": comment,
                "trajectory_image": os.path.basename(out_path), "observation_image": os.path.basename(observed_path),
                "trajectory": trajectory, "video_metadata": metadata, "goal_area": self.goal_area,
                "tracking_summary": summary, "tracking_video": os.path.basename(debug_path),
                "tracking_data": os.path.basename(track_path), "goal_entry": goal_entry,
                "physical_goal": physical_goal, "goal_prediction": prediction,
                "analysis_mode": measurement_context['analysis_mode'], "score_evaluation": score_evaluation,
                "goal_entry_image": os.path.basename(goal_image_path), **physics}


def process_video_with_progress(video_path, output_path, callback=None, capture_options=None):
    print(f"[MAIN] Processing started for {video_path}")
    analyzer = ProShootingAnalyzer(callback)
    return analyzer.run(video_path, output_path, capture_options)
