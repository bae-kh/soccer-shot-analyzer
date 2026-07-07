import cv2
import numpy as np
import time
import math
import os
import json
import traceback
from ultralytics import YOLO
from scipy.optimize import least_squares

class TrajectoryOptimizer:
    def __init__(self, F_px, cx, cy):
        self.F_px = F_px
        self.cx = cx
        self.cy = cy
        self.g = 9.81 

    def _project(self, params, t_arr):
        # [V2.5] Added Ax for Magnus Effect
        X0, Y0, Z0, Vx, Vy, Vz, Ax = params
        u_p, v_p = [], []
        for t in t_arr:
            X = X0 + Vx * t + 0.5 * Ax * t**2
            Y = Y0 + Vy * t + 0.5 * self.g * t**2
            Z = Z0 + Vz * t
            if Z < 0.1: Z = 0.1
            u_p.append(self.F_px * (X / Z) + self.cx)
            v_p.append(self.F_px * (Y / Z) + self.cy)
        return np.array(u_p), np.array(v_p)

    def _residuals(self, params, t_arr, u_obs, v_obs):
        u_p, v_p = self._project(params, t_arr)
        # [V2.5] Origin Pinning Formula
        weights = np.exp(-1.0 * t_arr)
        weights[0] += 100000.0  # Massive penalty for missing the exact kick point
        
        res_u = (u_obs - u_p) * weights
        res_v = (v_obs - v_p) * weights
        return np.concatenate((res_u, res_v))

    def optimize(self, t_arr, u_obs, v_obs, r0):
        if len(t_arr) < 5: return None
        
        Z0_guess = (self.F_px * 0.11) / r0 if r0 > 0 else 5.0
        X0_guess = (u_obs[0] - self.cx) * Z0_guess / self.F_px
        Y0_guess = (v_obs[0] - self.cy) * Z0_guess / self.F_px
        
        idx = min(len(t_arr)-1, max(3, len(t_arr)//2))
        dt = t_arr[idx] - t_arr[0]
        if dt <= 0: return None
        
        u_end, v_end = u_obs[idx], v_obs[idx]
        Z_end_guess = Z0_guess + 10.0 
        X_end_guess = (u_end - self.cx) * Z_end_guess / self.F_px
        Y_end_guess = (v_end - self.cy) * Z_end_guess / self.F_px
        
        Vx_guess = (X_end_guess - X0_guess) / dt
        Vy_guess = (Y_end_guess - Y0_guess) / dt
        Vz_guess = (Z_end_guess - Z0_guess) / dt
        Ax_guess = 0.0 # Assume straight initially
        
        initial_guess = [X0_guess, Y0_guess, Z0_guess, Vx_guess, Vy_guess, Vz_guess, Ax_guess]
        
        bounds = (
            [-20.0, -20.0,  0.5, -40.0, -40.0,   2.0, -40.0], 
            [ 20.0,  20.0, 30.0,  40.0,  40.0,  55.0,  40.0]  
        )
        
        try:
            res = least_squares(self._residuals, initial_guess, args=(t_arr, u_obs, v_obs), bounds=bounds, method='trf')
            if res.success:
                return res.x
        except Exception as e:
            print(f"[Optimizer] Failed: {e}")
            
        return None

class HybridTracker:
    def __init__(self):
        self.csrt = None
        self.is_tracking = False
        self.missed_yolo = 0
        
    def init(self, frame, u, v, w, h):
        x = int(max(0, u - w/2))
        y = int(max(0, v - h/2))
        w = int(w)
        h = int(h)
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
        print("[V2.5 Initializing] Loading YOLO Model...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_path = os.path.join(current_dir, 'yolov8s.pt')
        self.model = YOLO(yolo_path)
        self.cam_mtx = None
        self.cam_dist = None
        self.F_px = 640.0 
        self.REAL_RADIUS_M = 0.11
        self.calibrated_from_json = False
        
    def detect_goal_focal_length(self, cap, width, height):
        print("[V2.5] Scanning for goal to absolute-calibrate...")
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            goal_model_path = os.path.join(current_dir, 'goal_segment_best.pt')
            if not os.path.exists(goal_model_path): 
                print("[V2.5] Goal model not found.")
                return None
                
            goal_model = YOLO(goal_model_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            best_w = 0
            # [V2.5] Scan for up to 20 frames to find goal width
            for _ in range(20):
                ret, frame = cap.read()
                if not ret: break
                results = goal_model(frame, imgsz=640, verbose=False, conf=0.04)
                if results[0].masks:
                    for seg in results[0].masks.xy:
                        if len(seg) > 5:
                            rect = cv2.boundingRect(np.int32(seg))
                            gw = rect[2]
                            if best_w < gw < width * 0.95:
                                best_w = gw
            if best_w > 0:
                # [V2.5] Absolute Calibration (Assume 16 meters to goal, Goal width 7.32m)
                F_px = (16.0 * best_w) / 7.32
                print(f"[V2.5] 🥅 Goal detected! Width: {best_w}px. Absolute Calibrated F_px: {F_px:.1f}")
                return F_px
        except Exception as e:
            print(f"[V2.5] Goal calibration failed: {e}")
        return None
        
    def setup_calibration(self, width, height, calib_file="calibration.json"):
        if os.path.exists(calib_file):
            with open(calib_file, "r") as f:
                data = json.load(f)
                cam_mtx = np.array(data["camera_matrix"])
                self.cam_mtx = cam_mtx
                self.cam_dist = np.array(data["dist_coeffs"])
                calib_w = data.get("resolution", [1920, 1080])[0]
                scale = width / calib_w
                self.F_px = cam_mtx[0,0] * scale
                self.calibrated_from_json = True
                print(f"[V2.5] JSON Calibrated Focal Length loaded: {self.F_px:.1f}")
        else:
            self.calibrated_from_json = False
            self.F_px = (width / 2.0) / math.tan(math.radians(60) / 2.0)
            print(f"[V2.5] Fallback Initial Focal Length: {self.F_px:.1f}")

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.setup_calibration(width, height)
        # [V2.5] Dynamic Goal-Scaling Fallback
        if not self.calibrated_from_json:
             dynamic_F = self.detect_goal_focal_length(cap, width, height)
             if dynamic_F is not None:
                 self.F_px = dynamic_F
                 
        optimizer = TrajectoryOptimizer(self.F_px, width/2, height/2)
        tracker = HybridTracker()
        
        out_path = output_path
        
        # 1. Pre-flight sweep for initial ball
        print("[V2.5] Phase 1: Static Ball Sweep")
        static_candidates = []
        best_centered_score = -9999.0
        initial_ball_box = None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(15):
            ret, frame = cap.read()
            if not ret: break
            results = self.model(frame, imgsz=640, verbose=False, conf=0.1)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 32:
                        u, v, w, h = box.xywh[0].cpu().numpy()
                        conf = float(box.conf[0])
                        if 0.6 < w/h < 1.66:
                            cx_dist = abs(u - width/2) / width
                            cy_dist = abs(v - height*0.75) / height
                            score = conf - (cx_dist + cy_dist)*0.5
                            if score > best_centered_score:
                                best_centered_score = score
                                initial_ball_box = (u, v, w, h)
                                
        if initial_ball_box:
            print(f"[V2.5] Locked static ball at ({initial_ball_box[0]:.0f}, {initial_ball_box[1]:.0f})")
        else:
            print("[V2.5] No static ball found. Returning error.")
            return {"score": 0, "speed": 0, "comment": "공을 찾지 못했습니다.", "trajectory_image": os.path.basename(out_path)}
            
        # 2. Main Tracking Loop
        print("[V2.5] Phase 2: Main Tracking")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        flight_active = False
        tracked_t, tracked_u, tracked_v = [], [], []
        r0 = min(initial_ball_box[2], initial_ball_box[3]) / 2.0
        
        first_track_frame = None
        last_static_frame = None
        
        frame_idx = 0
        while True:
            ret, frame_raw = cap.read()
            if not ret: break
            frame_idx += 1
            t = frame_idx / fps
            
            if self.cam_mtx is not None:
                frame = cv2.undistort(frame_raw, self.cam_mtx, self.cam_dist, None, self.cam_mtx)
            else:
                frame = frame_raw
                
            if self.callback and frame_idx % 5 == 0:
                prog = 15 + int(frame_idx / total_frames * 75)
                self.callback(prog)
                
            yolo_res = self.model(frame, imgsz=640, verbose=False, conf=0.04)
            best_yolo = None
            min_dist = 9999
            
            # Extract all balls
            for r in yolo_res:
                for box in r.boxes:
                    if int(box.cls[0]) == 32:
                        u, v, w, h = box.xywh[0].cpu().numpy()
                        conf = float(box.conf[0])
                        ratio = w / (h + 1e-5)
                        
                        if flight_active:
                            if conf < 0.04: continue
                        else:
                            if conf < 0.15 or not (0.7 < ratio < 1.4): continue
                            
                        # If active, pick closest to tracker pred / last point
                        if flight_active and len(tracked_u) > 0:
                            dist = np.sqrt((u - tracked_u[-1])**2 + (v - tracked_v[-1])**2)
                            if dist < min_dist and dist < 200:
                                min_dist = dist
                                best_yolo = (u,v,w,h)
                        else:
                            # Pre kick, compare to static
                            dist = np.sqrt((u - initial_ball_box[0])**2 + (v - initial_ball_box[1])**2)
                            if dist < 15:
                                last_static_frame = frame.copy()
                            elif 30 < dist < 120 and conf > 0.05:
                                best_yolo = (u,v,w,h)
                                
            # State Machine
            if not flight_active:
                if best_yolo is not None:
                    # KICK DETECTED!
                    flight_active = True
                    first_track_frame = last_static_frame if last_static_frame is not None else frame.copy()
                    
                    # 앵커(Anchor) 추가
                    tracked_t.append(t)
                    tracked_u.append(initial_ball_box[0])
                    tracked_v.append(initial_ball_box[1])
                    
                    # 첫 트래커 초기화
                    tracker.init(frame, best_yolo[0], best_yolo[1], best_yolo[2], best_yolo[3])
                    tracked_t.append(t + 0.001) 
                    tracked_u.append(best_yolo[0])
                    tracked_v.append(best_yolo[1])
                    print(f"[V2.5] ⚽ KICK! Initialized at {best_yolo[0]:.0f}, {best_yolo[1]:.0f}")
            else:
                # Active tracking
                track_res = tracker.update(frame, best_yolo)
                if track_res is not None:
                    tracked_t.append(t)
                    tracked_u.append(track_res[0])
                    tracked_v.append(track_res[1])
                else:
                    # Tracking lost (e.g. out of frame or into net)
                    print(f"[V2.5] Tracking ended at frame {frame_idx}")
                    break
                    
        cap.release()
        
        # 3. Trajectory Fitting
        print(f"[V2.5] Phase 3: Trajectory Optimization with {len(tracked_t)} points")
        if len(tracked_t) < 5:
            return {"score": 0, "speed": 0, "comment": "오류: 추적된 공 궤적이 너무 짧습니다.", "trajectory_image": os.path.basename(out_path)}
            
        t_arr = np.array(tracked_t) - tracked_t[0]
        u_obs = np.array(tracked_u)
        v_obs = np.array(tracked_v)
        
        best_params = optimizer.optimize(t_arr, u_obs, v_obs, r0)
        
        if best_params is None:
            return {"score": 0, "speed": 0, "comment": "오류: 궤적 피팅에 실패했습니다.", "trajectory_image": os.path.basename(out_path)}
            
        X0, Y0, Z0, Vx, Vy, Vz, Ax = best_params
        
        # Calculate speed (magnitude of initial velocity vector) in km/h
        speed_ms = np.sqrt(Vx**2 + Vy**2 + Vz**2)
        speed_kmh = speed_ms * 3.6
        if speed_kmh > 150.0: speed_kmh = 150.0  # Biological limit cap
        print(f"[V2.5] Optimized Initial Speed: {speed_kmh:.1f} km/h (Magnus Ax: {Ax:.1f})")
        
        # 4. Visualization [V2.5] - Predictive Extrapolation
        if first_track_frame is not None:
            result_img = first_track_frame
        else:
            result_img = np.zeros((height, width, 3), dtype=np.uint8)
            
        # Add 0.4 seconds of extra prediction into the future
        future_frames = int(0.4 * fps)
        t_ext = list(t_arr)
        last_t = t_arr[-1]
        for i in range(1, future_frames + 1):
             t_ext.append(last_t + i / fps)
        t_ext = np.array(t_ext)
        
        u_p, v_p = optimizer._project(best_params, t_ext)
        
        points = []
        for i in range(len(u_p)):
            points.append((int(u_p[i]), int(v_p[i])))
            
        real_len = len(t_arr)
            
        # Draw actual track (solid, color gradient)
        for i in range(1, len(points)):
            t_ratio = i / real_len if i < real_len else 1.0
            
            if i < real_len:
                color = (0, int(255*(1-t_ratio)), int(255*t_ratio))
                thickness = max(2, int(2 + 4 * t_ratio))
                cv2.line(result_img, points[i-1], points[i], color, thickness, cv2.LINE_AA)
            else:
                # [V2.5] Future Prediction fading out (Broadcast style dashed/translucent effect)
                alpha = max(0, 1.0 - (i - real_len) / future_frames)
                color = (int(200*alpha), int(200*alpha), 255) # Fading white-red
                thickness = max(1, int(5 * alpha))
                
                # Manual dash by skipping every other frame
                if i % 2 == 0:
                     cv2.line(result_img, points[i-1], points[i], color, thickness, cv2.LINE_AA)
            
        if len(points) > 0:
            cv2.circle(result_img, points[0], 12, (0, 255, 0), -1) 
            cv2.circle(result_img, points[0], 15, (255, 255, 255), 2) # Anchor halo
            
            # End of actual tracking marker
            if real_len - 1 < len(points):
                cv2.circle(result_img, points[real_len-1], 8, (0, 80, 255), -1)
        
        # Put Text
        text_color = (0, 255, 255)
        text_str = f"SPEED: {speed_kmh:.1f} KM/H"
        cv2.putText(result_img, text_str, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 6) # Shadow
        cv2.putText(result_img, text_str, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3) 
        
        if abs(Ax) > 5.0:
            curve_str = "CURVE SHOT DETECTED!"
            cv2.putText(result_img, curve_str, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
            
        cv2.imwrite(out_path, result_img)
        
        if self.callback: self.callback(100)
        
        # Comment Gen
        comment = ""
        if speed_kmh > 100: comment = "상위 1% 급의 미친 대포알 슈팅입니다! 🚀"
        elif speed_kmh > 80: comment = "아마추어를 뛰어넘은 멋진 슈팅! 🔥"
        else: comment = "더 강하게 찰 수 있습니다! 화이팅! 💨"
        
        score = min(100, int(speed_kmh / 1.3))
        
        return {"score": score, "speed": round(speed_kmh, 1), "comment": comment, "trajectory_image": os.path.basename(out_path)}

def process_video_with_progress(video_path, output_path, callback=None):
    print(f"[MAIN] Processing started for {video_path}")
    analyzer = ProShootingAnalyzer(callback)
    return analyzer.run(video_path, output_path)
