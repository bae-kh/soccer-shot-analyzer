
import cv2
import numpy as np
import os
import traceback
from ultralytics import YOLO

class ProShootingAnalyzer:
    def __init__(self, callback=None):
        self.model = YOLO('yolov8n.pt') 
        if os.path.exists('yolov8s.pt'):
            self.model = YOLO('yolov8s.pt')
            
        # Goal Detection Model    
        self.goal_model = None
        
        # Absolute path fix (Robust based on file location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'goal_segment_best.pt')
        
        if os.path.exists(model_path):
            try:
                self.goal_model = YOLO(model_path)
                self.goal_model_loaded = True
                print(f"[SUCCESS] Goal Model Loaded from: {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                traceback.print_exc()
                self.goal_model_loaded = False
        else:
            print(f"[WARNING] Goal Model NOT FOUND at: {model_path}")
            self.goal_model_loaded = False
            
        self.callback = callback
        
        # --- 3D Kalman Filter Setup ---
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.identity(6, np.float32)
        self.kf.transitionMatrix[0, 3] = 1
        self.kf.transitionMatrix[1, 4] = 1
        self.kf.transitionMatrix[2, 5] = 1

        self.kf.measurementMatrix = np.identity(6, np.float32)[:3]
        
        # Tuning
        self.kf.processNoiseCov = np.identity(6, np.float32) * 0.01 
        self.kf.processNoiseCov[2, 2] = 0.001 
        self.kf.processNoiseCov[5, 5] = 0.001 
        self.kf.measurementNoiseCov = np.identity(3, np.float32) * 0.1
        self.kf.measurementNoiseCov[2, 2] = 1.0 
        
        self.trajectory_3d = []   
        self.velocities = []      
        self.timestamps = []      
        self.initialized = False
        self.recent_radii = []    
        self.F_px = 1000 
        self.REAL_RADIUS_M = 0.11 

    # --- Goal Detection Helpers (Modified for Robustness) ---
    def is_valid_goal(self, cnt, width, height, max_ratio_limit):
        area = cv2.contourArea(cnt)
        if area < (width * height) * 0.005: return False, "Too Small", 0

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        
        # Robust Angle/Ratio Normalization
        if w < h: 
            w, h = h, w
            angle += 90
            
        # Standardize angle to -45 to 45
        while angle > 45: angle -= 90


        ratio = w / (h + 1e-5)

        # Relax Ratio (allow squarish goals 0.8)
        if ratio < 1.3 or ratio > max_ratio_limit:
            return False, f"Ratio {ratio:.2f}", 0

        # Relax Angle Check (Issue with OpenCV version differences)
        if abs(angle) > 20: return False, f"Angle {angle:.1f}", 0

        screen_center_x = width / 2
        dist_from_center = abs(cx - screen_center_x)
        if dist_from_center > (width * 0.35):
            return False, "Far Side", dist_from_center

        return True, "Pass", dist_from_center

    def scan_for_goal_v21_logic(self, cap, model, width, height):
        # [Step 1] v21 Logic (Strict-ish)
        accumulated_mask = np.zeros((height, width), dtype=np.uint8)
        scan_limit = 60
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_cnt = 0
        while frame_cnt < scan_limit:
            ret, frame = cap.read()
            if not ret: break
            # Slight relax on conf to 0.02 to ensure masks are found
            results = model(frame, imgsz=640, conf=0.02, verbose=False)
            if results[0].masks:
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in results[0].masks.xy:
                    if len(seg) > 0:
                        cv2.fillPoly(temp_mask, [np.int32(seg)], 255)
                accumulated_mask = cv2.bitwise_or(accumulated_mask, temp_mask)
            frame_cnt += 1

        # Morph: Open(3,15) -> Dilate(5,5)
        kernel_split = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        accumulated_mask = cv2.morphologyEx(accumulated_mask, cv2.MORPH_OPEN, kernel_split)

        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        accumulated_mask = cv2.dilate(accumulated_mask, kernel_smooth, iterations=1)

        contours, _ = cv2.findContours(accumulated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        min_dist = 99999

        for cnt in contours:
            valid, msg, dist = self.is_valid_goal(cnt, width, height, max_ratio_limit=3.5)
            if valid:
                if dist < min_dist:
                    min_dist = dist
                    best_cnt = cnt
        return best_cnt

    def scan_for_goal_v22_logic(self, cap, model, width, height):
        # [Step 2] v22 Logic (Rescue)
        accumulated_mask = np.zeros((height, width), dtype=np.uint8)
        scan_limit = 60
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_cnt = 0
        while frame_cnt < scan_limit:
            ret, frame = cap.read()
            if not ret: break
            # Rescue Mode: Ultra High Sensitivity
            results = model(frame, imgsz=640, conf=0.001, verbose=False)
            if results[0].masks:
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in results[0].masks.xy:
                    if len(seg) > 0:
                        cv2.fillPoly(temp_mask, [np.int32(seg)], 255)
                accumulated_mask = cv2.bitwise_or(accumulated_mask, temp_mask)
            frame_cnt += 1

        # Morph: Dilate(5,5)x2 -> Open(3,5) -> Dilate(20,1)
        kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        accumulated_mask = cv2.dilate(accumulated_mask, kernel_base, iterations=2)
        
        kernel_gentle = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        accumulated_mask = cv2.morphologyEx(accumulated_mask, cv2.MORPH_OPEN, kernel_gentle)
        
        kernel_heal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        accumulated_mask = cv2.dilate(accumulated_mask, kernel_heal, iterations=1)

        contours, _ = cv2.findContours(accumulated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        min_dist = 99999

        for cnt in contours:
            valid, msg, dist = self.is_valid_goal(cnt, width, height, max_ratio_limit=5.0)
            if valid:
                if dist < min_dist:
                    min_dist = dist
                    best_cnt = cnt
        return best_cnt

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"score": 0, "speed": 0, "comment": "Video Load Error"}
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1
        
        # Camera Calibration
        fov_h_deg = 60
        self.F_px = (width / 2) / np.tan(np.deg2rad(fov_h_deg/2))
        
        # --- PHASE 1: Goal Detection (Pre-scan) ---
        best_goal_cnt = None
        goal_rect = None
        
        if self.goal_model_loaded:
            if self.callback: self.callback(5)
            # Try v21 (Strict)
            best_goal_cnt = self.scan_for_goal_v21_logic(cap, self.goal_model, width, height)
            
            # If fail, Try v22 (Rescue)
            if best_goal_cnt is None:
                if self.callback: self.callback(10)
                best_goal_cnt = self.scan_for_goal_v22_logic(cap, self.goal_model, width, height)
                
            if best_goal_cnt is not None:
                goal_rect = cv2.minAreaRect(best_goal_cnt)
        
        # --- PHASE 2: Main Processing ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start
        
        frame_idx = 0
        missed_frames = 0
        trajectory_draw = []   
        first_frame = None
        
        if self.callback: self.callback(15)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx == 0:
                    first_frame = frame.copy()
                
                frame_idx += 1
                dt = 1.0 / fps 
                
                if self.callback and frame_idx % 5 == 0:
                    # Map remaining 85% to frames
                    prog = 15 + int(frame_idx / total_frames * 75)
                    self.callback(prog)
                
                pred = self.kf.predict()
                pred_x_m, pred_y_m, pred_z_m = pred[0][0], pred[1][0], pred[2][0]
                
                PROCESS_W = 1280
                scale = PROCESS_W / width
                PROCESS_H = int(height * scale)
                frame_process = cv2.resize(frame, (PROCESS_W, PROCESS_H))
                
                results = self.model(frame_process, verbose=False, conf=0.1)
                
                best_box = None
                max_conf = -1
                min_dist_score = 9999
                
                cx, cy = PROCESS_W/2, PROCESS_H/2
                pred_u, pred_v = 0, 0
                if self.initialized and pred_z_m > 0:
                    pred_u = int((pred_x_m * self.F_px * scale) / pred_z_m + cx)
                    pred_v = int((pred_y_m * self.F_px * scale) / pred_z_m + cy)
                
                for r in results:
                    for box in r.boxes:
                        # Standard YOLO Class 32: Sports ball
                        if int(box.cls[0]) == 32: 
                            conf = float(box.conf[0])
                            u, v, w, h = box.xywh[0].cpu().numpy() # Center u,v and w,h in Process coords
                            
                            # Aspect Ratio Check
                            if not (0.6 < w/h < 1.66): continue 
                            
                            # Selection Logic
                            if self.initialized:
                                # Distance Gate in Pixels
                                dist = np.sqrt((u-pred_u)**2 + (v-pred_v)**2)
                                if dist < 150: # Gate
                                    if dist < min_dist_score:
                                        min_dist_score = dist
                                        best_box = (u, v, w, h)
                            else:
                                if conf > max_conf:
                                    max_conf = conf
                                    best_box = (u, v, w, h)
                                    
                if best_box:
                    u, v, w, h = best_box
                    r_px_raw = (w + h) / 4.0
                    
                    # --- Radius Smoothing (Crucial for Z stability) ---
                    self.recent_radii.append(r_px_raw)
                    if len(self.recent_radii) > 5: self.recent_radii.pop(0)
                    r_px = np.mean(self.recent_radii)
                    
                    # --- 1. 3D Depth Estimation (Pinhole) ---
                    F_proc = self.F_px * scale
                    z_meas = (F_proc * self.REAL_RADIUS_M) / r_px
                    
                    # X, Y (Meters)
                    x_meas = (u - cx) * z_meas / F_proc
                    y_meas = (v - cy) * z_meas / F_proc
                    
                    # --- 2. Kalman Update ---
                    meas = np.array([[np.float32(x_meas)], 
                                     [np.float32(y_meas)], 
                                     [np.float32(z_meas)]])
                                     
                    if not self.initialized:
                        self.kf.statePost = np.array([
                            [np.float32(x_meas)], [np.float32(y_meas)], [np.float32(z_meas)],
                            [0], [0], [0]], np.float32)
                        self.kf.statePre = self.kf.statePost
                        self.initialized = True
                    else:
                        self.kf.correct(meas)
                        
                    # Get Refined State (Post-Correction)
                    state = self.kf.statePost
                    curr_x, curr_y, curr_z = state[0][0], state[1][0], state[2][0]
                    curr_vx, curr_vy, curr_vz = state[3][0], state[4][0], state[5][0]
                    
                    # Store 3D Trajectory
                    self.trajectory_3d.append((curr_x, curr_y, curr_z))
                    
                    speed_m_per_frame = np.sqrt(curr_vx**2 + curr_vy**2 + curr_vz**2)
                    speed_mps = speed_m_per_frame * fps
                    self.velocities.append(speed_mps)
                    self.timestamps.append(frame_idx / fps)
                    
                    # Store Draw coords
                    if curr_z > 0.1:
                        draw_u = int((curr_x * F_proc) / curr_z + cx)
                        draw_v = int((curr_y * F_proc) / curr_z + cy)
                        draw_u_orig = int(draw_u / scale)
                        draw_v_orig = int(draw_v / scale)
                        trajectory_draw.append((draw_u_orig, draw_v_orig))
                        
                    missed_frames = 0
                    
                else:
                    if self.initialized:
                        missed_frames += 1
                        if missed_frames > 5:
                            break # Lost

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

        cap.release()
        
        # --- Visualization & Goal Logic ---
        if first_frame is not None:
             result_img = first_frame
        else:
             result_img = np.zeros((height, width, 3), np.uint8)
        
        # Draw Goal (if detected)
        is_goal = False
        dist_to_corner_score = 0
        
        if best_goal_cnt is not None:
            # Draw Goal Contour (Filled Overlay like User's Script)
            box = np.int32(cv2.boxPoints(goal_rect))
            
            overlay = result_img.copy()
            cv2.drawContours(overlay, [box], 0, (0, 255, 0), -1)
            result_img = cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0)
            cv2.drawContours(result_img, [box], 0, (0, 255, 0), 3)
            
            # Draw Text
            cv2.putText(result_img, "GOAL DETECTED", (box[1][0], box[1][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Goal Logic (Robust Segment Check)
            if trajectory_draw and len(trajectory_draw) >= 2:
                # Use the Bounding Box (box) for the check instead of best_goal_cnt
                # box is a numpy array of 4 points [(x,y), ...]. 
                # pointPolygonTest works with contours (N, 1, 2) or (N, 2).
                
                goal_poly = box  # box is derived from minAreaRect, robust convex shape
                
                # Check the last few segments (e.g. last 5 points = 4 segments)
                check_count = min(len(trajectory_draw), 5)
                
                # 1. Check if any recent point is inside the Bounding Box
                for i in range(1, check_count + 1):
                    pt = trajectory_draw[-i]
                    # Use goal_poly (box)
                    if cv2.pointPolygonTest(goal_poly, pt, False) >= 0:
                        is_goal = True
                        break
                
                # 2. Intersection Check (Segments vs Box)
                if not is_goal:
                     for i in range(1, check_count):
                        p1 = trajectory_draw[-(i+1)]
                        p2 = trajectory_draw[-i]
                        
                        # Interpolate
                        for t in np.linspace(0, 1, 5):
                            ix = int(p1[0] + (p2[0]-p1[0])*t)
                            iy = int(p1[1] + (p2[1]-p1[1])*t)
                            if cv2.pointPolygonTest(goal_poly, (ix, iy), False) >= 0:
                                is_goal = True
                                break
                        if is_goal: break

                # Corner Logic (Detailed)
                # Find top-left and top-right of the minRect
                # box points order is not guaranteed. Sort by Y then X.
                # Top corners have smallest Y.
                sorted_pts = sorted(box, key=lambda p: p[1])
                top_two = sorted_pts[:2]
                top_two = sorted(top_two, key=lambda p: p[0]) # Left and Right
                
                tl, tr = top_two[0], top_two[1]
                
                if is_goal:
                    last_pt = trajectory_draw[-1]
                    d_tl = np.linalg.norm(np.array(last_pt) - np.array(tl))
                    d_tr = np.linalg.norm(np.array(last_pt) - np.array(tr))
                    closest_d = min(d_tl, d_tr)
                    
                    # Normalize by goal width
                    goal_width = np.linalg.norm(np.array(tl) - np.array(tr))
                    if goal_width > 0:
                        normalized_dist = closest_d / goal_width
                        # Closer to 0 (corner) is better. Center is approx 0.5.
                        # Score: 0.1 or less -> 100. 
                        # 0.4 or more -> 10.
                        if normalized_dist < 0.1: 
                             dist_to_corner_score = 100
                        elif normalized_dist > 0.45: 
                             dist_to_corner_score = 10
                        else:
                             # Linear Map: 0.45->10, 0.1->100
                             # Slope = (100 - 10) / (0.1 - 0.45) = 90 / -0.35 = -257
                             # y - 10 = -257 * (x - 0.45)
                             dist_to_corner_score = 10 + 257 * (0.45 - normalized_dist)
                             
                        dist_to_corner_score = max(0, min(100, dist_to_corner_score))
        
        # Draw Trajectory
        if len(trajectory_draw) > 1:
            for i in range(1, len(trajectory_draw)):
                cv2.line(result_img, trajectory_draw[i-1], trajectory_draw[i], (0, 0, 255), 3)
        cv2.imwrite(output_path, result_img)
        
        # --- 3. Professional Physics & Scoring Calculation (Curve Fitting) ---
        final_speed = 0.0
        
        # We need enough data points for a valid regression
        if len(self.timestamps) > 5 and len(self.recent_radii) > 0:
            # 1. Gather Raw Data for Fitting
            # We want to fit Z(t). 
            # Z = (F * real_R) / img_R
            # img_R comes from self.recent_radii but we didn't store simple history.
            # Let's reconstruct from self.trajectory_3d which has (x, y, z)
            
            ts = np.array(self.timestamps)
            zs = np.array([p[2] for p in self.trajectory_3d]) 
            
            # 2. Outlier Rejection (Simple Z-score like)
            # Remove points where Z jumps crazily? 
            # For now, let's trust the KF output or raw?
            # KF is already smoothed. Let's use the KF Z data.
            
            # Align lengths (timestamps might be 1 ahead or behind depending on update loop)
            min_len = min(len(ts), len(zs))
            ts = ts[:min_len]
            zs = zs[:min_len]
            
            # 3. Fit Linear Model for Z (Constant Speed Approx for short distance)
            # Z(t) = v_z * t + z_0
            # Polyfit degree 1
            if min_len > 4:
                # Use only the first 50% of the flight for speed measurement (initial speed)
                # or the linear portion.
                # Projectile motion: Z velocity is roughly constant (drag is small for <20m).
                
                z_slope, z_intercept = np.polyfit(ts, zs, 1)
                vx_fit = 0
                vy_fit = 0
                
                # Fit X and Y? 
                xs = np.array([p[0] for p in self.trajectory_3d])[:min_len]
                ys = np.array([p[1] for p in self.trajectory_3d])[:min_len]
                
                x_slope, _ = np.polyfit(ts, xs, 1)
                y_slope, _ = np.polyfit(ts, ys, 1)
                # Note: Y has gravity, might want degree 2, but for speed scaler magnitude, linear is robust enough.
                
                # Calculate Total Speed Vector
                # Z slope is usually negative (moving away)
                speed_mps = np.sqrt(x_slope**2 + y_slope**2 + z_slope**2)
                
                # --- Sanity Check --- 
                # If fit error is huge, fallback or penalize?
                # Calculate R-squared
                p = np.poly1d([z_slope, z_intercept])
                yhat = p(ts)
                ybar = np.sum(zs)/len(zs)
                
                # If valid speed
                final_speed = speed_mps * 3.6
                print(f"[Results] Fitted Speed: {final_speed:.1f} km/h (Slope Z: {z_slope:.2f})")

        # Fallback to max peak if fit failed (too few points)
        if final_speed < 10 and len(self.velocities) > 0:
             final_speed = np.max(self.velocities) * 3.6
        
        # Cap physics limits (Amateurs rarely exceed 130, Pros 170)
        if final_speed > 165: final_speed = 165
        final_speed = round(final_speed, 1)
        
        # --- Scoring System ---
        power_score = 0
        if final_speed > 30:
            power_score = min(50, (final_speed - 30) * 0.6) 
            if final_speed > 90: power_score = 40 + (final_speed-90)*0.5 
            if power_score > 50: power_score = 50
            
        result_score = 0
        if is_goal:
            # Base Goal Score
            result_score = 30
            # Accuracy Bonus (0~20) based on corner proximity
            accuracy_bonus = (dist_to_corner_score / 100.0) * 20
            result_score += accuracy_bonus
        elif best_goal_cnt is None:
             result_score = 10
        else:
            result_score = 5 
            
        total_score = int(power_score + result_score)
        if total_score > 100: total_score = 100
        
        # --- Dynamic Comments ---
        s_tier = ""
        if final_speed < 60: s_tier = "아쉬운 속도"
        elif final_speed < 90: s_tier = "강력한 슈팅"
        else: s_tier = "프로급 파워"
        
        r_tier = ""
        if best_goal_cnt is None:
            r_tier = "(골대 미감지)"
        elif not is_goal:
            r_tier = "지만 골대를 벗어났네요 😅"
        else:
            if dist_to_corner_score > 60:
                r_tier = ", 그리고 완벽한 구석 궤적입니다! 🎯"
            else:
                r_tier = ", 골인입니다! 🎉"
                
        comment = f"{s_tier}({final_speed}km/h){r_tier}"
        if final_speed > 100 and is_goal:
            comment = "와우! 손흥민 선수가 울고 갈 완벽한 슈팅입니다! 🏆"
            
        if self.callback: self.callback(100)
        
        return {
            "score": total_score,
            "speed": final_speed,
            "comment": comment,
            "trajectory_image": os.path.basename(output_path)
        }

def process_video_with_progress(video_path, output_path, callback=None):
    analyzer = ProShootingAnalyzer(callback)
    return analyzer.run(video_path, output_path)
