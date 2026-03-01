import cv2
import numpy as np
import os
import traceback
import subprocess
import json
from ultralytics import YOLO

def get_exif_focal_length_px(video_path, img_width):
    """
    Step 1 of Auto Calibration: Attempt to parse EXIF metadata.
    Typically requires ExifTool for robust focal length extraction in mp4.
    Returns F_px if found, else None.
    """
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            # Placeholder for EXIF parsing logic. 
            # In a real environment, we would look for 'com.apple.quicktime.location.ISO6709' 
            # or use exiftool to get 'FocalLengthIn35mmFormat'.
            pass
    except Exception:
        pass
    return None

class ProShootingAnalyzer:
    def __init__(self, callback=None):
        self.model = YOLO('yolov8n.pt') 
        # Forced yolov8n for CPU inference speed (was yolov8s.pt before)
        
        # Goal Detection Model    
        self.goal_model = None
        
        # Absolute path fix (Robust based on file location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'goal_segment_best.pt')
        yolo_path = os.path.join(current_dir, 'yolov8s.pt')
        
        try:
            self.goal_model = YOLO(model_path)
            self.goal_model_loaded = True
        except:
            self.goal_model_loaded = False
            
        self.model = YOLO(yolo_path)
        
        self.known_static_objects = []
        self.saved_tracks = []
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
        self.kf.processNoiseCov[2, 2] = 0.05 # Relaxed from 0.001 to allow sudden Z-acceleration
        self.kf.processNoiseCov[5, 5] = 0.05 
        self.kf.measurementNoiseCov = np.identity(3, np.float32) * 0.1
        self.kf.measurementNoiseCov[2, 2] = 0.5 # Reduced from 1.0 to trust YOLO Z-depth more
        
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

        # Use boundingRect for stable width/height ratio calculation
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Still use minAreaRect for angle check
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (min_w, min_h), angle = rect
        
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
            print(f"[ERROR] Could not open video: {video_path}")
            return {"score": 0, "speed": 0, "comment": "Video Load Error"}
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1
        
        print(f"[INFO] Video Loaded: {video_path}, Frames: {total_frames}, FPS: {fps}")
        
        # Camera Calibration
        # Attempt to load calculated intrinsics
        calib_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
        self.cam_mtx = None
        self.cam_dist = None
        
        self.F_px = None
        self.calib_source = "None"
        
        if os.path.exists(calib_file):
            try:
                import json
                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)
                self.cam_mtx = np.array(calib_data["camera_matrix"], dtype=np.float32)
                self.cam_dist = np.array(calib_data["dist_coeffs"], dtype=np.float32)
                # Use the theoretically correct f_x instead of guessing based on 60 deg FOV
                self.F_px = self.cam_mtx[0][0] 
                self.calib_source = "Chessboard (JSON)"
                print(f"[INFO] Loaded Camera Calibration. Actual F_px: {self.F_px:.1f}")
            except Exception as e:
                print(f"[ERROR] Failed to load calibration data: {e}")
                
        # --- PHASE 0: Auto-Calibration Waterfall (Step 1: EXIF) ---
        if self.F_px is None:
            exif_fpx = get_exif_focal_length_px(video_path, width)
            if exif_fpx is not None:
                self.F_px = exif_fpx
                self.calib_source = "EXIF Metadata"
                print(f"[INFO] Fallback: Found EXIF focal length: {self.F_px:.1f}")
        
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
                print(f"[INFO] Goal Detected! Area: {cv2.contourArea(best_goal_cnt)}")
                x, y, w, h = cv2.boundingRect(best_goal_cnt)
                goal_rect = cv2.minAreaRect(best_goal_cnt) # Store goal_rect for later use
                
                # --- PHASE 0: Auto-Calibration Waterfall (Step 2: Goal Ratio) ---
                if self.F_px is None:
                    gw, gh = w, h
                    if gw > 0:
                        # Assume shot from 16.0m. Goal physical width = 7.32m
                        # Z = (F_px * Real_W) / px_W  ==>  F_px = (Z * px_W) / Real_W
                        self.F_px = (16.0 * gw) / 7.32
                        self.calib_source = "Environment (Goal Match)"
                        print(f"[INFO] Fallback: Calibrated F_px via Goal Match: {self.F_px:.1f}")
            else:
                print("[INFO] Goal NOT Detected.")
                
        # --- PHASE 0: Auto-Calibration Waterfall (Step 3: Default FOV) ---
        if self.F_px is None:
            fov_h_deg = 60
            self.F_px = (width / 2) / np.tan(np.deg2rad(fov_h_deg/2))
            self.calib_source = "Default (60deg FOV)"
            print(f"[INFO] Fallback: Using Default 60deg Focal Length: {self.F_px:.1f}")
            
        print(f"[INFO] Final active Calibration Source: {self.calib_source}")
        
        # --- PHASE 2: Main Processing ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start

        PROCESS_W = 640
        scale = PROCESS_W / width
        PROCESS_H = int(height * scale)
        
        self.known_static_objects = []

        frame_idx = 0
        missed_frames = 0
        self.trajectory_draw = []   
        first_frame = None
        self.saved_tracks = [] # Re-initialize for run
        
        if self.callback: self.callback(15)
        print("[INFO] Starting Main Tracking Loop...")
        
        stop_counter = 0 # For Net Hit Detection
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Apply Lens Distortion Correction if available
                if self.cam_mtx is not None and self.cam_dist is not None:
                    frame = cv2.undistort(frame, self.cam_mtx, self.cam_dist, None, self.cam_mtx)
                
                if frame_idx == 0:
                    first_frame = frame.copy()
                
                frame_idx += 1
                dt = 1.0 / fps 
                
                if self.callback and frame_idx % 5 == 0:
                    # Map remaining 85% to frames
                    prog = 15 + int(frame_idx / total_frames * 75)
                    self.callback(prog)
                
                if frame_idx % 30 == 0:
                    print(f"[INFO] Processing Frame {frame_idx}/{total_frames}")
                
                pred = self.kf.predict()
                pred_x_m, pred_y_m, pred_z_m = pred[0][0], pred[1][0], pred[2][0]
                
                PROCESS_W = 640
                scale = PROCESS_W / width
                PROCESS_H = int(height * scale)
                frame_process = cv2.resize(frame, (PROCESS_W, PROCESS_H))
                
                results = self.model(frame_process, imgsz=640, verbose=False, conf=0.1)
                
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
                            
                            if self.initialized:
                                # Distance Gate in Pixels
                                gate_size = 150
                                
                                dist = np.sqrt((u-pred_u)**2 + (v-pred_v)**2)
                                
                                if missed_frames > 0 or dist > 50:
                                    print(f"[DEBUG] Frame {frame_idx}: YOLO cls 32 conf={conf:.2f}, u={u:.1f}, pred_u={pred_u:.1f}, dist={dist:.1f}, gate={gate_size}")
                                
                                if dist < gate_size: # Dynamic Gate
                                    if dist < min_dist_score:
                                        min_dist_score = dist
                                        best_box = (u, v, w, h)
                                else:
                                    print(f"[DEBUG] Frame {frame_idx}: Rejected box due to distance gate ({dist:.1f} > {gate_size})")
                                    
                            else:
                                if conf > max_conf:
                                    max_conf = conf
                                    best_box = (u, v, w, h)
                                    
                if best_box:
                    if missed_frames > 0: print(f"[DEBUG] Frame {frame_idx}: Recovered tracking!")
                    missed_frames = 0
                    u, v, w, h = best_box
                    r_px_raw = (w + h) / 4.0
                    
                    # --- Radius Smoothing (Crucial for Z stability) ---
                    # [Fix 1] Increase buffer size to 5 for better Z stability
                    self.recent_radii.append(r_px_raw)
                    if len(self.recent_radii) > 5: self.recent_radii.pop(0)
                    r_px = np.mean(self.recent_radii)
                    
                    # --- 1. 3D Depth Estimation (Pinhole) ---
                    F_proc = self.F_px * scale
                    z_meas = (F_proc * self.REAL_RADIUS_M) / r_px
                    
                    # X, Y (Meters)
                    x_meas = (u - cx) * z_meas / F_proc
                    y_meas = (v - cy) * z_meas / F_proc
                    
                    # Filter extreme bad predictions (e.g., negative Z, Z going backwards towards camera)
                    if z_meas < 0.1: # Added this check as per the snippet's intent
                        print("[INFO] Target moved behind camera or Z is too small. Saving Track...")
                        if len(self.velocities) > 5:
                            self.saved_tracks.append((list(self.trajectory_3d), list(self.velocities), list(self.timestamps)))
                        self.initialized = False
                        self.trajectory_3d = []
                        self.velocities = []
                        self.timestamps = []
                        missed_frames = 0
                        continue
                            
                    # --- 2. Adaptive Kalman Update (Dynamic Z Noise) ---
                    # When ball is small (far away), 1px error causes massive Z jumps.
                    # Increase measurement noise so KF relies more on physics prediction.
                    base_z_noise = 0.5
                    if r_px < 10.0:
                        # Scale noise up to 10x as radius shrinks from 10 to 2
                        noise_scale = max(1.0, 10.0 / max(2.0, r_px))
                        self.kf.measurementNoiseCov[2, 2] = base_z_noise * (noise_scale ** 2)
                    else:
                        self.kf.measurementNoiseCov[2, 2] = base_z_noise
                        
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
                        self.trajectory_draw.append((draw_u_orig, draw_v_orig))
                        
                    missed_frames = 0
                    
                else:
                    if self.initialized:
                        missed_frames += 1
                        if missed_frames > 5:
                            break # Lost

        except Exception as e:
            print(f"[ERROR] Exception in Loop: {e}")
            traceback.print_exc()
 
        print(f"[INFO] Tracking Failed/Finished. Total Trajectory Points: {len(self.trajectory_3d)}")
        cap.release()
        
        # Save any remaining active trajectory
        if len(self.velocities) > 5:
            self.saved_tracks.append((list(self.trajectory_3d), list(self.velocities), list(self.timestamps), list(self.trajectory_draw)))
            
        if not self.saved_tracks:
            return {"score": 0, "speed": 0, "comment": "Error: 공의 궤적을 찾지 못했습니다."}

        # --- Track Selection ---
        best_track = None
        best_score = -1
        
        for num, (t_3d, t_vel, t_time, t_draw) in enumerate(self.saved_tracks):
            start_p = t_3d[0]
            end_p = t_3d[-1]
            dist_traveled = np.sqrt((end_p[0]-start_p[0])**2 + (end_p[1]-start_p[1])**2 + (end_p[2]-start_p[2])**2)
            mx_spd = np.max(t_vel) if len(t_vel) > 0 else 0
            
            score = dist_traveled * mx_spd
            
            print(f"[DEBUG] Track {num}: Pts={len(t_3d)}, Dist={dist_traveled:.1f}m, MaxVel={mx_spd:.1f}m/s -> Score={score:.1f}")
            if score > best_score:
                best_score = score
                best_track = (t_3d, t_vel, t_time, t_draw)
                
        self.trajectory_3d, self.velocities, self.timestamps, self.trajectory_draw = best_track
        
        print(f"[INFO] Selected best track with {len(self.trajectory_3d)} pts.")

        if len(self.trajectory_3d) < 5:
            return {"score": 0, "speed": 0, "comment": "Error: 공의 궤적을 충분히 추적하지 못했습니다."}
        
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
            if self.trajectory_draw and len(self.trajectory_draw) >= 2:
                # Use the Bounding Box (box) for the check instead of best_goal_cnt
                # box is a numpy array of 4 points [(x,y), ...]. 
                # pointPolygonTest works with contours (N, 1, 2) or (N, 2).
                
                goal_poly = box  # box is derived from minAreaRect, robust convex shape
                
                # Check the last few segments (e.g. last 5 points = 4 segments)
                check_count = min(len(self.trajectory_draw), 5)
                
                # 1. Check if any recent point is inside the Bounding Box
                for i in range(1, check_count + 1):
                    pt = self.trajectory_draw[-i]
                    # Use goal_poly (box)
                    if cv2.pointPolygonTest(goal_poly, pt, False) >= 0:
                        is_goal = True
                        break
                
                # 2. Intersection Check (Segments vs Box)
                if not is_goal:
                     for i in range(1, check_count):
                        p1 = self.trajectory_draw[-(i+1)]
                        p2 = self.trajectory_draw[-i]
                        
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
                    last_pt = self.trajectory_draw[-1]
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
        if len(self.trajectory_draw) > 1:
            for i in range(1, len(self.trajectory_draw)):
                cv2.line(result_img, self.trajectory_draw[i-1], self.trajectory_draw[i], (0, 0, 255), 3)
        cv2.imwrite(output_path, result_img)
        
        # --- 3. Professional Physics & Scoring Calculation (Curve Fitting) ---
        final_speed = 0.0
        
        # We need enough data points for a valid regression
        if len(self.timestamps) > 6:
            # 1. Gather Raw Data for Fitting
            ts = np.array(self.timestamps)
            
            # Align lengths
            min_len = min(len(ts), len(self.trajectory_3d))
            ts = ts[:min_len]
            
            if min_len > 6:
                try:
                    # Define xs, ys, zs for fitting
                    xs = np.array([p[0] for p in self.trajectory_3d])[:min_len]
                    ys = np.array([p[1] for p in self.trajectory_3d])[:min_len]
                    zs = np.array([p[2] for p in self.trajectory_3d])[:min_len]
                    
                    # --- NEW: Flight Period Extraction (Fixed) ---
                    # Find the exact moment of the kick. Look for sudden spike in Z velocity.
                    start_idx = 0
                    for i in range(len(ts) - 1):
                        dz = zs[i+1] - zs[i]
                        dt = ts[i+1] - ts[i]
                        vz = abs(dz / dt) if dt > 0 else 0
                        if vz > 1.5:  # Kick threshold
                            start_idx = i
                            break
                            
                    # Protect against completely flat runs
                    if len(ts) - start_idx < 5:
                        start_idx = max(0, len(ts) - 10)
                        
                    ts = ts[start_idx:]
                    xs = xs[start_idx:]
                    ys = ys[start_idx:]
                    zs = zs[start_idx:]
                    
                    if len(ts) < 3:
                        raise Exception("Not enough flight frames")
                    
                    # --- CRITICAL FIX: Time Normalization ---
                    # Start t from 0
                    ts_local = ts - ts[0]

                    # Fit Polynomials 
                    # X, Y, Z coordinates over time. Z and Y usually have quadratic shapes (gravity/drag)
                    # p(t) = a*t^2 + v0*t + p0 
                    # The velocity v(t) = 2*a*t + v0. Initial velocity is v(0) = v0 (which is coeff[1])
                    
                    z_coeffs = np.polyfit(ts_local, zs, 2)
                    y_coeffs = np.polyfit(ts_local, ys, 2)
                    x_coeffs = np.polyfit(ts_local, xs, 1) # X changes linearly (mostly)
                    
                    # Initial velocities from the coefficients (m/s)
                    vz_fit = abs(z_coeffs[1]) # v0 is the t^1 coefficient
                    vy_fit = abs(y_coeffs[1])
                    vx_fit = abs(x_coeffs[0]) # For linear fit, v is the t^1 coefficient
                    
                    # Calculate true 3D Initial Speed Vector
                    v_initial = np.sqrt(vx_fit**2 + vy_fit**2 + vz_fit**2)
                    
                    # --- NEW: Robust Instantaneous Speed Measurement ---
                    # Global curve fitting (polyfit) is too sensitive to YOLO bounding box noise
                    # and the exact moment the track starts.
                    # We will calculate frame-to-frame velocity and take the smoothed maximum.
                    
                    inst_velocities = []
                    for i in range(1, len(ts_local)):
                        dt = ts_local[i] - ts_local[i-1]
                        if dt <= 0: continue
                        
                        dx = xs[i] - xs[i-1]
                        dy = ys[i] - ys[i-1]
                        dz = zs[i] - zs[i-1]
                        
                        v_3d = np.sqrt(dx**2 + dy**2 + dz**2) / dt
                        inst_velocities.append(v_3d * 3.6) # Convert m/s -> km/h
                        
                    # Apply a 3-point Simple Moving Average (SMA) to remove noise spikes
                    if len(inst_velocities) >= 3:
                        smoothed_velocities = np.convolve(inst_velocities, np.ones(3)/3, mode='valid')
                        # True initial speed must happen at the beginning of the flight (first 15 frames ~ 0.5s)
                        # Any huge spike at the end is a tracking glitch (e.g., hitting the net).
                        search_window = min(15, len(smoothed_velocities))
                        max_speed = np.max(smoothed_velocities[:search_window])
                    elif len(inst_velocities) > 0:
                        max_speed = np.max(inst_velocities)
                    else:
                        max_speed = 0.0
                        
                    print(f"[DEBUG] Max Instantaneous Smoothed Speed: {max_speed:.1f} km/h")
                    
                    final_speed = max_speed
                    
                    # --- Compare with robust Linear Fit ---
                    # Linear is average speed over the whole flight.
                    # Real initial speed is always higher due to air drag.
                    x_slope, _ = np.polyfit(ts_local, xs, 1)
                    y_slope, _ = np.polyfit(ts_local, ys, 1)
                    z_slope, _ = np.polyfit(ts_local, zs, 1)
                    v_linear = np.sqrt(x_slope**2 + y_slope**2 + z_slope**2) * 3.6
                    
                    print(f"[DEBUG] Average Linear Speed: {v_linear:.1f} km/h")
                    
                    # Heuristic: If Max speed is wildly out of proportion to average (e.g., > 5.0x due to a glitch)
                    if final_speed > v_linear * 5.0 and v_linear > 30:
                        print(f"[INFO] Max speed outlier detected. Falling back to linear + drag.")
                        duration = ts_local[-1]
                        drag_compensation = 1.0 + (0.05 * duration)
                        final_speed = v_linear * drag_compensation

                    print(f"[Results] Final Processed Speed: {final_speed:.1f} km/h")
                    
                except Exception as e:
                    print(f"[Warning] Physics Fit Failed: {e}, using raw max velocity.")
                    if len(self.velocities) > 0:
                        final_speed = np.max(self.velocities) * 3.6
            else:
                 if len(self.velocities) > 0:
                     final_speed = np.max(self.velocities) * 3.6
                 else:
                     final_speed = 0.0
                     
        # Cap physics limits (Amateurs rarely exceed 130 km/h, Pros ~150 km/h)
        # We cap at 150 to keep it somewhat realistic while still allowing fast hits.
        if final_speed > 150: final_speed = 150

        # --- Final Calibration ---
        final_speed = float(round(final_speed, 1)) # Explicit float round
        
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
                
                
        comment = f"{s_tier}({final_speed:.1f}km/h){r_tier}"
        if final_speed > 100 and is_goal:
            comment = "와우! 손흥민 선수가 울고 갈 완벽한 슈팅입니다! 🏆"
            
        if self.callback: self.callback(100)
        
        return {
            "score": int(total_score),
            "speed": float(f"{final_speed:.1f}"), # Force clean float representation
            "comment": comment,
            "trajectory_image": os.path.basename(output_path)
        }

def process_video_with_progress(video_path, output_path, callback=None):
    print(f"[MAIN] Processing started for {video_path}")
    analyzer = ProShootingAnalyzer(callback)
    return analyzer.run(video_path, output_path)
