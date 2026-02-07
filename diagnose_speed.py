
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

def diagnose_speed(video_path):
    print(f"--- Diagnosing: {video_path} ---")
    model = YOLO('yolov8n.pt')
    if os.path.exists('yolov8s.pt'):
        model = YOLO('yolov8s.pt')
        
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Camera Intrinsics Assumption
    fov_h_deg = 60
    F_px = (width / 2) / np.tan(np.deg2rad(fov_h_deg/2))
    REAL_RADIUS_M = 0.11 # 22cm diameter
    
    raw_data = [] # frame, u, v, r_px, conf
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize for consistent detection like app
        PROCESS_W = 1280
        scale = PROCESS_W / width
        PROCESS_H = int(height * scale)
        frame_process = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        
        results = model(frame_process, verbose=False, conf=0.1)
        
        best_r = 0
        best_uv = None
        hit = False
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 32: # Sports ball
                    u, v, w, h = box.xywh[0].cpu().numpy()
                    r_px = (w + h) / 4.0
                    conf = float(box.conf[0])
                    
                    if r_px > best_r:
                        best_r = r_px
                        best_uv = (u, v)
                        hit = True
                        
        if hit:
            raw_data.append((frame_idx, best_uv[0], best_uv[1], best_r))
        
        frame_idx += 1
        if frame_idx % 10 == 0: print(f"Processing frame {frame_idx}...")
        
    cap.release()
    
    if not raw_data:
        print("No ball detected.")
        return

    # --- Analysis ---
    frames = [d[0] for d in raw_data]
    Rs = [d[3] for d in raw_data]
    
    # Calculate Raw Depth (Z)
    # Z = (F * real_R) / img_R
    # Note: img_R is in Process coordinates, so use F_proc
    F_proc = F_px * scale
    Zs = [(F_proc * REAL_RADIUS_M) / r for r in Rs]
    
    # Calculate Raw Speed (Point to Point)
    speeds = []
    for i in range(1, len(Zs)):
        dt = (frames[i] - frames[i-1]) / fps
        dz = Zs[i] - Zs[i-1] # Dominant factor usually
        # Simplified 1D Z-speed for diagnosis
        v = abs(dz / dt) * 3.6 # km/h
        speeds.append(v)
        
    print(f"\nStats:")
    print(f"Mean Speed (Raw): {np.mean(speeds):.1f} km/h")
    print(f"Max Speed (Raw): {np.max(speeds):.1f} km/h")
    print(f"Std Dev Speed: {np.std(speeds):.1f}")
    
    # --- Curve Fitting Proposal ---
    # Fit a line to Z(t) = Z0 + Vz * t
    times = [f/fps for f in frames]
    
    # Linear Fit (Constant Velocity assumption for short duration)
    # Z = m*t + c
    coeffs = np.polyfit(times, Zs, 1) # Degree 1
    fitted_Vz = abs(coeffs[0]) # m/s
    fitted_speed_kmh = fitted_Vz * 3.6
    
    print(f"\n[Proposed Solution] Global Curve Fitting:")
    print(f"Slope (Vz): {fitted_Vz:.2f} m/s")
    print(f"Robust Speed: {fitted_speed_kmh:.1f} km/h")
    
    # Check R-squared (Quality of fit)
    p = np.poly1d(coeffs)
    yhat = p(times)
    ybar = np.sum(Zs)/len(Zs)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((Zs-ybar)**2)
    r2 = ssreg / sstot
    print(f"Fit Quality (R^2): {r2:.4f}")
    
    return fitted_speed_kmh

# Run on the problematic video
if __name__ == "__main__":
    # Use video14 as reported in logs
    target = r"c:\Users\qorkd\.gemini\축구슈팅측정\backend\backend\uploads\video14.mp4"
    if os.path.exists(target):
        diagnose_speed(target)
    else:
        # Search for any
        import glob
        files = glob.glob(r"c:\Users\qorkd\.gemini\축구슈팅측정\backend\backend\uploads\*.mp4")
        if files:
            diagnose_speed(files[-1])
        else:
            print("No video found for diagnosis.")
