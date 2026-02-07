
import cv2
import numpy as np
import os
from ultralytics import YOLO

def is_valid_goal(cnt, width, height, max_ratio_limit):
    area = cv2.contourArea(cnt)
    if area < (width * height) * 0.005: 
        return False, f"Too Small (Area={area:.0f})", 0

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w < h: w, h = h, w; angle += 90

    ratio = w / (h + 1e-5)

    if ratio < 1.3 or ratio > max_ratio_limit:
        return False, f"Ratio {ratio:.2f}", 0

    # Angle normalization
    if angle > 45: angle -= 90
    if angle < -45: angle += 90
    
    if abs(angle) > 20: 
        return False, f"Angle {angle:.1f}", 0

    screen_center_x = width / 2
    dist_from_center = abs(cx - screen_center_x)
    if dist_from_center > (width * 0.35):
        return False, f"Far Side (Dist={dist_from_center:.0f})", dist_from_center

    return True, "Pass", dist_from_center

def scan_for_goal(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n--- Analyzing {os.path.basename(video_path)} ({width}x{height}) ---")

    # --- v21 Strict Logic ---
    print("Trying v21 (Strict)...")
    accumulated_mask = np.zeros((height, width), dtype=np.uint8)
    scan_limit = 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_cnt = 0
    mask_found_cnt = 0
    while frame_cnt < scan_limit:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, imgsz=640, conf=0.05, verbose=False)
        if results[0].masks:
            mask_found_cnt += 1
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in results[0].masks.xy:
                if len(seg) > 0:
                    cv2.fillPoly(temp_mask, [np.int32(seg)], 255)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, temp_mask)
        frame_cnt += 1
        
    print(f"  > Scanned {frame_cnt} frames, masks found in {mask_found_cnt} frames.")

    kernel_split = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    accumulated_mask = cv2.morphologyEx(accumulated_mask, cv2.MORPH_OPEN, kernel_split)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    accumulated_mask = cv2.dilate(accumulated_mask, kernel_smooth, iterations=1)

    contours, _ = cv2.findContours(accumulated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  > Found {len(contours)} contours.")
    
    best_cnt = None
    min_dist = 99999
    
    for i, cnt in enumerate(contours):
        valid, msg, dist = is_valid_goal(cnt, width, height, 3.5)
        print(f"    - Contour {i}: {msg}")
        if valid:
            if dist < min_dist:
                min_dist = dist
                best_cnt = cnt
                
    if best_cnt is not None:
        print("✅ v21 Found Goal!")
        return

    # --- v22 Rescue Logic ---
    print("Trying v22 (Rescue)...")
    accumulated_mask = np.zeros((height, width), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_cnt = 0
    mask_found_cnt = 0
    while frame_cnt < scan_limit:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, imgsz=640, conf=0.01, verbose=False) # High sensitivity
        if results[0].masks:
            mask_found_cnt += 1
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in results[0].masks.xy:
                if len(seg) > 0:
                    cv2.fillPoly(temp_mask, [np.int32(seg)], 255)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, temp_mask)
        frame_cnt += 1
        
    print(f"  > Scanned {frame_cnt} frames, masks found in {mask_found_cnt} frames.")

    kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    accumulated_mask = cv2.dilate(accumulated_mask, kernel_base, iterations=2)
    kernel_gentle = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    accumulated_mask = cv2.morphologyEx(accumulated_mask, cv2.MORPH_OPEN, kernel_gentle)
    kernel_heal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    accumulated_mask = cv2.dilate(accumulated_mask, kernel_heal, iterations=1)

    contours, _ = cv2.findContours(accumulated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  > Found {len(contours)} contours.")
    
    best_cnt = None
    min_dist = 99999
    
    for i, cnt in enumerate(contours):
        valid, msg, dist = is_valid_goal(cnt, width, height, 5.0)
        print(f"    - Contour {i}: {msg}")
        if valid:
            if dist < min_dist:
                min_dist = dist
                best_cnt = cnt
                
    if best_cnt is not None:
        print("✅ v22 Found Goal!")
    else:
        print("❌ Failed in v22 as well.")

if __name__ == "__main__":
    try:
        print("Checking imports...")
        import torch
        print(f"Torch: {torch.__version__}")
        
        if os.path.exists("goal_segment_best.pt"):
            print("Loading Model...")
            model = YOLO("goal_segment_best.pt")
            
            # Find all MP4s
            video_files = []
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith(".mp4"):
                        video_files.append(os.path.join(root, file))
            
            print(f"Found {len(video_files)} videos: {video_files}")

            for path in video_files:
                # Run scan and visually debug
                print(f"\nProcessing {path}...")
                cap = cv2.VideoCapture(path)
                if not cap.isOpened(): continue
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # --- v22 Logic Simulation with Visuals ---
                accumulated_mask = np.zeros((height, width), dtype=np.uint8)
                scan_limit = 45
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                first_frame = None
                
                frame_cnt = 0
                while frame_cnt < scan_limit:
                    ret, frame = cap.read()
                    if not ret: break
                    if first_frame is None: first_frame = frame.copy()
                    
                    results = model(frame, imgsz=640, conf=0.05, verbose=False)
                    if results[0].masks:
                        temp_mask = np.zeros((height, width), dtype=np.uint8)
                        for seg in results[0].masks.xy:
                            if len(seg) > 0:
                                cv2.fillPoly(temp_mask, [np.int32(seg)], 255)
                        accumulated_mask = cv2.bitwise_or(accumulated_mask, temp_mask)
                    frame_cnt += 1
                
                # Morphology
                kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                accumulated_mask = cv2.dilate(accumulated_mask, kernel_base, iterations=2)
                
                contours, _ = cv2.findContours(accumulated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw on first frame
                debug_img = first_frame.copy() if first_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
                
                found_any = False
                for i, cnt in enumerate(contours):
                    valid, msg, dist = is_valid_goal(cnt, width, height, 5.0)
                    color = (0, 255, 0) if valid else (0, 0, 255)
                    cv2.drawContours(debug_img, [cnt], -1, color, 2)
                    
                    # Draw label
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(debug_img, f"{i}:{msg}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    if valid: found_any = True
                
                status = "SUCCESS" if found_any else "FAIL"
                print(f"Result for {path}: {status}")
                
                # Save debug image
                out_name = f"debug_{os.path.basename(path)}.jpg"
                cv2.imwrite(out_name, debug_img)
                print(f"Saved {out_name}")

        else:
            print("goal_segment_best.pt not found")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
