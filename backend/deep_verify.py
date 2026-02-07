
import cv2
import numpy as np
import os
from ultralytics import YOLO
import sys

# Target Files
VIDEO_PATH = r"C:\Users\qorkd\.gemini\축구슈팅측정\backend\backend\uploads\video4.mp4"
MODEL_PATH = r"C:\Users\qorkd\.gemini\축구슈팅측정\backend\goal_segment_best.pt"

if not os.path.exists(VIDEO_PATH):
    # Fallback search
    for root, dirs, files in os.walk("."):
        if "video4.mp4" in files:
            VIDEO_PATH = os.path.join(root, "video4.mp4")
            break

print(f"Video: {VIDEO_PATH}")
print(f"Model: {MODEL_PATH}")

def is_valid_goal(cnt, width, height, max_ratio_limit):
    area = cv2.contourArea(cnt)
    if area < (width * height) * 0.005: 
        return False, f"Too Small {area:.0f}", 0

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w < h: w, h = h, w; angle += 90

    ratio = w / (h + 1e-5)

    if ratio < 1.3 or ratio > max_ratio_limit:
        return False, f"Ratio {ratio:.2f}", 0

    if angle > 45: angle -= 90
    if abs(angle) > 20: 
        return False, f"Angle {angle:.1f}", 0

    screen_center_x = width / 2
    dist_from_center = abs(cx - screen_center_x)
    if dist_from_center > (width * 0.35):
        return False, f"Far Side {dist_from_center:.0f}", dist_from_center

    return True, "Pass", dist_from_center

def run_debug():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found")
        return
        
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- V21 Logic Simulation ---
    print("\n--- Running V21 Logic Debug ---")
    accumulated_mask = np.zeros((height, width), dtype=np.uint8)
    scan_limit = 60
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_cnt = 0
    mask_hits = 0
    while frame_cnt < scan_limit:
        ret, frame = cap.read()
        if not ret: break
        
        # User V21: conf=0.05
        results = model(frame, imgsz=640, conf=0.05, verbose=False)
        if results[0].masks:
            mask_hits += 1
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in results[0].masks.xy:
                if len(seg) > 0:
                    cv2.fillPoly(temp_mask, [np.int32(seg)], 255)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, temp_mask)
        frame_cnt += 1
        
    print(f"Frames Scanned: {frame_cnt}")
    print(f"Frames with Detection (conf=0.05): {mask_hits}")
    
    # Save Accum Mask
    cv2.imwrite("debug_v21_acc_mask.jpg", accumulated_mask)
    
    if mask_hits == 0:
        print("❌ No detections at all. Model failing on these frames with conf=0.05.")
        return

    # Morphology
    kernel_split = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    acc_mask_open = cv2.morphologyEx(accumulated_mask, cv2.MORPH_OPEN, kernel_split)
    
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    acc_mask_final = cv2.dilate(acc_mask_open, kernel_smooth, iterations=1)
    
    cv2.imwrite("debug_v21_final_mask.jpg", acc_mask_final)
    
    contours, _ = cv2.findContours(acc_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours found after morph: {len(contours)}")
    
    best_cnt = None
    min_dist = 99999

    for i, cnt in enumerate(contours):
        valid, msg, dist = is_valid_goal(cnt, width, height, max_ratio_limit=3.5)
        print(f"Cnt {i}: {msg}")
        if valid:
            if dist < min_dist:
                min_dist = dist
                best_cnt = cnt
                
    if best_cnt is not None:
        print("✅ V21 SUCCESS")
    else:
        print("❌ V21 FAILED")

if __name__ == "__main__":
    run_debug()
