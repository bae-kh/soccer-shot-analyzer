
import os
import cv2
import sys

# Add current dir to path to find analyze_shot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyze_shot import ProShootingAnalyzer

def test_on_video(video_name):
    print(f"--- Testing {video_name} ---")
    analyzer = ProShootingAnalyzer()
    
    if not analyzer.goal_model_loaded:
        print("❌ CRITICAL: Goal Model NOT detected by ProShootingAnalyzer!")
        return

    video_path = os.path.join("uploads", video_name)
    if not os.path.exists(video_path):
        # Try parent dir uploads?
        video_path = os.path.join("../uploads", video_name)
        if not os.path.exists(video_path):
             print(f"❌ Video not found: {video_path}")
             return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}")

    print("Attempting v21 (Strict)...")
    cnt = analyzer.scan_for_goal_v21_logic(cap, analyzer.goal_model, width, height)
    
    if cnt is not None:
        print("✅ v21 Found Goal!")
    else:
        print("⚠️ v21 Failed. Attempting v22 (Rescue)...")
        cnt = analyzer.scan_for_goal_v22_logic(cap, analyzer.goal_model, width, height)
        
        if cnt is not None:
            print("✅ v22 Found Goal!")
        else:
            print("❌ v22 Failed. Total Failure.")

if __name__ == "__main__":
    # Hardcoded absolute path based on find_by_name result
    # Structure seems to be root/backend/backend/uploads/...
    target_video = r"C:\Users\qorkd\.gemini\축구슈팅측정\backend\backend\uploads\video4.mp4"
    
    if os.path.exists(target_video):
        print(f"Targeting confirmed video: {target_video}")
        test_on_video(target_video)
    else:
        print(f"❌ Absolute path not found: {target_video}")
        # Fallback search
        for root, dirs, files in os.walk("."):
             for f in files:
                 if f == "video4.mp4":
                     full_path = os.path.join(root, f)
                     print(f"Found alternative: {full_path}")
                     test_on_video(full_path)
                     exit()
