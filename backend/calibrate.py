import cv2
import numpy as np
import argparse
import os
import json

def calibrate_camera(video_path, output_json="calibration.json", chessboard_size=(9, 6), square_size_m=0.025, max_frames=50):
    """
    Calibrate a camera using a video of a chessboard sequence.
    
    Args:
        video_path: Path to the calibration video.
        output_json: Path to save the intrinsic parameters (K, D).
        chessboard_size: Tuple (corners_x, corners_y). Note: internal corners, not squares.
        square_size_m: Real-world size of a single chessboard square in meters.
        max_frames: Max number of frames to sample from the video.
    """
    print(f"[INFO] Starting Camera Calibration on: {video_path}")
    
    # termination criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare expected object points in 3D (0,0,0), (1,0,0), (2,0,0) ...
    # Multiplying by square_size_m puts them in metric space
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_m
    
    # Arrays to store object points and image points from all valid frames
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames in video: {total_frames}")
    
    # Calculate skip step to reasonably sample max_frames throughout the video
    skip_step = max(1, total_frames // max_frames)
    
    frame_idx = 0
    valid_captures = 0
    img_size = None
    
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Hard limits to prevent hanging the API/Frontend for minutes
        if time.time() - start_time > 45:
            print("[ERROR] Calibration timeout (45s). Video might be too complex or not a chessboard.")
            break
        if frame_idx > total_frames or frame_idx > 1000:
            break
            
        if frame_idx % skip_step == 0:
            original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = original_gray.shape[::-1] # (width, height)
                
            # Perform findChessboardCorners on a downscaled image for speed
            # High-res images without chessboards take SECONDS per frame
            MAX_WIDTH = 1000
            scale_ratio = 1.0
            if img_size[0] > MAX_WIDTH:
                scale_ratio = MAX_WIDTH / img_size[0]
                new_width = MAX_WIDTH
                new_height = int(img_size[1] * scale_ratio)
                gray_small = cv2.resize(original_gray, (new_width, new_height))
            else:
                gray_small = original_gray
                
            # Find the chessboard corners (Fast flag helps reject empty frames quickly)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            ret_corners, corners_small = cv2.findChessboardCorners(gray_small, chessboard_size, flags)
            
            # If found, add object points, image points (after refining them on ORIGINAL image)
            if ret_corners:
                objpoints.append(objp)
                
                # Scale corners back to original image size
                corners_orig = corners_small / scale_ratio
                
                # Refine pixel coordinates for accuracy on the ORIGINAL high-res grayscale image
                corners2 = cv2.cornerSubPix(original_gray, corners_orig, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                valid_captures += 1
                
                # Draw and display for debug (optional, can be disabled)
                # cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_corners)
                # cv2.imshow('Calibration', frame)
                # cv2.waitKey(1)
                
                print(f"[INFO] Frame {frame_idx}: Chessboard Found! (Total: {valid_captures})")
            else:
                pass
                
        frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    if img_size is None or valid_captures < 5:
         print(f"[ERROR] Not enough valid chessboard frames found ({valid_captures}/5 required). Please record a slower, clearer video covering multiple angles.")
         return False
         
    print(f"\n[INFO] Calibration using {valid_captures} valid images...")
    
    if len(objpoints) == 0 or len(imgpoints) == 0:
         print("[ERROR] No points extracted for calibration.")
         return False
         
    # Perform Calibration 
    # Mtx: Camera matrix (Intrinsic parameters K)
    # dist: Distortion coefficients
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        
        if ret:
            print("\n=== Calibration Successful ===")
            print(f"RMS Re-projection Error: {ret:.4f}")
            print("\nCamera Matrix (K):\n", mtx)
            print("\nDistortion Coefficients:\n", dist)
            
            # Save to JSON
            calib_data = {
                "camera_matrix": mtx.tolist(),
                "dist_coeffs": dist.tolist(),
                "rms_error": ret,
                "img_width": img_size[0],
                "img_height": img_size[1]
            }
            
            try:
                 with open(output_json, 'w') as f:
                     json.dump(calib_data, f, indent=4)
                 print(f"\n[SUCCESS] Calibration saved to {output_json}")
                 return True
            except Exception as e:
                 print(f"[ERROR] Failed to write JSON: {e}")
                 return False
        else:
            print("[ERROR] Internal Calibration Algorithm Failed.")
            return False
            
    except Exception as e:
        print(f"[ERROR] OpenCV calibrateCamera crashed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Camera using Chessboard Video")
    parser.add_argument("video", type=str, help="Path to the chessboard video.")
    parser.add_argument("--out", type=str, default="backend/calibration.json", help="Output JSON path")
    parser.add_argument("--corners_x", type=int, default=9, help="Number of internal corners in X (width)")
    parser.add_argument("--corners_y", type=int, default=6, help="Number of internal corners in Y (height)")
    parser.add_argument("--square_m", type=float, default=0.025, help="Square side length in meters (e.g., 0.025 for 2.5cm)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
         print(f"[ERROR] Video file not found: {args.video}")
         exit(1)
         
    calibrate_camera(
        video_path=args.video, 
        output_json=args.out, 
        chessboard_size=(args.corners_x, args.corners_y), 
        square_size_m=args.square_m
    )
