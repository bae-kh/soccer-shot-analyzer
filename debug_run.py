import os
from backend.analyze_shot import process_video_with_progress

import sys

def test_run():
    # Redirect output to file
    with open("debug_log.txt", "w") as log_file:
        sys.stdout = log_file
        
        # files = [f for f in os.listdir("backend/uploads") if f.endswith(".mp4")]
        # if not files:
        #     print("No video files found in backend/uploads")
        #     return

        # target = "f76c8899-43ea-4a97-8052-2d197ba2b506.mp4"
        target = "3c5b2e5d-e03e-4e08-be71-5b0fd46706c3.mp4"
        input_path = os.path.join("backend/uploads", target)
        output_path = os.path.join("backend/uploads", "test_debug_result.jpg")
        
        print(f"Testing with {target}...")
        
        def cb(p):
            # print(f"Progress: {p}%", end="\r") # Avoid cluttering log
            pass

        result = process_video_with_progress(input_path, output_path, cb)
        print("\nDone!")
        print(result)
        
    sys.stdout = sys.__stdout__ # Reset


if __name__ == "__main__":
    test_run()
