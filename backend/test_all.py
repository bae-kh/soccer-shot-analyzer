import os
import glob
from analyze_shot import ProShootingAnalyzer

def test_all_videos():
    base_dir = "backend/uploads"
    videos = [
        f"{base_dir}/video2.mp4",
        f"{base_dir}/video4.mp4"
    ]

    print("Starting batch analysis on videos...\n")
    print(f"{'Video Name':<15} | {'Max Speed (km/h)':<18} | {'Avg Speed':<15} | {'Goal?':<8}")
    print("-" * 65)

    for video_path in sorted(videos):
        analyzer = ProShootingAnalyzer()
        vid_name = os.path.basename(video_path)
        
        try:
            final_result = analyzer.run(video_path, "dummy.mp4")
                
            if isinstance(final_result, dict):
                max_speed = final_result.get('speed', 0)
                score = final_result.get('score', 0)
                is_goal = score > 0
                
                print(f"{vid_name:<15} | {max_speed:<18.2f} | -               | {str(is_goal):<8}")
            else:
                 print(f"{vid_name:<15} | No results returned")
            
        except Exception as e:
            print(f"{vid_name:<15} | ERROR: {e}")

if __name__ == "__main__":
    test_all_videos()
