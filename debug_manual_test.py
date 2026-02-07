
import os
import sys

# Ensure backend module is visible
sys.path.append(os.path.join(os.getcwd(), "backend"))

from analyze_shot import SoccerAnalyzer

def test_analysis():
    video_path = r"C:\Users\qorkd\.gemini\축구슈팅측정\backend\backend\uploads\video3.mp4"
    output_path = video_path.replace(".mp4", "_manual_test.jpg")
    
    print(f"Testing analysis on: {video_path}")
    
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return

    analyzer = SoccerAnalyzer(callback=lambda p: print(f"Progress: {p}%"))
    result = analyzer.run(video_path, output_path)
    
    print("\n--- Analysis Result ---")
    print(result)

if __name__ == "__main__":
    test_analysis()
