import sys
import os

# Add backend directory to path so it can import analyze_shot
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
from analyze_shot import ProShootingAnalyzer

def test_debug():
    videos = ["video4.mp4", "video11.mp4"]
    for v in videos:
        analyzer = ProShootingAnalyzer()
        input_path = f"backend/backend/uploads/{v}"
        if not os.path.exists(input_path):
            input_path = f"backend/uploads/{v}"
            if not os.path.exists(input_path):
                print(f"[ERROR] Could not find {v}")
                continue
            
        output_path = f"backend/backend/uploads/{v.split('.')[0]}_debug_result.jpg"
        
        print(f"\n{'='*40}")
        print(f"--- Running Debug on {input_path} ---")
        
        try:
            res = analyzer.run(input_path, output_path)
            print("\n[Final Result]")
            print(res)
        except Exception as e:
            print(f"\n[Exception] {e}")

if __name__ == "__main__":
    test_debug()
