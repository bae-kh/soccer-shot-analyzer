import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.analyze_shot import ProShootingAnalyzer

def test_batch():
    upload_dir = "backend/uploads"
    files = [f for f in os.listdir(upload_dir) if f.endswith(".mp4")]
    
    print(f"Found {len(files)} videos.")
    print("-" * 60)
    print(f"{'Filename':<40} | {'Score':<5} | {'Speed':<5}")
    print("-" * 60)

    for target in files:
        input_path = os.path.join(upload_dir, target)
        output_path = os.path.join(upload_dir, f"{target}_debug_result.jpg")
        
        try:
            # Instantiate manually like in debug_run.py
            analyzer = ProShootingAnalyzer()
            result = analyzer.run(input_path, output_path)
            
            score = result.get('score', 0)
            speed = result.get('speed', 0)
            print(f"{target:<40} | {score:<5} | {speed:<5}")
        except Exception as e:
            print(f"{target:<40} | ERROR | {e}")

if __name__ == "__main__":
    # Ensure backend is in path
    sys.path.append("backend")
    test_batch()
