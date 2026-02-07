import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.analyze_shot import SoccerAnalyzer

def diagnose_failures():
    # List of reported problematic videos
    target_files = [
        "e34dd928-f188-4341-ba8c-072ca4003cfe.mp4",
        "c3bd806b-2336-4e83-bc92-bb06461bf4c4.mp4",
        "09515148-7bcd-41cf-ab40-60e22132f6ba.mp4",
        "0d67d7f2-b178-4a42-9831-17eb4a743128.mp4"
    ]
    
    upload_dir = "backend/uploads"
    
    print(f"Starting diagnosis on {len(target_files)} files...")
    print("-" * 60)
    
    for filename in target_files:
        input_path = os.path.join(upload_dir, filename)
        output_path = os.path.join(upload_dir, f"diagnose_{filename}_debug.jpg")
        
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (Not Found)")
            continue
            
        print(f"Analyzing {filename}...")
        
        try:
            analyzer = SoccerAnalyzer()
            # We want to see debug prints, so we won't capture stdout/stderr here, just let it print to console
            # or we could modify analyzer to return more debug info, but console log is fastest for now.
            result = analyzer.run(input_path, output_path)
            
            print(f"Result for {filename}:")
            print(f"  Score: {result.get('score')}")
            print(f"  Speed: {result.get('speed_kmh')} km/h")
            print(f"  Trajectory Points: {len(result.get('trajectory_image', ''))}") # Just checking if image generated
            
            # Since analyzer.run stores trajectory in self.trajectory, we can inspect it if we had access, 
            # but run() returns a dict.
            # Ideally we check analyzer.trajectory length here if we refactor, but for now lets rely on prints.
            
        except Exception as e:
            print(f"ERROR analyzing {filename}: {e}")
            
        print("-" * 60)

if __name__ == "__main__":
    diagnose_failures()
