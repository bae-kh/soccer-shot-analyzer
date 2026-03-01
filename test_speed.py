import sys
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.analyze_shot import ProShootingAnalyzer

def run_test(v):
    print(f"Testing {v}")
    analyzer = ProShootingAnalyzer()
    res = analyzer.run(v, "output_test.jpg")
    print("Final result:", res)
    
    print("Trajectory length:", len(analyzer.trajectory_3d))
    
    ts = np.array(analyzer.timestamps)
    min_len = min(len(ts), len(analyzer.trajectory_3d))
    if min_len > 6:
        ts = ts[:min_len]
        ts_local = ts - ts[0]
        zs = np.array([p[2] for p in analyzer.trajectory_3d])[:min_len]
        xs = np.array([p[0] for p in analyzer.trajectory_3d])[:min_len]
        ys = np.array([p[1] for p in analyzer.trajectory_3d])[:min_len]
        print("Time:", ts_local)
        print("X:", xs)
        print("Y:", ys)
        print("Z:", zs)
        
        z_coeffs = np.polyfit(ts_local, zs, 2)
        y_coeffs = np.polyfit(ts_local, ys, 2)
        x_coeffs = np.polyfit(ts_local, xs, 1)
        z_slope, _ = np.polyfit(ts_local, zs, 1)
        
        print("Quad Vz_init:", z_coeffs[1])
        print("Quad Vy_init:", y_coeffs[1])
        print("Linear Vz:", z_slope)
    else:
        print("Not enough points:", min_len)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_test(sys.argv[1])
    else:
        print("Provide video path")
