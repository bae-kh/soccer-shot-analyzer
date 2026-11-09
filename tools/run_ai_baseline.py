"""Run the service analyzer in an isolated process for each sample video."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("yolov8s.pt", "goal_segment_best.pt")
FAILURES = {
    "공을 찾지 못했습니다.": "BALL_NOT_FOUND",
    "오류: 추적된 공 궤적이 너무 짧습니다.": "TRACK_TOO_SHORT",
    "오류: 궤적 피팅에 실패했습니다.": "FIT_FAILED",
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def video_key(path, input_dir):
    return path.relative_to(input_dir).with_suffix("")


def classify_result(result, image_ok):
    if not isinstance(result, dict):
        return "FAILED", "INVALID_RESULT"
    measurement = result.get('measurement_status')
    if measurement == 'UNAVAILABLE':
        if result.get('speed') is not None:
            return 'FAILED', 'INVALID_RESULT'
        if not image_ok:
            return 'FAILED', 'OUTPUT_IMAGE_MISSING'
        return 'UNAVAILABLE', result.get('failure_reason') or 'MEASUREMENT_UNAVAILABLE'
    if result.get("comment") in FAILURES:
        return "FAILED", FAILURES[result["comment"]]
    speed = result.get("speed")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
        return "FAILED", "INVALID_SPEED"
    if not image_ok:
        return "FAILED", "OUTPUT_IMAGE_MISSING"
    return ("ESTIMATED" if measurement == 'ESTIMATED' else "SUCCESS"), None


def preflight():
    import cv2
    import torch
    import ultralytics
    import scipy

    if not hasattr(cv2, "TrackerCSRT_create"):
        raise RuntimeError("OpenCV TrackerCSRT_create is missing; install the matching contrib build.")
    for name in MODEL_NAMES:
        if not (ROOT / "backend" / name).is_file():
            raise RuntimeError(f"Required local model missing: backend/{name}")
    packages = {}
    for name in ("numpy", "scipy", "torch", "torchvision", "ultralytics", "opencv-python", "opencv-contrib-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.executable, "python_version": sys.version,
        "packages": packages, "csrt_available": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_worker(video, image, result_path):
    started = time.perf_counter()
    record = {"status": "FAILED", "failure_reason": None, "raw_result": None}
    try:
        import cv2
        sys.path.insert(0, str(ROOT / "backend"))
        from analyze_shot import process_video_with_progress

        cap = cv2.VideoCapture(str(video))
        try:
            readable, _ = cap.read()
            record["video_metadata"] = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS), "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
        finally:
            cap.release()
        if not readable:
            record["failure_reason"] = "VIDEO_UNREADABLE"
        else:
            image.parent.mkdir(parents=True, exist_ok=True)
            progress = []
            record["raw_result"] = process_video_with_progress(str(video), str(image), progress.append)
            record["progress_last"] = progress[-1] if progress else None
            image_ok = image.is_file() and cv2.imread(str(image)) is not None
            record["status"], record["failure_reason"] = classify_result(record["raw_result"], image_ok)
            record["image_generated"] = image_ok
    except Exception as exc:
        traceback.print_exc()
        record.update(failure_reason="EXCEPTION", error=f"{type(exc).__name__}: {exc}")
    record["worker_time_sec"] = round(time.perf_counter() - started, 3)
    save_json(result_path, record)


def diagnostics(log_path):
    log = log_path.read_text(encoding="utf-8", errors="replace")
    count = re.search(r"Optimization with (\d+) points", log)
    focal = re.search(r"Absolute Calibrated F_px: ([\d.]+)", log)
    return {
        "tracked_point_count": int(count.group(1)) if count else None,
        "kick_detected": "Initialized at" in log,
        "goal_scale_detected": focal is not None,
        "goal_focal_px": float(focal.group(1)) if focal else None,
        "optimizer_error": next((line for line in log.splitlines() if "[Optimizer] Failed:" in line), None),
    }


def write_summary(output, metadata, records):
    counts = dict(Counter(row["status"] for row in records))
    reasons = dict(Counter(row["failure_reason"] for row in records if row["failure_reason"]))
    goal_counts = dict(Counter(row.get('goal_entry_status') for row in records))
    save_json(output / "summary.json", {"run": metadata, "counts": counts, "failure_counts": reasons,
                                      "goal_entry_counts": goal_counts, "videos": records})
    fields = ["video", "status", "failure_reason", "speed_kmh", "score", "elapsed_sec", "tracked_point_count", "tracking_status", "observed_points", "reacquisitions", "measurement_status", "flight_points", "model_error_px", "camera_motion", "goal_area_detected", "goal_scale_detected", "speed_capped", "goal_entry_status", "goal_entry_reason", "goal_entry_events", "image_path", "log_path"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    lines = ["# AI baseline", "", f"Completed: {len(records)}/{metadata['video_count']}",
             f"Statuses: {counts}", f"Failure reasons: {reasons}", "",
             "SUCCESS means a positive finite speed and a readable image, not correct tracking or physical accuracy.",
             "Stage 3: ESTIMATED means the conditional metric checks passed; UNAVAILABLE means speed is null with diagnostics/image retained.",
             "Metric estimates with assumed intrinsics/gravity are not calibrated speed measurements. No speed ground truth is available.",
             "Points is selected early-flight YOLO observations for stage 3, and legacy optimizer input count otherwise; it is not Track Coverage.",
             "elapsed_sec includes process startup/model loading, but excludes upload and UI.", "",
             f"Goal image-region statuses: {goal_counts}. Candidates are not confirmed goals; no candidate is not a miss.", "",
             "| Video | Status | Reason | km/h | Seconds | Points |", "|---|---|---|---:|---:|---:|"]
    for row in records:
        points = row.get('flight_points') if row.get('measurement_status') else row.get('tracked_point_count')
        lines.append(f"| {row['video']} | {row['status']} | {row['failure_reason'] or '-'} | {row.get('speed_kmh')} | {row['elapsed_sec']} | {points} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "soccer_videos")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=240, help="Per-video wall-clock timeout in seconds")
    parser.add_argument("--limit", type=int, help="Smoke run the first N videos")
    parser.add_argument("--worker", nargs=3, metavar=("VIDEO", "IMAGE", "RESULT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(*(Path(path) for path in args.worker))
        return 0
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeout and limit must be positive")
    input_dir = args.input_dir.resolve()
    videos = sorted((p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}),
                    key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.relative_to(input_dir).as_posix())])
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        parser.error(f"No videos found: {input_dir}")
    output = (args.output_dir or ROOT / "output" / "ai_baseline" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runtime" / "yolo").mkdir(parents=True)
    (output / "runtime" / "matplotlib").mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(output / "runtime" / "yolo"), MPLCONFIGDIR=str(output / "runtime" / "matplotlib"),
                      YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        environment = preflight()
    except Exception as exc:
        save_json(output / "preflight_error.json", {"error": f"{type(exc).__name__}: {exc}", "python": sys.executable})
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "working_directory": str(ROOT),
        "input_directory": str(input_dir), "video_count": len(videos), "timeout_sec": args.timeout,
        "environment": environment, "analyzer": fingerprint(ROOT / "backend" / "analyze_shot.py"),
        "tracking_modules": [fingerprint(ROOT / 'backend' / name) for name in ['ball_tracking.py', 'ball_observation.py', 'ball_size_quality.py', 'goal_geometry.py', 'goal_entry.py', 'goal_measurement.py', 'automatic_analysis.py', 'camera_geometry.py', 'capture_context.py', 'shot_physics.py', 'trajectory_render.py', 'calibrate.py'] if (ROOT / 'backend' / name).is_file()],
        "capture_conditions": fingerprint(ROOT / 'docs/video_capture_conditions.json'),
        "camera_profiles": [fingerprint(p) for p in sorted((ROOT / 'backend/camera_profiles').glob('*.json'))],
        "models": [fingerprint(ROOT / "backend" / name) for name in MODEL_NAMES],
        "calibration": {str(p.relative_to(ROOT)): fingerprint(p) if p.is_file() else None for p in [ROOT / "calibration.json", ROOT / "backend" / "calibration.json"]},
        "inputs": [fingerprint(p) for p in videos],
    }
    save_json(output / "run_metadata.json", metadata)
    records = []
    print(f"Running {len(videos)} videos; output: {output}", flush=True)
    for index, video in enumerate(videos, 1):
        key = video_key(video, input_dir)
        log_path = output / "logs" / f"{key}.log"
        result_path = output / "results" / f"{key}.json"
        image = output / "images" / f"{key}.jpg"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f"[{index}/{len(videos)}] START {video.relative_to(input_dir).as_posix()}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            try:
                child = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker", str(video), str(image), str(result_path)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                record = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
                    "status": "FAILED", "failure_reason": "WORKER_CRASH", "exit_code": child.returncode}
                if child.returncode:
                    record.update(status="FAILED", failure_reason="WORKER_CRASH", exit_code=child.returncode)
            except subprocess.TimeoutExpired:
                record = {"status": "FAILED", "failure_reason": "TIMEOUT"}
        raw = record.get("raw_result")
        if not isinstance(raw, dict):
            raw = {}
        tracking = raw.get("tracking_summary", {})
        physics = raw.get('physics_diagnostics', {})
        record.update(video=video.relative_to(input_dir).as_posix(), elapsed_sec=round(time.perf_counter() - started, 3),
                      speed_kmh=raw.get("speed"), score=raw.get("score"),
                      speed_capped=not raw.get('measurement_status') and raw.get("speed") == 150.0,
                      tracking_status=tracking.get("tracking_status"), observed_points=tracking.get("observed_points"),
                      reacquisitions=tracking.get("reacquisitions"),
                      measurement_status=raw.get('measurement_status'), flight_points=physics.get('flight_points', raw.get('flight_segment', {}).get('point_count')),
                      model_error_px=physics.get('median_reprojection_error_px'),
                      camera_motion=raw.get('camera_motion', {}).get('status'), goal_area_detected=raw.get('goal_area') is not None,
                      goal_entry_status=raw.get('goal_entry', {}).get('status'),
                      goal_entry_reason=raw.get('goal_entry', {}).get('reason'),
                      goal_entry_events=len(raw.get('goal_entry', {}).get('events', [])),
                      image_path=str(image) if record.get("image_generated") else None, log_path=str(log_path))
        record.update(diagnostics(log_path))
        save_json(result_path, record)
        records.append(record)
        write_summary(output, metadata, records)
        print(f"[{index}/{len(videos)}] {record['status']} {record['failure_reason'] or ''} speed={record['speed_kmh']} seconds={record['elapsed_sec']}", flush=True)
    print(f"Finished: {dict(Counter(r['status'] for r in records))}; see {output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
