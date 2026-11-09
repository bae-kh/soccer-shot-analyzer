
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import os
import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional
from uuid import uuid4
from analyze_shot import process_video_with_progress
from capture_context import validate_options, list_profiles, PROFILE_DIR

app = FastAPI()

# CORS Setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_stream(file: UploadFile = File(...), capture_options: Annotated[Optional[str], Form()] = None):
    try:
        options = validate_options(json.loads(capture_options)) if capture_options is not None else None
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail=str(exc))
    print(f"[INFO] New Request Received: {file.filename}")
    # 1. Save File
    upload_id = uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, upload_id + Path(file.filename or 'video.mp4').suffix)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    base_name = upload_id
    output_image_path = os.path.join(UPLOAD_DIR, f"{base_name}_result.jpg")
    
    # Queue for Progress
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sync_callback(progress):
        # Thread-safe put
        loop.call_soon_threadsafe(queue.put_nowait, progress)

    async def event_generator():
        # Run synchronous analysis in a separate thread
        from functools import partial
        analyze = partial(process_video_with_progress, capture_options=options) if options is not None else process_video_with_progress
        future = loop.run_in_executor(None, analyze, file_path, output_image_path, sync_callback)
        
        while not future.done():
            try:
                # Wait for progress
                progress = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield f"data: {json.dumps({'type': 'progress', 'value': progress})}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                continue
        
        # Get Result
        try:
            result = await future
            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
        except Exception as e:
            print(f"Analysis Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e) })}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/results/{image_name}")
def get_image(image_name: str):
    from fastapi.responses import FileResponse
    path = os.path.join(UPLOAD_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Image not found"}

@app.get('/camera-profiles')
def camera_profiles():
    return {'profiles': list_profiles(PROFILE_DIR)}


@app.post("/calibrate")
async def calibrate_stream(file: UploadFile = File(...), profile_label: Annotated[str, Form()] = '카메라 설정'):
    if not profile_label.strip() or len(profile_label) > 120:
        raise HTTPException(422, detail='Profile label must contain 1–120 characters')
    print(f"[INFO] New Calibration Request: {file.filename}")
    # 1. Save File
    profile_id = 'camera_' + uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, profile_id + Path(file.filename or 'video.mp4').suffix)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    output_json_path = str(PROFILE_DIR / (profile_id + '.json'))
    
    # We will simulate SSE progress for calibration just to keep UI responsive
    # Real progress tracking in OpenCV calibrateCamera is hard, so we just run it async
    # and send a 'processing' event, then the result.
    
    async def event_generator():
        yield f"data: {json.dumps({'type': 'progress', 'value': 10})}\n\n"
        
        loop = asyncio.get_running_loop()
        from calibrate import calibrate_camera
        
        # Run calibration in thread
        yield f"data: {json.dumps({'type': 'progress', 'value': 50})}\n\n"
        
        try:
             # Default to 9x6 inner corners, 25mm square size
             future = loop.run_in_executor(None, calibrate_camera, file_path, output_json_path, (9, 6), 0.025, 50)
             
             while not future.done():
                 try:
                     await asyncio.wait_for(asyncio.shield(future), timeout=0.5)
                 except asyncio.TimeoutError:
                     yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                     continue
                     
             success = future.result()
             
             if success:
                 path = Path(output_json_path)
                 profile_data = json.loads(path.read_text(encoding='utf-8'))
                 profile_data.update(profile_id=profile_id, profile_label=profile_label.strip())
                 path.write_text(json.dumps(profile_data, ensure_ascii=False, indent=2), encoding='utf-8')
                 yield f"data: {json.dumps({'type': 'progress', 'value': 100})}\n\n"
                 yield f"data: {json.dumps({'type': 'result', 'data': {'status': 'success', 'profile_id': profile_id, 'profile_label': profile_label.strip(), 'message': 'Calibration profile saved'}})}\n\n"
             else:
                 yield f"data: {json.dumps({'type': 'error', 'message': 'Calibration Failed. Please ensure the video clearly shows a 9x6 (internal corners) chessboard.'})}\n\n"
                 
        except Exception as e:
             print(f"Calibration Error: {e}")
             yield f"data: {json.dumps({'type': 'error', 'message': str(e) })}\n\n"
             
    return StreamingResponse(event_generator(), media_type="text/event-stream")
