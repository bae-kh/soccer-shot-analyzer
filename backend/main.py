
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import os
import asyncio
import json
from analyze_shot import process_video_with_progress

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
async def analyze_stream(file: UploadFile = File(...)):
    print(f"[INFO] New Request Received: {file.filename}")
    # 1. Save File
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_image_path = file_path.replace(".mp4", "_result.jpg")
    
    # Queue for Progress
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def sync_callback(progress):
        # Thread-safe put
        loop.call_soon_threadsafe(queue.put_nowait, progress)

    async def event_generator():
        # Run synchronous analysis in a separate thread
        future = loop.run_in_executor(None, process_video_with_progress, file_path, output_image_path, sync_callback)
        
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

@app.post("/calibrate")
async def calibrate_stream(file: UploadFile = File(...)):
    print(f"[INFO] New Calibration Request: {file.filename}")
    # 1. Save File
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    output_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    
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
                 yield f"data: {json.dumps({'type': 'progress', 'value': 100})}\n\n"
                 yield f"data: {json.dumps({'type': 'result', 'data': {'status': 'success', 'message': 'Calibration Complete!'}})}\n\n"
             else:
                 yield f"data: {json.dumps({'type': 'error', 'message': 'Calibration Failed. Please ensure the video clearly shows a 9x6 (internal corners) chessboard.'})}\n\n"
                 
        except Exception as e:
             print(f"Calibration Error: {e}")
             yield f"data: {json.dumps({'type': 'error', 'message': str(e) })}\n\n"
             
    return StreamingResponse(event_generator(), media_type="text/event-stream")
