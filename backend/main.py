
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
