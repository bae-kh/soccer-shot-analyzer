
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Union
import shutil
import os
import time
import uuid
import asyncio
import json
import logging

# opencv와 ultralytics 같은 분석 라이브러리는 영상을 실제로 처리할 때만 필요하다.
# 명세 확인과 예시 작업 조회에는 쓰이지 않으므로, 설치되지 않은 환경에서도
# 서버가 뜨도록 모듈 최상단이 아니라 사용하는 자리에서 불러온다.

logger = logging.getLogger("shot_analyzer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    # uvicorn이 설정한 상위 로거로 올라가 같은 줄이 두 번 찍히는 것을 막는다
    logger.propagate = False

app = FastAPI(
    title="축구 슈팅 영상 분석 API",
    description=(
        "스마트폰으로 촬영한 축구 슈팅 영상을 분석한다.\n\n"
        "분석은 영상 업로드와 진행률 구독을 분리한다. 업로드하면 분석 id를 받고, "
        "그 id로 진행률을 구독하거나 결과를 조회한다. 연결이 끊겨도 id만 있으면 다시 붙을 수 있다.\n\n"
        "카메라 보정은 분석과 달리 스트리밍 없이 주기적으로 상태를 조회한다."
    ),
)

# CORS Setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov"}
# 업로드 영상과 결과 이미지 보관 기간. 이 기간이 곧 결과를 다시 조회할 수 있는 기간이다
RETENTION_DAYS = 7
# 업로드 제한. 프론트엔드가 5주차에 적용한 기준을 그대로 채택했다
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
# 스마트폰 세로 영상을 고려해 긴 변이 아닌 짧은 변을 기준으로 판단한다
MIN_SHORT_SIDE_PX = 480

# 보정은 여러 각도의 프레임이 필요해 분석보다 긴 영상을 요구한다
CALIBRATION_MIN_DURATION_SEC = 5.0
# 인쇄용 체스보드 규격. 프론트엔드 안내 화면과 같은 값을 써야 한다
CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_M = 0.025

# 분석 작업과 보정 작업 상태. 서버 메모리에만 두므로 재시작하면 사라진다
jobs = {}
calibrations = {}
TERMINAL_STATUSES = {"VALID", "ANALYSIS_FAILED", "ERROR"}

# 연동용 예시 작업. 기본은 꺼져 있고 SAMPLE_DATA=1 일 때만 채운다
if os.getenv("SAMPLE_DATA") == "1":
    from sample_data import seed_samples
    _job_ids, _calib_ids = seed_samples(jobs, calibrations)
    logger.info("예시 작업 활성화 — 분석 %s / 보정 %s", _job_ids, _calib_ids)


# --- 응답 형태 선언 -------------------------------------------------
# 실제 응답은 각 엔드포인트가 직접 만들어 돌려준다. 아래 모델은 문서용이며,
# 프론트엔드가 스웨거에서 필드 이름과 타입을 바로 확인할 수 있게 하기 위한 것이다.

class ErrorResponse(BaseModel):
    error_code: str
    message: str


class CreatedResponse(BaseModel):
    id: str = Field(description="분석 또는 보정 작업 id (uuid4)")


class TrajectoryPoint(BaseModel):
    x: float = Field(description="원본 영상 해상도 기준 가로 좌표")
    y: float = Field(description="원본 영상 해상도 기준 세로 좌표")
    frame: int = Field(description="0부터 시작하는 프레임 번호")


class FrameSize(BaseModel):
    width: int
    height: int


class AnalysisResult(BaseModel):
    status: str = Field(description="VALID 또는 ANALYSIS_FAILED")
    score: int = Field(description="슈팅 점수 0~100")
    speed_kmh: float = Field(description="추정 슈팅 속도 (km/h)")
    goal: bool = Field(description="골인 여부. 판정 로직 미구현으로 현재는 항상 false")
    trajectory: List[TrajectoryPoint] = Field(description="공 궤적 좌표 시퀀스")
    fps: float = Field(description="영상의 초당 프레임 수")
    frame_size: FrameSize = Field(description="좌표 기준 해상도")
    confidence: float = Field(description="분석 신뢰도 0~1. 추적 프레임 중 탐지 성공 비율")
    processing_time_sec: float = Field(description="분석 소요 시간 (초)")
    trajectory_image: str = Field(description="결과 오버레이 이미지 파일명")
    comment: str
    failure_reason: Optional[str] = Field(
        default=None,
        description="status가 ANALYSIS_FAILED일 때만 포함. "
                    "BALL_NOT_DETECTED, TRACK_TOO_SHORT, FIT_FAILED, LOW_CONFIDENCE",
    )


class AnalysisProgress(BaseModel):
    status: str = Field(description="아직 분석 중일 때 RUNNING")
    progress: int = Field(description="0~100")


class FocalLength(BaseModel):
    fx: float
    fy: float


class PrincipalPoint(BaseModel):
    cx: float
    cy: float


class CalibrationProgress(BaseModel):
    status: str = Field(description="보정 진행 중일 때 CALIBRATING")
    progress: int


class CalibrationResult(BaseModel):
    status: str = Field(description="보정 성공 시 VALID")
    frame_size: FrameSize = Field(description="보정에 사용한 해상도")
    used_frames: int = Field(description="체스보드를 찾아 계산에 사용한 프레임 수")
    reprojection_error: float = Field(description="재투영 오차(px)")
    focal_length: FocalLength
    principal_point: PrincipalPoint


class CalibrationFailed(BaseModel):
    status: str = Field(description="보정 실패 시 FAILED")
    failure_reason: str = Field(
        description="CHESSBOARD_NOT_FOUND, NOT_ENOUGH_FRAMES, VIDEO_OPEN_FAILED, "
                    "CALIBRATION_FAILED, SAVE_FAILED",
    )
    used_frames: int


def cleanup_old_uploads():
    cutoff = time.time() - RETENTION_DAYS * 86400
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path) or os.path.getmtime(path) >= cutoff:
            continue
        try:
            os.remove(path)
            logger.info("보관 기간 경과 파일 삭제: %s", name)
        except OSError as e:
            # 동시에 들어온 다른 요청이 먼저 지웠을 수 있으므로 넘어간다
            logger.warning("파일 삭제 실패: %s (%s)", name, e)


def save_upload(file: UploadFile, max_bytes: Optional[int] = None) -> Optional[str]:
    cleanup_old_uploads()
    # 같은 이름으로 다시 업로드해도 이전 결과를 덮어쓰지 않도록 업로드 시각을 붙인다
    saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file.filename)}"
    path = os.path.join(UPLOAD_DIR, saved_name)

    # 한도를 넘는 파일을 끝까지 쓰지 않도록 저장하면서 크기를 센다
    written = 0
    over_limit = False
    with open(path, "wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                over_limit = True
                break
            buffer.write(chunk)

    if over_limit:
        os.remove(path)
        return None
    return path


def analysis_available() -> bool:
    # 분석 라이브러리 설치 여부. 없는 환경에서는 영상을 처리하는 기능만 쓸 수 없다
    try:
        import cv2  # noqa: F401
        import analyze_shot  # noqa: F401
    except ImportError:
        return False
    return True


def analysis_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error_code": "ANALYSIS_UNAVAILABLE",
                 "message": "분석 기능을 사용할 수 없는 환경입니다"},
    )


def check_video_constraints(path: str, min_duration_sec: float = MIN_DURATION_SEC):
    # 조건을 만족하면 None, 아니면 (상태 코드, 오류 코드, 메시지)를 돌려준다
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 415, "UNREADABLE_VIDEO", "영상을 읽을 수 없습니다"
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        return 415, "UNREADABLE_VIDEO", "영상 정보를 읽을 수 없습니다"

    duration = frame_count / fps
    if duration < min_duration_sec:
        return 413, "VIDEO_TOO_SHORT", "영상이 너무 짧습니다"
    if duration > MAX_DURATION_SEC:
        return 413, "VIDEO_TOO_LONG", "업로드 가능한 최대 길이를 초과했습니다"
    if min(width, height) < MIN_SHORT_SIDE_PX:
        return 415, "RESOLUTION_TOO_LOW", "영상 해상도가 너무 낮습니다"
    return None

@app.post(
    "/analyses",
    status_code=202,
    tags=["분석"],
    summary="영상 업로드 및 분석 작업 생성",
    description=(
        "슈팅 영상을 업로드하고 분석을 시작한다. 분석은 요청과 분리되어 백그라운드에서 끝까지 진행되므로, "
        "응답으로 받은 id로 나중에 다시 붙어 진행률이나 결과를 받을 수 있다.\n\n"
        "업로드 조건: mp4 또는 mov, 200MB 이하, 길이 1~30초, 짧은 변 480px 이상."
    ),
    responses={
        202: {"model": CreatedResponse, "description": "분석 작업 생성됨"},
        400: {"model": ErrorResponse, "description": "MISSING_FILE"},
        413: {"model": ErrorResponse, "description": "FILE_TOO_LARGE, VIDEO_TOO_LONG, VIDEO_TOO_SHORT"},
        415: {"model": ErrorResponse, "description": "UNSUPPORTED_MEDIA_TYPE, RESOLUTION_TOO_LOW, UNREADABLE_VIDEO"},
    },
)
async def create_analysis(file: Optional[UploadFile] = File(None)):
    if file is None or not file.filename:
        logger.warning("업로드 거절: 파일 없음 → 400 MISSING_FILE")
        return JSONResponse(
            status_code=400,
            content={"error_code": "MISSING_FILE", "message": "영상 파일이 필요합니다"},
        )

    if os.path.splitext(file.filename)[1].lower() not in ALLOWED_EXTENSIONS:
        logger.warning("업로드 거절: %s → 415 UNSUPPORTED_MEDIA_TYPE", file.filename)
        return JSONResponse(
            status_code=415,
            content={"error_code": "UNSUPPORTED_MEDIA_TYPE", "message": "지원하지 않는 형식의 파일입니다"},
        )

    # 처리할 수 없는 요청이면 영상을 디스크에 쓰기 전에 돌려보낸다
    if not analysis_available():
        logger.warning("분석 요청 거절: 분석 라이브러리 미설치 → 503 ANALYSIS_UNAVAILABLE")
        return analysis_unavailable_response()

    logger.info("분석 요청 수신: %s", file.filename)
    file_path = save_upload(file, MAX_UPLOAD_BYTES)
    if file_path is None:
        logger.warning("업로드 거절: %s → 413 FILE_TOO_LARGE", file.filename)
        return JSONResponse(
            status_code=413,
            content={"error_code": "FILE_TOO_LARGE", "message": "업로드 가능한 최대 용량을 초과했습니다"},
        )

    problem = check_video_constraints(file_path)
    if problem is not None:
        os.remove(file_path)
        status_code, error_code, message = problem
        logger.warning("업로드 거절: %s → %d %s", file.filename, status_code, error_code)
        return JSONResponse(
            status_code=status_code,
            content={"error_code": error_code, "message": message},
        )

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_image_path = os.path.join(UPLOAD_DIR, f"{base_name}_result.jpg")

    analysis_id = str(uuid.uuid4())
    # video_path는 원본 영상 반환 엔드포인트에서 어느 파일인지 찾는 데 쓴다
    jobs[analysis_id] = {"status": "RUNNING", "progress": 0, "result": None,
                         "error": None, "video_path": file_path}

    short_id = analysis_id[:8]
    logger.info("[%s] 분석 작업 생성: %s", short_id, os.path.basename(file_path))

    def on_progress(progress):
        # 분석 스레드는 진행률을 기록만 하고, 전달은 구독 요청이 맡는다
        jobs[analysis_id]["progress"] = progress

    def run_analysis():
        from analyze_shot import process_video_with_progress

        started = time.time()
        try:
            result = process_video_with_progress(file_path, output_image_path, on_progress)
            jobs[analysis_id]["result"] = result
            jobs[analysis_id]["status"] = result.get("status", "VALID")
            jobs[analysis_id]["progress"] = 100
            logger.info("[%s] 분석 완료 status=%s %s %.1f초", short_id,
                        jobs[analysis_id]["status"],
                        result.get("failure_reason", ""), time.time() - started)
        except Exception as e:
            logger.exception("[%s] 분석 중 예외: %s", short_id, e)
            jobs[analysis_id]["status"] = "ERROR"
            jobs[analysis_id]["error"] = str(e)

    # 구독 여부와 무관하게 분석이 끝까지 진행되도록 요청과 분리해 실행한다
    asyncio.get_running_loop().run_in_executor(None, run_analysis)

    return JSONResponse(status_code=202, content={"id": analysis_id})


@app.get(
    "/analyses/{analysis_id}/events",
    response_class=StreamingResponse,
    tags=["분석"],
    summary="분석 진행률 구독 (SSE)",
    description=(
        "분석 진행률과 결과를 서버 전송 이벤트로 받는다. 각 줄은 data: 뒤에 JSON이 오는 형식이다.\n\n"
        "- progress : 진행률이 바뀔 때 전송. {\"type\": \"progress\", \"value\": 45}\n"
        "- keepalive : 진행률 변화가 없는 동안 연결 유지용. {\"type\": \"keepalive\"}\n"
        "- complete : 분석이 끝났을 때. {\"type\": \"complete\", \"data\": 분석 결과}\n"
        "- failed : 분석 자체를 수행하지 못한 서버 예외. {\"type\": \"failed\", \"message\": \"...\"}\n\n"
        "분석이 이미 끝난 뒤에 연결하면 지난 진행률 없이 complete만 즉시 받고 스트림이 닫힌다. "
        "신뢰도 미달이나 공 미탐지처럼 분석이 끝까지 실행된 실패는 failed가 아니라 complete로 오며, "
        "결과의 status가 ANALYSIS_FAILED가 된다."
    ),
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE 스트림"},
        404: {"model": ErrorResponse, "description": "JOB_NOT_FOUND"},
    },
)
async def analysis_events(analysis_id: str):
    job = jobs.get(analysis_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error_code": "JOB_NOT_FOUND", "message": "분석 작업을 찾을 수 없습니다"},
        )

    async def event_generator():
        sent_progress = None
        while True:
            status = job["status"]
            if status in TERMINAL_STATUSES:
                break
            if job["progress"] != sent_progress:
                sent_progress = job["progress"]
                yield f"data: {json.dumps({'type': 'progress', 'value': sent_progress})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            await asyncio.sleep(0.2)

        # 이미 끝난 작업에 뒤늦게 붙은 경우에도 지난 진행률 없이 결과만 바로 받는다
        if job["status"] == "ERROR":
            yield f"data: {json.dumps({'type': 'failed', 'message': job['error']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'complete', 'data': job['result']})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get(
    "/analyses/{analysis_id}/result",
    tags=["분석"],
    summary="분석 결과 조회",
    description="스트림을 열지 않고 결과만 가져온다. 아직 분석 중이면 202로 진행 상태를 돌려준다.",
    responses={
        200: {"model": AnalysisResult, "description": "분석 완료"},
        202: {"model": AnalysisProgress, "description": "아직 분석 중"},
        404: {"model": ErrorResponse, "description": "JOB_NOT_FOUND"},
        500: {"model": ErrorResponse, "description": "ANALYSIS_ERROR"},
    },
)
def analysis_result(analysis_id: str):
    job = jobs.get(analysis_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error_code": "JOB_NOT_FOUND", "message": "분석 작업을 찾을 수 없습니다"},
        )

    if job["status"] not in TERMINAL_STATUSES:
        return JSONResponse(
            status_code=202,
            content={"status": job["status"], "progress": job["progress"]},
        )

    if job["status"] == "ERROR":
        return JSONResponse(
            status_code=500,
            content={"error_code": "ANALYSIS_ERROR", "message": job["error"]},
        )

    return JSONResponse(status_code=200, content=job["result"])

@app.get(
    "/analyses/{analysis_id}/video",
    response_class=FileResponse,
    tags=["분석"],
    summary="원본 영상 조회",
    description=(
        "분석에 사용한 원본 영상을 돌려준다. 프론트엔드가 이 영상 위에 궤적을 그린다. "
        "부분 요청(Range)을 지원하므로 브라우저에서 탐색이 가능하다.\n\n"
        "보관 기간 7일이 지나 파일이 정리되면 id가 살아 있어도 VIDEO_NOT_FOUND가 된다."
    ),
    responses={
        200: {"content": {"video/mp4": {}, "video/quicktime": {}}, "description": "원본 영상"},
        404: {"model": ErrorResponse, "description": "JOB_NOT_FOUND 또는 VIDEO_NOT_FOUND"},
    },
)
def analysis_video(analysis_id: str):
    job = jobs.get(analysis_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error_code": "JOB_NOT_FOUND", "message": "분석 작업을 찾을 수 없습니다"},
        )

    path = job.get("video_path")
    if not path or not os.path.isfile(path):
        # 보관 기간이 지나 정리된 경우에도 여기로 온다
        return JSONResponse(
            status_code=404,
            content={"error_code": "VIDEO_NOT_FOUND", "message": "원본 영상을 찾을 수 없습니다"},
        )

    media_type = "video/quicktime" if path.lower().endswith(".mov") else "video/mp4"
    return FileResponse(path, media_type=media_type)


@app.get(
    "/results/{image_name}",
    response_class=FileResponse,
    tags=["분석"],
    summary="결과 오버레이 이미지 조회",
    description="분석 결과의 trajectory_image 값을 그대로 넣어 호출한다.",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "결과 이미지"},
        404: {"model": ErrorResponse, "description": "IMAGE_NOT_FOUND"},
    },
)
def get_image(image_name: str):
    path = os.path.join(UPLOAD_DIR, os.path.basename(image_name))
    if os.path.isfile(path):
        return FileResponse(path)
    return JSONResponse(
        status_code=404,
        content={"error_code": "IMAGE_NOT_FOUND", "message": "Image not found"},
    )

@app.post(
    "/calibrations",
    status_code=202,
    tags=["카메라 보정"],
    summary="보정 영상 업로드 및 보정 작업 생성",
    description=(
        "체스보드가 촬영된 영상을 올려 카메라 보정을 시작한다. 체스보드는 내부 코너 9x6, 한 칸 25mm를 전제한다.\n\n"
        "업로드 조건은 분석과 같으나 최소 길이만 5초다. 여러 위치와 각도의 프레임이 필요하기 때문이다."
    ),
    responses={
        202: {"model": CreatedResponse, "description": "보정 작업 생성됨"},
        400: {"model": ErrorResponse, "description": "MISSING_FILE"},
        413: {"model": ErrorResponse, "description": "FILE_TOO_LARGE, VIDEO_TOO_SHORT"},
        415: {"model": ErrorResponse, "description": "UNSUPPORTED_MEDIA_TYPE, RESOLUTION_TOO_LOW, UNREADABLE_VIDEO"},
    },
)
async def create_calibration(file: Optional[UploadFile] = File(None)):
    if file is None or not file.filename:
        logger.warning("보정 업로드 거절: 파일 없음 → 400 MISSING_FILE")
        return JSONResponse(
            status_code=400,
            content={"error_code": "MISSING_FILE", "message": "영상 파일이 필요합니다"},
        )

    if os.path.splitext(file.filename)[1].lower() not in ALLOWED_EXTENSIONS:
        logger.warning("보정 업로드 거절: %s → 415 UNSUPPORTED_MEDIA_TYPE", file.filename)
        return JSONResponse(
            status_code=415,
            content={"error_code": "UNSUPPORTED_MEDIA_TYPE", "message": "지원하지 않는 형식의 파일입니다"},
        )

    if not analysis_available():
        logger.warning("보정 요청 거절: 분석 라이브러리 미설치 → 503 ANALYSIS_UNAVAILABLE")
        return analysis_unavailable_response()

    logger.info("보정 요청 수신: %s", file.filename)
    file_path = save_upload(file, MAX_UPLOAD_BYTES)
    if file_path is None:
        logger.warning("보정 업로드 거절: %s → 413 FILE_TOO_LARGE", file.filename)
        return JSONResponse(
            status_code=413,
            content={"error_code": "FILE_TOO_LARGE", "message": "업로드 가능한 최대 용량을 초과했습니다"},
        )

    # 보정은 여러 각도의 프레임이 필요해 분석보다 긴 최소 길이를 요구한다
    problem = check_video_constraints(file_path, CALIBRATION_MIN_DURATION_SEC)
    if problem is not None:
        os.remove(file_path)
        status_code, error_code, message = problem
        logger.warning("보정 업로드 거절: %s → %d %s", file.filename, status_code, error_code)
        return JSONResponse(
            status_code=status_code,
            content={"error_code": error_code, "message": message},
        )

    calibration_id = str(uuid.uuid4())
    calibrations[calibration_id] = {"status": "CALIBRATING", "progress": 0,
                                    "result": None, "failure_reason": None}

    output_json_path = os.path.join(UPLOAD_DIR, f"{calibration_id}.json")

    short_id = calibration_id[:8]
    logger.info("[%s] 보정 작업 생성: %s", short_id, os.path.basename(file_path))

    def run_calibration():
        from calibrate import calibrate_camera
        started = time.time()
        try:
            outcome = calibrate_camera(file_path, output_json_path, CHESSBOARD_SIZE,
                                       CHESSBOARD_SQUARE_M, 50)
            if outcome.get("ok"):
                calibrations[calibration_id]["result"] = outcome
                calibrations[calibration_id]["status"] = "VALID"
                logger.info("[%s] 보정 완료 프레임=%d 재투영오차=%.3f %.1f초", short_id,
                            outcome.get("used_frames", 0), outcome.get("rms_error", 0.0),
                            time.time() - started)
            else:
                calibrations[calibration_id]["failure_reason"] = outcome.get("failure_reason")
                calibrations[calibration_id]["result"] = outcome
                calibrations[calibration_id]["status"] = "FAILED"
                logger.warning("[%s] 보정 실패 %s 프레임=%d", short_id,
                               outcome.get("failure_reason"), outcome.get("used_frames", 0))
        except Exception as e:
            logger.exception("[%s] 보정 중 예외: %s", short_id, e)
            calibrations[calibration_id]["failure_reason"] = "CALIBRATION_FAILED"
            calibrations[calibration_id]["status"] = "FAILED"
        finally:
            calibrations[calibration_id]["progress"] = 100
            # 체스보드 영상은 보정값을 얻고 나면 보관할 이유가 없다
            if os.path.isfile(file_path):
                os.remove(file_path)

    asyncio.get_running_loop().run_in_executor(None, run_calibration)

    return JSONResponse(status_code=202, content={"id": calibration_id})


@app.get(
    "/calibrations/{calibration_id}",
    tags=["카메라 보정"],
    summary="보정 상태 및 결과 조회",
    description=(
        "보정 작업의 상태를 조회한다. 분석과 달리 스트리밍을 쓰지 않으므로 1초 간격으로 호출한다. "
        "보정은 처리 단계가 나뉘지 않고 시간도 짧아 진행률에 담을 정보가 없기 때문이다.\n\n"
        "보정에 실패해도 200으로 응답하며 status가 FAILED가 된다. 재투영 오차의 품질 해석은 화면에서 정한다."
    ),
    responses={
        200: {"model": Union[CalibrationResult, CalibrationFailed],
              "description": "보정 완료(VALID) 또는 보정 실패(FAILED)"},
        202: {"model": CalibrationProgress, "description": "보정 진행 중"},
        404: {"model": ErrorResponse, "description": "CALIBRATION_NOT_FOUND"},
    },
)
def get_calibration(calibration_id: str):
    job = calibrations.get(calibration_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error_code": "CALIBRATION_NOT_FOUND", "message": "보정 작업을 찾을 수 없습니다"},
        )

    if job["status"] == "CALIBRATING":
        return JSONResponse(
            status_code=202,
            content={"status": "CALIBRATING", "progress": job["progress"]},
        )

    if job["status"] == "FAILED":
        return JSONResponse(
            status_code=200,
            content={"status": "FAILED",
                     "failure_reason": job["failure_reason"],
                     "used_frames": (job["result"] or {}).get("used_frames", 0)},
        )

    r = job["result"]
    mtx = r["camera_matrix"]
    return JSONResponse(status_code=200, content={
        "status": "VALID",
        "frame_size": {"width": r["img_width"], "height": r["img_height"]},
        "used_frames": r["used_frames"],
        "reprojection_error": round(float(r["rms_error"]), 3),
        "focal_length": {"fx": round(float(mtx[0][0]), 2), "fy": round(float(mtx[1][1]), 2)},
        "principal_point": {"cx": round(float(mtx[0][2]), 2), "cy": round(float(mtx[1][2]), 2)},
    })
