"""Modal 배포 설정.

무료 호스팅 중 분석까지 수행할 수 있는 환경을 찾아 선택한 곳이다.
Render 무료 플랜은 메모리 한도가 512MB인데 분석이 900MB 이상을 써서
업로드 요청마다 서버가 재시작되었다.

올리기:   modal deploy modal_app.py
시험 실행: modal serve modal_app.py   (코드를 고치면 바로 반영된다)
"""
import modal

app = modal.App("soccer-shot-api")

image = (
    # scipy가 3.13용 설치 파일을 제공하지 않아 3.12로 고정한다
    modal.Image.debian_slim(python_version="3.12")
    # opencv가 필요로 하는 시스템 라이브러리
    .apt_install("libgl1", "libglib2.0-0")
    # requirements.txt는 다른 파일을 참조하는 형태라 Modal이 함께 올리지 못한다.
    # 내용이 같으므로 두 파일을 각각 지정한다.
    .pip_install_from_requirements("backend/requirements-ai.txt")
    .pip_install_from_requirements("backend/requirements-api.txt")
    # 공 추적에 쓰는 CSRT는 opencv 추가 기능 묶음에만 들어 있다
    .pip_install("opencv-contrib-python==4.13.0.92")
    # 분석 코드가 저장소 최상단 기준으로 경로를 잡으므로 구조를 그대로 올린다
    .add_local_dir("backend", remote_path="/root/backend")
    .add_local_file("docs/video_capture_conditions.json",
                    remote_path="/root/docs/video_capture_conditions.json")
)


@app.function(
    image=image,
    # 분석 작업 상태를 서버 메모리에 두므로 여러 대로 흩어지면 조회가 되지 않는다
    max_containers=1,
    # 분석이 20초가량 걸리고 그동안 진행률을 계속 조회하므로 넉넉히 깨워 둔다
    scaledown_window=300,
    timeout=600,
    memory=2048,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    sys.path.insert(0, "/root/backend")
    os.chdir("/root/backend")
    os.makedirs("/root/backend/uploads", exist_ok=True)
    os.makedirs("/root/backend/camera_profiles", exist_ok=True)
    # 연동용 예시 작업을 메모리에 적재한다
    os.environ["SAMPLE_DATA"] = "1"

    from main import app
    return app
