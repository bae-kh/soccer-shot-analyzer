import subprocess
import time
import os
import sys

def run_app():
    print("🚀 축구 슈팅 분석기 MVP를 시작합니다...")
    
    # capstone 환경의 파이썬 강제 지정
    capstone_python = r"C:\anaconda3\envs\capstone\python.exe"
    python_exe = capstone_python if os.path.exists(capstone_python) else sys.executable
    
    # Backend 실행
    print("Backend 서버 시작 중 (Port 8000)...")
    backend_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--reload"],
        cwd=os.path.join(os.getcwd(), "backend"),
        shell=True
    )
    
    time.sleep(2) # 백엔드 시작 대기

    # Frontend 실행
    print("Frontend 서버 시작 중 (Port 5173)...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(os.getcwd(), "frontend"),
        shell=True
    )

    print("\n✅ 서버가 실행되었습니다!")
    print("👉 http://localhost:5173 에 접속하여 테스트하세요.\n")
    print("종료하려면 Ctrl+C를 누르세요.")

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 서버를 종료합니다...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    run_app()
