# ⚽ AI Soccer Shot Analyzer (축구 슈팅 분석기)

> **"당신의 슈팅 속도는 몇 km/h 입니까?"**  
> 단일 카메라 영상만으로 축구 공의 3D 궤적을 추적하고, 슈팅 속도와 골 성공 여부를 분석하는 AI 웹 서비스입니다.

![Demo](https://velog.velcdn.com/images/example/post/1234/demo.gif)
*(여기에 실제 시연 GIF나 스크린샷을 추가하면 100배 더 매력적입니다!)*

## 🚀 Key Features (주요 기능)

1.  **AI Based 3D Tracking**: 
    - 단일 카메라(2D) 영상에서 공의 크기 변화를 분석하여 3D 좌표(x, y, z)를 추정합니다.
    - **Kalman Filter**를 적용하여 공이 잠시 가려지거나 인식이 불안정해도 부드러운 궤적을 그려냅니다.

2.  **Professional Speed Measurement**:
    - 단순 프레임 간 이동 거리가 아닌, **전체 궤적의 선형 회귀(Linear Regression)** 분석을 통해 정밀한 슈팅 속도(km/h)를 산출합니다.

3.  **Smart Goal Detection**:
    - **Custom Trained YOLOv8-seg**: 직접 수집한 데이터로 학습된 AI가 골대(그물, 기둥)를 정밀하게 인식합니다.
    - **Corner Score**: 골대 구석으로 찰수록 높은 점수를 부여하는 정교한 채점 시스템을 갖췄습니다.

4.  **Instant Feedback**:
    - "프로급 파워!", "손흥민 선수가 울고 갈 완벽한 궤적!" 등 상황에 맞는 재미있는 AI 코멘트를 제공합니다.

## 🛠 Tech Stack (기술 스택)

| Category | Technologies |
|----------|--------------|
| **Frontend** | React, Vite, Tailwind CSS |
| **Backend** | Python (FastAPI) |
| **AI / CV** | YOLOv8, OpenCV, NumPy |
| **Algorithms** | Kalman Filter, Linear Regression |

## 📦 Installation & Setup (설치 및 실행)

이 프로젝트는 Python(Backend)과 Node.js(Frontend) 환경이 필요합니다.

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/soccer-shot-analyzer.git
cd soccer-shot-analyzer
```

### 2. Backend Setup
```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r backend/requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

## ▶️ Usage (사용법)

프로젝트 루트에서 제공되는 실행 스크립트를 사용하면 한 번에 실행할 수 있습니다.

```bash
# 전체 실행 (Backend + Frontend)
python run_app.py
```

또는 개별 터미널에서 실행:
1.  **Backend**: `uvicorn main:app --reload` (in `/backend`)
2.  **Frontend**: `npm run dev` (in `/frontend`)

웹 브라우저에서 `http://localhost:5173`으로 접속하여 영상을 업로드하세요!

## 🧩 Trouble Shooting & Engineering

개발 과정에서 겪은 기술적 난제와 해결 방법들을 정리했습니다.

-   **Deep Dive into Speed Accuracy**: 
    -   *Problem*: 2D 영상에서 공의 크기 변화(Depth)가 노이즈로 인해 심하게 튐.
    -   *Solution*: 최근 5프레임 반경 평균(Radius Smoothing)과 칼만 필터(Kalman Filter) 융합으로 해결.

-   **Goal Detection in the Wild**:
    -   *Problem*: 야외 환경이나 각도에 따라 골대 인식이 실패함.
    -   *Solution*: 영상 초반 프레임의 마스크를 누적(Accumulate)하고 모폴로지 연산(Morphology)으로 복원하는 알고리즘 구현.

## 📝 License

This project is licensed under the MIT License.
