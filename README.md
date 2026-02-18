# ⚽ AI Soccer Shot Analyzer (축구 슈팅 분석기)

**"당신의 슈팅 속도는 몇 km/h 입니까?"** 단일 카메라(Monocular Camera) 영상만으로 축구공의 3D 궤적을 추적하고, 슈팅 속도와 골 성공 여부를 정밀하게 분석하는 AI 웹 서비스입니다.

## 🎥 Demo & UI



https://github.com/user-attachments/assets/93b3ed67-0f2a-4ccc-af72-0925b961d455


## 🚀 Key Features

### 1. Monocular 3D Trajectory Estimation & Tracking
- **2D to 3D Inference:** 단일 카메라 영상에서 바운딩 박스의 크기 변화(Scale)를 분석하여 깊이(Depth)를 포함한 3D 좌표(x, y, z)를 추정합니다.
- **Kalman Filter Integration:** 공이 플레이어에 의해 가려지거나(Occlusion) 객체 인식이 불안정한 구간에서도 **Kalman Filter**를 적용하여 물리적으로 자연스럽고 부드러운 궤적을 복원합니다.

### 2. Robust Speed Measurement
- **Linear Regression Analysis:** 단순 프레임 간 픽셀 이동 거리가 아닌, 전체 3D 궤적 데이터에 대한 **선형 회귀(Linear Regression)** 분석을 수행하여 아웃라이어(노이즈)에 강건한 정밀 슈팅 속도(km/h)를 산출합니다.

### 3. Smart Goal Detection & Scoring
- **Custom YOLOv8-seg:** 직접 수집하고 라벨링한 데이터셋으로 파인튜닝된 Instance Segmentation 모델이 골대(그물, 기둥)의 형태를 픽셀 단위로 정밀하게 분할(Segmentation)합니다.
- **Corner Score System:** 슈팅 궤적이 골대의 구석(사각지대)으로 향할수록 가중치를 부여하는 정교한 공간 채점 알고리즘을 구현했습니다.

## 🛠 Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend** | React, Vite, Tailwind CSS |
| **Backend** | Python, FastAPI |
| **AI / CV** | YOLOv8 (Detection & Segmentation), OpenCV, NumPy |
| **Algorithms** | Kalman Filter, Linear Regression, Morphological Operations |

## 🧩 Trouble Shooting & Engineering

### 1. Deep Dive into Speed Accuracy (Depth Noise Issue)
- **Problem:** 2D 영상에서 객체 크기 기반으로 Depth를 추정할 때, 바운딩 박스의 미세한 떨림이 Z축 좌표의 심각한 노이즈로 증폭되는 현상 발생.
- **Solution:** 최근 5프레임 반경의 크기 평균을 구하는 **Radius Smoothing** 기법과 **Kalman Filter**를 융합 파이프라인으로 구축하여 Z축 좌표의 튀는 현상을 성공적으로 제어하고 속도 측정의 신뢰도를 확보했습니다.

### 2. Goal Detection in the Wild (Robustness Issue)
- **Problem:** 야외 조명 변화, 카메라 각도, 그물의 색상 등에 따라 단일 프레임에서의 골대 Segmentation이 실패하거나 끊어지는 문제 발생.
- **Solution:** 영상 초반 프레임들의 Segmentation Mask를 **누적(Accumulate)**하고, OpenCV의 **모폴로지 연산(Morphological Operations)**을 적용해 끊어진 객체를 이어붙임으로써 환경 변화에 강건한 골대 영역 복원 알고리즘을 구현했습니다.

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
