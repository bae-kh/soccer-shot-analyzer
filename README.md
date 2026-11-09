# Soccer Shooting Analysis Prototype

단일 카메라 축구 슈팅 영상에서 공을 추적하고 초기 속도를 추정하는 창의학기제 프로젝트입니다. 촬영과 골대 측정 조건이 충족되면 골라인 통과 추정과 참고 점수도 제공합니다. Python AI 엔진을 FastAPI·SSE와 React 화면에 연결했습니다.

**최신 상태: 2026-11-09 — 9주차 골라인 판정·외곽 채점 구현.** 영상만 올리면 추적과 초기 속도를 분석합니다. 촬영 정보가 없을 때 파일 FPS·수평 화각 60°·공 반지름 0.11m를 가정합니다. 영상에서 골대와 겹친 관측은 후보로 표시하며, 보정과 골대 실측이 없으면 골인과 종합 점수를 보류합니다. 실측 정확도 검증은 남아 있습니다.

## 1. 현재 사용 흐름

영상 선택 → 업로드 → SSE 진행률 → 공 추적 사진·속도·골라인 결과·종합 점수 또는 보류 사유

처음에는 촬영 설정·카메라 보정·골대 실측 입력을 요구하지 않습니다. `분석 설정 변경 (선택)`에서 조건을 바꾸거나 보정 자료 기반 모드로 전환할 수 있습니다.

골라인 판정과 기본 종합 점수를 사용하려면 보정 자료 기반 모드에서 실제 촬영 시간을 확인하고, 같은 설정의 카메라 보정과 해당 영상의 골대 실측 설정을 입력합니다.

[이전 데모 영상](https://github.com/user-attachments/assets/1491df93-ec3d-40b1-942b-b77e2e096dba)은 과거 구현 기록입니다.

## 2. 구현 상태

| 기능 | 현재 상태 |
|---|---|
| 영상 업로드·진행률·결과 화면 | 구현: FastAPI multipart 업로드, SSE, React |
| 공 검출·추적·재탐지 | 구현: YOLOv8, Kalman, ROI 확대 재탐지, CSRT 보조 추적 |
| 오검출·추적 이탈 검사 | 구현: 초기 공 모습·배경 변화 검사, 약한 원거리 재탐지 연속 확인; 일부 오류 가능 |
| 골대 검출 | 구현: YOLOv8-seg 기둥·가로대 영역 조합; 실측 네 모서리는 아님 |
| 카메라·3D·속도 | 구현: 왜곡·해상도 처리, 배경 이동 검사, 공 크기 기반 3D 후보·초기 탄도 모델 |
| 기본 자동 분석 | 구현: 정상 속도·줌 없음 가정의 속도 추정, 영상 골대 영역 진입 후보 |
| 촬영 조건·보정 모드 | 구현: 슬로모션 시간 환산, 설정별 보정 저장·선택·유효성 검사 |
| 골대 면 통과·채점 | 골대 실측 및 보정 기반 통과 추정, 기본 외곽 점수와 명시적 사용자 지정 규칙 지원 |
| 추적 산출물 | 구현: 원본 픽셀 관측 JSON, 추적 MP4, 관측·결과·골대 근거 JPG |
| 실제 정확도 평가 | 준비 도구 구현, 독립 속도·골라인 정답 자료를 사용한 평가 미완료 |
| 서비스 마무리 | 반복 업로드·네이티브 종료 대응·실행 환경 재현성 추가 검수 필요 |

1~5단계의 추적·계산·화면 연결과 전체 실행 검수, 후속 추적·카메라 개선, 골대 면·채점 기능, 자동 분석 모드를 구현했습니다. 기존 YOLO 가중치를 사용했으며 이번 개발에서 새 모델을 학습하지 않았습니다. **기능 구현 완료와 실제 정확도 검증 완료는 구분합니다.**

## 3. AI 계산과 결과 해석

### 공 추적과 속도

초기 공·골대 선택 → YOLO 검출 + Kalman 예상 위치·크기 비교 → ROI 확대 재검출 / 제한된 CSRT → 원본 모습·배경·공 크기 검사 → 초기 최대 0.45초의 유효한 YOLO 비행 관측 → 공 크기·카메라 모델 기반 3D 후보 → 탄도 모델 least_squares → 속도 추정 또는 사유를 남기고 보류

- 속도는 선택한 초기 비행 구간의 평균 추정값입니다. 최고 속도나 임팩트 순간 속도가 아닙니다.
- CSRT·Kalman 예측점은 속도 계산용 크기 관측에서 제외합니다. 관측과 모델 선을 구분하고 미래 외삽을 골인 근거로 쓰지 않습니다.
- 자동 모드에서도 카메라 이동·관측 부족·불안정한 공 크기·모델 불일치·큰 불확실성을 검사합니다. 근거가 부족하면 `speed: null`을 반환합니다.
- 기본 가정으로 계산한 값은 `speed_basis: ASSUMPTION_BASED`입니다. 3D는 카메라 좌표계 후보이며 경기장 좌표나 실제 공 높이를 자동 복원한 값이 아닙니다.
- 녹색은 YOLO, 주황색은 CSRT, 자홍색은 예측, 하늘색은 검사를 통과한 모델입니다.

### 영상 진입 후보와 골라인 통과

| 자동 결과 상태 | 의미 |
|---|---|
| `GOAL_LIKELY` | 측정한 골대 면 안으로 공 전체가 통과한 것으로 추정 |
| `OUTSIDE_LIKELY` | 측정한 골대 면의 기둥 옆이나 크로스바 위로 공 전체가 통과한 것으로 추정 |
| `UNDETERMINED` | 골대 실측·보정·연속 관측 등의 근거 부족 |

`goal_entry`는 영상 영역 진입 근거, `physical_goal`은 실측 골대 면의 조건부 통과 추정입니다. `goal_prediction`은 자동 모드용 호환 필드이며, 화면 후보만 있으면 `UNDETERMINED`입니다. 물리 판정은 연속 YOLO 관측으로 공 전체가 앞에서 뒤로 넘어가는지 검사합니다. 관측 누락, 카메라 이동, 골대 경계와의 불확실한 접촉은 보류합니다.

### 기본 참고 점수

```text
속도 항목 = clamp((속도 km/h - 20) / 80, 0, 1)
위치 항목 = 정규화한 골대 면 교차 위치와 골대 중심의 거리 / sqrt(0.5)
참고 점수 = 100 × (속도 항목 × 0.4 + 위치 항목 × 0.6)
```

기본 규칙 `week9_outer_v1`은 같은 속도의 유효한 골인에서 중앙보다 외곽에 높은 위치 점수를 줍니다. 골인 점수에는 골대 면 교차 위치와 촬영 시간·보정을 확인한 속도가 모두 필요합니다. 충분한 골대 밖 통과 근거는 0점, 판단 불가나 입력 부족은 `score: null`입니다. 명시적으로 확인한 사용자 지정 `scoring_rules`는 기존 목표점 기준으로 계산합니다. 계산된 점수는 `PROVISIONAL`이며 실측 정확도 미검증입니다.

## 4. 검증 결과

9주차에는 현재 원본 20개를 모두 재실행했습니다. 기존 추적과 속도 결과는 모두 일치했고, 영상 진입 후보 9개를 유지했습니다. 골대 실측 입력이 없어 종합 점수는 모두 미평가입니다. 이전 원본 중 없는 12개는 저장 관측으로 확인했습니다. 자세한 비교는 [9주차 구현·검증 기록](./docs/week9_goal_scoring_results.md)을 참고하세요. 아래 표는 변경 전 `automatic_mode_20261012`의 **비교용 기록**입니다.

| 항목 | 결과 |
|---|---|
| 최종 결과 생성 | 32/32, 최초 네이티브 종료 1건 재시도 포함 |
| 속도 | 추정 15 / 보류 17 |
| 속도 보류 사유 | 공 크기 불안정 4 / 비행 관측 부족 5 / 큰 불확실성 2 / 모델 불일치 3 / 카메라 이동 3 |
| 골인 가능성 | 골인 추정 13 / 판단 어려움 19 / 골대 밖 추정 0 |
| 참고 점수 | 17개: 속도+위치 11 / 속도만 4 / 위치만 2; 미평가 15 |
| 직전 구현 대비 추적 | 32개 모두 기존 관측·궤적 유지, 변경 프레임 0 |
| 산출물 검사 | 소스·입력·모델 해시, JSON·이미지·영상 계약 검사 통과 |
| 자동 검사 | Python 141개 / 프론트엔드 Node 13개 / ESLint / 프로덕션 빌드 통과 |
| 평균 실행 시간 | 17.477초; 프로세스 시작·모델 로딩·재시도 포함, 업로드·UI 제외 |
| 실제 정확도 | 독립 정답 자료가 없어 미평가 |

실촬영 6번은 골대 검출 도중 최초 프로세스가 네이티브 종료 코드 `3221226505`로 종료됐습니다. 한 번 재시도해 결과를 만들었으며 초기 로그·종료 코드·재시도 이력을 보존했습니다. 원인과 반복 발생 여부는 아직 확정하지 않았습니다.

위 숫자는 **실행·반환 상태 수이며 정답 정확도나 골인 성공률이 아닙니다.**

상세 결과: [자동 분석 모드](./docs/ai_automatic_mode.md), [추적·카메라 후속 검수](./docs/ai_tracking_camera_followup.md), [골대 면·채점·실측 검증 준비](./docs/ai_goal_speed_followup.md).

9주차 기본 규칙에서 실측 골대 면이 없는 영상은 골라인 판정과 점수를 보류합니다. 자동 속도는 기존대로 품질 조건을 통과하면 표시합니다.

## 5. 팀원용 API와 프론트엔드 연결

| API | 용도 |
|---|---|
| `POST /analyze` | multipart `file`, 선택 JSON 문자열 `capture_options`; SSE 응답 |
| `GET /results/{image_name}` | 결과 이미지 조회 |
| `POST /calibrate` | 체커보드 영상 `file`, 설정 이름 `profile_label`; 보정 결과 SSE |
| `GET /camera-profiles` | 유효한 카메라 보정 프로필 목록 |

`capture_options`가 없거나 `{}`이면 자동 모드입니다. 명시적인 기본값은 `{"analysis_mode":"automatic","timing_mode":"normal"}`입니다. **기존 촬영 옵션을 보내면서 `analysis_mode`를 생략하면 호환성을 위해 보정 모드로 처리**하므로 새 프론트엔드에서는 모드를 명시하는 편이 좋습니다.

```text
data: {"type":"progress","value":30}

data: {"type":"keepalive"}

data: {"type":"result","data":{...분석 결과...}}

data: {"type":"error","message":"...오류 설명..."}
```

잘못된 촬영 옵션은 SSE 시작 전에 HTTP 422로 거절합니다. `frontend/src/lib/sse.js`가 청크 분할·UTF-8·여러 이벤트 처리를 담당합니다.

| 결과 필드 | 프론트엔드 사용 |
|---|---|
| `analysis_mode` | `automatic` / `calibrated`; 모드에 맞는 결과 표현 |
| `speed`, `measurement_status`, `failure_reason` | km/h 숫자 또는 null; `ESTIMATED` / `UNAVAILABLE` 및 보류 사유 |
| `speed_basis`, `measurement_context` | 가정 기반 여부·입력 조건·경고 |
| `goal_entry` | 영상 영역 진입 후보와 근거 프레임 |
| `goal_prediction` | 자동 모드의 골대 면 추정 호환 필드 |
| `physical_goal` | 결과 화면의 골라인 통과 추정 또는 보류 사유 |
| `score`, `score_status`, `score_evaluation` | 숫자 또는 null; 참고/미평가, 계산 근거·규칙 ID·보류 사유 |
| `comment` | 결과 설명 |
| `trajectory_image`, `goal_entry_image`, `observation_image` | 서버 반환 파일명; `/results/{파일명}`으로 조회 |
| `trajectory`, `tracking_summary`, `video_metadata`, `goal_area` | 원본 픽셀 관측·추적 요약·영상 정보·골대 영역 |
| `tracking_video`, `tracking_data` | 저장한 MP4·JSON 파일명; 전용 플레이어·다운로드 UI 미구현 |

테스트 11번 결과 객체의 일부입니다. 사진 파일명은 요청마다 달라지므로 반환값을 그대로 사용합니다.

```json
{
  "analysis_mode": "automatic",
  "speed": 81.8,
  "measurement_status": "ESTIMATED",
  "speed_basis": "ASSUMPTION_BASED",
  "score": null,
  "score_status": "NOT_EVALUATED",
  "goal_entry": {"status": "CANDIDATE_OBSERVED"},
  "physical_goal": {
    "status": "UNDETERMINED",
    "reason": "GOAL_PLANE_REQUIRED"
  },
  "score_evaluation": {
    "rule_set_id": "week9_outer_v1",
    "reason": "GOAL_PLANE_REQUIRED"
  }
}
```

프론트엔드 담당자는 `Result.jsx`·`Upload.jsx`·스타일을 중심으로 화면을 꾸밀 수 있습니다. 기존 `lib/sse.js`, `lib/goalResult.js`의 파싱·결과 표현 함수를 재사용하고 **null을 0으로 바꾸지 않으며 실제 0점은 표시**합니다. `UNDETERMINED`는 실축으로 표시하지 않습니다. 필드 이름·상태값을 바꿀 때는 AI/Backend 담당자와 함께 소비 코드·테스트를 갱신합니다.

현재 API 주소는 `http://localhost:8000`으로 고정돼 있습니다. 다른 PC·배포 주소 연결은 후속 환경 설정 작업입니다. SSE 분석은 서버 스레드에서 실행되며 영속 작업 큐·재시작 복구는 아직 없습니다.

## 6. 코드 구조

```text
backend/
  main.py                  # API·업로드·SSE
  analyze_shot.py          # AI 전체 분석 연결
  ball_tracking.py         # 초기 공 선택·Kalman·ROI·후보 연결
  ball_observation.py      # 원본 모습·배경·추적 이탈 검사
  ball_size_quality.py     # 속도 입력용 공 크기 품질
  goal_geometry.py         # 기둥·가로대 검출 영역 조합
  goal_entry.py            # 영상 영역 진입 근거
  goal_measurement.py      # 실측 골대 면 통과·기본 및 사용자 지정 채점
  automatic_analysis.py    # 자동 모드용 골라인 결과 표현
  camera_geometry.py      # 카메라 모델·왜곡·배경 이동
  capture_context.py      # 모드·시간·보정·가정 출처
  shot_physics.py          # 초기 3D 후보·탄도 fitting·속도
  trajectory_render.py     # 관측·모델·골대 근거 이미지
  calibrate.py             # 체커보드 내부 보정
  requirements*.txt        # 서비스·AI 의존성
frontend/src/
  App.jsx
  components/              # Upload·CaptureSettings·Result·CalibrationModal
  lib/                     # SSE 파서·결과 표현·입력 검사
tests/                     # Python 계산·추적·API 계약 검사
frontend/tests/            # Node SSE·결과·입력 계약 검사
tools/                     # 일괄 분석·합성 검증·회귀 비교·정답 평가
docs/                      # 단계별 구현·검수·촬영 조건 기록
assets/                    # Git 공유용 대표 화면·검증 사진
soccer_videos/             # 로컬 예시 영상; Git 제외
output/ai_baseline/        # 로컬 전체 실행 산출물; Git 제외
```

## 7. 실행 방법

### Backend — 프로젝트 루트에서 실행

```bash
python -m pip install -r backend/requirements.txt
python -c "import cv2; print('CSRT:', hasattr(cv2, 'TrackerCSRT_create'))"
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

검증 환경은 Python 3.9.25, PyTorch 2.5.1+cu121, NVIDIA RTX 4070 Ti입니다. GPU용 PyTorch 설치는 각 PC 환경에 맞춰 준비합니다. `requirements-ai.txt`는 현재 동작 환경 버전 기록이며 모든 PC에서 설치·성능을 검증한 것은 아닙니다. CSRT 확인 결과는 `True`여야 합니다.

### Frontend — 별도 터미널

```bash
cd frontend
npm ci
npm run dev
```

Vite가 표시한 주소(기본 `http://localhost:5173`)를 엽니다. 기존 페이지를 열어 둔 경우 새로고침해 자동 모드 기본값으로 시작합니다.

### 팀원이 별도로 받아야 하는 파일

- `backend/yolov8s.pt`: 공 검출 모델.
- `backend/goal_segment_best.pt`: 골대 분할 모델.
- 테스트할 슈팅 영상: 로컬 `soccer_videos` 등에 보관.
- 보정 모드를 쓸 때만 해당 촬영 설정의 `backend/camera_profiles/*.json`.

**모델 가중치와 원본 영상은 Git에 포함하지 않으므로 clone만으로 실제 AI 분석 준비가 끝나지는 않습니다.** 팀에서 합의한 별도 공유 경로로 두 모델과 필요한 영상을 전달해야 합니다. 자동 모드에서 체커보드·골대 실측은 필수가 아닙니다.

보정 모드는 정상 원본 또는 일정 슬로모션 배율/촬영 FPS와 해당 카메라·렌즈·줌·크롭 조건의 보정이 필요합니다. 체커보드는 내부 교차점 9×6개(칸 10×7), 칸 크기 25mm를 사용합니다. 입력은 [촬영 조건 연결](./docs/ai_capture_context_results.md), [골대 면·채점](./docs/ai_goal_speed_followup.md)을 참고하세요.

예시 촬영 기록은 `docs/video_capture_conditions.json`에 보존합니다. 테스트 1~4의 슬로모션, 테스트 12~14의 설정 차이 의심, 실촬영 10~13의 줌아웃 기록을 정상 촬영으로 바꾸지 않았습니다. 자동 모드 기본 가정과 기록이 다르면 경고를 남깁니다. 실제 배율·보정 없이 해당 영상의 자동 결과를 정답 속도로 사용하지 않습니다.

### 검사·일괄 분석

```bash
# 프로젝트 루트
python -m unittest discover -s tests
python -B tools/run_ai_baseline.py --output-dir output/ai_baseline/new_run

# frontend 폴더
node --test tests/*.test.js
npm run lint
npm run build
```

출력 폴더는 기존 결과와 겹치지 않는 새 이름을 사용합니다. [일괄 실행 안내](./docs/ai_baseline.md)에 산출물 구조가 있습니다. 전체 `output`은 로컬 기록이며 문서의 `../output/...` 링크는 산출물을 보유한 PC에서 확인하는 자료입니다.

## 8. 팀 협업 방식

`main`을 팀 공통 실행 기준으로 두고 **작업별 브랜치 → commit/push → Pull Request(PR) → 검토·실행 확인 → main 병합** 순서로 공유합니다. GitHub의 [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)를 기준으로 합니다.

| 역할 | 주요 작업 위치 | 작업 브랜치 예시 |
|---|---|---|
| AI | 추적·물리 계산·골인 추정·검증 | `codex/ai-auto-analysis` |
| Backend | API·실행 관리·파일 처리 | `feature/backend-jobs` |
| Frontend | 업로드·진행·결과 화면·스타일 | `feature/frontend-ui` |

브랜치는 작업 단위로 만듭니다. 이번 AI 버전을 main에 병합한 뒤 프론트엔드 담당자는 최신 main에서 자기 브랜치를 만들면 같은 AI·API를 기반으로 화면을 개발할 수 있습니다. API나 공용 파일이 겹칠 때는 먼저 변경 내용을 공유합니다. 같은 저장소에 collaborator 권한으로 참여하면 이 방식으로 진행할 수 있습니다.

커밋에는 필요한 AI·API·화면 코드, 실행 의존성, 테스트와 기술 문서를 포함합니다. `git diff --cached --stat`과 `git diff --cached --check`로 확인한 뒤 푸시합니다. 원본 영상, 가중치, 보정 프로필, 실험 산출물·캡처·보고서·캐시는 로컬에 보관합니다.

AI PR 병합 뒤, **작업 트리가 깨끗한 프론트엔드 담당자 PC**에서는 다음 순서로 시작합니다.

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/frontend-ui
```

프론트엔드 PR에는 변경 화면·API 필드·lint/build·결과 표시 확인을, AI PR에는 변경 계산·상태값·대표 실행·회귀 검사 결과를 기록합니다.

## 9. 검증 이력과 남은 개발

| 기록 | 문서 |
|---|---|
| 1단계 기준 실행 | [기준 결과](./docs/ai_baseline_stage1_results.md) |
| 2단계 추적·재탐지 | [구현·검증](./docs/ai_stage2_results.md), [구현 아이디어](./docs/ai_tracking_explained.md) |
| 3단계 3D·속도·촬영 조건 | [구현](./docs/ai_stage3_results.md), [검수](./docs/ai_stage3_review.md), [합성 검증](./docs/ai_capture_pipeline_validation.md) |
| 공 관측·속도 입력 개선 | [오검출](./docs/ai_identity_filter_results.md), [공 크기·비행 구간](./docs/ai_flight_quality_results.md) |
| 4~5단계 영역 진입·전체 검수 | [4단계](./docs/ai_stage4_results.md), [5단계](./docs/ai_stage5_results.md) |
| 후속 추적·카메라 개선 | [최신 검수](./docs/ai_tracking_camera_followup.md) |
| 실측 평가 준비·골대 면·채점 | [구현·한계](./docs/ai_goal_speed_followup.md) |
| 초기 자동 분석·32개 실행 | [이전 결과](./docs/ai_automatic_mode.md) |
| 9주차 골라인 판정·외곽 점수 | [설계](./docs/week9_goal_scoring_design.md), [구현·검증](./docs/week9_goal_scoring_results.md) |

단계별 문서는 당시 구현 기록입니다. 원본 영상과 결과 캡처, 일부 과거 실험 스크립트는 보유 PC의 기록입니다. 합성 속도 오차 약 0.20~1.66%·보정 초점거리 오차 최대 약 0.353%는 **합성 사례의 값이며 실제 영상 오차율이 아닙니다.**

후속 검수에서 테스트 12 f37~39의 약한 신발·가림 재탐지를 보완했습니다. 실촬영 5 f101은 원본 연속 프레임을 재검토해 실제 공 관측으로 보존했습니다. 과거의 “배경 추적 의심”을 현재 확정 오류로 취급하지 않습니다.

1. **보류·오류 영상 개선:** 속도 보류 영상의 공 관측과 카메라 조건을 검토하고, 골대 실측이 있는 현장 영상에서 통과 판정의 반환 품질을 확인.
2. **실제 정확도 검증:** 독립 속도·골라인 정답과 촬영 조건 확보, 공 중심·윤곽 라벨 및 오차 집계.
3. **채점 기준 검수:** 속도 40%·외곽 위치 60%와 속도 범위를 팀에서 확인하고 현장 조건에서 검증.
4. **실행·시연 마무리:** 네이티브 종료 원인·재발, 반복 업로드·오류 처리·환경 재현성과 최종 시연 검수.

단일 카메라 깊이 모호성, 체계적 공 박스 크기 오차, 가림·CSRT 이탈, 제한된 배경 이동 검사, 잘못 입력한 조건은 남은 한계입니다. 가변 프레임 간격·구간별 배속 편집·자동 렌즈/크롭 전환, 영속 작업 큐·취소·운영 배포는 현재 지원 또는 구현 범위에 포함하지 않습니다.

Agentic AI는 구현·디버깅·문서화 보조에 사용했습니다. 실제 실행과 합성·회귀·계약 검사로 확인한 범위만 보고하며 정답이 없는 결과를 정밀 계측이나 상용 수준 정확도로 주장하지 않습니다.
