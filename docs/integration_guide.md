# 백엔드 연동 가이드

프론트엔드 연동에 필요한 내용만 정리한 문서다. 설계 배경과 아직 정하지 못한 항목은 api_spec.md에 있다.

Base URL (개발 환경): http://localhost:8000

CORS는 모든 출처를 허용하도록 되어 있어 별도 설정 없이 호출할 수 있다.

---

## 1. 서버 실행

backend 폴더에서 아래를 실행한다.

pip install -r requirements.txt
uvicorn main:app

서버가 뜨면 아래에서 요청과 응답 형태를 직접 확인하고 시험 호출도 할 수 있다.

- 스웨거 화면: http://localhost:8000/docs
- 원본 문서(타입 생성용): http://localhost:8000/openapi.json

**분석 모델 파일이 따로 필요하다.** backend 폴더에 yolov8s.pt와 골대 탐지 모델이 있어야 실제 분석이 돌아간다. 저장소에 포함되어 있지 않으므로 AI 파트에서 받아야 한다. 모델이 없으면 업로드와 검증까지는 동작하지만 분석 단계에서 실패한다.

---

## 1-1. 예시 작업 (영상 없이 결과 받아보기)

영상을 올리고 분석이 끝날 때까지 기다리지 않아도 결과 화면을 만들어볼 수 있도록, 미리 준비된 작업을 넣어둘 수 있다. 아래처럼 켜고 서버를 실행한다.

SAMPLE_DATA=1 uvicorn main:app

고정된 id로 바로 조회된다. 결과 조회와 진행률 구독 둘 다 된다. 이미 끝난 작업이라 구독하면 지난 진행률 없이 complete만 즉시 온다.

| id                        | 조회 경로                               | 내용                                 |
| ------------------------- | --------------------------------------- | ------------------------------------ |
| sample-valid              | /analyses/sample-valid/result           | 정상 분석. 궤적 25점, 가로 영상      |
| sample-portrait           | /analyses/sample-portrait/result        | 정상 분석. 세로 영상(1080x1920)      |
| sample-no-ball            | /analyses/sample-no-ball/result         | 실패. BALL_NOT_DETECTED, 궤적 없음   |
| sample-low-confidence     | /analyses/sample-low-confidence/result  | 실패. LOW_CONFIDENCE, 궤적 일부 포함 |
| sample-track-short        | /analyses/sample-track-short/result     | 실패. TRACK_TOO_SHORT                |
| sample-calibration-ok     | /calibrations/sample-calibration-ok     | 보정 성공                            |
| sample-calibration-failed | /calibrations/sample-calibration-failed | 보정 실패. CHESSBOARD_NOT_FOUND      |

진행률 구독은 아래처럼 호출한다.

/analyses/sample-valid/events

**예시 작업에는 실제 영상 파일이 없다.** 원본 영상 조회와 결과 이미지 조회는 404가 난다. 그 두 가지는 영상을 직접 올려야 확인할 수 있다.

환경변수를 주지 않으면 예시는 들어가지 않는다. 개발 편의용이므로 연동이 끝나면 제거할 예정이다.

---

## 2. 분석 흐름

영상 업로드와 진행률 구독이 분리되어 있다. 업로드로 id를 받고, 그 id로 구독한다. 연결이 끊겨도 id만 있으면 다시 붙을 수 있다.

1. POST /analyses 로 영상을 올리고 id를 받는다
2. GET /analyses/{id}/events 로 진행률을 구독한다
3. complete 이벤트로 결과를 받는다
4. 필요하면 GET /analyses/{id}/video 로 원본 영상을, GET /results/{파일명} 으로 결과 이미지를 가져온다

구독을 안 하거나 중간에 끊어도 **분석은 서버에서 끝까지 진행된다.** 나중에 다시 구독하면 지난 진행률 없이 결과만 즉시 받는다.

### 2.1 POST /analyses

multipart/form-data로 file 필드에 영상을 담아 보낸다.

성공하면 202와 함께 아래를 돌려준다.

{ "id": "3f2a9c10-5d7e-4b61-9a2f-7c1e8d4b0a55" }

업로드 조건에 맞지 않으면 202 대신 오류를 돌려준다. 아래 5장 참고.

### 2.2 GET /analyses//events

서버 전송 이벤트로 진행률과 결과를 받는다. 각 줄은 data: 뒤에 JSON이 오는 형식이다.

| type      | 언제                        | 내용                                      |
| --------- | --------------------------- | ----------------------------------------- |
| progress  | 진행률이 바뀔 때            | { "type": "progress", "value": 45 }       |
| keepalive | 진행률 변화가 없는 동안     | { "type": "keepalive" }                   |
| complete  | 분석이 끝났을 때            | { "type": "complete", "data": 결과 객체 } |
| failed    | 분석 자체를 수행 못 한 경우 | { "type": "failed", "message": "..." }    |

**complete와 failed의 구분 기준**이 중요하다. 분석 파이프라인이 끝까지 실행되면 공을 못 찾았든 신뢰도가 낮든 **complete로 온다.** 결과 객체의 status가 ANALYSIS_FAILED가 되고 failure_reason에 원인이 담긴다. failed는 모델 로드 실패처럼 분석을 시작조차 못 한 경우에만 온다.

keepalive는 진행률이 안 바뀌는 동안 약 0.2초 간격으로 계속 나간다. 이름 있는 데이터 이벤트라 브라우저가 그대로 전달한다.

complete나 failed를 받으면 서버가 스트림을 닫는다. 받은 쪽에서도 연결을 닫아야 한다. 안 닫으면 브라우저가 끊긴 줄 알고 자동으로 다시 붙고, 이미 끝난 작업이라 complete를 또 받는 일이 반복된다.

존재하지 않는 id로 요청하면 스트림 대신 404가 온다.

### 2.3 GET /analyses//result

스트림을 열지 않고 결과만 가져온다. 새로고침 후 결과 화면을 복원할 때 쓴다.

| 상태 코드 | 내용                                                  |
| --------- | ----------------------------------------------------- |
| 200       | 결과 객체                                             |
| 202       | 아직 분석 중. { "status": "RUNNING", "progress": 45 } |
| 404       | 없는 id                                               |
| 500       | 분석 중 서버 예외                                     |

### 2.4 GET /analyses//video

분석에 사용한 원본 영상을 돌려준다. 궤적을 영상 위에 그릴 때 쓴다. 부분 요청을 지원하므로 브라우저에서 탐색이 된다.

보관 기간 7일이 지나 파일이 정리되면 id가 살아 있어도 VIDEO_NOT_FOUND가 온다.

### 2.5 GET /results/

결과 오버레이 이미지를 돌려준다. 결과 객체의 trajectory_image 값을 그대로 경로에 넣는다.

---

## 3. 분석 결과 객체

complete 이벤트의 data와 결과 조회의 200 응답이 같은 형태다.

| 필드                | 타입   | 설명                                 |
| ------------------- | ------ | ------------------------------------ |
| status              | string | VALID 또는 ANALYSIS_FAILED           |
| score               | int    | 슈팅 점수 0~100                      |
| speed_kmh           | float  | 추정 슈팅 속도                       |
| goal                | bool   | 골인 여부                            |
| trajectory          | array  | 공 궤적. { x, y, frame } 객체의 배열 |
| fps                 | float  | 영상의 초당 프레임 수                |
| frame_size          | object | 좌표 기준 해상도. { width, height }  |
| confidence          | float  | 분석 신뢰도 0~1                      |
| processing_time_sec | float  | 분석 소요 시간                       |
| trajectory_image    | string | 결과 이미지 파일명                   |
| comment             | string | 속도 구간별 코멘트                   |
| failure_reason      | string | status가 ANALYSIS_FAILED일 때만 포함 |

실제 응답 예시다.

{
  "status": "VALID",
  "score": 78,
  "speed_kmh": 93.7,
  "goal": false,
  "trajectory": [
    { "x": 150.0, "y": 540.0, "frame": 0 },
    { "x": 240.0, "y": 440.0, "frame": 3 },
    { "x": 350.0, "y": 355.0, "frame": 6 }
  ],
  "fps": 29.97,
  "frame_size": { "width": 1920, "height": 1080 },
  "confidence": 0.82,
  "processing_time_sec": 34.2,
  "trajectory_image": "20261102_141557_shot_result.jpg",
  "comment": "아마추어를 뛰어넘은 멋진 슈팅!"
}

**좌표 규칙**

- trajectory의 x와 y는 **원본 영상 해상도 기준 픽셀 좌표**다. 왜곡 보정을 적용한 경우에도 보정 전 원본 기준으로 변환해서 보낸다.
- frame은 **0부터 시작하는 프레임 번호**다. fps와 함께 쓰면 재생 시간을 계산할 수 있다.
- frame_size는 좌표가 기준으로 삼은 해상도다. 백엔드는 영상을 축소하지 않으므로 원본 해상도와 같지만, 화면 표시 배율 계산의 기준으로 쓰라고 함께 보낸다.

**실패한 경우**에도 같은 형태로 온다. score와 speed_kmh는 0이고, trajectory는 추적이 시작되기 전에 실패했으면 빈 배열, 그 이후 단계에서 실패했으면 그때까지 추적된 좌표가 담긴다. fps와 frame_size는 실패해도 채워진다.

---

## 4. 카메라 보정 흐름

분석과 달리 **스트리밍을 쓰지 않고 주기적으로 조회**한다. 보정은 처리 단계가 나뉘지 않고 시간도 짧아 진행률에 담을 정보가 없기 때문이다.

1. POST /calibrations 로 체스보드 영상을 올리고 id를 받는다
2. GET /calibrations/{id} 를 1초 간격으로 호출한다
3. 202가 오면 진행 중, 200이 오면 끝난 것이다

체스보드는 **내부 코너 9x6, 한 칸 25mm**를 전제한다. 안내 화면의 인쇄 파일도 같은 값이어야 한다.

보정 영상은 분석과 조건이 같지만 **최소 길이만 5초**다. 여러 위치와 각도의 프레임이 필요하기 때문이다.

### 조회 응답

진행 중이면 202와 함께 아래가 온다.

{ "status": "CALIBRATING", "progress": 40 }

성공하면 200과 함께 아래가 온다.

{
  "status": "VALID",
  "frame_size": { "width": 1920, "height": 1080 },
  "used_frames": 23,
  "reprojection_error": 0.418,
  "focal_length": { "fx": 1452.31, "fy": 1449.87 },
  "principal_point": { "cx": 963.52, "cy": 539.11 }
}

보정에 실패해도 **200으로 온다.** status가 FAILED가 된다.

{ "status": "FAILED", "failure_reason": "CHESSBOARD_NOT_FOUND", "used_frames": 0 }

재투영 오차의 품질 구분(양호, 보통, 재촬영 권장)은 화면에서 정한 기준이다. 백엔드는 값만 보낸다.

---

## 5. 오류 코드

### 업로드 단계 (분석과 보정 공통)

요청이 거절되면 아래 형태로 온다.

{ "error_code": "UNSUPPORTED_MEDIA_TYPE", "message": "지원하지 않는 형식의 파일입니다" }

| 상태 코드 | error_code             | 조건                       |
| --------- | ---------------------- | -------------------------- |
| 400       | MISSING_FILE           | file 필드가 없음           |
| 413       | FILE_TOO_LARGE         | 200MB 초과                 |
| 413       | VIDEO_TOO_LONG         | 30초 초과                  |
| 413       | VIDEO_TOO_SHORT        | 1초 미만 (보정은 5초 미만) |
| 415       | UNSUPPORTED_MEDIA_TYPE | mp4, mov가 아님            |
| 415       | RESOLUTION_TOO_LOW     | 짧은 변 480px 미만         |
| 415       | UNREADABLE_VIDEO       | 영상 정보를 읽을 수 없음   |
| 503       | ANALYSIS_UNAVAILABLE   | 분석 라이브러리가 없는 환경 |

### 조회 단계

| 상태 코드 | error_code            | 조건                           |
| --------- | --------------------- | ------------------------------ |
| 404       | JOB_NOT_FOUND         | 없는 분석 id                   |
| 404       | VIDEO_NOT_FOUND       | 영상이 보관 기간 경과로 삭제됨 |
| 404       | IMAGE_NOT_FOUND       | 없는 결과 이미지               |
| 404       | CALIBRATION_NOT_FOUND | 없는 보정 id                   |
| 500       | ANALYSIS_ERROR        | 분석 중 서버 예외              |

**오류 응답은 전부 error_code와 message 두 필드로 통일되어 있다.**

### 분석 실패 원인 (failure_reason)

결과 객체 안에 담겨 온다. 업로드 오류와는 다른 층위다.

| 코드              | 의미                       |
| ----------------- | -------------------------- |
| BALL_NOT_DETECTED | 정지 상태의 공을 찾지 못함 |
| TRACK_TOO_SHORT   | 추적된 궤적이 너무 짧음    |
| FIT_FAILED        | 궤적 계산 실패             |
| LOW_CONFIDENCE    | 신뢰도가 기준에 미달       |

	

뒤의 세 가지는 사용자 관점에서 모두 "공을 제대로 추적하지 못했다"에 해당하므로 같은 화면을 써도 된다. 백엔드가 나눠 두는 것은 데이터 파트가 실패 사례를 분류할 때 어느 단계에서 실패했는지 구분하기 위해서다.

GOAL_NOT_DETECTED 코드도 정의되어 있으나 골인 판정 로직이 없어 아직 반환되지 않는다.

### 보정 실패 원인

| 코드                 | 의미                                   |
| -------------------- | -------------------------------------- |
| CHESSBOARD_NOT_FOUND | 어느 프레임에서도 체스보드를 찾지 못함 |
| NOT_ENOUGH_FRAMES    | 찾긴 했으나 계산에 필요한 5장에 미달   |
| VIDEO_OPEN_FAILED    | 영상을 열 수 없음                      |
| CALIBRATION_FAILED   | 보정 계산 실패                         |
| SAVE_FAILED          | 보정값 저장 실패                       |

앞의 두 가지는 안내가 달라야 한다. 전자는 체스보드가 화면에 제대로 들어오지 않은 것이고, 후자는 더 길게 여러 각도로 촬영해야 한다는 뜻이다.

---

## 6. 업로드 제한값

| 항목        | 분석               | 보정               |
| ----------- | ------------------ | ------------------ |
| 확장자      | mp4, mov           | mp4, mov           |
| 최대 용량   | 200MB              | 200MB              |
| 길이        | 1초 이상 30초 이하 | 5초 이상 30초 이하 |
| 최소 해상도 | 짧은 변 480px 이상 | 짧은 변 480px 이상 |

프론트엔드가 먼저 정한 값을 그대로 채택했다. 한쪽에서만 바꾸면 화면에서는 통과한 영상이 서버에서 거절되므로, 변경이 필요하면 양쪽이 같이 바꿔야 한다.

---

## 7. 지금 안 되는 것

연동하기 전에 알고 계셔야 할 제약이다.

**배포된 서버에서는 영상 분석과 카메라 보정이 동작하지 않는다.** 분석 모델 파일이 아직 준비되지 않아, 배포된 서버에는 분석 라이브러리를 올리지 않았다. 영상을 올리면 503과 함께 ANALYSIS_UNAVAILABLE이 온다. 그 외의 기능인 예시 작업 조회, 결과 조회, 진행률 구독, 보정 상태 조회는 모두 정상 동작한다. 모델이 준비되면 같은 주소에 다시 배포하므로 주소는 바뀌지 않는다.

**보정값이 분석에 적용되지 않는다.** 보정 결과를 id별로 저장하는 데까지는 되어 있지만, 분석 요청에 calibration_id를 보내도 아직 쓰이지 않는다. 분석 코드가 고정된 파일에서 보정값을 읽고 있어 그 부분을 바꿔야 하는데, AI 파트와 협의가 필요한 작업이다.

**골인 여부는 항상 false다.** 필드는 있지만 판정 로직이 구현되지 않았다.

**서버를 재시작하면 진행 중이거나 끝난 작업을 조회할 수 없다.** 작업 정보를 서버 메모리에 두기 때문이다. 영상 파일은 7일 남지만 id로 찾을 수 없게 된다.

**분석 취소 기능이 없다.** 화면에서 연결을 끊으면 더 이상 진행률을 받지 않지만, 서버 쪽 분석은 끝까지 진행된다.

**사용자 구분이 없다.** id를 알면 누구나 조회할 수 있다. 로그인이나 권한 확인이 없다.

---

## 8. 문의가 있던 항목에 대한 답

프론트엔드 API 명세 v1에서 확인이 필요하다고 표시한 항목에 대한 현재 상태다.

**오류 응답 형식** — 모든 오류가 error_code와 message 두 필드로 통일되어 있다. 5장 참고.

**교차 출처 설정** — 모든 출처를 허용하도록 되어 있어 별도 설정 없이 호출된다. 배포 시점에는 좁혀야 한다.

**연결 유지 신호 형식** — 이름 있는 데이터 이벤트로 보내므로 브라우저가 그대로 전달한다. 약 0.2초 간격이라 60초 시간 초과에 걸릴 일은 없다.

**진행률 단계 값** — 어떤 값을 기대하는지 확인이 필요하다. 현재는 0에서 100 사이의 정수만 보내고 단계 이름은 보내지 않는다. 분석 내부적으로는 정지 공 탐색이 0~15, 프레임별 탐지와 추적이 15~90, 궤적 계산과 시각화가 90~100 구간이다. 단계 이름이 필요하면 추가할 수 있다.
