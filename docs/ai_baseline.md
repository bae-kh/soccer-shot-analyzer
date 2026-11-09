# AI 일괄 실행 안내와 단계별 판정

현재 서비스가 사용하는 `backend/analyze_shot.py`를 호출해 예시 영상을 분석한다.
이 작업의 목적은 영상별 실행 결과와 다음 단계의 실패 원인을 확보하는 것이다.
실행 성공과 내부 일관성을 확인하며 정답 속도 정확도·골인 정확도의 검증은 포함하지 않는다.

첫 실행의 요약은 [1단계 실행 결과](ai_baseline_stage1_results.md)에서 확인할 수 있다.
현재 계산은 3단계다. [3단계 구현 결과](ai_stage3_results.md)와 [2026-10-11 재검증·진행 판단](ai_stage3_review.md)을 참고한다.

## 확인한 실행 환경

- Python: `C:\anaconda3\envs\capstone\python.exe` (3.9.25)
- GPU: NVIDIA GeForce RTX 4070 Ti, CUDA 사용 가능
- PyTorch: 2.5.1+cu121 / torchvision: 0.20.1+cu121
- Ultralytics: 8.4.21 / NumPy: 2.0.2 / SciPy: 1.13.1
- OpenCV standard/contrib: 4.13.0.92, `TrackerCSRT_create` 확인

이번 실행에서는 기존 환경을 사용했으며 추가 패키지를 설치하지 않았다.
`backend/requirements-ai.txt`는 확인한 AI 의존성 버전을 기록한다.
CUDA 빌드 선택과 전체 환경의 잠금 파일을 대체하지 않으며 새 환경 설치는 이번 작업에서 검증하지 않았다.
Ultralytics가 standard OpenCV를 의존성으로 설치하므로 contrib와의 공존 시에는 CSRT 심볼을 반드시 확인한다.

## 필요한 로컬 파일

```text
backend/yolov8s.pt
backend/goal_segment_best.pt
soccer_videos/test_videos/*.mp4
soccer_videos/real_test_videos/*.mp4
```

모델 경로는 현재 분석기가 `__file__` 기준으로 정한다.
일괄 실행기는 작업 디렉터리를 프로젝트 루트로 고정한다.
3단계 분석기는 `backend/calibration.json`을 우선 읽고 기존 루트 파일도 지원한다.
두 파일 모두 없으면 수평 화각 60°를 가정하고, 잘못된 보정 파일이면 속도를 보류한다.
1·2단계 이력에는 이전 경로·고정 거리 fallback 계산이 포함되어 있다.

## 실행

프로젝트 루트에서 PowerShell로 실행한다.

```powershell
& 'C:\anaconda3\envs\capstone\python.exe' -B tools/run_ai_baseline.py
```

출력 경로를 지정할 때는 기존 결과를 덮어쓰지 않도록 새로운 폴더를 사용한다.

```powershell
& 'C:\anaconda3\envs\capstone\python.exe' -B tools/run_ai_baseline.py --output-dir output/ai_baseline/my_run --timeout 240
```

`--limit 1`은 간단한 실행 확인, `--input-dir`는 다른 영상 폴더 지정에 사용한다.
각 영상은 별도 프로세스로 실행하고 240초 기본 제한을 적용한다.
오류나 시간 초과가 발생해도 다음 영상을 계속 분석한다.

## 결과 파일과 판정 의미

- `report.md`: 영상별 결과 표
- `summary.csv`: 비교용 요약, UTF-8 BOM 포함
- `summary.json`: 원본 AI 결과, 실패 이유, 영상 메타데이터, 실행 진단
- `run_metadata.json`: 실행 환경과 분석 코드/모델/입력 파일 SHA-256
- `images/<영상그룹>/<영상명>.jpg`: 결과 이미지
- `images/<영상그룹>/<영상명>_observed.jpg`: 초기 구간 관측 이미지 (3단계)
- `images/<영상그룹>/<영상명>_tracking.json`, `_tracking.mp4`: 프레임별 추적과 물리 계산 진단
- `logs/<영상그룹>/<영상명>.log`: 영상별 로그와 예외
- `results/<영상그룹>/<영상명>.json`: 영상별 상세 결과

3단계의 `ESTIMATED`는 내부 검사 조건을 통과한 속도 추정, `UNAVAILABLE`은 속도 `null`과 진단·이미지를 남긴 계산 보류다. `FAILED`는 예외·프로세스 실패·시간 초과·반환 규약 오류 등 실행 오류다.

1·2단계의 `SUCCESS`는 양의 유한 속도와 읽을 수 있는 결과 이미지가 반환되었다는 뜻이다.
공을 정확하게 추적했거나 속도가 실측값과 일치한다는 뜻은 아니다.
`BALL_NOT_FOUND`, `TRACK_TOO_SHORT`, `FIT_FAILED`는 기존 AI의 실패 코멘트로 구분한다.
입력 오류, 예외, 결과 이미지 누락, 프로세스 실패, 시간 초과도 별도 기록한다.

처리 시간은 프로세스 시작과 모델 로딩을 포함한다. 웹 업로드와 UI 시간은 포함하지 않는다.
현재 표의 Points는 초기 비행 구간에서 선택한 YOLO 관측 수다. 이전 단계 표는 optimizer 입력 수이며 1단계에는 합성 앵커도 포함된다. 모두 Track Coverage로 해석하지 않는다.
`goal_area_detected`는 골대 영역 반환 여부다. 이전 `goal_scale_detected`는 기존 초점거리 계산 로그에서 읽은 값이며 골인 판정이 아니다.
150km/h 상한은 3단계에서 제거했다. `speed_capped`는 이전 코드 결과의 상한 도달 여부를 나타낸다.
실행 결과는 각 영상이 끝날 때 갱신하며, 대용량 결과와 런타임 설정은 Git에서 제외한다.

## 검사

```powershell
& 'C:\anaconda3\envs\capstone\python.exe' -B -m unittest discover -s tests -p 'test_ai_baseline.py' -v
```

실패를 성공으로 집계하지 않는지, 이미지 누락과 비정상 속도를 구분하는지,
두 폴더의 동명 영상 결과가 충돌하지 않는지, 피팅 실패 진단을 보존하는지 검사한다.
