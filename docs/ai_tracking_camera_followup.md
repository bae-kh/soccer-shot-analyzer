# 공 재탐지와 카메라 이동 판정 보완

2026-10-11. 5단계 이후의 작업 1(남은 추적 의심 구간 검토·보완), 작업 2(카메라 이동 검사 검증·보완)의 개발과 아래 범위의 검증을 완료했다. 실제 속도·골인·점수 정확도 검증은 별도 작업이다. README는 단계 종료 후 요청받을 때 업데이트한다.

## 원본에서 확인한 내용

- 실촬영 5번 f101~103은 이전 표본 검토에서 배경·가림 위치를 따라가는 것으로 의심했다. 연속 원본과 확대에서 공이 박스 부근에 보이는 것을 확인해 **이전 판단을 정정**했다. 실제 공 관측을 제거하지 않으며 f100~103·107~108을 추가 정상 대조로 보호한다. 중심·지름의 정밀 정답을 지정한 것은 아니다.
- 테스트 12번은 f30~32의 공 관측 뒤 검출이 끊기고, f37에서 신뢰도 0.069의 검출이 선수의 발·가림 위치로 옮겨가며 f38~39에 CSRT가 이어졌다. 원본에서 이 박스 안의 공을 분명하게 확인할 수 없어 관측으로 채택하지 않도록 보완한다. 가려진 공의 실제 위치를 복원했다고 주장하지 않는다.
- 슬로모션 테스트 1·3·4번은 선수와 떨어진 배경 특징들도 함께 변환된다. 기존 이동 보류를 단순 선수 움직임에 의한 오탐으로 해제하지 않는다. 테스트 2번도 시작→끝에 작은 변화가 누적되며 기존의 두 구간 검사로는 놓칠 수 있다.

원본 연속 구간 39프레임과 카메라 의심 9개 영상의 시작·중간·끝을 추출했다. 구간 검토·계산 자료는 `output/ai_baseline/followup_review_20261011`에 있다.

- [원본 구간 및 수정 전 진단](../output/ai_baseline/followup_review_20261011/original_review.json)
- 실촬영 5 원본: [1](../output/ai_baseline/followup_review_20261011/real_test_videos_video5_page1.jpg), [2](../output/ai_baseline/followup_review_20261011/real_test_videos_video5_page2.jpg)
- 테스트 12 원본: [1](../output/ai_baseline/followup_review_20261011/test_videos_video12_page1.jpg), [2](../output/ai_baseline/followup_review_20261011/test_videos_video12_page2.jpg)
- [배경 특징 대응 사전 대조](../output/ai_baseline/followup_review_20261011/camera_probe.json)
- [의심 박스의 원본 모습 검사](../output/ai_baseline/followup_review_20261011/tracking_probe.json)

## 코드 변경

### 약하고 불확실한 재검출의 확인

`backend/ball_tracking.py`에서 이미 시작된 추적의 마지막 YOLO 관측으로부터 3프레임 이상 지났고, 신뢰도 0.12 미만의 검출이 `max(마지막 공 반지름 × 3, 영상 대각선 × 0.02)`보다 먼 곳에 나타나면 추가 확인한다.

- 원본 공과 색·밝기 분포가 각각 0.55 이상 일치하면 기존처럼 재탐지를 허용한다. 약한 검출 자체를 전부 제거하지 않는다.
- 모습의 근거가 부족하면 같은 대상을 연속 두 프레임의 YOLO 관측으로 확인한 뒤 재탐지한다. 첫 프레임은 `WEAK_REACQUISITION_UNCONFIRMED` 사유를 기록하고 관측으로 채택하지 않는다.
- 실제 이미지 검사가 불가능할 때도 약한 먼 검출 하나로 대상을 바꾸지 않는다. 강한 검출·가까운 위치의 재검출·연속 관측은 기존 조건을 유지한다.
- ROI 검사를 위한 같은 프레임의 후보 조회는 시간적 확인 횟수에 포함하지 않는다. 새로 확인되지 않은 검출로 CSRT를 초기화하거나 Kalman 상태를 보정하지 않는다.

이 기준은 정답 라벨로 최적화한 확률 기준이 아닌 보수적인 휴리스틱이다. 연속 오검출, 조명 변화·공 패널 회전, 강한 오검출의 문제는 남을 수 있다. 가림 구간의 예측점은 공을 관측한 정답 위치가 아니다.

### 공간에 분산된 배경 대응과 누적 이동

`backend/camera_geometry.py`의 단방향 광학 흐름 대신 양방향 비율 검사를 통과한 SIFT 특징 대응과 RANSAC 유사 변환을 사용한다. 위쪽 배경 후보 영역에서 다음 조건을 검사한다.

- 대응·합의점 각각 20개 이상, 합의 비율 60% 이상.
- 4×3 공간 격자의 4칸 이상에 합의점이 있고 가로 35%·세로 20% 이상으로 분산.
- 변환 후 합의점의 중앙 오차 1px 이하. 분석 프레임은 최대 가로 960px로 축소한다.
- 시작→중간, 중간→끝에 더해 시작→끝도 검사해 누적 이동을 확인한다.
- 기존 이동 기준(대각선 대비 0.003, 배율 변화 0.005, 회전 0.25°)을 유지한다.
- 검증된 구간 하나에서 이동이 확인되면 `MOVING`; 모든 필요한 구간의 근거가 확보되고 기준 이하일 때만 `FIXED_CANDIDATE`; 프레임·특징·공간 근거가 부족하면 `UNVERIFIED`다.

`samples`에 검사 프레임·대응 수·공간 분포·변환·오차를, `rejected_pairs`에 부족한 근거를 저장한다. 변환 행렬은 축소된 분석 영상 좌표계의 진단용이며 실제 경기장 좌표·3D 계산 보정에 사용하지 않는다. 움직이는 선수·가림과 고정된 재생 UI의 영향은 남을 수 있고 전체 프레임의 고정 여부도 증명하지 않는다. `MOVING`은 영상 배경 변화이며 물리적 카메라 이동·안정화·편집 등 원인을 확정하는 값이 아니다.

## 검증

- 전체 Python 테스트 102개 통과.
- 약한 재탐지의 연속 확인·다른 대상과의 구분·강한/가까운 검출 보호·같은 프레임 ROI 조회, 실제 모습이 일치하는 약한 공 보호를 검사했다.
- 알려진 합성 영상에서 고정 배경과 움직이는 전경, 평행 이동·줌, 작은 이동의 누적, 전경에만 있는 특징·누락 프레임·단일 프레임의 보류를 검사했다.
- 수정 후 실제 32개 YOLO·CSRT 재분석과 결과 파일·현재 소스/입력/모델/보정 해시 검사를 모두 통과했다.
- 프론트엔드 Node 테스트 8개 통과. 실제 업로드에서 모델 계산을 대체하지 않고 AI 분석·SSE 진행 이벤트 26개·결과 사진 GET 200을 확인했다.

| 전체 재분석·회귀 비교 | 결과 |
|---|---|
| 기존 정상 관측 대조 | 29개 모두 중심·박스 유지 |
| 실촬영 5 추가 정상 대조 | f100~103·107~108의 6개 모두 실제 공 관측 유지 |
| 기존 오류 방지 대조 | 3단계의 8개와 5단계의 4개, 총 12개 유지 |
| 추적 관측 변경 | 테스트 12 f37·38은 예측, f39는 누락으로 변경; 다른 31개 영상의 관측 중심·박스·출처·신뢰도 동일 |
| 관측 출처 합계 | YOLO 1,789개, CSRT 688개, 예측 73개, 누락 355개 |
| 초기 계산 구간 | 32개 모두 선택 프레임·구간·관측 수 유지 |
| 속도 | 32개 모두 null: 촬영 시간 미확인 29개, 초기 카메라 이동 3개 |
| 골대 범위 카메라 검사 | 고정 후보 22개 / 이동 10개; 테스트 2번이 누적 이동으로 추가 보류 |
| 영상 영역 진입 | 후보 13개 / 후보 미관측 1개 / 보류 18개; 실제 골인·실패 확정 수가 아님 |
| 점수 | 32개 모두 미평가·null 유지 |

초기 속도 구간의 카메라 이동 3개와 전체 골대 검사 범위의 10개는 서로 다른 검사 범위다. 이동 보류 10개는 실촬영 3·18·20, 테스트 1·2·3·4·8·9·10이다. 테스트 2의 기존 영상 영역 진입 후보는 새 전체 범위 검사에서 보류로 바뀌었다.

슬로모션 1~4의 시작→끝 변환은 다음과 같다. 이동은 분석 영상 모서리 이동의 중앙값을 대각선으로 나눈 수치다.

| 테스트 영상 | 대각선 대비 이동 | 회전 | 합의 비율 | 결과 |
|---|---:|---:|---:|---|
| 1 | 0.445% | 0.491° | 98.1% | 이동 보류 유지 |
| 2 | 0.422% | 0.360° | 99.5% | 누적 이동 추가 보류 |
| 3 | 1.250% | 0.141° | 68.7% | 이동 보류 유지 |
| 4 | 0.573% | 0.251° | 82.8% | 이동 보류 유지 |

양성·음성 대조는 선택한 회귀 사례다. 실제 전체 관측의 검출 정확도·재현율이나 카메라 이동의 정답 정확도를 측정한 결과는 아니다.

최신 실행 산출물: `output/ai_baseline/followup_tracking_camera_20261011`.

- [전체 실행](../output/ai_baseline/followup_tracking_camera_20261011/summary.json)
- [실행 환경·소스·입력·모델 해시](../output/ai_baseline/followup_tracking_camera_20261011/run_metadata.json)
- [좌표·관측 근거·미디어 검사](../output/ai_baseline/followup_tracking_camera_20261011/geometry_audit.json)
- [추적·카메라·정상 보호·오류 회귀 비교](../output/ai_baseline/followup_tracking_camera_20261011/followup_comparison.json)
- [같은 원본의 추적 전후 사진](../output/ai_baseline/followup_tracking_camera_20261011/tracking_before_after.jpg)
- [슬로모션 1~4의 배경 대응](../output/ai_baseline/followup_tracking_camera_20261011/camera_evidence.jpg)
- [표시한 대응점과 카메라 수치](../output/ai_baseline/followup_tracking_camera_20261011/camera_evidence.json)
- [실제 업로드·AI·SSE·사진 조회](../output/ai_baseline/followup_tracking_camera_20261011/api_validation/api_validation.json)

## 재현

새 전체 실행의 출력 폴더는 기존 폴더와 다른 경로를 지정한다.

```powershell
& 'C:\anaconda3\envs\capstone\python.exe' -m unittest discover -s tests
& 'C:\anaconda3\envs\capstone\python.exe' -u tools/run_ai_baseline.py --output-dir output/ai_baseline/followup_new
& 'C:\anaconda3\envs\capstone\python.exe' tools/audit_ai_geometry.py output/ai_baseline/followup_new
& 'C:\anaconda3\envs\capstone\python.exe' tools/compare_ai_followup.py output/ai_baseline/followup_new
& 'C:\anaconda3\envs\capstone\python.exe' tools/render_camera_followup.py output/ai_baseline/followup_new
& 'C:\anaconda3\envs\capstone\python.exe' -u tools/verify_goal_api.py --output output/ai_baseline/followup_new/api_validation
```

## 남은 범위

실제 시간·설정별 보정·독립 기준 속도를 확보한 현장 속도 검증(작업 3), 실제 골라인 통과 근거·채점 기준과 점수 계산(작업 4)은 이번 작업에 포함하지 않는다. 표본 비교·합성 카메라 검사·전체 실행 성공을 실제 전체 추적·카메라·속도·골인 정확도의 합격률로 해석하지 않는다.

이후 진행한 작업 3~4의 계산·서비스 연결과 실측 검증의 남은 조건은 [골대 면·점수 및 실측 검증 후속 문서](ai_goal_speed_followup.md)에 정리했다. 위 실행 수치는 추적·카메라 보완 당시의 기록이다.
