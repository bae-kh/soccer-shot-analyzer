# 남은 AI 작업 3~4: 실측 검증 준비와 골대 면·채점 개발

2026-10-11. 이전 추적·카메라 보완 이후의 작업이다. **계산과 서비스 연결을 개발한 것과 실제 촬영 정확도를 검증한 것은 구분한다.** 사용자에게 촬영 기기·배율·보정·독립 정답 자료가 아직 없다는 답을 받은 상태로 개발했다.

| 작업 | 현재 상태 | 완료를 위해 남은 것 |
|---|---|---|
| 1. 추적 오류 보완 | 이전 작업 완료, 이번 변경의 회귀 검증 | 실제 정답 라벨을 통한 정량 평가 |
| 2. 카메라 움직임 검증 | 이전 작업 완료, 이번 변경의 회귀 검증 | 전체 프레임 고정 여부의 실측 검증 |
| 3. 실제 속도 검증 | 정답 입력·일치 확인·오차 집계 도구 구현 | 같은 촬영 설정의 보정·시간 정보 및 독립 속도 정답 |
| 4. 골인·점수 | 측정된 골대 면 통과 추정·선택한 채점 규칙·API/화면 연결 구현 | 골대 실측, 독립 골라인 정답, 실제 평가와 채점 기준 확정 |

따라서 기존 32개 영상에서 정확한 속도·확정 골인·최종 점수를 제공하는 AI가 완성됐다고 보고하지 않는다. 측정값을 입력하는 방식의 프로토타입이며, 실제 골대 면 네 모서리를 자동으로 확정하는 기능은 포함하지 않는다.

## 구현 아이디어

기존 사각형 검출은 “영상 속 골대 영역”이다. 멀리 있는 공이 골대 앞에서 겹쳐 보여도 이 영역 안에 보일 수 있다. 새 `backend/goal_measurement.py`는 별도로 측정한 골대 안쪽 폭·높이와 같은 골라인 경계면의 네 모서리를 받아 보정된 카메라의 골대 면 방향·거리를 계산한다.

공 위치는 **각 프레임에서 실제 YOLO가 관측한 공 박스**의 크기로 복원한다. 초기 0.45초 속도 모델을 골대까지 연장하지 않는다. 예측·CSRT 크기·낮은 신뢰도·가림이나 크기 불량 관측으로 깊이를 확정하지 않는다. 중간 관측이 빠지면 연속 통과 근거가 끊어진다.

공 전체가 골라인 앞쪽에 있던 관측에서 뒤쪽에 있는 관측까지 연결하고, 공이 면을 통과하는 구간 동안 기둥·크로스바 사이에 충분한 여유가 있는지 확인한다. 중심만 넘었거나 경계와 깊이가 불확실하면 보류한다. 여러 프레임에 걸친 느린 통과도 처리한다. 모서리와 공 박스의 픽셀 오차를 바꿔 결과의 민감도를 확인한다.

골대 사이에서 공 전체가 골라인을 통과해야 한다는 기하 조건은 [IFAB Law 10](https://www.theifab.com/laws/latest/determining-the-outcome-of-a-match/)을 참고했다. 팀의 반칙 등 심판 판정은 분석하지 않는다. 단일 카메라의 깊이 근사, 제한된 픽셀 민감도 검사, 프레임 사이 직선 이동 가정이 있어 **`CONFIRMED`로 반환하지 않는다.** 픽셀 민감도는 통계적 신뢰구간이나 전체 오차 상한이 아니다.

| 새 `physical_goal.status` | 의미 |
|---|---|
| `ESTIMATED_GOAL` | 입력한 골대 면 안에서 공 전체 통과를 조건부 추정 |
| `ESTIMATED_OUTSIDE` | 골대 면을 기둥 바깥·크로스바 위로 통과한 것으로 추정 |
| `UNDETERMINED` | 자료·연속 관측·경계 여유 부족 등으로 미확정 |

진입 후보가 없다는 이유만으로 실패·실축을 선언하지 않는다. 기존 `goal_entry.status`는 영상 영역 진입 후보를 계속 나타내고, 새 `physical_goal`은 별도의 근거·관측 프레임·좌표·민감도를 제공한다. 골대 면에서 멀리 떨어지거나 공이 작게 보이면 깊이 오차가 커 보류될 수 있다. 기둥·바닥과 가까운 통과도 보수적으로 보류할 수 있다.

## 점수

설계서에 구체적인 수치·가중치가 없으므로 실제 영상에 자동으로 적용되는 기본 점수표는 만들지 않았다. 명시적으로 선택한 규칙만 적용한다.

```
속도 항목 = clamp((추정 속도 - 최소 속도) / (최대 속도 - 최소 속도), 0, 1)
정확도 항목 = max(0, 1 - 골대 면 통과 위치와 목표 사이의 거리(m) / 기준 거리(m))
참고 점수 = 100 × (속도 비중 × 속도 항목 + 정확도 비중 × 정확도 항목)
```

정확도는 영상 사각형의 중심까지의 거리가 아니라, 측정한 골대 면의 미터 좌표와 선택한 목표의 거리다. 이 점수의 “정확도 항목”은 슈팅 목표 정확도이며, AI 알고리즘 정확도 지표와 다르다.

- 비중의 합은 1이어야 한다. 속도 범위·목표·기준 거리·골대 밖 통과 점수를 규칙에 기록한다.
- 속도 비중이 양수이면 유효한 속도 추정이 필요하다. 골대와 속도 계산의 공 반지름 가정이 다르면 채점을 보류한다. 현재 속도 엔진의 반지름은 0.11 m다.
- 속도 비중 0인 정확도 전용 규칙은 속도 없이도 계산할 수 있다. 골대 밖 통과 점수 역시 선택한 별도 규칙으로 계산한다.
- 영상 영역 진입 후보만 있으면 채점하지 않는다. 자료가 없으면 `score=null`, `NOT_EVALUATED`다.
- 계산한 점수는 `score_status=PROVISIONAL`, “참고 점수 · 실측 미검증”으로 표시한다. 공인·검증된 채점 기준이라고 주장하지 않는다.

합성 검증의 20~100 km/h, 속도 40%·정확도 60%, 중심 목표·기준 거리 1 m는 **산술 검증을 위한 예시**다. 실제 서비스 규칙으로 확정한 값이 아니다. 예를 들어 중심 통과와 입력 속도 60 km/h에서 80점이 나오는지 검증했다. 이 검증의 60 km/h는 속도 계산 결과가 아니라 채점 함수에 제공한 입력값이다.

## API와 화면 사용

기존 `/analyze` multipart 요청의 `capture_options` JSON에 선택 필드 `goal_plane`, `scoring_rules`를 추가했다. 이전 클라이언트의 요청은 계속 동작한다. 반환값에 `physical_goal`을 추가하고, 기존 `score`, `score_status`, `score_evaluation`에 계산값 또는 보류 이유를 연결했다. 저장한 추적 JSON에도 같은 값을 기록한다.

화면의 **골대 실측 · 참고 점수 설정**에서 JSON 파일을 선택한다. 파일을 읽는 것만으로 확인 체크가 켜지지 않는다. 영상 선택 전에 해당 영상의 측정값과 채점 기준을 확인해야 하며, 자료가 없으면 설정을 불러올 필요가 없다. 원본 해상도 기준 좌표와 0부터 시작하는 프레임 번호를 사용한다.

골대 설정은 다음 필드를 모두 요구한다.

| 필드 | 값 |
|---|---|
| `width_m`, `height_m` | 해당 골대 개구부의 실측 폭·높이 |
| `corners_px` | 좌상→우상→우하→좌하, 같은 평면의 네 안쪽 모서리 원본 좌표 |
| `reference_frame` | 측정한 원본 프레임 번호 |
| `plane_reference` | `rear_edge_goal_line` — 골라인 뒤쪽 경계의 수직 면 |
| `confirmed_by_user` | 해당 영상의 측정값이라는 확인 |
| `pixel_tolerance_px` | 모서리와 공 박스에 적용할 픽셀 민감도 검사값, 0 초과~5 이하 |
| `ball_radius_m` | 분석할 공의 반지름 가정 |

같은 카메라·렌즈·줌의 보정과 크롭 여부 확인이 필요하다. 전체 골대 관측 구간과 수동 측정 기준 프레임까지 카메라 움직임 검사 범위를 확장했다. 골대 면 계산 자체는 프레임 순서를 사용하므로 실제 촬영 FPS를 몰라도 가능하지만, 속도와 속도를 포함한 점수는 촬영 시간 확인이 필요하다.

현재 통과 방향은 카메라가 **골라인 앞 필드 쪽**에 있다는 조건으로 정한다. 골대 뒤에 둔 카메라는 이 모드에서 지원하지 않는다. 화면의 측정 확인 체크 및 API의 `confirmed_by_user=true`는 이 카메라 위치 조건까지 확인한다는 뜻이다. 카메라가 필드 쪽인지 AI가 독립적으로 검증하지는 않는다.

채점 규칙의 필드는 `id`, `confirmed_by_user`, `accept_conditional_estimates`, `speed_min_kmh`, `speed_max_kmh`, `speed_weight`, `accuracy_weight`, `target_normalized`, `accuracy_radius_m`, `outside_score`다. 목표 좌표의 `(0,0)`은 골대 좌상단, `(1,1)`은 우하단이다. 잘못된 형식·수치·미확인 설정은 `/analyze`에서 422로 거부한다.

[합성 설정 형식 예시](../output/ai_baseline/goal_plane_verified_20261011/synthetic_config_example.json)는 가상의 3×2 m 골대다. 실제 영상의 설정값으로 복사하면 안 된다. 저장된 예시의 확인 값은 `false`이며 직접 API에 제출하면 거부된다.

## 독립 정답 검증 도구

`tools/evaluate_references.py`가 실행 결과와 각 영상의 해시를 묶은 정답 입력표를 만든다. 정답은 AI 실행 입력으로 사용하지 않고, 분석이 끝난 뒤 별도로 비교한다. 결과·영상 파일이 바뀌거나 항목이 중복되면 비교를 중단한다. 실영상과 합성 자료를 별도 집계하며 슬로모션·줌 등 촬영 조건 그룹도 따로 보고한다.

속도 정답은 `initial_flight_mean_speed`여야 한다. 정답의 프레임 구간과 AI 초기 비행 구간이 일치해야 한다. 순간·최고 속도 또는 단순 2D 구간 거리 속도를 초기 3D 평균 속도의 정답으로 섞지 않는다. 지원하는 출처는 같은 구간의 레이더 평균(`radar_window_mean`)과 독립적인 3D 추적(`independent_3d_tracking`)이다. 각각 `independent=true`, 근거 자료 식별자와 구간을 요구한다.

골인 정답은 독립 골라인 시점의 판독(`independent_goal_line_view`)으로 입력한다. 같은 정면 영상에서 추정한 진입 후보를 정답으로 복사하지 않는다. 근거 식별자를 기록하지만 도구가 외부 근거 영상이나 장비의 신뢰성을 자동으로 인증하지는 않는다.

정답이 없으면 오차·정확도는 **null**이다. AI가 보류한 영상은 성공률·비교 가능 비율의 분모에 남고, 오차를 0으로 처리하지 않는다. 수치가 생겨도 프로젝트의 합격 기준까지 자동으로 충족했다고 선언하지 않는다.

```powershell
python tools/evaluate_references.py --run output/ai_baseline/remaining_goal_speed_final_20261011 --output output/ai_baseline/remaining_goal_speed_final_20261011/reference_ledger.json
python tools/evaluate_references.py --ledger output/ai_baseline/remaining_goal_speed_final_20261011/reference_ledger.json --output output/ai_baseline/remaining_goal_speed_final_20261011/reference_evaluation.json
```

`reference.speed`, `reference.goal`의 초기 값은 null이다. 독립 자료가 확보된 후 별도 사본에만 실제 정답을 입력한다. JSON 구조는 `tests/test_reference_validation.py`의 예시와 도구의 검증 조건을 참고한다.

## 검증

Python 127개, 프론트엔드 11개 테스트, lint와 프로덕션 빌드가 통과했다. 추가 테스트는 공 전체/중심만 통과, 앞쪽 겹침, 비스듬한 골대, 여러 프레임 통과, 역방향, 기둥 바깥·크로스바 위, 경계 오차, 관측 누락·CSRT·약한 검출, 보정·카메라 조건, 점수 산술·가정 불일치, 독립 정답 비교와 API 전달을 확인한다.

독립적인 구 투영 수식으로 생성한 합성 관측 10가지가 예상 상태를 반환했다. 이 검증은 YOLO 검출 정확도나 실제 영상의 골인 정확도를 검증하지 않는다.

- [합성 통과/보류 사례 모음](../output/ai_baseline/goal_plane_verified_20261011/synthetic_goal_cases.jpg)
- [합성 계산 기록](../output/ai_baseline/goal_plane_verified_20261011/goal_validation.json)
- [합성 결과 화면](../output/ai_baseline/goal_plane_verified_20261011/review/slow_inside_result.html)
- [합성 화면 캡처](../output/ai_baseline/goal_plane_verified_20261011/review/slow_inside_screen.png)

실영상의 최종 실행·이전 결과 비교·정답 입력표 결과는 아래 최종 검증 절에 기록한다.

## 최종 실영상 검증 기록

`output/ai_baseline/remaining_goal_speed_final_20261011`에서 실제 예시 영상 32개를 최종 코드와 로컬 모델로 새로 실행했다. 영상·모델·분석 코드·설정의 해시, 저장 JSON·이미지·추적 영상의 계약 검사가 통과했다. 이는 정확도 합격이 아니라 실행 및 결과 일관성 검사다.

| 확인 항목 | 결과 |
|---|---|
| 실행 오류 | 0개, 32개 모두 결과 이미지·분석 기록 생성 |
| 이전 실행 대비 공 관측·전체 프레임 기록 | 32개 모두 동일, 변경 프레임 0 |
| 초기 비행 선택·속도 상태·영상 영역 진입 후보 | 32개 모두 이전과 동일 |
| 관측 종류 | YOLO 1,789 / CSRT 688 / 예측 73 / 누락 355 |
| 속도 | 32개 null — 촬영 시간 미확인 29, 초기 구간 카메라 움직임 3 |
| 전체 골대 구간 카메라 | 고정 후보 22 / 움직임 10 |
| 골대 영상 영역 | 진입 후보 13 / 후보 없음 1 / 보류 18 |
| 골대 면 통과 | 32개 `UNDETERMINED`, 실측 골대 면 정보가 없어 보류 |
| 점수 | 32개 null, 채점 규칙·필요 자료 미입력 |
| 평균 처리 시간 | 15.276초, 프로세스 시작·모델 로딩 포함, 업로드·UI 제외 |
| 실제 정답 비교 | 속도·골인 각각 라벨 0/32, 비교 가능한 비율 0% |
| 실제 속도 MAE/MAPE 및 골인 정확도 | null — 평가 불가 |

FastAPI의 실제 multipart 업로드→YOLO/CSRT 분석→26개 SSE 진행 이벤트→결과 반환→결과 JPEG 조회(HTTP 200)도 확인했다. 이 API 검증의 모델은 대체하지 않았다. 별도의 설정 전달·422·참고 점수 상태 전달 테스트에서는 분석기를 대체해 요청/응답 계약을 확인했다.

- [전체 실행 요약](../output/ai_baseline/remaining_goal_speed_final_20261011/summary.json)
- [소스·영상·결과 계약 감사](../output/ai_baseline/remaining_goal_speed_final_20261011/geometry_audit.json)
- [이전 32개 결과와 비교](../output/ai_baseline/remaining_goal_speed_final_20261011/remaining_comparison.json)
- [실제 API 결과](../output/ai_baseline/remaining_goal_speed_final_20261011/api_validation/api_validation.json)
- [32개 영상 정답 입력표](../output/ai_baseline/remaining_goal_speed_final_20261011/reference_ledger.json)
- [정답 미확보 상태의 평가 기록](../output/ai_baseline/remaining_goal_speed_final_20261011/reference_evaluation.json)

재현할 때 출력 폴더·JSON 출력 경로는 기존과 다른 이름으로 지정한다.

```powershell
python -m unittest discover -s tests
node --test frontend/tests/sse.test.js frontend/tests/goalResult.test.js frontend/tests/measurementConfig.test.js
python tools/validate_goal_pipeline.py --output output/ai_baseline/goal_plane_new
python tools/run_ai_baseline.py --output-dir output/ai_baseline/remaining_new
python tools/audit_ai_geometry.py output/ai_baseline/remaining_new
python tools/compare_remaining.py output/ai_baseline/remaining_new
python tools/verify_goal_api.py --output output/ai_baseline/remaining_new/api_validation
```

## 다음 실제 검증

한 촬영 설정에서 먼저 검증한다. 같은 설정의 보정, 일정한 실제 시간 정보, 골대 실측, 독립 속도·골라인 정답을 확보하고 골대 안·밖과 경계를 포함한 최소 10개 슈팅을 분석한다. 비교 가능한 구간을 맞춰 오차와 보류 비율을 확인한 뒤 채점 기준과 허용 오차를 지도교수와 확정한다. 이후 슬로모션·다른 렌즈·줌 설정으로 확대한다.

이 자료가 확보되기 전까지 남은 일은 “코드를 더 수정하면 정확도가 확정되는 단계”가 아니라, 계산의 실제 오차와 사용 가능한 촬영 조건을 확인하는 단계다. 최종 보고서·발표·시연 영상 등 창의학기제 산출물은 별도로 남아 있다.

2026-10-12 후속 변경: 사용자 요청으로 정상 속도·줌 없음 가정의 [설정 없는 자동 추정 모드](ai_automatic_mode.md)를 기본값으로 추가했다. 이 문서의 32개 null 속도·점수 기록은 이전 보정 조건 기반 모드의 검증 이력이다.
