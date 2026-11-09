# 4단계: 영상 골대 영역 진입 후보와 판정 보류

2026-10-11. 전체 공 관측으로 진입 후보를 찾고, 근거가 부족하면 보류하는 계산과 결과 화면을 연결했다. **후보는 영상의 검출 사각형에 들어오는 관측이며 실제 골인 판정이 아니다.** 속도와 별도로 동작한다. README는 사용자가 단계 종료 후 갱신을 요청하면 반영한다.

## 구현 방식

- `backend/goal_entry.py`: 전체 `trajectory`의 원본 공 중심·박스를 사용한다. 초기 `fitted_trajectory`, Kalman 예측, 속도, 임의 골대 거리·크기를 진입 근거로 쓰지 않는다.
- 같은 관측 흐름에서 영역 밖의 신뢰 가능한 YOLO 공을 확인하고, 공 박스 전체가 안쪽 여유를 둔 골대 사각형 안에 들어온 YOLO 관측이 두 번 이상 이어지면 후보를 기록한다. 신뢰도 최소 0.12, 경계 여유는 `max(2px, 골대 높이의 1%)`다. 경계 여유와 관측 수는 휴리스틱이며 실제 정답 라벨로 최적화한 값이 아니다.
- 관측 프레임은 연속이어야 한다. 검증된 CSRT 관측은 연속성을 이어줄 수 있지만, 단독으로 영역 밖·안의 근거를 만들 수 없다. 예측·누락·낮은 신뢰도·거부된 관측이 끼면 흐름을 끊는다. 공 중심만 안에 있거나 경계에 걸친 박스, 한 번의 검출, 안에서 처음 나타난 공을 확정 진입으로 처리하지 않는다.
- `backend/analyze_shot.py`: 초기 속도 구간과 별도로 골대 검출 시점부터 추적 종료까지 배경 움직임을 검사한다. 골대 영역을 이후에 따라 움직이도록 갱신하는 기능은 없으므로 이동 감지·확인 불가는 보류한다. 검사 자체는 시작·중간·끝 세 프레임의 배경 흐름 휴리스틱이며, 고정 카메라의 증명이나 전 프레임 이동 검사가 아니다.
- `backend/trajectory_render.py`: 기존 초기 속도 사진에 더해 전체 관측 궤적과 후보 위치를 그린 `_goal_entry.jpg`를 저장한다. 첫 후보의 두 번째 YOLO 프레임을 원본 배경으로 사용하고, 후보가 없으면 마지막 공 관측 프레임을 사용한다. 초록은 YOLO, 주황은 CSRT, 분홍 원은 후보다. 예측점·미래 궤적은 이 사진에 그리지 않는다.
- `frontend/src/components/Result.jsx`, `Upload.jsx`, `frontend/src/lib/goalResult.js`: 후보·후보 관측 없음·보류 사유와 전체 관측 사진을 표시한다. 속도가 null이어도 후보 화면을 표시한다. 새 필드가 없는 이전 응답은 보류로 표시하고 없는 사진을 요청하지 않는다.

## 결과 계약

기존 필드·좌표·재생 시간 기준의 추적 레코드는 유지한다. 프레임 번호는 0부터 시작하고 좌표 원점은 원본 영상 왼쪽 위다.

| 추가 필드 | 의미 |
|---|---|
| `goal_entry.status` | `CANDIDATE_OBSERVED`, `NO_CANDIDATE_OBSERVED`, `UNAVAILABLE` |
| `goal_entry.reason` | 영상 영역 진입, 연속성 부족, 경계·짧은 겹침, 골대 누락, 카메라 이동 등 |
| `goal_entry.physical_goal_status` | 현재 항상 `UNDETERMINED`; 골인·실패 확정 아님 |
| `goal_entry.events` | 밖 관측 프레임, 안 관측·확인 프레임, 연속 관측 프레임 목록, 원본 진입 위치 |
| `entry_image_position_normalized` | 골대 사각형 기준 영상상의 상대 위치. 실제 골대 평면의 좌표·타격점이 아님 |
| `goal_entry.camera_motion` | 전체 추적 범위의 별도 카메라 검사 결과 |
| `goal_entry_image` | 전체 관측 궤적 결과 사진 이름 |
| `score_evaluation` | 규칙 정의 여부·실제 골인 확인 여부, 미평가 사유 |

점수 규칙 구조는 필요한 근거와 미평가 사유를 반환한다. 실제 골인 근거와 합의된 채점 기준이 없어 `rule_set_id: null`, `score: null`, `score_status: NOT_EVALUATED`를 유지한다. 진입 후보를 골인으로 간주하거나 속도에 비례하는 임의 점수를 만들지 않는다. 실제 점수 계산은 아직 보류다.

## 검증 자료

- Python 전체 테스트 84개 통과: 후보, 초기에 선택한 속도 구간 뒤의 관측, 촬영 시간 미확인, 경계·가림·예측·CSRT 단독·약한 검출·좌표 오류·누락 골대·누락 공·다중 후보·입력 불변성·SSE 계약.
- 프론트엔드 Node 테스트 8개, ESLint, 배포 빌드 통과.
- 실제 반환값으로 현재 결과 컴포넌트를 렌더링해 브라우저에서 후보·연속성 보류·후보 관측 없음의 문구와 사진 표시를 확인했다. 업로드 버튼을 통해 수행한 브라우저 전체 흐름 시험은 아니다.
- 실제 YOLO·CSRT로 32개 재실행 결과, 비교 검사, 실제 업로드/SSE/사진 조회 검사는 아래 산출물에 기록한다. 정답 속도·공 위치·골인 라벨 정확도를 측정한 결과가 아니다.

산출물은 `output/ai_baseline/stage4_goal_entry_20261011` 아래에 저장한다.

| 실제 32개 재실행 | 결과 |
|---|---|
| 영상 골대 영역 진입 후보 | 14개 (실촬영 10개, 테스트 4개) |
| 진입 후보 관측 없음 | 실촬영 2번 1개; 골인 실패 확정 아님 |
| 판정 보류 | 17개: 카메라 이동 휴리스틱 9개, 진입 연속성 미확인 4개, 경계·짧은 겹침 4개 |
| 기존 전체 추적 레코드·궤적 | 32/32 동일; 입력·모델 해시도 동일 |
| 속도 판정 | 32/32 기존 보류와 null 유지: 실제 시간 미확인 29개, 초기 카메라 이동 3개 |
| 실제 골인 확정·점수 평가 | 모두 0개; 근거·기준 미확보 |
| 결과·관측·골대 사진과 추적 MP4/JSON | 32개 모두 생성·검사 통과 |
| 새 후보의 관측 근거·좌표 검사 | 14개 모두 YOLO 신뢰도·연속 관측·박스 포함 검사 통과 |
| 실제 multipart 업로드·모델 실행·SSE·사진 GET | 실촬영 1번 통과; 모델 계산 모의 처리 없음 |

골대 판정의 카메라 이동 9개는 초기 속도 구간의 3개와 다른 범위의 검사 결과다. 골대 검출부터 추적 종료까지 검사하므로 실촬영 3·18·20, 테스트 1·3·4·8·9·10에서 보류됐다. 이 이동 판정의 정답 여부도 아직 라벨로 검증하지 않았다.

14개 영상의 첫 후보마다 밖·첫 안·확인 프레임, 총 42개 원본 공 주변 확대를 모아 육안으로 확인했다. 해당 확대에서는 공이 보였고 실촬영 1번처럼 아직 골대 앞에 있는 경우도 확인됐다. 이는 원본 근거 확인이며 공 중심·박스 크기·골인 사건의 정답 라벨 평가나 전체 추적 정확도 검증은 아니다.

- [전체 실행 보고서](../output/ai_baseline/stage4_goal_entry_20261011/report.md)
- [전체 결과 JSON](../output/ai_baseline/stage4_goal_entry_20261011/summary.json)
- [코드·모델·입력 해시와 실행 환경](../output/ai_baseline/stage4_goal_entry_20261011/run_metadata.json)
- [좌표·관측 근거·미디어 검사](../output/ai_baseline/stage4_goal_entry_20261011/geometry_audit.json)
- [이전 추적 비교와 후보 원본 프레임 목록](../output/ai_baseline/stage4_goal_entry_20261011/goal_entry_review.json)
- [실제 업로드·AI·SSE·결과 사진 조회 검사](../output/ai_baseline/stage4_goal_entry_20261011/api_validation/api_validation.json)
- [실제 결과 화면 캡처](../output/ai_baseline/stage4_goal_entry_20261011/review/result_candidate_ui.jpg)
- [14개 후보 원본 공 관측 확대 모음](../output/ai_baseline/stage4_goal_entry_20261011/review/candidate_crops_all.jpg)
- 전체 결과 사진: [실촬영 1/3](../output/ai_baseline/stage4_goal_entry_20261011/review/real_test_videos_page1.jpg), [2/3](../output/ai_baseline/stage4_goal_entry_20261011/review/real_test_videos_page2.jpg), [3/3](../output/ai_baseline/stage4_goal_entry_20261011/review/real_test_videos_page3.jpg), [테스트 1/2](../output/ai_baseline/stage4_goal_entry_20261011/review/test_videos_page1.jpg), [2/2](../output/ai_baseline/stage4_goal_entry_20261011/review/test_videos_page2.jpg).

```powershell
& 'C:\anaconda3\envs\capstone\python.exe' -m unittest discover -s tests
& 'C:\anaconda3\envs\capstone\python.exe' -u tools/run_ai_baseline.py --output-dir output/ai_baseline/stage4_new
& 'C:\anaconda3\envs\capstone\python.exe' tools/audit_ai_geometry.py output/ai_baseline/stage4_new
& 'C:\anaconda3\envs\capstone\python.exe' tools/review_goal_entry.py output/ai_baseline/stage4_new
& 'C:\anaconda3\envs\capstone\python.exe' -u tools/verify_goal_api.py --output output/ai_baseline/stage4_new/api_validation
# frontend 디렉터리
node --test tests/*.test.js
npm run lint
npm run build
# 프로젝트 루트: 현재 Result 컴포넌트로 저장 응답 미리보기 생성
node tools/render_goal_result.mjs output/ai_baseline/stage4_new
```

새 실행에는 새 출력 폴더를 지정해야 한다.

## 남은 정확도 한계와 다음 확인

원근 때문에 공이 골대 앞에 있는 상태에서도 골대 사각형과 겹친다. 실촬영 1번의 후보 사진도 이 한계를 보여주며, 후보 프레임을 실제 골라인 통과 시점으로 사용하면 안 된다. 현재 골대 영역은 기둥·네트 검출의 사각형이고 실제 안쪽 네 모서리나 골라인 평면이 아니다. 검출·CSRT·카메라 이동 검사의 오탐 및 공 가림으로 후보가 잘못 생기거나 보류될 수 있다.

다음 5단계에서는 모은 원본 후보 프레임과 보류 사례를 영상별로 확인하고 정답 공 관측·실제 사건과 비교한다. 실제 골인 확정에는 실제 골대 평면·공의 깊이·공 전체 통과를 확인할 추가 근거가 필요하다. 합의된 채점 기준과 검증된 측정 입력이 있어야 실제 점수 계산도 연결할 수 있다. 현재 기능 실행 성공률·후보 개수를 골인 정확도로 해석하지 않는다.
