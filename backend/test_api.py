"""백엔드 API 동작 확인.

업로드 검증, 분석 진행률 구독, 보정 상태 조회, 원본 영상 반환, 결과 응답 형태를 점검한다.

분석 라이브러리가 설치되지 않은 환경에서도 API 동작을 확인할 수 있도록,
무거운 라이브러리는 가짜 모듈로 끼워 넣고 main.py의 실제 코드를 불러온다.
영상 메타데이터는 가짜 VideoCapture로 원하는 값을 주입한다.
"""
import sys
import os
import time
import types
import asyncio
import json


def install_stubs(video_props):
    """main.py가 import되도록 외부 라이브러리를 가짜로 채운다."""
    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_COUNT = 7
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4

    class FakeCapture:
        def __init__(self, path):
            self.props = video_props
        def isOpened(self):
            return self.props.get("opened", True)
        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: self.props.get("fps", 30.0),
                cv2.CAP_PROP_FRAME_COUNT: self.props.get("frames", 300.0),
                cv2.CAP_PROP_FRAME_WIDTH: self.props.get("width", 1920.0),
                cv2.CAP_PROP_FRAME_HEIGHT: self.props.get("height", 1080.0),
            }.get(prop, 0.0)
        def release(self):
            pass

    cv2.VideoCapture = FakeCapture
    sys.modules["cv2"] = cv2

    for name in ("numpy", "ultralytics", "scipy", "scipy.optimize"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["ultralytics"].YOLO = object
    sys.modules["scipy.optimize"].least_squares = None

    # analyze_shot은 가짜로 두지 않는다. 위 스텁만으로 import가 되고,
    # 실제 코드가 만드는 응답 형태를 그대로 확인할 수 있다.


def load_main(video_props):
    for name in ("main", "cv2", "analyze_shot"):
        sys.modules.pop(name, None)
    install_stubs(video_props)
    import main
    return main


def check_upload_validation():
    print("=" * 62)
    print("업로드 검증 경계값 판정")
    print("=" * 62)

    cases = [
        ("정상 (10초, 1920x1080)",        dict(fps=30, frames=300, width=1920, height=1080), None),
        ("세로 영상 (10초, 1080x1920)",   dict(fps=30, frames=300, width=1080, height=1920), None),
        ("경계: 정확히 1초",              dict(fps=30, frames=30,  width=1920, height=1080), None),
        ("경계: 정확히 30초",             dict(fps=30, frames=900, width=1920, height=1080), None),
        ("0.9초 → 길이 미달",             dict(fps=30, frames=27,  width=1920, height=1080), (413, "VIDEO_TOO_SHORT")),
        ("31초 → 길이 초과",              dict(fps=30, frames=930, width=1920, height=1080), (413, "VIDEO_TOO_LONG")),
        ("320x240 → 해상도 미달",         dict(fps=30, frames=300, width=320,  height=240),  (415, "RESOLUTION_TOO_LOW")),
        ("세로 360x640 → 짧은 변 미달",   dict(fps=30, frames=300, width=360,  height=640),  (415, "RESOLUTION_TOO_LOW")),
        ("파일 열기 실패 → 손상",         dict(opened=False),                                 (415, "UNREADABLE_VIDEO")),
        ("fps 0 → 정보 없음",             dict(fps=0, frames=300, width=1920, height=1080),  (415, "UNREADABLE_VIDEO")),
    ]

    passed = 0
    for name, props, expected in cases:
        main = load_main(props)
        got = main.check_video_constraints("dummy.mp4")
        got = None if got is None else (got[0], got[1])
        ok = got == expected
        passed += ok
        shown = "통과" if got is None else f"{got[0]} {got[1]}"
        print(f"  {'OK  ' if ok else '실패'} {name:28} → {shown}")
    print(f"\n  {passed}/{len(cases)} 통과")
    return passed == len(cases)


def check_event_stream():
    print()
    print("=" * 62)
    print("분석 진행률 구독 및 재접속 동작")
    print("=" * 62)

    main = load_main({})

    async def collect(analysis_id):
        response = await main.analysis_events(analysis_id)
        if not hasattr(response, "body_iterator"):
            return [f"HTTP {response.status_code}"]
        events = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            payload = json.loads(chunk.removeprefix("data: ").strip())
            events.append(payload["type"] if payload["type"] != "progress"
                          else f"progress:{payload['value']}")
        return events

    async def scenario_running():
        main.jobs["run-1"] = {"status": "RUNNING", "progress": 0, "result": None, "error": None}

        async def analysis():
            for p in (15, 50, 90):
                await asyncio.sleep(0.25)
                main.jobs["run-1"]["progress"] = p
            await asyncio.sleep(0.25)
            main.jobs["run-1"]["result"] = {"status": "VALID", "speed_kmh": 93.7}
            main.jobs["run-1"]["status"] = "VALID"
            main.jobs["run-1"]["progress"] = 100

        asyncio.create_task(analysis())
        return await collect("run-1")

    async def scenario_reconnect():
        main.jobs["done-1"] = {"status": "VALID", "progress": 100,
                               "result": {"status": "VALID", "speed_kmh": 93.7}, "error": None}
        return await collect("done-1")

    async def scenario_error():
        main.jobs["err-1"] = {"status": "ERROR", "progress": 40,
                              "result": None, "error": "모델 로드 실패"}
        return await collect("err-1")

    async def scenario_missing():
        return await collect("없는-아이디")

    async def run_all():
        ok = True

        events = await scenario_running()
        meaningful = [e for e in events if e != "keepalive"]
        print("  [분석 진행 중 구독]")
        print(f"    수신 이벤트: {meaningful}")
        print(f"    keepalive {events.count('keepalive')}개로 연결 유지")
        ok &= meaningful[-1] == "complete"

        events = await scenario_reconnect()
        print("  [완료 후 재접속 — 새로고침 상황]")
        print(f"    수신 이벤트: {events}")
        print("    지난 진행률 없이 결과만 즉시 수신:", events == ["complete"])
        ok &= events == ["complete"]

        events = await scenario_error()
        print("  [분석 중 서버 예외]")
        print(f"    수신 이벤트: {events}")
        ok &= events == ["failed"]

        events = await scenario_missing()
        print("  [존재하지 않는 분석 id]")
        print(f"    응답: {events}")
        ok &= events == ["HTTP 404"]
        return ok

    return asyncio.run(run_all())


def check_calibration():
    print()
    print("=" * 62)
    print("보정 상태 조회 응답")
    print("=" * 62)

    main = load_main({})
    ok = True

    main.calibrations["running"] = {"status": "CALIBRATING", "progress": 40,
                                    "result": None, "failure_reason": None}
    main.calibrations["done"] = {"status": "VALID", "progress": 100, "failure_reason": None,
                                 "result": {"ok": True, "used_frames": 23, "rms_error": 0.4182,
                                            "camera_matrix": [[1452.31, 0, 963.52],
                                                              [0, 1449.87, 539.11], [0, 0, 1]],
                                            "dist_coeffs": [[0.1, -0.2, 0, 0, 0.05]],
                                            "img_width": 1920, "img_height": 1080}}
    main.calibrations["no_board"] = {"status": "FAILED", "progress": 100,
                                     "failure_reason": "CHESSBOARD_NOT_FOUND",
                                     "result": {"ok": False, "used_frames": 0}}
    main.calibrations["few"] = {"status": "FAILED", "progress": 100,
                                "failure_reason": "NOT_ENOUGH_FRAMES",
                                "result": {"ok": False, "used_frames": 3}}

    cases = [
        ("보정 진행 중", "running", 202),
        ("보정 성공", "done", 200),
        ("체스보드 미검출", "no_board", 200),
        ("체스보드 프레임 부족", "few", 200),
        ("존재하지 않는 보정 id", "없는-아이디", 404),
    ]
    for name, key, expected in cases:
        r = main.get_calibration(key)
        body = json.loads(r.body.decode())
        ok &= r.status_code == expected
        print(f"  {'OK  ' if r.status_code == expected else '실패'} {name}")
        print(f"       {r.status_code} {json.dumps(body, ensure_ascii=False)}")
    return ok


def check_video_endpoint():
    print()
    print("=" * 62)
    print("원본 영상 반환 엔드포인트")
    print("=" * 62)

    main = load_main({})
    ok = True

    mp4 = os.path.join(main.UPLOAD_DIR, "_test_clip.mp4")
    mov = os.path.join(main.UPLOAD_DIR, "_test_clip.mov")
    for path in (mp4, mov):
        with open(path, "wb") as f:
            f.write(b"X" * 2048)

    base = {"status": "VALID", "progress": 100, "result": {}, "error": None}
    main.jobs["mp4"] = {**base, "video_path": mp4}
    main.jobs["mov"] = {**base, "video_path": mov}
    main.jobs["cleaned"] = {**base, "video_path": os.path.join(main.UPLOAD_DIR, "_removed.mp4")}

    try:
        r = main.analysis_video("mp4")
        ok &= getattr(r, "media_type", None) == "video/mp4"
        print(f"  OK   mp4 정상 반환 → 200 {r.media_type}")

        r = main.analysis_video("mov")
        ok &= getattr(r, "media_type", None) == "video/quicktime"
        print(f"  OK   mov 정상 반환 → 200 {r.media_type}")

        r = main.analysis_video("cleaned")
        body = json.loads(r.body.decode())
        ok &= r.status_code == 404 and body["error_code"] == "VIDEO_NOT_FOUND"
        print(f"  OK   보관 기간 경과로 파일 없음 → {r.status_code} {body['error_code']}")

        r = main.analysis_video("없는-아이디")
        body = json.loads(r.body.decode())
        ok &= r.status_code == 404 and body["error_code"] == "JOB_NOT_FOUND"
        print(f"  OK   존재하지 않는 분석 id → {r.status_code} {body['error_code']}")
    finally:
        for path in (mp4, mov):
            if os.path.isfile(path):
                os.remove(path)
    return ok


def check_result_schema():
    print()
    print("=" * 62)
    print("분석 결과 응답 형태")
    print("=" * 62)

    load_main({})
    from analyze_shot import ProShootingAnalyzer

    traj = [{"x": 150.0, "y": 540.0, "frame": 0},
            {"x": 240.0, "y": 440.0, "frame": 3},
            {"x": 350.0, "y": 355.0, "frame": 6}]

    valid = ProShootingAnalyzer._build_result(
        None, "/uploads/20261027_shot_result.jpg", time.time() - 3.2, "VALID",
        score=78, speed_kmh=93.74, trajectory=traj, confidence=0.823,
        comment="아마추어를 뛰어넘은 멋진 슈팅!", fps=29.97,
        frame_size={"width": 1920, "height": 1080})

    failed = ProShootingAnalyzer._build_result(
        None, "/uploads/20261027_x_result.jpg", time.time() - 1.1, "ANALYSIS_FAILED",
        failure_reason="BALL_NOT_DETECTED", comment="공을 찾지 못했습니다.",
        fps=30.0, frame_size={"width": 1080, "height": 1920})

    print("  [정상 분석]")
    print(json.dumps(valid, ensure_ascii=False, indent=2))
    print()
    print("  [분석 실패 — 세로 영상]")
    print(json.dumps(failed, ensure_ascii=False, indent=2))

    ok = all(k in valid for k in ("fps", "frame_size", "trajectory"))
    ok &= all("frame" in p for p in valid["trajectory"])
    ok &= "failure_reason" in failed and "failure_reason" not in valid
    print()
    print("  프레임 번호·fps·frame_size 포함 여부:", ok)
    return ok


if __name__ == "__main__":
    results = [
        check_upload_validation(),
        check_event_stream(),
        check_calibration(),
        check_video_endpoint(),
        check_result_schema(),
    ]
    print()
    print("전체 결과:", "모두 통과" if all(results) else "실패 있음")
