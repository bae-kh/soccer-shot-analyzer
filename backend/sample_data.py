"""연동용 예시 작업.

영상을 올리지 않고도 결과 화면을 만들어볼 수 있도록, 고정된 id의 작업을
서버 메모리에 미리 넣어둔다. SAMPLE_DATA=1 일 때만 동작한다.

개발 편의용이므로 연동이 끝나면 이 파일과 main.py의 호출부를 함께 지우면 된다.
예시 작업에는 실제 영상 파일이 없어 원본 영상과 결과 이미지 조회는 404가 난다.
"""

FPS = 29.97
LANDSCAPE = {"width": 1920, "height": 1080}
PORTRAIT = {"width": 1080, "height": 1920}


def _arc(points, start, end, lift, frame_step=1):
    """차올린 공이 그리는 모양의 좌표를 만든다. 위로 솟았다 내려오는 곡선."""
    x0, y0 = start
    x1, y1 = end
    out = []
    for i in range(points):
        t = i / (points - 1)
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t - lift * (4 * t * (1 - t))
        out.append({"x": round(x, 1), "y": round(y, 1), "frame": i * frame_step})
    return out


def _result(status, score=0, speed_kmh=0.0, trajectory=None, confidence=0.0,
            comment="", frame_size=None, failure_reason=None, image="sample_result.jpg",
            processing_time=0.0, goal_status="UNDETERMINED", score_status="NOT_EVALUATED"):
    data = {
        "status": status,
        "score": score,
        "score_status": score_status,
        "speed_kmh": speed_kmh,
        "goal": goal_status == "GOAL",
        "goal_status": goal_status,
        "trajectory": trajectory or [],
        "fps": FPS,
        "frame_size": frame_size or LANDSCAPE,
        "confidence": confidence,
        "processing_time_sec": processing_time,
        "trajectory_image": image,
        "comment": comment,
    }
    if failure_reason:
        data["failure_reason"] = failure_reason
    return data


def _done(result):
    return {"status": result["status"], "progress": 100,
            "result": result, "error": None, "video_path": None}


def demo_outcomes():
    """진행 시연에서 고를 수 있는 결과. 키는 요청의 outcome 값이다."""
    return {
        "valid": _result(
            "VALID", score=78, speed_kmh=93.7,
            trajectory=_arc(25, (940, 905), (1486, 452), lift=150),
            confidence=0.82, comment="아마추어를 뛰어넘은 멋진 슈팅! 🔥",
            processing_time=20.0, image="sample_valid_result.jpg",
        ),
        "no-ball": _result(
            "ANALYSIS_FAILED", failure_reason="BALL_NOT_DETECTED",
            comment="공을 찾지 못했습니다.", processing_time=20.0,
        ),
        "low-confidence": _result(
            "ANALYSIS_FAILED", failure_reason="LOW_CONFIDENCE",
            trajectory=_arc(9, (952, 898), (1180, 690), lift=60, frame_step=3),
            confidence=0.41, comment="탐지가 불안정하여 분석 결과를 신뢰하기 어렵습니다.",
            processing_time=20.0,
        ),
        "track-short": _result(
            "ANALYSIS_FAILED", failure_reason="TRACK_TOO_SHORT",
            trajectory=_arc(3, (948, 902), (1010, 846), lift=20),
            confidence=0.33, comment="오류: 추적된 공 궤적이 너무 짧습니다.",
            processing_time=20.0,
        ),
        # 분석을 시작조차 못 한 경우. 구독하면 complete가 아니라 failed로 온다
        "error": None,
    }


def demo_steps():
    """진행률이 지나갈 구간. 연동 가이드에 적은 분석 단계와 같은 순서다."""
    return [
        (0, 15, "정지 공 탐색"),
        (15, 90, "프레임별 탐지·추적"),
        (90, 100, "궤적 계산"),
    ]


def seed_samples(jobs, calibrations):
    """분석 작업과 보정 작업 저장소에 예시를 채운다."""

    # 정상 분석. 궤적 25점, 가로 영상
    jobs["sample-valid"] = _done(_result(
        "VALID", score=78, speed_kmh=93.7,
        trajectory=_arc(25, (940, 905), (1486, 452), lift=150),
        confidence=0.82, comment="아마추어를 뛰어넘은 멋진 슈팅! 🔥",
        processing_time=34.2, image="sample_valid_result.jpg",
        goal_status="GOAL", score_status="EVALUATED",
    ))

    # 분석은 성공했으나 골대 면 정보가 없어 골인 판정과 채점을 보류한 경우.
    # 실제 분석에서 지금 가장 흔하게 나오는 결과다.
    jobs["sample-undetermined"] = _done(_result(
        "VALID", score=0, speed_kmh=81.8,
        trajectory=_arc(19, (930, 880), (1402, 498), lift=130),
        confidence=0.77, comment="초기 비행 구간의 속도 추정입니다.",
        processing_time=26.4, image="sample_undetermined_result.jpg",
        goal_status="UNDETERMINED", score_status="NOT_EVALUATED",
    ))

    # 세로 영상. 좌표 환산을 확인할 때 쓴다
    jobs["sample-portrait"] = _done(_result(
        "VALID", score=0, speed_kmh=79.4,
        trajectory=_arc(21, (540, 1620), (812, 889), lift=180),
        confidence=0.74, comment="골대를 벗어났습니다. 방향을 조금만 더 안쪽으로! 💨",
        frame_size=PORTRAIT, processing_time=28.9, image="sample_portrait_result.jpg",
        goal_status="OUTSIDE", score_status="EVALUATED",
    ))

    # 공을 끝내 찾지 못한 경우. 궤적이 비어 있다
    jobs["sample-no-ball"] = _done(_result(
        "ANALYSIS_FAILED", failure_reason="BALL_NOT_DETECTED",
        comment="공을 찾지 못했습니다.", processing_time=6.1,
    ))

    # 추적은 했으나 신뢰도가 기준에 미달한 경우. 궤적은 일부 담긴다
    jobs["sample-low-confidence"] = _done(_result(
        "ANALYSIS_FAILED", failure_reason="LOW_CONFIDENCE",
        trajectory=_arc(9, (952, 898), (1180, 690), lift=60, frame_step=3),
        confidence=0.41, comment="탐지가 불안정하여 분석 결과를 신뢰하기 어렵습니다.",
        processing_time=22.5,
    ))

    # 추적 구간이 너무 짧아 계산을 못 한 경우
    jobs["sample-track-short"] = _done(_result(
        "ANALYSIS_FAILED", failure_reason="TRACK_TOO_SHORT",
        trajectory=_arc(3, (948, 902), (1010, 846), lift=20),
        confidence=0.33, comment="오류: 추적된 공 궤적이 너무 짧습니다.",
        processing_time=11.8,
    ))

    # 보정 성공
    calibrations["sample-calibration-ok"] = {
        "status": "VALID", "progress": 100, "failure_reason": None,
        "result": {
            "ok": True, "used_frames": 23, "rms_error": 0.4182,
            "camera_matrix": [[1452.31, 0.0, 963.52], [0.0, 1449.87, 539.11], [0.0, 0.0, 1.0]],
            "dist_coeffs": [[0.108, -0.231, 0.0, 0.0, 0.052]],
            "img_width": 1920, "img_height": 1080,
        },
    }

    # 보정 실패. 체스보드를 한 번도 찾지 못한 경우
    calibrations["sample-calibration-failed"] = {
        "status": "FAILED", "progress": 100,
        "failure_reason": "CHESSBOARD_NOT_FOUND",
        "result": {"ok": False, "failure_reason": "CHESSBOARD_NOT_FOUND", "used_frames": 0},
    }

    return sorted(jobs), sorted(calibrations)
