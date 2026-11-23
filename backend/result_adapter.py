"""분석 결과를 API 응답 형식으로 바꾼다.

분석 코드는 AI 파트가 맡고 있어 반환 형태가 계속 바뀐다. 프론트엔드가 쓰는
응답 형식은 6~8주차에 합의하여 고정되어 있으므로, 둘 사이를 여기서 맞춘다.
분석 결과의 모양이 또 바뀌면 이 파일만 고치면 된다.
"""

# 골인 추정 결과. 영상만으로는 확정할 수 없는 경우가 많아 보류가 따로 있다.
GOAL_STATUS_MAP = {
    "ESTIMATED_GOAL": "GOAL",
    "ESTIMATED_OUTSIDE": "OUTSIDE",
}

# 분석 쪽 실패 사유를 명세에 적힌 네 가지로 옮긴다.
# 원래 값은 failure_detail로 함께 내보내므로 정보가 사라지지는 않는다.
FAILURE_REASON_MAP = {
    "BALL_NOT_FOUND": "BALL_NOT_DETECTED",
    "INSUFFICIENT_FLIGHT_OBSERVATIONS": "TRACK_TOO_SHORT",
    "FLIGHT_INTERVAL_TOO_SHORT": "TRACK_TOO_SHORT",
    "SPEED_UNCERTAINTY_HIGH": "LOW_CONFIDENCE",
    "BALL_SIZE_INCONSISTENT": "LOW_CONFIDENCE",
    "REPROJECTION_MISMATCH": "FIT_FAILED",
    "INVALID_DEPTH_ESTIMATE": "FIT_FAILED",
    # 촬영이나 보정 자체의 문제라 분석을 신뢰할 수 없는 경우
    "CAMERA_MOVING": "FIT_FAILED",
    "CALIBRATION_INVALID": "FIT_FAILED",
}




def _flight(raw, key):
    """속도 계산에 쓴 비행 구간의 프레임 번호."""
    value = (raw.get("flight_segment") or {}).get(key)
    return int(value) if value is not None else None


def _goal_area(raw):
    """골대 사각형. 궤적과 같은 원본 영상 좌표계다."""
    area = raw.get("goal_area") or {}
    bbox = area.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (round(float(v), 1) for v in bbox)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _first_event_frame(section, key):
    """판정 근거가 된 프레임. 판정을 못 했으면 사건 자체가 없다."""
    events = (section or {}).get("events") or []
    if not events:
        return None
    value = events[0].get(key)
    return int(value) if value is not None else None


def _trajectory(raw):
    """추적 기록을 좌표와 프레임 번호만 남긴 형태로 줄인다."""
    points = []
    for record in raw.get("trajectory") or []:
        x, y = record.get("x"), record.get("y")
        # 예측만 하고 실제로 보지 못한 프레임은 좌표가 없다
        if x is None or y is None:
            continue
        points.append({"x": round(float(x), 1), "y": round(float(y), 1),
                       "frame": int(record.get("frame_number", 0))})
    return points


def _confidence(raw, points):
    """추적된 구간 중 실제로 관측한 비율. 예측으로 메운 프레임은 빼고 센다."""
    summary = raw.get("tracking_summary") or {}
    observed = summary.get("observed_points")
    total = len(raw.get("trajectory") or [])
    if not observed or not total:
        return 0.0
    return round(min(observed / total, 1.0), 2)


def to_api_result(raw, started_at, elapsed_sec=None):
    """분석 결과를 프론트엔드가 쓰는 11개 필드로 바꾼다.

    raw는 analyze_shot.process_video_with_progress가 돌려준 값이다.
    """
    import time

    metadata = raw.get("video_metadata") or {}
    physical_goal = raw.get("physical_goal") or {}
    score_evaluation = raw.get("score_evaluation") or {}

    points = _trajectory(raw)
    succeeded = raw.get("measurement_status") == "ESTIMATED"
    detail = raw.get("failure_reason")

    goal_status = GOAL_STATUS_MAP.get(physical_goal.get("status"), "UNDETERMINED")
    scored = (raw.get("score_status") or score_evaluation.get("status")) not in (None, "NOT_EVALUATED")

    result = {
        "status": "VALID" if succeeded else "ANALYSIS_FAILED",
        "score": int(score_evaluation.get("value") or raw.get("score") or 0),
        # 점수를 매기지 못한 경우와 실제로 0점인 경우를 화면이 구분할 수 있게 한다
        "score_status": "EVALUATED" if scored else "NOT_EVALUATED",
        # 채점 근거가 일부만 모이면 만점이 100보다 작아진다. 속도를 재지 못하면
        # 속도 몫이 통째로 빠지므로, 만점을 함께 보내지 않으면 낮은 점수로 읽힌다.
        "score_max": round(float(score_evaluation.get("max_points") or 100), 1),
        "score_partial": bool(score_evaluation.get("partial")),
        "speed_kmh": round(float(raw.get("speed") or 0.0), 1),
        # 골인 판정은 추정이며 심판 판정이 아니다. 단정할 수 있을 때만 참으로 둔다
        "goal": goal_status == "GOAL",
        # goal의 false는 골대 밖 통과와 판정 불가를 모두 포함하므로 상태를 함께 준다
        "goal_status": goal_status,
        "trajectory": points,
        "fps": round(float(metadata.get("fps") or 0.0), 2),
        "frame_size": {"width": int(metadata.get("width") or 0),
                       "height": int(metadata.get("height") or 0)},
        "confidence": _confidence(raw, points),
        "processing_time_sec": round(
            elapsed_sec if elapsed_sec is not None else time.time() - started_at, 1),
        "trajectory_image": raw.get("trajectory_image") or "",
        "comment": raw.get("comment") or "",
        # 화면에서 골대를 그리고 궤적을 어디까지 보여줄지 정하는 데 쓴다.
        # 골대를 찾지 못했거나 판정이 서지 않으면 각각 null이 된다.
        # 속도를 잰 비행 구간. 궤적을 어디까지 보여줄지 정할 때 참고한다.
        # 다만 차고 난 직후 0.45초까지만 보는 구간이므로 공이 끝까지 날아간
        # 범위와 다를 수 있다. 끝난 이유를 함께 보내 화면이 판단하게 한다.
        "flight_start_frame": _flight(raw, "start_frame"),
        "flight_end_frame": _flight(raw, "end_frame"),
        "flight_end_reason": (raw.get("flight_segment") or {}).get("reason"),
        "goal_area": _goal_area(raw),
        "goal_entry_frame": _first_event_frame(raw.get("goal_entry"), "confirmation_frame"),
        "goal_line_frame": _first_event_frame(raw.get("physical_goal"), "behind_frame"),
    }

    if not succeeded:
        result["failure_reason"] = FAILURE_REASON_MAP.get(detail, "FIT_FAILED")
        # 분석 쪽 원래 사유. 실패 원인을 단계별로 보려면 이 값이 필요하다
        result["failure_detail"] = detail
    return result
