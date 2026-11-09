const reasons = {
    GOAL_NOT_FOUND: '골대 영역을 찾지 못했습니다.',
    INVALID_GOAL_GEOMETRY: '골대 영역의 좌표를 사용할 수 없습니다.',
    GOAL_CONFIDENCE_LOW: '골대 영역 검출이 불확실합니다.',
    CAMERA_MOVING: '카메라 움직임이 감지돼 처음 검출한 골대 영역을 사용할 수 없습니다.',
    CAMERA_MOTION_UNVERIFIED: '관측 구간의 카메라 고정 여부를 확인하지 못했습니다.',
    KICK_NOT_DETECTED: '슈팅 시작을 확인하지 못했습니다.',
    INVALID_OBSERVATION_SEQUENCE: '공 관측의 프레임 순서를 확인하지 못했습니다.',
    INSUFFICIENT_GOAL_OBSERVATIONS: '판정에 필요한 공 관측이 부족합니다.',
    ENTRY_CONTINUITY_UNVERIFIED: '영역 밖에서 안으로 들어오는 연속 관측을 확인하지 못했습니다. 공 가림이나 추적 중단이 있을 수 있습니다.',
    BORDER_OR_BRIEF_OVERLAP: '경계에 걸치거나 짧게 겹쳐 보여 진입 후보로 판단하기 어렵습니다.',
}

export function goalPresentation(assessment) {
    if (assessment?.status === 'CANDIDATE_OBSERVED') return {
        title: '영상 골대 영역 진입 후보',
        description: '공 전체의 검출 박스가 영상의 골대 영역 안에 연속으로 관측됐습니다. 실제 골라인 통과는 확인하지 못했습니다.',
    }
    if (assessment?.status === 'NO_CANDIDATE_OBSERVED') return {
        title: '진입 후보 관측 없음',
        description: '저장된 공 관측에서 영역 진입 후보가 없었습니다. 골인이 아니라고 확정한 결과는 아닙니다.',
    }
    return { title: '골대 진입 판정 보류', description: reasons[assessment?.reason] || '골대 진입을 판단할 관측 근거가 부족합니다.' }
}

export function withResultImages(data, baseUrl = 'http://localhost:8000') {
    const imageUrl = name => name ? `${baseUrl}/results/${encodeURIComponent(name)}` : undefined
    return { ...data, image_url: imageUrl(data.trajectory_image), goal_entry_image_url: imageUrl(data.goal_entry_image) }
}

export function physicalGoalPresentation(assessment) {
    if (assessment?.status === 'ESTIMATED_GOAL') return {
        title: '골라인 통과 추정',
        description: '측정한 골대 면과 연속 공 관측으로 공 전체가 골대 안에서 통과한 것으로 추정했습니다. 단일 카메라의 깊이 오차와 프레임 사이 움직임 가정이 포함되며 실측 정확도는 미검증입니다.',
    }
    if (assessment?.status === 'ESTIMATED_OUTSIDE') return {
        title: '골대 밖 통과 추정',
        description: '공이 골대 면을 기둥 옆 또는 크로스바 위로 통과한 것으로 추정했습니다. 실측 정확도는 미검증입니다.',
    }
    const detail = {
        GOAL_PLANE_REQUIRED: '골대 실제 치수와 골라인 경계면의 네 모서리 측정값이 필요합니다.',
        GOAL_CALIBRATION_REQUIRED: '같은 카메라·렌즈·줌 설정의 보정을 확인해야 합니다.',
        GOAL_POSE_AMBIGUOUS: '골대 면의 방향이 불확실합니다.',
        GOAL_POSE_SENSITIVITY_UNSTABLE: '모서리 오차에 따라 골대 면 계산이 크게 달라집니다.',
        GOAL_BOUNDARY_OR_DEPTH_UNCERTAIN: '골대 경계 또는 공 깊이의 오차 때문에 전체 통과를 판단하기 어렵습니다.',
        FULL_BALL_CROSSING_NOT_OBSERVED: '공 전체가 골라인 앞에서 뒤로 통과하는 연속 관측이 부족합니다.',
        INVALID_GOAL_PLANE_GEOMETRY: '골대 모서리나 기준 프레임이 영상과 맞지 않습니다.',
    }
    return { title: '골인 여부 미확정', description: detail[assessment?.reason] || reasons[assessment?.reason] || '골라인 전체 통과를 확인할 근거가 부족합니다.' }
}

export function scoreBasisLabel(score) {
    return {
        MEASURED_PLANE_OUTER_DISTANCE: '속도 40% · 골대 외곽 위치 60%',
        MEASURED_PLANE_OUTSIDE: '골대 밖 통과 추정 기준',
        USER_DEFINED_TARGET: '사용자 지정 목표 위치 기준',
        USER_DEFINED_OUTSIDE: '사용자 지정 골대 밖 기준',
        SPEED_ONLY: '이전 방식 · 속도 기준 일부 항목',
        IMAGE_ACCURACY_ONLY: '이전 방식 · 골대 중심 기준 일부 항목',
        SPEED_AND_IMAGE_ACCURACY: '이전 방식 · 속도 40% · 골대 중심 60%',
        OUTSIDE_ESTIMATE: '이전 방식 · 영상의 골대 밖 추정 기준',
    }[score?.basis] || '선택한 채점 기준'
}

export function scoreWithheldReason(score) {
    return {
        GOAL_PLANE_REQUIRED: '채점하려면 카메라 보정과 골대 실측 정보가 필요합니다.',
        PHYSICAL_GOAL_ESTIMATE_REQUIRED: '골라인 통과 여부를 판단할 근거가 부족합니다.',
        GOAL_CROSSING_EVIDENCE_REQUIRED: '골라인 통과 위치나 근거 프레임을 확인하지 못했습니다.',
        SPEED_ESTIMATE_REQUIRED: '종합 점수에 필요한 속도를 계산하지 못했습니다.',
        SPEED_BASIS_UNVERIFIED: '종합 점수에는 촬영 시간과 카메라 보정을 확인한 속도가 필요합니다.',
        BALL_RADIUS_ASSUMPTION_MISMATCH: '골라인과 속도 계산에 사용한 공 크기가 다릅니다.',
    }[score?.reason] || '종합 점수에 필요한 근거가 부족합니다.'
}

export function automaticInputLabel(data) {
    const context = data.measurement_context
    const time = context?.timing?.mode === 'slow_motion' ? '입력한 슬로모션 조건' :
        context?.timing?.mode === 'unknown' ? '촬영 시간 미확인' : '정상 재생 시간 가정'
    const camera = data.camera_model?.intrinsics_calibrated ? '카메라 보정 사용' : '줌 없음·기본 화각 가정'
    return `${time} · ${camera}`
}
