export function readMeasurementConfig(text) {
    const data = JSON.parse(text)
    if (!data || Array.isArray(data) || typeof data !== 'object'
        || Object.keys(data).some(key => !['goal_plane', 'scoring_rules'].includes(key))) {
        throw new Error('골대 측정값과 채점 규칙이 담긴 설정 파일을 선택해 주세요.')
    }
    if (!data.goal_plane || typeof data.goal_plane !== 'object' || Array.isArray(data.goal_plane)) {
        throw new Error('설정 파일에 골대 측정값이 필요합니다.')
    }
    if (data.scoring_rules != null && (typeof data.scoring_rules !== 'object' || Array.isArray(data.scoring_rules))) {
        throw new Error('채점 규칙의 형식을 확인해 주세요.')
    }
    // Importing a file never confirms measurements or accepts a scoring policy.
    return {
        goal_plane: { ...data.goal_plane, confirmed_by_user: false },
        scoring_rules: data.scoring_rules ? { ...data.scoring_rules, confirmed_by_user: false,
            accept_conditional_estimates: false } : null,
    }
}

export function measurementConfigProblem(options) {
    if (options.goal_plane && !options.goal_plane.confirmed_by_user) return '선택할 영상의 골대 측정값을 확인하거나 설정 파일을 해제해 주세요.'
    if (options.scoring_rules && (!options.scoring_rules.confirmed_by_user || !options.scoring_rules.accept_conditional_estimates)) {
        return '참고 점수의 채점 기준을 확인하거나 설정 파일을 해제해 주세요.'
    }
    return null
}
