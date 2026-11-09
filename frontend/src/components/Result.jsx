import { goalPresentation, physicalGoalPresentation, scoreBasisLabel, scoreWithheldReason, automaticInputLabel } from '../lib/goalResult'

export default function Result({ data, onReset }) {
    const hasSpeed = typeof data.speed === 'number' && Number.isFinite(data.speed) && data.measurement_status !== 'UNAVAILABLE'
    const hasScore = typeof data.score === 'number' && Number.isFinite(data.score)
    const automatic = data.analysis_mode === 'automatic'
    const goal = goalPresentation(data.goal_entry)
    const physical = physicalGoalPresentation(data.physical_goal)
    const unavailableMessages = {
        CAMERA_MOVING: '카메라 움직임이 감지됐습니다. 카메라를 고정하고 다시 촬영해 주세요.',
        BALL_SIZE_INCONSISTENT: '공 크기가 불안정하게 관측돼 속도를 계산하지 못했습니다. 공이 선명하게 보이도록 촬영해 주세요.',
        INSUFFICIENT_FLIGHT_OBSERVATIONS: '날아가는 공을 충분히 관측하지 못했습니다. 공과 비행 구간이 잘 보이도록 촬영해 주세요.',
        FLIGHT_INTERVAL_TOO_SHORT: '속도를 계산하기에 공이 관측된 구간이 너무 짧습니다.',
        REPROJECTION_MISMATCH: '관측한 공 움직임과 계산 결과의 차이가 커 속도를 표시하지 않았습니다.',
        SPEED_UNCERTAINTY_HIGH: '속도 계산의 불확실성이 커 값을 표시하지 않았습니다.',
        CALIBRATION_INVALID: '카메라 보정 정보를 사용할 수 없습니다. 같은 카메라로 다시 보정해 주세요.',
        BALL_NOT_FOUND: '영상에서 슈팅 공을 찾지 못했습니다.',
        CAPTURE_TIMING_UNVERIFIED: '실제 촬영 시간이 미확인입니다. 정상 속도 원본인지 확인하거나 일정한 슬로모션 배율을 입력해 주세요.',
        CAMERA_PROFILE_REQUIRED: '같은 카메라·렌즈·줌 설정의 보정을 선택하면 속도를 계산할 수 있습니다.',
        CAMERA_PROFILE_NOT_FOUND: '선택한 카메라 보정을 찾지 못했습니다. 보정 목록을 다시 확인해 주세요.',
        CAMERA_PROFILE_ASPECT_MISMATCH: '보정 영상과 화면 비율이 다릅니다. 같은 촬영 모드의 크롭하지 않은 영상을 사용해 주세요.',
        CAMERA_SETTINGS_UNCONFIRMED: '선택한 보정과 카메라·렌즈·줌·촬영 모드가 같은지 확인해 주세요.',
        UNCROPPED_IMAGE_UNCONFIRMED: '크롭하지 않은 영상인지 확인해 주세요.',
    }
    const comment = automatic ? data.comment : data.measurement_status === 'UNAVAILABLE'
        ? unavailableMessages[data.failure_reason] || '신뢰할 수 있는 속도를 계산하지 못했습니다. 촬영 조건과 공 관측 상태를 확인해 주세요.'
        : data.comment
    return (
        <div className="flex flex-col items-center">
            <h2 className="text-3xl font-extrabold text-emerald-400 drop-shadow-sm tracking-wide mb-6 uppercase">분석 결과</h2>
            {automatic && <p className="text-xs text-slate-400 mb-4">{automaticInputLabel(data)} · 실측 정확도 미검증</p>}

            <div className="relative w-full aspect-video rounded-xl overflow-hidden shadow-[0_0_20px_rgba(16,185,129,0.2)] mb-6 bg-slate-900 border border-slate-700/50 group">
                <img
                    src={data.image_url}
                    alt="Trajectory"
                    className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
                    onError={(e) => { e.target.src = "https://via.placeholder.com/640x360?text=Image+Load+Error" }}
                />
                <div className="absolute top-3 right-3 bg-emerald-500/80 backdrop-blur-sm text-slate-900 px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider shadow-[0_0_10px_rgba(16,185,129,0.5)]">
                    Speed Trace AI
                </div>
            </div>

            <section className="w-full bg-slate-800/80 border border-amber-400/30 p-5 rounded-xl mb-6" aria-label="골라인 판정과 영상 관측">
                <h3 className="text-xl font-bold text-amber-300 mb-2">{physical.title}</h3>
                <p className="text-slate-200 text-sm">{physical.description}</p>
                {data.physical_goal?.events?.length > 0 && <p className="text-sm text-slate-300 mt-2">
                    골라인 통과 근거 프레임: {data.physical_goal.events[0].before_frame}–{data.physical_goal.events[0].behind_frame}
                </p>}
                <h4 className="font-semibold text-emerald-300 mt-4">{goal.title}</h4>
                <p className="text-slate-400 text-sm mt-2">{goal.description}</p>
                {data.goal_entry?.events?.length > 0 && <p className="text-sm text-slate-300 mt-2">
                    화면 후보 프레임: {data.goal_entry.events[0].first_inside_frame}–{data.goal_entry.events[0].confirmation_frame}
                </p>}
                {data.goal_entry_image_url && <figure className="mt-4">
                    <img src={data.goal_entry_image_url} alt="전체 관측 궤적과 골대 영역 진입 근거" className="w-full rounded-lg" />
                    <figcaption className="text-xs text-slate-400 mt-2">전체 관측 궤적 · 초록: YOLO · 주황: CSRT · 분홍 원: 진입 후보 · 사각형: 검출한 골대 영역{data.physical_goal?.plane && ' · 청록: 측정한 골대 면과 통과 근거'}</figcaption>
                </figure>}
            </section>

            <div className="grid grid-cols-2 gap-4 w-full mb-6 relative">
                <div className="bg-slate-800 border border-slate-700/50 p-5 rounded-xl text-center shadow-inner relative overflow-hidden">
                    <div className="absolute -right-4 -top-4 w-16 h-16 bg-cyan-500/10 rounded-full blur-xl"></div>
                    <p className="text-sm text-slate-400 mb-1 font-light tracking-wider">{data.score_status === 'PROVISIONAL' ? '참고 점수 · 실측 미검증' : '종합 점수'}</p>
                    <div className="text-4xl font-black text-cyan-400 drop-shadow-[0_0_10px_rgba(34,211,238,0.5)]">
                        {hasScore ? data.score : '미평가'}{hasScore && <span className="text-lg text-slate-500 ml-1">점</span>}
                    </div>
                    <p className="text-xs text-slate-400 mt-2">{hasScore ? scoreBasisLabel(data.score_evaluation) : scoreWithheldReason(data.score_evaluation)}</p>
                    {hasScore && data.score_evaluation?.rule_set_id && <p className="text-xs text-slate-400 mt-1">규칙: {data.score_evaluation.rule_set_id}</p>}
                </div>
                <div className="bg-slate-800 border border-slate-700/50 p-5 rounded-xl text-center shadow-inner relative overflow-hidden">
                    <div className="absolute -left-4 -top-4 w-16 h-16 bg-emerald-500/10 rounded-full blur-xl"></div>
                    <p className="text-sm text-slate-400 mb-1 font-light tracking-wider">초기 비행 구간 추정 속도</p>
                    <div className="text-4xl font-black text-emerald-400 drop-shadow-[0_0_12px_rgba(52,211,153,0.6)]">
                        {hasSpeed ? data.speed.toFixed(1) : '계산 불가'}{hasSpeed && <span className="text-lg text-slate-500 ml-1">km/h</span>}
                    </div>
                    {hasSpeed && data.speed_basis === 'ASSUMPTION_BASED' && <p className="text-xs text-slate-400 mt-2">촬영 조건 가정에 따른 추정값</p>}
                </div>
            </div>

            <div className="bg-slate-800/80 border border-emerald-500/20 p-5 rounded-xl w-full text-center mb-8 shadow-[0_0_15px_rgba(0,0,0,0.3)]">
                <p className="text-emerald-300 font-semibold tracking-wide text-lg">"{comment}"</p>
                {hasScore && data.score_evaluation?.comment !== comment && <p className="text-xs text-slate-300 mt-3">{data.score_evaluation?.comment}</p>}
                {!hasSpeed && <p className="text-xs text-amber-300 mt-3">속도: {unavailableMessages[data.failure_reason] || '공의 비행 관측이 부족합니다.'}</p>}
                {data.measurement_context?.problems?.length > 0 && <ul className="text-left text-sm text-slate-300 mt-3 space-y-2">
                    {data.measurement_context.problems.map(problem => <li key={problem}>{unavailableMessages[problem] || problem}</li>)}
                </ul>}
            </div>

            <button
                onClick={onReset}
                className="px-8 py-4 bg-slate-700 text-emerald-400 font-extrabold rounded-lg hover:bg-slate-600 hover:text-emerald-300 border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.1)] hover:shadow-[0_0_25px_rgba(16,185,129,0.25)] transition-all transform hover:-translate-y-1 w-full text-lg uppercase tracking-widest"
            >
                다른 영상 분석하기
            </button>
        </div>
    )
}
