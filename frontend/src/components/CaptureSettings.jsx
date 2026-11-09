import { useEffect, useState } from 'react'
import { readMeasurementConfig } from '../lib/measurementConfig'

export default function CaptureSettings({ options, onChange, profileRevision }) {
    const [profiles, setProfiles] = useState([])
    const [profileError, setProfileError] = useState('')
    const [slowMethod, setSlowMethod] = useState('factor')
    const [configError, setConfigError] = useState('')
    useEffect(() => {
        const controller = new AbortController()
        fetch('http://localhost:8000/camera-profiles', { signal: controller.signal })
            .then(response => { if (!response.ok) throw new Error(); return response.json() })
            .then(data => { setProfiles(data.profiles); setProfileError('') })
            .catch(error => { if (error.name !== 'AbortError') setProfileError('보정 목록을 불러오지 못했습니다.') })
        return () => controller.abort()
    }, [profileRevision])
    const update = changes => onChange({ ...options, ...changes })
    const numeric = value => value === '' ? null : Number(value)
    const importConfig = async event => {
        const file = event.target.files?.[0]
        if (!file) return
        try {
            const config = readMeasurementConfig(await file.text())
            update({ ...config, analysis_mode: 'calibrated' })
            setConfigError('')
        } catch (error) {
            update({ goal_plane: null, scoring_rules: null })
            setConfigError(error.message)
        }
        event.target.value = ''
    }
    const style = 'w-full rounded-lg bg-slate-900 border border-slate-600 p-2 mt-1'
    return (
        <div className="mb-5 text-sm text-slate-300">
            <p className="text-emerald-300 mb-2">{options.analysis_mode === 'automatic'
                ? (options.timing_mode === 'normal' && !options.intrinsics_profile_id ? '영상만 올리면 자동 분석 · 정상 속도·줌 없음 기준' : '자동 분석 · 선택한 촬영 조건 기준')
                : '보정 자료를 사용하는 분석 모드'}</p>
            <details className="text-slate-400">
                <summary className="cursor-pointer">분석 설정 변경 (선택)</summary>
                <fieldset className="space-y-3 mt-3">
            <label className="block">분석 방식
                <select className={style} value={options.analysis_mode} onChange={e => update({ analysis_mode: e.target.value,
                    timing_mode: 'normal', slowdown_factor: null, capture_fps: null, one_frame_per_capture: false,
                    intrinsics_profile_id: null, same_camera_settings: false, uncropped: false, goal_plane: null, scoring_rules: null })}>
                    <option value="automatic">자동 추정 (기본) · 보정 없이 분석</option>
                    <option value="calibrated">보정 자료 기반 · 측정 조건 확인</option>
                </select>
            </label>
            <label className="block">촬영 시간 기준
                <select className={style} value={options.timing_mode} onChange={e => update({ timing_mode: e.target.value, slowdown_factor: null, capture_fps: null, one_frame_per_capture: false })}>
                    <option value="unknown">미확인 — 궤적만 분석</option>
                    <option value="normal">정상 속도 원본 · 일정 FPS</option>
                    <option value="slow_motion">슬로모션 · 일정 배율</option>
                </select>
            </label>
            {options.timing_mode === 'slow_motion' && <>
                <label className="block">알고 있는 촬영값
                    <select className={style} value={slowMethod} onChange={e => { setSlowMethod(e.target.value); update({ slowdown_factor: null, capture_fps: null, one_frame_per_capture: false }) }}>
                        <option value="factor">슬로모션 배율</option>
                        <option value="fps">실제 촬영 FPS</option>
                    </select>
                </label>
                <label className="block">{slowMethod === 'factor' ? '배율 (예: 실제 1초가 재생 4초면 4)' : '실제 촬영 FPS (재생 FPS와 다릅니다)'}
                    <input className={style} type="number" min={slowMethod === 'factor' ? 1 : 0.01} step="any"
                        value={(slowMethod === 'factor' ? options.slowdown_factor : options.capture_fps) ?? ''}
                        onChange={e => update({ [slowMethod === 'factor' ? 'slowdown_factor' : 'capture_fps']: numeric(e.target.value) })} />
                </label>
                {slowMethod === 'fps' && <label className="flex gap-2 items-start">
                    <input type="checkbox" checked={options.one_frame_per_capture} onChange={e => update({ one_frame_per_capture: e.target.checked })} />
                    <span>촬영 프레임을 중복·삭제·보간 없이 한 장씩 저장한 영상임을 확인했습니다.</span>
                </label>}
                <p className="text-xs text-slate-400">구간마다 배속이 달라지는 편집 영상은 미확인으로 두세요.</p>
            </>}
            <label className="block">카메라 · 렌즈 · 줌 보정
                <select className={style} value={options.intrinsics_profile_id || ''} onChange={e => update({ intrinsics_profile_id: e.target.value || null, same_camera_settings: false, uncropped: false })}>
                    <option value="">{options.analysis_mode === 'automatic' ? '기본 화각으로 자동 추정' : '미선택 — 속도 표시 보류'}</option>
                    {profiles.map(profile => <option key={profile.id} value={profile.id}>{profile.label} ({profile.resolution.join('×')})</option>)}
                </select>
            </label>
            {profileError && <p role="alert" className="text-amber-300">{profileError}</p>}
            {options.intrinsics_profile_id && <label className="flex gap-2 items-start">
                <input type="checkbox" checked={options.same_camera_settings && options.uncropped} onChange={e => update({ same_camera_settings: e.target.checked, uncropped: e.target.checked })} />
                <span>보정 영상과 카메라·렌즈·줌·촬영 모드가 같고, 크롭 없이 원본 또는 같은 비율로 크기를 바꾼 영상입니다.</span>
            </label>}
            <p className="text-xs text-slate-400">자동 모드는 기본 화각으로 속도를 추정합니다. 보정 자료와 실제 촬영 조건을 입력하면 계산 조건을 더 정확하게 맞출 수 있습니다.</p>
            <details className="border border-slate-600 rounded-lg p-3">
                <summary className="cursor-pointer text-amber-300">골대 실측 · 참고 점수 설정 (선택)</summary>
                <p className="text-xs mt-3">자동 모드는 영상 관측으로 골인 가능성과 참고 점수를 계산합니다. 골라인 면을 측정한 경우 설정 파일을 불러와 별도로 분석할 수 있습니다.</p>
                <label className="block mt-3">측정 설정 파일
                    <input type="file" accept=".json,application/json" className="block w-full mt-2" onChange={importConfig} />
                </label>
                {configError && <p role="alert" className="text-red-300 mt-2">{configError}</p>}
                {options.goal_plane && <>
                    <p className="mt-3">골대 {options.goal_plane.width_m} × {options.goal_plane.height_m} m · 기준 프레임 {options.goal_plane.reference_frame}</p>
                    <label className="flex gap-2 items-start mt-2">
                        <input type="checkbox" checked={options.goal_plane.confirmed_by_user} onChange={e => update({ goal_plane: { ...options.goal_plane, confirmed_by_user: e.target.checked } })} />
                        <span>카메라가 골라인 앞 필드 쪽에 있고, 선택할 영상의 원본 좌표와 실제 골대·공 치수를 측정했으며, 네 모서리가 골라인 뒤쪽 경계의 같은 평면임을 확인했습니다.</span>
                    </label>
                    {options.scoring_rules && <label className="flex gap-2 items-start mt-2">
                        <input type="checkbox" checked={options.scoring_rules.confirmed_by_user && options.scoring_rules.accept_conditional_estimates}
                            onChange={e => update({ scoring_rules: { ...options.scoring_rules, confirmed_by_user: e.target.checked, accept_conditional_estimates: e.target.checked } })} />
                        <span>채점 기준 {options.scoring_rules.id}의 목표·속도 범위·가중치를 확인했고, 실측 정확도 검증 전의 참고 점수로 사용합니다.</span>
                    </label>}
                    {options.scoring_rules && <p className="text-xs mt-2">속도 {options.scoring_rules.speed_min_kmh}–{options.scoring_rules.speed_max_kmh} km/h · 속도/정확도 비중 {options.scoring_rules.speed_weight}/{options.scoring_rules.accuracy_weight} · 목표 위치 {options.scoring_rules.target_normalized?.join(', ')} · 허용 거리 {options.scoring_rules.accuracy_radius_m} m · 골대 밖 통과 {options.scoring_rules.outside_score}점</p>}
                    <button type="button" className="underline mt-3" onClick={() => update({ goal_plane: null, scoring_rules: null })}>측정 설정 해제</button>
                </>}
            </details>
                </fieldset>
            </details>
        </div>
    )
}
