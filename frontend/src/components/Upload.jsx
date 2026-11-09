import { useState, useRef } from 'react'
import CaptureSettings from './CaptureSettings'
import { readJsonEvents } from '../lib/sse'
import { withResultImages } from '../lib/goalResult'
import { measurementConfigProblem } from '../lib/measurementConfig'

export default function Upload({ onUploadStart, onAnalysisComplete, onAnalysisError, onProgress, loading, progress, profileRevision }) {
    const [dragActive, setDragActive] = useState(false)
    const inputRef = useRef(null)
    const [error, setError] = useState('')
    const [options, setOptions] = useState({ analysis_mode: 'automatic', timing_mode: 'normal', slowdown_factor: null, capture_fps: null,
        one_frame_per_capture: false, intrinsics_profile_id: null, same_camera_settings: false, uncropped: false })

    const handleDrag = (e) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true)
        } else if (e.type === "dragleave") {
            setDragActive(false)
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0])
        }
    }

    const handleChange = (e) => {
        e.preventDefault()
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0])
        }
    }

    const handleFile = async (file) => {
        if (loading) return
        const problem = measurementConfigProblem(options)
        if (problem) { setError(problem); return }
        setError('')
        onUploadStart() // 로딩 시작 및 progress 0

        // One-Step Analyze Stream
        const formData = new FormData()
        formData.append("file", file)
        formData.append('capture_options', JSON.stringify(options))

        try {
            const response = await fetch("http://localhost:8000/analyze", {
                method: "POST",
                body: formData,
            })

            if (!response.ok) {
                const detail = await response.json()
                throw new Error(typeof detail.detail === 'string' ? detail.detail : '촬영 조건 입력을 확인해 주세요.')
            }
            let receivedFinal = false
            for await (const data of readJsonEvents(response.body)) {
                if (data.type === 'progress') onProgress(data.value)
                else if (data.type === 'result') {
                    onAnalysisComplete(withResultImages(data.data))
                    receivedFinal = true
                } else if (data.type === 'error') throw new Error(data.message)
            }
            if (!receivedFinal) throw new Error('분석 결과를 받지 못했습니다.')

        } catch (error) {
            console.error(error)
            setError(error.message || '서버 연결 오류가 발생했습니다.')
            onAnalysisError()
        }

    }

    const onButtonClick = () => {
        inputRef.current.click()
    }

    return (
        <div className="w-full">
            {!loading && <CaptureSettings options={options} onChange={setOptions} profileRevision={profileRevision} />}
            {error && <p role="alert" className="text-red-300 mb-3">{error}</p>}
            <div
                className={`flex flex-col items-center justify-center p-10 border-2 border-dashed rounded-xl transition-all duration-300 ${dragActive ? "border-emerald-400 bg-emerald-900/20 shadow-[0_0_20px_rgba(52,211,153,0.2)]" : "border-slate-600 bg-slate-900/50 hover:bg-slate-800"
                    }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    ref={inputRef}
                    type="file"
                    className="hidden"
                    accept="video/*"
                    onChange={handleChange}
                    disabled={loading}
                />

                {loading ? (
                    <div className="flex flex-col items-center w-full py-4">
                        <div className="w-full bg-slate-700 rounded-full h-3 mb-4 overflow-hidden shadow-inner">
                            <div
                                className="bg-gradient-to-r from-cyan-400 to-emerald-400 h-3 rounded-full transition-all duration-300 ease-out shadow-[0_0_10px_rgba(52,211,153,0.8)]"
                                style={{ width: `${progress}%` }}
                            ></div>
                        </div>
                        <p className="text-xl font-bold text-emerald-400 drop-shadow-md">{progress}% 분석 중...</p>
                        <p className="text-sm text-slate-400 mt-2 font-light tracking-wide">AI가 궤적을 추적하고 있습니다.</p>
                    </div>
                ) : (
                    <>
                        <div className="text-6xl mb-4 opacity-90 drop-shadow-lg drop-shadow-emerald-500/20">⚽</div>
                        <p className="text-lg font-semibold text-slate-200 mb-2">영상을 드래그하거나 클릭하세요</p>
                        <p className="text-sm text-slate-400 mb-6 font-light">MP4, MOV 등 축구 슈팅 영상</p>
                        <button
                            onClick={onButtonClick}
                            className="px-8 py-3 bg-emerald-500 text-slate-900 font-extrabold rounded-lg hover:bg-emerald-400 transition-all transform hover:scale-105 shadow-[0_0_15px_rgba(16,185,129,0.4)] hover:shadow-[0_0_25px_rgba(16,185,129,0.6)]"
                        >
                            파일 선택하기
                        </button>
                    </>
                )}
            </div>

            {/* 촬영 가이드 배너 */}
            {!loading && (
                <div className="mt-8 bg-slate-800/80 rounded-xl p-5 border border-emerald-500/30 shadow-[0_0_15px_rgba(0,0,0,0.5)]">
                    <h3 className="text-emerald-400 font-bold mb-3 flex items-center text-lg">
                        <span className="mr-2 text-xl shadow-emerald-400/50">💡</span> 속도 및 궤적 추정 가이드
                    </h3>
                    <ul className="text-slate-300 text-sm space-y-3 list-disc pl-5 font-light tracking-wide">
                        <li>
                            <strong className="text-emerald-300 font-semibold drop-shadow-sm">카메라 고정 필수:</strong> 렌즈가 움직이면 속도가 과대 계산될 수 있습니다. 삼각대나 기둥을 사용하세요.
                        </li>
                        <li>
                            <strong className="text-emerald-300 font-semibold drop-shadow-sm">화각 확보:</strong> 영상 안에서 골대 전체 윤곽이 멀리서 한눈에 들어오게 찍어주세요.
                        </li>
                        <li>
                            <strong className="text-emerald-300 font-semibold drop-shadow-sm">기본 분석:</strong> 정상 속도·줌 없는 영상을 기준으로 자동 추정합니다. 슬로모션 영상이나 다른 촬영 설정은 선택 설정에서 변경할 수 있습니다.
                        </li>
                    </ul>
                </div>
            )}
        </div>
    )
}
