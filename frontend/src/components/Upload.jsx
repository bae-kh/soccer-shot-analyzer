import { useState, useRef } from 'react'

export default function Upload({ onUploadStart, onAnalysisComplete, onProgress, loading, progress }) {
    const [dragActive, setDragActive] = useState(false)
    const inputRef = useRef(null)

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
        onUploadStart() // 로딩 시작 및 progress 0

        // One-Step Analyze Stream
        const formData = new FormData()
        formData.append("file", file)

        try {
            const response = await fetch("http://localhost:8000/analyze", {
                method: "POST",
                body: formData,
            })

            if (!response.body) throw new Error("No response body")

            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ""

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split("\n\n")
                buffer = lines.pop() // Last incomplete line

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const data = JSON.parse(line.slice(6))
                            if (data.type === "progress") {
                                onProgress(data.value)
                            } else if (data.type === "result") {
                                const resultData = data.data
                                resultData.image_url = `http://localhost:8000/results/${resultData.trajectory_image}`
                                onAnalysisComplete(resultData)
                            } else if (data.type === "error") {
                                alert("Error: " + data.message)
                                window.location.reload()
                            }
                        } catch (e) {
                            console.error("JSON Parse Error", e)
                        }
                    }
                }
            }

        } catch (error) {
            console.error(error)
            alert("서버 연결 오류가 발생했습니다.")
            window.location.reload()
        }

    }

    const onButtonClick = () => {
        inputRef.current.click()
    }

    return (
        <div className="w-full">
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
                        <span className="mr-2 text-xl shadow-emerald-400/50">💡</span> 정확한 측정 가이드
                    </h3>
                    <ul className="text-slate-300 text-sm space-y-3 list-disc pl-5 font-light tracking-wide">
                        <li>
                            <strong className="text-emerald-300 font-semibold drop-shadow-sm">카메라 고정 필수:</strong> 렌즈가 움직이면 속도가 부풀려집니다. 삼각대나 기둥을 사용하세요.
                        </li>
                        <li>
                            <strong className="text-emerald-300 font-semibold drop-shadow-sm">화각 확보:</strong> 영상 안에서 골대 전체 윤곽이 멀리서 한눈에 들어오게 찍어주세요.
                        </li>
                        <li>
                            <strong className="text-emerald-300 font-semibold drop-shadow-sm">스마트폰 자동 인식:</strong> 분석기가 영상을 올리기만 해도 <span className="text-cyan-300">내부 렌즈 데이터를 자동으로 추출(Auto-Calib)</span>하여 오차를 줄입니다.
                        </li>
                    </ul>
                </div>
            )}
        </div>
    )
}
