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
        <div
            className={`flex flex-col items-center justify-center p-10 border-2 border-dashed rounded-xl transition-colors ${dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-gray-50"
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
                <div className="flex flex-col items-center w-full">
                    <div className="w-full bg-gray-200 rounded-full h-4 mb-4 overflow-hidden">
                        <div
                            className="bg-blue-600 h-4 rounded-full transition-all duration-300 ease-out"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                    <p className="text-lg font-bold text-blue-600">{progress}% 분석 중...</p>
                    <p className="text-sm text-gray-500 mt-2">AI가 궤적을 추적하고 있습니다.</p>
                </div>
            ) : (
                <>
                    <div className="text-6xl mb-4">📹</div>
                    <p className="text-lg font-semibold text-gray-700 mb-2">영상을 드래그하거나 클릭하세요</p>
                    <p className="text-sm text-gray-500 mb-6">MP4, MOV 등 축구 슈팅 영상</p>
                    <button
                        onClick={onButtonClick}
                        className="px-6 py-2 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition-transform transform hover:scale-105"
                    >
                        파일 선택하기
                    </button>
                </>
            )}
        </div>
    )
}
