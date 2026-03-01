import React, { useState } from 'react';

const CalibrationModal = ({ isOpen, onClose }) => {
    const [file, setFile] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    if (!isOpen) return null;

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
            setResult(null);
            setError(null);
            setProgress(0);
        }
    };

    const startCalibration = async () => {
        if (!file) return;

        setIsProcessing(true);
        setError(null);
        setResult(null);
        setProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // 1. Upload the file
            const uploadRes = await fetch('http://localhost:8000/calibrate', {
                method: 'POST',
                body: formData,
            });

            if (!uploadRes.ok) {
                throw new Error('Upload failed');
            }

            // 2. We can't directly use EventSource with POST body, so the backend was designed
            // to return an SSE stream directly from the POST request in main.py.
            // We will read the stream manually using fetch iterators.
            const reader = uploadRes.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));

                            if (data.type === 'progress') {
                                setProgress(data.value);
                            } else if (data.type === 'result') {
                                setResult(data.data);
                                setIsProcessing(false);
                            } else if (data.type === 'error') {
                                setError(data.message);
                                setIsProcessing(false);
                            }
                        } catch (err) {
                            console.error("Parse error:", err);
                        }
                    }
                }
            }
        } catch (err) {
            setError(err.message || "An error occurred during calibration.");
            setIsProcessing(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex justify-center items-center">
            <div className="bg-white rounded-xl shadow-2xl p-6 w-11/12 max-w-lg">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-2xl font-bold text-gray-800">📷 카메라 캘리브레이션</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800 text-2xl font-bold">&times;</button>
                </div>

                <p className="text-sm text-gray-600 mb-6">
                    더 정확한 속도/궤적 측정을 위해 렌즈 왜곡을 교정합니다.<br />
                    <strong>9x6 체스보드</strong>를 프린트하여 10초 내외로 천천히 여러 각도에서 찍은 영상을 업로드해주세요.
                </p>

                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:bg-gray-50 transition-colors mb-4">
                    <input
                        type="file"
                        accept="video/*"
                        className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                        onChange={handleFileChange}
                        disabled={isProcessing}
                    />
                </div>

                {isProcessing && (
                    <div className="mb-4">
                        <div className="flex justify-between text-sm text-gray-600 mb-1">
                            <span>분석 중...</span>
                            <span>{progress}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${progress}%` }}
                            ></div>
                        </div>
                        <p className="text-xs text-gray-400 mt-2 text-center">OpenCV 처리 중입니다. 최대 1분 정도 소요될 수 있습니다.</p>
                    </div>
                )}

                {error && (
                    <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm mb-4 break-words">
                        ❌ {error}
                    </div>
                )}

                {result && result.status === 'success' && (
                    <div className="bg-green-50 text-green-700 p-3 rounded-lg text-sm mb-4">
                        ✅ <strong>캘리브레이션 완료!</strong><br />
                        렌즈 고유의 왜곡 계수와 초점 거리가 시스템에 적용되었습니다. 이제 메인 화면에서 슈팅 영상을 분석해보세요.
                    </div>
                )}

                <div className="flex justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 font-medium"
                        disabled={isProcessing}
                    >
                        닫기
                    </button>
                    <button
                        onClick={startCalibration}
                        className={`px-4 py-2 rounded-lg font-medium text-white transition-colors ${!file || isProcessing ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-md'
                            }`}
                        disabled={!file || isProcessing}
                    >
                        교정 시작
                    </button>
                </div>
            </div>
        </div>
    );
};

export default CalibrationModal;
