import { useState } from 'react';
import { readJsonEvents } from '../lib/sse';

const CalibrationModal = ({ isOpen, onClose, onCalibrationComplete }) => {
    const [file, setFile] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [label, setLabel] = useState('');

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
        if (!file || !label.trim()) return;

        setIsProcessing(true);
        setError(null);
        setResult(null);
        setProgress(0);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('profile_label', label.trim());

        try {
            // 1. Upload the file
            const uploadRes = await fetch('http://localhost:8000/calibrate', {
                method: 'POST',
                body: formData,
            });

            if (!uploadRes.ok) {
                throw new Error('Upload failed');
            }

            let receivedFinal = false;
            for await (const data of readJsonEvents(uploadRes.body)) {
                if (data.type === 'progress') {
                    setProgress(data.value);
                } else if (data.type === 'result') {
                    setResult(data.data);
                    onCalibrationComplete();
                    receivedFinal = true;
                } else if (data.type === 'error') {
                    setError(data.message);
                    receivedFinal = true;
                }
            }
            if (!receivedFinal) throw new Error('카메라 보정 결과를 받지 못했습니다.');
        } catch (err) {
            setError(err.message || "An error occurred during calibration.");
        } finally {
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
                    <strong>내부 교차점 9×6개인 체커보드(칸 수 10×7)</strong>를 프린트하여 천천히 여러 각도에서 찍은 영상을 업로드해주세요. 슈팅 촬영과 같은 카메라·렌즈·줌 설정을 사용하세요.
                </p>
                <label className="block text-sm text-gray-700 mb-4">카메라 · 렌즈 · 줌 설정 이름
                    <input className="w-full border rounded-lg p-2 mt-1 text-gray-800" maxLength={120} value={label}
                        onChange={e => setLabel(e.target.value)} disabled={isProcessing} placeholder="예: 휴대폰 후면 1× / 1080p" />
                </label>

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
                        ‘{result.profile_label}’ 보정을 저장했습니다. 메인 화면에서 해당 설정을 선택하세요. 렌즈·줌·촬영 모드가 바뀌면 별도로 보정하세요.
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
                        className={`px-4 py-2 rounded-lg font-medium text-white transition-colors ${!file || !label.trim() || isProcessing ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-md'
                            }`}
                        disabled={!file || !label.trim() || isProcessing}
                    >
                        교정 시작
                    </button>
                </div>
            </div>
        </div>
    );
};

export default CalibrationModal;
