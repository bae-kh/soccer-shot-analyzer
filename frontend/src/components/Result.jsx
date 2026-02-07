export default function Result({ data, onReset }) {
    return (
        <div className="flex flex-col items-center">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">분석 결과</h2>

            <div className="relative w-full aspect-video rounded-lg overflow-hidden shadow-md mb-6 bg-black">
                <img
                    src={data.image_url}
                    alt="Trajectory"
                    className="w-full h-full object-contain"
                    onError={(e) => { e.target.src = "https://via.placeholder.com/640x360?text=Image+Load+Error" }}
                />
                <div className="absolute top-2 right-2 bg-black/60 text-white px-3 py-1 rounded-full text-sm">
                    Speed Trace
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4 w-full mb-6">
                <div className="bg-blue-50 p-4 rounded-xl text-center">
                    <p className="text-sm text-gray-500 mb-1">슛 점수</p>
                    <div className="text-3xl font-black text-blue-600">
                        {data.score}<span className="text-lg text-gray-400">점</span>
                    </div>
                </div>
                <div className="bg-green-50 p-4 rounded-xl text-center">
                    <p className="text-sm text-gray-500 mb-1">예상 속도</p>
                    <div className="text-3xl font-black text-green-600">
                        {data.speed}<span className="text-lg text-gray-400">km/h</span>
                    </div>
                </div>
            </div>

            <div className="bg-gray-100 p-4 rounded-xl w-full text-center mb-6">
                <p className="text-gray-800 font-medium">"{data.comment}"</p>
            </div>

            <button
                onClick={onReset}
                className="px-8 py-3 bg-gray-800 text-white font-bold rounded-lg hover:bg-gray-900 transition-colors w-full"
            >
                다른 영상 분석하기
            </button>
        </div>
    )
}
