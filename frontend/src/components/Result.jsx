export default function Result({ data, onReset }) {
    return (
        <div className="flex flex-col items-center">
            <h2 className="text-3xl font-extrabold text-emerald-400 drop-shadow-sm tracking-wide mb-6 uppercase">분석 결과</h2>

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

            <div className="grid grid-cols-2 gap-4 w-full mb-6 relative">
                <div className="bg-slate-800 border border-slate-700/50 p-5 rounded-xl text-center shadow-inner relative overflow-hidden">
                    <div className="absolute -right-4 -top-4 w-16 h-16 bg-cyan-500/10 rounded-full blur-xl"></div>
                    <p className="text-sm text-slate-400 mb-1 font-light tracking-wider">슛 점수</p>
                    <div className="text-4xl font-black text-cyan-400 drop-shadow-[0_0_10px_rgba(34,211,238,0.5)]">
                        {data.score}<span className="text-lg text-slate-500 ml-1">점</span>
                    </div>
                </div>
                <div className="bg-slate-800 border border-slate-700/50 p-5 rounded-xl text-center shadow-inner relative overflow-hidden">
                    <div className="absolute -left-4 -top-4 w-16 h-16 bg-emerald-500/10 rounded-full blur-xl"></div>
                    <p className="text-sm text-slate-400 mb-1 font-light tracking-wider">최고 속도</p>
                    <div className="text-4xl font-black text-emerald-400 drop-shadow-[0_0_12px_rgba(52,211,153,0.6)]">
                        {Number(data.speed).toFixed(1)}<span className="text-lg text-slate-500 ml-1">km/h</span>
                    </div>
                </div>
            </div>

            <div className="bg-slate-800/80 border border-emerald-500/20 p-5 rounded-xl w-full text-center mb-8 shadow-[0_0_15px_rgba(0,0,0,0.3)]">
                <p className="text-emerald-300 font-semibold tracking-wide text-lg">"{data.comment}"</p>
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
