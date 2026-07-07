import { useState } from 'react'
import Upload from './components/Upload'
import Result from './components/Result'
import CalibrationModal from './components/CalibrationModal'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [showCalibration, setShowCalibration] = useState(false)

  const handleAnalysisComplete = (data) => {
    setResult(data)
    setLoading(false)
    setProgress(0)
  }

  const handleReset = () => {
    setResult(null)
    setProgress(0)
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 relative bg-[#0f172a]">
      {/* Settings / Calibration Button */}
      <div className="absolute top-4 right-4">
        <button
          onClick={() => setShowCalibration(true)}
          className="text-slate-400 hover:text-emerald-400 transition-colors p-2 rounded-full hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
          title="카메라 캘리브레이션 설정"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
          </svg>
        </button>
      </div>

      <header className="mb-8 text-center mt-12">
        <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400 mb-3 drop-shadow-[0_0_15px_rgba(52,211,153,0.3)] tracking-tight">
          PRO SHOT ANALYZER ⚽
        </h1>
        <p className="text-slate-400 text-lg font-light tracking-wide">축구 슈팅 궤적 및 속도 추정 프로토타입</p>
      </header>

      <main className="w-full max-w-md bg-slate-800/80 backdrop-blur-lg border border-slate-700/50 rounded-2xl shadow-[0_0_40px_rgba(16,185,129,0.1)] overflow-hidden p-6 transition-all duration-500">
        {!result ? (
          <Upload
            onUploadStart={() => {
              setLoading(true)
              setProgress(0)
            }}
            onAnalysisComplete={handleAnalysisComplete}
            onProgress={setProgress}
            loading={loading}
            progress={progress}
          />
        ) : (
          <Result data={result} onReset={handleReset} />
        )}
      </main>

      <CalibrationModal
        isOpen={showCalibration}
        onClose={() => setShowCalibration(false)}
      />
    </div>
  )
}

export default App
