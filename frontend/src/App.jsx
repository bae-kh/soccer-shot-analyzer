import { useState } from 'react'
import Upload from './components/Upload'
import Result from './components/Result'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)

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
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-green-500 mb-2">
          AI 축구 슈팅 분석기 ⚽
        </h1>
        <p className="text-gray-600">당신의 슛 속도와 궤적을 분석해드립니다.</p>
      </header>

      <main className="w-full max-w-md bg-white rounded-2xl shadow-xl overflow-hidden p-6 transition-all duration-300">
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

      <footer className="mt-8 text-sm text-gray-500">
        Developed with A16Z Style AI Stack
      </footer>
    </div>
  )
}

export default App
