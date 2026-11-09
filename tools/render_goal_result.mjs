// Render the real Result component using actual saved analyzer responses.
import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'

const root = process.cwd()
const run = path.resolve(process.argv[2])
const requireFrontend = createRequire(path.join(root, 'frontend/package.json'))
const { buildSync } = requireFrontend('esbuild')
const React = requireFrontend('react')
const { renderToStaticMarkup } = requireFrontend('react-dom/server')
const built = buildSync({ entryPoints: [path.join(root, 'frontend/src/components/Result.jsx')],
    bundle: true, write: false, platform: 'node', format: 'cjs', jsx: 'automatic', external: ['react', 'react/jsx-runtime'] })
const componentModule = { exports: {} }
new Function('require', 'module', 'exports', built.outputFiles[0].text)(requireFrontend, componentModule, componentModule.exports)
const Result = componentModule.exports.default
const summary = JSON.parse(fs.readFileSync(path.join(run, 'summary.json'), 'utf8'))
const previewLabel = summary.domain === 'synthetic'
    ? '합성 관측으로 계산한 검증용 반환값 · 실제 영상 아님 · 속도 60 km/h는 채점 입력 예시'
    : '저장된 실제 반환값으로 결과 컴포넌트를 렌더링한 미리보기'
const review = path.join(run, 'review')
fs.mkdirSync(review, { recursive: true })
const css = fs.readdirSync(path.join(root, 'frontend/dist/assets')).find(name => name.endsWith('.css'))
fs.copyFileSync(path.join(root, 'frontend/dist/assets', css), path.join(review, 'result.css'))
for (const row of summary.videos) {
    const raw = row.raw_result
    if (!raw?.goal_entry) continue
    const imagePrefix = `../images/${path.posix.dirname(row.video)}/`
    const data = { ...raw, image_url: imagePrefix + raw.trajectory_image,
        goal_entry_image_url: raw.goal_entry_image ? imagePrefix + raw.goal_entry_image : undefined }
    const content = renderToStaticMarkup(React.createElement(Result, { data, onReset: () => {} }))
    const filename = row.video.replaceAll('/', '_').replace(/\.mp4$/, '_result.html')
    fs.writeFileSync(path.join(review, filename), `<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="result.css"><title>슈팅 분석 결과 검증 화면</title><body class="bg-slate-900 text-white"><main class="max-w-4xl mx-auto px-6 py-8"><p class="text-slate-400 mb-5">${row.video} · ${previewLabel}</p>${content}</main></body></html>`)
}
console.log(`Rendered ${summary.videos.length} actual result previews into ${review}`)
