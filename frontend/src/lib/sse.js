/** Read JSON SSE events across arbitrary byte chunks, including Korean UTF-8. */
export async function* readJsonEvents(body) {
    if (!body) throw new Error('서버 응답이 없습니다.')
    const reader = body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    try {
        while (true) {
            const { value, done } = await reader.read()
            buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
            let boundary
            while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
                const block = buffer.slice(0, boundary.index)
                buffer = buffer.slice(boundary.index + boundary[0].length)
                const lines = block.split(/\r?\n/).filter(line => line.startsWith('data:'))
                if (lines.length) {
                    const data = lines.map(line => line.slice(5).replace(/^ /, '')).join('\n')
                    yield JSON.parse(data)
                }
            }
            if (done) break
        }
        if (buffer.trim()) throw new Error('서버 응답이 완전히 도착하지 않았습니다.')
    } finally {
        reader.releaseLock()
    }
}
