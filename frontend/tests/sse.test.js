import assert from 'node:assert/strict'
import test from 'node:test'
import { readJsonEvents } from '../src/lib/sse.js'

function body(text, size = 7) {
    const bytes = new TextEncoder().encode(text)
    return new ReadableStream({
        start(controller) {
            for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size))
            controller.close()
        },
    })
}

async function collect(stream) {
    const events = []
    for await (const event of readJsonEvents(stream)) events.push(event)
    return events
}

test('buffers split Korean JSON and several events with LF or CRLF', async () => {
    const events = [{ type: 'progress', value: 50 }, { type: 'result', data: { message: '보정 완료' } }]
    for (const newline of ['\n', '\r\n']) {
        const text = events.map(event => `data: ${JSON.stringify(event)}${newline}${newline}`).join('')
        for (const size of [1, 7, 1000]) assert.deepEqual(await collect(body(text, size)), events)
    }
})

test('ignores heartbeat comments and combines multiline data', async () => {
    assert.deepEqual(await collect(body(': heartbeat\n\ndata: {"type":\ndata: "keepalive"}\n\n')), [{ type: 'keepalive' }])
})

test('rejects a truncated response instead of losing the result', async () => {
    await assert.rejects(collect(body('data: {"type":"result"}')), /완전히/)
})

test('reports malformed JSON and missing response body', async () => {
    await assert.rejects(collect(body('data: invalid\n\n')), SyntaxError)
    await assert.rejects(collect(null), /응답/)
})
