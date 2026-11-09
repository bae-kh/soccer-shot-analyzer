import test from 'node:test'
import assert from 'node:assert/strict'
import { goalPresentation, withResultImages } from '../src/lib/goalResult.js'

test('null speed does not prevent candidate presentation or image mapping', () => {
    const data = withResultImages({ speed: null, goal_entry: { status: 'CANDIDATE_OBSERVED' },
        trajectory_image: 'early.jpg', goal_entry_image: 'whole path.jpg' })
    assert.equal(goalPresentation(data.goal_entry).title, '영상 골대 영역 진입 후보')
    assert.equal(data.goal_entry_image_url, 'http://localhost:8000/results/whole%20path.jpg')
    assert.equal(data.speed, null)
})

test('no candidate explicitly preserves unknown physical goal', () => {
    assert.match(goalPresentation({ status: 'NO_CANDIDATE_OBSERVED' }).description, /확정한 결과는 아닙니다/)
})

test('deferred camera and observation reasons are explained', () => {
    assert.match(goalPresentation({ status: 'UNAVAILABLE', reason: 'CAMERA_MOVING' }).description, /카메라 움직임/)
    assert.match(goalPresentation({ status: 'UNAVAILABLE', reason: 'ENTRY_CONTINUITY_UNVERIFIED' }).description, /연속 관측/)
})

test('old or unknown response stays deferred without a broken evidence image', () => {
    assert.equal(goalPresentation().title, '골대 진입 판정 보류')
    assert.equal(goalPresentation({ status: 'GOAL' }).title, '골대 진입 판정 보류')
    assert.equal(withResultImages({ trajectory_image: 'a.jpg' }).goal_entry_image_url, undefined)
})
