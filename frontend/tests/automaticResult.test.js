import test from 'node:test'
import assert from 'node:assert/strict'
import { goalPresentation, physicalGoalPresentation, scoreBasisLabel, scoreWithheldReason, automaticInputLabel } from '../src/lib/goalResult.js'

test('an image candidate is shown independently of an undetermined goal line', () => {
    assert.equal(goalPresentation({status:'CANDIDATE_OBSERVED'}).title,'영상 골대 영역 진입 후보')
    assert.equal(physicalGoalPresentation({status:'UNDETERMINED', reason:'GOAL_PLANE_REQUIRED'}).title,'골인 여부 미확정')
    assert.match(scoreWithheldReason({reason:'GOAL_PLANE_REQUIRED'}),/골대 실측/)
})

test('a goal worth zero and an outside crossing use different score labels', () => {
    assert.match(scoreBasisLabel({value:0, basis:'MEASURED_PLANE_OUTER_DISTANCE'}),/외곽 위치/)
    assert.match(scoreBasisLabel({value:0, basis:'MEASURED_PLANE_OUTSIDE'}),/골대 밖/)
    assert.match(scoreBasisLabel({value:0, basis:'USER_DEFINED_TARGET'}),/사용자 지정/)
    assert.match(scoreWithheldReason({value:null, reason:'SPEED_BASIS_UNVERIFIED'}),/촬영 시간/)
})

test('partial scores and supplied timing do not appear fully measured', () => {
    assert.match(scoreBasisLabel({basis:'SPEED_ONLY'}),/일부 항목/)
    assert.match(scoreBasisLabel({basis:'SPEED_AND_IMAGE_ACCURACY'}),/40%/)
    assert.match(automaticInputLabel({measurement_context:{timing:{mode:'slow_motion'}}}),/슬로모션/)
    assert.match(automaticInputLabel({camera_model:{intrinsics_calibrated:true}}),/보정 사용/)
})
