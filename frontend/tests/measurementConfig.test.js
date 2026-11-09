import test from 'node:test'
import assert from 'node:assert/strict'
import { readMeasurementConfig, measurementConfigProblem } from '../src/lib/measurementConfig.js'
import { physicalGoalPresentation } from '../src/lib/goalResult.js'

test('import never trusts confirmations stored in a file', () => {
    const config = readMeasurementConfig(JSON.stringify({ goal_plane: { confirmed_by_user: true },
        scoring_rules: { confirmed_by_user: true, accept_conditional_estimates: true } }))
    assert.equal(config.goal_plane.confirmed_by_user, false)
    assert.equal(config.scoring_rules.accept_conditional_estimates, false)
    assert.match(measurementConfigProblem(config), /測定|측정값/)
    config.goal_plane.confirmed_by_user = true
    assert.match(measurementConfigProblem(config), /채점 기준/)
    config.scoring_rules.confirmed_by_user = true
    config.scoring_rules.accept_conditional_estimates = true
    assert.equal(measurementConfigProblem(config), null)
})

test('invalid settings and unrelated capture keys are refused', () => {
    for (const text of ['null', '[]', '{}', '{invalid', '{"goal_plane":{},"capture_fps":120}']) {
        assert.throws(() => readMeasurementConfig(text))
    }
    assert.equal(measurementConfigProblem({ timing_mode: 'unknown' }), null)
})

test('physical estimates stay distinct from image candidates and confirmed goals', () => {
    assert.equal(physicalGoalPresentation({ status: 'ESTIMATED_GOAL' }).title, '골라인 통과 추정')
    assert.match(physicalGoalPresentation({ status: 'ESTIMATED_GOAL' }).description, /미검증/)
    assert.equal(physicalGoalPresentation({ status: 'CANDIDATE_OBSERVED' }).title, '골인 여부 미확정')
    assert.equal(physicalGoalPresentation().title, '골인 여부 미확정')
})
