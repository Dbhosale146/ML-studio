import { describe, expect, it } from 'vitest'
import { parseEvent, type AgentEvent } from '../../src/features/agent/events'
import { agentReducer, initialAgentState, type AgentState } from '../../src/features/agent/reducer'

let seq = 0
const ev = (type: string, payload: Record<string, unknown>): AgentEvent => {
  const e = parseEvent({ seq: ++seq, type, at: 1_700_000_000 + seq, payload })
  if (!e) throw new Error(`invalid ${type}`)
  return e
}
const run = (events: AgentEvent[], from: AgentState = initialAgentState) => events.reduce(agentReducer, from)

describe('agent event reducer', () => {
  it('streams messages from deltas and tracks stages', () => {
    seq = 0
    const s = run([
      ev('stage', { index: 0, key: 'data', label: 'Data', status: 'active' }),
      ev('message', { id: 'm1', text: '' }),
      ev('message.delta', { id: 'm1', delta: 'Hello ' }),
      ev('message.delta', { id: 'm1', delta: '**world**' }),
      ev('stage', { index: 0, key: 'data', label: 'Data', status: 'done' }),
    ])
    expect(s.items).toHaveLength(1)
    expect(s.items[0]).toMatchObject({ kind: 'agent', text: 'Hello **world**' })
    expect(s.stages.data).toBe('done')
    expect(s.stages.detect).toBe('pending')
  })

  it('collects tool logs, card updates, blueprint, decisions and experiments', () => {
    seq = 0
    const s = run([
      ev('tool.start', { id: 't1', label: 'profiler' }),
      ev('tool.log', { id: 't1', line: 'reading' }),
      ev('tool.end', { id: 't1', duration_ms: 1200, status: 'done' }),
      ev('card', { id: 'c1', kind: 'leaderboard', data: { round: 1, rows: [] } }),
      ev('card.update', { id: 'c1', data: { round: 1, rows: [{ name: 'LightGBM' }], done: true } }),
      ev('blueprint', { field: 'Target', value: 'churned' }),
      ev('decision', { step: 'Critic', decision: 'Removed leak', reason: 'AUC 1.0' }),
      ev('experiment', { run_id: 'r1', model: 'LightGBM', round: 1, score: 0.9, display: '0.900' }),
    ])
    expect(s.items[0]).toMatchObject({ kind: 'tool', logs: ['reading'], status: 'done', durationMs: 1200 })
    expect(s.items[1]).toMatchObject({ kind: 'card', data: { done: true } })
    expect(s.blueprint.Target).toBe('churned')
    expect(s.decisions[0].alternatives).toBe('')
    expect(s.experiments).toHaveLength(1)
  })

  it('pauses on asks, clears them on reply, and shows errors', () => {
    seq = 0
    let s = run([
      ev('session', { status: 'waiting' }),
      ev('ask', { ask_id: 'a1', kind: 'options', prompt: 'Target?', options: [{ label: 'Yes', value: 'y' }], default: 'y' }),
    ])
    expect(s.pendingAsk?.ask_id).toBe('a1')
    s = run([ev('reply', { ask_id: 'a1', text: 'Yes', value: 'y' })], s)
    expect(s.pendingAsk).toBeNull()
    expect(s.items.at(-1)).toMatchObject({ kind: 'user', text: 'Yes' })
    s = run([ev('error', { code: 'no_signal', message: 'No model beats the baseline', retryable: false })], s)
    expect(s.items.at(-1)).toMatchObject({ kind: 'error', retryable: false })
    s = run([ev('done', { model_id: 'mdl_1' })], s)
    expect(s.doneModelId).toBe('mdl_1')
  })

  it('ignores duplicate events replayed after a reconnect', () => {
    seq = 0
    const a = ev('message', { id: 'm1', text: 'once' })
    const s = run([a, a, a])
    expect(s.items).toHaveLength(1)
    expect(s.lastSeq).toBe(1)
  })

  it('rejects malformed events', () => {
    expect(parseEvent({ seq: 1, type: 'stage', at: 0, payload: { key: 'x' } })).toBeNull()
    expect(parseEvent({ seq: 1, type: 'unknown', at: 0, payload: {} })).toBeNull()
    expect(parseEvent('garbage')).toBeNull()
  })
})
