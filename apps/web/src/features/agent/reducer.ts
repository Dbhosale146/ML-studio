/** Pure reducer: folds agent events into everything the chat and workspace render. Safe to replay. */
import type { AgentEvent, Ask, Payload } from './events'

export const STAGES: Array<{ key: string; label: string }> = [
  { key: 'data', label: 'Data & profile' },
  { key: 'detect', label: 'Problem detection' },
  { key: 'strategy', label: 'ML strategy' },
  { key: 'prep', label: 'Cleaning & features' },
  { key: 'train', label: 'Model search + HPO' },
  { key: 'critic', label: 'Critic review + round 2' },
  { key: 'final', label: 'Evaluation & explain' },
  { key: 'publish', label: 'Publish for testing' },
]

export type StageStatus = 'pending' | 'active' | 'done' | 'failed' | 'skipped'

export type ChatItem =
  | { kind: 'agent'; id: string; text: string; at: number; streaming: boolean }
  | { kind: 'user'; id: string; text: string; at: number }
  | { kind: 'tool'; id: string; label: string; logs: string[]; status: 'running' | 'done' | 'error'; durationMs: number | null; at: number }
  | { kind: 'card'; id: string; cardKind: string; data: Record<string, unknown>; at: number }
  | { kind: 'error'; id: string; code: string; message: string; retryable: boolean; at: number }

export interface AgentState {
  lastSeq: number
  items: ChatItem[]
  stages: Record<string, StageStatus>
  thinking: string | null
  pendingAsk: Ask | null
  blueprint: Record<string, string>
  decisions: Array<Payload<'decision'>>
  experiments: Array<Payload<'experiment'>>
  model: Payload<'model'> | null
  status: Payload<'session'>['status'] | null
  doneModelId: string | null
}

export const initialAgentState: AgentState = {
  lastSeq: 0,
  items: [],
  stages: Object.fromEntries(STAGES.map((s) => [s.key, 'pending' as StageStatus])),
  thinking: null,
  pendingAsk: null,
  blueprint: {},
  decisions: [],
  experiments: [],
  model: null,
  status: null,
  doneModelId: null,
}

function updateItem(items: ChatItem[], id: string, fn: (item: ChatItem) => ChatItem): ChatItem[] {
  const i = items.findIndex((x) => x.id === id)
  if (i === -1) return items
  const next = items.slice()
  next[i] = fn(items[i])
  return next
}

export function agentReducer(state: AgentState, ev: AgentEvent): AgentState {
  if (ev.seq <= state.lastSeq) return state // duplicates after an SSE reconnect
  const s: AgentState = { ...state, lastSeq: ev.seq }
  switch (ev.type) {
    case 'stage':
      return { ...s, stages: { ...s.stages, [ev.payload.key]: ev.payload.status } }
    case 'thinking':
      return { ...s, thinking: ev.payload.text ?? null }
    case 'message': {
      const exists = s.items.some((x) => x.id === ev.payload.id)
      if (exists) return { ...s, items: updateItem(s.items, ev.payload.id, (x) => ({ ...x, text: ev.payload.text }) as ChatItem) }
      return {
        ...s,
        thinking: null,
        items: [...s.items, { kind: 'agent', id: ev.payload.id, text: ev.payload.text, at: ev.at, streaming: ev.payload.text === '' }],
      }
    }
    case 'message.delta':
      return { ...s, items: updateItem(s.items, ev.payload.id, (x) => (x.kind === 'agent' ? { ...x, text: x.text + ev.payload.delta } : x)) }
    case 'tool.start':
      return {
        ...s,
        items: [...s.items, { kind: 'tool', id: ev.payload.id, label: ev.payload.label, logs: [], status: 'running', durationMs: null, at: ev.at }],
      }
    case 'tool.log':
      return { ...s, items: updateItem(s.items, ev.payload.id, (x) => (x.kind === 'tool' ? { ...x, logs: [...x.logs, ev.payload.line] } : x)) }
    case 'tool.end':
      return {
        ...s,
        items: updateItem(s.items, ev.payload.id, (x) =>
          x.kind === 'tool' ? { ...x, status: ev.payload.status, durationMs: ev.payload.duration_ms } : x,
        ),
      }
    case 'card':
      return { ...s, items: [...s.items, { kind: 'card', id: ev.payload.id, cardKind: ev.payload.kind, data: ev.payload.data, at: ev.at }] }
    case 'card.update':
      return { ...s, items: updateItem(s.items, ev.payload.id, (x) => (x.kind === 'card' ? { ...x, data: ev.payload.data } : x)) }
    case 'ask':
      return { ...s, thinking: null, pendingAsk: ev.payload }
    case 'reply':
      return {
        ...s,
        pendingAsk: s.pendingAsk && (ev.payload.ask_id == null || s.pendingAsk.ask_id === ev.payload.ask_id) ? null : s.pendingAsk,
        items: [...s.items, { kind: 'user', id: `u${ev.seq}`, text: ev.payload.text, at: ev.at }],
      }
    case 'blueprint':
      return { ...s, blueprint: { ...s.blueprint, [ev.payload.field]: ev.payload.value } }
    case 'decision':
      return { ...s, decisions: [...s.decisions, ev.payload] }
    case 'experiment':
      return { ...s, experiments: [...s.experiments, ev.payload] }
    case 'model':
      return { ...s, model: ev.payload }
    case 'session':
      return {
        ...s,
        status: ev.payload.status,
        pendingAsk: ev.payload.status === 'waiting' ? s.pendingAsk : ev.payload.status === 'running' ? s.pendingAsk : null,
      }
    case 'error':
      return {
        ...s,
        thinking: null,
        pendingAsk: null,
        items: [
          ...s.items,
          { kind: 'error', id: `e${ev.seq}`, code: ev.payload.code, message: ev.payload.message, retryable: ev.payload.retryable, at: ev.at },
        ],
      }
    case 'done':
      return { ...s, thinking: null, pendingAsk: null, doneModelId: ev.payload.model_id ?? null, status: 'done' }
    default:
      return s
  }
}

/** Stream finished agent messages instantly; animate only those that arrive live. */
export function isLive(item: ChatItem, connectedAt: number): boolean {
  return item.at * 1000 >= connectedAt - 500
}
