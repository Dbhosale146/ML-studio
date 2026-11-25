/** Agent event protocol (mirrors services/api/src/datalyon/schemas/events.py; see docs/agent-events.md). */
import { z } from 'zod'

const record = z.record(z.string(), z.unknown())

export const AskOption = z.object({ label: z.string(), value: z.unknown().optional(), hint: z.string().nullish() })

export const payloads = {
  stage: z.object({ index: z.number(), key: z.string(), label: z.string(), status: z.enum(['active', 'done', 'failed', 'skipped']) }),
  thinking: z.object({ text: z.string().nullish() }),
  message: z.object({ id: z.string(), text: z.string() }),
  'message.delta': z.object({ id: z.string(), delta: z.string() }),
  'tool.start': z.object({ id: z.string(), label: z.string() }),
  'tool.log': z.object({ id: z.string(), line: z.string() }),
  'tool.end': z.object({ id: z.string(), duration_ms: z.number(), status: z.enum(['done', 'error']) }),
  card: z.object({ id: z.string(), kind: z.string(), data: record }),
  'card.update': z.object({ id: z.string(), data: record }),
  ask: z.object({
    ask_id: z.string(),
    kind: z.enum(['options', 'free_text', 'approval', 'upload', 'sheet']),
    prompt: z.string(),
    options: z.array(AskOption).default([]),
    items: z.array(record).default([]),
    default: z.unknown().optional(),
  }),
  reply: z.object({ ask_id: z.string().nullish(), text: z.string(), value: z.unknown().optional() }),
  blueprint: z.object({ field: z.string(), value: z.string() }),
  decision: z.object({ step: z.string(), decision: z.string(), reason: z.string(), alternatives: z.string().default('') }),
  experiment: z.object({
    run_id: z.string(),
    model: z.string(),
    round: z.number(),
    score: z.number().nullish(),
    display: z.string(),
    best: z.boolean().default(false),
    tuned: z.boolean().default(false),
  }),
  model: z.object({
    id: z.string(),
    status: z.enum(['draft', 'training', 'ready', 'failed']),
    progress: z.number(),
    stage: z.string().nullish(),
    name: z.string().nullish(),
  }),
  session: z.object({ status: z.enum(['running', 'waiting', 'done', 'failed', 'cancelled']) }),
  error: z.object({ code: z.string(), message: z.string(), retryable: z.boolean() }),
  done: z.object({ model_id: z.string().nullish() }),
} as const

export type EventType = keyof typeof payloads
export type Payload<T extends EventType> = z.infer<(typeof payloads)[T]>
export type AskOptionT = z.infer<typeof AskOption>
export type Ask = Payload<'ask'>

export type AgentEvent = { [K in EventType]: { seq: number; type: K; at: number; payload: Payload<K> } }[EventType]

const Envelope = z.object({ seq: z.number(), type: z.string(), at: z.number(), payload: record })

/** Validate one SSE `data:` line. Unknown or malformed events return null (and are logged), never crash the UI. */
export function parseEvent(raw: unknown): AgentEvent | null {
  const env = Envelope.safeParse(raw)
  if (!env.success) return null
  const schema = payloads[env.data.type as EventType]
  if (!schema) return null
  const p = schema.safeParse(env.data.payload)
  if (!p.success) {
    console.warn('Ignoring invalid agent event', env.data.type, p.error.issues[0])
    return null
  }
  return { ...env.data, payload: p.data } as AgentEvent
}
