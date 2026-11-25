import { parseEvent, type AgentEvent } from '../features/agent/events'

export type StreamStatus = 'connecting' | 'open' | 'reconnecting' | 'closed'

/**
 * Typed Server-Sent Events client. EventSource reconnects on its own and sends Last-Event-ID, so the
 * server replays only what we missed; the stream is closed for good when the server sends `end`.
 */
export function openEventStream(
  url: string,
  handlers: { onEvent: (e: AgentEvent) => void; onStatus?: (s: StreamStatus) => void; onEnd?: () => void },
): () => void {
  const es = new EventSource(url)
  handlers.onStatus?.('connecting')
  es.onopen = () => handlers.onStatus?.('open')
  es.onmessage = (msg) => {
    try {
      const ev = parseEvent(JSON.parse(msg.data))
      if (ev) handlers.onEvent(ev)
    } catch {
      // ignore malformed frames
    }
  }
  es.addEventListener('end', () => {
    es.close()
    handlers.onStatus?.('closed')
    handlers.onEnd?.()
  })
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) handlers.onStatus?.('closed')
    else handlers.onStatus?.('reconnecting')
  }
  return () => {
    es.close()
    handlers.onStatus?.('closed')
  }
}
