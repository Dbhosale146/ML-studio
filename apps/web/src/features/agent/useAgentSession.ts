import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api } from '../../lib/apiClient'
import { openEventStream, type StreamStatus } from '../../lib/sse'
import type { Schemas } from '../../lib/apiClient'
import { agentReducer, initialAgentState } from './reducer'

type Session = Schemas['SessionOut']

/** Live agent session: replays history, streams new events, and exposes the reply/message/cancel/retry commands. */
export function useAgentSession(sessionId: string) {
  const qc = useQueryClient()
  const [state, dispatch] = useReducer(agentReducer, initialAgentState)
  const [stream, setStream] = useState<StreamStatus>('connecting')
  const [generation, setGeneration] = useState(0)
  const connectedAt = useRef(Date.now())
  const lastSeq = useRef(0)
  lastSeq.current = state.lastSeq

  const session = useQuery({
    queryKey: ['sessions', sessionId],
    queryFn: () => api.get<Session>(`/sessions/${sessionId}`),
    refetchInterval: (q) => (q.state.data && ['done', 'failed', 'cancelled'].includes(q.state.data.status) ? false : 5000),
  })

  useEffect(() => {
    connectedAt.current = Date.now()
    const close = openEventStream(`/api/v1/sessions/${sessionId}/events?after=${lastSeq.current}`, {
      onEvent: (ev) => {
        dispatch(ev)
        if (ev.type === 'model' || ev.type === 'done') void qc.invalidateQueries({ queryKey: ['models'] })
        if (ev.type === 'done' || ev.type === 'error' || ev.type === 'session') void qc.invalidateQueries({ queryKey: ['sessions', sessionId] })
      },
      onStatus: setStream,
    })
    return close
    // reconnect only when the session changes or after a retry
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, generation])

  const reply = useMutation({
    mutationFn: (body: { ask_id: string; value?: unknown; text?: string | null }) => api.post(`/sessions/${sessionId}/reply`, body),
  })
  const message = useMutation({ mutationFn: (text: string) => api.post(`/sessions/${sessionId}/message`, { text }) })
  const cancel = useMutation({
    mutationFn: () => api.post<Session>(`/sessions/${sessionId}/cancel`),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['sessions', sessionId] }),
  })
  const retry = useMutation({
    mutationFn: () => api.post<Session>(`/sessions/${sessionId}/retry`),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['sessions', sessionId] })
      setGeneration((g) => g + 1) // the old stream ended with the failed run: open a new one
    },
  })

  const answer = useCallback(
    (value: unknown, text?: string) => {
      if (!state.pendingAsk) return
      reply.mutate({ ask_id: state.pendingAsk.ask_id, value, text: text ?? null })
    },
    [reply, state.pendingAsk],
  )

  return { state, stream, session: session.data, connectedAt: connectedAt.current, answer, reply, message, cancel, retry }
}
