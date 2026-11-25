import { useMutation, useQuery } from '@tanstack/react-query'
import { Brain, Database, FileSpreadsheet, LoaderCircle, Plus, Square, Wand2, X } from 'lucide-react'
import { useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { Badge, Button, Card, Spinner } from '../components/ui'
import { AgentAvatar, AskPanel, ChatItems, Composer, useStickToBottom } from '../features/agent/AgentChat'
import { useAgentSession } from '../features/agent/useAgentSession'
import { Workspace, type WorkspaceTab } from '../features/agent/Workspace'
import { useDataset, useHealth } from '../features/datasets/hooks'
import { UploadDropzone } from '../features/datasets/UploadDropzone'
import { api, errorMessage, type Schemas } from '../lib/apiClient'
import { cx } from '../lib/cx'

type Sample = { name: string; goal: string; available: boolean }

function NewSession() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [goal, setGoal] = useState(params.get('goal') ?? '')
  const datasetId = params.get('dataset') ?? undefined
  const dataset = useDataset(datasetId)
  const samples = useQuery({ queryKey: ['samples'], queryFn: () => api.get<Sample[]>('/datasets/samples') })
  const start = useMutation({
    mutationFn: (body: { goal: string; dataset_id?: string }) => api.post<Schemas['SessionOut']>('/sessions', body),
    onSuccess: (s) => navigate(`/agent/${s.id}`),
  })
  const sample = useMutation({
    mutationFn: async (s: Sample) => {
      const ds = await api.post<Schemas['DatasetOut']>(`/datasets/samples/${encodeURIComponent(s.name)}`)
      return api.post<Schemas['SessionOut']>('/sessions', { goal: s.goal, dataset_id: ds.id })
    },
    onSuccess: (s) => navigate(`/agent/${s.id}`),
  })
  const busy = start.isPending || sample.isPending
  const submit = () => goal.trim().length >= 3 && start.mutate({ goal: goal.trim(), dataset_id: datasetId })

  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <div className="flex items-center gap-3">
        <AgentAvatar />
        <div>
          <h1 className="text-[24px] font-semibold tracking-tight text-slate-900">What would you like to predict?</h1>
          <p className="text-sm text-slate-600">Describe your goal in plain English. I'll ask for your Excel file next, unless you add it now.</p>
        </div>
      </div>
      <Card className="mt-6 p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
          className="flex flex-col gap-2 rounded-xl bg-white ring-1 ring-slate-200 focus-within:ring-2 focus-within:ring-emerald-400 sm:flex-row sm:items-center sm:p-2"
        >
          <div className="flex flex-1 items-center gap-2 px-2">
            <Brain size={20} className="shrink-0 text-indigo-500" aria-hidden />
            <label htmlFor="goal" className="sr-only">
              Your goal
            </label>
            <input
              id="goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Predict which customers will churn next month"
              className="w-full bg-transparent py-2.5 text-[14px] outline-none placeholder:text-slate-400"
            />
          </div>
          <Button type="submit" variant="primary" size="lg" icon={Wand2} disabled={busy || goal.trim().length < 3}>
            {start.isPending ? 'Starting…' : 'Start'}
          </Button>
        </form>
        <div className="mt-3">
          {datasetId && dataset.data ? (
            <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[13px]">
              <FileSpreadsheet size={16} className="text-emerald-600" aria-hidden />
              <span className="min-w-0 flex-1 truncate">
                Using <strong>{dataset.data.file_name}</strong>
                {dataset.data.sheet ? ` · ${dataset.data.sheet}` : ''}
              </span>
              <button
                type="button"
                className="rounded p-1 text-slate-500 hover:bg-white"
                aria-label="Remove dataset"
                onClick={() => {
                  params.delete('dataset')
                  setParams(params)
                }}
              >
                <X size={14} />
              </button>
            </div>
          ) : (
            <UploadDropzone
              compact
              onUploaded={(ds) => {
                params.set('dataset', ds.id)
                setParams(params)
              }}
            />
          )}
        </div>
        {(start.isError || sample.isError) && (
          <p role="alert" className="mt-2 text-[13px] text-red-700">
            {errorMessage(start.error ?? sample.error)}
          </p>
        )}
      </Card>
      <div className="mt-6">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Or try a sample dataset</div>
        <div className="grid gap-2 sm:grid-cols-2">
          {(samples.data ?? []).map((s) => (
            <button
              key={s.name}
              type="button"
              disabled={!s.available || busy}
              onClick={() => sample.mutate(s)}
              className="group rounded-xl border border-slate-200 bg-white p-3 text-left shadow-sm transition-all hover:-translate-y-px hover:border-indigo-300 focus-visible:outline-2 focus-visible:outline-emerald-500 disabled:opacity-50"
            >
              <div className="flex items-center gap-1.5 font-mono text-[11px] text-indigo-600">
                <Database size={12} aria-hidden /> {s.name}
              </div>
              <div className="mt-0.5 text-[13px] font-medium text-slate-800">{s.goal}</div>
              {!s.available && <div className="text-[11px] text-slate-400">Run `make samples` to generate it</div>}
            </button>
          ))}
        </div>
        {sample.isPending && (
          <div className="mt-3">
            <Spinner label="Loading the sample and starting the agent" />
          </div>
        )}
      </div>
    </div>
  )
}

function SessionView({ sessionId }: { sessionId: string }) {
  const navigate = useNavigate()
  const health = useHealth()
  const { state, stream, session, answer, reply, message, cancel, retry } = useAgentSession(sessionId)
  const [tab, setTab] = useState<WorkspaceTab>('blueprint')
  const [showWs, setShowWs] = useState(false)
  const { scrollRef, contentRef, pin } = useStickToBottom(sessionId)
  const status = state.status ?? session?.status
  const running = status === 'running' || status === 'waiting'
  const ask = state.pendingAsk
  const llm = health.data?.llm

  const onAnswer = (value: unknown, text?: string) => {
    pin()
    answer(value, text)
  }
  const onSend = (text: string) => {
    pin()
    message.mutate(text)
  }

  return (
    <div className="flex h-[calc(100vh-4rem)] min-h-0">
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3 lg:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <AgentAvatar />
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-[14px] font-semibold text-slate-900">AI Data Scientist</h1>
                <span
                  className={cx('flex items-center gap-1 text-[11px] font-medium', stream === 'reconnecting' ? 'text-amber-700' : 'text-emerald-700')}
                  aria-live="polite"
                >
                  <span className={cx('h-1.5 w-1.5 rounded-full', stream === 'reconnecting' ? 'bg-amber-500' : 'bg-emerald-500')} />
                  {stream === 'reconnecting' ? 'reconnecting…' : (status ?? 'connecting')}
                </span>
              </div>
              <div className="truncate text-[11px] text-slate-500" title={session?.goal}>
                {session?.goal}
              </div>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Badge color={llm?.provider === 'gemini' ? 'indigo' : 'slate'}>
              {llm?.provider === 'gemini' ? `Gemini · ${llm.model}` : 'MockLLM (offline)'}
            </Badge>
            {running && (
              <Button size="sm" variant="danger" icon={Square} onClick={() => cancel.mutate()} disabled={cancel.isPending}>
                Cancel
              </Button>
            )}
            <Button size="sm" onClick={() => setShowWs((v) => !v)} className="lg:hidden" aria-expanded={showWs}>
              Workspace
            </Button>
            <Button size="sm" icon={Plus} onClick={() => navigate('/agent')}>
              New
            </Button>
          </div>
        </div>
        <div ref={scrollRef} className="thin-scroll min-h-0 flex-1 overflow-y-auto bg-canvas">
          <div ref={contentRef} className="mx-auto max-w-3xl space-y-4 px-4 py-6 lg:px-6" aria-live="polite">
            {state.items.length === 0 && (
              <div className="flex justify-center py-10">
                <Spinner label="Connecting to the agent" />
              </div>
            )}
            <ChatItems
              state={state}
              onRetry={() => retry.mutate()}
              retrying={retry.isPending}
              onShowDecisions={() => (setTab('decisions'), setShowWs(true))}
            />
            {ask && <AskPanel ask={ask} onAnswer={onAnswer} busy={reply.isPending} />}
            {reply.isError && (
              <p role="alert" className="pl-11 text-[13px] text-red-700">
                {errorMessage(reply.error)}
              </p>
            )}
            {state.doneModelId && (
              <div className="flex justify-center">
                <Button variant="primary" onClick={() => navigate(`/models/${state.doneModelId}?tab=test`)}>
                  Test this model
                </Button>
              </div>
            )}
            <div className="h-2" />
          </div>
        </div>
        <Composer
          enabled={Boolean(ask) || !running}
          busy={message.isPending}
          placeholder={
            ask
              ? 'Pick an option above, or type your answer…'
              : running
                ? 'The agent is working…'
                : 'Ask about this model, e.g. "which features matter most?"'
          }
          onSend={onSend}
          footer={
            running && !ask ? (
              <span className="flex items-center gap-1">
                <LoaderCircle size={12} className="animate-spin" aria-hidden /> working
              </span>
            ) : null
          }
        />
      </section>
      <Workspace
        state={state}
        tab={tab}
        setTab={setTab}
        className={cx(showWs ? 'fixed inset-y-16 right-0 z-30 flex shadow-2xl' : 'hidden', 'lg:static lg:flex')}
      />
    </div>
  )
}

export default function AgentPage() {
  const { sessionId } = useParams()
  return sessionId ? <SessionView key={sessionId} sessionId={sessionId} /> : <NewSession />
}
