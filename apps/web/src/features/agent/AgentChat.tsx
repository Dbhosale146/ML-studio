import { ArrowUp, Bot, Check, ChevronDown, CircleAlert, CircleCheck, LoaderCircle, Paperclip, RotateCcw, Sparkles, Terminal } from 'lucide-react'
import { useEffect, useRef, useState, type DragEvent, type ReactNode } from 'react'
import { Badge, Button } from '../../components/ui'
import { errorMessage } from '../../lib/apiClient'
import { cx } from '../../lib/cx'
import { num } from '../../lib/format'
import { useUploadDataset } from '../datasets/hooks'
import { ACCEPT, UploadDropzone } from '../datasets/UploadDropzone'
import StreamText from './StreamText'
import { AgentCard } from './cards'
import type { Ask } from './events'
import type { AgentState, ChatItem } from './reducer'

const clock = (at: number) => new Date(at * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })

export function AgentAvatar() {
  return (
    <span
      className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-emerald-500 text-white shadow-sm"
      aria-hidden
    >
      <Sparkles size={15} />
    </span>
  )
}

function ToolBlock({ item }: { item: Extract<ChatItem, { kind: 'tool' }> }) {
  const [open, setOpen] = useState(false)
  const running = item.status === 'running'
  const expanded = running || open
  return (
    <div className="animate-fade-up overflow-hidden rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        onClick={() => !running && setOpen((v) => !v)}
        aria-expanded={expanded}
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left focus-visible:outline-2 focus-visible:outline-emerald-500"
      >
        {running ? (
          <LoaderCircle size={14} className="shrink-0 animate-spin text-indigo-500" aria-label="Running" />
        ) : item.status === 'error' ? (
          <CircleAlert size={14} className="shrink-0 text-red-500" aria-label="Failed" />
        ) : (
          <CircleCheck size={14} className="shrink-0 text-emerald-500" aria-label="Done" />
        )}
        <Terminal size={13} className="shrink-0 text-slate-400" aria-hidden />
        <code className={cx('min-w-0 flex-1 truncate font-mono text-[12px]', running ? 'shimmer-text' : 'text-slate-700')}>{item.label}</code>
        <span className="tabular shrink-0 font-mono text-[11px] text-slate-400">
          {running ? 'running…' : `${((item.durationMs ?? 0) / 1000).toFixed(1)}s`}
        </span>
        {!running && item.logs.length > 0 && (
          <ChevronDown size={14} className={cx('shrink-0 text-slate-400 transition-transform', open && 'rotate-180')} aria-hidden />
        )}
      </button>
      {expanded && item.logs.length > 0 && (
        <div className="border-t border-slate-100 bg-slate-900 px-3 py-2 font-mono text-[11px] leading-5 text-slate-300">
          {item.logs.map((l, i) => (
            <div key={i} className="animate-fade-in truncate" title={l}>
              <span className="text-slate-500">{String(i + 1).padStart(2, '0')} </span>
              <span className="text-emerald-400">›</span> {l}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Thinking({ text }: { text: string }) {
  return (
    <div className="flex animate-fade-in items-center gap-3" role="status">
      <AgentAvatar />
      <div className="flex items-center gap-2.5 rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
        <span className="dot-typing flex items-center gap-1" aria-hidden>
          <span />
          <span />
          <span />
        </span>
        <span className="shimmer-text text-[13px] font-medium">{text}</span>
      </div>
    </div>
  )
}

export function ChatItems({
  state,
  onRetry,
  retrying,
  onShowDecisions,
}: {
  state: AgentState
  onRetry: () => void
  retrying: boolean
  onShowDecisions: () => void
}) {
  const items = state.items
  return (
    <>
      {items.map((m, i) => {
        if (m.kind === 'user')
          return (
            <div key={m.id} className="flex animate-fade-up flex-col items-end gap-1">
              <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-md bg-slate-900 px-4 py-2.5 text-[14px] leading-6 text-white shadow-sm">
                {m.text}
              </div>
              <span className="pr-1 text-[10px] text-slate-400">{clock(m.at)} · you</span>
            </div>
          )
        const prev = items[i - 1]
        const showAvatar = !prev || prev.kind === 'user'
        let body: ReactNode = null
        if (m.kind === 'agent')
          body = (
            <div className="text-[14px] leading-6 text-slate-700">
              <StreamText text={m.text} />
            </div>
          )
        else if (m.kind === 'tool') body = <ToolBlock item={m} />
        else if (m.kind === 'card') body = <AgentCard kind={m.cardKind} data={m.data} onShowDecisions={onShowDecisions} />
        else if (m.kind === 'error')
          body = (
            <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-3">
              <div className="flex items-start gap-2 text-[13px] text-red-900">
                <CircleAlert size={16} className="mt-0.5 shrink-0" aria-hidden />
                <div className="min-w-0 flex-1">
                  <div className="font-semibold">
                    {m.code === 'cancelled' ? 'Run cancelled' : m.code === 'no_signal' ? 'Stopped: no reliable model' : 'Something went wrong'}
                  </div>
                  <div className="mt-0.5 text-red-800">{m.message}</div>
                </div>
              </div>
              {m.retryable && i === items.length - 1 && (
                <Button size="sm" className="mt-2" icon={RotateCcw} onClick={onRetry} disabled={retrying}>
                  {retrying ? 'Retrying…' : 'Retry from this step'}
                </Button>
              )}
            </div>
          )
        return (
          <div key={m.id} className="flex gap-3">
            <div className="w-8 shrink-0">{showAvatar && <AgentAvatar />}</div>
            <div className="min-w-0 flex-1">
              {showAvatar && (
                <div className="mb-1 flex items-center gap-2 text-[11px]">
                  <span className="font-semibold text-slate-900">AI Data Scientist</span>
                  <span className="text-slate-400">{clock(m.at)}</span>
                </div>
              )}
              {body}
            </div>
          </div>
        )
      })}
      {state.thinking && <Thinking text={state.thinking} />}
    </>
  )
}

function ApprovalPanel({ ask, onSubmit, busy }: { ask: Ask; onSubmit: (value: unknown) => void; busy: boolean }) {
  const items = ask.items as Array<{ id: string; title: string; detail: string; optional: boolean; enabled?: boolean; count?: number }>
  const [enabled, setEnabled] = useState<string[]>(
    Array.isArray(ask.default) ? (ask.default as string[]) : items.filter((i) => i.optional).map((i) => i.id),
  )
  const optional = items.filter((i) => i.optional)
  return (
    <div className="space-y-2 rounded-xl border border-indigo-200 bg-white p-3 shadow-sm">
      {optional.length > 0 && (
        <fieldset>
          <legend className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Optional fixes</legend>
          <div className="space-y-1">
            {optional.map((it) => (
              <label key={it.id} className="flex cursor-pointer items-start gap-2 rounded-md px-2 py-1 text-[13px] hover:bg-slate-50">
                <input
                  type="checkbox"
                  aria-label={it.title}
                  className="mt-1 accent-emerald-600"
                  checked={enabled.includes(it.id)}
                  onChange={(e) => setEnabled((v) => (e.target.checked ? [...v, it.id] : v.filter((x) => x !== it.id)))}
                />
                <span>
                  <span className="font-medium text-slate-900">{it.title}</span>
                  <span className="block text-[12px] text-slate-500">{it.detail}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      )}
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-slate-500">{items.length - optional.length} required step(s) always applied</span>
        <Button
          variant="primary"
          size="sm"
          icon={Check}
          disabled={busy}
          onClick={() => onSubmit(optional.length ? enabled : (ask.options[0]?.value ?? []))}
        >
          {ask.options[0]?.label ?? 'Approve'}
        </Button>
      </div>
    </div>
  )
}

export function AskPanel({ ask, onAnswer, busy }: { ask: Ask; onAnswer: (value: unknown, text?: string) => void; busy: boolean }) {
  if (ask.kind === 'upload')
    return (
      <div className="animate-fade-up pl-11">
        <UploadDropzone onUploaded={(ds) => onAnswer(ds.id, `Uploaded ${ds.file_name}`)} />
      </div>
    )
  if (ask.kind === 'approval')
    return (
      <div className="animate-fade-up pl-11">
        <ApprovalPanel ask={ask} onSubmit={(v) => onAnswer(v)} busy={busy} />
      </div>
    )
  if (ask.kind === 'sheet')
    return (
      <div className="grid animate-fade-up gap-2 pl-11 sm:grid-cols-2" role="group" aria-label="Choose a sheet">
        {ask.options.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            disabled={busy}
            onClick={() => onAnswer(o.value)}
            className={cx(
              'flex items-center justify-between gap-3 rounded-xl border bg-white p-3 text-left shadow-sm transition-all hover:-translate-y-px focus-visible:outline-2 focus-visible:outline-emerald-500',
              o.value === ask.default ? 'border-emerald-300 ring-2 ring-emerald-100' : 'border-slate-200 hover:border-indigo-300',
            )}
          >
            <span className="min-w-0">
              <span className="block truncate text-[13px] font-semibold text-slate-900">{o.label}</span>
              {o.hint && <span className="block text-[11px] text-slate-500">{o.hint}</span>}
            </span>
            {o.value === ask.default && <Badge color="emerald">Largest</Badge>}
          </button>
        ))}
      </div>
    )
  return (
    <div className="flex animate-fade-up flex-wrap gap-2 pl-11" role="group" aria-label="Quick replies">
      {ask.options.map((o, i) => (
        <button
          key={i}
          type="button"
          disabled={busy}
          onClick={() => onAnswer(o.value)}
          className="rounded-full border border-indigo-200 bg-white px-3.5 py-1.5 text-[13px] font-medium text-indigo-700 shadow-sm transition-all hover:-translate-y-px hover:border-indigo-400 hover:bg-indigo-50 focus-visible:outline-2 focus-visible:outline-indigo-500 disabled:opacity-50"
          title={o.hint ?? undefined}
        >
          {o.label}
          {o.hint && <span className="ml-1 text-[11px] text-indigo-400">· {o.hint}</span>}
        </button>
      ))}
    </div>
  )
}

/** Free-text composer. Typed text answers a pending question (the LLM maps it) or asks the agent something. */
export function Composer({
  enabled,
  placeholder,
  onSend,
  onFile,
  busy,
  footer,
}: {
  enabled: boolean
  placeholder: string
  onSend: (text: string) => void
  onFile?: (file: File) => void
  busy: boolean
  footer?: ReactNode
}) {
  const [text, setText] = useState('')
  const [over, setOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const send = () => {
    const t = text.trim()
    if (!t || !enabled) return
    onSend(t)
    setText('')
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f && onFile) onFile(f)
  }
  return (
    <div
      className="border-t border-slate-200 bg-white px-4 py-3 lg:px-6"
      onDragOver={(e) => {
        if (onFile) {
          e.preventDefault()
          setOver(true)
        }
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <div className="mx-auto max-w-3xl">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
          className={cx(
            'flex items-end gap-2 rounded-xl border bg-white p-2 transition-all',
            over
              ? 'border-emerald-400 ring-2 ring-emerald-100'
              : enabled
                ? 'border-slate-300 shadow-sm focus-within:border-emerald-400 focus-within:ring-2 focus-within:ring-emerald-100'
                : 'border-slate-200 bg-slate-50',
          )}
        >
          {onFile && (
            <>
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-100"
                aria-label="Attach an Excel or CSV file"
              >
                <Paperclip size={18} />
              </button>
              <input
                ref={fileRef}
                type="file"
                accept={ACCEPT}
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) onFile(f)
                  e.target.value = ''
                }}
              />
            </>
          )}
          <label className="sr-only" htmlFor="agent-composer">
            Message the agent
          </label>
          <textarea
            id="agent-composer"
            rows={1}
            value={text}
            disabled={!enabled}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            placeholder={over ? 'Drop the file to upload it' : placeholder}
            className="max-h-32 min-h-[36px] flex-1 resize-none bg-transparent px-1 py-2 text-[14px] outline-none placeholder:text-slate-400 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={!enabled || !text.trim() || busy}
            className="grid h-9 w-9 place-items-center rounded-lg bg-emerald-500 text-white shadow-sm transition-colors hover:bg-emerald-600 disabled:bg-slate-200 disabled:text-slate-400"
            aria-label="Send"
          >
            <ArrowUp size={18} />
          </button>
        </form>
        <div className="mt-1.5 flex items-center justify-between px-1 text-[11px] text-slate-400">
          <span className="flex items-center gap-1">
            <Bot size={12} aria-hidden /> Enter to send · Shift + Enter for a new line{onFile ? ' · drop an Excel file here' : ''}
          </span>
          {footer}
        </div>
      </div>
    </div>
  )
}

export function useFileReply(onDataset: (id: string, name: string) => void) {
  const upload = useUploadDataset()
  return {
    upload: (f: File) => upload.mutate(f, { onSuccess: (ds) => onDataset(ds.id, ds.file_name) }),
    pending: upload.isPending,
    error: upload.isError ? errorMessage(upload.error) : null,
  }
}

/** Keep the chat pinned to the bottom while content streams in, unless the user scrolled up to read. */
export function useStickToBottom(dep: unknown) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  useEffect(() => {
    const el = scrollRef.current
    const content = contentRef.current
    if (!el || !content) return
    let lastTop = el.scrollTop
    const onScroll = () => {
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 120) stick.current = true
      else if (el.scrollTop < lastTop - 4) stick.current = false
      lastTop = el.scrollTop
    }
    el.addEventListener('scroll', onScroll)
    const ro = new ResizeObserver(() => {
      if (stick.current) el.scrollTop = el.scrollHeight
    })
    ro.observe(content)
    return () => {
      el.removeEventListener('scroll', onScroll)
      ro.disconnect()
    }
  }, [dep])
  return { scrollRef, contentRef, pin: () => (stick.current = true) }
}

export const fmtCount = num
