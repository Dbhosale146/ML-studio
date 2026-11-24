// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  ArrowUp,
  Bot,
  Check,
  ChevronDown,
  CircleCheck,
  Clock,
  Database,
  FlaskConical,
  LoaderCircle,
  Mic,
  Paperclip,
  Plus,
  ScrollText,
  Sparkles,
  Terminal,
  Upload,
  Wrench,
} from 'lucide-react'
import { SCENARIOS, SUGGESTED_GOALS, detectScenario } from './scenarios'
import { bestCandidate, buildModel } from './models'
import { useStore } from './store'
import { num, sleep } from '../lib/format'
import { Badge, ConnectorLogo, cx } from './ui'
import StreamText, { streamDuration } from '../features/agent/StreamText'
import { CleaningCard, CriticCard, DeployedCard, DetectionCard, FeaturesCard, FinalCard, LeaderboardCard, ProfileCard, StrategyCard } from './Cards'

const STAGES = [
  { key: 'detect', label: 'Problem detection', hint: 'Understand the goal' },
  { key: 'data', label: 'Data analysis', hint: 'Profile & map tables' },
  { key: 'strategy', label: 'ML strategy', hint: 'Metric & validation' },
  { key: 'prep', label: 'Cleaning & features', hint: 'Fix issues, engineer signals' },
  { key: 'train', label: 'Model search + HPO', hint: 'Train candidates' },
  { key: 'critic', label: 'Critic review', hint: 'Leakage, overfit, residuals' },
  { key: 'final', label: 'Evaluation & explain', hint: 'Select & explain' },
  { key: 'deploy', label: 'Deploy', hint: 'Ship to production' },
]

class Cancelled extends Error {}

const uid = () => Math.random().toString(36).slice(2, 10)
const clock = (ts) => new Date(ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })

/* --------------------------------- Messages ---------------------------------- */

function AgentAvatar() {
  return (
    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-emerald-500 text-white shadow-sm">
      <Sparkles size={15} />
    </span>
  )
}

function ToolCall({ msg }) {
  const [open, setOpen] = useState(false)
  const running = msg.status === 'running'
  const expanded = running || open
  return (
    <div className="animate-fade-up overflow-hidden rounded-lg border border-slate-200 bg-white">
      <button onClick={() => !running && setOpen((v) => !v)} className="flex w-full items-center gap-2.5 px-3 py-2 text-left">
        {running ? (
          <LoaderCircle size={14} className="shrink-0 animate-spin text-indigo-500" />
        ) : (
          <CircleCheck size={14} className="shrink-0 text-emerald-500" />
        )}
        <Terminal size={13} className="shrink-0 text-slate-400" />
        <code className={cx('min-w-0 flex-1 truncate font-mono text-[12px]', running ? 'shimmer-text' : 'text-slate-700')}>{msg.label}</code>
        <span className="shrink-0 font-mono text-[11px] text-slate-400 tabular">{running ? 'running…' : `${(msg.duration / 1000).toFixed(1)}s`}</span>
        {!running && <ChevronDown size={14} className={cx('shrink-0 text-slate-400 transition-transform', open && 'rotate-180')} />}
      </button>
      {expanded && msg.logs.length > 0 && (
        <div className="border-t border-slate-100 bg-slate-900 px-3 py-2 font-mono text-[11px] leading-5 text-slate-300">
          {msg.logs.map((l, i) => (
            <div key={i} className="animate-fade-in truncate">
              <span className="text-slate-500">{String(i + 1).padStart(2, '0')} </span>
              <span className="text-emerald-400">›</span> {l}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Thinking({ text }) {
  return (
    <div className="flex animate-fade-in items-center gap-3">
      <AgentAvatar />
      <div className="flex items-center gap-2.5 rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">
        <span className="dot-typing flex items-center gap-1">
          <span />
          <span />
          <span />
        </span>
        <span className="shimmer-text text-[13px] font-medium">{text}</span>
      </div>
    </div>
  )
}

function QuickReplies({ pending, onPick }) {
  if (!pending?.options) return null
  return (
    <div className="flex animate-fade-up flex-wrap gap-2 pl-11">
      {pending.options.map((o) => (
        <button
          key={o.label}
          onClick={() => onPick(o)}
          className="rounded-full border border-indigo-200 bg-white px-3.5 py-1.5 text-[13px] font-medium text-indigo-700 shadow-sm transition-all hover:-translate-y-px hover:border-indigo-400 hover:bg-indigo-50"
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

function SourcePicker({ pending, onPick, onUpload }) {
  const fileRef = useRef(null)
  return (
    <div className="animate-fade-up pl-11">
      <div className="grid gap-2 sm:grid-cols-2">
        {pending.sources.map((s) => (
          <button
            key={s.id}
            onClick={() => onPick({ label: `Use ${s.name}`, value: s.id, source: s })}
            className={cx(
              'group flex items-center gap-3 rounded-xl border bg-white p-3 text-left shadow-sm transition-all hover:-translate-y-px',
              s.id === pending.recommended
                ? 'border-emerald-300 ring-2 ring-emerald-100 hover:border-emerald-400'
                : 'border-slate-200 hover:border-indigo-300',
            )}
          >
            <ConnectorLogo id={s.id} size={34} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="truncate text-[13px] font-semibold text-slate-900">{s.name}</span>
                {s.id === pending.recommended && <Badge color="emerald">Best match</Badge>}
              </div>
              <div className="truncate text-[11px] text-slate-500">{s.id === pending.recommended ? pending.recommendedDetail : s.detail}</div>
            </div>
          </button>
        ))}
        <button
          onClick={() => fileRef.current?.click()}
          className="flex items-center gap-3 rounded-xl border border-dashed border-slate-300 bg-white/60 p-3 text-left transition-all hover:border-emerald-400 hover:bg-emerald-50/40"
        >
          <span className="grid h-[34px] w-[34px] place-items-center rounded-lg bg-slate-100 text-slate-500">
            <Upload size={16} />
          </span>
          <div>
            <div className="text-[13px] font-semibold text-slate-900">Upload CSV / Excel</div>
            <div className="text-[11px] text-slate-500">or drag a file into the chat</div>
          </div>
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.tsv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onUpload(f)
            e.target.value = ''
          }}
        />
      </div>
    </div>
  )
}

function GoalSuggestions({ onPick }) {
  return (
    <div className="grid animate-fade-up gap-2 pl-11 sm:grid-cols-2">
      {SUGGESTED_GOALS.map((g) => {
        const s = SCENARIOS[g.key]
        return (
          <button
            key={g.key}
            onClick={() => onPick({ label: g.text, text: g.text })}
            className="group rounded-xl border border-slate-200 bg-white p-3 text-left shadow-sm transition-all hover:-translate-y-px hover:border-indigo-300"
          >
            <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-600">{s.problemType.split(' ·')[0]}</div>
            <div className="mt-0.5 text-[13px] font-medium text-slate-800 group-hover:text-slate-900">{g.text}</div>
          </button>
        )
      })}
    </div>
  )
}

/* ------------------------------ Workspace panel ------------------------------ */

function Stepper({ stage, finished }) {
  return (
    <ol className="space-y-0.5">
      {STAGES.map((s, i) => {
        const done = finished || i < stage
        const active = !finished && i === stage
        return (
          <li key={s.key} className={cx('flex items-center gap-3 rounded-lg px-2 py-1.5 transition-colors', active && 'bg-indigo-50')}>
            <span
              className={cx(
                'grid h-6 w-6 shrink-0 place-items-center rounded-full text-[11px] font-bold',
                done ? 'bg-emerald-500 text-white' : active ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-slate-400',
              )}
            >
              {done ? <Check size={13} strokeWidth={3} /> : active ? <LoaderCircle size={13} className="animate-spin" /> : i + 1}
            </span>
            <div className="min-w-0">
              <div className={cx('text-[13px] font-semibold', done ? 'text-slate-900' : active ? 'text-indigo-900' : 'text-slate-400')}>
                {s.label}
              </div>
              {active && <div className="text-[11px] text-indigo-600">{s.hint}</div>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}

function Field({ label, value, mono }) {
  return (
    <div className="border-b border-slate-100 py-2 last:border-0">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{label}</div>
      {value ? (
        <div className={cx('mt-0.5 animate-fade-in text-[13px] font-medium text-slate-900', mono && 'font-mono text-[12px]')}>{value}</div>
      ) : (
        <div className="mt-1 h-3.5 w-28 animate-shimmer rounded bg-gradient-to-r from-slate-100 via-slate-200/70 to-slate-100 bg-[length:200%_100%]" />
      )}
    </div>
  )
}

function Workspace({ stage, finished, blueprint, decisions, experiments, tab, setTab, elapsed }) {
  return (
    <aside className="hidden w-[380px] shrink-0 flex-col border-l border-slate-200 bg-white lg:flex xl:w-[400px]">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-center justify-between">
          <div className="text-[13px] font-semibold text-slate-900">Live workspace</div>
          <span className="flex items-center gap-1 font-mono text-[11px] text-slate-500 tabular">
            <Clock size={12} />
            {elapsed}
          </span>
        </div>
        <div className="mt-3">
          <Stepper stage={stage} finished={finished} />
        </div>
      </div>
      <div className="flex gap-1 border-b border-slate-200 px-3">
        {[
          ['blueprint', 'Blueprint', null],
          ['decisions', 'Decisions', decisions.length],
          ['experiments', 'Experiments', experiments.length],
        ].map(([id, label, count]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={cx(
              '-mb-px flex items-center gap-1.5 border-b-2 px-2.5 py-2.5 text-[12px] font-semibold',
              tab === id ? 'border-emerald-500 text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-800',
            )}
          >
            {label}
            {count != null && <span className="rounded-full bg-slate-100 px-1.5 text-[10px] tabular">{count}</span>}
          </button>
        ))}
      </div>
      <div className="thin-scroll min-h-0 flex-1 overflow-y-auto px-5 py-3">
        {tab === 'blueprint' && (
          <div>
            <Field label="Business goal" value={blueprint.goal} />
            <Field label="Problem type" value={blueprint.problemType} />
            <Field label="Data source" value={blueprint.source} />
            <Field label="Target · entity" value={blueprint.target} mono />
            <Field label="Training rows" value={blueprint.rows} />
            <Field label="Metric" value={blueprint.metric} />
            <Field label="Validation" value={blueprint.validation} />
            <Field label="Business constraint" value={blueprint.setup} />
            <Field label="Features" value={blueprint.features} />
            <Field label="Best model" value={blueprint.best} />
            <Field label="Deployment" value={blueprint.deploy} />
          </div>
        )}
        {tab === 'decisions' &&
          (decisions.length === 0 ? (
            <p className="py-8 text-center text-[13px] text-slate-500">Decisions appear here as the agent makes them.</p>
          ) : (
            <ol className="space-y-2.5">
              {decisions.map((d, i) => (
                <li key={i} className="animate-fade-up rounded-lg border border-slate-200 p-3">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] font-bold text-slate-400">{String(i + 1).padStart(2, '0')}</span>
                    <Badge color="indigo">{d.step}</Badge>
                  </div>
                  <div className="mt-1.5 text-[13px] font-semibold text-slate-900">{d.decision}</div>
                  <div className="mt-0.5 text-[12px] text-slate-600">{d.reason}</div>
                </li>
              ))}
            </ol>
          ))}
        {tab === 'experiments' &&
          (experiments.length === 0 ? (
            <p className="py-8 text-center text-[13px] text-slate-500">Training runs will be tracked here.</p>
          ) : (
            <table className="w-full text-left">
              <thead>
                <tr className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="py-1.5">Run</th>
                  <th className="py-1.5">Model</th>
                  <th className="py-1.5 text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((e) => (
                  <tr key={e.id} className="animate-fade-in border-t border-slate-100">
                    <td className="py-2 font-mono text-[11px] text-slate-500">{e.id}</td>
                    <td className="py-2 text-[12px] font-medium text-slate-800">
                      {e.name}
                      <div className="text-[10px] text-slate-400">round {e.round}</div>
                    </td>
                    <td className={cx('py-2 text-right font-mono text-[12px] tabular', e.best ? 'font-semibold text-emerald-700' : 'text-slate-700')}>
                      {e.display}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
      </div>
    </aside>
  )
}

/* ----------------------------------- Page ------------------------------------ */

export default function Agent() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const store = useStore()
  const storeRef = useRef(store)
  storeRef.current = store

  const [messages, setMessages] = useState([])
  const [pending, setPending] = useState(null)
  const [status, setStatus] = useState(null)
  const [input, setInput] = useState('')
  const [stage, setStage] = useState(0)
  const [finished, setFinished] = useState(false)
  const [blueprint, setBlueprint] = useState({})
  const [decisions, setDecisions] = useState([])
  const [experiments, setExperiments] = useState([])
  const [tab, setTab] = useState('blueprint')
  const [startedAt] = useState(() => Date.now())
  const [now, setNow] = useState(() => Date.now())
  const [sessionId] = useState(() => `SES-${Math.random().toString(36).slice(2, 6).toUpperCase()}-${Math.floor(1000 + Math.random() * 8999)}`)

  const alive = useRef(true)
  const resolver = useRef(null)
  const pendingRef = useRef(null)
  const modelRef = useRef(null)
  const finishedRef = useRef(false)
  const decisionsRef = useRef([])
  const scrollRef = useRef(null)
  const contentRef = useRef(null)
  const stickRef = useRef(true)
  const inputRef = useRef(null)

  // elapsed timer
  useEffect(() => {
    if (finished) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [finished])
  const secs = Math.floor((now - startedAt) / 1000)
  const elapsed = `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`

  // keep the chat pinned to the bottom while content streams in (unless the user scrolled up)
  useEffect(() => {
    const el = scrollRef.current
    const content = contentRef.current
    if (!el || !content) return
    // Programmatic scrolls only move down, so moving up means the user is reading back.
    let lastTop = el.scrollTop
    const onScroll = () => {
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 120) stickRef.current = true
      else if (el.scrollTop < lastTop - 4) stickRef.current = false
      lastTop = el.scrollTop
    }
    el.addEventListener('scroll', onScroll)
    const ro = new ResizeObserver(() => {
      if (stickRef.current) el.scrollTop = el.scrollHeight
    })
    ro.observe(content)
    return () => {
      el.removeEventListener('scroll', onScroll)
      ro.disconnect()
    }
  }, [])

  /* ---- primitives used by the script ---- */
  const push = useCallback((msg) => {
    const id = uid()
    setMessages((ms) => [...ms, { id, at: Date.now(), ...msg }])
    return id
  }, [])
  const update = useCallback((id, patch) => {
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...(typeof patch === 'function' ? patch(m) : patch) } : m)))
  }, [])

  const answer = useCallback((opt) => {
    if (!resolver.current) return
    const r = resolver.current
    resolver.current = null
    pendingRef.current = null
    setPending(null)
    stickRef.current = true
    setMessages((ms) => [
      ...ms,
      {
        id: uid(),
        at: Date.now(),
        role: 'user',
        type: 'text',
        text: opt.label,
        file: opt.file ? { name: opt.file.name, size: opt.file.size } : null,
      },
    ])
    r(opt)
  }, [])

  const submitText = () => {
    const text = input.trim()
    const p = pendingRef.current
    if (!text || !p) return
    setInput('')
    if (p.freeText) return answer({ label: text, text })
    const lower = text.toLowerCase()
    const opts = p.options || (p.sources || []).map((s) => ({ label: `Use ${s.name}`, value: s.id, source: s }))
    const match =
      opts.find((o) => lower.includes(String(o.value).toLowerCase())) ||
      opts.find((o) =>
        o.label
          .toLowerCase()
          .split(/[^a-z0-9]+/)
          .filter((w) => w.length > 3)
          .some((w) => lower.includes(w)),
      ) ||
      (/^(y|yes|ok|okay|sure|go|sounds good|correct|right|yep|do it)\b/.test(lower) ? opts[0] : null) ||
      opts[0]
    answer({ ...match, label: text })
  }

  useEffect(() => {
    alive.current = true

    const wait = async (ms) => {
      await sleep(ms)
      if (!alive.current) throw new Cancelled()
    }
    const think = async (text, ms) => {
      setStatus(text)
      await wait(ms)
      setStatus(null)
    }
    const say = async (text, extra = {}) => {
      setStatus(null)
      push({ role: 'agent', type: 'text', text, stream: true, ...extra })
      await wait(streamDuration(text) + 350)
    }
    const tool = async (label, logs, ms = 1800) => {
      const id = push({ role: 'agent', type: 'tool', label, status: 'running', logs: [] })
      const step = ms / (logs.length + 1)
      for (const l of logs) {
        await wait(step)
        update(id, (m) => ({ logs: [...m.logs, l] }))
      }
      await wait(step)
      update(id, { status: 'done', duration: ms + Math.round(Math.random() * 400) })
      await wait(250)
    }
    const card = (kind, data) => push({ role: 'agent', type: 'card', kind, data })
    const ask = (cfg) =>
      new Promise((res) => {
        if (!alive.current) return
        resolver.current = res
        pendingRef.current = cfg
        setPending(cfg)
        setStatus(null)
        setTimeout(() => inputRef.current?.focus(), 50)
      })
    const addDecision = (d) => {
      decisionsRef.current = [...decisionsRef.current, d]
      setDecisions(decisionsRef.current)
    }
    const stageDecisions = (spec, key) => spec.decisions.filter((d) => d.stage === key).forEach(addDecision)
    const patch = (p) => modelRef.current && storeRef.current.patchModel(modelRef.current.id, p)

    async function trainRound(spec, round) {
      const speeds = [2.2, 1.0, 0.75, 1.1, 0.65]
      let rows = spec.candidates.map((c, i) => ({
        name: c.name,
        family: c.family,
        progress: 0,
        score: null,
        interim: null,
        target: round === 1 ? c.r1 : c.r2,
        prev: round === 2 ? c.r1 : null,
        speed: speeds[i % speeds.length],
      }))
      const id = card('leaderboard', { spec, rows, trials: 0, round, done: false })
      const base = round === 1 ? 38 : 72
      const span = round === 1 ? 26 : 16
      while (rows.some((r) => r.score == null)) {
        await wait(round === 1 ? 420 : 340)
        rows = rows.map((r) => {
          if (r.score != null) return r
          const progress = Math.min(100, r.progress + r.speed * (round === 1 ? 8 : 12) * (0.6 + Math.random() * 0.8))
          const t = progress / 100
          const interim = spec.better === 'lower' ? r.target * (1.25 - 0.25 * t) : r.target * (0.9 + 0.1 * t)
          return { ...r, progress, interim, score: progress >= 100 ? r.target : null }
        })
        const avg = rows.reduce((s, r) => s + r.progress, 0) / rows.length
        update(id, {
          data: {
            spec,
            rows,
            trials: Math.round(spec.hpoTrials * (avg / 100) * (round === 1 ? 0.7 : 0.3) + (round === 2 ? spec.hpoTrials * 0.7 : 0)),
            round,
            done: false,
          },
        })
        patch({ progress: Math.round(base + span * (avg / 100)) })
      }
      update(id, { data: { spec, rows, trials: round === 1 ? Math.round(spec.hpoTrials * 0.7) : spec.hpoTrials, round, done: true } })
      const dir = spec.better === 'lower' ? 1 : -1
      const leader = [...rows].sort((a, b) => dir * (a.score - b.score))[0]
      setExperiments((ex) => [
        ...ex,
        ...rows.map((r, i) => ({
          id: `r${round}-${String(i + 1).padStart(2, '0')}`,
          name: r.name,
          round,
          display: spec.fmt(r.score),
          best: round === 2 && r.name === leader.name,
        })),
      ])
      return leader
    }

    async function run() {
      await think('Starting a new session…', 700)
      await say(
        "Hi Akash! I'm your **AI Data Scientist**. Tell me what you want to predict in plain English. I'll frame the problem, analyse your data, train and critique candidate models, and explain every decision I make along the way.",
      )

      // 1 · goal
      const goalParam = params.get('goal')
      let goal
      if (goalParam) {
        await wait(500)
        push({ role: 'user', type: 'text', text: goalParam })
        goal = goalParam
      } else {
        const a = await ask({ kind: 'goal', freeText: true, placeholder: 'Describe what you want to predict…' })
        goal = a.text
      }
      setBlueprint({ goal })
      setStage(0)

      await think('Reading your goal…', 1000)
      let key = detectScenario(goal)
      await tool(
        'problem_detector.analyze(goal, sources)',
        [
          `parsing intent: "${goal.length > 48 ? goal.slice(0, 48) + '…' : goal}"`,
          'matching against 6 ML problem families',
          'scanning connected sources for candidate target columns',
          key ? `top match: ${SCENARIOS[key].problemType}` : 'no dominant signal: confidence below 0.6',
        ],
        2000,
      )

      if (!key) {
        await say('I want to frame this correctly before touching any data. Which of these is closest to what you need?')
        const a = await ask({
          kind: 'options',
          options: [
            { label: 'Predict a yes / no outcome (like churn)', value: 'churn' },
            { label: 'Forecast a value over time (like demand)', value: 'forecast' },
            { label: 'Estimate a number (like price)', value: 'price' },
            { label: 'Flag rare, unusual events (like fraud)', value: 'fraud' },
          ],
        })
        key = a.value
        addDecision({
          step: 'Clarification',
          decision: `User clarified problem family: ${a.value}`,
          reason: 'Goal wording was ambiguous, so the agent asked instead of guessing.',
        })
      }
      const spec = SCENARIOS[key]

      // create the model record immediately so it shows up live in the catalog & dashboard
      const existing = storeRef.current.models
      const clash = existing.some((m) => m.name === spec.title)
      const monthTag = new Date().toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
      const name = clash ? `${spec.title} · ${monthTag}` : spec.title
      const slug = clash ? `${spec.slug}_v2` : spec.slug
      const model = buildModel(spec, {
        id: `${spec.slug}_${Date.now().toString(36)}`,
        name,
        slug,
        objective: goal,
        status: 'draft',
        stage: 'Problem framing',
        progress: 5,
        createdBy: 'agent',
        predictions30d: 0,
        deployment: 'Not deployed',
        tags: ['built-with-agent'],
        decisions: [],
      })
      model.primary = { name: spec.metricName, display: '—' }
      model.baseline = { name: spec.final.baseline.name, display: '—' }
      modelRef.current = model
      storeRef.current.upsertModel(model)
      storeRef.current.addActivity('agent', `Started building **${name}**: "${goal}"`)

      card('detection', { spec })
      stageDecisions(spec, 'detect')
      setBlueprint((b) => ({ ...b, problemType: spec.problemType }))
      await wait(600)
      await say(
        `This is a **${spec.problemType}** problem (${Math.round(spec.detection.confidence * 100)}% confidence). ${spec.detection.alternatives}`,
      )

      // 2 · data
      setStage(1)
      await say(
        'Which data should I use? I scanned your connected sources and highlighted the one with the best-matching tables. You can also upload a file.',
      )
      const connected = storeRef.current.connectors.filter((c) => c.status === 'connected' && c.id !== 'files')
      const src = await ask({
        kind: 'source',
        sources: connected,
        recommended: spec.source.connectorId,
        recommendedDetail: `${spec.source.database} · ${spec.source.tables
          .map((t) => t.name)
          .slice(0, 3)
          .join(', ')}`,
      })
      let sourceLabel = `${spec.source.connector} · ${spec.source.database}`
      if (src.value === 'upload') {
        const f = src.file
        let header = null
        if (/\.(csv|tsv)$/i.test(f.name)) {
          try {
            const head = await f.slice(0, 4096).text()
            header = head.split(/\r?\n/)[0].split(/[,\t;]/).length
          } catch {
            header = null
          }
        }
        await tool(
          `ingest.upload("${f.name}")`,
          [
            `receiving ${f.name} (${(f.size / 1024).toFixed(0)} KB)`,
            'detecting encoding & delimiter: UTF-8',
            header ? `parsed header: ${header} columns` : 'parsed workbook: 1 sheet',
            'staging to encrypted workspace storage',
          ],
          2000,
        )
        sourceLabel = `${f.name} + ${spec.source.connector}`
        await say(
          `Got **${f.name}**. I mapped its columns to the matching reference tables in ${spec.source.connector}.${spec.source.database} so the agent has full history and labels.`,
        )
        addDecision({
          step: 'Data',
          decision: `Ingested uploaded file ${f.name}`,
          reason: 'User-provided dataset, enriched with warehouse reference tables.',
        })
      } else if (src.value !== spec.source.connectorId) {
        await say(
          `I'll use **${src.source.name}**. The best-matching tables for this goal live in ${spec.source.connector}.${spec.source.database}, so I'll federate the query across both sources.`,
        )
        sourceLabel = `${src.source.name} + ${spec.source.connector}`
      }
      setBlueprint((b) => ({ ...b, source: sourceLabel }))
      patch({ stage: 'Data analysis', progress: 12 })

      await tool(
        `warehouse.connect("${spec.source.connector}.${spec.source.database}")`,
        [
          'authenticating with read-only service role',
          `listing schemas: ${spec.source.tables.length + 9} tables visible`,
          'ranking tables by semantic relevance to the goal',
        ],
        1700,
      )
      await tool(
        'data_profiler.run(tables)',
        [
          ...spec.source.tables.map((t) => `profiling ${t.name}: ${num(t.rows)} rows`),
          'inferring column types & semantic roles',
          'computing missingness, cardinality & distributions',
        ],
        2800,
      )

      let target = spec.dataset.target
      const profileId = card('profile', { spec, target })
      stageDecisions(spec, 'data')
      setBlueprint((b) => ({ ...b, target: `${target} · ${spec.dataset.entity}`, rows: `${num(spec.dataset.rows)} (${spec.dataset.period})` }))
      await say(
        `I found **${num(spec.dataset.rows)} rows** across ${spec.source.tables.length} tables (${spec.dataset.period}). The target looks like \`${target}\` at the \`${spec.dataset.entity}\` grain (${spec.dataset.balance}). Data quality is **${spec.dataset.quality}/100**, and I'll fix the issues next.\n\nShould I use \`${target}\` as the target?`,
      )
      const t = await ask({
        kind: 'options',
        options: [{ label: `Yes, use ${target}`, value: 'yes' }, ...spec.altTargets.map((a) => ({ label: `Use ${a} instead`, value: a }))],
      })
      if (t.value !== 'yes') {
        target = t.value
        update(profileId, { data: { spec, target } })
        setBlueprint((b) => ({ ...b, target: `${target} · ${spec.dataset.entity}` }))
        addDecision({ step: 'Target', decision: `Target changed to ${target}`, reason: 'User override of the detected target column.' })
        await say(`Switched the target to \`${target}\`. The pipeline stays the same, and the change is in the decision log.`)
      } else {
        await say("Target confirmed. Next I'll pick the right metric and validation scheme.")
      }

      // 3 · strategy
      setStage(2)
      patch({ stage: 'ML strategy', progress: 18 })
      await think('Choosing metric & validation strategy…', 1300)
      card('strategy', { spec })
      stageDecisions(spec, 'strategy')
      setBlueprint((b) => ({ ...b, metric: spec.strategy.metric, validation: spec.strategy.validation }))
      await wait(500)
      await say(spec.clarify.question)
      const c = await ask({ kind: 'options', options: spec.clarify.options.map((o) => ({ label: o.label, value: o.label, opt: o })) })
      const chosen = c.opt || spec.clarify.options[0]
      addDecision(chosen.decision)
      setBlueprint((b) => ({ ...b, setup: chosen.label }))
      await say(chosen.reply)

      // 4 · cleaning & features
      setStage(3)
      patch({ status: 'training', stage: 'Cleaning & features', progress: 24 })
      await tool(
        'data_cleaner.apply(profile)',
        spec.cleaning.map((x) => `${x.issue} → fixed (${x.count})`),
        2400,
      )
      card('cleaning', { spec })
      stageDecisions(spec, 'clean')
      await wait(400)
      await tool(
        'feature_engineer.synthesize(target, horizon)',
        [
          'generating temporal, lag & aggregate candidates',
          'LLM proposing domain features from column semantics',
          'point-in-time leakage guard on every feature',
          `kept ${spec.featureCount} features after redundancy pruning`,
        ],
        2600,
      )
      card('features', { spec })
      stageDecisions(spec, 'features')
      setBlueprint((b) => ({ ...b, features: `${spec.featureCount} engineered (${spec.features.filter((f) => f.genai).length} GenAI)` }))
      await say(
        `I engineered **${spec.featureCount} features**. The ones in violet were proposed by the GenAI feature synthesiser, and every one passed the point-in-time leakage guard.`,
      )

      // 5 · model search
      setStage(4)
      patch({ stage: 'Model search + HPO', progress: 34 })
      await say(`Now the model search. I won't brute-force all ${spec.shortlist.considered} model families. ${spec.shortlist.reason}`)
      stageDecisions(spec, 'train')
      setTab('experiments')
      const lead1 = await trainRound(spec, 1)
      await say(
        `Round 1 is done. **${lead1.name}** leads with ${spec.metricName} **${spec.fmt(lead1.score)}**. Before I accept that, the critic agent reviews the results.`,
      )

      // 6 · critic
      setStage(5)
      patch({ stage: 'Critic review', progress: 66 })
      await think('Critic agent reviewing round 1…', 1200)
      await tool(
        'critic.review(round=1)',
        [
          'scanning feature importance for leakage signatures',
          'comparing train vs validation gap per candidate',
          'analysing residuals by segment and time',
          'checking metric alignment with business goal',
        ],
        2600,
      )
      card('critic', { spec })
      stageDecisions(spec, 'critic')
      setTab('decisions')
      await wait(500)
      await say("I've applied the fixes. Retraining with the corrected pipeline (round 2)…")
      setTab('experiments')
      await trainRound(spec, 2)
      await say(spec.critic.after)

      // 7 · final
      setStage(6)
      patch({ stage: 'Final evaluation', progress: 92 })
      await tool(
        'evaluator.final_report(best)',
        [
          'scoring untouched holdout set',
          'computing SHAP values on 2,000-row sample',
          'calibrating & generating prediction intervals',
          'writing plain-language explanation',
        ],
        2200,
      )
      const best = bestCandidate(spec)
      const finalModel = { ...modelRef.current, ...buildModel(spec, { id: modelRef.current.id, name, slug }) }
      card('final', { spec, model: finalModel })
      stageDecisions(spec, 'final')
      setBlueprint((b) => ({ ...b, best: `${spec.final.algorithm} · ${spec.final.primary.name} ${spec.final.primary.display}` }))
      patch({
        status: 'draft',
        stage: 'Ready, awaiting deployment',
        progress: 100,
        decisions: decisionsRef.current,
        candidates: finalModel.candidates,
        primary: finalModel.primary,
        baseline: finalModel.baseline,
      })
      await wait(500)
      await say(
        `**${best.name}** wins with ${spec.final.primary.name} **${spec.final.primary.display}** vs ${spec.final.baseline.display} for the baseline. Ask **"Why this model?"** on the card, or open the decision log on the right.\n\nHow would you like to deploy it?`,
      )

      // 8 · deploy
      setStage(7)
      const d = await ask({
        kind: 'options',
        options: [
          { label: 'Deploy as real-time REST API', value: 'api' },
          { label: 'Schedule daily batch scoring', value: 'batch' },
          { label: 'Keep in staging for review', value: 'staging' },
        ],
      })
      const mode = d.value
      if (mode === 'staging') {
        await tool(
          'registry.register(model, env="staging")',
          ['serialising pipeline + model artifact', 'writing model card & decision log', 'registering version v1.0 in staging'],
          1800,
        )
      } else {
        await tool(
          'deployer.package_and_ship(model)',
          [
            'serialising pipeline + model (ONNX)',
            'building container image: 412 MB',
            'running smoke tests on 500 holdout rows… 500/500 passed',
            mode === 'api' ? 'provisioning autoscaling endpoint (2–8 replicas)' : 'creating schedule: daily 06:00 UTC',
            'health check passed',
          ],
          3200,
        )
      }
      const series = Array.from({ length: 30 }, (_, i) => ({ day: i, value: i === 29 ? 500 : 0 }))
      const deployment = mode === 'api' ? 'REST API · real-time' : mode === 'batch' ? 'Batch · daily 06:00 UTC' : 'REST API · staging'
      const deployed = {
        ...finalModel,
        status: mode === 'staging' ? 'staging' : 'live',
        stage: 'Deployed',
        progress: 100,
        deployment,
        decisions: [
          ...decisionsRef.current,
          { step: 'Deployment', decision: deployment, reason: `Chosen by ${'Akash Kesarkar'} in the agent session.`, alternatives: '' },
        ],
        usage: { ...finalModel.usage, predictions30d: 500, uptime: '100%', series },
        versions: [
          {
            version: 'v1.0',
            date: Date.now(),
            metric: spec.final.primary.display,
            note: 'Initial build by AI Data Scientist',
            status: mode === 'staging' ? 'staging' : 'live',
          },
        ],
        createdBy: 'agent',
        objective: goal,
        tags: ['built-with-agent'],
        createdAt: modelRef.current.createdAt,
        updatedAt: Date.now(),
      }
      storeRef.current.upsertModel(deployed)
      storeRef.current.addActivity('deploy', mode === 'staging' ? `Registered **${name}** in staging` : `Deployed **${name}** (${deployment})`)
      addDecision({ step: 'Deployment', decision: deployment, reason: 'Chosen in the agent session.' })
      setBlueprint((b) => ({ ...b, deploy: deployment }))
      finishedRef.current = true
      setFinished(true)
      card('deployed', { model: deployed, mode })
      await say(
        `Done! **${name}** is ${mode === 'staging' ? 'registered in staging' : 'live'}, and it's already in your Model Catalog and on the dashboard. The model page has the full data profile, metrics, decision log and API snippets.`,
      )
      const next = await ask({
        kind: 'options',
        options: [
          { label: 'Open the model page', value: `/models/${deployed.id}` },
          { label: 'Go to the dashboard', value: '/' },
          { label: 'Build another model', value: 'again' },
        ],
      })
      if (next.value === 'again') navigate('/agent', { replace: false, state: { fresh: Date.now() } })
      else navigate(next.value)
    }

    run().catch((e) => {
      if (!(e instanceof Cancelled)) console.error(e)
    })

    return () => {
      alive.current = false
      // a conversation abandoned mid-way leaves a resumable draft instead of a forever-"training" model
      if (modelRef.current && !finishedRef.current) {
        storeRef.current.patchModel(modelRef.current.id, (m) =>
          m.status === 'live' || m.status === 'staging' ? {} : { status: 'draft', stage: 'Paused: conversation closed' },
        )
      }
    }
    // the script runs exactly once per mounted session
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const onUpload = (f) => answer({ label: `Uploaded ${f.name}`, value: 'upload', file: f })

  const canType = !!pending
  const lastAgentIdx = messages.reduce((idx, m, i) => (m.role === 'agent' ? i : idx), -1)

  return (
    <div className="flex h-[calc(100vh-4rem)] min-h-0">
      <section className="flex min-w-0 flex-1 flex-col">
        {/* chat header */}
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3 lg:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <AgentAvatar />
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate text-[14px] font-semibold text-slate-900">AI Data Scientist</span>
                <span className="flex items-center gap-1 text-[11px] font-medium text-emerald-700">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                  online
                </span>
              </div>
              <div className="truncate font-mono text-[11px] text-slate-500">datalyon-reasoner-5 · Phase 3 · {sessionId}</div>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <span className="hidden items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-[11px] font-medium text-slate-600 md:flex">
              <Database size={12} /> {store.connectors.filter((c) => c.status === 'connected').length} sources
            </span>
            <span className="hidden items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-[11px] font-medium text-slate-600 md:flex">
              <Wrench size={12} /> 9 tools
            </span>
            <button
              onClick={() => navigate('/agent', { state: { fresh: Date.now() } })}
              className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1.5 text-[12px] font-semibold text-slate-700 hover:bg-slate-50"
            >
              <Plus size={14} /> New session
            </button>
          </div>
        </div>

        {/* mobile stage strip */}
        <div className="flex items-center gap-2 border-b border-slate-200 bg-white px-4 py-2 lg:hidden">
          <span className="text-[11px] font-semibold text-slate-500">Step {Math.min(stage + 1, 8)}/8</span>
          <span className="text-[12px] font-semibold text-slate-900">{finished ? 'Complete' : STAGES[stage].label}</span>
          <div className="ml-auto h-1.5 w-24 overflow-hidden rounded-full bg-slate-100">
            <div className="h-full bg-emerald-500 transition-all" style={{ width: `${finished ? 100 : (stage / 8) * 100}%` }} />
          </div>
        </div>

        {/* messages */}
        <div ref={scrollRef} className="thin-scroll min-h-0 flex-1 overflow-y-auto bg-canvas">
          <div ref={contentRef} className="mx-auto max-w-3xl space-y-4 px-4 py-6 lg:px-6">
            <div className="flex justify-center">
              <span className="rounded-full bg-white px-3 py-1 text-[11px] font-medium text-slate-500 ring-1 ring-slate-200">
                Session started {clock(startedAt)} · your data never leaves your workspace
              </span>
            </div>
            {messages.map((m, i) => {
              const prev = messages[i - 1]
              const showAvatar = m.role === 'agent' && (!prev || prev.role !== 'agent')
              if (m.role === 'user') {
                return (
                  <div key={m.id} className="flex animate-fade-up flex-col items-end gap-1">
                    <div className="max-w-[85%] rounded-2xl rounded-tr-md bg-slate-900 px-4 py-2.5 text-[14px] leading-6 text-white shadow-sm">
                      {m.file && (
                        <div className="mb-1.5 flex items-center gap-2 rounded-lg bg-white/10 px-2.5 py-1.5 text-[12px]">
                          <Paperclip size={13} /> {m.file.name} <span className="text-white/60">{(m.file.size / 1024).toFixed(0)} KB</span>
                        </div>
                      )}
                      {m.text}
                    </div>
                    <span className="pr-1 text-[10px] text-slate-400">{clock(m.at)} · Akash</span>
                  </div>
                )
              }
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
                    {m.type === 'text' && (
                      <div className="animate-fade-in text-[14px] leading-6 text-slate-700">
                        <StreamText text={m.text} stream={m.stream} />
                      </div>
                    )}
                    {m.type === 'tool' && <ToolCall msg={m} />}
                    {m.type === 'card' && m.kind === 'detection' && <DetectionCard spec={m.data.spec} />}
                    {m.type === 'card' && m.kind === 'profile' && <ProfileCard spec={m.data.spec} target={m.data.target} />}
                    {m.type === 'card' && m.kind === 'strategy' && <StrategyCard spec={m.data.spec} />}
                    {m.type === 'card' && m.kind === 'cleaning' && <CleaningCard spec={m.data.spec} />}
                    {m.type === 'card' && m.kind === 'features' && <FeaturesCard spec={m.data.spec} />}
                    {m.type === 'card' && m.kind === 'leaderboard' && <LeaderboardCard data={m.data} spec={m.data.spec} />}
                    {m.type === 'card' && m.kind === 'critic' && <CriticCard spec={m.data.spec} />}
                    {m.type === 'card' && m.kind === 'final' && (
                      <FinalCard spec={m.data.spec} model={m.data.model} onShowDecisions={() => setTab('decisions')} />
                    )}
                    {m.type === 'card' && m.kind === 'deployed' && <DeployedCard model={m.data.model} mode={m.data.mode} onNavigate={navigate} />}
                  </div>
                </div>
              )
            })}

            {status && <Thinking text={status} />}
            {pending?.kind === 'goal' && lastAgentIdx >= 0 && <GoalSuggestions onPick={answer} />}
            {pending?.kind === 'options' && <QuickReplies pending={pending} onPick={answer} />}
            {pending?.kind === 'source' && <SourcePicker pending={pending} onPick={answer} onUpload={onUpload} />}
            <div className="h-4" />
          </div>
        </div>

        {/* composer */}
        <div className="border-t border-slate-200 bg-white px-4 py-3 lg:px-6">
          <div className="mx-auto max-w-3xl">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                submitText()
              }}
              className={cx(
                'flex items-end gap-2 rounded-xl border bg-white p-2 transition-all',
                canType
                  ? 'border-slate-300 shadow-sm focus-within:border-emerald-400 focus-within:ring-2 focus-within:ring-emerald-100'
                  : 'border-slate-200 bg-slate-50',
              )}
            >
              <label
                className={cx(
                  'rounded-lg p-2',
                  pending?.kind === 'source' ? 'cursor-pointer text-slate-600 hover:bg-slate-100' : 'cursor-not-allowed text-slate-300',
                )}
                title={pending?.kind === 'source' ? 'Attach a CSV / Excel file' : 'Attachments are available when the agent asks for data'}
              >
                <Paperclip size={18} />
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls,.tsv"
                  className="hidden"
                  disabled={pending?.kind !== 'source'}
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) onUpload(f)
                    e.target.value = ''
                  }}
                />
              </label>
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                disabled={!canType}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    submitText()
                  }
                }}
                placeholder={
                  canType
                    ? pending.placeholder || 'Reply to the agent, or pick an option above…'
                    : finished
                      ? 'Session complete'
                      : 'The agent is working…'
                }
                className="max-h-32 min-h-[36px] flex-1 resize-none bg-transparent px-1 py-2 text-[14px] outline-none placeholder:text-slate-400 disabled:cursor-not-allowed"
              />
              <button
                type="button"
                className="hidden rounded-lg p-2 text-slate-400 hover:bg-slate-100 sm:block"
                title="Voice input"
                aria-label="Voice input"
              >
                <Mic size={18} />
              </button>
              <button
                type="submit"
                disabled={!canType || !input.trim()}
                className="grid h-9 w-9 place-items-center rounded-lg bg-emerald-500 text-white shadow-sm transition-colors hover:bg-emerald-600 disabled:bg-slate-200 disabled:text-slate-400"
                aria-label="Send"
              >
                <ArrowUp size={18} />
              </button>
            </form>
            <div className="mt-1.5 flex items-center justify-between px-1 text-[11px] text-slate-400">
              <span className="flex items-center gap-1">
                {canType || finished ? (
                  <>
                    <Bot size={12} /> Enter to send · Shift + Enter for a new line
                  </>
                ) : (
                  <>
                    <LoaderCircle size={12} className="animate-spin" /> Agent is working. You can reply when it asks you something.
                  </>
                )}
              </span>
              <span className="hidden items-center gap-3 sm:flex">
                <span className="flex items-center gap-1">
                  <ScrollText size={12} /> {decisions.length} decisions
                </span>
                <span className="flex items-center gap-1">
                  <FlaskConical size={12} /> {experiments.length} runs
                </span>
              </span>
            </div>
          </div>
        </div>
      </section>

      <Workspace
        stage={stage}
        finished={finished}
        blueprint={blueprint}
        decisions={decisions}
        experiments={experiments}
        tab={tab}
        setTab={setTab}
        elapsed={elapsed}
      />
    </div>
  )
}
