// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  Boxes,
  Brain,
  CircleCheck,
  Database,
  FlaskConical,
  Gauge,
  LoaderCircle,
  Plug,
  Plus,
  Rocket,
  Sparkles,
  TrendingUp,
  Upload,
  Wand2,
  X,
  Zap,
} from 'lucide-react'
import { useStore } from './store'
import { compact, num, timeAgo } from '../lib/format'
import { SUGGESTED_GOALS } from './scenarios'
import { Badge, Button, Card, CardHeader, ConnectorLogo, KIND, KindIcon, ProgressBar, Rich, Sparkline, StatusPill, cx } from './ui'
import { UsageChart } from './charts'

function greeting() {
  const h = new Date().getHours()
  return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
}

function AiHero() {
  const navigate = useNavigate()
  const [goal, setGoal] = useState('')
  const start = (text) => navigate(`/agent${text ? `?goal=${encodeURIComponent(text)}` : ''}`)
  return (
    <Card className="relative overflow-hidden p-6 lg:p-7">
      <div className="pointer-events-none absolute -right-24 -top-28 h-96 w-96 rounded-full bg-gradient-to-br from-indigo-300/40 via-emerald-200/30 to-transparent blur-3xl" />
      <div className="pointer-events-none absolute -bottom-28 -left-16 h-72 w-72 rounded-full bg-gradient-to-tr from-sky-200/40 via-indigo-100/40 to-transparent blur-3xl" />
      <div className="relative">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-[11px] font-bold uppercase tracking-wider text-indigo-700 ring-1 ring-indigo-100">
            <Sparkles size={13} className="text-indigo-500" /> AI Data Scientist
          </span>
          <span className="hidden items-center gap-1.5 font-mono text-[11px] text-slate-500 sm:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> Orchestrator · Critic · Explainer agents online
          </span>
        </div>
        <h2 className="mt-4 text-2xl font-semibold tracking-tight text-slate-900 lg:text-[28px]">What would you like to predict today?</h2>
        <p className="mt-1.5 max-w-2xl text-sm text-slate-600">
          Describe your business goal in plain English. The agent detects the problem type, analyses your data, trains and critiques candidate models,
          then explains every decision.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            start(goal.trim())
          }}
          className="mt-5 flex flex-col gap-2 rounded-xl bg-white p-2 shadow-sm ring-1 ring-slate-200 focus-within:ring-2 focus-within:ring-emerald-400 sm:flex-row sm:items-center"
        >
          <div className="flex flex-1 items-center gap-2 pl-2">
            <Brain size={20} className="shrink-0 text-indigo-500" />
            <input
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Predict which customers will churn in the next 30 days"
              className="w-full bg-transparent py-2 text-[14px] outline-none placeholder:text-slate-400"
            />
          </div>
          <Button type="submit" variant="primary" size="lg" icon={Wand2}>
            Create model with AI
          </Button>
        </form>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Try:</span>
          {SUGGESTED_GOALS.map((g) => (
            <button
              key={g.key}
              onClick={() => start(g.text)}
              className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-[12px] text-slate-700 transition-colors hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-800"
            >
              {g.text}
            </button>
          ))}
        </div>
        <div className="mt-6 grid grid-cols-2 gap-3 border-t border-slate-100 pt-5 md:grid-cols-4">
          {[
            [Brain, 'Detects the problem', 'Classification, regression, forecasting, anomaly or clustering'],
            [Database, 'Understands your data', 'Profiles, cleans and engineers features'],
            [FlaskConical, 'Trains & self-critiques', 'Backtests candidates, catches leakage'],
            [Rocket, 'Explains & deploys', 'Decision log, API endpoint in one click'],
          ].map(([Icon, t, d], i) => (
            <div key={t} className="flex gap-2.5">
              <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-slate-900 font-mono text-[11px] font-bold text-white">
                {i + 1}
              </span>
              <div>
                <div className="flex items-center gap-1.5 text-[13px] font-semibold text-slate-900">
                  <Icon size={13} className="text-slate-400" />
                  {t}
                </div>
                <div className="text-xs text-slate-500">{d}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

function useLiveCounter(start, perTick) {
  const [n, setN] = useState(start)
  useEffect(() => {
    const t = setInterval(() => setN((v) => v + Math.round(perTick * (0.4 + Math.random()))), 1200)
    return () => clearInterval(t)
  }, [perTick])
  return n
}

function Kpis({ models }) {
  const live = models.filter((m) => m.status === 'live')
  const training = models.filter((m) => m.status === 'training')
  const pred30 = models.reduce((s, m) => s + (m.usage?.predictions30d || 0), 0)
  const dayFraction = (new Date().getHours() * 60 + new Date().getMinutes()) / 1440
  const today = useLiveCounter(Math.round((pred30 / 30) * dayFraction), Math.max(3, pred30 / 30 / 72000))
  const warnings = models.filter((m) => m.drift?.status === 'warning').length
  const agentBuilt = models.filter((m) => m.createdBy === 'agent').length
  const spark = (k) => Array.from({ length: 12 }, (_, i) => ({ value: 10 + i * k + Math.sin(i * 1.3) * 3 }))

  const cards = [
    {
      label: 'Live models',
      icon: Rocket,
      value: live.length,
      unit: `of ${models.length}`,
      foot: (
        <span className="flex items-center gap-1 text-emerald-700">
          <TrendingUp size={13} /> {agentBuilt ? `+${agentBuilt} built by you today` : '+2 this month'}
        </span>
      ),
      chart: <Sparkline data={spark(1.2)} />,
    },
    {
      label: 'Predictions today',
      icon: Zap,
      value: num(today),
      unit: 'live',
      foot: <span className="text-slate-500">{compact(pred30)} in the last 30 days · avg 31 ms</span>,
      chart: <Sparkline data={spark(2)} color="#6366f1" />,
    },
    {
      label: 'Model health',
      icon: Gauge,
      value: `${models.length - warnings}/${models.length}`,
      unit: 'healthy',
      foot: warnings ? (
        <span className="flex items-center gap-1 text-amber-700">
          <AlertTriangle size={13} /> {warnings} drift warning{warnings > 1 ? 's' : ''}
        </span>
      ) : (
        <span className="text-emerald-700">No drift detected</span>
      ),
    },
    {
      label: 'Agent jobs running',
      icon: Activity,
      value: training.length,
      unit: training.length === 1 ? 'job' : 'jobs',
      foot: training[0] ? (
        <div className="w-full">
          <div className="mb-1 flex justify-between text-[11px] text-slate-500">
            <span className="truncate">{training[0].name}</span>
            <span className="tabular font-mono">{training[0].progress}%</span>
          </div>
          <ProgressBar value={training[0].progress} animated />
        </div>
      ) : (
        <span className="text-slate-500">Idle · ready for a new goal</span>
      ),
    },
  ]
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map((c) => (
        <Card key={c.label} className="flex flex-col p-4">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{c.label}</span>
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-slate-50 text-slate-500 ring-1 ring-slate-100">
              <c.icon size={16} />
            </span>
          </div>
          <div className="mt-2 flex items-baseline gap-1.5">
            <span className="text-[28px] font-semibold tracking-tight text-slate-900 tabular">{c.value}</span>
            <span className="text-sm text-slate-500">{c.unit}</span>
          </div>
          {c.chart && <div className="mt-1">{c.chart}</div>}
          <div className="mt-auto pt-2 text-xs">{c.foot}</div>
        </Card>
      ))}
    </div>
  )
}

function ModelsTable({ models }) {
  const navigate = useNavigate()
  const rows = [...models].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 7)
  return (
    <Card className="overflow-hidden">
      <CardHeader
        title="Model status"
        subtitle="Most recently updated models in this workspace"
        icon={Boxes}
        right={
          <Link to="/models" className="flex items-center gap-1 text-[13px] font-semibold text-emerald-700 hover:text-emerald-800">
            View catalog <ArrowRight size={14} />
          </Link>
        }
      />
      <div className="thin-scroll overflow-x-auto">
        <table className="w-full min-w-[640px] text-left">
          <thead>
            <tr className="border-y border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              <th className="px-5 py-2.5">Model</th>
              <th className="px-3 py-2.5">Type</th>
              <th className="px-3 py-2.5 text-right">Key metric</th>
              <th className="px-3 py-2.5">Status</th>
              <th className="px-3 py-2.5">Health</th>
              <th className="px-5 py-2.5 text-right">Updated</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr
                key={m.id}
                onClick={() => navigate(`/models/${m.id}`)}
                className="cursor-pointer border-b border-slate-100 transition-colors last:border-0 hover:bg-slate-50"
              >
                <td className="px-5 py-3">
                  <div className="flex items-center gap-3">
                    <KindIcon kind={m.kind} size="sm" />
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5 truncate text-[13px] font-semibold text-slate-900">
                        {m.name}
                        {m.createdBy === 'agent' && <Badge color="indigo">New</Badge>}
                      </div>
                      <div className="truncate font-mono text-[11px] text-slate-500">
                        {m.slug} · {m.version}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-3 py-3 text-[12px] text-slate-600">{KIND[m.kind]?.label}</td>
                <td className="px-3 py-3 text-right">
                  <div className="font-mono text-[13px] font-semibold text-slate-900 tabular">{m.primary.display}</div>
                  <div className="text-[11px] text-slate-500">{m.primary.name}</div>
                </td>
                <td className="px-3 py-3">
                  <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
                </td>
                <td className="px-3 py-3 text-[12px]">
                  {m.status === 'training' || m.status === 'draft' ? (
                    <span className="text-slate-400">{m.stage}</span>
                  ) : m.drift?.status === 'warning' ? (
                    <span className="flex items-center gap-1 text-amber-700">
                      <AlertTriangle size={13} /> Drift
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-emerald-700">
                      <CircleCheck size={13} /> Healthy
                    </span>
                  )}
                </td>
                <td className="whitespace-nowrap px-5 py-3 text-right text-[12px] text-slate-500">{timeAgo(m.updatedAt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

const NEW_CONNECTORS = [
  { id: 'databricks', name: 'Databricks', detail: 'Unity Catalog · lakehouse', tables: 31, volume: '840M rows' },
  { id: 'redshift', name: 'Amazon Redshift', detail: 'analytics-cluster', tables: 22, volume: '510M rows' },
  { id: 'mysql', name: 'MySQL', detail: 'app-primary (read replica)', tables: 14, volume: '22M rows' },
  { id: 'gsheets', name: 'Google Sheets', detail: 'Finance planning sheets', tables: 4, volume: '9 sheets' },
]

function ConnectModal({ onClose }) {
  const { connectors, upsertConnector, addActivity } = useStore()
  const [uploading, setUploading] = useState(null)
  const connect = (c) => {
    upsertConnector({ ...c, status: 'syncing', lastSync: Date.now() })
    addActivity('data', `Connecting **${c.name}**: schema discovery started`)
    setTimeout(() => {
      upsertConnector({ id: c.id, status: 'connected', lastSync: Date.now(), error: undefined })
      addActivity('data', `**${c.name}** connected: ${c.tables} tables discovered`)
    }, 2800)
    onClose()
  }
  const available = NEW_CONNECTORS.filter((n) => !connectors.some((c) => c.id === n.id))
  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4">
      <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-[2px]" onClick={onClose} />
      <Card className="relative w-full max-w-lg animate-fade-up p-0">
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
          <div>
            <div className="text-[15px] font-semibold">Add a data source</div>
            <div className="text-xs text-slate-500">Read-only credentials · encrypted at rest · no training on your data</div>
          </div>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100" aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="grid gap-2 p-4 sm:grid-cols-2">
          {available.map((c) => (
            <button
              key={c.id}
              onClick={() => connect(c)}
              className="flex items-center gap-3 rounded-lg border border-slate-200 p-3 text-left transition-colors hover:border-emerald-300 hover:bg-emerald-50/50"
            >
              <ConnectorLogo id={c.id} size={34} />
              <div className="min-w-0">
                <div className="text-[13px] font-semibold">{c.name}</div>
                <div className="truncate text-[11px] text-slate-500">{c.detail}</div>
              </div>
            </button>
          ))}
          <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-dashed border-slate-300 p-3 transition-colors hover:border-emerald-400 hover:bg-emerald-50/50 sm:col-span-2">
            <span className="grid h-[34px] w-[34px] place-items-center rounded-lg bg-slate-100 text-slate-500">
              <Upload size={16} />
            </span>
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-semibold">{uploading ? `Uploading ${uploading}…` : 'Upload CSV or Excel'}</div>
              <div className="text-[11px] text-slate-500">Drop a file up to 2 GB, profiled automatically</div>
            </div>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (!f) return
                setUploading(f.name)
                setTimeout(() => {
                  const files = connectors.find((c) => c.id === 'files')
                  upsertConnector({
                    id: 'files',
                    tables: (files?.tables || 0) + 1,
                    volume: `${(files?.tables || 0) + 1} files`,
                    lastSync: Date.now(),
                    status: 'connected',
                  })
                  addActivity('data', `Uploaded **${f.name}** (${(f.size / 1024).toFixed(0)} KB) to CSV / Excel uploads`)
                  onClose()
                }, 1200)
              }}
            />
          </label>
        </div>
      </Card>
    </div>
  )
}

function Connectors() {
  const { connectors, upsertConnector, addActivity } = useStore()
  const [open, setOpen] = useState(false)
  const reconnect = (c) => {
    upsertConnector({ id: c.id, status: 'syncing', error: undefined })
    setTimeout(() => {
      upsertConnector({ id: c.id, status: 'connected', lastSync: Date.now() })
      addActivity('data', `**${c.name}** re-authenticated and synced`)
    }, 2400)
  }
  const ok = connectors.filter((c) => c.status === 'connected').length
  return (
    <Card className="flex flex-col">
      <CardHeader
        title="Data sources"
        subtitle={`${ok} of ${connectors.length} connected`}
        icon={Plug}
        right={
          <Button size="sm" icon={Plus} onClick={() => setOpen(true)}>
            Add
          </Button>
        }
      />
      <div className="divide-y divide-slate-100 border-t border-slate-100">
        {connectors.map((c) => (
          <div key={c.id} className="flex items-center gap-3 px-5 py-3">
            <ConnectorLogo id={c.id} size={34} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-[13px] font-semibold text-slate-900">{c.name}</span>
              </div>
              <div className="truncate text-[11px] text-slate-500">{c.detail}</div>
              {c.status === 'syncing' && <ProgressBar value={62} animated className="mt-1.5" />}
            </div>
            <div className="shrink-0 text-right">
              {c.status === 'connected' && (
                <Badge color="emerald">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                  Connected
                </Badge>
              )}
              {c.status === 'syncing' && (
                <Badge color="amber">
                  <LoaderCircle size={10} className="animate-spin" />
                  Syncing
                </Badge>
              )}
              {c.status === 'error' && (
                <button
                  onClick={() => reconnect(c)}
                  className="rounded-md border border-red-200 bg-red-50 px-1.5 py-0.5 text-[11px] font-semibold text-red-700 hover:bg-red-100"
                >
                  Reconnect
                </button>
              )}
              <div className="mt-1 text-[11px] text-slate-500">{c.status === 'error' ? c.error : `${c.tables} tables · ${timeAgo(c.lastSync)}`}</div>
            </div>
          </div>
        ))}
      </div>
      {open && <ConnectModal onClose={() => setOpen(false)} />}
    </Card>
  )
}

const ACTIVITY_ICON = {
  train: [FlaskConical, 'bg-amber-50 text-amber-600'],
  improve: [TrendingUp, 'bg-emerald-50 text-emerald-600'],
  warn: [AlertTriangle, 'bg-red-50 text-red-600'],
  deploy: [Rocket, 'bg-indigo-50 text-indigo-600'],
  data: [Database, 'bg-sky-50 text-sky-600'],
  agent: [Sparkles, 'bg-indigo-50 text-indigo-600'],
}

function ActivityFeed() {
  const { activity } = useStore()
  return (
    <Card>
      <CardHeader title="Agent activity" subtitle="What your AI Data Scientist has been doing" icon={Activity} />
      <ol className="relative space-y-0 border-t border-slate-100 px-5 py-3">
        {activity.slice(0, 7).map((a, i) => {
          const [Icon, tint] = ACTIVITY_ICON[a.type] || ACTIVITY_ICON.agent
          return (
            <li key={a.id} className={cx('relative flex gap-3 py-2.5', i === 0 && 'animate-fade-up')}>
              <span className={cx('z-10 grid h-7 w-7 shrink-0 place-items-center rounded-full ring-4 ring-white', tint)}>
                <Icon size={14} />
              </span>
              {i < Math.min(activity.length, 7) - 1 && <span className="absolute left-[13px] top-9 bottom-[-10px] w-px bg-slate-200" />}
              <div className="min-w-0 pt-0.5">
                <div className="text-[13px] leading-5 text-slate-700">
                  <Rich text={a.text} />
                </div>
                <div className="text-[11px] text-slate-400">{timeAgo(a.at)}</div>
              </div>
            </li>
          )
        })}
      </ol>
    </Card>
  )
}

function PredictionVolume({ models }) {
  const series = useMemo(() => {
    const out = Array.from({ length: 30 }, (_, day) => ({ day, value: 0 }))
    models.forEach((m) =>
      m.usage?.series?.forEach((p) => {
        out[p.day].value += p.value
      }),
    )
    return out
  }, [models])
  const top = [...models]
    .filter((m) => m.usage?.predictions30d)
    .sort((a, b) => b.usage.predictions30d - a.usage.predictions30d)
    .slice(0, 3)
  return (
    <Card>
      <CardHeader title="Prediction volume" subtitle="All deployed models · last 30 days" icon={ArrowUpRight} />
      <div className="px-3">
        <UsageChart series={series} height={190} />
      </div>
      <div className="grid grid-cols-3 gap-2 border-t border-slate-100 px-5 py-3">
        {top.map((m) => (
          <Link to={`/models/${m.id}`} key={m.id} className="min-w-0 rounded-lg p-1.5 hover:bg-slate-50">
            <div className="truncate text-[11px] text-slate-500">{m.name}</div>
            <div className="font-mono text-sm font-semibold tabular">{compact(m.usage.predictions30d)}</div>
          </Link>
        ))}
      </div>
    </Card>
  )
}

export default function Dashboard() {
  const { models } = useStore()
  const navigate = useNavigate()
  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-[26px] font-semibold tracking-tight text-slate-900">{greeting()}, Akash</h1>
            <Badge color="emerald">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
              Production workspace
            </Badge>
          </div>
          <p className="mt-1 text-sm text-slate-600">Here's how your models, data and AI agent are doing right now.</p>
        </div>
        <div className="flex gap-2">
          <Button icon={Boxes} onClick={() => navigate('/models')}>
            Model catalog
          </Button>
          <Button variant="primary" icon={Plus} onClick={() => navigate('/agent')}>
            Create new model
          </Button>
        </div>
      </div>

      <AiHero />
      <Kpis models={models} />

      <div className="grid gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <ModelsTable models={models} />
        </div>
        <Connectors />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <ActivityFeed />
        <PredictionVolume models={models} />
      </div>
    </div>
  )
}
