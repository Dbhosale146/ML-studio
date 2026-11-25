import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Boxes,
  Brain,
  CircleCheck,
  Database,
  FileSpreadsheet,
  FlaskConical,
  PlayCircle,
  Plus,
  Rocket,
  Sparkles,
  Wand2,
} from 'lucide-react'
import { useState, type ComponentType } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Badge, Button, Card, CardHeader, Empty, KindIcon, ProgressBar, Rich, Spinner, StatusPill, type BadgeColor } from '../components/ui'
import { useActivity, useDatasets, useHealth } from '../features/datasets/hooks'
import { UploadDropzone } from '../features/datasets/UploadDropzone'
import { PROBLEM_KIND, useModels } from '../features/models/hooks'
import { cx } from '../lib/cx'
import { num, timeAgo } from '../lib/format'

function greeting() {
  const h = new Date().getHours()
  return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
}

function GoalBox() {
  const navigate = useNavigate()
  const [goal, setGoal] = useState('')
  return (
    <Card className="relative overflow-hidden p-6 lg:p-7">
      <div
        className="pointer-events-none absolute -right-24 -top-28 h-96 w-96 rounded-full bg-gradient-to-br from-indigo-300/40 via-emerald-200/30 to-transparent blur-3xl"
        aria-hidden
      />
      <div className="relative">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-[11px] font-bold uppercase tracking-wider text-indigo-700 ring-1 ring-indigo-100">
          <Sparkles size={13} aria-hidden /> AI Data Scientist
        </span>
        <h2 className="mt-4 text-2xl font-semibold tracking-tight text-slate-900 lg:text-[28px]">What would you like to predict today?</h2>
        <p className="mt-1.5 max-w-2xl text-sm text-slate-600">
          Describe your goal and upload an Excel file. The agent frames the problem, cleans the data, trains and critiques models, and publishes one
          you can test right away.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            navigate(`/agent${goal.trim() ? `?goal=${encodeURIComponent(goal.trim())}` : ''}`)
          }}
          className="mt-5 flex flex-col gap-2 rounded-xl bg-white p-2 shadow-sm ring-1 ring-slate-200 focus-within:ring-2 focus-within:ring-emerald-400 sm:flex-row sm:items-center"
        >
          <div className="flex flex-1 items-center gap-2 pl-2">
            <Brain size={20} className="shrink-0 text-indigo-500" aria-hidden />
            <label htmlFor="dash-goal" className="sr-only">
              Your goal
            </label>
            <input
              id="dash-goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Predict which customers will churn next month"
              className="w-full bg-transparent py-2 text-[14px] outline-none placeholder:text-slate-400"
            />
          </div>
          <Button type="submit" variant="primary" size="lg" icon={Wand2}>
            Create new model
          </Button>
        </form>
        <div className="mt-6 grid grid-cols-2 gap-3 border-t border-slate-100 pt-5 md:grid-cols-4">
          {(
            [
              [FileSpreadsheet, 'Upload Excel', 'Messy sheets are cleaned up'],
              [Brain, 'Frame the problem', 'Target, metric and validation'],
              [FlaskConical, 'Train & critique', 'HPO, leakage and baseline checks'],
              [PlayCircle, 'Test it', 'Form, batch Excel or forecast'],
            ] as const
          ).map(([Icon, t, d], i) => (
            <div key={t} className="flex gap-2.5">
              <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-slate-900 font-mono text-[11px] font-bold text-white">
                {i + 1}
              </span>
              <div>
                <div className="flex items-center gap-1.5 text-[13px] font-semibold text-slate-900">
                  <Icon size={13} className="text-slate-400" aria-hidden />
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

const ACT: Record<string, [ComponentType<{ size?: number }>, BadgeColor]> = {
  data: [Database, 'sky'],
  agent: [Sparkles, 'indigo'],
  ready: [Rocket, 'emerald'],
  predict: [PlayCircle, 'amber'],
  model: [Boxes, 'slate'],
}
const TINT: Record<BadgeColor, string> = {
  sky: 'bg-sky-50 text-sky-600',
  indigo: 'bg-indigo-50 text-indigo-600',
  emerald: 'bg-emerald-50 text-emerald-600',
  amber: 'bg-amber-50 text-amber-600',
  slate: 'bg-slate-100 text-slate-600',
  red: 'bg-red-50 text-red-600',
  dark: 'bg-slate-900 text-white',
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const health = useHealth()
  const models = useModels()
  const datasets = useDatasets()
  const activity = useActivity(8)
  const items = models.data?.items ?? []
  const ready = items.filter((m) => m.status === 'ready').length
  const training = items.filter((m) => m.status === 'training')
  const predictions = items.reduce((s, m) => s + (m.prediction_count ?? 0), 0)
  const kpis = [
    { label: 'Models', value: num(items.length), sub: `${items.filter((m) => m.status === 'failed').length} failed`, icon: Boxes },
    { label: 'Ready to test', value: num(ready), sub: training.length ? `${training.length} training now` : 'none training', icon: CircleCheck },
    { label: 'Datasets', value: num(datasets.data?.total ?? 0), sub: 'Excel & CSV uploads', icon: Database },
    { label: 'Playground predictions', value: num(predictions), sub: 'single + batch rows', icon: PlayCircle },
  ]
  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
        <div>
          <h1 className="text-[26px] font-semibold tracking-tight text-slate-900">
            {greeting()}
            {health.data?.display_name && health.data.display_name !== 'there' ? `, ${health.data.display_name}` : ''}
          </h1>
          <p className="mt-1 text-sm text-slate-600">Your models, data and AI agent at a glance.</p>
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
      <GoalBox />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {kpis.map((k) => (
          <Card key={k.label} className="p-4">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{k.label}</span>
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-slate-50 text-slate-500 ring-1 ring-slate-100">
                <k.icon size={16} aria-hidden />
              </span>
            </div>
            <div className="tabular mt-2 text-[28px] font-semibold tracking-tight text-slate-900">{models.isLoading ? '…' : k.value}</div>
            <div className="text-xs text-slate-500">{k.sub}</div>
          </Card>
        ))}
      </div>
      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="overflow-hidden xl:col-span-2">
          <CardHeader
            title="Model status"
            subtitle="Live from the agent"
            icon={Boxes}
            right={
              <Link to="/models" className="flex items-center gap-1 text-[13px] font-semibold text-emerald-700 hover:text-emerald-800">
                View catalog <ArrowRight size={14} aria-hidden />
              </Link>
            }
          />
          {models.isLoading ? (
            <div className="p-5">
              <Spinner />
            </div>
          ) : items.length === 0 ? (
            <Empty
              icon={Sparkles}
              title="No models yet"
              text="Start with a goal above, or pick a sample dataset on the Create New Model page."
              action={
                <Button variant="primary" icon={Plus} onClick={() => navigate('/agent')}>
                  Create your first model
                </Button>
              }
            />
          ) : (
            <ul className="divide-y divide-slate-100 border-t border-slate-100">
              {items.slice(0, 7).map((m) => (
                <li key={m.id}>
                  <Link
                    to={`/models/${m.id}`}
                    className="flex items-center gap-3 px-5 py-3 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-emerald-500"
                  >
                    <KindIcon kind={PROBLEM_KIND[m.problem_type]} size="sm" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-semibold text-slate-900">{m.name}</div>
                      {m.status === 'training' ? (
                        <div className="mt-1 flex items-center gap-2">
                          <ProgressBar value={m.progress} animated className="max-w-[220px]" label={`${m.name} progress`} />
                          <span className="truncate text-[11px] text-slate-500">{m.stage}</span>
                        </div>
                      ) : (
                        <div className="truncate text-[11px] text-slate-500">
                          {m.problem_label} · {m.target}
                        </div>
                      )}
                    </div>
                    <div className="hidden text-right sm:block">
                      <div className="tabular font-mono text-[13px] font-semibold text-slate-900">{m.primary_metric?.display ?? '—'}</div>
                      <div className="text-[11px] text-slate-500">{m.primary_metric?.name}</div>
                    </div>
                    <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
                    <span className="hidden w-20 text-right text-[11px] text-slate-500 md:block">{timeAgo(m.updated_at)}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card>
          <CardHeader
            title="Datasets"
            subtitle={`${datasets.data?.total ?? 0} uploaded`}
            icon={Database}
            right={
              <Link to="/datasets" className="text-[13px] font-semibold text-emerald-700 hover:text-emerald-800">
                All
              </Link>
            }
          />
          <div className="space-y-2 px-5 pb-5">
            <UploadDropzone compact onUploaded={(ds) => navigate(`/datasets/${ds.id}`)} />
            {(datasets.data?.items ?? []).slice(0, 5).map((d) => (
              <Link
                key={d.id}
                to={`/datasets/${d.id}`}
                className="flex items-center gap-3 rounded-lg border border-slate-200 p-2.5 hover:border-slate-300 focus-visible:outline-2 focus-visible:outline-emerald-500"
              >
                <FileSpreadsheet size={16} className="shrink-0 text-emerald-600" aria-hidden />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-semibold text-slate-900">{d.file_name}</span>
                  <span className="block truncate text-[11px] text-slate-500">
                    {d.status === 'ready' ? `${num(d.rows)} rows · quality ${d.quality}` : 'choose a sheet'} · {timeAgo(d.created_at)}
                  </span>
                </span>
              </Link>
            ))}
          </div>
        </Card>
      </div>
      <Card>
        <CardHeader title="Recent activity" subtitle="What happened in this workspace" icon={Activity} />
        {(activity.data ?? []).length === 0 ? (
          <p className="px-5 pb-5 text-[13px] text-slate-500">Nothing yet. Uploads, agent runs and predictions show up here.</p>
        ) : (
          <ol className="border-t border-slate-100 px-5 py-3">
            {(activity.data ?? []).map((a) => {
              const [Icon, color] = ACT[a.type] ?? [AlertTriangle, 'slate' as BadgeColor]
              return (
                <li key={a.id} className="flex gap-3 py-2">
                  <span className={cx('grid h-7 w-7 shrink-0 place-items-center rounded-full', TINT[color])}>
                    <Icon size={14} />
                  </span>
                  <div className="min-w-0 pt-0.5">
                    <div className="text-[13px] leading-5 text-slate-700">
                      <Rich text={a.text} />
                    </div>
                    <div className="text-[11px] text-slate-400">{timeAgo(a.created_at)}</div>
                  </div>
                  {a.model_id && (
                    <Link to={`/models/${a.model_id}`} className="ml-auto self-center text-[12px] font-semibold text-emerald-700 hover:underline">
                      Open
                    </Link>
                  )}
                </li>
              )
            })}
          </ol>
        )}
      </Card>
      {health.isError && (
        <Badge color="red" className="hidden">
          offline
        </Badge>
      )}
    </div>
  )
}
