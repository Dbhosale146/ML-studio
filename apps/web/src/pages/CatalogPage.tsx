import { AlertTriangle, Boxes, Database, LayoutGrid, List, Plus, Search, Zap } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { Badge, Button, Card, Empty, KindIcon, ProgressBar, Spinner, StatusPill } from '../components/ui'
import { PROBLEM_KIND, useModels, type ModelSummary } from '../features/models/hooks'
import { errorMessage } from '../lib/apiClient'
import { cx } from '../lib/cx'
import { num, timeAgo } from '../lib/format'

const TYPES = [
  ['', 'All types'],
  ['binary', 'Binary'],
  ['multiclass', 'Multiclass'],
  ['regression', 'Regression'],
  ['forecasting', 'Forecasting'],
] as const
const STATUSES = ['', 'ready', 'training', 'draft', 'failed'] as const

function ModelCard({ m }: { m: ModelSummary }) {
  return (
    <Link to={`/models/${m.id}`} className="group rounded-xl focus-visible:outline-2 focus-visible:outline-emerald-500">
      <Card className="flex h-full flex-col p-5 transition-all duration-150 group-hover:-translate-y-0.5 group-hover:border-emerald-300 group-hover:shadow-[0_8px_24px_rgba(15,23,42,0.08)]">
        <div className="flex items-start justify-between gap-3">
          <KindIcon kind={PROBLEM_KIND[m.problem_type]} />
          <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
        </div>
        <h3 className="mt-3 truncate text-[15px] font-semibold tracking-tight text-slate-900 group-hover:text-emerald-800">{m.name}</h3>
        <div className="mt-0.5 truncate font-mono text-[11px] text-slate-500">
          {m.target ?? '—'}
          {m.latest_version ? ` · v${m.latest_version}` : ''}
        </div>
        <p className="mt-2 line-clamp-2 min-h-[2.5rem] text-[13px] text-slate-600">{m.objective}</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Badge>{m.problem_label}</Badge>
          {m.algorithm && (
            <Badge mono color="slate">
              {m.algorithm}
            </Badge>
          )}
        </div>
        {m.primary_metric ? (
          <div className="mt-4 flex items-end justify-between gap-3 rounded-lg bg-slate-50 p-3">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{m.primary_metric.name}</div>
              <div className="tabular text-[22px] font-semibold tracking-tight text-slate-900">{m.primary_metric.display}</div>
            </div>
            <div className="text-right text-[11px] text-slate-500">baseline {m.baseline_metric?.display ?? '—'}</div>
          </div>
        ) : m.status === 'training' ? (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-[11px] text-slate-500">
              <span>{m.stage}</span>
              <span className="tabular font-mono">{m.progress}%</span>
            </div>
            <ProgressBar value={m.progress} animated label="Training progress" />
          </div>
        ) : m.status === 'failed' ? (
          <div className="mt-4 flex gap-1.5 rounded-lg bg-red-50 p-2.5 text-[12px] text-red-800">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" aria-hidden />
            <span className="line-clamp-2">{m.error}</span>
          </div>
        ) : (
          <div className="mt-4 rounded-lg bg-slate-50 p-3 text-[12px] text-slate-500">{m.stage ?? 'Not trained yet'}</div>
        )}
        <div className="mt-auto flex items-center justify-between gap-2 pt-4 text-[11px] text-slate-500">
          <span className="flex min-w-0 items-center gap-1 truncate">
            <Database size={12} aria-hidden />
            {m.dataset_name ?? 'dataset deleted'}
          </span>
          <span className="flex shrink-0 items-center gap-2">
            {m.prediction_count > 0 && (
              <span className="flex items-center gap-0.5">
                <Zap size={12} aria-hidden />
                {num(m.prediction_count)}
              </span>
            )}
            {timeAgo(m.updated_at)}
          </span>
        </div>
      </Card>
    </Link>
  )
}

export default function CatalogPage() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [view, setView] = useState<'grid' | 'list'>(() => (localStorage.getItem('catalog-view') as 'grid' | 'list') ?? 'grid')
  const q = params.get('q') ?? ''
  const type = params.get('type') ?? ''
  const status = params.get('status') ?? ''
  const models = useModels({ q, problem_type: type, status })
  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params)
    if (v) next.set(k, v)
    else next.delete(k)
    setParams(next, { replace: true })
  }
  const setViewMode = (v: 'grid' | 'list') => {
    setView(v)
    try {
      localStorage.setItem('catalog-view', v)
    } catch {
      // storage unavailable
    }
  }
  const items = models.data?.items ?? []

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
        <div>
          <h1 className="text-[26px] font-semibold tracking-tight text-slate-900">Model Catalog</h1>
          <p className="mt-1 text-sm text-slate-600">Every model the agent built, with its data, metrics, decisions and a place to test it.</p>
        </div>
        <Button variant="primary" icon={Plus} onClick={() => navigate('/agent')}>
          Create new model
        </Button>
      </div>
      <Card className="p-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 focus-within:border-emerald-400 focus-within:ring-2 focus-within:ring-emerald-100">
            <Search size={15} className="text-slate-400" aria-hidden />
            <input
              value={q}
              onChange={(e) => set('q', e.target.value)}
              placeholder="Search by name, goal or target…"
              aria-label="Search models"
              className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-slate-400"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={status}
              onChange={(e) => set('status', e.target.value)}
              aria-label="Filter by status"
              className="h-9 rounded-lg border border-slate-200 bg-white px-2.5 text-[13px] font-medium text-slate-700 outline-none focus:border-emerald-400"
            >
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s ? s[0].toUpperCase() + s.slice(1) : 'All statuses'}
                </option>
              ))}
            </select>
            <div className="flex rounded-lg border border-slate-200 p-0.5">
              {(
                [
                  ['grid', LayoutGrid],
                  ['list', List],
                ] as const
              ).map(([v, Icon]) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => setViewMode(v)}
                  className={cx('rounded-md p-1.5', view === v ? 'bg-slate-900 text-white' : 'text-slate-500 hover:bg-slate-100')}
                  aria-label={`${v} view`}
                  aria-pressed={view === v}
                >
                  <Icon size={15} />
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5" role="group" aria-label="Filter by problem type">
          {TYPES.map(([k, label]) => (
            <button
              key={k}
              type="button"
              onClick={() => set('type', k)}
              aria-pressed={type === k}
              className={cx(
                'rounded-full border px-3 py-1 text-[12px] font-semibold transition-colors',
                type === k ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </Card>
      {models.isLoading && <Spinner />}
      {models.isError && (
        <Card>
          <Empty icon={AlertTriangle} title="Could not load models" text={errorMessage(models.error)} />
        </Card>
      )}
      {models.isSuccess && items.length === 0 && (
        <Card>
          <Empty
            icon={Boxes}
            title={q || type || status ? 'No models match your filters' : 'No models yet'}
            text={
              q || type || status
                ? 'Try a different search or filter.'
                : 'Describe a goal and upload an Excel file; the agent builds and publishes a model you can test.'
            }
            action={
              <Button variant="primary" icon={Plus} onClick={() => navigate('/agent')}>
                Create a model
              </Button>
            }
          />
        </Card>
      )}
      {items.length > 0 &&
        (view === 'grid' ? (
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {items.map((m) => (
              <ModelCard key={m.id} m={m} />
            ))}
          </div>
        ) : (
          <Card className="overflow-hidden">
            <div className="thin-scroll overflow-x-auto">
              <table className="w-full min-w-[820px] text-left">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    <th className="px-5 py-2.5">Model</th>
                    <th className="px-3 py-2.5">Problem</th>
                    <th className="px-3 py-2.5">Algorithm</th>
                    <th className="px-3 py-2.5 text-right">Key metric</th>
                    <th className="px-3 py-2.5">Status</th>
                    <th className="px-5 py-2.5 text-right">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((m) => (
                    <tr
                      key={m.id}
                      className="cursor-pointer border-b border-slate-100 last:border-0 hover:bg-slate-50"
                      onClick={() => navigate(`/models/${m.id}`)}
                    >
                      <td className="px-5 py-3">
                        <Link to={`/models/${m.id}`} className="flex items-center gap-3 focus-visible:outline-2 focus-visible:outline-emerald-500">
                          <KindIcon kind={PROBLEM_KIND[m.problem_type]} size="sm" />
                          <span className="min-w-0">
                            <span className="block truncate text-[13px] font-semibold">{m.name}</span>
                            <span className="block truncate font-mono text-[11px] text-slate-500">{m.target}</span>
                          </span>
                        </Link>
                      </td>
                      <td className="px-3 py-3 text-[12px] text-slate-600">{m.problem_label}</td>
                      <td className="px-3 py-3 font-mono text-[12px] text-slate-700">{m.algorithm ?? '—'}</td>
                      <td className="px-3 py-3 text-right">
                        <span className="tabular font-mono text-[13px] font-semibold">{m.primary_metric?.display ?? '—'}</span>{' '}
                        <span className="text-[11px] text-slate-500">{m.primary_metric?.name}</span>
                      </td>
                      <td className="px-3 py-3">
                        <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
                      </td>
                      <td className="px-5 py-3 text-right text-[12px] text-slate-500">{timeAgo(m.updated_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        ))}
    </div>
  )
}
