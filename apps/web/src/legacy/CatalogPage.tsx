// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { AlertTriangle, Boxes, Database, LayoutGrid, List, Plus, Search, Sparkles, Zap } from 'lucide-react'
import { useStore } from './store'
import { compact, timeAgo } from '../lib/format'
import { Badge, Button, Card, Empty, KIND, KindIcon, ProgressBar, Sparkline, StatusPill, cx } from './ui'

const STATUSES = ['all', 'live', 'staging', 'training', 'draft']

function ModelCard({ m }) {
  const spark = m.usage?.series?.filter((p) => p.value > 0).length >= 5 ? m.usage.series : null
  return (
    <Link to={`/models/${m.id}`} className="group">
      <Card className="flex h-full flex-col p-5 transition-all duration-150 group-hover:-translate-y-0.5 group-hover:border-emerald-300 group-hover:shadow-[0_8px_24px_rgba(15,23,42,0.08)]">
        <div className="flex items-start justify-between gap-3">
          <KindIcon kind={m.kind} />
          <div className="flex items-center gap-1.5">
            {m.createdBy === 'agent' && (
              <Badge color="indigo">
                <Sparkles size={10} />
                New
              </Badge>
            )}
            <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
          </div>
        </div>
        <div className="mt-3 min-w-0">
          <h3 className="truncate text-[15px] font-semibold tracking-tight text-slate-900 group-hover:text-emerald-800">{m.name}</h3>
          <div className="mt-0.5 truncate font-mono text-[11px] text-slate-500">
            {m.slug} · {m.version}
          </div>
        </div>
        <p className="mt-2 line-clamp-2 min-h-[2.5rem] text-[13px] text-slate-600">{m.objective}</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Badge>{KIND[m.kind]?.label}</Badge>
          <Badge color="slate" mono>
            {m.algorithm.split(' (')[0]}
          </Badge>
        </div>

        <div className="mt-4 flex items-end justify-between gap-3 rounded-lg bg-slate-50 p-3">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{m.primary.name}</div>
            <div className="text-[22px] font-semibold tracking-tight text-slate-900 tabular">{m.primary.display}</div>
            <div className="text-[11px] text-slate-500">baseline {m.baseline.display}</div>
          </div>
          <div className="w-24">
            {spark ? <Sparkline data={spark} height={34} color={m.drift?.status === 'warning' ? '#f59e0b' : '#10b981'} /> : null}
          </div>
        </div>

        {m.status === 'training' && (
          <div className="mt-3">
            <div className="mb-1 flex justify-between text-[11px] text-slate-500">
              <span>{m.stage}</span>
              <span className="font-mono tabular">{m.progress}%</span>
            </div>
            <ProgressBar value={m.progress} animated />
          </div>
        )}

        <div className="mt-auto flex items-center justify-between gap-2 pt-4 text-[11px] text-slate-500">
          <span className="flex min-w-0 items-center gap-1 truncate">
            <Database size={12} />
            {m.source.connector} · {m.source.database}
          </span>
          <span className="flex shrink-0 items-center gap-2">
            {m.drift?.status === 'warning' && (
              <span className="flex items-center gap-0.5 text-amber-700">
                <AlertTriangle size={12} />
                Drift
              </span>
            )}
            {m.usage?.predictions30d ? (
              <span className="flex items-center gap-0.5">
                <Zap size={12} />
                {compact(m.usage.predictions30d)}
              </span>
            ) : null}
            <span>{timeAgo(m.updatedAt)}</span>
          </span>
        </div>
      </Card>
    </Link>
  )
}

function ModelRow({ m }) {
  const navigate = useNavigate()
  return (
    <tr onClick={() => navigate(`/models/${m.id}`)} className="cursor-pointer border-b border-slate-100 last:border-0 hover:bg-slate-50">
      <td className="px-5 py-3">
        <div className="flex items-center gap-3">
          <KindIcon kind={m.kind} size="sm" />
          <div className="min-w-0">
            <div className="truncate text-[13px] font-semibold">{m.name}</div>
            <div className="truncate font-mono text-[11px] text-slate-500">
              {m.slug} · {m.version}
            </div>
          </div>
        </div>
      </td>
      <td className="px-3 py-3 text-[12px] text-slate-600">{m.problemType}</td>
      <td className="px-3 py-3 font-mono text-[12px] text-slate-700">{m.algorithm}</td>
      <td className="px-3 py-3 text-right">
        <span className="font-mono text-[13px] font-semibold tabular">{m.primary.display}</span>{' '}
        <span className="text-[11px] text-slate-500">{m.primary.name}</span>
      </td>
      <td className="px-3 py-3">
        <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
      </td>
      <td className="px-3 py-3 text-right font-mono text-[12px] tabular text-slate-600">
        {m.usage?.predictions30d ? compact(m.usage.predictions30d) : '—'}
      </td>
      <td className="px-5 py-3 text-right text-[12px] text-slate-500">{timeAgo(m.updatedAt)}</td>
    </tr>
  )
}

export default function Catalog() {
  const { models } = useStore()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [kind, setKind] = useState('all')
  const [status, setStatus] = useState('all')
  const [sort, setSort] = useState('updated')
  const [view, setView] = useState('grid')
  const q = params.get('q') || ''

  const list = useMemo(() => {
    const needle = q.toLowerCase()
    const out = models.filter(
      (m) =>
        (kind === 'all' || m.kind === kind) &&
        (status === 'all' || m.status === status) &&
        (!needle ||
          [m.name, m.slug, m.objective, m.algorithm, m.problemType, m.source.connector, ...(m.tags || [])].join(' ').toLowerCase().includes(needle)),
    )
    const by = {
      updated: (a, b) => b.updatedAt - a.updatedAt,
      name: (a, b) => a.name.localeCompare(b.name),
      usage: (a, b) => (b.usage?.predictions30d || 0) - (a.usage?.predictions30d || 0),
    }
    return out.sort(by[sort])
  }, [models, kind, status, sort, q])

  const counts = useMemo(() => {
    const c = { all: models.length }
    models.forEach((m) => {
      c[m.kind] = (c[m.kind] || 0) + 1
    })
    return c
  }, [models])

  return (
    <div className="space-y-5">
      <div className="flex flex-col justify-between gap-3 md:flex-row md:items-end">
        <div>
          <h1 className="text-[26px] font-semibold tracking-tight text-slate-900">Model Catalog</h1>
          <p className="mt-1 text-sm text-slate-600">Every model your AI Data Scientist has built, with data, metrics, decisions and usage.</p>
        </div>
        <Button variant="primary" icon={Plus} onClick={() => navigate('/agent')}>
          Create new model
        </Button>
      </div>

      <Card className="p-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 focus-within:border-emerald-400 focus-within:ring-2 focus-within:ring-emerald-100">
            <Search size={15} className="text-slate-400" />
            <input
              value={q}
              onChange={(e) => setParams(e.target.value ? { q: e.target.value } : {}, { replace: true })}
              placeholder="Search by name, objective, algorithm, source or tag…"
              className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-slate-400"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              className="h-9 rounded-lg border border-slate-200 bg-white px-2.5 text-[13px] font-medium text-slate-700 outline-none focus:border-emerald-400"
            >
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s === 'all' ? 'All statuses' : s[0].toUpperCase() + s.slice(1)}
                </option>
              ))}
            </select>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value)}
              className="h-9 rounded-lg border border-slate-200 bg-white px-2.5 text-[13px] font-medium text-slate-700 outline-none focus:border-emerald-400"
            >
              <option value="updated">Recently updated</option>
              <option value="usage">Most predictions</option>
              <option value="name">Name A–Z</option>
            </select>
            <div className="flex rounded-lg border border-slate-200 p-0.5">
              {[
                ['grid', LayoutGrid],
                ['list', List],
              ].map(([v, Icon]) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={cx('rounded-md p-1.5', view === v ? 'bg-slate-900 text-white' : 'text-slate-500 hover:bg-slate-100')}
                  aria-label={`${v} view`}
                >
                  <Icon size={15} />
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {['all', ...Object.keys(KIND)].map((k) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              className={cx(
                'rounded-full border px-3 py-1 text-[12px] font-semibold transition-colors',
                kind === k ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300',
              )}
            >
              {k === 'all' ? 'All types' : KIND[k].label}
              <span className={cx('ml-1.5 tabular', kind === k ? 'text-slate-300' : 'text-slate-400')}>{counts[k] || 0}</span>
            </button>
          ))}
        </div>
      </Card>

      {list.length === 0 ? (
        <Card>
          <Empty
            icon={Boxes}
            title="No models match your filters"
            text="Try a different search, or ask the AI Data Scientist to build one."
            action={
              <Button variant="primary" icon={Sparkles} onClick={() => navigate(`/agent${q ? `?goal=${encodeURIComponent(q)}` : ''}`)}>
                Build it with AI
              </Button>
            }
          />
        </Card>
      ) : view === 'grid' ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {list.map((m) => (
            <ModelCard key={m.id} m={m} />
          ))}
          <button
            onClick={() => navigate('/agent')}
            className="flex min-h-[260px] flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 text-slate-500 transition-colors hover:border-emerald-300 hover:bg-emerald-50/40 hover:text-emerald-700"
          >
            <span className="grid h-11 w-11 place-items-center rounded-xl bg-white shadow-sm ring-1 ring-slate-200">
              <Plus size={20} />
            </span>
            <span className="mt-3 text-sm font-semibold">Create a new model</span>
            <span className="mt-0.5 text-xs">Describe a goal, and the agent does the rest</span>
          </button>
        </div>
      ) : (
        <Card className="overflow-hidden">
          <div className="thin-scroll overflow-x-auto">
            <table className="w-full min-w-[860px] text-left">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="px-5 py-2.5">Model</th>
                  <th className="px-3 py-2.5">Problem type</th>
                  <th className="px-3 py-2.5">Algorithm</th>
                  <th className="px-3 py-2.5 text-right">Key metric</th>
                  <th className="px-3 py-2.5">Status</th>
                  <th className="px-3 py-2.5 text-right">Predictions 30d</th>
                  <th className="px-5 py-2.5 text-right">Updated</th>
                </tr>
              </thead>
              <tbody>
                {list.map((m) => (
                  <ModelRow key={m.id} m={m} />
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
