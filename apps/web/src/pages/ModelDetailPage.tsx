import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Boxes,
  ChevronRight,
  CircleCheck,
  CircleDashed,
  Database,
  Download,
  FlaskConical,
  GitBranch,
  Info,
  Layers,
  Lightbulb,
  LoaderCircle,
  PlayCircle,
  ScrollText,
  ShieldCheck,
  Sparkles,
  Trash2,
  Wand2,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { EvaluationCharts, ImportanceBars } from '../components/charts'
import { Badge, Button, Card, CardHeader, Empty, KindIcon, ProgressBar, Rich, Spinner, StatusPill, Tabs, type BadgeColor } from '../components/ui'
import { QualityRing } from '../features/datasets/ProfileView'
import { PROBLEM_KIND, useDeleteModel, useModel, type ModelDetail } from '../features/models/hooks'
import { Playground } from '../features/playground/Playground'
import { errorMessage } from '../lib/apiClient'
import { cx } from '../lib/cx'
import { fmtDate, num, timeAgo } from '../lib/format'

type Data = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any -- model card JSON

const SEV: Record<string, BadgeColor> = { high: 'red', medium: 'amber', low: 'sky', info: 'slate', pass: 'emerald' }

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="border-b border-slate-100 px-5 py-2.5">
      <dt className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</dt>
      <dd className="mt-0.5 break-words text-[13px] text-slate-800">{children}</dd>
    </div>
  )
}

function downloadCard(m: ModelDetail) {
  const blob = new Blob([JSON.stringify(m.card, null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${m.slug}_v${m.latest_version}_model_card.json`
  a.click()
  URL.revokeObjectURL(a.href)
}

function Overview({ m, card, ui }: { m: ModelDetail; card: Data; ui: Data }) {
  const final: Data = ui.final ?? {}
  const strategy: Data = ui.strategy ?? {}
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card className="p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{final.primary?.name ?? 'Score'}</div>
          <div className="tabular mt-1 text-[28px] font-semibold tracking-tight">{final.primary?.display ?? '—'}</div>
          <div className="text-xs text-slate-500">baseline {final.primary?.baseline ?? '—'}</div>
        </Card>
        <Card className="p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            {m.problem_type === 'forecasting' ? 'Horizon' : 'Cross-validation'}
          </div>
          <div className="tabular mt-1 text-[22px] font-semibold tracking-tight">
            {m.problem_type === 'forecasting' ? `${final.horizon ?? '—'} ${final.unit ?? ''}s` : (final.cv?.display ?? '—')}
          </div>
          <div className="text-xs text-slate-500">{strategy.validation}</div>
        </Card>
        <Card className="p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Training data</div>
          <div className="tabular mt-1 text-[22px] font-semibold tracking-tight">{num(card.dataset?.training_rows)} rows</div>
          <div className="truncate text-xs text-slate-500">
            {card.dataset?.file_name} · {card.dataset?.sheet}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Playground predictions</div>
          <div className="tabular mt-1 text-[28px] font-semibold tracking-tight">{num(m.prediction_count)}</div>
          <div className="text-xs text-slate-500">single + batch rows</div>
        </Card>
      </div>
      <div className="grid gap-6 xl:grid-cols-5">
        <div className="space-y-6 xl:col-span-3">
          <Card>
            <CardHeader
              title="What this model does"
              subtitle={`Written by the ${card.llm?.provider === 'gemini' ? 'Gemini' : 'offline'} assistant from the computed metrics`}
              icon={Lightbulb}
            />
            <div className="space-y-4 px-5 pb-5">
              <p className="text-[14px] leading-6 text-slate-700">{final.explanation}</p>
              <div className="rounded-lg border border-indigo-100 bg-indigo-50/60 p-4">
                <div className="mb-1 flex items-center gap-1.5 text-[12px] font-bold uppercase tracking-wider text-indigo-700">
                  <Sparkles size={13} aria-hidden />
                  Why this model?
                </div>
                <p className="text-[13px] leading-6 text-slate-700">{final.why_model}</p>
              </div>
            </div>
          </Card>
          <Card>
            <CardHeader title="Model card" icon={Info} subtitle="Everything needed to reproduce this version" />
            <dl className="grid grid-cols-1 border-t border-slate-100 sm:grid-cols-2">
              <Fact label="Goal">{m.objective}</Fact>
              <Fact label="Problem">{m.problem_label}</Fact>
              <Fact label="Target">
                <code className="font-mono text-[12px]">{m.target}</code>
                {ui.plan?.positive_class ? ` (positive: ${ui.plan.positive_class})` : ''}
              </Fact>
              <Fact label="Metric">{strategy.metric_label}</Fact>
              <Fact label="Algorithm">{card.algorithm}</Fact>
              <Fact label="Validation">{strategy.validation}</Fact>
              <Fact label="Data hash">
                <code className="font-mono text-[11px]">{String(card.data_hash ?? '').slice(0, 16)}…</code>
              </Fact>
              <Fact label="Training time · seed">
                {card.training_seconds}s · random_state {card.random_state}
              </Fact>
              <Fact label="Excluded columns">
                {card.excluded?.length ? card.excluded.map((e: Data) => `${e.column} (${e.reason})`).join(', ') : 'none'}
              </Fact>
              <Fact label="Libraries">
                {Object.entries(card.library_versions ?? {})
                  .map(([k, v]) => `${k} ${v}`)
                  .join(' · ')}
              </Fact>
            </dl>
          </Card>
        </div>
        <Card className="h-fit xl:col-span-2">
          <CardHeader
            title="Top drivers"
            subtitle={m.problem_type === 'forecasting' ? 'Gain importance of the model features' : 'Permutation importance on the holdout'}
            icon={BarChart3}
          />
          <div className="px-5 pb-5">
            <ImportanceBars items={final.importance ?? []} max={10} />
          </div>
        </Card>
      </div>
    </div>
  )
}

function DataTab({ card, ui }: { card: Data; ui: Data }) {
  const ds: Data = ui.dataset ?? {}
  const used = new Set<string>(card.columns ?? [])
  const excluded = new Map<string, string>((card.excluded ?? []).map((e: Data) => [e.column, e.reason]))
  const features: Data = ui.features ?? {}
  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="flex items-center gap-4 p-4">
          <QualityRing value={ds.quality ?? 0} />
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Data quality</div>
            <div className="truncate text-[13px] font-semibold">{ds.file_name}</div>
            <div className="text-xs text-slate-500">
              {ds.sheet} · {num(ds.rows)} rows × {ds.cols} columns
              {ds.period ? ` · ${ds.period}` : ''}
            </div>
          </div>
        </Card>
        <Card className="p-4 lg:col-span-2">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Quality deductions</div>
          <ul className="mt-1 space-y-0.5 text-[12px] text-slate-600">
            {(ds.quality_reasons ?? []).length === 0 && <li>No issues found</li>}
            {(ds.quality_reasons ?? []).map((r: Data) => (
              <li key={r.factor}>
                −{r.penalty} {r.detail}
              </li>
            ))}
          </ul>
        </Card>
      </div>
      <Card className="overflow-hidden">
        <CardHeader title="Columns" subtitle="Role detected at profiling and whether the model uses it" icon={Database} />
        <div className="thin-scroll overflow-x-auto">
          <table className="w-full min-w-[640px] text-left">
            <thead>
              <tr className="border-y border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                <th className="px-5 py-2.5">Column</th>
                <th className="px-3 py-2.5">Role</th>
                <th className="px-3 py-2.5 text-right">Missing</th>
                <th className="px-3 py-2.5 text-right">Unique</th>
                <th className="px-5 py-2.5">In the model</th>
              </tr>
            </thead>
            <tbody>
              {(ds.columns ?? []).map((c: Data) => {
                const isTarget = c.name === ui.plan?.target
                return (
                  <tr key={c.name} className={cx('border-b border-slate-100 last:border-0', isTarget && 'bg-emerald-50/40')}>
                    <td className={cx('px-5 py-2 font-mono text-[12px]', excluded.has(c.name) && 'text-slate-400 line-through')}>{c.name}</td>
                    <td className="px-3 py-2">
                      <Badge color={isTarget ? 'emerald' : c.kind === 'id' ? 'indigo' : 'slate'}>{isTarget ? 'Target' : c.role}</Badge>
                    </td>
                    <td className="tabular px-3 py-2 text-right font-mono text-[12px] text-slate-600">{c.missing_pct}%</td>
                    <td className="tabular px-3 py-2 text-right font-mono text-[12px] text-slate-600">{num(c.unique)}</td>
                    <td className="px-5 py-2 text-[12px]">
                      {isTarget ? (
                        <span className="text-emerald-700">predicted</span>
                      ) : excluded.has(c.name) ? (
                        <span className="text-red-700">excluded: {excluded.get(c.name)}</span>
                      ) : used.has(c.name) ? (
                        <span className="text-slate-700">input</span>
                      ) : (
                        <span className="text-slate-400">not used</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Cleaning applied" subtitle="Approved in the conversation; fitted on training folds only" icon={ShieldCheck} />
          <ul className="divide-y divide-slate-100 border-t border-slate-100">
            {(ui.cleaning ?? []).map((s: Data) => (
              <li key={s.id ?? s.title} className={cx('flex gap-3 px-5 py-3', s.enabled === false && 'opacity-50')}>
                {s.enabled === false ? (
                  <CircleDashed size={16} className="mt-0.5 shrink-0 text-slate-400" aria-label="Skipped" />
                ) : (
                  <CircleCheck size={16} className="mt-0.5 shrink-0 text-emerald-500" aria-label="Applied" />
                )}
                <div className="min-w-0">
                  <div className="text-[13px] font-semibold text-slate-900">{s.title}</div>
                  <div className="text-[12px] text-slate-600">{s.detail}</div>
                </div>
              </li>
            ))}
          </ul>
        </Card>
        <Card>
          <CardHeader title="Features" subtitle={`${features.total ?? '—'} model features`} icon={Layers} />
          <div className="flex flex-wrap gap-1.5 border-t border-slate-100 p-4">
            {(features.items ?? []).map((f: Data) => (
              <span
                key={f.name}
                title={f.desc}
                className={cx(
                  'rounded-md border px-2 py-1 font-mono text-[11px]',
                  f.kind === 'derived'
                    ? f.accepted
                      ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                      : 'border-slate-200 text-slate-400 line-through'
                    : 'border-slate-200 bg-slate-50 text-slate-700',
                )}
              >
                {f.name}
                {f.source === 'llm' && ' ✦'}
              </span>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}

function MetricsTab({ m, ui }: { m: ModelDetail; ui: Data }) {
  const final: Data = ui.final ?? {}
  const rounds = [
    ['Round 1', ui.round1 ?? []],
    ['Round 2', ui.round2 ?? []],
  ] as const
  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {(final.metrics ?? []).map((x: Data) => (
          <Card key={x.name} className="p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{x.name}</div>
            <div className="tabular mt-1 text-[22px] font-semibold tracking-tight">{x.display}</div>
            <div className={cx('text-[11px]', x.good === false ? 'text-red-700' : 'text-slate-500')}>baseline {x.baseline}</div>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader
          title="Evaluation"
          subtitle={m.problem_type === 'forecasting' ? 'Rolling-origin backtest' : `Held-out rows · ${ui.strategy?.validation ?? ''}`}
          icon={BarChart3}
        />
        <div className="px-5 pb-5">
          <EvaluationCharts problemType={m.problem_type} charts={final.charts ?? {}} />
        </div>
      </Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Model search" subtitle="Cross-validated score of every run (round 2 includes the critic's fixes)" icon={FlaskConical} />
          <div className="space-y-4 px-5 pb-5">
            {rounds.map(([label, runs]) =>
              runs.length ? (
                <div key={label}>
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</div>
                  <table className="w-full text-[12px]">
                    <tbody>
                      {(runs as Data[]).map((r) => (
                        <tr key={r.run_id} className="border-b border-slate-100 last:border-0">
                          <td className="py-1.5 font-medium text-slate-800">
                            {r.candidate}
                            {r.tuned ? ' · tuned' : ''}
                          </td>
                          <td className="tabular py-1.5 text-right font-mono">
                            {Number(r.mean).toFixed(4)} <span className="text-slate-400">± {Number(r.std).toFixed(4)}</span>
                          </td>
                          <td className="tabular py-1.5 text-right font-mono text-slate-400">{r.fit_seconds}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null,
            )}
          </div>
        </Card>
        <Card>
          <CardHeader title="Holdout comparison" subtitle="Top candidates refit and scored on unseen rows" icon={GitBranch} />
          <div className="px-5 pb-5">
            {(final.ranking ?? []).length === 0 ? (
              <p className="text-[13px] text-slate-500">Forecasting models are compared on the backtest windows above.</p>
            ) : (
              <table className="w-full text-[12px]">
                <tbody>
                  {(final.ranking as Data[]).map((r) => (
                    <tr key={r.candidate} className={cx('border-b border-slate-100 last:border-0', r.selected && 'font-semibold')}>
                      <td className="py-1.5">
                        {r.candidate} {r.selected && <Badge color="emerald">selected</Badge>}
                      </td>
                      <td className="tabular py-1.5 text-right font-mono">{r.holdout != null ? Number(r.holdout).toFixed(4) : '—'}</td>
                      <td className="tabular py-1.5 text-right font-mono text-slate-400">{r.fit_seconds != null ? `${r.fit_seconds}s` : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

function DecisionsTab({ ui }: { ui: Data }) {
  const decisions: Data[] = ui.decisions ?? []
  const critic: Data = ui.critic ?? {}
  return (
    <div className="grid gap-6 xl:grid-cols-3">
      <Card className="xl:col-span-2">
        <CardHeader title="Decision log" subtitle="Every significant choice, the alternatives considered and the reason" icon={ScrollText} />
        <ol className="border-t border-slate-100 px-5 py-4">
          {decisions.map((d, i) => (
            <li key={i} className="relative flex gap-4 pb-5 last:pb-0">
              {i < decisions.length - 1 && <span className="absolute left-[15px] top-8 bottom-0 w-px bg-slate-200" aria-hidden />}
              <span className="z-10 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-slate-900 font-mono text-[11px] font-bold text-white">
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0 flex-1 rounded-lg border border-slate-200 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge color="indigo">{d.step}</Badge>
                  <span className="text-[13px] font-semibold text-slate-900">
                    <Rich text={d.decision} />
                  </span>
                </div>
                {d.reason && (
                  <p className="mt-1.5 text-[13px] text-slate-600">
                    <span className="font-semibold text-slate-700">Reason: </span>
                    <Rich text={d.reason} />
                  </p>
                )}
                {d.alternatives && (
                  <p className="mt-1 text-[12px] text-slate-500">
                    <span className="font-semibold">Alternatives considered: </span>
                    {d.alternatives}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      </Card>
      <Card className="h-fit">
        <CardHeader title="Critic review" subtitle={critic.headline} icon={ShieldCheck} />
        <ul className="space-y-2 border-t border-slate-100 p-4">
          {(critic.findings ?? []).map((f: Data) => (
            <li key={f.id} className="rounded-lg border border-slate-200 p-3">
              <div className="flex items-start justify-between gap-2">
                <span className="text-[13px] font-semibold text-slate-900">
                  <Rich text={f.title} />
                </span>
                <Badge color={SEV[f.severity] ?? 'slate'}>{f.severity}</Badge>
              </div>
              <p className="mt-1 text-[12px] text-slate-600">
                <Rich text={f.detail} />
              </p>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}

function VersionsTab({ m }: { m: ModelDetail }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader title="Versions" subtitle="Each version is a saved pipeline + model card + input schema under storage/artifacts" icon={GitBranch} />
      <table className="w-full text-left">
        <thead>
          <tr className="border-y border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            <th className="px-5 py-2.5">Version</th>
            <th className="px-3 py-2.5">Created</th>
            <th className="px-3 py-2.5">Algorithm</th>
            <th className="px-3 py-2.5 text-right">Key metric</th>
            <th className="px-5 py-2.5">Data</th>
          </tr>
        </thead>
        <tbody>
          {(m.versions ?? []).map((v) => (
            <tr key={v.version} className="border-b border-slate-100 last:border-0">
              <td className="px-5 py-3 font-mono text-[13px] font-semibold">v{v.version}</td>
              <td className="px-3 py-3 text-[12px] text-slate-600">{fmtDate(v.created_at)}</td>
              <td className="px-3 py-3 text-[12px]">{v.algorithm}</td>
              <td className="tabular px-3 py-3 text-right font-mono text-[13px]">
                {v.primary_metric?.display} <span className="text-[11px] text-slate-500">{v.primary_metric?.name}</span>
              </td>
              <td className="px-5 py-3 text-[12px] text-slate-600">
                {v.note} · <code className="font-mono text-[11px]">{v.data_hash.slice(0, 10)}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

export default function ModelDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const model = useModel(id)
  const del = useDeleteModel()
  const tab = params.get('tab') ?? 'overview'
  const setTab = (t: string) => setParams({ tab: t }, { replace: true })

  if (model.isLoading) return <Spinner />
  if (model.isError || !model.data)
    return (
      <Card>
        <Empty
          icon={Boxes}
          title="Model not found"
          text={errorMessage(model.error)}
          action={
            <Button onClick={() => navigate('/models')} icon={ArrowRight}>
              Back to the catalog
            </Button>
          }
        />
      </Card>
    )
  const m = model.data
  const card = (m.card ?? {}) as Data
  const ui = (card.ui ?? {}) as Data
  const ready = m.status === 'ready' && Boolean(m.card)
  const tabs = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'data', label: 'Data', icon: Database },
    { id: 'metrics', label: 'Metrics', icon: BarChart3 },
    { id: 'decisions', label: 'Decision log', icon: ScrollText, count: ui.decisions?.length ?? null },
    { id: 'versions', label: 'Versions', icon: GitBranch, count: m.versions?.length ?? null },
    { id: 'test', label: 'Test model', icon: PlayCircle },
  ]

  return (
    <div className="space-y-5">
      <nav className="flex items-center gap-1.5 text-[13px] text-slate-500" aria-label="Breadcrumb">
        <Link to="/models" className="hover:text-slate-900">
          Model Catalog
        </Link>
        <ChevronRight size={14} aria-hidden />
        <span className="truncate font-medium text-slate-900">{m.name}</span>
      </nav>
      <Card className="p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex min-w-0 gap-4">
            <KindIcon kind={PROBLEM_KIND[m.problem_type]} size="lg" />
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-[22px] font-semibold tracking-tight text-slate-900">{m.name}</h1>
                <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
                {m.latest_version > 0 && (
                  <Badge color="dark" mono>
                    v{m.latest_version}
                  </Badge>
                )}
              </div>
              <p className="mt-1 text-[14px] text-slate-600">{m.objective}</p>
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-slate-500">
                <span>{m.problem_label}</span>
                {m.target && (
                  <span>
                    target <code className="font-mono">{m.target}</code>
                  </span>
                )}
                {m.algorithm && <span className="font-mono">{m.algorithm}</span>}
                <span>
                  created {fmtDate(m.created_at)} · updated {timeAgo(m.updated_at)}
                </span>
              </div>
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            {ready && (
              <Button variant="primary" icon={PlayCircle} onClick={() => setTab('test')}>
                Test this model
              </Button>
            )}
            <Button
              icon={Wand2}
              onClick={() => navigate(`/agent?${new URLSearchParams({ goal: m.objective, ...(m.dataset_id ? { dataset: m.dataset_id } : {}) })}`)}
            >
              Retrain with agent
            </Button>
            {m.card && (
              <Button icon={Download} onClick={() => downloadCard(m)} aria-label="Download model card">
                Model card
              </Button>
            )}
            <Button
              variant="danger"
              icon={Trash2}
              aria-label="Delete model"
              disabled={del.isPending}
              onClick={() => {
                if (window.confirm(`Delete ${m.name} and all its versions?`)) del.mutate(m.id, { onSuccess: () => navigate('/models') })
              }}
            />
          </div>
        </div>
        {(m.status === 'training' || m.status === 'draft') && (
          <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
            <LoaderCircle size={18} className="shrink-0 animate-spin text-amber-600" aria-hidden />
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-semibold text-amber-900">The agent is working: {m.stage}</div>
              <ProgressBar value={m.progress} animated className="mt-1.5 bg-amber-100" label="Training progress" />
            </div>
            {m.session_id && (
              <Button size="sm" iconRight={ArrowRight} onClick={() => navigate(`/agent/${m.session_id}`)}>
                Open the conversation
              </Button>
            )}
          </div>
        )}
        {m.status === 'failed' && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 p-3" role="alert">
            <span className="flex items-start gap-2 text-[13px] text-red-900">
              <AlertTriangle size={16} className="mt-0.5 shrink-0" aria-hidden />
              {m.error}
            </span>
            {m.session_id && (
              <Button size="sm" onClick={() => navigate(`/agent/${m.session_id}`)}>
                Open the conversation
              </Button>
            )}
          </div>
        )}
      </Card>
      {!m.card ? (
        <Card>
          <Empty
            icon={FlaskConical}
            title="No published version yet"
            text="Details, metrics and the Playground appear once the agent publishes the model."
          />
        </Card>
      ) : (
        <>
          <Tabs tabs={tabs} value={tab} onChange={setTab} />
          <div key={tab} className="animate-fade-in" role="tabpanel">
            {tab === 'overview' && <Overview m={m} card={card} ui={ui} />}
            {tab === 'data' && <DataTab card={card} ui={ui} />}
            {tab === 'metrics' && <MetricsTab m={m} ui={ui} />}
            {tab === 'decisions' && <DecisionsTab ui={ui} />}
            {tab === 'versions' && <VersionsTab m={m} />}
            {tab === 'test' && <Playground model={m} />}
          </div>
        </>
      )}
    </div>
  )
}
