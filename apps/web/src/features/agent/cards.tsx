/** Rich cards the agent posts in the chat. Each renders the real payload of a `card` event. */
import {
  ArrowRight,
  BarChart3,
  CircleCheck,
  CircleDashed,
  Database,
  FlaskConical,
  Gauge,
  Layers,
  LoaderCircle,
  Rocket,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
  Wand2,
  type LucideProps,
} from 'lucide-react'
import { useState, type ComponentType, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { EvaluationCharts, ImportanceBars } from '../../components/charts'
import { Badge, Button, KindIcon, ProgressBar, Rich, type BadgeColor } from '../../components/ui'
import { cx } from '../../lib/cx'
import { num } from '../../lib/format'
import { QualityRing } from '../datasets/ProfileView'

type Data = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any -- card payloads are validated server-side

const TONES = {
  slate: 'bg-slate-100 text-slate-600',
  ai: 'bg-indigo-100 text-indigo-600',
  good: 'bg-emerald-100 text-emerald-700',
  warn: 'bg-amber-100 text-amber-700',
}

export function CardShell({
  icon: Icon,
  title,
  subtitle,
  tone = 'slate',
  right,
  children,
}: {
  icon: ComponentType<LucideProps>
  title: ReactNode
  subtitle?: ReactNode
  tone?: keyof typeof TONES
  right?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="animate-fade-up overflow-hidden rounded-xl border border-slate-200 bg-white shadow-[0_1px_3px_rgba(15,23,42,0.05)]">
      <header className="flex items-center justify-between gap-3 border-b border-slate-100 bg-slate-50/70 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className={cx('grid h-6 w-6 shrink-0 place-items-center rounded-md', TONES[tone])}>
            <Icon size={13} aria-hidden />
          </span>
          <div className="min-w-0">
            <h3 className="truncate text-[13px] font-semibold text-slate-900">{title}</h3>
            {subtitle && <div className="truncate text-[11px] text-slate-500">{subtitle}</div>}
          </div>
        </div>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  )
}

const KIND_OF: Record<string, string> = {
  binary: 'classification',
  multiclass: 'classification',
  regression: 'regression',
  forecasting: 'forecasting',
}

export function ProfileCard({ d }: { d: Data }) {
  const [all, setAll] = useState(false)
  const cols: Data[] = d.columns ?? []
  return (
    <CardShell icon={Database} title="Data profile" subtitle={`${d.file_name} · ${d.sheet}`}>
      <div className="flex flex-wrap items-center gap-4">
        <QualityRing value={d.quality} size={60} />
        <div className="grid flex-1 grid-cols-3 gap-2">
          {[
            ['Rows', num(d.rows)],
            ['Columns', num(d.cols)],
            ['Duplicates', num(d.duplicates)],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg bg-slate-50 px-3 py-2">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{k}</div>
              <div className="tabular text-[15px] font-semibold text-slate-900">{v}</div>
            </div>
          ))}
        </div>
      </div>
      {d.period && <div className="mt-2 text-[12px] text-slate-500">Period: {d.period}</div>}
      <div className="thin-scroll mt-3 overflow-x-auto rounded-lg border border-slate-200">
        <table className="w-full min-w-[460px] text-left">
          <thead>
            <tr className="bg-slate-50 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              <th className="px-3 py-1.5">Column</th>
              <th className="px-2 py-1.5">Role</th>
              <th className="px-2 py-1.5 text-right">Unique</th>
              <th className="px-3 py-1.5 text-right">Missing</th>
            </tr>
          </thead>
          <tbody>
            {(all ? cols : cols.slice(0, 8)).map((c) => (
              <tr key={c.name} className="border-t border-slate-100">
                <td className="px-3 py-1.5 font-mono text-[12px]">
                  {c.name}
                  {c.pii && (
                    <Badge color="amber" className="ml-1.5">
                      masked for LLM
                    </Badge>
                  )}
                </td>
                <td className="px-2 py-1.5">
                  <Badge color={c.kind === 'id' ? 'indigo' : ['constant', 'empty', 'text'].includes(c.kind) ? 'slate' : 'sky'}>{c.role}</Badge>
                </td>
                <td className="tabular px-2 py-1.5 text-right font-mono text-[11px] text-slate-500">{num(c.unique)}</td>
                <td className={cx('tabular px-3 py-1.5 text-right font-mono text-[11px]', c.missing_pct > 20 ? 'text-amber-700' : 'text-slate-500')}>
                  {c.missing_pct}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {cols.length > 8 && (
        <button type="button" onClick={() => setAll((v) => !v)} className="mt-2 text-[12px] font-semibold text-emerald-700 hover:underline">
          {all ? 'Show fewer columns' : `Show all ${cols.length} columns`}
        </button>
      )}
      {(d.notes?.length > 0 || d.issues?.length > 0) && (
        <ul className="mt-3 space-y-1 text-[12px] text-slate-600">
          {(d.notes ?? []).map((n: string) => (
            <li key={n}>• {n}</li>
          ))}
          {(d.issues ?? [])
            .filter((i: Data) => i.severity !== 'info')
            .slice(0, 5)
            .map((i: Data, k: number) => (
              <li key={k} className="text-amber-800">
                • {i.detail}
              </li>
            ))}
        </ul>
      )}
    </CardShell>
  )
}

export function DetectionCard({ d }: { d: Data }) {
  const conf = Math.round((d.confidence ?? 0) * 100)
  return (
    <CardShell
      icon={Target}
      title="Problem detection"
      subtitle="Rules + LLM reading of your goal, checked against the data"
      tone="ai"
      right={<Badge color="emerald">{conf}% confidence</Badge>}
    >
      <div className="flex items-center gap-3">
        <KindIcon kind={KIND_OF[d.problem_type] ?? 'classification'} size="lg" />
        <div className="min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Detected problem</div>
          <div className="text-lg font-semibold tracking-tight text-slate-900">{d.problem_label}</div>
          <div className="text-[12px] text-slate-600">
            Target <code className="rounded bg-slate-100 px-1 font-mono">{d.target}</code>
            {d.positive_class && (
              <>
                {' '}
                · positive class <code className="rounded bg-slate-100 px-1 font-mono">{d.positive_class}</code>
              </>
            )}
            {d.time_column && (
              <>
                {' '}
                · time <code className="rounded bg-slate-100 px-1 font-mono">{d.time_column}</code>
              </>
            )}
            {d.series_column && (
              <>
                {' '}
                · series <code className="rounded bg-slate-100 px-1 font-mono">{d.series_column}</code>
              </>
            )}
          </div>
        </div>
      </div>
      <ProgressBar value={conf} className="mt-3" label="Detection confidence" />
      <ul className="mt-3 space-y-1.5">
        {(d.signals ?? []).map((s: string) => (
          <li key={s} className="flex gap-2 text-[13px] text-slate-700">
            <CircleCheck size={15} className="mt-0.5 shrink-0 text-emerald-500" aria-hidden />
            <Rich text={s} />
          </li>
        ))}
      </ul>
      {d.alternatives?.length > 0 && (
        <div className="mt-3 rounded-lg bg-slate-50 p-2.5 text-[12px] text-slate-600">
          <span className="font-semibold text-slate-700">Also considered: </span>
          {d.alternatives.join(' ')}
        </div>
      )}
    </CardShell>
  )
}

export function StrategyCard({ d }: { d: Data }) {
  return (
    <CardShell icon={Gauge} title="ML strategy" subtitle="Metric and validation picked by rules, explained by the assistant">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-emerald-200 bg-emerald-50/50 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-emerald-700">Optimisation metric</div>
          <div className="mt-0.5 text-[15px] font-semibold text-slate-900">
            {d.metric_label} <span className="text-xs font-normal text-slate-500">({d.better} is better)</span>
          </div>
          {d.secondary?.length > 0 && <div className="mt-1 text-[11px] text-slate-500">Also reported: {d.secondary.join(', ')}</div>}
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Validation</div>
          <div className="mt-0.5 text-[13px] font-semibold text-slate-900">{d.validation}</div>
        </div>
      </div>
      <p className="mt-3 text-[13px] leading-6 text-slate-700">{d.explanation}</p>
      {d.extras?.length > 0 && (
        <dl className="mt-3 grid gap-x-4 gap-y-1.5 sm:grid-cols-3">
          {d.extras.map(([k, v]: [string, string]) => (
            <div key={k}>
              <dt className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{k}</dt>
              <dd className="text-[12px] font-medium text-slate-800">{v}</dd>
            </div>
          ))}
        </dl>
      )}
    </CardShell>
  )
}

export function CleaningCard({ d }: { d: Data }) {
  const items: Data[] = d.items ?? []
  return (
    <CardShell
      icon={ShieldCheck}
      title="Cleaning plan"
      subtitle={`${items.length} steps · fitted on training folds only`}
      tone="good"
      right={d.approved ? <Badge color="emerald">Approved</Badge> : <Badge color="amber">Waiting for approval</Badge>}
    >
      <ul className="space-y-2">
        {items.map((s) => (
          <li key={s.id} className={cx('flex gap-2.5', s.enabled === false && 'opacity-50')}>
            {s.enabled === false ? (
              <CircleDashed size={15} className="mt-0.5 shrink-0 text-slate-400" aria-label="Skipped" />
            ) : (
              <CircleCheck size={15} className="mt-0.5 shrink-0 text-emerald-500" aria-label="Applied" />
            )}
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-medium text-slate-900">
                {s.title}
                {!s.optional && <span className="ml-1.5 text-[10px] font-semibold uppercase text-slate-400">required</span>}
              </div>
              <div className="text-[12px] text-slate-600">{s.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </CardShell>
  )
}

export function FeaturesCard({ d }: { d: Data }) {
  const items: Data[] = d.items ?? []
  const derived = items.filter((i) => i.kind === 'derived')
  const other = items.filter((i) => i.kind !== 'derived')
  return (
    <CardShell
      icon={Layers}
      title="Feature engineering"
      subtitle={`${d.total} model features from ${d.inputs} input column${d.inputs === 1 ? '' : 's'}`}
      tone="ai"
    >
      <div className="flex flex-wrap gap-1.5">
        {other.slice(0, 24).map((f) => (
          <span
            key={f.name}
            title={f.desc}
            className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-700"
          >
            {f.name}
          </span>
        ))}
        {other.length > 24 && (
          <span className="rounded-md border border-dashed border-slate-300 px-2 py-1 font-mono text-[11px] text-slate-500">
            +{other.length - 24} more
          </span>
        )}
      </div>
      {derived.length > 0 && (
        <div className="mt-3 space-y-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Derived features (kept only if they improve CV)</div>
          {derived.map((f) => (
            <div
              key={f.name}
              className={cx(
                'flex flex-wrap items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[12px]',
                f.accepted ? 'border-emerald-200 bg-emerald-50/50' : 'border-slate-200 opacity-70',
              )}
            >
              <code className="font-mono font-semibold">{f.name}</code>
              <span className="text-slate-500">= {f.desc}</span>
              {f.source === 'llm' && (
                <Badge color="indigo">
                  <Sparkles size={10} aria-hidden />
                  LLM idea
                </Badge>
              )}
              <Badge color={f.accepted ? 'emerald' : 'slate'}>{f.accepted ? `kept (+${Number(f.gain).toFixed(4)})` : 'rejected: no CV gain'}</Badge>
            </div>
          ))}
        </div>
      )}
    </CardShell>
  )
}

export function LeaderboardCard({ d }: { d: Data }) {
  const rows: Data[] = d.rows ?? []
  return (
    <CardShell
      icon={FlaskConical}
      title={d.round === 1 ? 'Model search · round 1' : 'Model search · round 2 (critic fixes)'}
      subtitle={`${d.metric_label} (${d.better} is better)${d.tuning ? ` · tuning ${d.tuning}` : ''}`}
      tone={d.done ? 'good' : 'warn'}
      right={
        <span className="tabular flex items-center gap-1.5 font-mono text-[11px] text-slate-500" aria-live="polite">
          {!d.done && <LoaderCircle size={12} className="animate-spin text-amber-500" aria-hidden />}
          {d.trials} HPO trials
        </span>
      }
    >
      <div className="space-y-2">
        {rows.map((r) => {
          const done = r.status === 'done'
          return (
            <div
              key={r.key}
              className={cx(
                'grid grid-cols-[minmax(0,1fr)_7rem] items-center gap-3 rounded-lg px-3 py-2 transition-colors',
                r.best ? 'bg-emerald-50 ring-1 ring-emerald-200' : 'bg-slate-50',
              )}
            >
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-x-2">
                  <span className="truncate text-[13px] font-semibold text-slate-900">{r.name}</span>
                  <span className="text-[11px] text-slate-500">{r.family}</span>
                  {r.best && <Badge color="emerald">Leader</Badge>}
                  {r.tuned && <Badge color="indigo">tuned</Badge>}
                  {r.status === 'skipped' && <Badge>skipped</Badge>}
                  {r.status === 'failed' && <Badge color="red">failed</Badge>}
                </div>
                {r.status === 'running' || r.status === 'queued' ? (
                  <ProgressBar
                    value={r.progress}
                    className="mt-1.5 bg-white"
                    animated={r.status === 'running'}
                    color=""
                    label={`${r.name} progress`}
                  />
                ) : (
                  r.note && <div className="mt-0.5 truncate text-[11px] text-slate-500">{r.note}</div>
                )}
              </div>
              <div className="text-right">
                {done ? (
                  <>
                    <div className="tabular font-mono text-[13px] font-semibold text-slate-900">{r.display}</div>
                    <div className="tabular font-mono text-[10px] text-slate-400">
                      {r.prev_display && r.prev_display !== r.display ? (
                        <span className="line-through">{r.prev_display}</span>
                      ) : r.std != null ? (
                        `± ${Number(r.std).toFixed(3)}`
                      ) : (
                        ''
                      )}
                    </div>
                  </>
                ) : r.status === 'running' ? (
                  <div className="tabular font-mono text-[12px] text-slate-400">{r.interim ?? '…'}</div>
                ) : (
                  <div className="text-[11px] text-slate-400">{r.status}</div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </CardShell>
  )
}

const SEV: Record<string, [BadgeColor, string]> = {
  high: ['red', 'High'],
  medium: ['amber', 'Medium'],
  low: ['sky', 'Low'],
  info: ['slate', 'Info'],
  pass: ['emerald', 'Pass'],
}

export function CriticCard({ d }: { d: Data }) {
  const findings: Data[] = d.findings ?? []
  const [showPass, setShowPass] = useState(false)
  const issues = findings.filter((f) => f.severity !== 'pass')
  const passes = findings.filter((f) => f.severity === 'pass')
  return (
    <CardShell icon={ShieldAlert} title="Critic review" subtitle="Leakage · overfitting · stability · calibration · baseline · residuals" tone="warn">
      <p className="mb-3 text-[13px] text-slate-700">
        <Rich text={d.headline} />
      </p>
      <ul className="space-y-2">
        {issues.map((f) => {
          const [color, label] = SEV[f.severity] ?? SEV.info
          return (
            <li key={f.id} className={cx('rounded-lg border p-3', f.severity === 'high' ? 'border-red-200 bg-red-50/50' : 'border-slate-200')}>
              <div className="flex items-start justify-between gap-2">
                <span className="text-[13px] font-semibold text-slate-900">
                  <Rich text={f.title} />
                </span>
                <div className="flex shrink-0 gap-1">
                  {(d.applied ?? []).includes(f.id) && <Badge color="indigo">fixed in round 2</Badge>}
                  <Badge color={color}>{label}</Badge>
                </div>
              </div>
              <p className="mt-1 text-[12px] text-slate-600">
                <Rich text={f.detail} />
              </p>
            </li>
          )
        })}
      </ul>
      {passes.length > 0 && (
        <div className="mt-2">
          <button type="button" onClick={() => setShowPass((v) => !v)} className="text-[12px] font-semibold text-emerald-700 hover:underline">
            {showPass ? 'Hide' : 'Show'} {passes.length} passed check{passes.length > 1 ? 's' : ''}
          </button>
          {showPass && (
            <ul className="mt-2 space-y-1">
              {passes.map((f) => (
                <li key={f.id} className="flex gap-2 text-[12px] text-slate-600">
                  <CircleCheck size={14} className="mt-0.5 shrink-0 text-emerald-500" aria-hidden />
                  <span>
                    <strong className="font-semibold text-slate-800">{f.title}.</strong> <Rich text={f.detail} />
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </CardShell>
  )
}

export function FinalCard({ d, onShowDecisions }: { d: Data; onShowDecisions?: () => void }) {
  const [why, setWhy] = useState(false)
  const metrics: Data[] = d.metrics ?? []
  return (
    <CardShell
      icon={Sparkles}
      title="Final model"
      subtitle={d.holdout_rows ? `Evaluated on ${num(d.holdout_rows)} held-out rows` : 'Evaluated on the backtest'}
      tone="good"
      right={<Badge color="emerald">Validated</Badge>}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <KindIcon kind={KIND_OF[d.problem_type] ?? 'classification'} size="lg" />
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Best model</div>
            <div className="text-lg font-semibold tracking-tight text-slate-900">{d.algorithm}</div>
            {d.cv?.display && <div className="text-[11px] text-slate-500">cross-validation {d.cv.display}</div>}
          </div>
        </div>
        <div className="flex items-end gap-4">
          <div className="text-right">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{d.primary?.name}</div>
            <div className="tabular text-[28px] font-semibold leading-8 tracking-tight text-emerald-600">{d.primary?.display}</div>
          </div>
          <div className="pb-0.5 text-right text-[11px] text-slate-500">
            baseline
            <br />
            <span className="font-mono text-[13px] font-semibold text-slate-700">{d.primary?.baseline}</span>
          </div>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {metrics.slice(0, 8).map((m) => (
          <div key={m.name} className="rounded-lg bg-slate-50 px-2.5 py-2">
            <div className="truncate text-[10px] font-semibold uppercase tracking-wider text-slate-500">{m.name}</div>
            <div className="tabular font-mono text-[14px] font-semibold text-slate-900">{m.display}</div>
            <div className="truncate text-[10px] text-slate-400">baseline {m.baseline}</div>
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-lg border border-slate-200 p-3">
        <EvaluationCharts problemType={d.problem_type} charts={d.charts ?? {}} compactMode />
      </div>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <div className="mb-2 flex items-center gap-1.5 text-[12px] font-semibold text-slate-900">
            <BarChart3 size={14} className="text-slate-400" aria-hidden />
            Top drivers (permutation importance)
          </div>
          <ImportanceBars items={d.importance ?? []} max={6} />
        </div>
        <div>
          <div className="mb-2 flex items-center gap-1.5 text-[12px] font-semibold text-slate-900">
            <Wand2 size={14} className="text-slate-400" aria-hidden />
            In plain language
          </div>
          <p className="text-[13px] leading-6 text-slate-700">{d.explanation}</p>
        </div>
      </div>
      {why && (
        <div className="mt-4 animate-fade-up rounded-lg border border-indigo-100 bg-indigo-50/60 p-3 text-[13px] leading-6 text-slate-700">
          <div className="mb-1 text-[11px] font-bold uppercase tracking-wider text-indigo-700">Why this model?</div>
          {d.why_model}
          {d.notes?.length > 0 && <div className="mt-1 text-[12px] text-slate-500">{d.notes.join(' ')}</div>}
        </div>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <Button size="sm" variant={why ? 'dark' : 'secondary'} icon={Sparkles} onClick={() => setWhy((v) => !v)} aria-expanded={why}>
          Why this model?
        </Button>
        {onShowDecisions && (
          <Button size="sm" onClick={onShowDecisions}>
            View decisions
          </Button>
        )}
      </div>
    </CardShell>
  )
}

export function PublishedCard({ d }: { d: Data }) {
  return (
    <CardShell
      icon={Rocket}
      title="Published for testing"
      subtitle={`${d.name} · v${d.version}`}
      tone="good"
      right={<Badge color="emerald">Ready</Badge>}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-[13px] text-slate-700">
          <strong className="font-semibold">{d.algorithm}</strong> · {d.primary?.name} <strong className="font-mono">{d.primary?.display}</strong>
          <div className="text-[11px] text-slate-500">Saved as {d.artifact}: pipeline, model card and input schema</div>
        </div>
        <Link
          to={d.test_url}
          className="inline-flex h-9 items-center gap-2 rounded-lg bg-emerald-500 px-3.5 text-[13px] font-semibold text-white shadow-[0_4px_14px_rgba(16,185,129,0.3)] hover:bg-emerald-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-500"
        >
          Test this model <ArrowRight size={16} aria-hidden />
        </Link>
      </div>
    </CardShell>
  )
}

export function AgentCard({ kind, data, onShowDecisions }: { kind: string; data: Data; onShowDecisions?: () => void }) {
  switch (kind) {
    case 'profile':
      return <ProfileCard d={data} />
    case 'detection':
      return <DetectionCard d={data} />
    case 'strategy':
      return <StrategyCard d={data} />
    case 'cleaning':
      return <CleaningCard d={data} />
    case 'features':
      return <FeaturesCard d={data} />
    case 'leaderboard':
      return <LeaderboardCard d={data} />
    case 'critic':
      return <CriticCard d={data} />
    case 'final':
      return <FinalCard d={data} onShowDecisions={onShowDecisions} />
    case 'published':
      return <PublishedCard d={data} />
    default:
      return null
  }
}
