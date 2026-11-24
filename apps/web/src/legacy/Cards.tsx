// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { useState } from 'react'
import {
  ArrowRight,
  BarChart3,
  Boxes,
  CircleCheck,
  Database,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  Layers,
  LoaderCircle,
  Rocket,
  ScrollText,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
  Wand2,
} from 'lucide-react'
import { Badge, Button, CodeBlock, ConnectorLogo, KindIcon, ProgressBar, Rich, cx } from './ui'
import { ImportanceBars, ModelEvalCharts } from './charts'
import { num } from '../lib/format'

export function CardShell({ icon: Icon, title, subtitle, tone = 'slate', right, children, className }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-600',
    ai: 'bg-indigo-100 text-indigo-600',
    good: 'bg-emerald-100 text-emerald-700',
    warn: 'bg-amber-100 text-amber-700',
  }
  return (
    <div
      className={cx('animate-fade-up overflow-hidden rounded-xl border border-slate-200 bg-white shadow-[0_1px_3px_rgba(15,23,42,0.05)]', className)}
    >
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 bg-slate-50/70 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className={cx('grid h-6 w-6 shrink-0 place-items-center rounded-md', tones[tone])}>
            <Icon size={13} />
          </span>
          <div className="min-w-0">
            <div className="truncate text-[13px] font-semibold text-slate-900">{title}</div>
            {subtitle && <div className="truncate text-[11px] text-slate-500">{subtitle}</div>}
          </div>
        </div>
        {right}
      </div>
      <div className="p-4">{children}</div>
    </div>
  )
}

export function DetectionCard({ spec }) {
  const conf = Math.round(spec.detection.confidence * 100)
  return (
    <CardShell
      icon={Target}
      title="Problem detection"
      subtitle="problem_detector · goal wording + data signals"
      tone="ai"
      right={<Badge color="emerald">{conf}% confidence</Badge>}
    >
      <div className="flex items-center gap-3">
        <KindIcon kind={spec.kind} size="lg" />
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Detected problem type</div>
          <div className="text-lg font-semibold tracking-tight text-slate-900">{spec.problemType}</div>
        </div>
      </div>
      <ProgressBar value={conf} className="mt-3" />
      <ul className="mt-3 space-y-1.5">
        {spec.detection.signals.map((s) => (
          <li key={s} className="flex gap-2 text-[13px] text-slate-700">
            <CircleCheck size={15} className="mt-0.5 shrink-0 text-emerald-500" />
            <span>{s}</span>
          </li>
        ))}
      </ul>
    </CardShell>
  )
}

export function ProfileCard({ spec, target }) {
  const d = spec.dataset
  return (
    <CardShell
      icon={Database}
      title="Data profile"
      subtitle={`${spec.source.connector} · ${spec.source.database}`}
      right={<ConnectorLogo id={spec.source.connectorId} size={24} />}
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ['Rows', num(d.rows)],
          ['Tables', spec.source.tables.length],
          ['Columns', d.cols],
          ['Quality', `${d.quality}/100`],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg bg-slate-50 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{k}</div>
            <div className="text-[15px] font-semibold tabular text-slate-900">{v}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-[12px] text-slate-500">
        {d.period} · {d.balance}
      </div>
      <div className="thin-scroll mt-3 overflow-x-auto rounded-lg border border-slate-200">
        <table className="w-full min-w-[460px] text-left">
          <thead>
            <tr className="bg-slate-50 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              <th className="px-3 py-1.5">Column</th>
              <th className="px-2 py-1.5">Type</th>
              <th className="px-2 py-1.5">Role</th>
              <th className="px-3 py-1.5 text-right">Missing</th>
            </tr>
          </thead>
          <tbody>
            {spec.schema.slice(0, 9).map((c) => {
              const role = c.role.startsWith('Leakage')
                ? 'Feature'
                : c.name === spec.dataset.target && target !== spec.dataset.target
                  ? 'Feature'
                  : c.role
              return (
                <tr key={c.name} className={cx('border-t border-slate-100', c.name === target && 'bg-emerald-50/60')}>
                  <td className="px-3 py-1.5 font-mono text-[12px]">{c.name}</td>
                  <td className="px-2 py-1.5 font-mono text-[11px] text-slate-500">{c.type}</td>
                  <td className="px-2 py-1.5">
                    <Badge color={c.name === target ? 'emerald' : /key|ID|index/i.test(role) ? 'indigo' : 'slate'}>
                      {c.name === target ? 'Target' : role}
                    </Badge>
                  </td>
                  <td className={cx('px-3 py-1.5 text-right font-mono text-[11px] tabular', c.missing > 20 ? 'text-amber-700' : 'text-slate-500')}>
                    {c.missing}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </CardShell>
  )
}

export function StrategyCard({ spec }) {
  const s = spec.strategy
  return (
    <CardShell icon={Gauge} title="ML strategy" subtitle="Metric & validation chosen for this problem type">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-emerald-200 bg-emerald-50/50 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-emerald-700">Optimisation metric</div>
          <div className="mt-0.5 text-[15px] font-semibold text-slate-900">{s.metric}</div>
          <p className="mt-1 text-[12px] leading-5 text-slate-600">{s.metricWhy}</p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Validation scheme</div>
          <div className="mt-0.5 text-[15px] font-semibold text-slate-900">{s.validation}</div>
          <p className="mt-1 text-[12px] leading-5 text-slate-600">{s.validationWhy}</p>
        </div>
      </div>
      <dl className="mt-3 grid gap-x-4 gap-y-1.5 sm:grid-cols-3">
        {s.extras.map(([k, v]) => (
          <div key={k}>
            <dt className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{k}</dt>
            <dd className="text-[12px] font-medium text-slate-800">{v}</dd>
          </div>
        ))}
      </dl>
    </CardShell>
  )
}

export function CleaningCard({ spec }) {
  return (
    <CardShell icon={ShieldCheck} title="Data cleaning" subtitle={`${spec.cleaning.length} issues found and fixed`} tone="good">
      <ul className="space-y-2">
        {spec.cleaning.map((c) => (
          <li key={c.issue} className="flex gap-2.5">
            <CircleCheck size={15} className="mt-0.5 shrink-0 text-emerald-500" />
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-medium text-slate-900">{c.issue}</div>
              <div className="text-[12px] text-slate-600">{c.action}</div>
            </div>
            <span className="shrink-0 font-mono text-[11px] text-slate-400">{c.count}</span>
          </li>
        ))}
      </ul>
    </CardShell>
  )
}

export function FeaturesCard({ spec }) {
  return (
    <CardShell icon={Layers} title="Feature engineering" subtitle={`${spec.featureCount} features · leakage guard passed`} tone="ai">
      <div className="flex flex-wrap gap-1.5">
        {spec.features.map((f) => (
          <span
            key={f.name}
            title={f.desc}
            className={cx(
              'inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[11px]',
              f.genai ? 'border-indigo-200 bg-indigo-50 text-indigo-800' : 'border-slate-200 bg-slate-50 text-slate-700',
            )}
          >
            {f.genai && <Sparkles size={10} />}
            {f.name}
          </span>
        ))}
        <span className="rounded-md border border-dashed border-slate-300 px-2 py-1 font-mono text-[11px] text-slate-500">
          +{spec.featureCount - spec.features.length} more
        </span>
      </div>
      <div className="mt-3 flex items-center gap-4 text-[11px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-indigo-400" />
          GenAI-synthesised
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-slate-300" />
          Standard
        </span>
      </div>
    </CardShell>
  )
}

export function LeaderboardCard({ data, spec }) {
  const { rows, trials, round, done } = data
  const finished = rows.filter((r) => r.score != null)
  const dir = spec.better === 'lower' ? 1 : -1
  const leader = finished.length ? [...finished].sort((a, b) => dir * (a.score - b.score))[0].name : null
  return (
    <CardShell
      icon={FlaskConical}
      title={round === 1 ? 'Model search · round 1' : 'Model search · round 2 (after critic fixes)'}
      subtitle={`${rows.length} candidates · ${spec.metricName} (${spec.better} is better) · ${spec.strategy.validation.split(' ·')[0]}`}
      tone={done ? 'good' : 'warn'}
      right={
        <span className="flex items-center gap-1.5 font-mono text-[11px] text-slate-500 tabular">
          {!done && <LoaderCircle size={12} className="animate-spin text-amber-500" />}
          HPO {trials}/{spec.hpoTrials}
        </span>
      }
    >
      <div className="space-y-2">
        {rows.map((r) => {
          const isLeader = done && r.name === leader
          return (
            <div
              key={r.name}
              className={cx(
                'grid grid-cols-[minmax(0,1fr)_6.5rem] items-center gap-3 rounded-lg px-3 py-2 transition-colors',
                isLeader ? 'bg-emerald-50 ring-1 ring-emerald-200' : 'bg-slate-50',
              )}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate text-[13px] font-semibold text-slate-900">{r.name}</span>
                  <span className="hidden text-[11px] text-slate-500 sm:inline">{r.family}</span>
                  {isLeader && <Badge color="emerald">Leader</Badge>}
                </div>
                <ProgressBar
                  value={r.progress}
                  className="mt-1.5 bg-white"
                  color={r.score != null ? (isLeader ? 'bg-emerald-500' : 'bg-slate-400') : ''}
                  animated={r.score == null && r.progress > 0}
                />
              </div>
              <div className="text-right">
                {r.score != null ? (
                  <>
                    <div className="font-mono text-[13px] font-semibold tabular text-slate-900">{spec.fmt(r.score)}</div>
                    {r.prev != null && r.prev !== r.score && (
                      <div className="font-mono text-[10px] text-slate-400 tabular line-through">{spec.fmt(r.prev)}</div>
                    )}
                  </>
                ) : r.progress > 0 ? (
                  <div className="font-mono text-[12px] tabular text-slate-400">{r.interim != null ? spec.fmt(r.interim) : '…'}</div>
                ) : (
                  <div className="text-[11px] text-slate-400">queued</div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </CardShell>
  )
}

const SEV = { high: ['red', 'High'], medium: ['amber', 'Medium'], low: ['sky', 'Low'], pass: ['emerald', 'Pass'] }

export function CriticCard({ spec }) {
  return (
    <CardShell icon={ShieldAlert} title="Critic agent review" subtitle="Leakage · overfitting · residual patterns · metric fit" tone="warn">
      <p className="mb-3 text-[13px] text-slate-700">
        <Rich text={spec.critic.headline} />
      </p>
      <ul className="space-y-2">
        {spec.critic.findings.map((f) => {
          const [color, label] = SEV[f.severity]
          return (
            <li key={f.title} className={cx('rounded-lg border p-3', f.severity === 'high' ? 'border-red-200 bg-red-50/50' : 'border-slate-200')}>
              <div className="flex items-start justify-between gap-2">
                <span className="text-[13px] font-semibold text-slate-900">{f.title}</span>
                <Badge color={color}>{label}</Badge>
              </div>
              <p className="mt-1 text-[12px] text-slate-600">{f.detail}</p>
            </li>
          )
        })}
      </ul>
    </CardShell>
  )
}

export function FinalCard({ spec, model, onShowDecisions }) {
  const [why, setWhy] = useState(false)
  return (
    <CardShell
      icon={Sparkles}
      title="Final model"
      subtitle={`${spec.problemType} · ready for deployment`}
      tone="good"
      right={<Badge color="emerald">Validated</Badge>}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <KindIcon kind={spec.kind} size="lg" />
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Best model</div>
            <div className="text-lg font-semibold tracking-tight text-slate-900">{spec.final.algorithm}</div>
          </div>
        </div>
        <div className="flex items-end gap-4">
          <div className="text-right">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{spec.final.primary.name}</div>
            <div className="text-[28px] font-semibold leading-8 tracking-tight text-emerald-600 tabular">{spec.final.primary.display}</div>
          </div>
          <div className="pb-0.5 text-right text-[11px] text-slate-500">
            vs baseline
            <br />
            <span className="font-mono text-[13px] font-semibold text-slate-700">{spec.final.baseline.display}</span>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-5">
        {spec.final.metrics.map((m) => (
          <div key={m.name} className="rounded-lg bg-slate-50 px-2.5 py-2">
            <div className="truncate text-[10px] font-semibold uppercase tracking-wider text-slate-500">{m.name}</div>
            <div className="font-mono text-[14px] font-semibold tabular text-slate-900">{m.display}</div>
            <div className="truncate text-[10px] text-slate-400">base {m.baseline}</div>
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-lg border border-slate-200 p-3">
        <ModelEvalCharts model={model} compact />
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <div className="mb-2 flex items-center gap-1.5 text-[12px] font-semibold text-slate-900">
            <BarChart3 size={14} className="text-slate-400" />
            Top drivers
          </div>
          <ImportanceBars items={spec.final.importance} max={5} />
        </div>
        <div>
          <div className="mb-2 flex items-center gap-1.5 text-[12px] font-semibold text-slate-900">
            <Wand2 size={14} className="text-slate-400" />
            In plain language
          </div>
          <p className="text-[13px] leading-6 text-slate-700">{spec.final.explanation}</p>
        </div>
      </div>

      {why && (
        <div className="mt-4 animate-fade-up rounded-lg border border-indigo-100 bg-indigo-50/60 p-3 text-[13px] leading-6 text-slate-700">
          <div className="mb-1 text-[11px] font-bold uppercase tracking-wider text-indigo-700">Why this model?</div>
          {spec.final.whyModel}
        </div>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <Button size="sm" variant={why ? 'dark' : 'secondary'} icon={Sparkles} onClick={() => setWhy((v) => !v)}>
          Why this model?
        </Button>
        <Button size="sm" icon={ScrollText} onClick={onShowDecisions}>
          View decisions
        </Button>
      </div>
    </CardShell>
  )
}

export function DeployedCard({ model, mode, onNavigate }) {
  const live = mode !== 'staging'
  return (
    <CardShell
      icon={Rocket}
      title={live ? 'Deployed to production' : 'Registered in staging'}
      subtitle={`${model.name} · ${model.version}`}
      tone="good"
      right={
        <Badge color={live ? 'emerald' : 'indigo'}>
          <span className={cx('h-1.5 w-1.5 rounded-full', live ? 'animate-pulse bg-emerald-500' : 'bg-indigo-500')} />
          {live ? 'Live' : 'Staging'}
        </Badge>
      }
    >
      <div className="space-y-3">
        <div className="flex items-center gap-2 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
          <Badge color="emerald" mono>
            POST
          </Badge>
          <code className="truncate font-mono text-[12px] text-slate-700">{model.endpoint}</code>
        </div>
        {live && (
          <CodeBlock
            label="request.sh"
            code={`curl -X POST ${model.endpoint} \\\n  -H "Authorization: Bearer $DATALYON_API_KEY" \\\n  -d '${JSON.stringify(model.sampleInput)}'`}
          />
        )}
        <div className="grid grid-cols-3 gap-2 text-center">
          {[
            ['Mode', mode === 'api' ? 'Real-time API' : mode === 'batch' ? 'Daily batch' : 'Staging'],
            ['Latency p95', `${model.usage.latency} ms`],
            ['Smoke tests', '500 / 500 ✓'],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg bg-slate-50 px-2 py-2">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{k}</div>
              <div className="text-[13px] font-semibold text-slate-900">{v}</div>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap gap-2 pt-1">
          <Button size="sm" variant="primary" icon={Boxes} iconRight={ArrowRight} onClick={() => onNavigate(`/models/${model.id}`)}>
            Open model page
          </Button>
          <Button size="sm" icon={LayoutDashboard} onClick={() => onNavigate('/')}>
            Dashboard
          </Button>
        </div>
      </div>
    </CardShell>
  )
}
