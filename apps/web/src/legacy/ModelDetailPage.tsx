// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Boxes,
  ChevronRight,
  CircleCheck,
  Clock,
  Code2,
  Database,
  Download,
  FlaskConical,
  GitBranch,
  Info,
  Layers,
  Lightbulb,
  ListChecks,
  LoaderCircle,
  Rocket,
  ScrollText,
  ShieldCheck,
  Sparkles,
  Table2,
  Target,
  Wand2,
  Zap,
} from 'lucide-react'
import { useStore } from './store'
import { compact, fmtDate, num, rng, seedFrom, timeAgo } from '../lib/format'
import { Badge, Button, Card, CardHeader, CodeBlock, ConnectorLogo, Empty, KindIcon, ProgressBar, Rich, StatusPill, Tabs, cx } from './ui'
import { ImportanceBars, ModelEvalCharts, UsageChart } from './charts'

const SEVERITY = {
  high: ['red', 'High'],
  medium: ['amber', 'Medium'],
  low: ['sky', 'Low'],
  pass: ['emerald', 'Pass'],
}

function improvement(m) {
  const a = parseFloat(String(m.primary.display).replace(/[^0-9.]/g, ''))
  const b = parseFloat(String(m.baseline.display).replace(/[^0-9.]/g, ''))
  if (!a || !b) return null
  const rel = m.better === 'lower' ? (b - a) / b : (a - b) / b
  return Math.round(rel * 100)
}

function downloadModelCard(m) {
  const card = {
    model: m.name,
    slug: m.slug,
    version: m.version,
    problem_type: m.problemType,
    algorithm: m.algorithm,
    objective: m.objective,
    target: m.dataset.target,
    entity: m.dataset.entity,
    metrics: m.metrics,
    baseline: m.baseline,
    validation: m.strategy.validation,
    features: m.features.map((f) => f.name),
    decision_log: m.decisions,
    endpoint: m.endpoint,
    exported_at: new Date().toISOString(),
  }
  const blob = new Blob([JSON.stringify(card, null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${m.slug}_${m.version}_model_card.json`
  a.click()
  URL.revokeObjectURL(a.href)
}

/* ---------------------------------- Overview --------------------------------- */

function Lineage({ m }) {
  const steps = [
    { icon: Database, title: m.source.connector, sub: `${m.source.database} · ${m.source.tables.length} tables` },
    { icon: Table2, title: `${compact(m.dataset.rows)} rows`, sub: `${m.dataset.cols} raw columns` },
    { icon: Layers, title: `${m.featureCount} features`, sub: `${m.features.filter((f) => f.genai).length} GenAI-synthesised` },
    { icon: FlaskConical, title: m.algorithm.split(' (')[0], sub: `${m.candidates.length} candidates compared` },
    { icon: Rocket, title: m.status === 'live' ? 'Deployed' : m.status === 'staging' ? 'Staging' : 'Not deployed', sub: m.deployment },
  ]
  return (
    <div className="flex flex-col gap-2 md:flex-row md:items-stretch">
      {steps.map((s, i) => (
        <div key={i} className="flex flex-1 items-center gap-2 md:flex-col md:items-stretch">
          <div className="flex flex-1 items-center gap-2.5 rounded-lg border border-slate-200 bg-slate-50/60 p-3">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-white text-slate-600 ring-1 ring-slate-200">
              <s.icon size={15} />
            </span>
            <div className="min-w-0">
              <div className="truncate text-[13px] font-semibold text-slate-900">{s.title}</div>
              <div className="truncate text-[11px] text-slate-500">{s.sub}</div>
            </div>
          </div>
          {i < steps.length - 1 && <ChevronRight size={16} className="hidden shrink-0 self-center text-slate-300 md:hidden" />}
        </div>
      ))}
    </div>
  )
}

function Overview({ m }) {
  const imp = improvement(m)
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card className="p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{m.primary.name}</div>
          <div className="mt-1 text-[28px] font-semibold tracking-tight tabular">{m.primary.display}</div>
          <div className="text-xs text-slate-500">
            {imp != null && <span className="mr-1 font-semibold text-emerald-700">{imp}% better</span>}than {m.baseline.name} ({m.baseline.display})
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Predictions · 30 days</div>
          <div className="mt-1 text-[28px] font-semibold tracking-tight tabular">
            {m.usage.predictions30d ? compact(m.usage.predictions30d) : '—'}
          </div>
          <div className="text-xs text-slate-500">{m.deployment}</div>
        </Card>
        <Card className="p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Latency p95</div>
          <div className="mt-1 text-[28px] font-semibold tracking-tight tabular">{m.usage.latency} ms</div>
          <div className="text-xs text-slate-500">Uptime {m.usage.uptime}</div>
        </Card>
        <Card className={cx('p-4', m.drift.status === 'warning' && 'border-amber-200 bg-amber-50/40')}>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Data drift</div>
          <div
            className={cx(
              'mt-1 flex items-center gap-2 text-[28px] font-semibold tracking-tight',
              m.drift.status === 'warning' ? 'text-amber-700' : 'text-emerald-700',
            )}
          >
            {m.drift.status === 'warning' ? <AlertTriangle size={22} /> : <ShieldCheck size={22} />}
            {m.drift.status === 'warning' ? 'Warning' : 'Stable'}
          </div>
          <div className="text-xs text-slate-500">{m.drift.note}</div>
        </Card>
      </div>

      <Card>
        <CardHeader title="Pipeline lineage" subtitle="From raw data to a serving endpoint, reproducible end to end" icon={GitBranch} />
        <div className="px-5 pb-5">
          <Lineage m={m} />
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-5">
        <div className="space-y-6 xl:col-span-3">
          <Card>
            <CardHeader title="What this model does" subtitle="Plain-language summary from the Explainer agent" icon={Lightbulb} />
            <div className="space-y-4 px-5 pb-5">
              <p className="text-[14px] leading-6 text-slate-700">{m.explanation}</p>
              <div className="rounded-lg border border-indigo-100 bg-indigo-50/60 p-4">
                <div className="mb-1 flex items-center gap-1.5 text-[12px] font-bold uppercase tracking-wider text-indigo-700">
                  <Sparkles size={13} />
                  Why this model?
                </div>
                <p className="text-[13px] leading-6 text-slate-700">{m.whyModel}</p>
              </div>
            </div>
          </Card>
          <Card>
            <CardHeader title="Model card" icon={Info} />
            <dl className="grid grid-cols-1 gap-x-6 border-t border-slate-100 sm:grid-cols-2">
              {[
                ['Objective', m.objective],
                ['Problem type', m.problemType],
                [
                  'Target',
                  <code key="t" className="font-mono text-[12px]">
                    {m.dataset.target}
                  </code>,
                ],
                [
                  'Entity / grain',
                  <code key="e" className="font-mono text-[12px]">
                    {m.dataset.entity}
                  </code>,
                ],
                ['Optimisation metric', m.strategy.metric],
                ['Validation', m.strategy.validation],
                ['Training window', m.dataset.period],
                ['Requested by', `${m.requestedBy} · built by ${m.owner}`],
              ].map(([k, v]) => (
                <div key={k} className="border-b border-slate-100 px-5 py-2.5">
                  <dt className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{k}</dt>
                  <dd className="mt-0.5 text-[13px] text-slate-800">{v}</dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>
        <Card className="xl:col-span-2">
          <CardHeader title="Top drivers" subtitle="Global feature importance (mean |SHAP|)" icon={BarChart3} />
          <div className="px-5 pb-5">
            <ImportanceBars items={m.importance} />
          </div>
        </Card>
      </div>
    </div>
  )
}

/* ------------------------------------ Data ----------------------------------- */

function QualityRing({ value }) {
  const r = 26,
    c = 2 * Math.PI * r
  const color = value >= 90 ? '#10b981' : value >= 80 ? '#f59e0b' : '#ef4444'
  return (
    <svg width="68" height="68" viewBox="0 0 68 68" className="shrink-0">
      <circle cx="34" cy="34" r={r} stroke="#eef2f7" strokeWidth="7" fill="none" />
      <circle
        cx="34"
        cy="34"
        r={r}
        stroke={color}
        strokeWidth="7"
        fill="none"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - value / 100)}
        strokeLinecap="round"
        transform="rotate(-90 34 34)"
      />
      <text x="34" y="39" textAnchor="middle" fontSize="16" fontWeight="600" fill="#0f172a">
        {value}
      </text>
    </svg>
  )
}

function roleColor(role) {
  if (role === 'Target') return 'emerald'
  if (role.startsWith('Leakage')) return 'red'
  if (/key|ID|index/i.test(role)) return 'indigo'
  return 'slate'
}

function DataTab({ m }) {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card className="flex items-center gap-4 p-4">
          <QualityRing value={m.dataset.quality} />
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Data quality score</div>
            <div className="text-xs text-slate-500">before cleaning · now 98 after fixes</div>
          </div>
        </Card>
        {[
          ['Rows', num(m.dataset.rows), m.dataset.balance],
          ['Columns', `${m.dataset.cols} raw → ${m.featureCount}`, 'raw columns → model features'],
          ['Time period', m.dataset.period, `entity: ${m.dataset.entity}`],
        ].map(([k, v, s]) => (
          <Card key={k} className="p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{k}</div>
            <div className="mt-1 truncate text-lg font-semibold tracking-tight tabular">{v}</div>
            <div className="truncate text-xs text-slate-500">{s}</div>
          </Card>
        ))}
      </div>

      <Card className="overflow-hidden">
        <CardHeader
          title="Source tables"
          subtitle={`${m.source.connector} · ${m.source.database} · point-in-time joins`}
          icon={Database}
          right={<ConnectorLogo id={m.source.connectorId} size={30} />}
        />
        <div className="thin-scroll overflow-x-auto">
          <table className="w-full min-w-[560px] text-left">
            <thead>
              <tr className="border-y border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                <th className="px-5 py-2.5">Table</th>
                <th className="px-3 py-2.5">Role</th>
                <th className="px-3 py-2.5 text-right">Rows</th>
                <th className="px-5 py-2.5">Join condition</th>
              </tr>
            </thead>
            <tbody>
              {m.source.tables.map((t) => (
                <tr key={t.name} className="border-b border-slate-100 last:border-0">
                  <td className="px-5 py-2.5 font-mono text-[12px] font-semibold">
                    {m.source.database}.{t.name}
                  </td>
                  <td className="px-3 py-2.5">
                    <Badge>{t.role}</Badge>
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono text-[12px] tabular">{num(t.rows)}</td>
                  <td className="px-5 py-2.5 font-mono text-[12px] text-slate-600">{t.join}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card className="overflow-hidden">
        <CardHeader title="Schema & column profile" subtitle="Detected types and roles assigned by the Data Analysis agent" icon={Table2} />
        <div className="thin-scroll overflow-x-auto">
          <table className="w-full min-w-[640px] text-left">
            <thead>
              <tr className="border-y border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                <th className="px-5 py-2.5">Column</th>
                <th className="px-3 py-2.5">Type</th>
                <th className="px-3 py-2.5">Role</th>
                <th className="px-3 py-2.5 w-48">Missing</th>
                <th className="px-5 py-2.5 text-right">Unique</th>
              </tr>
            </thead>
            <tbody>
              {m.schema.map((c) => (
                <tr key={c.name} className={cx('border-b border-slate-100 last:border-0', c.role === 'Target' && 'bg-emerald-50/40')}>
                  <td className={cx('px-5 py-2.5 font-mono text-[12px]', c.role.startsWith('Leakage') && 'text-slate-400 line-through')}>{c.name}</td>
                  <td className="px-3 py-2.5 font-mono text-[11px] text-slate-500">{c.type}</td>
                  <td className="px-3 py-2.5">
                    <Badge color={roleColor(c.role)}>{c.role}</Badge>
                  </td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={cx('h-full rounded-full', c.missing > 20 ? 'bg-amber-500' : 'bg-slate-400')}
                          style={{ width: `${Math.max(c.missing, c.missing ? 3 : 0)}%` }}
                        />
                      </div>
                      <span className="font-mono text-[11px] tabular text-slate-600">{c.missing}%</span>
                    </div>
                  </td>
                  <td className="px-5 py-2.5 text-right font-mono text-[12px] tabular text-slate-600">{c.unique}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="Cleaning applied" subtitle="Each fix is recorded in the decision log" icon={ListChecks} />
          <ul className="divide-y divide-slate-100 border-t border-slate-100">
            {m.cleaning.map((c) => (
              <li key={c.issue} className="flex gap-3 px-5 py-3">
                <CircleCheck size={16} className="mt-0.5 shrink-0 text-emerald-500" />
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-semibold text-slate-900">{c.issue}</div>
                  <div className="text-[12px] text-slate-600">{c.action}</div>
                </div>
                <span className="shrink-0 font-mono text-[11px] text-slate-500">{c.count}</span>
              </li>
            ))}
          </ul>
        </Card>
        <Card>
          <CardHeader title="Engineered features" subtitle={`${m.featureCount} total · top features shown`} icon={Layers} />
          <div className="grid gap-2 border-t border-slate-100 p-4 sm:grid-cols-2">
            {m.features.map((f) => (
              <div
                key={f.name}
                className={cx('rounded-lg border p-2.5', f.genai ? 'border-indigo-200 bg-indigo-50/60' : 'border-slate-200 bg-white')}
              >
                <div className="flex items-center justify-between gap-2">
                  <code className={cx('truncate font-mono text-[12px] font-semibold', f.genai ? 'text-indigo-800' : 'text-slate-800')}>{f.name}</code>
                  {f.genai && <Sparkles size={12} className="shrink-0 text-indigo-500" />}
                </div>
                <div className="mt-0.5 text-[11px] text-slate-500">{f.desc}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}

/* ---------------------------------- Metrics ---------------------------------- */

export function Leaderboard({ candidates, metricName, better }) {
  const scores = candidates.map((c) => c.score)
  const lo = Math.min(...scores),
    hi = Math.max(...scores)
  const width = (s) => {
    if (hi === lo) return 100
    const t = (s - lo) / (hi - lo)
    return 25 + 75 * (better === 'lower' ? 1 - t : t)
  }
  const sorted = [...candidates].sort((a, b) => (better === 'lower' ? a.score - b.score : b.score - a.score))
  return (
    <div className="space-y-2">
      {sorted.map((c, i) => (
        <div
          key={c.name}
          className={cx(
            'grid grid-cols-[1.5rem_minmax(0,1fr)_7.5rem] items-center gap-3 rounded-lg px-3 py-2',
            c.selected ? 'bg-emerald-50 ring-1 ring-emerald-200' : 'bg-slate-50',
          )}
        >
          <span className="font-mono text-[12px] font-semibold text-slate-400">#{i + 1}</span>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate text-[13px] font-semibold text-slate-900">{c.name}</span>
              <span className="hidden text-[11px] text-slate-500 sm:inline">{c.family}</span>
              {c.selected && <Badge color="emerald">Selected</Badge>}
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white">
              <div className={cx('h-full rounded-full', c.selected ? 'bg-emerald-500' : 'bg-slate-300')} style={{ width: `${width(c.score)}%` }} />
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-[13px] font-semibold tabular">{c.display}</div>
            <div className="whitespace-nowrap text-[10px] text-slate-500">
              {metricName} · {c.time}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function MetricsTab({ m }) {
  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {m.metrics.map((x) => (
          <Card key={x.name} className="p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{x.name}</div>
            <div className="mt-1 text-[22px] font-semibold tracking-tight tabular">{x.display}</div>
            <div className="text-[11px] text-slate-500">baseline {x.baseline}</div>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader title="Evaluation" subtitle={`${m.strategy.validation}`} icon={Target} />
        <div className="px-5 pb-5">
          <ModelEvalCharts model={m} />
        </div>
      </Card>
      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader
            title="Model comparison"
            subtitle={`${m.candidates.length} candidates · ranked by ${m.metricName} (${m.better} is better)`}
            icon={FlaskConical}
          />
          <div className="px-5 pb-5">
            <Leaderboard candidates={m.candidates} metricName={m.metricName} better={m.better} />
          </div>
        </Card>
        <Card>
          <CardHeader title="Feature importance" subtitle="Mean |SHAP value| on holdout" icon={BarChart3} />
          <div className="px-5 pb-5">
            <ImportanceBars items={m.importance} />
          </div>
        </Card>
      </div>
    </div>
  )
}

/* --------------------------------- Decisions --------------------------------- */

function DecisionsTab({ m }) {
  return (
    <div className="grid gap-6 xl:grid-cols-3">
      <Card className="xl:col-span-2">
        <CardHeader title="Decision log" subtitle="Every significant choice, the alternatives considered and the evidence" icon={ScrollText} />
        <ol className="border-t border-slate-100 px-5 py-4">
          {m.decisions.map((d, i) => (
            <li key={i} className="relative flex gap-4 pb-5 last:pb-0">
              {i < m.decisions.length - 1 && <span className="absolute left-[15px] top-8 bottom-0 w-px bg-slate-200" />}
              <span className="z-10 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-slate-900 font-mono text-[11px] font-bold text-white">
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0 flex-1 rounded-lg border border-slate-200 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge color="indigo">{d.step}</Badge>
                  <span className="text-[13px] font-semibold text-slate-900">{d.decision}</span>
                </div>
                <p className="mt-1.5 text-[13px] text-slate-600">
                  <span className="font-semibold text-slate-700">Reason: </span>
                  {d.reason}
                </p>
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
        <CardHeader title="Critic agent review" subtitle="Automatic checks for leakage, overfitting & bias" icon={ShieldCheck} />
        <ul className="space-y-2 border-t border-slate-100 p-4">
          {m.critic.map((f) => {
            const [color, label] = SEVERITY[f.severity] || SEVERITY.low
            return (
              <li key={f.title} className="rounded-lg border border-slate-200 p-3">
                <div className="flex items-start justify-between gap-2">
                  <span className="text-[13px] font-semibold text-slate-900">{f.title}</span>
                  <Badge color={color}>{label}</Badge>
                </div>
                <p className="mt-1 text-[12px] text-slate-600">{f.detail}</p>
              </li>
            )
          })}
        </ul>
      </Card>
    </div>
  )
}

/* ----------------------------------- Usage ----------------------------------- */

function recentPredictions(m) {
  const r = rng(seedFrom(m.slug))
  return Array.from({ length: 6 }, (_, i) => {
    const id = `${m.dataset.entity.replace(/_id$/, '').slice(0, 4)}_${Math.floor(10000 + r() * 89999)}`
    let out
    if (m.kind === 'classification') {
      const p = r()
      out = `${(p * 100).toFixed(0)}% · ${p > 0.6 ? 'High' : p > 0.3 ? 'Medium' : 'Low'} risk`
    } else if (m.kind === 'anomaly') {
      const p = Math.pow(r(), 3)
      out = `score ${p.toFixed(2)} · ${p > 0.34 ? 'review' : 'approve'}`
    } else if (m.kind === 'forecasting') out = `${num(Math.round(1100 + r() * 500))} units · next 13 wks`
    else if (m.kind === 'regression') out = `$${num(Math.round(250 + r() * 600) * 1000)}`
    else out = ['Champions', 'Loyal', 'Promising', 'At Risk', 'Hibernating'][Math.floor(r() * 5)]
    return { id, out, at: Date.now() - (i * 47 + Math.floor(r() * 40)) * 1000, ms: Math.round(m.usage.latency * (0.6 + r() * 0.5)) }
  })
}

function UsageTab({ m }) {
  const [lang, setLang] = useState('curl')
  const preds = useMemo(() => recentPredictions(m), [m])
  if (m.status === 'training' || m.status === 'draft') {
    return (
      <Card>
        <Empty
          icon={Rocket}
          title="Not deployed yet"
          text="Once training finishes and you approve deployment, the endpoint, code snippets and live usage will appear here."
        />
      </Card>
    )
  }
  const body = JSON.stringify(m.sampleInput, null, 2)
  const code = {
    curl: `curl -X POST ${m.endpoint} \\\n  -H "Authorization: Bearer $DATALYON_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(m.sampleInput)}'`,
    python: `from datalyon import Client\n\nclient = Client(api_key=os.environ["DATALYON_API_KEY"])\nmodel = client.models.get("${m.slug}", version="${m.version}")\n\nresult = model.predict(${body.replace(/\n/g, '\n  ')})\nprint(result)`,
    sql: `-- Batch scoring directly in your warehouse\nSELECT *\nFROM TABLE(DATALYON.PREDICT(\n  model   => '${m.slug}',\n  version => '${m.version}',\n  input   => (SELECT * FROM ${m.source.database}.${m.source.tables[0].name})\n));`,
  }
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['Predictions · 30 days', compact(m.usage.predictions30d), m.deployment],
          ['Latency p95', `${m.usage.latency} ms`, 'measured at the endpoint'],
          ['Uptime', m.usage.uptime, 'last 30 days'],
          ['Serving version', m.version, `updated ${timeAgo(m.updatedAt)}`],
        ].map(([k, v, s]) => (
          <Card key={k} className="p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{k}</div>
            <div className="mt-1 text-[22px] font-semibold tracking-tight tabular">{v}</div>
            <div className="truncate text-xs text-slate-500">{s}</div>
          </Card>
        ))}
      </div>
      <div className="grid gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-3">
          <CardHeader title="Use this model" subtitle="REST endpoint · Python SDK · in-warehouse SQL" icon={Code2} />
          <div className="space-y-3 px-5 pb-5">
            <div className="flex items-center gap-2 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
              <Badge color="emerald" mono>
                POST
              </Badge>
              <code className="truncate font-mono text-[12px] text-slate-700">{m.endpoint}</code>
            </div>
            <div className="flex gap-1">
              {['curl', 'python', 'sql'].map((l) => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={cx(
                    'rounded-md px-2.5 py-1 font-mono text-[12px] font-semibold',
                    lang === l ? 'bg-slate-900 text-white' : 'text-slate-500 hover:bg-slate-100',
                  )}
                >
                  {l}
                </button>
              ))}
            </div>
            <CodeBlock code={code[lang]} label={lang === 'curl' ? 'request.sh' : lang === 'python' ? 'predict.py' : 'batch_score.sql'} />
            <CodeBlock code={JSON.stringify(m.sampleOutput, null, 2)} label="response.json" />
          </div>
        </Card>
        <div className="space-y-6 xl:col-span-2">
          <Card>
            <CardHeader title="Prediction volume" subtitle="Daily calls, last 30 days" icon={Zap} />
            {m.usage.series.filter((p) => p.value > 0).length >= 5 ? (
              <div className="px-3 pb-3">
                <UsageChart series={m.usage.series} height={180} />
              </div>
            ) : (
              <div className="mx-5 mb-5 rounded-lg border border-dashed border-emerald-300 bg-emerald-50/50 p-4 text-[13px] text-slate-700">
                <div className="flex items-center gap-2 font-semibold text-emerald-800">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
                  Just deployed
                </div>
                <p className="mt-1">
                  {num(m.usage.predictions30d)} smoke-test predictions so far. The daily volume chart fills in as production traffic arrives.
                </p>
              </div>
            )}
          </Card>
          <Card>
            <CardHeader title="Recent predictions" subtitle="Live request log" icon={Clock} />
            <ul className="divide-y divide-slate-100 border-t border-slate-100">
              {preds.map((p) => (
                <li key={p.id} className="flex items-center justify-between gap-3 px-5 py-2.5 text-[12px]">
                  <code className="font-mono text-slate-600">{p.id}</code>
                  <span className="truncate font-medium text-slate-900">{p.out}</span>
                  <span className="shrink-0 font-mono text-[11px] text-slate-400">
                    {p.ms} ms · {timeAgo(p.at)}
                  </span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </div>
    </div>
  )
}

/* ---------------------------------- Versions --------------------------------- */

function VersionsTab({ m }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader title="Version history" subtitle="Every version is reproducible from its pipeline snapshot" icon={GitBranch} />
      <div className="thin-scroll overflow-x-auto">
        <table className="w-full min-w-[600px] text-left">
          <thead>
            <tr className="border-y border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              <th className="px-5 py-2.5">Version</th>
              <th className="px-3 py-2.5">Date</th>
              <th className="px-3 py-2.5 text-right">{m.primary.name}</th>
              <th className="px-3 py-2.5">Change</th>
              <th className="px-5 py-2.5">Status</th>
            </tr>
          </thead>
          <tbody>
            {m.versions.map((v) => (
              <tr key={v.version} className="border-b border-slate-100 last:border-0">
                <td className="px-5 py-3 font-mono text-[13px] font-semibold">{v.version}</td>
                <td className="px-3 py-3 text-[12px] text-slate-600">{fmtDate(v.date)}</td>
                <td className="px-3 py-3 text-right font-mono text-[13px] tabular">{v.metric}</td>
                <td className="px-3 py-3 text-[13px] text-slate-700">{v.note}</td>
                <td className="px-5 py-3">{v.status === 'archived' ? <Badge>Archived</Badge> : <StatusPill status={v.status} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ----------------------------------- Page ------------------------------------ */

export default function ModelDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { models, patchModel, addActivity } = useStore()
  const [tab, setTab] = useState('overview')
  const m = models.find((x) => x.id === id)

  if (!m) {
    return (
      <Card>
        <Empty
          icon={Boxes}
          title="Model not found"
          text="It may have been removed, or the demo data was reset."
          action={<Button onClick={() => navigate('/models')}>Back to catalog</Button>}
        />
      </Card>
    )
  }

  const promote = () => {
    patchModel(m.id, (cur) => ({
      status: 'live',
      deployment: 'REST API · real-time',
      usage: { ...cur.usage, uptime: '100%' },
      versions: cur.versions.map((v, i) => (i === 0 ? { ...v, status: 'live' } : v)),
    }))
    addActivity('deploy', `Promoted **${m.name} ${m.version}** to production`)
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'data', label: 'Data', icon: Database },
    { id: 'metrics', label: 'Metrics', icon: BarChart3 },
    { id: 'decisions', label: 'Decision log', icon: ScrollText, count: m.decisions.length },
    { id: 'usage', label: 'Usage & API', icon: Code2 },
    { id: 'versions', label: 'Versions', icon: GitBranch, count: m.versions.length },
  ]

  return (
    <div className="space-y-5">
      <nav className="flex items-center gap-1.5 text-[13px] text-slate-500">
        <Link to="/models" className="hover:text-slate-900">
          Model Catalog
        </Link>
        <ChevronRight size={14} />
        <span className="truncate font-medium text-slate-900">{m.name}</span>
      </nav>

      <Card className="p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex min-w-0 gap-4">
            <KindIcon kind={m.kind} size="lg" />
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-[22px] font-semibold tracking-tight text-slate-900">{m.name}</h1>
                <StatusPill status={m.status} progress={m.status === 'training' ? m.progress : null} />
                <Badge color="dark" mono>
                  {m.version}
                </Badge>
                {m.createdBy === 'agent' && (
                  <Badge color="indigo">
                    <Sparkles size={10} />
                    Built in this session
                  </Badge>
                )}
              </div>
              <p className="mt-1 text-[14px] text-slate-600">{m.objective}</p>
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-slate-500">
                <span className="font-mono">{m.slug}</span>
                <span>{m.problemType}</span>
                <span className="font-mono">{m.algorithm}</span>
                <span>Created {fmtDate(m.createdAt)}</span>
                {(m.tags || []).map((t) => (
                  <Badge key={t}>#{t}</Badge>
                ))}
              </div>
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <Button icon={Download} onClick={() => downloadModelCard(m)}>
              Download
            </Button>
            <Button icon={Wand2} onClick={() => navigate(`/agent?goal=${encodeURIComponent(m.objective)}`)}>
              Retrain with agent
            </Button>
            {m.status === 'staging' && (
              <Button variant="primary" icon={Rocket} onClick={promote}>
                Promote to production
              </Button>
            )}
            {m.status === 'live' && (
              <Button variant="dark" icon={Code2} onClick={() => setTab('usage')}>
                Use model
              </Button>
            )}
          </div>
        </div>
        {m.status === 'training' && (
          <div className="mt-4 flex items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
            <LoaderCircle size={18} className="shrink-0 animate-spin text-amber-600" />
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-semibold text-amber-900">Agent is working: {m.stage}</div>
              <ProgressBar value={m.progress} animated className="mt-1.5 bg-amber-100" />
            </div>
            <span className="font-mono text-[13px] font-semibold text-amber-800 tabular">{m.progress}%</span>
          </div>
        )}
        {m.status === 'draft' && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <span className="text-[13px] text-slate-700">
              <Rich text={`This model is a **draft**: ${m.stage}.`} />
            </span>
            <Button size="sm" variant="primary" iconRight={ArrowRight} onClick={() => navigate(`/agent?goal=${encodeURIComponent(m.objective)}`)}>
              Continue with agent
            </Button>
          </div>
        )}
        {m.drift?.status === 'warning' && m.status === 'live' && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
            <span className="flex items-center gap-2 text-[13px] text-amber-900">
              <AlertTriangle size={16} />
              {m.drift.note}
            </span>
            <Button size="sm" icon={Wand2} onClick={() => navigate(`/agent?goal=${encodeURIComponent(m.objective)}`)}>
              Retrain now
            </Button>
          </div>
        )}
      </Card>

      <Tabs tabs={tabs} value={tab} onChange={setTab} />
      <div key={tab} className="animate-fade-in">
        {tab === 'overview' && <Overview m={m} />}
        {tab === 'data' && <DataTab m={m} />}
        {tab === 'metrics' && <MetricsTab m={m} />}
        {tab === 'decisions' && <DecisionsTab m={m} />}
        {tab === 'usage' && <UsageTab m={m} />}
        {tab === 'versions' && <VersionsTab m={m} />}
      </div>
    </div>
  )
}
