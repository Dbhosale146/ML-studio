/** Evaluation charts. Every prop is real data computed by the backend from holdout or backtest predictions. */
import type { ReactNode } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { cx } from '../../lib/cx'
import { compact, num, pct } from '../../lib/format'

const AXIS = { fontSize: 11, fill: '#64748b' }
const TT = {
  contentStyle: { borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12, boxShadow: '0 4px 14px rgba(15,23,42,0.08)' },
  labelStyle: { color: '#0f172a', fontWeight: 600 },
}

export function ChartFrame({
  title,
  subtitle,
  children,
  height = 240,
  right,
}: {
  title?: ReactNode
  subtitle?: ReactNode
  children: ReactNode
  height?: number
  right?: ReactNode
}) {
  return (
    <figure className="min-w-0">
      {(title || right) && (
        <figcaption className="mb-2 flex items-end justify-between gap-2">
          <div>
            <div className="text-[13px] font-semibold text-slate-900">{title}</div>
            {subtitle && <div className="text-xs text-slate-500">{subtitle}</div>}
          </div>
          {right}
        </figcaption>
      )}
      <div style={{ height }}>{children}</div>
    </figure>
  )
}

export function RocChart({ points, auc, height = 240 }: { points: Array<{ fpr: number; tpr: number }>; auc?: number | null; height?: number }) {
  const data = points.map((p) => ({ ...p, chance: p.fpr }))
  return (
    <ChartFrame
      title="ROC curve"
      subtitle={auc != null ? `Area under the curve ${auc.toFixed(3)} · dashed = random` : 'dashed = random'}
      height={height}
    >
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 14, left: -14 }}>
          <CartesianGrid stroke="#eef2f7" vertical={false} />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tick={AXIS}
            tickCount={6}
            label={{ value: 'False positive rate', position: 'insideBottom', offset: -8, style: AXIS }}
          />
          <YAxis domain={[0, 1]} tick={AXIS} tickCount={6} />
          <Tooltip {...TT} formatter={(v) => Number(v).toFixed(3)} labelFormatter={(l) => `FPR ${Number(l).toFixed(2)}`} />
          <Area
            type="linear"
            dataKey="tpr"
            name="True positive rate"
            stroke="#10b981"
            strokeWidth={2}
            fill="#10b981"
            fillOpacity={0.12}
            isAnimationActive={false}
          />
          <Line type="linear" dataKey="chance" stroke="#94a3b8" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function PrChart({
  points,
  prevalence,
  height = 240,
}: {
  points: Array<{ recall: number; precision: number }>
  prevalence?: number | null
  height?: number
}) {
  return (
    <ChartFrame
      title="Precision–recall curve"
      subtitle={prevalence != null ? `Baseline precision = positive rate ${pct(prevalence)}` : undefined}
      height={height}
    >
      <ResponsiveContainer>
        <AreaChart data={points} margin={{ top: 6, right: 8, bottom: 14, left: -14 }}>
          <CartesianGrid stroke="#eef2f7" vertical={false} />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            tick={AXIS}
            tickCount={6}
            label={{ value: 'Recall', position: 'insideBottom', offset: -8, style: AXIS }}
          />
          <YAxis domain={[0, 1]} tick={AXIS} tickCount={6} />
          <Tooltip {...TT} formatter={(v) => Number(v).toFixed(3)} />
          {prevalence != null && <ReferenceLine y={prevalence} stroke="#94a3b8" strokeDasharray="4 4" />}
          <Area type="stepAfter" dataKey="precision" stroke="#6366f1" strokeWidth={2} fill="#6366f1" fillOpacity={0.12} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ConfusionMatrix({ labels, matrix, threshold }: { labels: string[]; matrix: number[][]; threshold?: number | null }) {
  const total = matrix.flat().reduce((a, b) => a + b, 0) || 1
  const max = Math.max(...matrix.flat(), 1)
  return (
    <div>
      <div className="text-[13px] font-semibold text-slate-900">Confusion matrix</div>
      <div className="mb-3 text-xs text-slate-500">
        Holdout · {num(total)} rows{threshold != null ? ` · threshold ${threshold.toFixed(2)}` : ''}
      </div>
      <div className="thin-scroll overflow-x-auto">
        <table className="text-xs" aria-label="Confusion matrix: rows are actual, columns are predicted">
          <thead>
            <tr>
              <th className="p-1">
                <span className="sr-only">Actual \ predicted</span>
              </th>
              {labels.map((l) => (
                <th key={l} scope="col" className="max-w-[90px] truncate p-1 text-center font-medium text-slate-500" title={l}>
                  pred. {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={labels[i]}>
                <th scope="row" className="max-w-[110px] truncate p-1 pr-2 text-right font-medium text-slate-500" title={labels[i]}>
                  actual {labels[i]}
                </th>
                {row.map((v, j) => (
                  <td key={j} className="p-0.5" aria-label={`actual ${labels[i]}, predicted ${labels[j]}: ${num(v)}`}>
                    <div
                      className={cx('grid min-w-[64px] place-items-center rounded-md px-2 py-2', i === j ? 'text-emerald-950' : 'text-red-950')}
                      style={{ background: i === j ? `rgba(16,185,129,${0.1 + (0.5 * v) / max})` : `rgba(239,68,68,${0.06 + (0.4 * v) / max})` }}
                    >
                      <span className="tabular text-sm font-semibold">{num(v)}</span>
                      <span className="tabular text-[10px] opacity-70">{((v / total) * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function CalibrationChart({
  points,
  height = 220,
}: {
  points: Array<{ predicted: number; observed: number; count: number }>
  height?: number
}) {
  const data = points.map((p) => ({ ...p, ideal: p.predicted }))
  return (
    <ChartFrame title="Calibration" subtitle="Predicted probability vs observed rate (dashed = perfect)" height={height}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 14, left: -14 }}>
          <CartesianGrid stroke="#eef2f7" />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[0, 1]}
            tick={AXIS}
            tickCount={6}
            label={{ value: 'Predicted', position: 'insideBottom', offset: -8, style: AXIS }}
          />
          <YAxis domain={[0, 1]} tick={AXIS} tickCount={6} />
          <Tooltip {...TT} formatter={(v) => Number(v).toFixed(3)} />
          <Line dataKey="ideal" stroke="#94a3b8" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
          <Line dataKey="observed" name="Observed" stroke="#f59e0b" strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ScoreDistChart({
  bins,
  threshold,
  labels,
  height = 220,
}: {
  bins: Array<{ bin: string; negative: number; positive: number }>
  threshold?: number | null
  labels: string[]
  height?: number
}) {
  return (
    <ChartFrame
      title="Score distribution"
      subtitle={`Holdout rows by predicted probability${threshold != null ? ` · threshold ${threshold.toFixed(2)}` : ''}`}
      height={height}
    >
      <ResponsiveContainer>
        <BarChart data={bins} margin={{ top: 6, right: 8, bottom: 0, left: -14 }} barGap={0}>
          <CartesianGrid stroke="#eef2f7" vertical={false} />
          <XAxis dataKey="bin" tick={AXIS} interval={3} />
          <YAxis tick={AXIS} scale="sqrt" />
          <Tooltip {...TT} />
          <Legend wrapperStyle={{ fontSize: 11 }} iconSize={8} />
          <Bar dataKey="negative" name={labels[0] ?? 'negative'} fill="#cbd5e1" isAnimationActive={false} />
          <Bar dataKey="positive" name={labels[1] ?? 'positive'} fill="#ef4444" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ScatterFit({ points, height = 250 }: { points: Array<{ actual: number; predicted: number }>; height?: number }) {
  const lo = Math.min(...points.map((p) => Math.min(p.actual, p.predicted)))
  const hi = Math.max(...points.map((p) => Math.max(p.actual, p.predicted)))
  return (
    <ChartFrame title="Actual vs predicted" subtitle={`${points.length} holdout rows · dashed = perfect prediction`} height={height}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 6, right: 8, bottom: 14, left: 4 }}>
          <CartesianGrid stroke="#eef2f7" />
          <XAxis
            dataKey="actual"
            type="number"
            name="Actual"
            tick={AXIS}
            domain={[lo, hi]}
            tickFormatter={(v) => compact(Number(v))}
            label={{ value: 'Actual', position: 'insideBottom', offset: -8, style: AXIS }}
          />
          <YAxis dataKey="predicted" type="number" name="Predicted" tick={AXIS} domain={[lo, hi]} tickFormatter={(v) => compact(Number(v))} />
          <Tooltip {...TT} cursor={{ strokeDasharray: '3 3' }} formatter={(v) => num(Number(v))} />
          <ReferenceLine
            segment={[
              { x: lo, y: lo },
              { x: hi, y: hi },
            ]}
            stroke="#94a3b8"
            strokeDasharray="4 4"
          />
          <Scatter data={points} fill="#f59e0b" fillOpacity={0.55} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ResidualChart({ bins, height = 220 }: { bins: Array<{ x0: number; x1: number; count: number }>; height?: number }) {
  const data = bins.map((b) => ({ ...b, label: compact((b.x0 + b.x1) / 2) }))
  return (
    <ChartFrame title="Residuals" subtitle="Actual − predicted on the holdout" height={height}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
          <CartesianGrid stroke="#eef2f7" vertical={false} />
          <XAxis dataKey="label" tick={AXIS} interval={4} />
          <YAxis tick={AXIS} />
          <Tooltip {...TT} />
          <Bar dataKey="count" fill="#f59e0b" radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export type ForecastPoint = { date: string; actual?: number; backtest?: number; forecast?: number; band?: [number, number] }

export function ForecastChart({
  points,
  title = 'Forecast',
  subtitle,
  height = 280,
}: {
  points: ForecastPoint[]
  title?: string
  subtitle?: string
  height?: number
}) {
  const today = points.find((p) => p.forecast != null)?.date
  return (
    <ChartFrame title={title} subtitle={subtitle ?? 'History, last backtest window and forecast with 80% interval'} height={height}>
      <ResponsiveContainer>
        <ComposedChart data={points} margin={{ top: 6, right: 8, bottom: 0, left: -2 }}>
          <CartesianGrid stroke="#eef2f7" vertical={false} />
          <XAxis dataKey="date" tick={AXIS} minTickGap={32} />
          <YAxis tick={AXIS} tickFormatter={(v) => compact(Number(v))} domain={['auto', 'auto']} />
          <Tooltip {...TT} formatter={(v, n) => [Array.isArray(v) ? `${num(Number(v[0]))} – ${num(Number(v[1]))}` : num(Number(v)), n]} />
          <Legend wrapperStyle={{ fontSize: 11 }} iconSize={8} />
          <Area dataKey="band" name="80% interval" stroke="none" fill="#10b981" fillOpacity={0.15} isAnimationActive={false} />
          <Line dataKey="actual" name="Actual" stroke="#0f172a" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line
            dataKey="backtest"
            name="Backtest forecast"
            stroke="#6366f1"
            strokeWidth={1.25}
            strokeDasharray="3 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line dataKey="forecast" name="Forecast" stroke="#10b981" strokeWidth={2.25} dot={false} isAnimationActive={false} />
          {today && <ReferenceLine x={today} stroke="#94a3b8" strokeDasharray="4 4" />}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function BacktestChart({
  windows,
  metric,
  metricLabel,
  height = 200,
}: {
  windows: Array<Record<string, number | string>>
  metric: string
  metricLabel: string
  height?: number
}) {
  const pctMetric = metric === 'mape' || metric === 'wape'
  const fmt = (v: unknown) => (pctMetric ? pct(Number(v)) : num(Number(v)))
  return (
    <ChartFrame title="Backtest windows" subtitle={`${metricLabel} per rolling window vs seasonal naive`} height={height}>
      <ResponsiveContainer>
        <BarChart data={windows} margin={{ top: 6, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid stroke="#eef2f7" vertical={false} />
          <XAxis dataKey="window" tick={AXIS} />
          <YAxis tick={AXIS} tickFormatter={fmt} />
          <Tooltip {...TT} formatter={(v) => fmt(v)} labelFormatter={(l, p) => `${l} · from ${String(p?.[0]?.payload?.start ?? '')}`} />
          <Legend wrapperStyle={{ fontSize: 11 }} iconSize={8} />
          <Bar dataKey={metric} name="Model" fill="#10b981" radius={[4, 4, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="baseline" name="Seasonal naive" fill="#cbd5e1" radius={[4, 4, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ImportanceBars({ items, max = 8, labels }: { items: Array<[string, number]>; max?: number; labels?: Record<string, string> }) {
  const top = items.slice(0, max)
  const peak = Math.max(...top.map((i) => i[1]), 1e-9)
  if (!top.length) return <p className="text-xs text-slate-500">No feature carried measurable importance on the holdout.</p>
  return (
    <ul className="space-y-2" aria-label="Feature importance">
      {top.map(([name, v], idx) => (
        <li key={name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1">
          <span className="truncate font-mono text-[12px] text-slate-700" title={name}>
            {labels?.[name] ?? name}
          </span>
          <span className="tabular font-mono text-[11px] text-slate-500">{(v * 100).toFixed(1)}%</span>
          <div className="col-span-2 h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full transition-[width] duration-700"
              style={{ width: `${(v / peak) * 100}%`, background: idx < 3 ? '#10b981' : '#6ee7b7' }}
            />
          </div>
        </li>
      ))}
    </ul>
  )
}

export function PerClassTable({ rows }: { rows: Array<{ class: string; precision: number; recall: number; f1: number; support: number }> }) {
  return (
    <div className="thin-scroll overflow-x-auto">
      <table className="w-full text-left text-[12px]">
        <thead>
          <tr className="border-b border-slate-200 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            <th className="py-1.5">Class</th>
            <th className="py-1.5 text-right">Precision</th>
            <th className="py-1.5 text-right">Recall</th>
            <th className="py-1.5 text-right">F1</th>
            <th className="py-1.5 text-right">Rows</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.class} className="border-b border-slate-100 last:border-0">
              <td className="py-1.5 font-medium text-slate-800">{r.class}</td>
              <td className="tabular py-1.5 text-right font-mono">{pct(r.precision)}</td>
              <td className="tabular py-1.5 text-right font-mono">{pct(r.recall)}</td>
              <td className="tabular py-1.5 text-right font-mono">{pct(r.f1)}</td>
              <td className="tabular py-1.5 text-right font-mono text-slate-500">{num(r.support)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Picks the right evaluation visuals for a model from its `final.charts` payload. */
export function EvaluationCharts({
  problemType,
  charts,
  compactMode = false,
}: {
  problemType: string
  charts: Record<string, unknown>
  compactMode?: boolean
}) {
  const h = compactMode ? 200 : 250
  const c = charts as Record<string, never>
  if (problemType === 'forecasting') {
    const fc = c.forecast as unknown as { series: Record<string, ForecastPoint[]>; default: string } | undefined
    return (
      <div className="grid gap-6">
        {fc && <ForecastChart points={fc.series[fc.default] ?? []} title={`Forecast · ${fc.default}`} height={compactMode ? 230 : 290} />}
        {!compactMode && c.backtest && (
          <BacktestChart
            windows={c.backtest as unknown as Array<Record<string, number | string>>}
            metric={String(c.metric)}
            metricLabel={String(c.metric).toUpperCase()}
          />
        )}
      </div>
    )
  }
  if (problemType === 'binary') {
    const conf = c.confusion as unknown as { labels: string[]; matrix: number[][]; threshold: number | null }
    return (
      <div className="grid gap-6 lg:grid-cols-2">
        {c.roc && <RocChart points={c.roc} height={h} />}
        {conf && <ConfusionMatrix labels={conf.labels} matrix={conf.matrix} threshold={conf.threshold} />}
        {!compactMode && c.pr && <PrChart points={c.pr} prevalence={c.prevalence as unknown as number} height={h} />}
        {!compactMode && c.scores && <ScoreDistChart bins={c.scores} threshold={conf?.threshold} labels={conf?.labels ?? []} height={h} />}
        {!compactMode && c.calibration && <CalibrationChart points={c.calibration} height={h} />}
      </div>
    )
  }
  if (problemType === 'multiclass') {
    const conf = c.confusion as unknown as { labels: string[]; matrix: number[][] }
    return (
      <div className="grid gap-6 lg:grid-cols-2">
        {conf && <ConfusionMatrix labels={conf.labels} matrix={conf.matrix} />}
        {c.per_class && <PerClassTable rows={c.per_class} />}
      </div>
    )
  }
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {c.scatter && <ScatterFit points={c.scatter} height={h} />}
      {c.residuals && <ResidualChart bins={c.residuals} height={h} />}
    </div>
  )
}
