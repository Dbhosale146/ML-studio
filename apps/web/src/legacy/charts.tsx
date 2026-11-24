// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { forecastSeries, prCurve, regressionScatter, residualHistogram, rocCurve, scoreDistribution } from './chartData'
import { num } from '../lib/format'

const AXIS = { fontSize: 11, fill: '#64748b' }
const GRID = <CartesianGrid stroke="#eef2f7" vertical={false} />
const TT = {
  contentStyle: { borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12, boxShadow: '0 4px 14px rgba(15,23,42,0.08)' },
  labelStyle: { color: '#0f172a', fontWeight: 600 },
}

export function ChartFrame({ title, subtitle, children, height = 240, right }) {
  return (
    <div className="min-w-0">
      {(title || right) && (
        <div className="mb-2 flex items-end justify-between gap-2">
          <div>
            <div className="text-[13px] font-semibold text-slate-900">{title}</div>
            {subtitle && <div className="text-xs text-slate-500">{subtitle}</div>}
          </div>
          {right}
        </div>
      )}
      <div style={{ height }}>{children}</div>
    </div>
  )
}

export function RocChart({ auc, height }) {
  const data = rocCurve(auc)
  return (
    <ChartFrame title="ROC curve" subtitle={`Area under curve = ${auc.toFixed(3)}`} height={height}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 14, left: -14 }}>
          <defs>
            <linearGradient id="roc" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.22} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          {GRID}
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tick={AXIS}
            tickCount={6}
            label={{ value: 'False positive rate', position: 'insideBottom', offset: -8, style: AXIS }}
          />
          <YAxis domain={[0, 1]} tick={AXIS} tickCount={6} />
          <Tooltip {...TT} formatter={(v, n) => [v, n === 'tpr' ? 'True positive rate' : 'Random']} labelFormatter={(l) => `FPR ${l}`} />
          <Area type="monotone" dataKey="tpr" stroke="#10b981" strokeWidth={2} fill="url(#roc)" />
          <Line type="linear" dataKey="chance" stroke="#94a3b8" strokeDasharray="4 4" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ConfusionMatrix({ cm }) {
  const total = cm.tp + cm.fn + cm.fp + cm.tn
  const cell = (v, good, label) => (
    <div className={`rounded-lg p-3 ${good ? 'bg-emerald-50 text-emerald-900' : 'bg-red-50/70 text-red-900'}`}>
      <div className="text-[11px] font-medium opacity-70">{label}</div>
      <div className="mt-0.5 text-xl font-semibold tabular">{num(v)}</div>
      <div className="text-[11px] tabular opacity-70">{((v / total) * 100).toFixed(1)}%</div>
    </div>
  )
  return (
    <div>
      <div className="text-[13px] font-semibold text-slate-900">Confusion matrix</div>
      <div className="mb-3 text-xs text-slate-500">Holdout set · {num(total)} rows</div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-xs">
        <div />
        <div className="text-center font-medium text-slate-500">Predicted positive</div>
        <div className="text-center font-medium text-slate-500">Predicted negative</div>
        <div className="flex items-center font-medium text-slate-500 [writing-mode:vertical-rl] rotate-180">Actual +</div>
        {cell(cm.tp, true, 'True positive')}
        {cell(cm.fn, false, 'False negative')}
        <div className="flex items-center font-medium text-slate-500 [writing-mode:vertical-rl] rotate-180">Actual −</div>
        {cell(cm.fp, false, 'False positive')}
        {cell(cm.tn, true, 'True negative')}
      </div>
    </div>
  )
}

export function PrChart({ prAuc, height }) {
  const data = prCurve(prAuc)
  return (
    <ChartFrame title="Precision–recall curve" subtitle={`PR-AUC = ${prAuc.toFixed(3)} · prevalence 0.17%`} height={height}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 14, left: -14 }}>
          {GRID}
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            tick={AXIS}
            tickCount={6}
            label={{ value: 'Recall', position: 'insideBottom', offset: -8, style: AXIS }}
          />
          <YAxis domain={[0, 1]} tick={AXIS} tickCount={6} />
          <Tooltip {...TT} />
          <Area type="monotone" dataKey="precision" stroke="#6366f1" strokeWidth={2} fill="#6366f1" fillOpacity={0.12} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ScoreDistChart({ seed, height }) {
  const data = scoreDistribution(seed)
  return (
    <ChartFrame title="Score distribution" subtitle="Share of transactions by model score (threshold 0.34)" height={height}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -14 }} barGap={0}>
          {GRID}
          <XAxis dataKey="bin" tick={AXIS} interval={3} />
          <YAxis tick={AXIS} />
          <Tooltip {...TT} />
          <Legend wrapperStyle={{ fontSize: 11 }} iconSize={8} />
          <ReferenceLine x="0.30" stroke="#0f172a" strokeDasharray="3 3" />
          <Bar dataKey="legit" name="Legitimate" fill="#cbd5e1" radius={[3, 3, 0, 0]} />
          <Bar dataKey="fraud" name="Fraud" fill="#ef4444" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ForecastChart({ seed, height = 260, horizon = 13 }) {
  const data = forecastSeries(seed, horizon)
  const split = data.find((d) => d.forecast != null && d.actual != null)?.week
  const vals = data.flatMap((d) => [d.actual, d.forecast, ...(d.band || [])]).filter((v) => v != null)
  const domain = [Math.floor((Math.min(...vals) - 60) / 100) * 100, Math.ceil((Math.max(...vals) + 60) / 100) * 100]
  return (
    <ChartFrame title="Forecast vs actuals" subtitle={`52 weeks of history · ${horizon}-week forecast with 80% interval`} height={height}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -6 }}>
          {GRID}
          <XAxis dataKey="week" tick={AXIS} interval={7} />
          <YAxis tick={AXIS} domain={domain} allowDataOverflow={false} tickFormatter={(v) => num(Math.round(v))} />
          <Tooltip {...TT} formatter={(v, n) => [Array.isArray(v) ? `${num(v[0])} – ${num(v[1])}` : num(v), n]} />
          <Legend wrapperStyle={{ fontSize: 11 }} iconSize={8} />
          <Area dataKey="band" name="80% interval" stroke="none" fill="#10b981" fillOpacity={0.14} />
          <Line dataKey="actual" name="Actual" stroke="#0f172a" strokeWidth={1.5} dot={false} />
          <Line dataKey="fitted" name="Backtest fit" stroke="#6366f1" strokeWidth={1.25} strokeDasharray="3 3" dot={false} />
          <Line dataKey="forecast" name="Forecast" stroke="#10b981" strokeWidth={2.25} dot={false} />
          {split && (
            <ReferenceLine x={split} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: 'Today', position: 'insideTopRight', style: AXIS }} />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function BacktestChart({ values, height = 200 }) {
  const data = values.map((v, i) => ({ window: `W${i + 1}`, mape: v }))
  return (
    <ChartFrame title="Backtest error by window" subtitle="Rolling-origin, 13-week windows · MAPE %" height={height}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
          {GRID}
          <XAxis dataKey="window" tick={AXIS} />
          <YAxis tick={AXIS} domain={[0, 16]} />
          <Tooltip {...TT} formatter={(v) => [`${v}%`, 'MAPE']} />
          <ReferenceLine
            y={14.8}
            stroke="#ef4444"
            strokeDasharray="4 4"
            label={{ value: 'Seasonal naive 14.8%', position: 'insideTopRight', style: { ...AXIS, fill: '#ef4444' } }}
          />
          <Bar dataKey="mape" fill="#10b981" radius={[4, 4, 0, 0]} barSize={28} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ScatterFit({ seed, r2, height }) {
  const data = regressionScatter(seed, r2)
  return (
    <ChartFrame title="Predicted vs actual" subtitle={`2026 holdout · R² = ${r2.toFixed(3)} · $K`} height={height}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 6, right: 8, bottom: 14, left: -8 }}>
          <CartesianGrid stroke="#eef2f7" />
          <XAxis
            dataKey="actual"
            type="number"
            name="Actual"
            tick={AXIS}
            domain={[100, 1050]}
            label={{ value: 'Actual ($K)', position: 'insideBottom', offset: -8, style: AXIS }}
          />
          <YAxis dataKey="predicted" type="number" name="Predicted" tick={AXIS} domain={[100, 1050]} />
          <Tooltip {...TT} cursor={{ strokeDasharray: '3 3' }} />
          <ReferenceLine
            segment={[
              { x: 100, y: 100 },
              { x: 1050, y: 1050 },
            ]}
            stroke="#94a3b8"
            strokeDasharray="4 4"
          />
          <Scatter data={data} fill="#f59e0b" fillOpacity={0.55} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ResidualChart({ seed, height }) {
  const data = residualHistogram(seed)
  return (
    <ChartFrame title="Residual distribution" subtitle="Prediction error, centred on zero" height={height}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
          {GRID}
          <XAxis dataKey="bin" tick={AXIS} interval={2} />
          <YAxis tick={AXIS} />
          <Tooltip {...TT} />
          <Bar dataKey="count" fill="#f59e0b" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

const SEG_COLORS = ['#10b981', '#6366f1', '#0ea5e9', '#f59e0b', '#94a3b8']

export function SegmentsChart({ segments, height = 230 }) {
  return (
    <ChartFrame title="Segment sizes" subtitle="Share of customers per discovered segment" height={height}>
      <ResponsiveContainer>
        <BarChart data={segments} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 10 }}>
          <XAxis type="number" tick={AXIS} unit="%" />
          <YAxis type="category" dataKey="name" tick={{ ...AXIS, fill: '#0f172a' }} width={90} />
          <Tooltip {...TT} formatter={(v) => [`${v}%`, 'Share']} />
          <Bar dataKey="share" radius={[0, 4, 4, 0]} barSize={18}>
            {segments.map((s, i) => (
              <Cell key={s.name} fill={SEG_COLORS[i % SEG_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}

export function ImportanceBars({ items, max = 8 }) {
  const top = items.slice(0, max)
  const peak = Math.max(...top.map((i) => i[1]))
  return (
    <div className="space-y-2">
      {top.map(([name, v], idx) => (
        <div key={name} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1">
          <span className="truncate font-mono text-[12px] text-slate-700">{name}</span>
          <span className="font-mono text-[11px] text-slate-500 tabular">{(v * 100).toFixed(0)}%</span>
          <div className="col-span-2 h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full transition-[width] duration-700"
              style={{ width: `${(v / peak) * 100}%`, background: idx < 3 ? '#10b981' : '#6ee7b7' }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

export function UsageChart({ series, height = 200 }) {
  const data = series.map((p) => ({ ...p, label: `D-${30 - p.day}` }))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -4 }}>
        <defs>
          <linearGradient id="usage" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6366f1" stopOpacity={0.25} />
            <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
          </linearGradient>
        </defs>
        {GRID}
        <XAxis dataKey="label" tick={AXIS} interval={6} />
        <YAxis tick={AXIS} tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}K` : v)} />
        <Tooltip {...TT} formatter={(v) => [num(v), 'Predictions']} />
        <Area type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} fill="url(#usage)" />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function MiniLeaderLines({ data, height = 120 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <Line dataKey="v" stroke="#10b981" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  )
}

// Picks the right evaluation visuals for a model's problem type.
export function ModelEvalCharts({ model, compact }) {
  const h = compact ? 200 : 250
  const seed = model.slug
  switch (model.kind) {
    case 'classification':
      return (
        <div className="grid gap-6 lg:grid-cols-2">
          <RocChart auc={model.auc || 0.9} height={h} />
          {model.confusion && <ConfusionMatrix cm={model.confusion} />}
        </div>
      )
    case 'anomaly':
      return (
        <div className="grid gap-6 lg:grid-cols-2">
          <PrChart prAuc={parseFloat(model.primary.display) || 0.8} height={h} />
          <ScoreDistChart seed={seed} height={h} />
        </div>
      )
    case 'forecasting':
      return (
        <div className="grid gap-6">
          <ForecastChart seed={seed} height={compact ? 220 : 280} />
          {!compact && model.backtest && <BacktestChart values={model.backtest} />}
        </div>
      )
    case 'regression':
      return (
        <div className="grid gap-6 lg:grid-cols-2">
          <ScatterFit seed={seed} r2={model.r2 || 0.9} height={h} />
          <ResidualChart seed={seed} height={h} />
        </div>
      )
    case 'clustering':
      return model.segments ? <SegmentsChart segments={model.segments} /> : null
    default:
      return null
  }
}
