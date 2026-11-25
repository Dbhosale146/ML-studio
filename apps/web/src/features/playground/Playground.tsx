import { useMutation, useQuery } from '@tanstack/react-query'
import {
  ArrowDown,
  ArrowUp,
  ChevronDown,
  Dices,
  Download,
  FileSpreadsheet,
  LoaderCircle,
  PlayCircle,
  RotateCcw,
  TriangleAlert,
  Upload,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ForecastChart, type ForecastPoint } from '../../components/charts'
import { Badge, Button, Card, CardHeader, Empty, Spinner, Tabs } from '../../components/ui'
import { api, errorMessage, type Schemas } from '../../lib/apiClient'
import { cx } from '../../lib/cx'
import { num } from '../../lib/format'
import { ACCEPT } from '../datasets/UploadDropzone'
import { PreviewTable } from '../datasets/PreviewTable'
import { useInputSchema, type InputSchema, type ModelDetail } from '../models/hooks'
import { defaultValues, fieldWarning, fromExample, splitFields, toRow, type Field, type FormValues } from './form'

type PredictOut = Schemas['PredictOut']
type Row = Schemas['PredictionRow']
type Job = Schemas['BatchJobOut']

function FieldInput({ f, value, onChange }: { f: Field; value: string; onChange: (v: string) => void }) {
  const id = `f-${f.name}`
  const warn = fieldWarning(f, value)
  const [other, setOther] = useState(false)
  const opts = (f.options ?? []) as Array<{ value: string; label?: string; count?: number }>
  let control
  if (f.kind === 'boolean') {
    control = (
      <div className="flex gap-1" role="radiogroup" aria-labelledby={`${id}-label`}>
        {opts.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={value === o.value}
            onClick={() => onChange(o.value)}
            className={cx(
              'h-9 flex-1 rounded-lg border text-[13px] font-semibold',
              value === o.value ? 'border-emerald-400 bg-emerald-50 text-emerald-800' : 'border-slate-200 text-slate-600 hover:bg-slate-50',
            )}
          >
            {o.label ?? o.value}
          </button>
        ))}
      </div>
    )
  } else if (f.kind === 'category' && !other && opts.length) {
    control = (
      <select
        id={id}
        value={opts.some((o) => o.value === value) ? value : ''}
        onChange={(e) => (e.target.value === '__other__' ? setOther(true) : onChange(e.target.value))}
        className="h-9 w-full rounded-lg border border-slate-200 bg-white px-2 text-[13px] outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-100"
      >
        <option value="">(empty: use the most common value)</option>
        {opts.map((o) => (
          <option key={o.value} value={o.value}>
            {o.value}
            {o.count ? ` (${num(o.count)})` : ''}
          </option>
        ))}
        <option value="__other__">Other…</option>
      </select>
    )
  } else {
    control = (
      <input
        id={id}
        type={f.kind === 'date' ? 'date' : f.kind === 'number' || f.kind === 'integer' ? 'number' : 'text'}
        step={f.kind === 'integer' ? 1 : 'any'}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={f.kind === 'category' ? 'Type a value' : f.default != null ? String(f.default) : ''}
        className={cx(
          'h-9 w-full rounded-lg border bg-white px-2.5 text-[13px] outline-none focus:ring-2',
          warn ? 'border-amber-300 focus:border-amber-400 focus:ring-amber-100' : 'border-slate-200 focus:border-emerald-400 focus:ring-emerald-100',
        )}
        aria-describedby={`${id}-hint`}
      />
    )
  }
  const hint =
    f.kind === 'number' || f.kind === 'integer'
      ? f.p05 != null
        ? `typical ${num(f.p05)}–${num(f.p95)} · median ${num(f.median)}`
        : ''
      : f.kind === 'date' && f.min
        ? `${f.min} → ${f.max}`
        : f.kind === 'category'
          ? `${opts.length}${f.allow_other ? '+' : ''} known values`
          : ''
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label id={`${id}-label`} htmlFor={id} className="truncate text-[12px] font-semibold text-slate-800" title={f.name}>
          {f.label}
        </label>
        {f.importance > 0 && <span className="tabular shrink-0 font-mono text-[10px] text-slate-400">{(f.importance * 100).toFixed(0)}% impact</span>}
      </div>
      {control}
      <div id={`${id}-hint`} className={cx('mt-0.5 min-h-[14px] truncate text-[11px]', warn ? 'text-amber-700' : 'text-slate-400')}>
        {warn ?? f.help ?? hint}
        {!warn && f.help && hint ? ` · ${hint}` : !warn && !f.help ? '' : ''}
      </div>
    </div>
  )
}

function Result({ row, actual, explainer }: { row: Row; actual: string | null; explainer?: string | null }) {
  const probs = Object.entries(row.probabilities ?? {}).sort((a, b) => b[1] - a[1])
  const peak = Math.max(...(row.drivers ?? []).map((d) => Math.abs(d.effect)), 1e-9)
  return (
    <div className="space-y-4" aria-live="polite">
      <div className="rounded-xl bg-gradient-to-br from-emerald-50 to-indigo-50 p-4 ring-1 ring-emerald-100">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Prediction</div>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="tabular text-[28px] font-semibold tracking-tight text-slate-900">{row.display}</span>
          {row.risk_band && (
            <Badge color={row.risk_band === 'High' ? 'red' : row.risk_band === 'Medium' ? 'amber' : 'emerald'}>{row.risk_band} risk</Badge>
          )}
        </div>
        {row.interval && (
          <div className="mt-1 text-[12px] text-slate-600">
            80% of similar holdout errors fall within {num(row.interval[0])} – {num(row.interval[1])}
          </div>
        )}
        {actual != null && (
          <div className="mt-1 text-[12px] text-slate-600">
            Actual value in your data: <strong>{actual}</strong>
          </div>
        )}
      </div>
      {probs.length > 0 && (
        <div>
          <div className="mb-1.5 text-[12px] font-semibold text-slate-900">Probabilities</div>
          <ul className="space-y-1.5">
            {probs.slice(0, 6).map(([k, p]) => (
              <li key={k} className="grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-2 text-[12px]">
                <div>
                  <div className="truncate text-slate-700">{k}</div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full rounded-full bg-indigo-500" style={{ width: `${p * 100}%` }} />
                  </div>
                </div>
                <span className="tabular text-right font-mono">{(p * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {(row.drivers ?? []).length > 0 && (
        <div>
          <div className="mb-1.5 text-[12px] font-semibold text-slate-900">
            Top drivers for this prediction {explainer && <span className="font-normal text-slate-400">({explainer})</span>}
          </div>
          <ul className="space-y-1.5" aria-label="Drivers">
            {(row.drivers ?? []).map((d) => (
              <li key={d.feature} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-2 text-[12px]">
                <span className="truncate text-slate-700" title={d.feature}>
                  {d.label} <span className="text-slate-400">= {d.value == null ? 'empty' : String(d.value)}</span>
                </span>
                <span className={cx('flex items-center gap-0.5 font-semibold', d.direction === 'up' ? 'text-red-600' : 'text-emerald-600')}>
                  {d.direction === 'up' ? <ArrowUp size={12} aria-label="raises" /> : <ArrowDown size={12} aria-label="lowers" />}
                </span>
                <div className="col-span-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={cx('h-full rounded-full', d.direction === 'up' ? 'bg-red-400' : 'bg-emerald-400')}
                    style={{ width: `${(Math.abs(d.effect) / peak) * 100}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-1.5 text-[11px] text-slate-400">Red pushes the prediction up, green pushes it down, relative to an average row.</p>
        </div>
      )}
      {(row.warnings ?? []).length > 0 && (
        <ul className="space-y-1 rounded-lg bg-amber-50 p-2.5 text-[12px] text-amber-900">
          {(row.warnings ?? []).map((w) => (
            <li key={w} className="flex gap-1.5">
              <TriangleAlert size={13} className="mt-0.5 shrink-0" aria-hidden />
              {w}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function SinglePredict({ model, schema }: { model: ModelDetail; schema: InputSchema }) {
  const fields = schema.fields
  const [values, setValues] = useState<FormValues>(() => defaultValues(fields))
  const [actual, setActual] = useState<string | null>(null)
  const [showMore, setShowMore] = useState(false)
  const { main, more } = useMemo(() => splitFields(fields), [fields])
  const predict = useMutation({
    mutationFn: (row: Record<string, unknown>) => api.post<PredictOut>(`/models/${model.id}/predict`, { rows: [row], explain: true }),
  })
  const set = (k: string, v: string) => setValues((s) => ({ ...s, [k]: v }))
  const random = () => {
    const ex = (schema.examples ?? [])[Math.floor(Math.random() * (schema.examples ?? []).length)] as
      { inputs: Record<string, unknown>; actual?: string } | undefined
    if (!ex) return
    setValues(fromExample(fields, ex.inputs))
    setActual(ex.actual ?? null)
  }
  const result = predict.data?.rows?.[0]
  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <Card className="xl:col-span-3">
        <CardHeader
          title="Enter values"
          subtitle={`${fields.length} inputs from the training data, most influential first`}
          icon={PlayCircle}
          right={
            <div className="flex gap-1.5">
              <Button size="sm" icon={Dices} onClick={random} disabled={!(schema.examples ?? []).length}>
                Random training row
              </Button>
              <Button
                size="sm"
                variant="ghost"
                icon={RotateCcw}
                aria-label="Reset to typical values"
                onClick={() => {
                  setValues(defaultValues(fields))
                  setActual(null)
                }}
              />
            </div>
          }
        />
        <form
          className="px-5 pb-5"
          onSubmit={(e) => {
            e.preventDefault()
            predict.mutate(toRow(fields, values))
          }}
        >
          <div className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
            {main.map((f) => (
              <FieldInput key={f.name} f={f} value={values[f.name] ?? ''} onChange={(v) => (set(f.name, v), setActual(null))} />
            ))}
          </div>
          {more.length > 0 && (
            <div className="mt-2">
              <button
                type="button"
                onClick={() => setShowMore((v) => !v)}
                aria-expanded={showMore}
                className="flex items-center gap-1 text-[12px] font-semibold text-emerald-700 hover:underline"
              >
                <ChevronDown size={14} className={cx('transition-transform', showMore && 'rotate-180')} aria-hidden />
                {showMore ? 'Hide' : 'Show'} {more.length} more input{more.length > 1 ? 's' : ''} (prefilled with typical values)
              </button>
              {showMore && (
                <div className="mt-2 grid gap-x-4 gap-y-2 sm:grid-cols-2">
                  {more.map((f) => (
                    <FieldInput key={f.name} f={f} value={values[f.name] ?? ''} onChange={(v) => (set(f.name, v), setActual(null))} />
                  ))}
                </div>
              )}
            </div>
          )}
          <div className="mt-4 flex items-center gap-3">
            <Button type="submit" variant="primary" icon={PlayCircle} disabled={predict.isPending}>
              {predict.isPending ? 'Predicting…' : 'Predict'}
            </Button>
            {predict.isError && (
              <span role="alert" className="text-[13px] text-red-700">
                {errorMessage(predict.error)}
              </span>
            )}
          </div>
        </form>
      </Card>
      <Card className="h-fit xl:col-span-2">
        <CardHeader title="Result" subtitle={`${model.name} v${predict.data?.version ?? model.latest_version}`} icon={PlayCircle} />
        <div className="px-5 pb-5">
          {result ? (
            <Result row={result} actual={actual} explainer={predict.data?.explainer} />
          ) : (
            <p className="text-[13px] text-slate-500">Fill in the form (or load a random training row) and press Predict.</p>
          )}
        </div>
      </Card>
    </div>
  )
}

function BatchPredict({ model, schema }: { model: ModelDetail; schema: InputSchema }) {
  const input = useRef<HTMLInputElement>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const start = useMutation({
    mutationFn: (file: File) => api.upload<Job>(`/models/${model.id}/predict-batch`, file),
    onSuccess: (j) => setJobId(j.id),
  })
  const job = useQuery({
    queryKey: ['prediction-job', jobId],
    queryFn: () => api.get<Job>(`/predictions/${jobId}`),
    enabled: Boolean(jobId),
    refetchInterval: (q) => (q.state.data && ['done', 'failed'].includes(q.state.data.status) ? false : 800),
  })
  const j = job.data
  const summary = (j?.summary ?? {}) as {
    missing_columns?: string[]
    extra_columns?: string[]
    distribution?: Record<string, number>
    preview?: { columns: string[]; rows: string[][] }
    sheet?: string
  }
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Score a whole file"
          subtitle={`Upload an Excel or CSV with the same columns as the training data (${schema.fields.length} inputs). The target column is not needed.`}
          icon={FileSpreadsheet}
        />
        <div className="space-y-3 px-5 pb-5">
          <div className="flex flex-wrap gap-1.5">
            {schema.fields.map((f) => (
              <code key={f.name} className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 font-mono text-[11px] text-slate-600">
                {f.name}
              </code>
            ))}
          </div>
          <Button
            variant="primary"
            icon={Upload}
            onClick={() => input.current?.click()}
            disabled={start.isPending || j?.status === 'running' || j?.status === 'queued'}
          >
            {start.isPending ? 'Uploading…' : 'Upload file to score'}
          </Button>
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) start.mutate(f)
              e.target.value = ''
            }}
          />
          {start.isError && (
            <p role="alert" className="text-[13px] text-red-700">
              {errorMessage(start.error)}
            </p>
          )}
        </div>
      </Card>
      {j && (
        <Card>
          <CardHeader
            title={j.file_name}
            subtitle={j.status === 'done' ? `${num(j.rows)} rows scored${summary.sheet ? ` · sheet ${summary.sheet}` : ''}` : j.status}
            icon={j.status === 'done' ? Download : LoaderCircle}
            right={
              j.status === 'done' && j.download_url ? (
                <a
                  href={j.download_url}
                  className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-emerald-500 px-3 text-xs font-semibold text-white hover:bg-emerald-600"
                  download
                >
                  <Download size={14} aria-hidden /> Download .xlsx
                </a>
              ) : null
            }
          />
          <div className="space-y-3 px-5 pb-5">
            {(j.status === 'queued' || j.status === 'running') && <Spinner label="Scoring rows in the background" />}
            {j.status === 'failed' && (
              <p role="alert" className="rounded-lg bg-red-50 p-3 text-[13px] text-red-800">
                {j.error}
              </p>
            )}
            {j.status === 'done' && (
              <>
                {(summary.missing_columns?.length ?? 0) > 0 && (
                  <p className="rounded-lg bg-amber-50 p-2.5 text-[12px] text-amber-900">
                    Missing columns (filled with typical values): <strong>{summary.missing_columns!.join(', ')}</strong>
                  </p>
                )}
                {(summary.extra_columns?.length ?? 0) > 0 && (
                  <p className="text-[12px] text-slate-500">Ignored extra columns: {summary.extra_columns!.join(', ')}</p>
                )}
                {summary.distribution && (
                  <div className="flex flex-wrap gap-2 text-[12px]">
                    {Object.entries(summary.distribution).map(([k, v]) => (
                      <Badge key={k} color="slate">
                        {k}: {typeof v === 'number' ? num(v) : String(v)}
                      </Badge>
                    ))}
                  </div>
                )}
                {summary.preview && (
                  <PreviewTable
                    preview={{
                      columns: summary.preview.columns,
                      rows: summary.preview.rows.map((r) => r.map((v) => v ?? '')),
                      total_rows: j.rows ?? 0,
                    }}
                    maxHeight={360}
                  />
                )}
              </>
            )}
          </div>
        </Card>
      )}
    </div>
  )
}

function ForecastPlayground({ model, schema }: { model: ModelDetail; schema: InputSchema }) {
  const fc = (schema.forecast ?? {}) as { series?: string[]; horizon?: number; unit?: string }
  const seriesList = fc.series ?? []
  const [series, setSeries] = useState(seriesList.includes('Total') ? 'Total' : (seriesList[0] ?? ''))
  const [horizon, setHorizon] = useState(fc.horizon ?? 1)
  const predict = useMutation({ mutationFn: () => api.post<PredictOut>(`/models/${model.id}/predict`, { series, horizon }) })
  useEffect(() => {
    predict.mutate()
    // re-forecast whenever the series or horizon changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, horizon])
  const view = predict.data?.forecast as
    { points: ForecastPoint[]; table: Array<{ date: string; forecast: number; lower_80: number; upper_80: number }>; series: string } | undefined
  return (
    <div className="space-y-6">
      <Card className="p-5">
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-[12px] font-semibold text-slate-700">
            Series
            <select
              value={series}
              onChange={(e) => setSeries(e.target.value)}
              className="mt-1 block h-9 min-w-[180px] rounded-lg border border-slate-200 bg-white px-2 text-[13px] font-normal outline-none focus:border-emerald-400"
            >
              {seriesList.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="text-[12px] font-semibold text-slate-700">
            Horizon: {horizon} {fc.unit}
            {horizon > 1 ? 's' : ''}
            <input
              type="range"
              min={1}
              max={fc.horizon ?? 1}
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="mt-2 block w-56 accent-emerald-600"
            />
          </label>
          <a
            href={`/api/v1/models/${model.id}/forecast/download?${new URLSearchParams({ series, horizon: String(horizon) })}`}
            className="ml-auto inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-[13px] font-semibold text-slate-700 hover:bg-slate-50"
            download
          >
            <Download size={15} aria-hidden /> Download forecast
          </a>
        </div>
      </Card>
      {predict.isError && (
        <p role="alert" className="text-[13px] text-red-700">
          {errorMessage(predict.error)}
        </p>
      )}
      {view && (
        <div className="grid gap-6 xl:grid-cols-3">
          <Card className="p-5 xl:col-span-2">
            <ForecastChart points={view.points} title={`Forecast · ${view.series}`} height={320} />
          </Card>
          <Card className="overflow-hidden">
            <CardHeader title="Forecast table" subtitle="With the 80% interval from backtest errors" icon={Download} />
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-y border-slate-200 bg-slate-50 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="px-4 py-2 text-left">Date</th>
                  <th className="px-2 py-2 text-right">Forecast</th>
                  <th className="px-4 py-2 text-right">80% interval</th>
                </tr>
              </thead>
              <tbody>
                {view.table.map((r) => (
                  <tr key={r.date} className="border-b border-slate-100 last:border-0">
                    <td className="px-4 py-1.5 font-mono">{r.date}</td>
                    <td className="tabular px-2 py-1.5 text-right font-mono font-semibold">{num(r.forecast)}</td>
                    <td className="tabular px-4 py-1.5 text-right font-mono text-slate-500">
                      {num(r.lower_80)} – {num(r.upper_80)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}

export function Playground({ model }: { model: ModelDetail }) {
  const schema = useInputSchema(model.id, model.status === 'ready')
  const [mode, setMode] = useState('single')
  if (model.status !== 'ready')
    return <Empty icon={PlayCircle} title="Not ready yet" text="The Playground opens once the agent has published the model." />
  if (schema.isLoading) return <Spinner />
  if (schema.isError || !schema.data) return <Empty icon={TriangleAlert} title="Could not load the input schema" text={errorMessage(schema.error)} />
  if (model.problem_type === 'forecasting') return <ForecastPlayground model={model} schema={schema.data} />
  return (
    <div className="space-y-4">
      <Tabs
        tabs={[
          { id: 'single', label: 'Single prediction', icon: PlayCircle },
          { id: 'batch', label: 'Batch (Excel / CSV)', icon: FileSpreadsheet },
        ]}
        value={mode}
        onChange={setMode}
      />
      {mode === 'single' ? <SinglePredict model={model} schema={schema.data} /> : <BatchPredict model={model} schema={schema.data} />}
    </div>
  )
}
