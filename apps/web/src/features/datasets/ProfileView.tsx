import { AlertTriangle, Info, Lock } from 'lucide-react'
import { Badge, type BadgeColor } from '../../components/ui'
import { cx } from '../../lib/cx'
import { num } from '../../lib/format'
import type { ColumnProfile, Profile } from '../../types/domain'

const SEVERITY: Record<string, BadgeColor> = { high: 'red', medium: 'amber', low: 'sky', info: 'slate' }

export function QualityRing({ value, size = 68 }: { value: number; size?: number }) {
  const r = size / 2 - 8
  const c = 2 * Math.PI * r
  const color = value >= 90 ? '#10b981' : value >= 75 ? '#f59e0b' : '#ef4444'
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0" role="img" aria-label={`Data quality ${value} out of 100`}>
      <circle cx={size / 2} cy={size / 2} r={r} stroke="#eef2f7" strokeWidth="7" fill="none" />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        stroke={color}
        strokeWidth="7"
        fill="none"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - value / 100)}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x={size / 2} y={size / 2 + 5} textAnchor="middle" fontSize="16" fontWeight="600" fill="#0f172a">
        {value}
      </text>
    </svg>
  )
}

function stats(c: ColumnProfile): string {
  if ((c.kind === 'number' || c.kind === 'integer') && c.min !== undefined)
    return `${num(c.min as number)} – ${num(c.max as number)} · median ${num(c.median)}`
  if (c.kind === 'date' && c.min) return `${String(c.min).slice(0, 10)} → ${String(c.max).slice(0, 10)}`
  if (c.top_values?.length)
    return c.top_values
      .slice(0, 3)
      .map((t) => `${t.value} ${t.pct}%`)
      .join(' · ')
  return c.sample.join(', ')
}

function Mini({ c }: { c: ColumnProfile }) {
  if (c.histogram?.length) {
    const peak = Math.max(...c.histogram.map((h) => h.count))
    return (
      <div className="flex h-6 w-24 items-end gap-px" aria-hidden>
        {c.histogram.map((h, i) => (
          <span key={i} className="flex-1 rounded-t-sm bg-emerald-300" style={{ height: `${Math.max(4, (h.count / peak) * 100)}%` }} />
        ))}
      </div>
    )
  }
  if (c.top_values?.length) {
    return (
      <div className="flex h-2 w-24 overflow-hidden rounded-full bg-slate-100" aria-hidden>
        {c.top_values.slice(0, 5).map((t, i) => (
          <span
            key={t.value}
            style={{ width: `${t.pct}%` }}
            className={['bg-sky-400', 'bg-indigo-400', 'bg-emerald-400', 'bg-amber-400', 'bg-slate-400'][i]}
          />
        ))}
      </div>
    )
  }
  return null
}

export function ProfileView({ profile }: { profile: Profile }) {
  return (
    <div className="space-y-5">
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="flex items-center gap-4 rounded-xl border border-slate-200 p-4">
          <QualityRing value={profile.quality} />
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Data quality</div>
            <ul className="mt-1 space-y-0.5 text-xs text-slate-600">
              {profile.quality_reasons.length === 0 && <li>No issues found</li>}
              {profile.quality_reasons.map((r) => (
                <li key={r.factor}>
                  −{r.penalty} {r.detail}
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 p-4 lg:col-span-2">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">What we found</div>
          <ul className="mt-2 grid gap-1.5 sm:grid-cols-2">
            {profile.notes.map((n) => (
              <li key={n} className="flex gap-2 text-xs text-slate-600">
                <Info size={13} className="mt-0.5 shrink-0 text-sky-500" aria-hidden />
                {n}
              </li>
            ))}
            {profile.issues.slice(0, 10).map((i, k) => (
              <li key={k} className="flex gap-2 text-xs text-slate-600">
                <AlertTriangle size={13} className={cx('mt-0.5 shrink-0', i.severity === 'high' ? 'text-red-500' : 'text-amber-500')} aria-hidden />
                {i.detail}
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="thin-scroll overflow-x-auto rounded-xl border border-slate-200">
        <table className="w-full min-w-[760px] text-left">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              <th scope="col" className="px-4 py-2.5">
                Column
              </th>
              <th scope="col" className="px-3 py-2.5">
                Role
              </th>
              <th scope="col" className="w-40 px-3 py-2.5">
                Missing
              </th>
              <th scope="col" className="px-3 py-2.5 text-right">
                Unique
              </th>
              <th scope="col" className="px-3 py-2.5">
                Distribution
              </th>
              <th scope="col" className="px-4 py-2.5">
                Summary
              </th>
            </tr>
          </thead>
          <tbody>
            {profile.columns.map((c) => (
              <tr key={c.name} className="border-b border-slate-100 last:border-0">
                <td className="px-4 py-2 font-mono text-[12px] text-slate-800">
                  <span className="inline-flex items-center gap-1.5">
                    {c.name}
                    {c.pii && (
                      <span title="Looks like personal data: masked before anything is sent to the LLM">
                        <Lock size={12} className="text-amber-500" aria-label="Personal data" />
                      </span>
                    )}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <Badge color={c.kind === 'id' ? 'indigo' : c.kind === 'constant' || c.kind === 'empty' || c.kind === 'text' ? 'slate' : 'sky'}>
                    {c.role}
                  </Badge>
                </td>
                <td className="px-3 py-2" aria-label={`${c.missing_pct}% missing`}>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-100" aria-hidden>
                      <div
                        className={cx('h-full rounded-full', c.missing_pct > 20 ? 'bg-amber-500' : 'bg-slate-400')}
                        style={{ width: `${c.missing_pct ? Math.max(3, c.missing_pct) : 0}%` }}
                      />
                    </div>
                    <span className="tabular font-mono text-[11px] text-slate-600">{c.missing_pct}%</span>
                  </div>
                </td>
                <td className="tabular px-3 py-2 text-right font-mono text-[12px] text-slate-600">{c.unique.toLocaleString()}</td>
                <td className="px-3 py-2">
                  <Mini c={c} />
                </td>
                <td className="max-w-[320px] truncate px-4 py-2 text-[12px] text-slate-600" title={stats(c)}>
                  {stats(c)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap gap-2">
        {profile.issues
          .filter((i) => i.type === 'id' || i.type === 'text')
          .map((i, k) => (
            <Badge key={k} color={SEVERITY[i.severity] ?? 'slate'}>
              {i.detail}
            </Badge>
          ))}
      </div>
    </div>
  )
}
