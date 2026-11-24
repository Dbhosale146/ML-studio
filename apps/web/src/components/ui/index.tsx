import { useState, type ButtonHTMLAttributes, type ComponentType, type HTMLAttributes, type ReactNode } from 'react'
import { Area, AreaChart, ResponsiveContainer } from 'recharts'
import { Calculator, Check, Copy, LoaderCircle, Target, TrendingUp, type LucideProps } from 'lucide-react'
import { cx } from '../../lib/cx'

export { cx }

type Icon = ComponentType<LucideProps>

export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cx('rounded-xl border border-slate-200 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)]', className)} {...rest}>
      {children}
    </div>
  )
}

export function CardHeader({
  title,
  subtitle,
  icon: IconC,
  right,
  className,
}: {
  title: ReactNode
  subtitle?: ReactNode
  icon?: Icon
  right?: ReactNode
  className?: string
}) {
  return (
    <div className={cx('flex items-start justify-between gap-3 px-5 pt-4 pb-3', className)}>
      <div className="flex min-w-0 items-start gap-2.5">
        {IconC && (
          <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-slate-100 text-slate-600">
            <IconC size={15} aria-hidden />
          </span>
        )}
        <div className="min-w-0">
          <h3 className="text-[15px] font-semibold tracking-tight text-slate-900">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        </div>
      </div>
      {right}
    </div>
  )
}

const BADGE = {
  slate: 'bg-slate-100 text-slate-700 border-slate-200',
  emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  indigo: 'bg-indigo-50 text-indigo-700 border-indigo-200',
  amber: 'bg-amber-50 text-amber-700 border-amber-200',
  red: 'bg-red-50 text-red-700 border-red-200',
  sky: 'bg-sky-50 text-sky-700 border-sky-200',
  dark: 'bg-slate-900 text-white border-slate-900',
} as const
export type BadgeColor = keyof typeof BADGE

export function Badge({
  color = 'slate',
  className,
  children,
  mono,
}: {
  color?: BadgeColor
  className?: string
  children: ReactNode
  mono?: boolean
}) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-4',
        mono && 'font-mono',
        BADGE[color],
        className,
      )}
    >
      {children}
    </span>
  )
}

export type ModelStatus = 'draft' | 'training' | 'ready' | 'failed'
export const STATUS: Record<ModelStatus, { label: string; color: BadgeColor; dot: string }> = {
  draft: { label: 'Draft', color: 'slate', dot: 'bg-slate-400' },
  training: { label: 'Training', color: 'amber', dot: 'bg-amber-500' },
  ready: { label: 'Ready', color: 'emerald', dot: 'bg-emerald-500' },
  failed: { label: 'Failed', color: 'red', dot: 'bg-red-500' },
}

export function StatusPill({ status, progress }: { status: string; progress?: number | null }) {
  const s = STATUS[status as ModelStatus] ?? STATUS.draft
  return (
    <span
      className={cx('inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold', BADGE[s.color])}
    >
      {status === 'training' ? (
        <LoaderCircle size={11} className="animate-spin" aria-hidden />
      ) : (
        <span className={cx('relative h-1.5 w-1.5 rounded-full', s.dot)} />
      )}
      {s.label}
      {status === 'training' && progress != null && <span className="tabular font-mono">{progress}%</span>}
    </span>
  )
}

export type ProblemKind = 'classification' | 'regression' | 'forecasting'
export const KIND: Record<ProblemKind, { label: string; icon: Icon; tint: string }> = {
  classification: { label: 'Classification', icon: Target, tint: 'bg-sky-50 text-sky-600 ring-sky-100' },
  regression: { label: 'Regression', icon: Calculator, tint: 'bg-amber-50 text-amber-600 ring-amber-100' },
  forecasting: { label: 'Forecasting', icon: TrendingUp, tint: 'bg-emerald-50 text-emerald-600 ring-emerald-100' },
}

export function KindIcon({ kind, size = 'md' }: { kind: string; size?: 'sm' | 'md' | 'lg' }) {
  const k = KIND[kind as ProblemKind] ?? KIND.classification
  const IconC = k.icon
  const dims = size === 'lg' ? 'h-11 w-11 rounded-xl' : size === 'sm' ? 'h-7 w-7 rounded-lg' : 'h-9 w-9 rounded-lg'
  return (
    <span className={cx('grid shrink-0 place-items-center ring-1', dims, k.tint)} aria-hidden>
      <IconC size={size === 'lg' ? 20 : size === 'sm' ? 14 : 17} />
    </span>
  )
}

const BTN = {
  primary: 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-[0_4px_14px_rgba(16,185,129,0.3)]',
  dark: 'bg-slate-900 text-white hover:bg-slate-800',
  secondary: 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 hover:border-slate-300',
  ghost: 'text-slate-600 hover:bg-slate-100 hover:text-slate-900',
  danger: 'bg-white text-red-700 border border-red-200 hover:bg-red-50',
  ai: 'bg-indigo-500 text-white hover:bg-indigo-600 shadow-[0_4px_14px_rgba(99,102,241,0.3)]',
} as const

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: keyof typeof BTN
  size?: 'sm' | 'md' | 'lg'
  icon?: Icon
  iconRight?: Icon
}

export function Button({
  variant = 'secondary',
  size = 'md',
  className,
  children,
  icon: IconL,
  iconRight: IconR,
  type = 'button',
  ...rest
}: ButtonProps) {
  const sz = size === 'sm' ? 'h-8 px-3 text-xs gap-1.5' : size === 'lg' ? 'h-11 px-5 text-sm gap-2' : 'h-9 px-3.5 text-[13px] gap-2'
  return (
    <button
      type={type}
      className={cx(
        'inline-flex items-center justify-center rounded-lg font-semibold transition-all duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-500 disabled:opacity-50',
        sz,
        BTN[variant],
        className,
      )}
      {...rest}
    >
      {IconL && <IconL size={size === 'sm' ? 14 : 16} aria-hidden />}
      {children}
      {IconR && <IconR size={size === 'sm' ? 14 : 16} aria-hidden />}
    </button>
  )
}

export function ProgressBar({
  value,
  color = 'bg-emerald-500',
  className,
  animated,
  label,
}: {
  value: number
  color?: string
  className?: string
  animated?: boolean
  label?: string
}) {
  return (
    <div
      className={cx('h-1.5 w-full overflow-hidden rounded-full bg-slate-100', className)}
      role="progressbar"
      aria-valuenow={Math.round(value)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div
        className={cx(
          'h-full rounded-full transition-[width] duration-700 ease-out',
          color,
          animated && 'animate-shimmer bg-gradient-to-r from-amber-400 via-amber-300 to-amber-400 bg-[length:200%_100%]',
        )}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}

/** Minimal inline formatting: **bold** and `code`. */
export function Rich({ text, className }: { text: string; className?: string }) {
  const parts = String(text).split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return (
    <span className={className}>
      {parts.map((p, i) => {
        if (p.startsWith('**') && p.endsWith('**'))
          return (
            <strong key={i} className="font-semibold text-slate-900">
              {p.slice(2, -2)}
            </strong>
          )
        if (p.startsWith('`') && p.endsWith('`'))
          return (
            <code key={i} className="rounded bg-slate-100 px-1 py-px font-mono text-[0.86em] text-slate-800">
              {p.slice(1, -1)}
            </code>
          )
        return p
      })}
    </span>
  )
}

export function Sparkline({
  data,
  color = '#10b981',
  height = 36,
  dataKey = 'value',
}: {
  data: Array<Record<string, number | string>>
  color?: string
  height?: number
  dataKey?: string
}) {
  const id = `sg-${color.replace('#', '')}`
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.25} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.75} fill={`url(#${id})`} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export type TabDef = { id: string; label: string; icon?: Icon; count?: number | null }

export function Tabs({ tabs, value, onChange, className }: { tabs: TabDef[]; value: string; onChange: (id: string) => void; className?: string }) {
  return (
    <div className={cx('thin-scroll flex gap-1 overflow-x-auto border-b border-slate-200', className)} role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          role="tab"
          aria-selected={value === t.id}
          onClick={() => onChange(t.id)}
          className={cx(
            'relative -mb-px flex items-center gap-1.5 whitespace-nowrap border-b-2 px-3 py-2.5 text-[13px] font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-emerald-500',
            value === t.id ? 'border-emerald-500 text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-800',
          )}
        >
          {t.icon && <t.icon size={15} aria-hidden />}
          {t.label}
          {t.count != null && <span className="tabular rounded-full bg-slate-100 px-1.5 text-[10px] font-semibold text-slate-600">{t.count}</span>}
        </button>
      ))}
    </div>
  )
}

export function CodeBlock({ code, label, className }: { code: string; label: string; className?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    void navigator.clipboard?.writeText(code).catch(() => undefined)
    setCopied(true)
    setTimeout(() => setCopied(false), 1400)
  }
  return (
    <div className={cx('overflow-hidden rounded-lg bg-slate-900', className)}>
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-1.5">
        <span className="font-mono text-[11px] text-slate-400">{label}</span>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-slate-300 hover:bg-white/10"
          aria-label="Copy to clipboard"
        >
          {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="thin-scroll overflow-x-auto p-3 font-mono text-[12px] leading-5 text-slate-200">{code}</pre>
    </div>
  )
}

export function Stat({ label, value, sub, className }: { label: string; value: ReactNode; sub?: ReactNode; className?: string }) {
  return (
    <div className={cx('min-w-0', className)}>
      <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</div>
      <div className="tabular mt-1 truncate text-lg font-semibold tracking-tight text-slate-900">{value}</div>
      {sub && <div className="mt-0.5 truncate text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

export function Empty({ icon: IconC, title, text, action }: { icon?: Icon; title: string; text?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      {IconC && (
        <span className="mb-3 grid h-11 w-11 place-items-center rounded-xl bg-slate-100 text-slate-500">
          <IconC size={20} aria-hidden />
        </span>
      )}
      <div className="text-sm font-semibold text-slate-900">{title}</div>
      {text && <p className="mt-1 max-w-sm text-sm text-slate-500">{text}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-500" role="status">
      <LoaderCircle size={16} className="animate-spin" aria-hidden /> {label}
    </span>
  )
}
