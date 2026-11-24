export function timeAgo(ts: number | string | null | undefined): string {
  if (ts == null) return '—'
  const t = typeof ts === 'string' ? Date.parse(ts) : ts
  const s = Math.max(1, Math.round((Date.now() - t) / 1000))
  if (s < 60) return 'just now'
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} hr${h > 1 ? 's' : ''} ago`
  const d = Math.round(h / 24)
  if (d < 30) return `${d} day${d > 1 ? 's' : ''} ago`
  return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

export function fmtDate(ts: number | string | null | undefined): string {
  if (ts == null) return '—'
  const t = typeof ts === 'string' ? Date.parse(ts) : ts
  return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

export function compact(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—'
  const a = Math.abs(n)
  if (a >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 1 : 2)}M`
  if (a >= 1e3) return `${(n / 1e3).toFixed(a >= 1e4 ? 0 : 1)}K`
  return String(Math.round(n * 100) / 100)
}

export function num(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toLocaleString('en-US', { maximumFractionDigits: 4 })
}

export function bytes(n: number | null | undefined): string {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

// Deterministic RNG, used only by the legacy scripted prototype until milestone 6 removes it.
export function rng(seed = 1): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function seedFrom(str = ''): number {
  let h = 2166136261
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 16777619)
  return h >>> 0
}
