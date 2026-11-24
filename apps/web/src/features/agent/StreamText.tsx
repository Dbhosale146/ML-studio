import { useEffect, useMemo, useState } from 'react'

const CHARS_PER_TICK = 3
const TICK_MS = 14

type Part = { kind: 'b' | 'c' | 't'; text: string }

function parse(text: string): Part[] {
  return text
    .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    .filter(Boolean)
    .map((p): Part => {
      if (p.startsWith('**') && p.endsWith('**')) return { kind: 'b', text: p.slice(2, -2) }
      if (p.startsWith('`') && p.endsWith('`')) return { kind: 'c', text: p.slice(1, -1) }
      return { kind: 't', text: p }
    })
}

function renderPlain(s: string, key: number) {
  const lines = s.split('\n')
  return lines.map((l, i) => (
    <span key={`${key}-${i}`}>
      {l}
      {i < lines.length - 1 && <br />}
    </span>
  ))
}

/** Text that keeps **bold** and `code` formatting while it reveals. `stream` animates new text only. */
export default function StreamText({ text, stream = false }: { text: string; stream?: boolean }) {
  const parts = useMemo(() => parse(text), [text])
  const total = useMemo(() => parts.reduce((s, p) => s + p.text.length, 0), [parts])
  const [n, setN] = useState(stream ? 0 : Number.POSITIVE_INFINITY)

  useEffect(() => {
    if (!stream) return
    const t = setInterval(() => {
      setN((v) => {
        if (v >= total) {
          clearInterval(t)
          return v
        }
        return v + CHARS_PER_TICK
      })
    }, TICK_MS)
    return () => clearInterval(t)
  }, [stream, total])

  let budget = n
  const out = []
  for (let i = 0; i < parts.length && budget > 0; i++) {
    const p = parts[i]
    const s = p.text.slice(0, budget)
    budget -= p.text.length
    if (p.kind === 'b')
      out.push(
        <strong key={i} className="font-semibold text-slate-900">
          {s}
        </strong>,
      )
    else if (p.kind === 'c')
      out.push(
        <code key={i} className="rounded bg-slate-100 px-1 py-px font-mono text-[0.86em] text-slate-800">
          {s}
        </code>,
      )
    else out.push(...renderPlain(s, i))
  }
  return (
    <span>
      {out}
      {n < total && (
        <span className="ml-0.5 inline-block h-[1.05em] w-[7px] translate-y-[2px] animate-blink rounded-[1px] bg-indigo-500" aria-hidden />
      )}
    </span>
  )
}

export function streamDuration(text: string): number {
  return Math.ceil(text.replace(/\*\*|`/g, '').length / CHARS_PER_TICK) * TICK_MS
}
