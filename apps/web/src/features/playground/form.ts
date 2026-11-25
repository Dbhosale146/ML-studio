/** Pure helpers behind the auto-generated Test-model form (unit-tested). */
import type { Schemas } from '../../lib/apiClient'

export type Field = Schemas['InputField']
export type FormValues = Record<string, string>

export function defaultValues(fields: Field[]): FormValues {
  const out: FormValues = {}
  for (const f of fields) out[f.name] = f.default == null ? '' : String(f.default)
  return out
}

export function fromExample(fields: Field[], inputs: Record<string, unknown>): FormValues {
  const out: FormValues = {}
  for (const f of fields) {
    const v = inputs[f.name]
    out[f.name] = v == null ? '' : String(v)
  }
  return out
}

/** Form strings -> the row sent to /predict (numbers as numbers, blanks as null so the model imputes them). */
export function toRow(fields: Field[], values: FormValues): Record<string, string | number | null> {
  const row: Record<string, string | number | null> = {}
  for (const f of fields) {
    const raw = (values[f.name] ?? '').trim()
    if (raw === '') {
      row[f.name] = null
    } else if (f.kind === 'number' || f.kind === 'integer') {
      const n = Number(raw.replace(/,/g, ''))
      row[f.name] = Number.isFinite(n) ? n : raw
    } else {
      row[f.name] = raw
    }
  }
  return row
}

/** Client-side hints shown under a field before submitting. */
export function fieldWarning(f: Field, value: string): string | null {
  const raw = value.trim()
  if (raw === '') return null
  if (f.kind === 'number' || f.kind === 'integer') {
    const n = Number(raw.replace(/,/g, ''))
    if (!Number.isFinite(n)) return 'Not a number'
    if (f.kind === 'integer' && !Number.isInteger(n)) return 'Expected a whole number'
    if (typeof f.min === 'number' && typeof f.max === 'number' && (n < f.min || n > f.max)) return 'Outside the training range: less reliable'
  }
  if (f.kind === 'category' && f.options && !f.allow_other) {
    const known = f.options.some((o) => String(o.value).toLowerCase() === raw.toLowerCase())
    if (!known) return 'Never seen in training'
  }
  return null
}

export function splitFields(fields: Field[], visible = 8): { main: Field[]; more: Field[] } {
  const sorted = [...fields].sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0))
  return { main: sorted.slice(0, visible), more: sorted.slice(visible) }
}
