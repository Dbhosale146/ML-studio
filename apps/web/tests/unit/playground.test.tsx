import { describe, expect, it } from 'vitest'
import { defaultValues, fieldWarning, fromExample, splitFields, toRow, type Field } from '../../src/features/playground/form'

const fields: Field[] = [
  { name: 'age', label: 'Age', kind: 'integer', importance: 0.1, default: 40, min: 18, max: 90, help: '' },
  { name: 'spend', label: 'Spend', kind: 'number', importance: 0.5, default: 120.5, min: 0, max: 1000, help: '' },
  {
    name: 'plan',
    label: 'Plan',
    kind: 'category',
    importance: 0.3,
    default: 'Pro',
    options: [{ value: 'Pro' }, { value: 'Basic' }],
    allow_other: false,
    help: '',
  },
  { name: 'active', label: 'Active', kind: 'boolean', importance: 0.0, default: 'true', help: '' },
].map((f) => ({ required: false, allow_other: false, ...f })) as Field[]

describe('Playground form helpers', () => {
  it('prefills typical values', () => {
    expect(defaultValues(fields)).toEqual({ age: '40', spend: '120.5', plan: 'Pro', active: 'true' })
  })

  it('builds a typed row; blanks become null so the model imputes them', () => {
    const row = toRow(fields, { age: '35', spend: '1,250', plan: 'Basic', active: '' })
    expect(row).toEqual({ age: 35, spend: 1250, plan: 'Basic', active: null })
  })

  it('warns about out-of-range numbers, non-numbers and unseen categories', () => {
    expect(fieldWarning(fields[0], '150')).toMatch(/training range/)
    expect(fieldWarning(fields[0], '30.5')).toMatch(/whole number/)
    expect(fieldWarning(fields[1], 'abc')).toMatch(/Not a number/)
    expect(fieldWarning(fields[2], 'Gold')).toMatch(/Never seen/)
    expect(fieldWarning(fields[2], 'pro')).toBeNull()
    expect(fieldWarning(fields[1], '')).toBeNull()
  })

  it('loads a training example and orders fields by importance', () => {
    expect(fromExample(fields, { age: 22, plan: null })).toEqual({ age: '22', spend: '', plan: '', active: '' })
    const { main, more } = splitFields(fields, 2)
    expect(main.map((f) => f.name)).toEqual(['spend', 'plan'])
    expect(more.map((f) => f.name)).toEqual(['age', 'active'])
  })
})
