import type { Schemas } from '../lib/apiClient'

export type Dataset = Schemas['DatasetOut']
export type Health = Schemas['HealthOut']
export type Activity = Schemas['ActivityOut']
export type Preview = Schemas['PreviewOut']

export type ColumnKind = 'id' | 'number' | 'integer' | 'category' | 'boolean' | 'date' | 'text' | 'constant' | 'empty'

export interface ColumnProfile {
  name: string
  kind: ColumnKind
  role: string
  missing_pct: number
  missing: number
  unique: number
  pii: boolean
  sample: string[]
  min?: number | string
  max?: number | string
  mean?: number
  median?: number
  std?: number
  skew?: number
  outliers?: number
  histogram?: Array<{ x0: number; x1: number; count: number }>
  top_values?: Array<{ value: string; count: number; pct: number }>
}

export interface QualityIssue {
  column: string | null
  type: string
  severity: 'high' | 'medium' | 'low' | 'info'
  detail: string
}

export interface Profile {
  file_name: string
  sheet: string
  rows: number
  cols: number
  duplicates: number
  missing_pct: number
  quality: number
  quality_reasons: Array<{ factor: string; penalty: number; detail: string }>
  period: string | null
  columns: ColumnProfile[]
  issues: QualityIssue[]
  notes: string[]
}
