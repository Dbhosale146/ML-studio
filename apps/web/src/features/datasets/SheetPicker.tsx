import { Sheet } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button } from '../../components/ui'
import { errorMessage } from '../../lib/apiClient'
import { cx } from '../../lib/cx'
import type { Dataset } from '../../types/domain'
import { useChooseSheet } from './hooks'

export function SheetPicker({ dataset, onChosen }: { dataset: Dataset; onChosen?: (ds: Dataset) => void }) {
  const largest = [...dataset.sheets].sort((a, b) => b.rows - a.rows)[0]?.name
  const [sheet, setSheet] = useState(dataset.sheet ?? largest ?? '')
  const [headerRow, setHeaderRow] = useState('')
  const choose = useChooseSheet(dataset.id)

  return (
    <div className="space-y-3">
      <div className="text-[13px] text-slate-600">
        <strong className="text-slate-900">{dataset.file_name}</strong> has {dataset.sheets.length} sheets. Which one holds your data?
      </div>
      <div className="grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Sheets">
        {dataset.sheets.map((s) => (
          <button
            key={s.name}
            type="button"
            role="radio"
            aria-checked={sheet === s.name}
            onClick={() => setSheet(s.name)}
            className={cx(
              'flex items-center gap-3 rounded-lg border p-3 text-left transition-colors focus-visible:outline-2 focus-visible:outline-emerald-500',
              sheet === s.name ? 'border-emerald-400 bg-emerald-50 ring-2 ring-emerald-100' : 'border-slate-200 bg-white hover:border-slate-300',
            )}
          >
            <Sheet size={18} className="shrink-0 text-slate-400" aria-hidden />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-semibold text-slate-900">{s.name}</span>
              <span className="tabular block text-xs text-slate-500">
                {s.rows.toLocaleString()} rows · {s.cols} columns
              </span>
            </span>
            {s.name === largest && <Badge color="emerald">Largest</Badge>}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-xs text-slate-600">
          Header row (optional)
          <input
            type="number"
            min={1}
            value={headerRow}
            onChange={(e) => setHeaderRow(e.target.value)}
            placeholder="auto"
            className="mt-1 block h-9 w-28 rounded-lg border border-slate-200 px-2 text-[13px] outline-none focus:border-emerald-400"
          />
        </label>
        <Button
          variant="primary"
          disabled={!sheet || choose.isPending}
          onClick={() =>
            choose.mutate({ sheet, header_row: headerRow ? Math.max(0, Number(headerRow) - 1) : null }, { onSuccess: (ds) => onChosen?.(ds) })
          }
        >
          {choose.isPending ? 'Reading sheet…' : 'Use this sheet'}
        </Button>
      </div>
      {choose.isError && (
        <p role="alert" className="text-[13px] text-red-700">
          {errorMessage(choose.error)}
        </p>
      )}
    </div>
  )
}
