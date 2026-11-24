import { FileSpreadsheet, LoaderCircle, Upload } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'
import { errorMessage } from '../../lib/apiClient'
import { cx } from '../../lib/cx'
import type { Dataset } from '../../types/domain'
import { useUploadDataset } from './hooks'

export const ACCEPT = '.xlsx,.xls,.xlsm,.csv'

export function UploadDropzone({ onUploaded, compact = false }: { onUploaded?: (ds: Dataset) => void; compact?: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const upload = useUploadDataset()

  const send = (file: File | undefined) => {
    if (!file || upload.isPending) return
    upload.mutate(file, { onSuccess: (ds) => onUploaded?.(ds) })
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    send(e.dataTransfer.files?.[0])
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        disabled={upload.isPending}
        aria-label="Upload an Excel or CSV file"
        className={cx(
          'flex w-full items-center gap-3 rounded-xl border-2 border-dashed text-left transition-colors focus-visible:outline-2 focus-visible:outline-emerald-500',
          compact ? 'p-3' : 'flex-col justify-center p-8 text-center',
          over ? 'border-emerald-400 bg-emerald-50' : 'border-slate-300 bg-white hover:border-emerald-300 hover:bg-emerald-50/40',
        )}
      >
        <span className={cx('grid shrink-0 place-items-center rounded-xl bg-slate-100 text-slate-500', compact ? 'h-9 w-9' : 'h-12 w-12')}>
          {upload.isPending ? <LoaderCircle size={20} className="animate-spin" /> : compact ? <Upload size={18} /> : <FileSpreadsheet size={22} />}
        </span>
        <span>
          <span className="block text-[13px] font-semibold text-slate-900">
            {upload.isPending ? 'Uploading and reading the file…' : 'Drop an Excel or CSV file here'}
          </span>
          <span className="block text-xs text-slate-500">or click to browse · .xlsx, .xls, .csv</span>
        </span>
      </button>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          send(e.target.files?.[0])
          e.target.value = ''
        }}
      />
      {upload.isError && (
        <p role="alert" className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[13px] text-red-800">
          {errorMessage(upload.error)}
        </p>
      )}
    </div>
  )
}
