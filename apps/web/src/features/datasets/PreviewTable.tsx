import type { Preview } from '../../types/domain'

export function PreviewTable({ preview, maxHeight = 420 }: { preview: Preview; maxHeight?: number }) {
  return (
    <div className="thin-scroll overflow-auto rounded-lg border border-slate-200" style={{ maxHeight }}>
      <table className="w-full min-w-max text-left">
        <thead className="sticky top-0 z-10 bg-slate-50">
          <tr className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            <th scope="col" className="border-b border-slate-200 px-3 py-2 text-right">
              #
            </th>
            {preview.columns.map((c) => (
              <th key={c} scope="col" className="whitespace-nowrap border-b border-slate-200 px-3 py-2 font-mono normal-case tracking-normal">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, i) => (
            <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
              <td className="tabular px-3 py-1.5 text-right font-mono text-[11px] text-slate-400">{i + 1}</td>
              {row.map((v, j) => (
                <td key={j} className="max-w-[240px] truncate whitespace-nowrap px-3 py-1.5 text-[12px] text-slate-700" title={v}>
                  {v === '' ? <span className="text-slate-300">·</span> : v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="sticky bottom-0 border-t border-slate-200 bg-white px-3 py-1.5 text-[11px] text-slate-500">
        Showing {preview.rows.length.toLocaleString()} of {preview.total_rows.toLocaleString()} rows
      </div>
    </div>
  )
}
