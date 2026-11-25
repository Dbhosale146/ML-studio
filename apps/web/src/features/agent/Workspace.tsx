import { Check, LoaderCircle, X } from 'lucide-react'
import { cx } from '../../lib/cx'
import type { AgentState } from './reducer'
import { STAGES } from './reducer'

export type WorkspaceTab = 'blueprint' | 'decisions' | 'experiments'

function Stepper({ stages }: { stages: AgentState['stages'] }) {
  return (
    <ol className="space-y-0.5" aria-label="Progress">
      {STAGES.map((s, i) => {
        const st = stages[s.key]
        return (
          <li
            key={s.key}
            className={cx('flex items-center gap-3 rounded-lg px-2 py-1.5 transition-colors', st === 'active' && 'bg-indigo-50')}
            aria-current={st === 'active' ? 'step' : undefined}
          >
            <span
              className={cx(
                'grid h-6 w-6 shrink-0 place-items-center rounded-full text-[11px] font-bold',
                st === 'done'
                  ? 'bg-emerald-500 text-white'
                  : st === 'active'
                    ? 'bg-indigo-500 text-white'
                    : st === 'failed'
                      ? 'bg-red-500 text-white'
                      : 'bg-slate-100 text-slate-400',
              )}
            >
              {st === 'done' ? (
                <Check size={13} strokeWidth={3} aria-label="done" />
              ) : st === 'active' ? (
                <LoaderCircle size={13} className="animate-spin" aria-label="in progress" />
              ) : st === 'failed' ? (
                <X size={13} strokeWidth={3} aria-label="failed" />
              ) : (
                i + 1
              )}
            </span>
            <span
              className={cx(
                'text-[13px] font-semibold',
                st === 'done' ? 'text-slate-900' : st === 'active' ? 'text-indigo-900' : st === 'failed' ? 'text-red-700' : 'text-slate-400',
              )}
            >
              {s.label}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

export function Workspace({
  state,
  tab,
  setTab,
  className,
}: {
  state: AgentState
  tab: WorkspaceTab
  setTab: (t: WorkspaceTab) => void
  className?: string
}) {
  const blueprint = Object.entries(state.blueprint)
  return (
    <aside className={cx('w-[380px] shrink-0 flex-col border-l border-slate-200 bg-white xl:w-[400px]', className)} aria-label="Live workspace">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-center justify-between">
          <h2 className="text-[13px] font-semibold text-slate-900">Live workspace</h2>
          {state.model && (
            <span className="tabular font-mono text-[11px] text-slate-500">
              {state.model.status} · {state.model.progress}%
            </span>
          )}
        </div>
        <div className="mt-3">
          <Stepper stages={state.stages} />
        </div>
      </div>
      <div className="flex gap-1 border-b border-slate-200 px-3" role="tablist">
        {(
          [
            ['blueprint', 'Blueprint', null],
            ['decisions', 'Decisions', state.decisions.length],
            ['experiments', 'Experiments', state.experiments.length],
          ] as const
        ).map(([id, label, count]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={cx(
              '-mb-px flex items-center gap-1.5 border-b-2 px-2.5 py-2.5 text-[12px] font-semibold focus-visible:outline-2 focus-visible:outline-emerald-500',
              tab === id ? 'border-emerald-500 text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-800',
            )}
          >
            {label}
            {count != null && <span className="tabular rounded-full bg-slate-100 px-1.5 text-[10px]">{count}</span>}
          </button>
        ))}
      </div>
      <div className="thin-scroll min-h-0 flex-1 overflow-y-auto px-5 py-3" role="tabpanel">
        {tab === 'blueprint' &&
          (blueprint.length === 0 ? (
            <p className="py-8 text-center text-[13px] text-slate-500">The blueprint fills in as the agent works.</p>
          ) : (
            <dl>
              {blueprint.map(([k, v]) => (
                <div key={k} className="border-b border-slate-100 py-2 last:border-0">
                  <dt className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{k}</dt>
                  <dd className="mt-0.5 animate-fade-in break-words text-[13px] font-medium text-slate-900">{v}</dd>
                </div>
              ))}
            </dl>
          ))}
        {tab === 'decisions' &&
          (state.decisions.length === 0 ? (
            <p className="py-8 text-center text-[13px] text-slate-500">Decisions appear here as the agent makes them.</p>
          ) : (
            <ol className="space-y-2.5">
              {state.decisions.map((d, i) => (
                <li key={i} className="animate-fade-up rounded-lg border border-slate-200 p-3">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] font-bold text-slate-400">{String(i + 1).padStart(2, '0')}</span>
                    <span className="rounded-md border border-indigo-200 bg-indigo-50 px-1.5 text-[11px] font-medium text-indigo-700">{d.step}</span>
                  </div>
                  <div className="mt-1.5 text-[13px] font-semibold text-slate-900">{d.decision}</div>
                  <div className="mt-0.5 text-[12px] text-slate-600">{d.reason}</div>
                  {d.alternatives && <div className="mt-0.5 text-[11px] text-slate-400">Considered: {d.alternatives}</div>}
                </li>
              ))}
            </ol>
          ))}
        {tab === 'experiments' &&
          (state.experiments.length === 0 ? (
            <p className="py-8 text-center text-[13px] text-slate-500">Every training run is tracked here (and in MLflow).</p>
          ) : (
            <table className="w-full text-left">
              <thead>
                <tr className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="py-1.5">Run</th>
                  <th className="py-1.5">Model</th>
                  <th className="py-1.5 text-right">Score</th>
                </tr>
              </thead>
              <tbody>
                {state.experiments.map((e) => (
                  <tr key={e.run_id} className="animate-fade-in border-t border-slate-100">
                    <td className="py-2 font-mono text-[10px] text-slate-500">{e.run_id}</td>
                    <td className="py-2 text-[12px] font-medium text-slate-800">
                      {e.model}
                      <div className="text-[10px] text-slate-400">
                        round {e.round}
                        {e.tuned ? ' · tuned' : ''}
                      </div>
                    </td>
                    <td className="tabular py-2 text-right font-mono text-[12px] text-slate-700">{e.display}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
      </div>
    </aside>
  )
}
