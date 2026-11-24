import { Boxes, Building2, Database, LayoutDashboard, Menu, Search, Sparkles, X, type LucideProps } from 'lucide-react'
import { useState, type ComponentType } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useHealth } from '../../features/datasets/hooks'
import { cx } from '../../lib/cx'

type NavItem = { to: string; label: string; icon: ComponentType<LucideProps>; end?: boolean; ai?: boolean }

const NAV: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/agent', label: 'Create New Model', icon: Sparkles, ai: true },
  { to: '/models', label: 'Model Catalog', icon: Boxes },
  { to: '/datasets', label: 'Datasets', icon: Database },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="relative grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-emerald-400 to-indigo-500 shadow-lg shadow-emerald-500/20">
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-white" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden>
          <path d="M4 18 9 12l4 3 7-9" />
          <circle cx="20" cy="6" r="1.6" fill="currentColor" stroke="none" />
        </svg>
      </span>
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-tight text-white">Datalyon</div>
        <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-slate-400">AI Data Scientist</div>
      </div>
    </div>
  )
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const health = useHealth()
  const online = health.isSuccess
  const llm = health.data?.llm
  return (
    <div className="flex h-full flex-col justify-between bg-navy">
      <div>
        <div className="flex h-16 items-center border-b border-white/10 px-5">
          <Logo />
        </div>
        <nav className="flex flex-col gap-1 p-3" aria-label="Main">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              onClick={onNavigate}
              className={({ isActive }) =>
                cx(
                  'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13px] font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-emerald-400',
                  isActive ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/25' : 'text-slate-300 hover:bg-white/5 hover:text-white',
                )
              }
            >
              <n.icon size={18} aria-hidden />
              <span className="flex-1">{n.label}</span>
              {n.ai && <span className="rounded bg-indigo-500/90 px-1.5 py-px text-[9px] font-bold uppercase tracking-wider text-white">AI</span>}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="space-y-2 border-t border-white/10 p-3">
        <div className="rounded-lg bg-white/5 px-3 py-2.5" aria-live="polite">
          <div className="flex items-center justify-between text-[11px]">
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className={cx('relative inline-flex h-2 w-2 rounded-full', online ? 'bg-emerald-500' : 'bg-red-500')} />
              Backend
            </span>
            <span className={cx('font-mono', online ? 'text-emerald-300' : 'text-red-300')}>
              {online ? 'Online' : health.isLoading ? '…' : 'Offline'}
            </span>
          </div>
          <div className="mt-1.5 truncate font-mono text-[10px] text-slate-400" title={llm?.note ?? undefined}>
            {llm ? (llm.provider === 'gemini' ? `LLM: ${llm.model}` : 'LLM: MockLLM (offline rules)') : 'LLM: unknown'}
          </div>
        </div>
        <div className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
          <span className="grid h-8 w-8 place-items-center rounded-full bg-gradient-to-br from-indigo-400 to-indigo-600 text-[12px] font-bold uppercase text-white">
            {(health.data?.display_name ?? 'U').slice(0, 2)}
          </span>
          <div className="min-w-0 leading-tight">
            <div className="truncate text-[13px] font-semibold text-white">{health.data?.display_name ?? 'Workspace'}</div>
            <div className="truncate text-[11px] text-slate-400">Local workspace</div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Topbar({ onMenu }: { onMenu: () => void }) {
  const navigate = useNavigate()
  const [q, setQ] = useState('')
  const health = useHealth()
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-slate-200 bg-white/90 px-4 backdrop-blur-xl lg:px-6">
      <button type="button" onClick={onMenu} className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 lg:hidden" aria-label="Open navigation">
        <Menu size={20} />
      </button>
      <span className="hidden items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-[13px] font-semibold text-slate-800 md:flex">
        <Building2 size={16} className="text-slate-500" aria-hidden />
        Local workspace
        <span className="rounded bg-emerald-500 px-1.5 py-px text-[9px] font-bold uppercase tracking-wider text-white">POC</span>
      </span>
      <form
        role="search"
        className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 focus-within:border-emerald-400 lg:max-w-md"
        onSubmit={(e) => {
          e.preventDefault()
          navigate(`/models${q ? `?q=${encodeURIComponent(q)}` : ''}`)
        }}
      >
        <Search size={15} className="shrink-0 text-slate-400" aria-hidden />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search models…"
          aria-label="Search models"
          className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-slate-400"
        />
      </form>
      {health.isError && (
        <span role="status" className="ml-auto rounded-full border border-red-200 bg-red-50 px-3 py-1 text-[12px] font-semibold text-red-700">
          Backend offline: start it with make dev
        </span>
      )}
    </header>
  )
}

export default function Layout() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  const fullBleed = pathname.startsWith('/agent')
  return (
    <div className="min-h-full">
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 lg:block">
        <Sidebar />
      </aside>
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/50" onClick={() => setOpen(false)} aria-hidden />
          <div className="absolute inset-y-0 left-0 w-64 animate-fade-in">
            <Sidebar onNavigate={() => setOpen(false)} />
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="absolute right-2 top-4 rounded p-1 text-slate-300 hover:bg-white/10"
              aria-label="Close navigation"
            >
              <X size={18} />
            </button>
          </div>
        </div>
      )}
      <div className="lg:pl-64">
        <Topbar onMenu={() => setOpen(true)} />
        <main className={fullBleed ? '' : 'mx-auto max-w-[1440px] px-4 py-6 lg:px-8'}>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
