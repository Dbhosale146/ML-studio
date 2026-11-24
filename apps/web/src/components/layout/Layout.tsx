// @ts-nocheck -- uses the legacy demo store until milestone 6 wires it to the API
import { useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { Bell, Boxes, Building2, ChevronDown, Coins, LayoutDashboard, Menu, RotateCcw, Search, Sparkles, X } from 'lucide-react'
import { useStore } from '../../legacy/store'
import { cx } from '../ui'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/agent', label: 'Create New Model', icon: Sparkles, ai: true },
  { to: '/models', label: 'Model Catalog', icon: Boxes },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="relative grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-emerald-400 to-indigo-500 shadow-lg shadow-emerald-500/20">
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-white" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
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

function Sidebar({ onNavigate }) {
  const { models, resetDemo } = useStore()
  const training = models.filter((m) => m.status === 'training').length
  return (
    <div className="flex h-full flex-col justify-between bg-navy">
      <div>
        <div className="flex h-16 items-center border-b border-white/10 px-5">
          <Logo />
        </div>
        <nav className="flex flex-col gap-1 p-3">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              onClick={onNavigate}
              className={({ isActive }) =>
                cx(
                  'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13px] font-semibold transition-colors',
                  isActive ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/25' : 'text-slate-300 hover:bg-white/5 hover:text-white',
                )
              }
            >
              <n.icon size={18} />
              <span className="flex-1">{n.label}</span>
              {n.ai && <span className="rounded bg-indigo-500/90 px-1.5 py-px text-[9px] font-bold uppercase tracking-wider text-white">AI</span>}
              {n.to === '/models' && <span className="rounded-full bg-white/10 px-1.5 text-[10px] tabular text-slate-200">{models.length}</span>}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="space-y-2 border-t border-white/10 p-3">
        <div className="rounded-lg bg-white/5 px-3 py-2.5">
          <div className="flex items-center justify-between text-[11px]">
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
              </span>
              Agent engine
            </span>
            <span className="font-mono text-emerald-300">Optimal</span>
          </div>
          <div className="mt-1.5 flex items-center justify-between font-mono text-[10px] text-slate-400">
            <span>
              {training} job{training === 1 ? '' : 's'} running
            </span>
            <span>GPU 38%</span>
          </div>
        </div>
        <button
          onClick={() => {
            if (window.confirm('Reset the prototype to its original demo data? Models you built will be removed.')) resetDemo()
          }}
          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-[12px] font-medium text-slate-400 hover:bg-white/5 hover:text-slate-200"
        >
          <RotateCcw size={14} /> Reset demo data
        </button>
        <div className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
          <span className="grid h-8 w-8 place-items-center rounded-full bg-gradient-to-br from-indigo-400 to-indigo-600 text-[12px] font-bold text-white">
            AK
          </span>
          <div className="min-w-0 leading-tight">
            <div className="truncate text-[13px] font-semibold text-white">Akash Kesarkar</div>
            <div className="truncate text-[11px] text-slate-400">Workspace admin</div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Topbar({ onMenu }) {
  const navigate = useNavigate()
  const [q, setQ] = useState('')
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-slate-200 bg-white/90 px-4 backdrop-blur-xl lg:px-6">
      <button onClick={onMenu} className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 lg:hidden" aria-label="Open navigation">
        <Menu size={20} />
      </button>
      <button className="hidden items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-[13px] font-semibold text-slate-800 md:flex">
        <Building2 size={16} className="text-slate-500" />
        Acme Corp
        <span className="rounded bg-emerald-500 px-1.5 py-px text-[9px] font-bold uppercase tracking-wider text-white">Prod</span>
        <ChevronDown size={15} className="text-slate-400" />
      </button>
      <form
        className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 lg:max-w-md"
        onSubmit={(e) => {
          e.preventDefault()
          navigate(`/models${q ? `?q=${encodeURIComponent(q)}` : ''}`)
        }}
      >
        <Search size={15} className="shrink-0 text-slate-400" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search models, datasets, metrics…"
          className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-slate-400"
        />
        <kbd className="hidden rounded border border-slate-200 bg-white px-1.5 font-mono text-[10px] text-slate-500 sm:block">⏎</kbd>
      </form>
      <div className="ml-auto flex items-center gap-2">
        <span className="hidden items-center gap-1.5 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-[12px] font-semibold text-indigo-700 sm:flex">
          <Coins size={14} /> 1,840 compute credits
        </span>
        <button className="relative rounded-lg p-2 text-slate-500 hover:bg-slate-100" aria-label="Notifications">
          <Bell size={18} />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-red-500 ring-2 ring-white" />
        </button>
      </div>
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
          <div className="absolute inset-0 bg-slate-900/50" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-64 animate-fade-in">
            <Sidebar onNavigate={() => setOpen(false)} />
            <button
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
