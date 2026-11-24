// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { seedActivity, seedConnectors, seedModels } from './models'

const KEY = 'datalyon-prototype-v1'
const StoreCtx = createContext(null)

function initialState() {
  return { version: 1, models: seedModels(), connectors: seedConnectors(), activity: seedActivity() }
}

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY))
    if (saved && saved.version === 1 && Array.isArray(saved.models)) return saved
  } catch {
    // storage unavailable or corrupt: fall back to seed data
  }
  return initialState()
}

export function StoreProvider({ children }) {
  const [state, setState] = useState(load)

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(state))
    } catch {
      // ignore quota / private-mode errors
    }
  }, [state])

  // Keep multiple open tabs in sync: build a model in one tab, watch it appear in another.
  useEffect(() => {
    const onStorage = (e) => {
      if (e.key !== KEY || !e.newValue) return
      try {
        setState(JSON.parse(e.newValue))
      } catch {
        // ignore malformed payloads
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  // Seeded background training jobs keep progressing so the dashboard feels live.
  useEffect(() => {
    const t = setInterval(() => {
      setState((s) => {
        const busy = s.models.some((m) => m.status === 'training' && m.createdBy === 'seed' && m.progress < 97)
        if (!busy) return s
        return {
          ...s,
          models: s.models.map((m) =>
            m.status === 'training' && m.createdBy === 'seed' && m.progress < 97 ? { ...m, progress: m.progress + 1, updatedAt: Date.now() } : m,
          ),
        }
      })
    }, 4000)
    return () => clearInterval(t)
  }, [])

  const upsertModel = useCallback((model) => {
    setState((s) => {
      const exists = s.models.some((m) => m.id === model.id)
      return { ...s, models: exists ? s.models.map((m) => (m.id === model.id ? model : m)) : [model, ...s.models] }
    })
  }, [])

  const patchModel = useCallback((id, patch) => {
    setState((s) => ({
      ...s,
      models: s.models.map((m) => (m.id === id ? { ...m, ...(typeof patch === 'function' ? patch(m) : patch), updatedAt: Date.now() } : m)),
    }))
  }, [])

  const removeModel = useCallback((id) => {
    setState((s) => ({ ...s, models: s.models.filter((m) => m.id !== id) }))
  }, [])

  const addActivity = useCallback((type, text) => {
    setState((s) => ({
      ...s,
      activity: [{ id: `a${Date.now()}${Math.random().toString(36).slice(2, 6)}`, type, text, at: Date.now() }, ...s.activity].slice(0, 40),
    }))
  }, [])

  const upsertConnector = useCallback((conn) => {
    setState((s) => {
      const exists = s.connectors.some((c) => c.id === conn.id)
      return { ...s, connectors: exists ? s.connectors.map((c) => (c.id === conn.id ? { ...c, ...conn } : c)) : [...s.connectors, conn] }
    })
  }, [])

  const resetDemo = useCallback(() => setState(initialState()), [])

  const api = useMemo(
    () => ({ ...state, upsertModel, patchModel, removeModel, addActivity, upsertConnector, resetDemo }),
    [state, upsertModel, patchModel, removeModel, addActivity, upsertConnector, resetDemo],
  )

  return <StoreCtx.Provider value={api}>{children}</StoreCtx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useStore() {
  return useContext(StoreCtx)
}
