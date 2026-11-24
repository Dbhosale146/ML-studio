// @ts-nocheck -- compatibility shim for the scripted prototype, removed in milestone 6
export * from '../components/ui'

export function ConnectorLogo({ id, size = 36 }) {
  return (
    <span
      className="grid shrink-0 place-items-center rounded-lg bg-slate-500 font-mono font-bold text-white"
      style={{ width: size, height: size, fontSize: size * 0.3 }}
    >
      {String(id ?? '')
        .slice(0, 2)
        .toUpperCase()}
    </span>
  )
}
