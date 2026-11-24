import type { components } from '../types/api.gen'

export type Schemas = components['schemas']

export const API_BASE = '/api/v1'

export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly details: unknown
  readonly requestId: string | null

  constructor(status: number, code: string, message: string, details: unknown, requestId: string | null) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
    this.requestId = requestId
  }
}

async function parseError(res: Response): Promise<ApiError> {
  const rid = res.headers.get('X-Request-ID')
  try {
    const body = (await res.json()) as Partial<Schemas['ErrorEnvelope']>
    if (body.error) return new ApiError(res.status, body.error.code, body.error.message, body.error.details, body.request_id ?? rid)
  } catch {
    // not a JSON error envelope
  }
  if (res.status === 502 || res.status === 503 || res.status === 504)
    return new ApiError(res.status, 'backend_unreachable', 'The backend is not reachable. Is it running on port 8000?', null, rid)
  return new ApiError(res.status, 'http_error', `Request failed (${res.status})`, null, rid)
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers: { Accept: 'application/json', ...(init.headers ?? {}) } })
  } catch {
    throw new ApiError(0, 'network_error', 'Cannot reach the backend. Start it with `make dev` (or ./make.ps1 dev).', null, null)
  }
  if (!res.ok) throw await parseError(res)
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  upload: <T>(path: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<T>(path, { method: 'POST', body: form })
  },
  del: (path: string) => request<void>(path, { method: 'DELETE' }),
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message
  if (e instanceof Error) return e.message
  return 'Something went wrong.'
}
