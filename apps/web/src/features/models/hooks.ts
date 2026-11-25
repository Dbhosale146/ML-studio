import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type Schemas } from '../../lib/apiClient'

export type ModelSummary = Schemas['ModelSummaryOut']
export type ModelDetail = Schemas['ModelDetailOut']
export type InputSchema = Schemas['InputSchemaOut']
type ModelPage = { items: ModelSummary[]; total: number; limit: number; offset: number }

export const PROBLEM_KIND: Record<string, 'classification' | 'regression' | 'forecasting'> = {
  binary: 'classification',
  multiclass: 'classification',
  regression: 'regression',
  forecasting: 'forecasting',
}

export function useModels(filters: { q?: string; status?: string; problem_type?: string } = {}) {
  const qs = new URLSearchParams({ limit: '200' })
  Object.entries(filters).forEach(([k, v]) => v && qs.set(k, v))
  return useQuery({
    queryKey: ['models', filters],
    queryFn: () => api.get<ModelPage>(`/models?${qs.toString()}`),
    refetchInterval: (q) => (q.state.data?.items.some((m) => m.status === 'training') ? 3000 : 15000),
  })
}

export function useModel(id: string | undefined) {
  return useQuery({
    queryKey: ['models', 'detail', id],
    queryFn: () => api.get<ModelDetail>(`/models/${id}`),
    enabled: Boolean(id),
    refetchInterval: (q) => (q.state.data?.status === 'training' || q.state.data?.status === 'draft' ? 3000 : false),
  })
}

export function useInputSchema(id: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ['models', 'schema', id],
    queryFn: () => api.get<InputSchema>(`/models/${id}/input-schema`),
    enabled: Boolean(id) && enabled,
  })
}

export function useDeleteModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.del(`/models/${id}`),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['models'] }),
  })
}
