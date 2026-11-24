import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../../lib/apiClient'
import type { Activity, Dataset, Health, Preview, Profile } from '../../types/domain'

type DatasetPage = { items: Dataset[]; total: number; limit: number; offset: number }

export const keys = {
  datasets: ['datasets'] as const,
  dataset: (id: string) => ['datasets', id] as const,
  preview: (id: string) => ['datasets', id, 'preview'] as const,
  profile: (id: string) => ['datasets', id, 'profile'] as const,
  health: ['health'] as const,
  activity: ['activity'] as const,
}

export function useHealth() {
  return useQuery({ queryKey: keys.health, queryFn: () => api.get<Health>('/health'), refetchInterval: 30_000, retry: 0 })
}

export function useActivity(limit = 20) {
  return useQuery({ queryKey: [...keys.activity, limit], queryFn: () => api.get<Activity[]>(`/activity?limit=${limit}`), refetchInterval: 10_000 })
}

export function useDatasets() {
  return useQuery({ queryKey: keys.datasets, queryFn: () => api.get<DatasetPage>('/datasets?limit=200') })
}

export function useDataset(id: string | undefined) {
  return useQuery({ queryKey: keys.dataset(id ?? ''), queryFn: () => api.get<Dataset>(`/datasets/${id}`), enabled: Boolean(id) })
}

export function usePreview(id: string | undefined, enabled = true) {
  return useQuery({
    queryKey: keys.preview(id ?? ''),
    queryFn: () => api.get<Preview>(`/datasets/${id}/preview?limit=50`),
    enabled: Boolean(id) && enabled,
  })
}

export function useProfile(id: string | undefined, enabled = true) {
  return useQuery({ queryKey: keys.profile(id ?? ''), queryFn: () => api.get<Profile>(`/datasets/${id}/profile`), enabled: Boolean(id) && enabled })
}

export function useUploadDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => api.upload<Dataset>('/datasets', file),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.datasets })
      void qc.invalidateQueries({ queryKey: keys.activity })
    },
  })
}

export function useChooseSheet(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { sheet: string; header_row?: number | null }) => api.post<Dataset>(`/datasets/${id}/sheet`, body),
    onSuccess: (ds) => {
      qc.setQueryData(keys.dataset(id), ds)
      void qc.invalidateQueries({ queryKey: keys.datasets })
      void qc.invalidateQueries({ queryKey: keys.preview(id) })
      void qc.invalidateQueries({ queryKey: keys.profile(id) })
    },
  })
}

export function useDeleteDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.del(`/datasets/${id}`),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.datasets })
      void qc.invalidateQueries({ queryKey: keys.activity })
    },
  })
}
