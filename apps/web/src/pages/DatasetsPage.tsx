import { Database, FileSpreadsheet, Sparkles, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Badge, Button, Card, CardHeader, Empty, Spinner, Tabs } from '../components/ui'
import { useDataset, useDatasets, useDeleteDataset, usePreview, useProfile } from '../features/datasets/hooks'
import { PreviewTable } from '../features/datasets/PreviewTable'
import { ProfileView } from '../features/datasets/ProfileView'
import { SheetPicker } from '../features/datasets/SheetPicker'
import { UploadDropzone } from '../features/datasets/UploadDropzone'
import { errorMessage } from '../lib/apiClient'
import { cx } from '../lib/cx'
import { bytes, timeAgo } from '../lib/format'
import type { Dataset } from '../types/domain'

function DatasetDetail({ id }: { id: string }) {
  const navigate = useNavigate()
  const [tab, setTab] = useState('preview')
  const ds = useDataset(id)
  const ready = ds.data?.status === 'ready'
  const preview = usePreview(id, ready && tab === 'preview')
  const profile = useProfile(id, ready && tab === 'profile')
  const del = useDeleteDataset()

  if (ds.isLoading) return <Spinner />
  if (ds.isError || !ds.data) return <Empty icon={Database} title="Dataset not found" text={errorMessage(ds.error)} />
  const d = ds.data

  return (
    <Card>
      <CardHeader
        icon={FileSpreadsheet}
        title={d.file_name}
        subtitle={
          ready
            ? `${d.sheet} · ${d.rows?.toLocaleString()} rows × ${d.cols} columns · ${bytes(d.size_bytes)} · uploaded ${timeAgo(d.created_at)}`
            : 'Choose a sheet'
        }
        right={
          <div className="flex gap-2">
            {ready && (
              <Button variant="primary" size="sm" icon={Sparkles} onClick={() => navigate(`/agent?dataset=${d.id}`)}>
                Build a model
              </Button>
            )}
            <Button
              variant="danger"
              size="sm"
              icon={Trash2}
              aria-label="Delete dataset"
              disabled={del.isPending}
              onClick={() => {
                if (window.confirm(`Delete ${d.file_name}? Models trained on it keep working.`))
                  del.mutate(d.id, { onSuccess: () => navigate('/datasets') })
              }}
            >
              Delete
            </Button>
          </div>
        }
      />
      <div className="px-5 pb-5">
        {!ready ? (
          <SheetPicker dataset={d} />
        ) : (
          <>
            <Tabs
              tabs={[
                { id: 'preview', label: 'Preview' },
                { id: 'profile', label: 'Profile' },
              ]}
              value={tab}
              onChange={setTab}
              className="mb-4"
            />
            {tab === 'preview' && (preview.data ? <PreviewTable preview={preview.data} /> : <Spinner label="Loading preview" />)}
            {tab === 'profile' && (profile.data ? <ProfileView profile={profile.data} /> : <Spinner label="Loading profile" />)}
          </>
        )}
      </div>
    </Card>
  )
}

function DatasetRow({ d, active, onClick }: { d: Dataset; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cx(
        'flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors focus-visible:outline-2 focus-visible:outline-emerald-500',
        active ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200 bg-white hover:border-slate-300',
      )}
    >
      <FileSpreadsheet size={18} className="shrink-0 text-emerald-600" aria-hidden />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px] font-semibold text-slate-900">{d.file_name}</span>
        <span className="block truncate text-xs text-slate-500">
          {d.status === 'ready' ? `${d.sheet} · ${d.rows?.toLocaleString()} rows` : `${d.sheets.length} sheets · choose one`} ·{' '}
          {timeAgo(d.created_at)}
        </span>
      </span>
      {d.status !== 'ready' && <Badge color="amber">Pick sheet</Badge>}
      {(d.used_by ?? []).length > 0 && <Badge color="indigo">{(d.used_by ?? []).length} model</Badge>}
    </button>
  )
}

export default function DatasetsPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const list = useDatasets()
  const items = list.data?.items ?? []
  const selected = id ?? items[0]?.id

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-[26px] font-semibold tracking-tight text-slate-900">Datasets</h1>
        <p className="mt-1 text-sm text-slate-600">Excel and CSV files you uploaded. Preview them, check data quality, or build a model from one.</p>
      </div>
      <div className="grid gap-5 xl:grid-cols-[340px_minmax(0,1fr)]">
        <div className="space-y-3">
          <UploadDropzone compact onUploaded={(ds) => navigate(`/datasets/${ds.id}`)} />
          {list.isLoading && <Spinner />}
          {list.isError && (
            <p role="alert" className="text-[13px] text-red-700">
              {errorMessage(list.error)}
            </p>
          )}
          {items.map((d) => (
            <DatasetRow key={d.id} d={d} active={d.id === selected} onClick={() => navigate(`/datasets/${d.id}`)} />
          ))}
        </div>
        <div className="min-w-0">
          {selected ? (
            <DatasetDetail key={selected} id={selected} />
          ) : (
            !list.isLoading && (
              <Card>
                <Empty
                  icon={Database}
                  title="No datasets yet"
                  text="Upload an Excel or CSV file to get started. Sample files are in data/samples/ in the repository."
                />
              </Card>
            )
          )}
        </div>
      </div>
    </div>
  )
}
