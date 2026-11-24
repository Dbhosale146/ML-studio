import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import Layout from '../components/layout/Layout'
import Dashboard from '../legacy/DashboardPage'
import Agent from '../legacy/AgentPage'
import Catalog from '../legacy/CatalogPage'
import ModelDetail from '../legacy/ModelDetailPage'
import DatasetsPage from '../pages/DatasetsPage'

function AgentRoute() {
  // Re-mount the agent for every new session (e.g. a second "Create new model" click).
  const { key } = useLocation()
  return <Agent key={key} />
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="agent" element={<AgentRoute />} />
        <Route path="models" element={<Catalog />} />
        <Route path="models/:id" element={<ModelDetail />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="datasets/:id" element={<DatasetsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
