import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from '../components/layout/Layout'
import AgentPage from '../pages/AgentPage'
import CatalogPage from '../pages/CatalogPage'
import DashboardPage from '../pages/DashboardPage'
import DatasetsPage from '../pages/DatasetsPage'
import ModelDetailPage from '../pages/ModelDetailPage'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<DashboardPage />} />
        <Route path="agent" element={<AgentPage />} />
        <Route path="agent/:sessionId" element={<AgentPage />} />
        <Route path="models" element={<CatalogPage />} />
        <Route path="models/:id" element={<ModelDetailPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="datasets/:id" element={<DatasetsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
