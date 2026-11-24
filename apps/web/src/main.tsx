import { createRoot } from 'react-dom/client'
import './styles/index.css'
import App from './app/App'
import { Providers } from './app/providers'
import { StoreProvider } from './legacy/store'

createRoot(document.getElementById('root')!).render(
  <Providers>
    <StoreProvider>
      <App />
    </StoreProvider>
  </Providers>,
)
