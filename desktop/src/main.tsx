import { createRoot } from 'react-dom/client';
import { AppErrorBoundary } from './App';
import App from './Studio';

createRoot(document.getElementById('root')!).render(<AppErrorBoundary><App /></AppErrorBoundary>);
