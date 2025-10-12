import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Detections from './pages/Detections';
import Events from './pages/Events';
import Stats from './pages/Stats';
import About from './pages/About';
import Config from './pages/Config';
import DebugStats from './pages/DebugStats';
import BackupStatus from './pages/BackupStatus';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/detections" replace />} />
        <Route path="/detections" element={<Detections />} />
        <Route path="/events" element={<Events />} />
        <Route path="/stats" element={<Stats />} />
        <Route path="/about" element={<About />} />
        <Route path="/config" element={<Config />} />
        <Route path="/debug-stats" element={<DebugStats />} />
        <Route path="/backup-status" element={<BackupStatus />} />
      </Routes>
    </Router>
  );
}

export default App;
