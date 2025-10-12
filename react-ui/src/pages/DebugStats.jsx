import { useState, useEffect } from 'react';
import Header from '../components/Header';
import './DebugStats.css';

function DebugStats() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [perf, setPerf] = useState(null);
  const [scanStatus, setScanStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAllStats();
  }, []);

  const loadAllStats = async () => {
    setLoading(true);
    try {
      const [healthRes, statsRes, perfRes, scanRes] = await Promise.all([
        fetch('/db-health').catch(() => null),
        fetch('/stats').catch(() => null),
        fetch('/perf').catch(() => null),
        fetch('/scan-status').catch(() => null)
      ]);

      if (healthRes?.ok) setHealth(await healthRes.json());
      if (statsRes?.ok) setStats(await statsRes.json());
      if (perfRes?.ok) setPerf(await perfRes.json());
      if (scanRes?.ok) setScanStatus(await scanRes.json());
    } catch (e) {
      console.error('Failed to load stats:', e);
    } finally {
      setLoading(false);
    }
  };

  const renderJSON = (data) => {
    return <pre className="json-display">{JSON.stringify(data, null, 2)}</pre>;
  };

  return (
    <div className="debug-stats-page">
      <Header title="Debug Stats" />
      
      <main>
        <div className="controls">
          <button className="primary" onClick={loadAllStats}>Refresh All</button>
        </div>

        {loading && <div className="loading">Loading stats...</div>}

        {!loading && (
          <>
            <div className="endpoint-section">
              <div className="endpoint-header">
                <span className="method">GET</span>
                <span className="url">/db-health</span>
              </div>
              <div className="description">Database health check</div>
              {health && renderJSON(health)}
            </div>

            <div className="endpoint-section">
              <div className="endpoint-header">
                <span className="method">GET</span>
                <span className="url">/stats</span>
              </div>
              <div className="description">Basic system statistics</div>
              {stats && renderJSON(stats)}
            </div>

            <div className="endpoint-section">
              <div className="endpoint-header">
                <span className="method">GET</span>
                <span className="url">/perf</span>
              </div>
              <div className="description">Performance metrics</div>
              {perf && renderJSON(perf)}
            </div>

            <div className="endpoint-section">
              <div className="endpoint-header">
                <span className="method">GET</span>
                <span className="url">/scan-status</span>
              </div>
              <div className="description">Live scanning status</div>
              {scanStatus && renderJSON(scanStatus)}
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default DebugStats;
