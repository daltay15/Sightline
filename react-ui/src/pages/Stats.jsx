import { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import Header from '../components/Header';
import './Stats.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function Stats() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('24h');

  useEffect(() => {
    loadStats();
  }, [timeRange]);

  const loadStats = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/stats/motion?range=${timeRange}`);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error('Failed to load stats:', e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="stats-page">
        <Header title="Motion Detection Stats" />
        <main>
          <div className="loading">Loading statistics...</div>
        </main>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="stats-page">
        <Header title="Motion Detection Stats" />
        <main>
          <div className="error">Failed to load statistics.</div>
        </main>
      </div>
    );
  }

  // Prepare chart data
  const timelineData = {
    labels: stats.timeline?.map(d => d.hour || d.date) || [],
    datasets: [{
      label: 'Events',
      data: stats.timeline?.map(d => d.count) || [],
      borderColor: '#375dfb',
      backgroundColor: 'rgba(55, 93, 251, 0.1)',
      tension: 0.4
    }]
  };

  const cameraData = {
    labels: stats.by_camera?.map(d => d.camera) || [],
    datasets: [{
      label: 'Events by Camera',
      data: stats.by_camera?.map(d => d.count) || [],
      backgroundColor: '#375dfb'
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#e8e8e8'
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#e8e8e8' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        ticks: { color: '#e8e8e8' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      }
    }
  };

  return (
    <div className="stats-page">
      <Header title="Motion Detection Stats" />
      
      <main>
        <div className="controls">
          <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)}>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <button className="primary" onClick={loadStats}>Refresh</button>
        </div>

        <div className="summary-stats">
          <div className="summary-item">
            <div className="value">{stats.total_events || 0}</div>
            <div className="label">Total Events</div>
          </div>
          <div className="summary-item">
            <div className="value">{stats.unique_cameras || 0}</div>
            <div className="label">Active Cameras</div>
          </div>
          <div className="summary-item">
            <div className="value">{stats.avg_events_per_day?.toFixed(1) || 0}</div>
            <div className="label">Avg Events/Day</div>
          </div>
          <div className="summary-item">
            <div className="value">{stats.peak_hour || 'N/A'}</div>
            <div className="label">Peak Hour</div>
          </div>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <h3>Events Timeline</h3>
            <div className="chart-container">
              <Line data={timelineData} options={chartOptions} />
            </div>
          </div>

          <div className="stat-card">
            <h3>Events by Camera</h3>
            <div className="chart-container">
              <Bar data={cameraData} options={chartOptions} />
            </div>
          </div>
        </div>

        {stats.top_cameras && stats.top_cameras.length > 0 && (
          <div className="stat-card">
            <h3>Top Cameras</h3>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Camera</th>
                    <th>Events</th>
                    <th>Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.top_cameras.map((cam, idx) => (
                    <tr key={idx}>
                      <td>{cam.camera}</td>
                      <td>{cam.count}</td>
                      <td>{cam.percentage?.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default Stats;
