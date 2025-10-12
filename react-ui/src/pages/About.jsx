import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import './About.css';

function About() {
  const navigate = useNavigate();

  return (
    <div className="about-page">
      <Header title="About - Security Camera UI" />
      
      <main>
        <div className="info-card">
          <h2>🎥 Security Camera UI</h2>
          <p>
            A modern, high-performance web interface for managing and viewing security camera footage.
            Built for speed, scalability, and ease of use.
          </p>
        </div>

        <div className="info-card">
          <h2>✨ Features</h2>
          <ul>
            <li><strong>Real-time Monitoring:</strong> Live updates when new events are detected</li>
            <li><strong>AI Detection:</strong> Automatic object detection (people, cars, animals)</li>
            <li><strong>Smart Filtering:</strong> Filter by camera, date range, and detection type</li>
            <li><strong>Video Streaming:</strong> Smooth video playback with seek support</li>
            <li><strong>Thumbnail Generation:</strong> Automatic thumbnails for quick browsing</li>
            <li><strong>Analytics:</strong> Motion detection statistics and trends</li>
            <li><strong>Mobile Responsive:</strong> Works great on phones and tablets</li>
          </ul>
        </div>

        <div className="info-card">
          <h2>🛠️ Technology Stack</h2>
          <ul>
            <li><strong>Frontend:</strong> React with Vite</li>
            <li><strong>Backend:</strong> Go with Gin web framework</li>
            <li><strong>Database:</strong> SQLite for event storage and indexing</li>
            <li><strong>Streaming:</strong> HTTP range requests for efficient video streaming</li>
            <li><strong>Thumbnails:</strong> Automatic generation and caching system</li>
          </ul>
          
          <h3>Performance Features</h3>
          <ul>
            <li><strong>Lazy Loading:</strong> Images and videos load on demand</li>
            <li><strong>Compression:</strong> Gzip compression for faster data transfer</li>
            <li><strong>Caching:</strong> Aggressive caching for thumbnails and static assets</li>
            <li><strong>Pagination:</strong> Efficient pagination for large event collections</li>
            <li><strong>Background Processing:</strong> Non-blocking thumbnail generation</li>
          </ul>
        </div>

        <div className="info-card">
          <h2>🔌 API Endpoints</h2>
          
          <h3>Core Endpoints</h3>
          <ul>
            <li><code>GET /events</code> - List security camera events with filtering</li>
            <li><code>GET /stats/motion</code> - Motion detection statistics and analytics</li>
            <li><code>GET /thumb/:id</code> - Serve thumbnail images</li>
            <li><code>GET /stream/:id</code> - Stream video files with range support</li>
            <li><code>POST /events/:id/reviewed</code> - Mark events as reviewed</li>
          </ul>
          
          <h3>System Endpoints</h3>
          <ul>
            <li><code>GET /scan-status</code> - Current scanning and indexing status</li>
            <li><code>GET /health</code> - System health check</li>
            <li><code>GET /stats</code> - Basic system statistics</li>
            <li><code>GET /perf</code> - Performance metrics</li>
          </ul>
        </div>

        <div className="action-buttons">
          <button className="primary" onClick={() => navigate('/debug-stats')}>
            📊 Debug Stats
          </button>
          <button className="primary" onClick={() => navigate('/config')}>
            ⚙️ Configuration
          </button>
          <button className="primary" onClick={() => navigate('/backup-status')}>
            💾 Backup Status
          </button>
        </div>
      </main>
    </div>
  );
}

export default About;
