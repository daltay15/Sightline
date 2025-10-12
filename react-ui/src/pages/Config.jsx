import Header from '../components/Header';
import './Config.css';

function Config() {
  return (
    <div className="config-page">
      <Header title="Configuration" />
      
      <main>
        <div className="info-card">
          <h2>⚙️ Configuration</h2>
          <p>
            Configuration management is available in the original HTML interface.
            This React version focuses on viewing and monitoring functionality.
          </p>
          <p>
            To access full configuration features, please use the configuration endpoint directly
            or access the HTML version at <code>/ui/config.html</code>
          </p>
        </div>
      </main>
    </div>
  );
}

export default Config;
