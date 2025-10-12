import Header from '../components/Header';
import './BackupStatus.css';

function BackupStatus() {
  return (
    <div className="backup-status-page">
      <Header title="Backup Status" />
      
      <main>
        <div className="info-card">
          <h2>💾 Backup Status</h2>
          <p>
            Backup status monitoring is available in the original HTML interface.
            This React version focuses on viewing and monitoring functionality.
          </p>
          <p>
            To access full backup management features, please use the backup endpoint directly
            or access the HTML version at <code>/ui/backup-status.html</code>
          </p>
        </div>
      </main>
    </div>
  );
}

export default BackupStatus;
