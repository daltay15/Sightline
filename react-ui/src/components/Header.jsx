import { Link, useLocation } from 'react-router-dom';
import './Header.css';

function Header({ title = 'Security Camera UI', controls = null, count = null, status = null }) {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <header className="sticky top-0 z-10 bg-security-bg px-[18px] py-[14px] flex gap-[10px] flex-wrap items-end">
      <h1 className="m-0 text-lg font-bold">{title}</h1>
      <nav className="flex gap-2 flex-wrap">
        <Link 
          to="/detections" 
          className={`bg-security-card text-security-text border border-security-border rounded-lg px-[10px] py-2 no-underline transition-all hover:bg-security-border ${
            isActive('/detections') || location.pathname === '/' 
              ? 'bg-security-primary border-security-primary text-white' 
              : ''
          }`}
        >
          AI Detections
        </Link>
        <Link 
          to="/events" 
          className={`bg-security-card text-security-text border border-security-border rounded-lg px-[10px] py-2 no-underline transition-all hover:bg-security-border ${
            isActive('/events') ? 'bg-security-primary border-security-primary text-white' : ''
          }`}
        >
          Events
        </Link>
        <Link 
          to="/stats" 
          className={`bg-security-card text-security-text border border-security-border rounded-lg px-[10px] py-2 no-underline transition-all hover:bg-security-border ${
            isActive('/stats') ? 'bg-security-primary border-security-primary text-white' : ''
          }`}
        >
          Stats
        </Link>
        <Link 
          to="/about" 
          className={`bg-security-card text-security-text border border-security-border rounded-lg px-[10px] py-2 no-underline transition-all hover:bg-security-border ${
            isActive('/about') ? 'bg-security-primary border-security-primary text-white' : ''
          }`}
        >
          About
        </Link>
      </nav>
      {controls && (
        <div className="flex gap-2 flex-wrap ml-auto">
          {controls}
        </div>
      )}
      {count && <span className="text-security-text text-sm whitespace-nowrap flex-shrink-0">{count}</span>}
      {status && (
        <div className={`bg-[#1a1d29] border border-security-border rounded-lg px-3 py-2 text-xs opacity-80 ${
          status.type === 'loading' ? 'opacity-100 bg-security-primary/20 border-security-primary' :
          status.type === 'error' ? 'bg-red-500/20 border-red-500 text-red-400' :
          status.type === 'success' ? 'bg-security-success/20 border-security-success text-green-400' : ''
        }`}>
          {status.message}
        </div>
      )}
    </header>
  );
}

export default Header;
