import { Link, useLocation } from 'react-router-dom';
import './Header.css';

function Header({ title = 'Security Camera UI' }) {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <header>
      <h1>{title}</h1>
      <nav className="nav">
        <Link 
          to="/detections" 
          className={isActive('/detections') || location.pathname === '/' ? 'active' : ''}
        >
          AI Detections
        </Link>
        <Link 
          to="/events" 
          className={isActive('/events') ? 'active' : ''}
        >
          Events
        </Link>
        <Link 
          to="/stats" 
          className={isActive('/stats') ? 'active' : ''}
        >
          Stats
        </Link>
        <Link 
          to="/about" 
          className={isActive('/about') ? 'active' : ''}
        >
          About
        </Link>
      </nav>
    </header>
  );
}

export default Header;
