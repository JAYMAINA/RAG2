import { useState, useEffect } from 'react';
import Chat from './components/Chat';
import './App.css';

export default function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Apply theme class to body
    document.body.classList.toggle('dark-theme', isDarkMode);
  }, [isDarkMode]);

  return (
    <div className={`app ${isDarkMode ? 'dark-theme' : ''}`}>
      <header className="app-header">
        <h1>poa! gpt</h1>
        <button 
          className="theme-toggle"
          onClick={() => setIsDarkMode(!isDarkMode)}
          aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDarkMode ? '☀️' : '🌙'}
        </button>
      </header>
      <main className="app-main">
        <Chat />
      </main>
    </div>
  );
}
      