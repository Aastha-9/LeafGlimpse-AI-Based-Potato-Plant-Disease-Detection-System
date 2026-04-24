import React from 'react';
import { Sprout } from 'lucide-react';

const Navbar = ({ language, setLanguage }) => {
  return (
    <nav className="navbar">
      <div className="nav-brand">
        <Sprout size={32} color="var(--primary-dark)" />
        LeafGlimpse
      </div>
      <div className="nav-controls">
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="en">English (US)</option>
          <option value="hi">Hindi (हिन्दी)</option>
          <option value="mr">Marathi (मराठी)</option>
        </select>
      </div>
    </nav>
  );
};

export default Navbar;
