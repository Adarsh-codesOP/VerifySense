import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';

function Header() {
  return (
    <header className="bg-white shadow-sm">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="flex items-center space-x-2">
          <ShieldCheckIcon className="h-8 w-8 text-primary-600" />
          <span className="text-xl font-bold text-gray-900">VerifySense</span>
        </Link>
        <nav className="flex space-x-6">
          <Link to="/" className="text-gray-700 hover:text-primary-600 font-medium">
            Home
          </Link>
          <Link to="/about" className="text-gray-700 hover:text-primary-600 font-medium">
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}

export default Header;