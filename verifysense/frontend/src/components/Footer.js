import React from 'react';

function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-6">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">
              &copy; {new Date().getFullYear()} VerifySense - Google Gen AI Hackathon Project
            </p>
          </div>
          <div className="flex space-x-4">
            <p className="text-sm">Powered by Google AI Services</p>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;