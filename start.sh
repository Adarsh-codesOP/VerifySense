#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting backend server..."
# Navigate to backend and run the app in the background
cd backend
nohup python3 app.py > backend.log 2>&1 &

echo "🌐 Starting frontend server..."
# Navigate to frontend and start the server
cd ../frontend
npm start
