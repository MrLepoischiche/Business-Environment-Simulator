#!/bin/bash

# Banking Simulator Frontend Setup Script
echo "🚀 Setting up Banking Simulator Frontend..."

# Load NVM if available (for Node.js installed via NVM)
if [ -f "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    echo "✅ NVM loaded"
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "❌ Node.js version 16+ required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Install additional Tailwind CSS plugins
echo "🎨 Installing Tailwind CSS plugins..."
npm install -D @tailwindcss/forms

# Create necessary directories
mkdir -p src/components
mkdir -p src/hooks
mkdir -p src/utils
mkdir -p src/types

echo "✅ Frontend setup completed!"
echo ""
echo "🚀 To start the development server:"
echo "   npm start"
echo ""
echo "🔗 The application will be available at:"
echo "   http://localhost:3000"
echo ""
echo "📝 Make sure the API is running on:"
echo "   http://localhost:8000"