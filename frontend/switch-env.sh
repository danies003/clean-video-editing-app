#!/bin/bash

# Environment Switcher Script
# Usage: ./switch-env.sh [development|production]

ENV=${1:-development}

case $ENV in
  "development"|"dev")
    echo "🔄 Switching to DEVELOPMENT environment..."
    cp .env.development .env.local
    echo "✅ Environment set to DEVELOPMENT"
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: http://localhost:3000"
    ;;
  "production"|"prod")
    echo "🔄 Switching to PRODUCTION environment..."
    cp .env.production .env.local
    echo "✅ Environment set to PRODUCTION"
    echo "   Backend: https://organic-swim-production.up.railway.app"
    echo "   Frontend: http://localhost:3000 (still local for testing)"
    ;;
  *)
    echo "❌ Invalid environment. Use 'development' or 'production'"
    echo "Usage: ./switch-env.sh [development|production]"
    exit 1
    ;;
esac

echo ""
echo "🔄 Restart your frontend server to apply changes:"
echo "   npm run dev" 