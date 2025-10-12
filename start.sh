#!/bin/bash
set -e

echo "🚀 Starting Railway Services..."
echo "================================"

# Get port from environment
# Railway provides PORT, local defaults to 8000
PORT=${PORT:-8000}
echo "📍 Port: $PORT"

# Set environment variables
export BYPASS_RENDER=1

# Start RQ worker in background (with output redirected to main stdout/stderr)
echo "👷 Starting RQ worker in background..."
python run_worker.py 2>&1 | sed 's/^/[WORKER] /' &
WORKER_PID=$!
echo "✅ RQ worker started (PID: $WORKER_PID)"

# Give worker time to start
sleep 10

# Start FastAPI server in foreground (this will block)
echo "🔌 Starting FastAPI server on port $PORT..."
echo "================================"
exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info

