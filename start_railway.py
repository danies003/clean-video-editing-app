#!/usr/bin/env python3
"""
Railway Startup Script - Runs both FastAPI backend and RQ worker
This script starts both the API server and background worker in Railway.
"""

import os
import sys
import time
import signal
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("railway_startup")

# Global process tracking
processes = []

def cleanup_processes(signum=None, frame=None):
    """Clean up all child processes."""
    logger.info("🛑 Shutting down all services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
            try:
                proc.kill()
            except:
                pass
    logger.info("✅ All services stopped")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup_processes)
signal.signal(signal.SIGTERM, cleanup_processes)

def main():
    """Start both FastAPI and worker."""
    logger.info("=" * 60)
    logger.info("🚀 Railway Startup - Starting Backend + Worker")
    logger.info("=" * 60)
    
    # Get port from environment (Railway provides this)
    port = os.environ.get("PORT", "8080")
    logger.info(f"📍 Port: {port}")
    
    # Start FastAPI server in background
    logger.info("🔌 Starting FastAPI server...")
    fastapi_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "main_railway:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--log-level", "info"
    ])
    processes.append(fastapi_process)
    logger.info(f"✅ FastAPI server started (PID: {fastapi_process.pid})")
    
    # Give FastAPI time to start
    time.sleep(5)
    
    # Start RQ worker in background
    logger.info("👷 Starting RQ worker...")
    worker_process = subprocess.Popen([
        sys.executable, "worker_railway.py"
    ])
    processes.append(worker_process)
    logger.info(f"✅ RQ worker started (PID: {worker_process.pid})")
    
    logger.info("=" * 60)
    logger.info("🎉 All services running!")
    logger.info(f"   - FastAPI: http://0.0.0.0:{port}")
    logger.info(f"   - Worker: Processing jobs from Redis")
    logger.info("=" * 60)
    
    # Keep the main process alive and monitor child processes
    try:
        while True:
            # Check if processes are still running
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    logger.error(f"❌ Process {i} (PID: {proc.pid}) exited with code {proc.returncode}")
                    cleanup_processes()
            time.sleep(5)
    except KeyboardInterrupt:
        cleanup_processes()

if __name__ == "__main__":
    main()

