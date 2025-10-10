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
    
    # Set environment variables for Railway
    env = os.environ.copy()
    env["BYPASS_RENDER"] = "1"
    
    # Start RQ worker in background first
    logger.info("👷 Starting RQ worker in background...")
    worker_process = subprocess.Popen([
        sys.executable, "run_worker.py"
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    processes.append(worker_process)
    logger.info(f"✅ RQ worker started (PID: {worker_process.pid})")
    
    # Give worker a moment to start
    time.sleep(2)
    
    # Start FastAPI server in foreground (will block)
    logger.info("=" * 60)
    logger.info("🔌 Starting FastAPI server in foreground...")
    logger.info(f"   - FastAPI: http://0.0.0.0:{port}")
    logger.info(f"   - Worker: Processing jobs (PID: {worker_process.pid})")
    logger.info("=" * 60)
    
    # Start FastAPI server - this will block and keep the container running
    # Use Popen with wait() instead of execvp to keep the worker alive
    fastapi_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", port,
        "--log-level", "info"
    ], env=env)
    processes.append(fastapi_process)
    
    # Wait for FastAPI to finish (it won't, it runs forever)
    # This keeps the main process alive and both subprocesses running
    try:
        fastapi_process.wait()
    except KeyboardInterrupt:
        cleanup_processes()

if __name__ == "__main__":
    main()

