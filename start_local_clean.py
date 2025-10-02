#!/usr/bin/env python3
"""
Local startup script for the new clean Video Editing App.
Runs both FastAPI server and RQ worker in parallel.
"""

import os
import signal
import sys
import time
import subprocess
from pathlib import Path

# Load environment variables
def load_env():
    """Load environment variables from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

def cleanup_existing_processes():
    """Kill any existing processes that might conflict."""
    print("🧹 Cleaning up existing processes...")
    
    # Kill existing processes
    processes_to_kill = [
        "uvicorn main:app",
        "python run_worker.py", 
        "npm run dev",
        "next dev"
    ]
    
    for process in processes_to_kill:
        try:
            subprocess.run(["pkill", "-f", process], capture_output=True)
        except:
            pass
    
    # Kill processes on specific ports
    for port in [3000, 3001, 8000]:
        try:
            result = subprocess.run(["lsof", "-ti:" + str(port)], capture_output=True, text=True)
            if result.stdout.strip():
                subprocess.run(["kill", "-9"] + result.stdout.strip().split("\n"), capture_output=True)
        except:
            pass
    
    # Wait a moment for processes to fully terminate
    time.sleep(2)
    print("✅ Cleanup complete")

def main():
    """Main startup function."""
    print("🎬 Starting New Clean Video Editing App")
    print("======================================")
    
    # Load environment variables
    load_env()
    
    # Clean up existing processes first
    cleanup_existing_processes()
    
    print("📍 Starting FastAPI server and RQ worker...")
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Use the virtual environment Python
    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found. Please run 'python3 -m venv venv' first.")
        sys.exit(1)
    
    # Set environment variables for backend
    env = os.environ.copy()
    env["BYPASS_RENDER"] = "1"
    
    # Start FastAPI server in background process
    print("🔌 Starting FastAPI server in background...")
    fastapi_process = subprocess.Popen([
        venv_python, "-m", "uvicorn", 
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info",
        "--reload"
    ], env=env)
    
    # Give FastAPI time to start
    time.sleep(5)
    
    # Start worker in background process
    print("👷 Starting RQ worker in background...")
    worker_process = subprocess.Popen([venv_python, "run_worker.py"], env=env)
    
    # Give background processes time to start
    time.sleep(3)
    
    print("✅ All services started!")
    print("📱 Backend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("")
    print("⌨️ Press Ctrl+C to stop all services")
    print("======================================")
    
    # Keep main process alive and handle shutdown
    try:
        while True:
            time.sleep(1)
            # Check if any process has died
            if fastapi_process.poll() is not None:
                print("❌ FastAPI server died unexpectedly")
                break
            if worker_process.poll() is not None:
                print("❌ Worker process died unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        # Terminate processes
        fastapi_process.terminate()
        worker_process.terminate()
        
        # Wait for graceful shutdown
        try:
            fastapi_process.wait(timeout=5)
            worker_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Forcing shutdown...")
            fastapi_process.kill()
            worker_process.kill()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
