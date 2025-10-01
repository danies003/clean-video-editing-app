#!/usr/bin/env python3
"""
Local startup script that mimics Railway deployment.
Runs both FastAPI server and RQ worker in parallel.
Supports HTTPS frontend for OAuth providers that require it.
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
        "next dev",
        "caddy"
    ]
    
    for process in processes_to_kill:
        try:
            subprocess.run(["pkill", "-f", process], capture_output=True)
        except:
            pass
    
    # Kill processes on specific ports
    for port in [3000, 3001, 8000, 8443]:
        try:
            result = subprocess.run(["lsof", "-ti:" + str(port)], capture_output=True, text=True)
            if result.stdout.strip():
                subprocess.run(["kill", "-9"] + result.stdout.strip().split("\n"), capture_output=True)
        except:
            pass
    
    # Wait a moment for processes to fully terminate
    time.sleep(2)
    print("✅ Cleanup complete")

def check_ssl_certificates():
    """Check if SSL certificates exist for HTTPS frontend."""
    # Use our existing certificates in the project root
    cert_file = Path("localhost+2.pem")
    key_file = Path("localhost+2-key.pem")
    
    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)
    else:
        print("⚠️  SSL certificates not found. Frontend will run on HTTP only.")
        print("   Expected certificates: localhost+2.pem and localhost+2-key.pem")
        return None, None

def main():
    """Main startup function."""
    print("🎬 Starting AI Video Editor (Local Development)")
    print("===============================================")
    
    # Load environment variables
    load_env()
    
    # Check SSL certificates first, as use_https is needed for the override
    cert_file, key_file = check_ssl_certificates()
    use_https = cert_file is not None
    
    # Override CORS settings to allow HTTPS frontend to connect to HTTP backend
    os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000", "https://localhost:3000", "http://127.0.0.1:3000", "https://127.0.0.1:3000"]'
    
    # Override frontend API URL to use HTTPS backend when available
    if cert_file is not None:
        os.environ["NEXT_PUBLIC_API_URL"] = "https://localhost:8443"
    
    # Clean up existing processes first
    cleanup_existing_processes()
    
    print("📍 Starting FastAPI server, RQ worker, and Frontend...")
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check if frontend exists
    if not Path("frontend").exists():
        print("❌ frontend directory not found.")
        sys.exit(1)
    
    # Start FastAPI server in background process FIRST
    print("🔌 Starting FastAPI server in background...")
    # Use the virtual environment Python for FastAPI
    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable  # Fallback to system Python
    
    # Set environment variables for backend
    env = os.environ.copy()
    env["BYPASS_RENDER"] = "1"
    
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
    
    # Start HTTPS backend proxy if SSL is available
    backend_proxy_process = None
    if use_https:
        print("🔐 Starting HTTPS backend proxy...")
        backend_proxy_process = subprocess.Popen([
            "local-ssl-proxy",
            "--source", "8443",
            "--target", "8000",
            "--cert", cert_file,
            "--key", key_file
        ])
        backend_url = "https://localhost:8443"
    else:
        backend_url = "http://localhost:8000"
    
    # Start worker in background process
    print("👷 Starting RQ worker in background...")
    worker_process = subprocess.Popen([venv_python, "run_worker.py"], env=env)
    
    # Start frontend based on SSL availability
    if use_https:
        print("🔒 Starting Frontend with HTTPS support...")
        # Start frontend on port 3001 (HTTP)
        frontend_process = subprocess.Popen(["npm", "run", "dev", "--", "--port", "3001"], cwd="frontend")
        
        # Wait for frontend to start
        time.sleep(5)
        
        # Start HTTPS proxy on port 3000
        print("🔐 Starting HTTPS proxy...")
        proxy_process = subprocess.Popen([
            "local-ssl-proxy",
            "--source", "3000",
            "--target", "3001",
            "--cert", cert_file,
            "--key", key_file
        ])
        
        frontend_url = "https://localhost:3000"
    else:
        print("🌐 Starting Frontend on HTTP...")
        # Start frontend on port 3000 (HTTP)
        frontend_process = subprocess.Popen(["npm", "run", "dev"], cwd="frontend")
        proxy_process = None
        frontend_url = "http://localhost:3000"
    
    # Give background processes time to start
    time.sleep(3)
    
    print("✅ All services started!")
    print(f"📱 Frontend: {frontend_url}")
    print(f"🔌 Backend: {backend_url}")
    print(f"📚 API Docs: {backend_url}/docs")
    if use_https:
        print("🔒 HTTPS: Enabled (required for OAuth providers)")
    else:
        print("⚠️  HTTPS: Disabled (OAuth may not work)")
    print("")
    print("⌨️ Press Ctrl+C to stop all services")
    print("===============================================")
    
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
            if frontend_process.poll() is not None:
                print("❌ Frontend process died unexpectedly")
                break
            if proxy_process and proxy_process.poll() is not None:
                print("❌ HTTPS frontend proxy died unexpectedly")
                break
            if backend_proxy_process and backend_proxy_process.poll() is not None:
                print("❌ HTTPS backend proxy died unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        fastapi_process.terminate()
        worker_process.terminate()
        frontend_process.terminate()
        if proxy_process:
            proxy_process.terminate()
        if backend_proxy_process:
            backend_proxy_process.terminate()
        fastapi_process.wait()
        worker_process.wait()
        frontend_process.wait()
        if proxy_process:
            proxy_process.wait()
        if backend_proxy_process:
            backend_proxy_process.wait()
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 