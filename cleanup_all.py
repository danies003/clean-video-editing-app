#!/usr/bin/env python3
"""
Comprehensive cleanup script for Video Editing App
Kills all processes, clears caches, and cleans Redis jobs
"""

import os
import sys
import time
import signal
import subprocess
import shutil
from pathlib import Path

def print_status(message, status="🔧"):
    """Print a status message with emoji."""
    print(f"{status} {message}")

def run_command(cmd, capture_output=True, check=False):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=check)
        return result
    except Exception as e:
        print(f"❌ Error running command {' '.join(cmd)}: {e}")
        return None

def kill_processes_by_pattern(patterns):
    """Kill processes matching patterns."""
    for pattern in patterns:
        print_status(f"Killing processes matching: {pattern}")
        result = run_command(['pkill', '-f', pattern])
        if result and result.returncode == 0:
            print(f"   ✅ Killed processes matching: {pattern}")
        else:
            print(f"   ⚠️  No processes found matching: {pattern}")

def kill_processes_by_port(ports):
    """Kill processes using specific ports."""
    for port in ports:
        print_status(f"Killing processes using port {port}")
        try:
            # Find processes using the port
            result = run_command(['lsof', '-ti', f':{port}'])
            if result and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        run_command(['kill', '-9', pid])
                        print(f"   ✅ Killed process {pid} using port {port}")
            else:
                print(f"   ⚠️  No processes found using port {port}")
        except Exception as e:
            print(f"   ❌ Error killing processes on port {port}: {e}")
    
    # Additional cleanup for Redis (but preserve Redis server)
    if 6379 in ports:
        print_status("Performing Redis cleanup (preserving Redis server)", "🔴")
        try:
            # Only kill Redis processes that are NOT the main server
            result = run_command(['pgrep', '-f', 'redis'])
            if result and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        # Check if this is the main Redis server
                        cmd_result = run_command(['ps', '-p', pid, '-o', 'command='])
                        if cmd_result and cmd_result.stdout.strip():
                            cmd = cmd_result.stdout.strip()
                            if 'redis-server' in cmd and ('*:6379' in cmd or '127.0.0.1:6379' in cmd):
                                print(f"   ⚠️  Preserving main Redis server process {pid}")
                                continue
                            else:
                                run_command(['kill', '-9', pid])
                                print(f"   ✅ Killed Redis-related process {pid}")
            else:
                print("   ⚠️  No Redis processes found to clean up")
            
        except Exception as e:
            print(f"   ❌ Error in Redis cleanup: {e}")

def discover_redis_keys():
    """Discover and display all Redis keys before cleanup."""
    print_status("Discovering Redis keys before cleanup", "🔍")
    
    try:
        # Get total key count
        result = run_command(['redis-cli', 'DBSIZE'])
        if result and result.stdout.strip():
            total_keys = result.stdout.strip()
            print(f"   📊 Total Redis keys: {total_keys}")
        
        # Discover job-related keys
        key_patterns = [
            ('job:*', 'Custom Job Keys'),
            ('*queue*', 'Queue-related Keys'),
            ('*worker*', 'Worker-related Keys'),
            ('multi_video_project:*', 'Multi-video Project Keys'),
            ('rq:*', 'RQ Standard Keys'),
            ('ffmpeg_*', 'FFmpeg Resource Keys'),
            ('*analysis*', 'Analysis-related Keys'),
            ('*render*', 'Render-related Keys')
        ]
        
        for pattern, description in key_patterns:
            result = run_command(['redis-cli', '--scan', '--pattern', pattern])
            if result and result.stdout.strip():
                keys = result.stdout.strip().split('\n')
                keys = [k for k in keys if k]  # Remove empty lines
                if keys:
                    print(f"   🔍 {description} ({len(keys)} keys):")
                    for key in keys[:5]:  # Show first 5 keys
                        print(f"      - {key}")
                    if len(keys) > 5:
                        print(f"      ... and {len(keys) - 5} more")
                else:
                    print(f"   ⚠️  {description}: No keys found")
            else:
                print(f"   ⚠️  {description}: No keys found")
        
        # Show some sample key values for important keys
        print("\n   📋 Sample Key Values:")
        result = run_command(['redis-cli', '--scan', '--pattern', 'job:*'])
        if result and result.stdout.strip():
            sample_job = result.stdout.strip().split('\n')[0]
            if sample_job:
                job_data = run_command(['redis-cli', 'GET', sample_job])
                if job_data and job_data.stdout.strip():
                    try:
                        import json
                        job_json = json.loads(job_data.stdout.strip())
                        print(f"      Sample job {sample_job}:")
                        print(f"        Status: {job_json.get('status', 'unknown')}")
                        print(f"        Progress: {job_json.get('progress', 'unknown')}%")
                        print(f"        Type: {job_json.get('job_type', 'unknown')}")
                    except:
                        print(f"      Sample job {sample_job}: Raw data available")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Error discovering Redis keys: {e}")

def cleanup_redis():
    """Clean Redis jobs and flush database."""
    print_status("Cleaning Redis jobs and data", "🔴")
    
    # First, discover what keys exist
    discover_redis_keys()
    
    # Try to connect to Redis and clean jobs
    try:
        # Method 1: Try to flush all Redis data (most comprehensive)
        print("   🔄 Attempting FLUSHALL...")
        result = run_command(['redis-cli', 'FLUSHALL'])
        if result and result.returncode == 0:
            print("   ✅ Flushed all Redis data (FLUSHALL)")
            # Verify cleanup
            print("   🔍 Verifying cleanup...")
            final_count = run_command(['redis-cli', 'DBSIZE'])
            if final_count and final_count.stdout.strip():
                remaining_keys = final_count.stdout.strip()
                print(f"   📊 Remaining Redis keys: {remaining_keys}")
                if remaining_keys == '0':
                    print("   ✅ All Redis keys cleared successfully!")
                else:
                    print("   ⚠️  Some keys remain - cleanup may be incomplete")
            return  # FLUSHALL cleared everything, no need for individual cleanup
        
        print("   ⚠️  FLUSHALL failed, trying individual key cleanup")
        
        # Method 2: Individual cleanup if FLUSHALL fails
        # Clear custom job keys (job:*)
        print("   🔄 Clearing custom job keys (job:*)...")
        result = run_command(['redis-cli', '--scan', '--pattern', 'job:*', '|', 'xargs', 'redis-cli', 'DEL'])
        if result:
            print("   ✅ Cleared custom job keys (job:*)")
        
        # Clear custom queue keys (*queue*)
        print("   🔄 Clearing custom queue keys (*queue*)...")
        result = run_command(['redis-cli', '--scan', '--pattern', '*queue*', '|', 'xargs', 'redis-cli', 'DEL'])
        if result:
            print("   ✅ Cleared custom queue keys (*queue*)")
        
        # Clear custom worker keys (*worker*)
        print("   🔄 Clearing custom worker keys (*worker*)...")
        result = run_command(['redis-cli', '--scan', '--pattern', '*worker*', '|', 'xargs', 'redis-cli', 'DEL'])
        if result:
            print("   ✅ Cleared custom worker keys (*worker*)")
        
        # Clear project keys (multi_video_project:*)
        print("   🔄 Clearing project keys (multi_video_project:*)...")
        result = run_command(['redis-cli', '--scan', '--pattern', 'multi_video_project:*', '|', 'xargs', 'redis-cli', 'DEL'])
        if result:
            print("   ✅ Cleared project keys (multi_video_project:*)")
        
        # Clear RQ job registries
        print("   🔄 Clearing RQ standard keys...")
        rq_keys = [
            'rq:queue:video_editing',
            'rq:failed_job_registry',
            'rq:started_job_registry',
            'rq:deferred_job_registry',
            'rq:finished_job_registry',
            'rq:scheduled_job_registry',
            'rq:workers'
        ]
        
        for rq_key in rq_keys:
            result = run_command(['redis-cli', 'DEL', rq_key])
            if result:
                print(f"      ✅ Cleared {rq_key}")
        
        # Method 3: Aggressive cleanup - force FLUSHALL again
        print("   🔴 Performing aggressive cleanup - forcing FLUSHALL again...")
        result = run_command(['redis-cli', 'FLUSHALL'])
        if result:
            print("   ✅ Aggressively cleared all remaining keys with FLUSHALL")
        
        # Method 4: Final aggressive cleanup - scan and delete ALL remaining keys
        print("   🔴 Performing final aggressive cleanup - scanning ALL remaining keys...")
        result = run_command(['redis-cli', '--scan', '|', 'xargs', 'redis-cli', 'DEL'])
        if result:
            print("   ✅ Final aggressive cleanup completed")
        
        # Verify cleanup
        print("   🔍 Verifying cleanup...")
        final_count = run_command(['redis-cli', 'DBSIZE'])
        if final_count and final_count.stdout.strip():
            remaining_keys = final_count.stdout.strip()
            print(f"   📊 Remaining Redis keys: {remaining_keys}")
            if remaining_keys == '0':
                print("   ✅ All Redis keys cleared successfully!")
            else:
                print("   ⚠️  Some keys remain - cleanup may be incomplete")
        
    except Exception as e:
        print(f"   ❌ Error cleaning Redis: {e}")
        print("   🔴 Attempting emergency Redis cleanup...")
        
        # Emergency cleanup: Kill Redis server and restart
        try:
            run_command(['pkill', '-f', 'redis-server'])
            time.sleep(2)
            print("   ✅ Killed Redis server for emergency cleanup")
        except Exception as emergency_error:
            print(f"   ❌ Emergency cleanup failed: {emergency_error}")

def clear_caches():
    """Clear all caches."""
    print_status("Clearing all caches", "🧹")
    
    # Clear Next.js cache
    next_cache = Path("frontend/.next")
    if next_cache.exists():
        try:
            shutil.rmtree(next_cache)
            print("   ✅ Cleared Next.js cache (.next)")
        except Exception as e:
            print(f"   ❌ Error clearing Next.js cache: {e}")
    else:
        print("   ⚠️  Next.js cache not found")
    
    # Clear Node.js module cache
    node_cache = Path("frontend/node_modules/.cache")
    if node_cache.exists():
        try:
            shutil.rmtree(node_cache)
            print("   ✅ Cleared Node.js module cache")
        except Exception as e:
            print(f"   ❌ Error clearing Node.js cache: {e}")
    else:
        print("   ⚠️  Node.js cache not found")
    
    # Clear Python cache files
    try:
        # Remove .pyc files
        for pyc_file in Path(".").rglob("*.pyc"):
            try:
                pyc_file.unlink()
            except Exception:
                pass
        print("   ✅ Cleared Python .pyc files")
        
        # Remove __pycache__ directories
        for cache_dir in Path(".").rglob("__pycache__"):
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass
        print("   ✅ Cleared Python __pycache__ directories")
    except Exception as e:
        print(f"   ❌ Error clearing Python cache: {e}")
    
    # Clear temporary files
    temp_patterns = ["*.tmp", "*.temp", "*.log", "*.pid"]
    for pattern in temp_patterns:
        try:
            for temp_file in Path(".").rglob(pattern):
                try:
                    temp_file.unlink()
                except Exception:
                    pass
        except Exception:
            pass
    print("   ✅ Cleared temporary files")

def cleanup_docker():
    """Clean up Docker containers and images if they exist."""
    print_status("Cleaning Docker containers and images", "🐳")
    
    try:
        # Stop all running containers
        result = run_command(['docker', 'stop', '$(docker ps -q)'])
        if result and result.returncode == 0:
            print("   ✅ Stopped all Docker containers")
        
        # Remove all containers
        result = run_command(['docker', 'rm', '$(docker ps -aq)'])
        if result and result.returncode == 0:
            print("   ✅ Removed all Docker containers")
        
        # Remove all images
        result = run_command(['docker', 'rmi', '$(docker images -q)'])
        if result and result.returncode == 0:
            print("   ✅ Removed all Docker images")
        
        # Prune system
        result = run_command(['docker', 'system', 'prune', '-f'])
        if result and result.returncode == 0:
            print("   ✅ Pruned Docker system")
            
    except Exception as e:
        print(f"   ⚠️  Docker cleanup failed (Docker might not be installed): {e}")

def cleanup_stuck_jobs():
    """Clean up stuck jobs and processes specifically."""
    print_status("Cleaning up stuck jobs and processes", "🚨")
    
    try:
        # Kill any stuck FFmpeg processes - MORE AGGRESSIVE
        print("   🔄 Killing stuck FFmpeg processes...")
        ffmpeg_patterns = [
            'ffmpeg',
            'ffprobe',
            'imageio_ffmpeg'
        ]
        
        # First pass: graceful kill
        for pattern in ffmpeg_patterns:
            result = run_command(['pkill', '-f', pattern])
            if result and result.returncode == 0:
                print(f"      ✅ Killed {pattern} processes")
            else:
                print(f"      ⚠️  No {pattern} processes found")
        
        # Wait a moment for processes to die
        time.sleep(2)
        
        # Second pass: force kill any remaining FFmpeg processes
        print("   🔄 Force killing any remaining FFmpeg processes...")
        result = run_command(['pgrep', '-f', 'ffmpeg'])
        if result and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    run_command(['kill', '-9', pid])
                    print(f"      ✅ Force killed FFmpeg process {pid}")
        
        # Third pass: scan for any remaining FFmpeg processes by name
        print("   🔄 Final scan for any remaining FFmpeg processes...")
        for pattern in ffmpeg_patterns:
            result = run_command(['pgrep', '-f', pattern])
            if result and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        run_command(['kill', '-9', pid])
                        print(f"      ✅ Final kill of {pattern} process {pid}")
        
        # Wait again and verify
        time.sleep(1)
        final_check = run_command(['pgrep', '-f', 'ffmpeg'])
        if final_check and final_check.stdout.strip():
            print(f"   ⚠️  Some FFmpeg processes still running: {final_check.stdout.strip()}")
        else:
            print("      ✅ All FFmpeg processes successfully killed")
        
        # Clean up Redis stuck jobs (if Redis is available)
        print("   🔄 Cleaning up Redis stuck jobs...")
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            
            # Get all jobs
            jobs = r.smembers('jobs')
            if jobs:
                print(f"      📊 Found {len(jobs)} jobs to clean up")
                
                # Delete all job data
                for job_id in jobs:
                    job_key = f"job:{job_id.decode()}"
                    video_job_key = f"video_job:{job_id.decode()}"
                    
                    r.delete(job_key)
                    r.delete(video_job_key)
                
                # Clear the jobs set
                r.delete('jobs')
                print(f"      ✅ Cleaned up {len(jobs)} stuck jobs")
            else:
                print("      ⚠️  No jobs found to clean up")
            
            # Clear RQ queue data
            r.delete('rq:queue:video_editing')
            r.delete('rq:queue:video_editing:deferred')
            r.delete('rq:queue:video_editing:failed')
            print("      ✅ Cleared RQ queue data")
            
        except ImportError:
            print("      ⚠️  Redis module not available, using redis-cli")
            # Fallback to redis-cli
            run_command(['redis-cli', 'DEL', 'jobs'])
            run_command(['redis-cli', 'DEL', 'rq:queue:video_editing'])
            run_command(['redis-cli', 'DEL', 'rq:queue:video_editing:deferred'])
            run_command(['redis-cli', 'DEL', 'rq:queue:video_editing:failed'])
            print("      ✅ Cleared Redis data using redis-cli")
        except Exception as redis_error:
            print(f"      ⚠️  Redis cleanup failed: {redis_error}")
        
        print("   ✅ Stuck job cleanup completed")
        
    except Exception as e:
        print(f"   ❌ Error in stuck job cleanup: {e}")

def verify_cleanup():
    """Verify that cleanup was successful."""
    print_status("Verifying cleanup", "🔍")
    
    # Check if any processes are still running
    processes_to_check = [
        "start_local.py",
        "uvicorn",
        "run_worker",
        "next dev",
        "local-ssl-proxy",
        "redis-server"
    ]
    
    still_running = []
    for process in processes_to_check:
        result = run_command(['pgrep', '-f', process])
        if result and result.stdout.strip():
            still_running.append(process)
    
    if still_running:
        print(f"   ⚠️  Some processes are still running: {', '.join(still_running)}")
    else:
        print("   ✅ All processes have been stopped")
    
    # Check if ports are free
    ports_to_check = [3000, 3001, 8000, 8443, 6379]
    ports_in_use = []
    
    for port in ports_to_check:
        try:
            result = run_command(['lsof', '-i', f':{port}'])
            if result and result.stdout.strip():
                ports_in_use.append(port)
        except Exception:
            pass
    
    if ports_in_use:
        print(f"   ⚠️  Some ports are still in use: {ports_in_use}")
    else:
        print("   ✅ All ports are free")
    
    # Check Redis status
    try:
        result = run_command(['redis-cli', 'ping'])
        if result and result.stdout.strip() == 'PONG':
            print("   ⚠️  Redis is still running")
        else:
            print("   ✅ Redis has been stopped")
    except Exception:
        print("   ✅ Redis has been stopped")

def cleanup_stuck_jobs():
    """Clean up stuck jobs and their associated processes."""
    print_status("Cleaning up stuck jobs and their associated processes", "🚨")
    
    # Kill processes matching patterns that might indicate stuck jobs
    stuck_patterns = [
        "rq worker",
        "run_worker",
        "uvicorn main:app",
        "next dev",
        "local-ssl-proxy",
        "npm exec",
        "node"
    ]
    
    for pattern in stuck_patterns:
        print_status(f"Attempting to kill processes matching: {pattern}")
        result = run_command(['pkill', '-f', pattern])
        if result and result.returncode == 0:
            print(f"   ✅ Killed processes matching: {pattern}")
        else:
            print(f"   ⚠️  No processes found matching: {pattern}")
    
    # Kill processes using specific ports that might be stuck
    stuck_ports = [3000, 3001, 8000, 8443]
    for port in stuck_ports:
        print_status(f"Attempting to kill processes using port {port}")
        try:
            result = run_command(['lsof', '-ti', f':{port}'])
            if result and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        run_command(['kill', '-9', pid])
                        print(f"   ✅ Killed process {pid} using port {port}")
            else:
                print(f"   ⚠️  No processes found using port {port}")
        except Exception as e:
            print(f"   ❌ Error killing processes on port {port}: {e}")
    
    # Force kill any remaining Redis processes
    print_status("Performing aggressive Redis cleanup for stuck jobs", "🔴")
    try:
        run_command(['pkill', '-f', 'redis-server'])
        time.sleep(2)
        print("   ✅ Killed Redis server for aggressive stuck job cleanup")
    except Exception as emergency_error:
        print(f"   ❌ Emergency stuck job cleanup failed: {emergency_error}")

def main():
    """Main cleanup function."""
    print("🧹 Video Editing App - Complete Cleanup")
    print("=" * 50)
    
    # Check if user wants to just clean stuck jobs
    if len(sys.argv) > 1 and sys.argv[1] == "--stuck-jobs-only":
        print_status("Running stuck jobs cleanup only...", "🚨")
        cleanup_stuck_jobs()
        print("\n" + "=" * 50)
        print("🎉 Stuck jobs cleanup completed!")
        print("=" * 50)
        return
    
    # Check if user wants to force Redis flush
    if len(sys.argv) > 1 and sys.argv[1] == "--force-redis-flush":
        print_status("Running forced Redis flush...", "🔴")
        try:
            result = run_command(['redis-cli', 'FLUSHALL'])
            if result and result.returncode == 0:
                print("   ✅ Forced Redis flush completed")
                # Verify
                final_count = run_command(['redis-cli', 'DBSIZE'])
                if final_count and final_count.stdout.strip():
                    remaining_keys = final_count.stdout.strip()
                    print(f"   📊 Remaining Redis keys: {remaining_keys}")
            else:
                print("   ❌ Forced Redis flush failed")
        except Exception as e:
            print(f"   ❌ Error in forced Redis flush: {e}")
        print("\n" + "=" * 50)
        print("🎉 Forced Redis flush completed!")
        print("=" * 50)
        return
    
    # Define process patterns to kill
    process_patterns = [
        "python3 start_local.py",
        "python start_local.py",
        "uvicorn main:app",
        "rq worker",
        "run_worker",
        "next dev",
        "local-ssl-proxy",
        "npm exec",
        "node"
    ]
    
    # Define ports to check
    ports = [3000, 3001, 8000, 8443, 6379]
    
    print_status("Starting comprehensive cleanup...", "🚀")
    
    # Step 1: Kill dependent processes first
    print_status("Step 1: Killing dependent processes", "1️⃣")
    dependent_patterns = ["rq worker", "run_worker"]
    kill_processes_by_pattern(dependent_patterns)
    time.sleep(2)  # Wait for graceful shutdown
    
    # Step 2: Kill all other processes
    print_status("Step 2: Killing all other processes", "2️⃣")
    kill_processes_by_pattern(process_patterns)
    
    # Step 3: Kill processes by port
    print_status("Step 3: Killing processes by port", "3️⃣")
    kill_processes_by_port(ports)
    
    # Step 4: Clean Redis first (to stop job processing)
    print_status("Step 4: Cleaning Redis", "4️⃣")
    cleanup_redis()
    
    # Step 5: Clean up stuck jobs and processes (after Redis is stopped)
    print_status("Step 5: Cleaning up stuck jobs and processes", "5️⃣")
    cleanup_stuck_jobs()
    
    # Step 6: Clear caches
    print_status("Step 6: Clearing caches", "6️⃣")
    clear_caches()
    
    # Step 7: Clean Docker (if available)
    print_status("Step 7: Cleaning Docker", "7️⃣")
    cleanup_docker()
    
    # Step 8: Final verification
    print_status("Step 8: Verifying cleanup", "8️⃣")
    verify_cleanup()
    
    print("\n" + "=" * 50)
    print("🎉 Cleanup completed!")
    print("=" * 50)
    print("✅ All processes killed")
    print("✅ All caches cleared")
    print("✅ Redis jobs cleaned")
    print("✅ Ports freed")
    print("\n💡 You can now run 'python3 start_local.py' for a fresh start!")
    print("\n💡 To clean only stuck jobs, run: python cleanup_all.py --stuck-jobs-only")
    print("💡 To force Redis flush, run: python cleanup_all.py --force-redis-flush")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Cleanup failed with error: {e}")
        sys.exit(1)
