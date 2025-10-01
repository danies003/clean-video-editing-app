"""
Enhanced Global FFmpeg Process Manager

This module provides centralized control over all FFmpeg operations
to prevent resource contention and hanging processes with real-time monitoring.
"""

import asyncio
import logging
import os
import signal
import subprocess
import time
import psutil
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class FFmpegProcess:
    """Represents a running FFmpeg process."""
    pid: int
    command: List[str]
    start_time: datetime
    timeout: int
    description: str
    status: str = "running"  # running, stuck, completed, failed, timed_out
    last_heartbeat: datetime = None
    progress_check_count: int = 0

class GlobalFFmpegManager:
    """
    Enhanced Global manager for all FFmpeg operations across the application.
    
    This manager:
    1. Limits concurrent FFmpeg processes
    2. Monitors for hanging processes with heartbeat detection
    3. Provides timeout protection
    4. Automatically cleans up stuck processes
    5. Uses psutil for real-time process monitoring
    6. Prevents 59% hangs with proactive process management
    """
    
    def __init__(self, max_concurrent: int = 2, cleanup_interval: int = 15, heartbeat_interval: int = 10):
        self.max_concurrent = max_concurrent
        self.cleanup_interval = cleanup_interval
        self.heartbeat_interval = heartbeat_interval
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_processes: Dict[int, FFmpegProcess] = {}
        self.process_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._start_background_tasks()
        logger.info(f"🚀 [FFMPEG MANAGER] Enhanced manager initialized with {max_concurrent} max concurrent, {cleanup_interval}s cleanup, {heartbeat_interval}s heartbeat")
    
    def _start_background_tasks(self):
        """Start the background monitoring tasks."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("✅ [FFMPEG MANAGER] Background monitoring tasks started")
    
    async def _cleanup_loop(self):
        """Background loop to clean up stuck FFmpeg processes."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stuck_processes()
            except Exception as e:
                logger.error(f"FFmpeg cleanup loop error: {e}")
    
    async def _heartbeat_loop(self):
        """Background loop to monitor process heartbeats and detect stuck processes."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._check_process_heartbeats()
            except Exception as e:
                logger.error(f"FFmpeg heartbeat loop error: {e}")
    
    async def _check_process_heartbeats(self):
        """Check process heartbeats and detect stuck processes proactively."""
        async with self.process_lock:
            current_time = datetime.now()
            stuck_processes = []
            
            for pid, process in list(self.active_processes.items()):
                if process.status != "running":
                    continue
                
                # Check if process is alive at OS level
                if not self._is_process_alive(pid):
                    logger.warning(f"💀 [FFMPEG MANAGER] Process {pid} is not alive at OS level: {process.description}")
                    process.status = "failed"
                    stuck_processes.append(pid)
                    continue
                
                # Check if process is consuming CPU (heartbeat)
                if not self._is_process_consuming_cpu(pid):
                    process.progress_check_count += 1
                    logger.warning(f"⚠️ [FFMPEG MANAGER] Process {pid} not consuming CPU (check {process.progress_check_count}): {process.description}")
                    if process.progress_check_count >= 3:
                        logger.error(f"🚨 [FFMPEG MANAGER] Process {pid} appears stuck (no CPU activity for {process.progress_check_count} checks): {process.description}")
                        process.status = "stuck"
                        stuck_processes.append(pid)
                else:
                    process.progress_check_count = 0
                    process.last_heartbeat = current_time
                
                # Check timeout
                elapsed = (current_time - process.start_time).total_seconds()
                if elapsed > process.timeout:
                    logger.warning(f"⏰ [FFMPEG MANAGER] Process {pid} exceeded timeout ({elapsed}s > {process.timeout}s): {process.description}")
                    process.status = "timed_out"
                    stuck_processes.append(pid)
            
            # Kill stuck processes
            for pid in stuck_processes:
                await self._kill_process(pid)
    
    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is alive at the OS level."""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
    
    def _is_process_stuck(self, pid: int) -> bool:
        """Check if a process is in a stopped/traced state (T state)."""
        try:
            process = psutil.Process(pid)
            status = process.status()
            return status == psutil.STATUS_STOPPED
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
    
    def _is_process_consuming_cpu(self, pid: int) -> bool:
        """Check if a process is consuming CPU (heartbeat)."""
        try:
            process = psutil.Process(pid)
            cpu_percent = process.cpu_percent(interval=0.1)
            return cpu_percent > 0.1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False
    
    async def _cleanup_stuck_processes(self):
        """Clean up stuck FFmpeg processes (legacy method, now handled by heartbeat)."""
        # This is now handled by the heartbeat loop
        pass
    
    async def _kill_process(self, pid: int):
        """Kill a specific process."""
        try:
            if pid in self.active_processes:
                process_info = self.active_processes[pid]
                process_info.status = "timed_out"
                
                # Try graceful termination first
                try:
                    os.kill(pid, signal.SIGTERM)
                    await asyncio.sleep(2)
                    
                    # Check if process is still running
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        # Process still running, force kill
                        os.kill(pid, signal.SIGKILL)
                        logger.info(f"🧹 [FFMPEG MANAGER] Force killed stuck process {pid}")
                    except OSError:
                        logger.info(f"🧹 [FFMPEG MANAGER] Process {pid} terminated gracefully")
                        
                except OSError as e:
                    logger.warning(f"🧹 [FFMPEG MANAGER] Failed to kill process {pid}: {e}")
                
                # Remove from active processes
                del self.active_processes[pid]
                
        except Exception as e:
            logger.error(f"🧹 [FFMPEG MANAGER] Error killing process {pid}: {e}")
    
    async def run_ffmpeg(
        self,
        command: List[str],
        timeout: int = 300,
        description: str = "FFmpeg operation",
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run an FFmpeg command with enhanced global process management.
        
        Args:
            command: FFmpeg command list
            timeout: Timeout in seconds
            description: Human-readable description of the operation
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            CompletedProcess result
            
        Raises:
            subprocess.TimeoutExpired: If operation times out
            Exception: If operation fails
        """
        # Acquire semaphore to limit concurrent operations
        async with self.semaphore:
            logger.info(f"🎬 [FFMPEG MANAGER] Acquired semaphore for: {description}")
            
            try:
                # Use asyncio.create_subprocess_exec with proper process group isolation
                if capture_output:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True
                    )
                else:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        start_new_session=True
                    )
                
                # Register the process with enhanced monitoring
                async with self.process_lock:
                    process_info = FFmpegProcess(
                        pid=process.pid,
                        command=command,
                        start_time=datetime.now(),
                        timeout=timeout,
                        description=description,
                        last_heartbeat=datetime.now()
                    )
                    self.active_processes[process.pid] = process_info
                
                logger.info(f"🎬 [FFMPEG MANAGER] Started process {process.pid}: {description}")
                
                # Enhanced monitoring: Check if process gets stuck immediately
                # For fast commands like ffprobe, use shorter delay
                if "ffprobe" in command[0]:
                    await asyncio.sleep(0.05)  # Very short delay for ffprobe
                else:
                    await asyncio.sleep(0.2)  # Standard delay for other commands
                
                # Check if process is alive and not stuck
                if not self._is_process_alive(process.pid):
                    raise Exception(f"Process {process.pid} failed to start or got stuck immediately")
                
                # Check if process is in a stopped/traced state
                if self._is_process_stuck(process.pid):
                    logger.error(f"🚨 [FFMPEG MANAGER] Process {process.pid} is in stopped/traced state immediately after creation")
                    # Try to kill and restart with fallback method
                    await self._kill_process(process.pid)
                    raise Exception(f"Process {process.pid} got stuck in T state immediately - this indicates a system-level subprocess issue")
                
                logger.info(f"✅ [FFMPEG MANAGER] Process {process.pid} started successfully and is running")
                
                try:
                    # Wait for completion with timeout
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                    
                    # Decode output if needed
                    if capture_output:
                        stdout = stdout.decode('utf-8') if stdout else ""
                        stderr = stderr.decode('utf-8') if stderr else ""
                    
                    # Update process status
                    async with self.process_lock:
                        if process.pid in self.active_processes:
                            process_info = self.active_processes[process.pid]
                            process_info.status = "completed"
                            del self.active_processes[process.pid]
                    
                    logger.info(f"🎬 [FFMPEG MANAGER] Process {process.pid} completed: {description}")
                    
                    # Create CompletedProcess result
                    result = subprocess.CompletedProcess(
                        args=command,
                        returncode=process.returncode,
                        stdout=stdout,
                        stderr=stderr
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"🎬 [FFMPEG MANAGER] Process {process.pid} failed with return code {result.returncode}: {description}")
                        logger.error(f"🎬 [FFMPEG MANAGER] stderr: {stderr}")
                        raise Exception(f"FFmpeg operation failed: {stderr}")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    # Process timed out
                    async with self.process_lock:
                        if process.pid in self.active_processes:
                            process_info = self.active_processes[process.pid]
                            process_info.status = "timed_out"
                    
                    logger.error(f"🎬 [FFMPEG MANAGER] Process {process.pid} timed out after {timeout}s: {description}")
                    
                    # Kill the timed out process
                    await self._kill_process(process.pid)
                    
                    raise subprocess.TimeoutExpired(command, timeout)
                    
            except Exception as e:
                logger.error(f"🎬 [FFMPEG MANAGER] Error running FFmpeg command: {e}")
                raise e  # Re-raise the error without fallback
            finally:
                logger.info(f"🎬 [FFMPEG MANAGER] Released semaphore for: {description}")
    
    async def get_active_processes(self) -> List[FFmpegProcess]:
        """Get list of currently active FFmpeg processes."""
        async with self.process_lock:
            return list(self.active_processes.values())
    
    async def kill_all_processes(self):
        """Kill all active FFmpeg processes."""
        async with self.process_lock:
            for pid in list(self.active_processes.keys()):
                await self._kill_process(pid)
    
    async def shutdown(self):
        """Shutdown the manager and clean up all processes."""
        logger.info("🛑 [FFMPEG MANAGER] Shutting down enhanced manager...")
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        await self.kill_all_processes()
        logger.info("🧹 [FFMPEG MANAGER] Enhanced manager shutdown complete")

# Global instance
_manager: Optional[GlobalFFmpegManager] = None

async def get_ffmpeg_manager() -> GlobalFFmpegManager:
    """Get the global FFmpeg manager instance."""
    global _manager
    if _manager is None:
        _manager = GlobalFFmpegManager()
    return _manager

async def shutdown_ffmpeg_manager():
    """Shutdown the global FFmpeg manager."""
    global _manager
    if _manager:
        await _manager.shutdown()
        _manager = None
