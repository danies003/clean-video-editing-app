#!/usr/bin/env python3
"""
Railway Worker - Background Job Processor
This script runs as a separate Railway service to process video editing jobs.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("railway_worker")

def initialize_services():
    """Initialize all services needed for the worker."""
    logger.info("🔄 WORKER STARTUP: Initializing Railway Worker...")
    
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        
        logger.info(f"📍 Redis URL: {settings.redis_url}")
        logger.info(f"📍 Queue Name: {settings.queue_name}")
        logger.info(f"📍 S3 Bucket: {settings.s3_bucket_name}")
        
        # Initialize Redis connection
        logger.info("🔗 Connecting to Redis...")
        from app.job_queue.worker import initialize_redis_connection
        asyncio.run(initialize_redis_connection(settings.redis_url))
        logger.info("✅ Redis connection established")

        # Initialize S3 storage
        logger.info("☁️ Initializing S3 storage client...")
        from app.ingestion.storage import initialize_storage_client
        asyncio.run(initialize_storage_client(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_region=settings.aws_region,
            bucket_name=settings.s3_bucket_name
        ))
        logger.info("✅ S3 storage client initialized")

        # Try to initialize optional services (they'll be lazy-loaded if they fail)
        optional_services = [
            ("analysis engine", "app.analyzer.engine", "initialize_analysis_engine"),
            ("template manager", "app.templates.manager", "initialize_template_manager"),
            ("timeline builder", "app.timeline.builder", "initialize_timeline_builder"),
            ("video renderer", "app.editor.renderer", "initialize_renderer"),
        ]
        
        for service_name, module_name, func_name in optional_services:
            try:
                logger.info(f"🔄 Initializing {service_name}...")
                module = __import__(module_name, fromlist=[func_name])
                init_func = getattr(module, func_name)
                asyncio.run(init_func())
                logger.info(f"✅ {service_name} ready")
            except Exception as e:
                logger.warning(f"⚠️ {service_name} initialization failed (will be lazy-loaded): {e}")

        logger.info("🚀 CORE SERVICES INITIALIZED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"❌ CRITICAL WORKER INITIALIZATION FAILED: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def start_worker():
    """Start the RQ worker to process jobs."""
    try:
        from rq import Worker, Queue
        from redis import Redis
        from app.config.settings import get_settings
        
        settings = get_settings()
        start_time = datetime.now()
        
        logger.info(f"🎯 WORKER STARTING: Queue '{settings.queue_name}' at {start_time}")
        
        # Connect to Redis
        conn = Redis.from_url(settings.redis_url)
        conn.ping()
        logger.info("📊 Redis connection verified")
        
        # Check queue status
        queue = Queue(settings.queue_name, connection=conn)
        job_count = len(queue)
        logger.info(f"📦 Queue '{settings.queue_name}' has {job_count} pending jobs")
        
        # Create custom worker with job logging
        class LoggingWorker(Worker):
            def perform_job(self, job, queue):
                logger.info(f"📝 JOB STARTED: {job.id} - {job.func_name}")
                logger.info(f"   Args: {job.args[:2] if len(job.args) > 2 else job.args}")  # Don't log full args
                logger.info(f"   Created: {job.created_at}")
                
                try:
                    result = super().perform_job(job, queue)
                    logger.info(f"✅ JOB COMPLETED: {job.id}")
                    if job.ended_at and job.started_at:
                        duration = job.ended_at - job.started_at
                        logger.info(f"   Duration: {duration}")
                    return result
                except Exception as e:
                    logger.error(f"❌ JOB FAILED: {job.id}")
                    logger.error(f"   Error: {type(e).__name__}: {str(e)[:200]}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()[:500]}")
                    raise
        
        # Create and start worker
        worker = LoggingWorker([settings.queue_name], connection=conn)
        logger.info(f"👷 Worker created: {worker.name}")
        logger.info("🏃 WORKER RUNNING - Waiting for jobs...")
        logger.info("=" * 60)
        
        # Start processing jobs
        worker.work()
        
    except KeyboardInterrupt:
        logger.info("⏹️ WORKER STOPPED: Received keyboard interrupt")
    except Exception as e:
        logger.error(f"💥 WORKER ERROR: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"🏁 WORKER SHUTDOWN: Runtime {duration}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Railway Worker Starting...")
    logger.info("=" * 60)
    
    # Initialize services
    if not initialize_services():
        logger.error("❌ Failed to initialize services. Exiting.")
        sys.exit(1)
    
    # Start worker
    logger.info("")
    logger.info("Starting RQ worker process...")
    start_worker()

