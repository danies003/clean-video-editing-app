import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import asyncio
import logging
import sys
from datetime import datetime
from app.config.settings import settings
from app.job_queue.worker import initialize_redis_connection
from app.ingestion.storage import initialize_storage_client
from app.analyzer.engine import initialize_analysis_engine
from app.templates.manager import initialize_template_manager
from app.timeline.builder import initialize_timeline_builder

# Configure MoviePy to use system FFMPEG
import os
import shutil

# Try to find ffmpeg in system PATH, fallback to common locations
ffmpeg_path = shutil.which('ffmpeg')
if not ffmpeg_path:
    # Try common macOS location
    if os.path.exists('/opt/homebrew/bin/ffmpeg'):
        ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
    # Try common Linux location
    elif os.path.exists('/usr/bin/ffmpeg'):
        ffmpeg_path = '/usr/bin/ffmpeg'
    
if ffmpeg_path:
    os.environ['FFMPEG_BINARY'] = ffmpeg_path

# Configure detailed logging for worker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("worker.log", mode="a")
    ]
)
logger = logging.getLogger("video_worker")

def initialize_services():
    """Initialize all services with detailed logging."""
    logger.info("🔄 WORKER STARTUP: Initializing Video Editing Worker...")
    logger.info(f"📍 Redis URL: {settings.redis_url}")
    logger.info(f"📍 Queue Name: {settings.queue_name}")
    logger.info(f"📍 S3 Bucket: {settings.s3_bucket_name}")
    
    try:
        # Initialize the job queue before starting the worker
        logger.info("🔗 Connecting to Redis...")
        asyncio.run(initialize_redis_connection(settings.redis_url))
        logger.info("✅ Redis connection established")

        # Initialize the storage client before starting the worker
        logger.info("☁️ Initializing S3 storage client...")
        asyncio.run(initialize_storage_client(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_region=settings.aws_region,
            bucket_name=settings.s3_bucket_name,
            endpoint_url=settings.s3_endpoint_url
        ))
        logger.info("✅ S3 storage client initialized")

        # Try to initialize analysis engine (optional - might fail due to heavy dependencies)
        try:
            logger.info("🔍 Initializing video analysis engine...")
            asyncio.run(initialize_analysis_engine())
            logger.info("✅ Analysis engine ready")
        except Exception as e:
            logger.warning(f"⚠️ Analysis engine initialization failed (will be lazy-loaded): {e}")

        # Try to initialize template manager (optional)
        try:
            logger.info("📋 Initializing template manager...")
            asyncio.run(initialize_template_manager())
            logger.info("✅ Template manager ready")
        except Exception as e:
            logger.warning(f"⚠️ Template manager initialization failed (will be lazy-loaded): {e}")

        # Try to initialize timeline builder (optional)
        try:
            logger.info("⏱️ Initializing timeline builder...")
            asyncio.run(initialize_timeline_builder())
            logger.info("✅ Timeline builder ready")
        except Exception as e:
            logger.warning(f"⚠️ Timeline builder initialization failed (will be lazy-loaded): {e}")
        
        logger.info("🚀 CORE SERVICES INITIALIZED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"❌ CRITICAL WORKER INITIALIZATION FAILED: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Custom worker class with enhanced logging
class LoggingWorker:
    """Custom worker wrapper with detailed logging."""
    
    def __init__(self, queue_name, redis_url):
        self.queue_name = queue_name
        self.redis_url = redis_url
        self.start_time = datetime.now()
        
    def start_worker(self):
        """Start the worker with detailed logging."""
        try:
            from rq import Worker, Queue
            from redis import Redis
        except ImportError as e:
            logger.error(f"❌ RQ import failed: {e}")
            logger.error("Make sure RQ is installed: pip install rq")
            return
        
        logger.info(f"🎯 WORKER STARTING: Queue '{self.queue_name}' at {self.start_time}")
        
        try:
            conn = Redis.from_url(self.redis_url)
            # Test Redis connection
            try:
                if conn:
                    conn.ping()
                    logger.info("📊 Redis connection established successfully")
                else:
                    logger.error("❌ Redis connection is None")
                    return
            except Exception as e:
                logger.error(f"❌ Redis connection failed: {e}")
                return
            
            # Check queue status
            queue = Queue(self.queue_name, connection=conn)
            job_count = len(queue)
            logger.info(f"📦 Queue '{self.queue_name}' has {job_count} pending jobs")
            
            # Create worker with logging
            worker = Worker([self.queue_name], connection=conn)
            logger.info(f"👷 Worker created for queue: {self.queue_name}")
            
            # Log worker details
            logger.info(f"🔧 Worker PID: {worker.pid if hasattr(worker, 'pid') else 'N/A'}")
            logger.info(f"⚙️ Worker name: {worker.name}")
            
            logger.info("🏃 WORKER RUNNING - Waiting for jobs...")
            logger.info("=" * 60)
            
            # Custom worker with job logging
            class LoggingWorker(Worker):
                def perform_job(self, job, queue):
                    logger.info(f"📝 JOB STARTED: {job.id} - {job.func_name}")
                    logger.info(f"   Args: {job.args}")
                    logger.info(f"   Created: {job.created_at}")
                    
                    try:
                        result = super().perform_job(job, queue)
                        logger.info(f"✅ JOB COMPLETED: {job.id}")
                        logger.info(f"   Duration: {job.ended_at - job.started_at if job.ended_at and job.started_at else 'unknown'}")
                        logger.info(f"   Result type: {type(result).__name__}")
                        return result
                    except Exception as e:
                        logger.error(f"❌ JOB FAILED: {job.id}")
                        logger.error(f"   Error: {type(e).__name__}: {e}")
                        import traceback
                        logger.error(f"   Traceback: {traceback.format_exc()}")
                        raise
            
            # Use the logging worker
            logging_worker = LoggingWorker([self.queue_name], connection=conn)
            logging_worker.work()
            
        except KeyboardInterrupt:
            logger.info("⏹️ WORKER STOPPED: Received keyboard interrupt")
        except Exception as e:
            logger.error(f"💥 WORKER ERROR: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            end_time = datetime.now()
            duration = end_time - self.start_time
            logger.info(f"🏁 WORKER SHUTDOWN: Runtime {duration}")

if __name__ == "__main__":
    # Initialize services
    if not initialize_services():
        logger.error("Failed to initialize services. Exiting.")
        sys.exit(1)
    
    # Start worker
    worker = LoggingWorker(settings.queue_name, settings.redis_url)
    worker.start_worker()