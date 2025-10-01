#!/usr/bin/env python3
"""
Test script to manually process the stuck job
"""

import sys
import os
import asyncio
import json
from uuid import UUID

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_job_processing():
    """Test processing the stuck job manually"""
    try:
        print("Testing job processing manually...")
        
        # Initialize Redis connection
        from app.job_queue.worker import initialize_redis_connection
        from app.config.settings import settings
        
        await initialize_redis_connection(settings.redis_url)
        from app.job_queue.worker import get_job_queue
        job_queue = await get_job_queue()
        
        # Process the stuck job
        job_id = UUID("882e765f-f818-4073-b1a1-e0d4f9dbe969")
        print(f"Processing job: {job_id}")
        
        result = await job_queue._process_editing_job(job_id)
        print(f"Job processing result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Job processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing job processing...")
    print("=" * 50)
    
    success = asyncio.run(test_job_processing())
    
    if success:
        print("\n✅ Job processing successful!")
    else:
        print("\n❌ Job processing failed!")
        sys.exit(1)
