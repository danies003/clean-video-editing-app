#!/usr/bin/env python3
"""
Debug job processing to see why it's stuck at 30%
"""
import sys
import os
import traceback
from uuid import UUID

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_job_processing():
    try:
        print("🔍 [DEBUG] Starting job processing debug...")
        
        # Import required modules
        from app.job_queue.worker import process_editing_job_standalone
        
        # Job ID that's stuck
        job_id = "1c3ed5f6-5f10-456a-9ebd-8e04baca0d7a"
        print(f"🔍 [DEBUG] Processing job: {job_id}")
        
        # Call the job processing function directly
        result = process_editing_job_standalone(job_id)
        print(f"✅ [DEBUG] Job processing result: {result}")
        
    except Exception as e:
        print(f"❌ [DEBUG] Error during job processing: {e}")
        print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_job_processing()
