#!/usr/bin/env python3
"""
Startup script for Railway deployment
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting AI Video Editor API...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Environment variables:")
    for key, value in os.environ.items():
        if key.startswith(('PORT', 'RAILWAY', 'NIXPACKS')):
            logger.info(f"  {key}={value}")
    
    try:
        import simple_main
        logger.info("Successfully imported simple_main module")
        
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting server on port {port}")
        
        uvicorn.run(simple_main.app, host="0.0.0.0", port=port)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
