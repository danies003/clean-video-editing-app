#!/usr/bin/env python3
"""
Quick single video test - should complete in ~30 seconds
"""

import sys
import os
import requests
import time
from pathlib import Path

def test_single_video():
    """Test with a single video for quick completion."""
    print("🎬 Quick Single Video Test")
    print("=" * 50)
    
    # Find one video
    video_dir = Path("../Video Editing App/testing _video_source")
    if not video_dir.exists():
        print(f"❌ Test video directory not found: {video_dir}")
        return False
    
    video_files = list(video_dir.glob("*.MOV"))
    if not video_files:
        print("❌ No video files found")
        return False
    
    # Use just ONE video
    video_file = video_files[0]
    print(f"✅ Using video: {video_file.name}")
    
    # Check backend
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            print("❌ Backend not ready")
            return False
        print("✅ Backend ready")
    except:
        print("❌ Backend not responding")
        return False
    
    # Upload single video
    files = [('files', (video_file.name, open(video_file, 'rb'), 'video/mp4'))]
    data = {"project_name": "Quick Test", "video_ids": [str(video_file)]}
    
    try:
        print("📤 Uploading video...")
        response = requests.post(
            "http://localhost:8000/api/v1/multi-video/projects",
            files=files,
            data=data,
            timeout=30
        )
        
        files[0][1][1].close()
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return False
        
        project = response.json()
        project_id = project["project_id"]
        print(f"✅ Project created: {project_id}")
        
        # Wait for completion (max 2 minutes)
        print("⏳ Waiting for processing...")
        for i in range(24):  # 24 * 5s = 2 minutes
            time.sleep(5)
            
            status_resp = requests.get(
                f"http://localhost:8000/api/v1/multi-video/projects/{project_id}/status",
                timeout=10
            )
            
            if status_resp.status_code == 200:
                status = status_resp.json()
                progress = status.get("progress", 0)
                state = status.get("status", "unknown")
                
                print(f"📊 {i*5}s: {progress}% - {state}")
                
                if state == "completed":
                    print("✅ Processing completed!")
                    
                    # Get output URL
                    output_url = status.get("output_video_url")
                    if output_url:
                        print(f"🎬 Video URL: {output_url}")
                        return True
                    else:
                        print("⚠️ No output URL")
                        return False
                
                elif state == "failed":
                    error = status.get("error")
                    print(f"❌ Processing failed: {error}")
                    return False
        
        print("❌ Timeout after 2 minutes")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_single_video()
    sys.exit(0 if success else 1)

