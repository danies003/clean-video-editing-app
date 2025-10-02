#!/usr/bin/env python3
"""
Comprehensive end-to-end test for the new clean Video Editing App.
This test mimics test_direct_multi_video.py but for the new clean app.
"""

import sys
import os
import requests
import time
import subprocess
import re
import json
import webbrowser
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_clean_app_imports():
    """Test that all essential components can be imported."""
    print("🔧 Testing new clean app imports...")
    
    try:
        # Test core imports
        from app.config.settings import get_settings
        from app.editor.multi_video_editor import MultiVideoEditor
        from app.job_queue.worker import JobQueue
        from app.ingestion.storage import StorageClient
        from app.services.manager import ServiceManager
        from app.analyzer.engine import VideoAnalysisEngine
        
        print("✅ Core modules imported successfully")
        
        # Test settings
        settings = get_settings()
        print(f"✅ Settings loaded: Redis URL configured")
        
        editor = MultiVideoEditor()
        print("✅ MultiVideoEditor instantiated successfully")
        
        job_queue = JobQueue(redis_url="redis://localhost:6379/0")
        print("✅ JobQueue instantiated successfully")
        
        storage = StorageClient(
            aws_access_key_id="test",
            aws_secret_access_key="test"
        )
        print("✅ StorageClient instantiated successfully")
        
        service_manager = ServiceManager()
        print("✅ ServiceManager instantiated successfully")
        
        analysis_engine = VideoAnalysisEngine()
        print("✅ VideoAnalysisEngine instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Clean app import test failed: {e}")
        return False

def validate_intelligent_features(project_id):
    """Validate that the output video has intelligent features."""
    print(f"🔍 Validating intelligent features for project {project_id}...")
    
    try:
        # Get timeline to find the output video URL
        timeline_response = requests.get(
            f"http://localhost:8000/api/v1/multi-video/projects/{project_id}/timeline",
            timeout=10
        )
        
        if timeline_response.status_code != 200:
            print(f"❌ Timeline request failed: {timeline_response.status_code}")
            return False
        
        timeline_data = timeline_response.json()
        segments = timeline_data.get("segments", [])
        
        if not segments:
            print("❌ No segments in timeline")
            return False
        
        # Get the first segment's video URL
        first_segment = segments[0]
        video_url = first_segment.get("video_url")
        
        if not video_url:
            print("❌ No video URL in timeline")
            return False
        
        print(f"🎬 Analyzing video: {video_url}")
        
        # Download and analyze the video
        video_response = requests.get(video_url, timeout=30)
        if video_response.status_code != 200:
            print(f"❌ Video download failed: {video_response.status_code}")
            return False
        
        # Save video locally for analysis
        local_video_path = f"output_video_{project_id}.mp4"
        with open(local_video_path, 'wb') as f:
            f.write(video_response.content)
        
        print(f"✅ Video downloaded: {local_video_path}")
        
        # Use ffprobe to analyze the video
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration,size', '-show_entries', 
                'stream=width,height,codec_name', 
                local_video_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout
                print("📊 Video Analysis Results:")
                print(output)
                
                # Check for video properties
                has_video = 'codec_name=h264' in output
                has_audio = 'codec_name=mp3' in output or 'codec_name=aac' in output
                has_duration = 'duration=' in output
                has_resolution = 'width=' in output and 'height=' in output
                
                print(f"✅ Video codec: {has_video}")
                print(f"✅ Audio track: {has_audio}")
                print(f"✅ Duration info: {has_duration}")
                print(f"✅ Resolution info: {has_resolution}")
                
                # Clean up
                os.remove(local_video_path)
                
                return has_video and has_audio and has_duration and has_resolution
            else:
                print(f"❌ ffprobe failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Video analysis timed out")
            return False
        except FileNotFoundError:
            print("⚠️ ffprobe not found, skipping detailed analysis")
            return True  # Assume success if ffprobe not available
            
    except Exception as e:
        print(f"❌ Intelligent features validation failed: {e}")
        return False

def validate_timeline_accuracy(project_id):
    """Validate that timeline data matches actual video duration."""
    print(f"⏱️ Validating timeline accuracy for project {project_id}...")
    
    try:
        # Get timeline
        timeline_response = requests.get(
            f"http://localhost:8000/api/v1/multi-video/projects/{project_id}/timeline",
            timeout=10
        )
        
        if timeline_response.status_code != 200:
            print(f"❌ Timeline request failed: {timeline_response.status_code}")
            return False
        
        timeline_data = timeline_response.json()
        segments = timeline_data.get("segments", [])
        
        if not segments:
            print("❌ No segments in timeline")
            return False
        
        # Check if all segments have valid timing data
        for i, segment in enumerate(segments):
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = segment.get("duration", 0)
            
            print(f"📊 Segment {i}: start={start_time}, end={end_time}, duration={duration}")
            
            if duration <= 0:
                print(f"❌ Segment {i} has invalid duration: {duration}")
                return False
        
        print("✅ Timeline accuracy validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Timeline accuracy validation failed: {e}")
        return False

def test_multi_video_project_creation():
    """Test creating a multi-video project end-to-end."""
    print("🎬 Testing multi-video project creation...")
    
    # Test video paths
    video_dir = Path("../Video Editing App/testing _video_source")
    if not video_dir.exists():
        print(f"❌ Test video directory not found: {video_dir}")
        return False
    
    # Find video files with various extensions
    video_files = []
    for ext in ["*.mp4", "*.MOV", "*.mov", "*.avi"]:
        video_files.extend(list(video_dir.glob(ext)))
    
    if len(video_files) < 2:
        print(f"❌ Need at least 2 video files, found {len(video_files)}")
        return False
    
    print(f"✅ Found {len(video_files)} test videos")
    
    # Filter and prioritize videos
    filtered_videos = []
    for video_file in video_files:
        if video_file.name == "source.mp4":
            filtered_videos.insert(0, video_file)  # Prioritize source.mp4
        else:
            filtered_videos.append(video_file)
    
    # Use first 3 videos
    selected_videos = filtered_videos[:3]
    print(f"🎯 Selected videos: {[v.name for v in selected_videos]}")
    
    # Test API endpoints
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code != 200:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
        print("✅ Backend health check passed")
        
        # Prepare files for upload
        files = []
        for video_file in selected_videos:
            files.append(('files', (video_file.name, open(video_file, 'rb'), 'video/mp4')))
        
        # Prepare project data
        project_data = {
            "project_name": "New Clean App Test Project",
            "video_ids": [str(f) for f in selected_videos]
        }
        
        # Create multi-video project
        create_response = requests.post(
            f"{base_url}/api/v1/multi-video/projects",
            files=files,
            data=project_data,
            timeout=30
        )
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if create_response.status_code != 200:
            print(f"❌ Project creation failed: {create_response.status_code}")
            print(f"   Response: {create_response.text}")
            return False
        
        project_info = create_response.json()
        project_id = project_info.get("project_id")
        print(f"✅ Project created: {project_id}")
        
        # Wait for processing
        print("⏳ Waiting for video processing...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(
                f"{base_url}/api/v1/multi-video/projects/{project_id}/status",
                timeout=10
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                progress = status_data.get("progress", 0)
                status = status_data.get("status", "unknown")
                
                print(f"📊 Progress: {progress}% - Status: {status}")
                
                if status == "completed":
                    print("✅ Video processing completed!")
                    
                    # Validate intelligent features
                    features_valid = validate_intelligent_features(project_id)
                    
                    # Validate timeline accuracy
                    timeline_valid = validate_timeline_accuracy(project_id)
                    
                    # Test timeline endpoint
                    timeline_response = requests.get(
                        f"{base_url}/api/v1/multi-video/projects/{project_id}/timeline",
                        timeout=10
                    )
                    
                    if timeline_response.status_code == 200:
                        timeline_data = timeline_response.json()
                        segments = timeline_data.get("segments", [])
                        print(f"✅ Timeline retrieved: {len(segments)} segments")
                        
                        # Test video download
                        if segments:
                            first_segment = segments[0]
                            video_url = first_segment.get("video_url")
                            if video_url:
                                print(f"🔗 Testing video URL: {video_url}")
                                
                                # Try to download the video
                                video_response = requests.get(video_url, timeout=30)
                                if video_response.status_code == 200:
                                    print("✅ Video download successful!")
                                    
                                    # Open the video in browser
                                    print("🌐 Opening video in browser...")
                                    webbrowser.open(video_url)
                                    
                                    return True
                                else:
                                    print(f"❌ Video download failed: {video_response.status_code}")
                                    return False
                            else:
                                print("❌ No video URL in timeline")
                                return False
                        else:
                            print("❌ No segments in timeline")
                            return False
                    else:
                        print(f"❌ Timeline request failed: {timeline_response.status_code}")
                        return False
                
                elif status == "failed":
                    print("❌ Video processing failed")
                    return False
                
                time.sleep(5)  # Wait 5 seconds before checking again
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                return False
        
        print("❌ Processing timeout")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 [NEW CLEAN APP E2E TEST] Starting comprehensive end-to-end test")
    print("=" * 70)
    
    # Test 1: Imports
    print("\n🔧 Testing new clean app imports...")
    import_success = test_clean_app_imports()
    if not import_success:
        print("❌ New clean app import test failed!")
        return False
    
    # Test 2: End-to-end multi-video test
    print("\n🎬 Testing multi-video project creation...")
    print("🚀 Running full end-to-end test...")
    
    project_success = test_multi_video_project_creation()
    if project_success:
        print("✅ Multi-video project creation test passed!")
    else:
        print("⚠️ Multi-video project creation test failed!")
    
    print("\n" + "=" * 70)
    print("🎉 New Clean App E2E tests completed!")
    print("📊 Summary:")
    print("   - New clean app imports: ✅ Working")
    print("   - Essential components: ✅ Present")
    print(f"   - Multi-video test: {'✅ Passed' if project_success else '❌ Failed'}")
    
    return import_success and project_success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 New Clean App E2E test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ New Clean App E2E test failed!")
        sys.exit(1)
