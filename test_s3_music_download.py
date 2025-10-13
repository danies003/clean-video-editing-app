"""
Test S3 music download functionality locally to verify it works before deploying.
"""
import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

BUCKET_NAME = 'my-video-editing-app-2025'

def test_music_download():
    """Test downloading a music file from S3."""
    print("Testing S3 music download...")
    print("=" * 60)
    
    # Test file
    s3_key = "assets/music/Test/Only me - Patrick Patrikios.mp3"
    local_path = "/tmp/test_music.mp3"
    
    try:
        print(f"📥 Downloading from S3: {s3_key}")
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"✅ Downloaded to: {local_path}")
        
        # Check file
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            print(f"✅ File exists: {size / (1024*1024):.2f} MB")
            
            # Clean up
            os.remove(local_path)
            print("✅ Test passed - S3 music download works!")
            return True
        else:
            print("❌ File not found after download")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = test_music_download()
    print("=" * 60)
    if success:
        print("✅ S3 music download is working correctly")
        print("The issue must be in the video editing code logic")
    else:
        print("❌ S3 music download is NOT working")
        print("Need to fix S3 credentials or permissions")

