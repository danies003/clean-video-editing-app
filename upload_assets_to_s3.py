"""
Upload music, fonts, and LUT assets to S3 for cloud deployment.
This solves the issue of large asset folders in Railway deployments.
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

BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'my-video-editing-app-2025')

def upload_directory_to_s3(local_directory: str, s3_prefix: str):
    """Upload a directory to S3 with progress tracking."""
    local_path = Path(local_directory)
    
    if not local_path.exists():
        print(f"❌ Directory not found: {local_directory}")
        return
    
    files_uploaded = 0
    total_size = 0
    
    for file_path in local_path.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            # Get relative path from local_directory
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
            
            try:
                file_size = file_path.stat().st_size
                print(f"📤 Uploading: {file_path.name} ({file_size / (1024*1024):.2f} MB)")
                
                s3.upload_file(
                    str(file_path),
                    BUCKET_NAME,
                    s3_key
                )
                
                files_uploaded += 1
                total_size += file_size
                print(f"   ✅ Uploaded to: s3://{BUCKET_NAME}/{s3_key}")
                
            except Exception as e:
                print(f"   ❌ Failed to upload {file_path}: {e}")
    
    print(f"\n✅ Uploaded {files_uploaded} files ({total_size / (1024*1024):.2f} MB total)")

def main():
    print("=" * 60)
    print("📦 Uploading Assets to S3")
    print("=" * 60)
    
    # Upload music files
    print("\n🎵 Uploading music files...")
    upload_directory_to_s3("app/assets/music", "assets/music")
    
    # Upload fonts (if needed)
    # print("\n🔤 Uploading font files...")
    # upload_directory_to_s3("app/assets/fonts", "assets/fonts")
    
    # Upload LUTs (if needed)
    # print("\n🎨 Uploading LUT files...")
    # upload_directory_to_s3("app/assets/luts", "assets/luts")
    
    print("\n" + "=" * 60)
    print("✅ All assets uploaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

