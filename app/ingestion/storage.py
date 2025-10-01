"""
S3-compatible cloud storage client for video file management.

This module provides a unified interface for cloud storage operations,
supporting AWS S3, MinIO, and other S3-compatible storage services.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, Union
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import HttpUrl

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class StorageClient:
    """
    S3-compatible storage client for video file operations.
    
    Handles file uploads, downloads, and metadata management
    for the video editing automation engine.
    """
    
    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str = "us-east-1",
        bucket_name: str = "",
        endpoint_url: Optional[str] = None
    ):
        """
        Initialize the storage client.
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_region: AWS region
            bucket_name: S3 bucket name
            endpoint_url: Optional custom endpoint URL for S3-compatible services
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        
        # Initialize S3 client
        s3_kwargs = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'region_name': aws_region
        }
        if endpoint_url is not None and str(endpoint_url).strip():
            s3_kwargs['endpoint_url'] = endpoint_url
        self.s3_client = boto3.client('s3', **s3_kwargs)
        
        # Initialize S3 resource for advanced operations
        s3_resource_kwargs = dict(s3_kwargs)
        self.s3_resource = boto3.resource('s3', **s3_resource_kwargs)
        
        self.bucket = self.s3_resource.Bucket(bucket_name)
    
    async def ping(self) -> bool:
        """
        Test storage connectivity.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            # Test bucket access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Storage ping failed: {e}")
            return False
    
    async def create_upload_url(
        self,
        filename: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        expiration: int = 3600
    ) -> Tuple[HttpUrl, datetime]:
        """
        Create a pre-signed URL for direct file upload.
        
        Args:
            filename: Original filename
            content_type: MIME type of the file
            metadata: Optional metadata to store with the file
            expiration: URL expiration time in seconds
            
        Returns:
            Tuple[HttpUrl, datetime]: Upload URL and expiration time
        """
        try:
            # Generate unique key for the file
            file_key = f"uploads/{datetime.utcnow().strftime('%Y/%m/%d')}/{filename}"
            
            # Prepare upload parameters
            upload_params = {
                'Bucket': self.bucket_name,
                'Key': file_key,
                'ContentType': content_type,
                'ExpiresIn': expiration
            }
            
            # Add metadata if provided
            if metadata:
                upload_params['Metadata'] = {
                    k: str(v) for k, v in metadata.items()
                }
            
            # Generate pre-signed URL
            upload_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params=upload_params,
                ExpiresIn=expiration
            )
            
            expires_at = datetime.utcnow() + timedelta(seconds=expiration)
            
            logger.info(f"Created upload URL for {filename}: {file_key}")
            return HttpUrl(upload_url), expires_at
            
        except Exception as e:
            logger.error(f"Failed to create upload URL for {filename}: {e}")
            raise
    
    async def create_download_url(
        self,
        file_url: HttpUrl,
        expiration: int = 3600
    ) -> HttpUrl:
        """
        Create a pre-signed URL for file download.
        
        Args:
            file_url: Original file URL
            expiration: URL expiration time in seconds
            
        Returns:
            HttpUrl: Download URL
        """
        try:
            # Extract key from URL
            parsed_url = urlparse(str(file_url))
            file_key = parsed_url.path.lstrip('/')
            
            # Generate pre-signed URL
            download_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_key
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"Created download URL for {file_key}")
            return HttpUrl(download_url)
            
        except Exception as e:
            logger.error(f"Failed to create download URL: {e}")
            raise
    
    async def create_download_url_by_key(
        self,
        file_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Create a pre-signed URL for file download by file key.
        
        Args:
            file_key: S3 key of the file
            expiration: URL expiration time in seconds
            
        Returns:
            str: Download URL
        """
        try:
            # Generate pre-signed URL
            download_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_key
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"Created download URL for key {file_key}")
            return download_url
            
        except Exception as e:
            logger.error(f"Failed to create download URL for key {file_key}: {e}")
            raise
    
    async def get_video_access_url(self, file_key: str, expiration: int = 86400) -> HttpUrl:
        """
        Generate a pre-signed URL for video access.
        
        Args:
            file_key: S3 key of the video file
            expiration: URL expiration time in seconds (default 24 hours)
            
        Returns:
            HttpUrl: Pre-signed URL for video access
        """
        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated video access URL for {file_key}")
            return HttpUrl(str(presigned_url))
        except Exception as e:
            logger.error(f"Failed to generate video access URL for {file_key}: {e}")
            raise
    
    async def upload_file(
        self,
        file_path: str,
        file_key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HttpUrl:
        """
        Upload a file to storage.
        
        Args:
            file_path: Local file path
            file_key: Storage key for the file
            content_type: Optional MIME type
            metadata: Optional metadata
            
        Returns:
            HttpUrl: URL of the uploaded file
        """
        try:
            upload_params = {
                'Filename': file_path,
                'Bucket': self.bucket_name,
                'Key': file_key
            }
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            if metadata:
                extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
            if extra_args:
                upload_params['ExtraArgs'] = extra_args
            self.s3_client.upload_file(**upload_params)
            
            # Generate pre-signed URL for public access (24 hours)
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': file_key},
                ExpiresIn=86400  # 24 hours
            )
            
            logger.info(f"Uploaded file {file_path} to {file_key}")
            return HttpUrl(str(presigned_url))
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            raise
    
    async def download_file(
        self,
        file_url: Union[str, HttpUrl],
        local_path: str
    ) -> bool:
        """
        Download a file from storage.
        
        Args:
            file_url: URL of the file to download
            local_path: Local path to save the file
            
        Returns:
            bool: True if download successful
        """
        try:
            # Extract key from URL
            parsed_url = urlparse(str(file_url))
            file_key = parsed_url.path.lstrip('/')
            
            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                file_key,
                local_path
            )
            
            logger.info(f"Downloaded file {file_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file {file_url}: {e}")
            return False
    
    async def delete_file(self, file_url: HttpUrl) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_url: URL of the file to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            # Extract key from URL
            parsed_url = urlparse(str(file_url))
            file_key = parsed_url.path.lstrip('/')
            
            # Delete file
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            logger.info(f"Deleted file {file_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_url}: {e}")
            return False
    
    async def get_file_metadata(self, file_url: HttpUrl) -> Dict[str, Any]:
        """
        Get metadata for a file.
        
        Args:
            file_url: URL of the file
            
        Returns:
            Dict[str, Any]: File metadata
        """
        try:
            # Extract key from URL
            parsed_url = urlparse(str(file_url))
            file_key = parsed_url.path.lstrip('/')
            
            # Get object metadata
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            return {
                'content_type': response.get('ContentType'),
                'content_length': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'metadata': response.get('Metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {file_url}: {e}")
            raise


# Global storage client instance
_storage_client: Optional[StorageClient] = None


async def initialize_storage_client(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str = "us-east-1",
    bucket_name: str = "",
    endpoint_url: Optional[str] = None
) -> StorageClient:
    """
    Initialize the global storage client.
    
    Args:
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_region: AWS region
        bucket_name: S3 bucket name
        endpoint_url: Optional custom endpoint URL
        
    Returns:
        StorageClient: Initialized storage client
    """
    global _storage_client
    
    settings = get_settings()
    
    _storage_client = StorageClient(
        aws_access_key_id=aws_access_key_id or settings.aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key or settings.aws_secret_access_key,
        aws_region=aws_region or settings.aws_region,
        bucket_name=bucket_name or settings.s3_bucket_name,
        endpoint_url=endpoint_url or settings.s3_endpoint_url or None
    )
    
    # Test connection
    if not await _storage_client.ping():
        raise ConnectionError("Failed to connect to storage service")
    
    logger.info("Storage client initialized successfully")
    return _storage_client


def get_storage_client() -> StorageClient:
    """
    Get the global storage client instance.
    
    Returns:
        StorageClient: Storage client instance
        
    Raises:
        RuntimeError: If storage client is not initialized
    """
    if _storage_client is None:
        raise RuntimeError("Storage client not initialized. Call initialize_storage_client first.")
    
    return _storage_client 