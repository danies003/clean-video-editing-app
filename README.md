# Clean Video Editing App

A streamlined, intelligent video editing application with AI-powered analysis and automated editing features.

## Features

- **Intelligent Multi-Video Editing**: AI-powered analysis and editing using Gemini
- **Background Music Integration**: Automatic music selection and volume control
- **Super Font Captions**: Dynamic text overlays with custom fonts
- **Smart Transitions**: Automated transition detection and application
- **Visual Effects**: LUTs, color grading, and cinematic effects
- **Real-time Processing**: Redis job queue for background processing
- **Cloud Storage**: S3 integration for video uploads and outputs
- **Modern UI**: Next.js frontend with real-time progress tracking

## Quick Start

1. **Install Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Services**
   ```bash
   python3 start_local.py
   ```

3. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## Architecture

- **Backend**: FastAPI with Redis job queue
- **Frontend**: Next.js with TypeScript
- **Video Processing**: MoviePy with FFmpeg
- **AI Analysis**: Google Gemini API
- **Storage**: AWS S3
- **Database**: Redis for job management

## Key Components

- `app/editor/multi_video_editor.py` - Core video editing engine
- `app/analyzer/engine.py` - AI content analysis
- `app/job_queue/worker.py` - Background job processing
- `frontend/` - Next.js web interface

## Environment Setup

Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key
REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your_bucket_name
```

## Testing

Run the end-to-end test:
```bash
python3 test_new_clean_e2e.py
```

## License

MIT License
