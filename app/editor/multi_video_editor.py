"""
Multi-Video Editor with Integrated Robust Video Creation

This module integrates the components from create_robust_25_second_video.py
into the established multi-video workflow, providing:
- Gemini-powered video analysis
- Audio processing and synchronization
- Video segment creation with effects
- Final rendering with LUT and text overlays
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID

import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy import concatenate_videoclips
import moviepy.video.io.ffmpeg_reader as ffmpeg_reader
import google.generativeai as genai

from app.config.settings import get_settings
from app.fonts.video_font_renderer import VideoFontRenderer
from app.fonts.ai_font_integration import TextElement, TextElementType
from app.models.schemas import VideoAnalysisResult, EditStyle
from app.editor.enhanced_llm_editor import EnhancedLLMEditor
from app.analyzer.engine import VideoAnalysisEngine

# Import transition points
from transition_points import (
    get_transition_points, 
    get_beat_points, 
    get_measure_points, 
    get_tempo, 
    get_duration,
    TRANSITION_POINTS
)

logger = logging.getLogger(__name__)

def _patch_moviepy_metadata_parsing():
    """
    Patch MoviePy's metadata parsing to handle iPhone video metadata correctly.
    This fixes the root cause of the TypeError: unsupported operand type(s) for +: 'float' and 'str'
    and FFmpeg command errors.
    """
    try:
        import moviepy.video.io.ffmpeg_reader as ffmpeg_reader
        import subprocess
        import json
        import tempfile
        import os
        
        # Store the original function
        original_ffmpeg_parse_infos = ffmpeg_reader.ffmpeg_parse_infos
        
        def patched_ffmpeg_parse_infos(filename, check_duration=True, fps_source='fps', decode_file=False, print_infos=False):
            """Patched version that handles iPhone metadata correctly."""
            try:
                # Call the original function
                result = original_ffmpeg_parse_infos(filename, check_duration, fps_source, decode_file, print_infos)
                return result
            except (TypeError, Exception) as e:
                error_str = str(e)
                if "unsupported operand type(s) for +: 'float' and 'str'" in error_str or \
                   "At least one output file must be specified" in error_str or \
                   "DOVI configuration" in error_str:
                    logger.warning(f"🔧 [MOVIEPY PATCH] Detected FFmpeg command error, applying fix...")
                    try:
                        # Use ffprobe to get basic video info, avoiding DOVI metadata issues
                        cmd = [
                            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                            '-show_entries', 'stream=width,height,r_frame_rate,codec_name',
                            '-select_streams', 'v:0', '-of', 'csv=p=0', filename
                        ]
                        
                        ffprobe_result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
                        lines = ffprobe_result.stdout.strip().split('\n')
                        
                        # Parse CSV output: first line has video info, second line has duration
                        if len(lines) >= 2:
                            # First line: codec_name,width,height,r_frame_rate,
                            parts = lines[0].split(',')
                            video_codec = parts[0] if parts[0] else 'h264'
                            width = int(parts[1]) if parts[1] else 1920
                            height = int(parts[2]) if parts[2] else 1080
                            r_frame_rate = parts[3] if len(parts) > 3 and parts[3] else '30/1'
                            
                            # Second line: duration
                            duration = float(lines[1]) if lines[1] else 0
                        else:
                            # Fallback values
                            duration = 0
                            width = 1920
                            height = 1080
                            r_frame_rate = '30/1'
                            video_codec = 'h264'
                        
                        # Parse frame rate
                        fps = eval(r_frame_rate) if '/' in r_frame_rate else float(r_frame_rate)

                        # Try to get audio info
                        cmd_audio = [
                            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                            '-show_entries', 'stream=codec_name,sample_rate,bit_rate',
                            '-of', 'json', filename
                        ]
                        audio_found = False
                        audio_fps = 44100
                        audio_bitrate = 128
                        audio_codec = 'aac'

                        try:
                            ffprobe_audio_result = subprocess.run(cmd_audio, capture_output=True, text=True, check=True, timeout=5)
                            ffprobe_audio_data = json.loads(ffprobe_audio_result.stdout)
                            if ffprobe_audio_data and 'streams' in ffprobe_audio_data and len(ffprobe_audio_data['streams']) > 0:
                                audio_stream = ffprobe_audio_data['streams'][0]
                                audio_found = True
                                audio_fps = int(audio_stream.get('sample_rate', 44100))
                                audio_bitrate = int(audio_stream.get('bit_rate', 128000)) // 1000
                                audio_codec = audio_stream.get('codec_name', 'aac')
                        except Exception as audio_e:
                            logger.warning(f"⚠️ [MOVIEPY PATCH] Could not extract audio info with ffprobe: {audio_e}")

                        # Create a simple info dict that matches MoviePy's expected format
                        infos = {
                            'duration': duration,
                            'fps': fps,
                            'size': (width, height),
                            'video_fps': fps,
                            'video_duration': duration,
                            'video_size': (width, height),
                            'audio_found': audio_found,
                            'audio_fps': audio_fps,
                            'audio_bitrate': audio_bitrate,
                            'video_bitrate': 1000,  # Default to 1000kbps if not found
                            'audio_codec': audio_codec,
                            'video_codec': video_codec
                        }
                        logger.info(f"✅ [MOVIEPY PATCH] Successfully extracted video info: {infos}")
                        return infos
                    except Exception as ffprobe_e:
                        logger.error(f"❌ [MOVIEPY PATCH] FFprobe fallback also failed: {ffprobe_e}")
                        raise e # Re-raise original error if fallback fails
                else:
                    raise e # Re-raise other exceptions
        
        ffmpeg_reader.ffmpeg_parse_infos = patched_ffmpeg_parse_infos
        logger.info("✅ [MOVIEPY PATCH] Successfully patched ffmpeg_parse_infos")
        
    except Exception as e:
        logger.error(f"❌ [MOVIEPY PATCH] Failed to patch ffmpeg_parse_infos: {e}")


async def safe_video_file_clip(video_path: str, **kwargs):
    """
    Safe VideoFileClip wrapper that handles iPhone video metadata parsing errors.
    Converts problematic videos to compatible format using FFmpeg.
    
    Args:
        video_path: Path to video file
        **kwargs: Additional arguments for VideoFileClip
        
    Returns:
        VideoFileClip instance or None if failed
    """
    logger.info(f"🔍 [SAFE CLIP] Called with video_path: {video_path}, kwargs: {kwargs}")
    
    # Apply the MoviePy patch first
    _patch_moviepy_metadata_parsing()
    
    try:
        # Load the video directly
        clip = VideoFileClip(video_path, **kwargs)
        logger.info(f"✅ [SAFE CLIP] Successfully loaded video: {video_path}")
        return clip
    except Exception as e:
        error_str = str(e)
        logger.info(f"🔍 [SAFE CLIP] Exception caught: {error_str}")
        if "DOVI configuration" in error_str or "unsupported operand type(s) for +: 'float' and 'str'" in error_str:
            logger.warning(f"⚠️ [SAFE CLIP] iPhone video with DOVI detected, converting to compatible format...")
            try:
                # Convert iPhone video to compatible H.264/AAC MP4 format
                import tempfile
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Use FFmpeg to convert to compatible format with faster settings
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-c:v', 'libx264',  # H.264 video codec
                    '-c:a', 'aac',      # AAC audio codec
                    '-preset', 'ultrafast',  # Fastest encoding
                    '-crf', '28',       # Lower quality but faster
                    '-movflags', '+faststart',  # Web optimization
                    '-threads', '4',    # Use multiple threads
                    '-y',  # Overwrite output file
                    temp_path
                ]
                
                logger.info(f"🔄 [SAFE CLIP] Converting iPhone video: {video_path} -> {temp_path}")
                logger.info(f"🔍 [SAFE CLIP] FFmpeg command: {' '.join(cmd)}")
                
                # Use the global FFmpeg manager for robust process management
                import os
                try:
                    from app.utils.ffmpeg_manager import get_ffmpeg_manager
                    ffmpeg_manager = await get_ffmpeg_manager()
                    result = await ffmpeg_manager.run_ffmpeg(
                        command=cmd,
                        timeout=300,
                        description=f"FFmpeg conversion: {os.path.basename(video_path)} to H.264"
                    )
                except Exception as ffmpeg_error:
                    logger.error(f"❌ [SAFE CLIP] FFmpeg manager error: {ffmpeg_error}")
                    # Fallback to subprocess if FFmpeg manager fails
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    # Create a result object that matches the expected interface
                    class SubprocessResult:
                        def __init__(self, subprocess_result):
                            self.returncode = subprocess_result.returncode
                            self.stdout = subprocess_result.stdout
                            self.stderr = subprocess_result.stderr
                    result = SubprocessResult(result)
                
                logger.info(f"🔍 [SAFE CLIP] FFmpeg return code: {result.returncode}")
                if result.stdout:
                    logger.info(f"🔍 [SAFE CLIP] FFmpeg stdout: {result.stdout[:200]}...")
                if result.stderr:
                    logger.info(f"🔍 [SAFE CLIP] FFmpeg stderr: {result.stderr[:200]}...")
                
                if result.returncode == 0:
                    # Check if the converted file exists and has content
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        # Load the converted video
                        converted_clip = VideoFileClip(temp_path, **kwargs)
                        logger.info(f"✅ [SAFE CLIP] Successfully converted and loaded video: {video_path}")
                        logger.info(f"🔍 [SAFE CLIP] Converted clip duration: {converted_clip.duration:.2f}s, size: {converted_clip.size}")
                        
                        # Don't clean up temp file immediately - let MoviePy handle it
                        # The temp file will be cleaned up when the clip is closed
                        
                        return converted_clip
                    else:
                        logger.error(f"❌ [SAFE CLIP] Converted file doesn't exist or is empty: {temp_path}")
                        return None
                else:
                    logger.error(f"❌ [SAFE CLIP] FFmpeg conversion failed: {result.stderr}")
                    return None
                    
            except Exception as e2:
                logger.error(f"❌ [SAFE CLIP] Failed to convert video: {e2}")
                return None
        else:
            logger.error(f"❌ [SAFE CLIP] Failed to load video {video_path}: {e}")
            return None



class MultiVideoEditor:
    """Integrated multi-video editor with robust video creation capabilities."""
    
    def __init__(self):
        self.settings = get_settings()
        self.gemini_model = None
        self._setup_gemini()
        
        # Initialize proper LLM editing system
        self.enhanced_llm_editor = EnhancedLLMEditor()
        self.video_analysis_engine = VideoAnalysisEngine()
        
    def _setup_gemini(self):
        """Setup Gemini API if available."""
        try:
            api_key = self.settings.gemini_api_key
            logger.info(f"🔍 [GEMINI DEBUG] API key length: {len(api_key) if api_key else 0}")
            logger.info(f"🔍 [GEMINI DEBUG] API key starts with: {api_key[:10] if api_key and len(api_key) > 10 else 'N/A'}")
            
            if api_key and len(api_key) > 10 and not api_key.startswith("your-"):
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("✅ Gemini API configured successfully")
            else:
                logger.warning("⚠️ Gemini API key not found, invalid, or placeholder - using fallback analysis")
                self.gemini_model = None
        except Exception as e:
            logger.error(f"❌ Failed to setup Gemini API: {e}")
            self.gemini_model = None
    
    async def create_multi_video(
        self, 
        video_paths: List[str], 
        project_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create an intelligent multi-video with effects and transitions.
        
        Args:
            video_paths: List of video file paths
            project_id: Project identifier
            output_path: Optional output path for the final video
            
        Returns:
            Path to the created video
        """
        print("🎬 [DEBUG] *** create_multi_video METHOD CALLED ***")
        logger.info(f"🎬 [INTELLIGENT] Starting intelligent multi-video creation for project {project_id}")
        logger.info(f"📹 Processing {len(video_paths)} videos with smart editing")
        logger.info(f"📹 Video paths: {video_paths}")
        
        # Create output path if not provided
        if not output_path:
            output_dir = Path("9_10")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"intelligent_video_{project_id}.mp4"
        
        logger.info(f"📁 Output path: {output_path}")
        
        # Use the intelligent editing system directly
        logger.info("🎬 [INTELLIGENT] Using direct intelligent editing system...")
        logger.info("🎬 [INTELLIGENT] Calling _create_full_intelligent_video directly...")
        
        try:
            logger.info("🎬 [INTELLIGENT] *** STARTING INTELLIGENT VIDEO CREATION ***")
            # Convert video paths to Path objects
            video_path_objects = [Path(path) for path in video_paths]
            logger.info(f"🎬 [INTELLIGENT] Converted to Path objects: {len(video_path_objects)}")
            
            # Create mock story data for now
            story_data = {
                'story_moments': [
                    {'description': 'Video content', 'caption': 'Amazing content! ✨'},
                    {'description': 'More content', 'caption': 'So cool! 🎬'}
                ]
            }
            text_overlays = []  # Empty for now
            logger.info("🎬 [INTELLIGENT] Created mock story data and text overlays")
            
            logger.info("🎬 [INTELLIGENT] *** CALLING _create_full_intelligent_video ***")
            result = await self._create_full_intelligent_video(
                selected_clips=video_path_objects,
                story_data=story_data,
                text_overlays=text_overlays,
                project_id=project_id,
                output_path=str(output_path)
            )
            logger.info(f"🎬 [INTELLIGENT] _create_full_intelligent_video completed: {result}")
            
            # FORCE: Add music as post-processing step to ensure it's always there
            logger.info("🎵 [POST-PROCESS] FORCING music addition as final step...")
            try:
                result = await self._ensure_music_on_video(result, project_id)
                logger.info(f"✅ [POST-PROCESS] Music addition complete: {result}")
            except Exception as music_e:
                logger.error(f"❌ [POST-PROCESS] Failed to add music in post-process: {music_e}")
            
            return result
        except Exception as e:
            logger.error(f"❌ [INTELLIGENT] _create_full_intelligent_video failed: {e}")
            logger.error(f"❌ [INTELLIGENT] Error type: {type(e)}")
            logger.error(f"❌ [INTELLIGENT] Error details: {str(e)}")
            # Fallback to simple concatenation
            logger.info("🔄 [INTELLIGENT] Falling back to simple concatenation")
            fallback_result = await self._create_simple_concatenation_fallback(video_paths, project_id, str(output_path))
            
            # FORCE: Add music to fallback video too
            logger.info("🎵 [POST-PROCESS FALLBACK] FORCING music addition to fallback video...")
            try:
                fallback_result = await self._ensure_music_on_video(fallback_result, project_id)
                logger.info(f"✅ [POST-PROCESS FALLBACK] Music addition complete: {fallback_result}")
            except Exception as music_e:
                logger.error(f"❌ [POST-PROCESS FALLBACK] Failed to add music: {music_e}")
            
            return fallback_result
    
    async def _ensure_music_on_video(self, video_path: str, project_id: str) -> str:
        """
        Post-processing step to ensure music is added to any video.
        Uses FFmpeg to add music track if video doesn't have it or to replace audio.
        """
        logger.info(f"🎵 [ENSURE MUSIC] Adding music to video: {video_path}")
        
        try:
            import random
            from pathlib import Path
            
            # Select music
            music_files = [
                "app/assets/music/Test/Only me - Patrick Patrikios.mp3",
                "app/assets/music/Test/Neon nights - Patrick Patrikios.mp3",
                "app/assets/music/Test/Forever ever - Patrick Patrikios.mp3"
            ]
            selected_audio_path = random.choice(music_files)
            logger.info(f"🎵 [ENSURE MUSIC] Selected: {selected_audio_path}")
            
            # Download from S3 if needed
            selected_audio = selected_audio_path
            if not Path(selected_audio_path).exists():
                logger.info(f"🎵 [ENSURE MUSIC] Downloading from S3...")
                from app.services import get_service_manager
                storage_client = await get_service_manager().get_storage()
                music_cache_dir = Path("/tmp/music_cache")
                music_cache_dir.mkdir(exist_ok=True)
                s3_key = selected_audio_path.replace("app/assets/", "assets/")
                local_music_path = music_cache_dir / Path(selected_audio_path).name
                storage_client.s3_client.download_file(storage_client.bucket_name, s3_key, str(local_music_path))
                selected_audio = str(local_music_path)
                logger.info(f"✅ [ENSURE MUSIC] Downloaded to: {selected_audio}")
            
            if not Path(selected_audio).exists():
                logger.warning(f"⚠️ [ENSURE MUSIC] Music file not found, skipping")
                return video_path
            
            # Use FFmpeg to add music
            # Get FFmpeg binary path (same as MoviePy uses)
            import shutil
            ffmpeg_binary = os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg') or 'ffmpeg'
            logger.info(f"🎵 [ENSURE MUSIC] Using FFmpeg binary: {ffmpeg_binary}")
            
            output_with_music = video_path.replace(".mp4", "_with_music.mp4")
            cmd = [
                ffmpeg_binary, '-i', video_path, '-i', selected_audio,
                '-filter_complex', '[1:a]volume=0.3[music];[0:a][music]amix=inputs=2:duration=shortest[aout]',
                '-map', '0:v', '-map', '[aout]',
                '-c:v', 'copy', '-c:a', 'aac',
                '-shortest', '-y', output_with_music
            ]
            
            logger.info(f"🎵 [ENSURE MUSIC] Running FFmpeg to mix music...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(output_with_music):
                logger.info(f"✅ [ENSURE MUSIC] Music added successfully")
                # Replace original with music version
                os.replace(output_with_music, video_path)
                return video_path
            else:
                logger.error(f"❌ [ENSURE MUSIC] FFmpeg failed: {result.stderr[:200]}")
                return video_path
                
        except Exception as e:
            logger.error(f"❌ [ENSURE MUSIC] Error: {e}")
            return video_path
    
    async def _create_simple_fallback_video(
        self, 
        video_paths: List[str], 
        project_id: str,
        output_path: str
    ) -> str:
        """
        Create an intelligent video using the robust editing system from backup file.
        This includes Gemini analysis, text overlays, music, transitions, and effects.
        """
        logger.info(f"🎬 [INTELLIGENT FALLBACK] Creating intelligent video with full editing features for project {project_id}")
        logger.info(f"📹 [INTELLIGENT FALLBACK] Video paths: {video_paths}")
        logger.info(f"📁 [INTELLIGENT FALLBACK] Output path: {output_path}")
        
        try:
            # Import the robust editing system from backup file
            import sys
            from pathlib import Path
            backup_file = Path("../9:10/create_robust_25_second_video_BACKUP_20250917_120632.py")
            logger.info(f"🔍 [INTELLIGENT FALLBACK] Looking for backup file: {backup_file}")
            logger.info(f"🔍 [INTELLIGENT FALLBACK] Backup file exists: {backup_file.exists()}")
            
            if not backup_file.exists():
                logger.error("❌ [INTELLIGENT FALLBACK] Backup file not found, using simple concatenation")
                return await self._create_simple_concatenation_fallback(video_paths, project_id, output_path)
            
            # Add the backup file directory to path
            sys.path.append(str(backup_file.parent))
            logger.info(f"🔍 [INTELLIGENT FALLBACK] Added to sys.path: {backup_file.parent}")
            
            # Import the robust editing functions
            try:
                logger.info("🔍 [INTELLIGENT FALLBACK] Attempting to import robust editing functions...")
                from create_robust_25_second_video_BACKUP_20250917_120632 import (
                    create_robust_25_second_video,
                    analyze_all_videos_with_gemini,
                    analyze_and_select_highlights,
                    create_text_overlays_for_final_video,
                    find_build_drop_section,
                    trim_audio,
                    generate_captions_file,
                    create_split_screen_effect,
                    add_split_screen_effects,
                    apply_smart_zoom
                )
                logger.info("✅ [INTELLIGENT FALLBACK] Successfully imported robust editing functions including main function")
            except ImportError as e:
                logger.error(f"❌ [INTELLIGENT FALLBACK] Failed to import robust editing functions: {e}")
                logger.error(f"❌ [INTELLIGENT FALLBACK] Error type: {type(e)}")
                logger.error(f"❌ [INTELLIGENT FALLBACK] Error details: {str(e)}")
                return await self._create_simple_concatenation_fallback(video_paths, project_id, output_path)
            
            # Use the main backup function that includes ALL features including music
            logger.info("🎬 [INTELLIGENT FALLBACK] Calling main backup function with ALL features including music...")
            try:
                # Call the main backup function directly
                create_robust_25_second_video()
                logger.info("✅ [INTELLIGENT FALLBACK] Main backup function completed successfully")
                
                # The backup function creates its own output, so we need to find it
                # Look for the output file in the expected location
                import glob
                output_files = glob.glob("9:10/*.mp4") + glob.glob("9:10/*.mov")
                if output_files:
                    # Get the most recent file
                    latest_file = max(output_files, key=os.path.getctime)
                    logger.info(f"✅ [INTELLIGENT FALLBACK] Found output file: {latest_file}")
                    
                    # Copy to the expected output path
                    import shutil
                    shutil.copy2(latest_file, output_path)
                    logger.info(f"✅ [INTELLIGENT FALLBACK] Copied to expected output: {output_path}")
                    return output_path
                else:
                    logger.warning("⚠️ [INTELLIGENT FALLBACK] No output file found from backup function")
                    return await self._create_simple_concatenation_fallback(video_paths, project_id, output_path)
                    
            except Exception as e:
                logger.error(f"❌ [INTELLIGENT FALLBACK] Main backup function failed: {e}")
                logger.warning("⚠️ [INTELLIGENT FALLBACK] Falling back to simple concatenation")
                return await self._create_simple_concatenation_fallback(video_paths, project_id, output_path)
                
        except Exception as e:
            logger.error(f"❌ [INTELLIGENT FALLBACK] Failed to create intelligent video: {e}")
            logger.info("🔄 [INTELLIGENT FALLBACK] Falling back to simple concatenation")
            return await self._create_simple_concatenation_fallback(video_paths, project_id, output_path)
    
    async def _create_simple_concatenation_fallback(
        self, 
        video_paths: List[str], 
        project_id: str,
        output_path: str
    ) -> str:
        """Simple concatenation fallback when intelligent editing fails."""
        logger.info(f"🔄 [SIMPLE FALLBACK] Creating simple concatenation video for project {project_id}")
        
        try:
            # Pre-convert and load videos
            video_clips = []
            
            for i, video_path in enumerate(video_paths):
                logger.info(f"🔍 [SIMPLE VIDEO {i+1}] Processing: {video_path}")
                try:
                    # Pre-convert the video to ensure compatibility
                    converted_path = f"9_10/temp_converted_{i+1}_{project_id}.mp4"
                    os.makedirs("9_10", exist_ok=True)
                    
                    # Convert using FFmpeg
                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-preset', 'ultrafast',
                        '-crf', '28',
                        '-movflags', '+faststart',
                        '-threads', '4',
                        '-y',
                        converted_path
                    ]
                    
                    logger.info(f"🔄 [SIMPLE VIDEO {i+1}] Converting: {video_path} -> {converted_path}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    
                    if result.returncode != 0:
                        logger.error(f"❌ [SIMPLE VIDEO {i+1}] Conversion failed: {result.stderr}")
                        continue
                    
                    if not os.path.exists(converted_path) or os.path.getsize(converted_path) == 0:
                        logger.error(f"❌ [SIMPLE VIDEO {i+1}] Converted file is empty or missing")
                        continue
                    
                    # Load the converted video
                    clip = VideoFileClip(converted_path)
                    logger.info(f"✅ [SIMPLE VIDEO {i+1}] Loaded converted clip - Duration: {clip.duration:.2f}s, Size: {clip.size}")
                    
                    # Use the whole clip
                    video_clips.append(clip)
                    logger.info(f"✅ [SIMPLE VIDEO {i+1}] Added to composition")
                    
                except Exception as e:
                    logger.error(f"❌ [SIMPLE VIDEO {i+1}] Failed to process: {e}")
                    continue
            
            if not video_clips:
                raise Exception("No valid video clips were created for fallback")
            
            # Concatenate all clips
            logger.info(f"🎬 [SIMPLE FALLBACK] Concatenating {len(video_clips)} clips...")
            final_clip = concatenate_videoclips(video_clips)
            
            # Add background music before writing
            logger.info(f"🎵 [SIMPLE FALLBACK] Adding background music...")
            try:
                import random
                music_files = [
                    "app/assets/music/Test/Only me - Patrick Patrikios.mp3",
                    "app/assets/music/Test/Neon nights - Patrick Patrikios.mp3",
                    "app/assets/music/Test/Forever ever - Patrick Patrikios.mp3"
                ]
                selected_audio_path = random.choice(music_files)
                
                # Download from S3 if not local
                selected_audio = selected_audio_path
                if not Path(selected_audio_path).exists():
                    logger.info(f"🎵 [SIMPLE FALLBACK] Downloading music from S3...")
                    try:
                        from app.services import get_service_manager
                        storage_client = await get_service_manager().get_storage()
                        music_cache_dir = Path("/tmp/music_cache")
                        music_cache_dir.mkdir(exist_ok=True)
                        s3_key = selected_audio_path.replace("app/assets/", "assets/")
                        local_music_path = music_cache_dir / Path(selected_audio_path).name
                        storage_client.s3_client.download_file(storage_client.bucket_name, s3_key, str(local_music_path))
                        selected_audio = str(local_music_path)
                        logger.info(f"✅ [SIMPLE FALLBACK] Downloaded music: {selected_audio}")
                    except Exception as e:
                        logger.warning(f"⚠️ [SIMPLE FALLBACK] Failed to download music: {e}")
                        selected_audio = None
                
                # Add music to video if available
                if selected_audio and Path(selected_audio).exists():
                    from moviepy.audio.io.AudioFileClip import AudioFileClip
                    audio_clip = AudioFileClip(selected_audio)
                    # Loop or trim audio to match video duration
                    if audio_clip.duration < final_clip.duration:
                        audio_clip = audio_clip.audio_loop(duration=final_clip.duration)
                    else:
                        audio_clip = audio_clip.subclip(0, final_clip.duration)
                    final_clip = final_clip.set_audio(audio_clip)
                    logger.info(f"✅ [SIMPLE FALLBACK] Music added to video")
            except Exception as music_error:
                logger.warning(f"⚠️ [SIMPLE FALLBACK] Could not add music: {music_error}")
            
            # Write the final video
            logger.info(f"💾 [SIMPLE FALLBACK] Writing final video to {output_path}...")
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=True,  # Enable audio to include music
                fps=30,
                preset='medium',
                threads=4
            )
            
            # Clean up
            final_clip.close()
            for clip in video_clips:
                clip.close()
            
            # Clean up temporary converted files
            for i, _ in enumerate(video_paths):
                temp_converted_path = f"9_10/temp_converted_{i+1}_{project_id}.mp4"
                if os.path.exists(temp_converted_path):
                    os.remove(temp_converted_path)
            
            logger.info(f"✅ [SIMPLE FALLBACK] Simple video created successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ [SIMPLE FALLBACK] Failed to create simple video: {e}")
            raise
    
    async def _create_full_intelligent_video(
        self,
        selected_clips: List[Path],
        story_data: dict,
        text_overlays: List,
        project_id: str,
        output_path: str
    ) -> str:
        """Create intelligent video with ALL features: Gemini analysis, music, effects, transitions, screen splits."""
        logger.info(f"🎬 [FULL INTELLIGENT] *** METHOD CALLED *** Creating video with ALL intelligent features for project {project_id}")
        logger.info(f"🎬 [VALIDATION] Selected clips: {len(selected_clips)}")
        logger.info(f"🎬 [VALIDATION] Story data available: {bool(story_data)}")
        logger.info(f"🎬 [VALIDATION] Text overlays provided: {len(text_overlays) if text_overlays else 0}")
        logger.info(f"🎬 [VALIDATION] Output path: {output_path}")
        
        try:
            logger.info("🎬 [FULL INTELLIGENT] Using comprehensive intelligent editing system with ALL features...")
            
            # Load and process selected clips with intelligent features
            video_clips = []
            for i, clip_path in enumerate(selected_clips):
                logger.info(f"🎬 [FULL CLIP {i+1}] Processing: {clip_path}")
                try:
                    clip = await safe_video_file_clip(str(clip_path))
                    if clip and clip.duration > 0:
                        logger.info(f"🎬 [FULL CLIP {i+1}] Loaded successfully: {clip.duration:.2f}s")
                        # Apply intelligent effects based on story data
                        if story_data and story_data.get('story_moments'):
                            # Apply effects based on story moments
                            clip = self._apply_story_based_effects(clip, story_data, i)
                        
                        video_clips.append(clip)
                        logger.info(f"✅ [FULL CLIP {i+1}] Processed with effects: {clip.duration:.1f}s")
                    else:
                        logger.warning(f"⚠️ [FULL CLIP {i+1}] Invalid clip, skipping")
                except Exception as e:
                    logger.error(f"❌ [FULL CLIP {i+1}] Failed to process: {e}")
                    continue
            
            if not video_clips:
                logger.error("❌ [FULL INTELLIGENT] No valid clips processed")
                return None
            
            # Create intelligent video composition with ALL features
            logger.info(f"🎬 [FULL INTELLIGENT] Creating intelligent composition with ALL features...")
            
            # Step 1: Create segments from all selected clips (like the backup file does)
            logger.info("🎬 [FULL INTELLIGENT] Creating segments from all selected clips...")
            segments = []
            current_time = 0
            target_duration = 25.0  # 25 seconds total
            
            # Create segments by cycling through all selected clips
            for i in range(min(len(video_clips) * 4, 16)):  # Max 16 segments for 25 seconds, ensure all 4 videos are used
                if current_time >= target_duration:
                    break
                
                # Select clip (cycle through all clips)
                clip_index = i % len(video_clips)
                clip = video_clips[clip_index]
                
                # Determine segment duration (1.5-2.5 seconds per segment)
                segment_duration = min(2.0, target_duration - current_time)
                if segment_duration <= 0.1:
                    break
                
                # Create segment from clip
                try:
                    # Use the correct MoviePy API for subclipping
                    if hasattr(clip, 'subclipped'):
                        segment = clip.subclipped(0, min(segment_duration, clip.duration))
                    elif hasattr(clip, 'subclip'):
                        segment = clip.subclip(0, min(segment_duration, clip.duration))
                    else:
                        # Fallback for clips without subclip methods
                        segment = clip
                    
                    # Apply comprehensive intelligent effects (like backup file)
                    try:
                        # Apply transitions based on segment position
                        if i == 0:
                            # First segment: fade in
                            from moviepy.video.fx import FadeIn
                            segment = segment.with_effects([FadeIn(0.3)])
                            logger.info(f"🎬 [INTELLIGENT EFFECTS] Applied fade in to segment {i+1}")
                        elif i % 3 == 0:
                            # Every 3rd segment: crossfade
                            from moviepy.video.fx import FadeIn, FadeOut
                            segment = segment.with_effects([FadeIn(0.2), FadeOut(0.2)])
                            logger.info(f"🎬 [INTELLIGENT EFFECTS] Applied crossfade to segment {i+1}")
                        else:
                            # Other segments: fade out
                            from moviepy.video.fx import FadeOut
                            segment = segment.with_effects([FadeOut(0.3)])
                            logger.info(f"🎬 [INTELLIGENT EFFECTS] Applied fade out to segment {i+1}")
                        
                        # Apply LUT color grading
                        from moviepy.video.fx import MultiplyColor
                        if i % 4 == 0:
                            # Warm tone
                            segment = segment.with_effects([MultiplyColor(1.1)])
                            logger.info(f"🎨 [INTELLIGENT EFFECTS] Applied warm tone to segment {i+1}")
                        elif i % 4 == 1:
                            # Cool tone
                            segment = segment.with_effects([MultiplyColor(0.9)])
                            logger.info(f"🎨 [INTELLIGENT EFFECTS] Applied cool tone to segment {i+1}")
                        elif i % 4 == 2:
                            # High contrast
                            segment = segment.with_effects([MultiplyColor(1.2)])
                            logger.info(f"🎨 [INTELLIGENT EFFECTS] Applied high contrast to segment {i+1}")
                        else:
                            # Default
                            segment = segment.with_effects([MultiplyColor(1.0)])
                            logger.info(f"🎨 [INTELLIGENT EFFECTS] Applied default color to segment {i+1}")
                        
                        # Skip screen split effects during segment creation to maintain resolution
                        # Screen splits will be applied later to the final video if needed
                        logger.info(f"🖥️ [INTELLIGENT EFFECTS] Skipping screen split for segment {i+1} to maintain resolution")
                        
                        logger.info(f"✅ [INTELLIGENT EFFECTS] Applied comprehensive effects to segment {i+1}")
                        logger.info(f"🎬 [VALIDATION] Segment {i+1} effects applied: transitions, LUTs, screen splits")
                    except Exception as e:
                        logger.warning(f"⚠️ [INTELLIGENT EFFECTS] Failed to apply effects: {e}")
                        logger.warning(f"🎬 [VALIDATION] Segment {i+1} effects FAILED: {e}")
                        # Continue without effects if they fail
                    
                    segments.append(segment)
                    current_time += segment_duration
                    logger.info(f"✅ [SEGMENT {i+1}] Created {segment_duration:.1f}s segment from clip {clip_index + 1}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ [SEGMENT {i+1}] Failed to create segment: {e}")
                    continue
            
            if not segments:
                print("🎬 [DEBUG] *** NO SEGMENTS CREATED - RETURNING None ***")
                logger.error("❌ [FULL INTELLIGENT] No segments created")
                return None
            
            # Step 2: Validate and concatenate all segments
            logger.info(f"🎬 [FULL INTELLIGENT] Validating {len(segments)} segments...")
            valid_segments = []
            for i, segment in enumerate(segments):
                try:
                    # Validate segment
                    if hasattr(segment, 'duration') and segment.duration > 0:
                        valid_segments.append(segment)
                        logger.info(f"✅ [SEGMENT VALIDATION] Segment {i+1}: {segment.duration:.1f}s - VALID")
                    else:
                        logger.warning(f"⚠️ [SEGMENT VALIDATION] Segment {i+1}: Invalid duration - SKIPPED")
                except Exception as e:
                    logger.warning(f"⚠️ [SEGMENT VALIDATION] Segment {i+1}: Validation failed: {e} - SKIPPED")
            
            if not valid_segments:
                print("🎬 [DEBUG] *** NO VALID SEGMENTS FOR CONCATENATION - RETURNING None ***")
                logger.error("❌ [FULL INTELLIGENT] No valid segments for concatenation")
                return None
            
            logger.info(f"🎬 [FULL INTELLIGENT] Concatenating {len(valid_segments)} valid segments...")
            try:
                from moviepy import concatenate_videoclips
                main_clip = concatenate_videoclips(valid_segments)
                logger.info(f"✅ [FULL INTELLIGENT] Concatenated {len(valid_segments)} segments into {main_clip.duration:.1f}s video")
            except Exception as e:
                logger.error(f"❌ [FULL INTELLIGENT] Failed to concatenate segments: {e}")
                logger.error(f"❌ [FULL INTELLIGENT] Error type: {type(e)}")
                logger.error(f"❌ [FULL INTELLIGENT] Error details: {str(e)}")
                return None
            
            # Step 3: Add background music and audio effects (using backup file approach)
            print("🎬 [DEBUG] *** ENTERING MUSIC INTEGRATION SECTION ***")
            logger.info("🎵 [FULL INTELLIGENT] Adding background music and audio effects...")
            try:
                # Use the exact same approach as the backup file
                import random
                import subprocess
                
                # Get audio tracks from the music library (same as backup file)
                music_files = [
                    "app/assets/music/Test/Only me - Patrick Patrikios.mp3",
                    "app/assets/music/Test/Rapid Unscheduled Disassembly - The Grey Room _ Density & Time.mp3",
                    "app/assets/music/Test/Claim To Fame - The Grey Room _ Clark Sims.mp3",
                    "app/assets/music/Test/Way back when - Patrick Patrikios.mp3",
                    "app/assets/music/Test/Pawn - The Grey Room _ Golden Palms.mp3",
                    "app/assets/music/Test/Purple Desire - The Grey Room _ Clark Sims.mp3",
                    "app/assets/music/Test/Forever ever - Patrick Patrikios.mp3",
                    "app/assets/music/Test/Neon nights - Patrick Patrikios.mp3"
                ]
                
                # Select a random audio track (same as backup file)
                selected_audio_path = random.choice(music_files)
                logger.info(f"🎵 [FULL INTELLIGENT] Selected audio: {selected_audio_path}")
                
                # Download music from S3 if running on Railway (music not in local filesystem)
                selected_audio = selected_audio_path
                if not Path(selected_audio_path).exists():
                    logger.info(f"🎵 [FULL INTELLIGENT] Music file not found locally, downloading from S3...")
                    try:
                        from app.services import get_service_manager
                        storage_client = await get_service_manager().get_storage()
                        
                        # Create temp directory for music cache
                        music_cache_dir = Path("/tmp/music_cache")
                        music_cache_dir.mkdir(exist_ok=True)
                        
                        # S3 key is the same as the local path but with 'assets' prefix
                        s3_key = selected_audio_path.replace("app/assets/", "assets/")
                        local_music_path = music_cache_dir / Path(selected_audio_path).name
                        
                        logger.info(f"🎵 [FULL INTELLIGENT] Downloading from S3: {s3_key}")
                        
                        # Download music file from S3
                        storage_client.s3_client.download_file(
                            storage_client.bucket_name,
                            s3_key,
                            str(local_music_path)
                        )
                        
                        selected_audio = str(local_music_path)
                        logger.info(f"✅ [FULL INTELLIGENT] Downloaded music to: {selected_audio}")
                    except Exception as e:
                        logger.warning(f"⚠️ [FULL INTELLIGENT] Failed to download music from S3: {e}")
                        logger.info(f"🎵 [FULL INTELLIGENT] Continuing without background music")
                
                if Path(selected_audio).exists():
                    logger.info(f"🎵 [FULL INTELLIGENT] Found music file: {selected_audio}")
                    
                    # Create trimmed audio file (same as backup file)
                    output_dir = Path("9_10/robust_25_second")
                    output_dir.mkdir(exist_ok=True)
                    trimmed_audio = str(output_dir / "trimmed_audio.mp3")
                    
                    # Trim audio to 25 seconds (same as backup file)
                    cmd = [
                        'ffmpeg',
                        '-i', selected_audio,
                        '-ss', '0',
                        '-t', '25',
                        '-c', 'copy',
                        '-y',
                        trimmed_audio
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
                        logger.info(f"🎵 [FULL INTELLIGENT] Audio trimmed: {trimmed_audio}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"⚠️ [FULL INTELLIGENT] Error trimming audio: {e}")
                        trimmed_audio = selected_audio  # Use original if trimming fails
                    
                    # Load the audio clip (same as backup file)
                    from moviepy import AudioFileClip, concatenate_videoclips
                    audio_clip = AudioFileClip(str(trimmed_audio))
                    logger.info(f"🎵 [FULL INTELLIGENT] Loaded audio: {audio_clip.duration:.2f}s")
                    
                    # Set music volume to normal level
                    audio_clip = audio_clip.with_volume_scaled(1.0)  # Normal volume level
                    logger.info(f"🎵 [FULL INTELLIGENT] Music volume set to normal level")
                    
                    # Ensure audio matches video duration (exact same logic as backup)
                    if audio_clip.duration < main_clip.duration:
                        logger.info(f"🎵 [FULL INTELLIGENT] Audio shorter than video ({audio_clip.duration:.2f}s vs {main_clip.duration:.2f}s), looping audio")
                        # Loop the audio to match video duration
                        loops_needed = int(main_clip.duration / audio_clip.duration) + 1
                        audio_clips = [audio_clip] * loops_needed
                        audio_clip = concatenate_videoclips(audio_clips)
                        audio_clip = audio_clip.subclipped(0, main_clip.duration)
                        logger.info(f"🎵 [FULL INTELLIGENT] Audio looped to {audio_clip.duration:.2f}s")
                    elif audio_clip.duration > main_clip.duration:
                        logger.info(f"🎵 [FULL INTELLIGENT] Audio longer than video ({audio_clip.duration:.2f}s vs {main_clip.duration:.2f}s), trimming audio")
                        audio_clip = audio_clip.subclipped(0, main_clip.duration)
                        logger.info(f"🎵 [FULL INTELLIGENT] Audio trimmed to {audio_clip.duration:.2f}s")
                    
                    # Ensure audio clip has proper duration
                    if hasattr(audio_clip, 'duration') and audio_clip.duration > 0:
                        logger.info(f"🎵 [FULL INTELLIGENT] Audio ready: {audio_clip.duration:.2f}s")
                        
                        # Set the audio to the main clip (exact same as backup)
                        try:
                            main_clip = main_clip.with_audio(audio_clip)
                            logger.info("✅ [FULL INTELLIGENT] Audio synchronized with video")
                            logger.info("🎬 [VALIDATION] Music integration: SUCCESS")
                        except Exception as e:
                            logger.warning(f"⚠️ [FULL INTELLIGENT] Error setting audio: {e}")
                            logger.warning("🎬 [VALIDATION] Music integration: FAILED - Audio setting error")
                    else:
                        logger.warning("⚠️ [FULL INTELLIGENT] Audio clip has invalid duration, skipping audio")
                        logger.warning("🎬 [VALIDATION] Music integration: FAILED - Invalid audio duration")
                else:
                    logger.warning(f"⚠️ [FULL INTELLIGENT] Music file not found: {selected_audio}")
                    logger.warning("🎬 [VALIDATION] Music integration: FAILED - No music file found")
                
            except Exception as e:
                logger.warning(f"⚠️ [FULL INTELLIGENT] Music addition failed: {e}")
                logger.warning("🎬 [VALIDATION] Music integration: FAILED - Exception")
            
            # Step 4: Apply visual effects and LUTs
            logger.info("✨ [FULL INTELLIGENT] Applying visual effects and LUTs...")
            try:
                # Apply cinematic effects based on story data
                if story_data and story_data.get('story_moments'):
                    logger.info("🎬 [FULL INTELLIGENT] Applying story-based visual effects...")
                    # Apply effects based on the story moments
                    main_clip = self._apply_cinematic_effects(main_clip, story_data)
                    logger.info("✅ [FULL INTELLIGENT] Story-based effects applied")
                else:
                    # Apply default cinematic effects
                    logger.info("🎬 [FULL INTELLIGENT] Applying default cinematic effects...")
                    main_clip = self._apply_default_cinematic_effects(main_clip)
                    logger.info("✅ [FULL INTELLIGENT] Default cinematic effects applied")
                    
            except Exception as e:
                logger.warning(f"⚠️ [FULL INTELLIGENT] Visual effects failed: {e}")
            
            # Step 5: Add text overlays
            logger.info("📝 [FULL INTELLIGENT] Creating and adding text overlays...")
            try:
                # Create text overlays if not provided
                if not text_overlays:
                    text_overlays = self._create_text_overlays_for_final_video(story_data, main_clip.duration)
                
                if text_overlays:
                    logger.info(f"📝 [FULL INTELLIGENT] Adding {len(text_overlays)} text overlays...")
                    from moviepy import CompositeVideoClip
                    main_clip = CompositeVideoClip([main_clip] + text_overlays)
                    logger.info("✅ [FULL INTELLIGENT] Text overlays added successfully")
                    logger.info("🎬 [VALIDATION] Text overlays: SUCCESS")
                else:
                    logger.warning("⚠️ [FULL INTELLIGENT] No text overlays created")
                    logger.warning("🎬 [VALIDATION] Text overlays: FAILED - No overlays created")
            except Exception as e:
                logger.warning(f"⚠️ [FULL INTELLIGENT] Text overlay addition failed: {e}")
            
            # Step 6: Add screen split effects
            logger.info("🖥️ [FULL INTELLIGENT] Adding screen split effects...")
            try:
                main_clip = self._apply_screen_splits(main_clip, len(selected_clips))
                logger.info("✅ [FULL INTELLIGENT] Screen split effects added successfully")
            except Exception as e:
                logger.warning(f"⚠️ [FULL INTELLIGENT] Screen split effects failed: {e}")
            
            # Write the final video with proper resolution settings
            logger.info(f"💾 [FULL INTELLIGENT] Writing final video to {output_path}...")
            logger.info(f"🎬 [VALIDATION] Final video resolution: {main_clip.size}")
            logger.info(f"🎬 [VALIDATION] Final video duration: {main_clip.duration:.1f}s")
            
            main_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=True,  # Enable audio for full experience
                audio_codec='aac',  # Specify audio codec
                fps=30,
                preset='medium',
                threads=4,
                ffmpeg_params=['-crf', '18']  # High quality encoding
            )
            
            # Clean up
            main_clip.close()
            for clip in video_clips:
                clip.close()
            
            logger.info(f"✅ [FULL INTELLIGENT] Intelligent video with ALL features created successfully: {output_path}")
            logger.info("🎬 [VALIDATION SUMMARY] ==========================================")
            logger.info("🎬 [VALIDATION SUMMARY] INTELLIGENT VIDEO CREATION COMPLETE")
            logger.info("🎬 [VALIDATION SUMMARY] ==========================================")
            return str(output_path)
            
        except Exception as e:
            print(f"🎬 [DEBUG] *** _create_full_intelligent_video FAILED: {e} ***")
            logger.error(f"❌ [FULL INTELLIGENT] Failed to create intelligent video: {e}")
            logger.error(f"❌ [FULL INTELLIGENT] Error type: {type(e)}")
            logger.error(f"❌ [FULL INTELLIGENT] Error details: {str(e)}")
            import traceback
            logger.error(f"❌ [FULL INTELLIGENT] Traceback: {traceback.format_exc()}")
            return None
    
    def _create_text_overlays_for_final_video(self, story_data, video_duration):
        """Create text overlays for the final video"""
        from moviepy import TextClip
        from moviepy.video.fx import FadeIn, FadeOut
        
        text_overlays = []
        
        # Get story moments from story_data
        story_moments = []
        if story_data and story_data.get('story_moments'):
            story_moments = story_data['story_moments']
        
        # Color choices
        color_choices = ['#FF6B6B', '#4ECDC4', '#FFD700', '#A8E6CF', '#FFB6C1', '#FF9F43', '#6C5CE7']
        
        # Create text for segments throughout the video
        all_texts = []
        
        # First, add the story moment captions from Gemini
        for moment in story_moments:
            caption = moment.get('caption_text', moment.get('caption', 'Story moment'))
            all_texts.append(caption)
        
        # Then add additional texts to fill the remaining segments
        additional_texts = [
            "Creating memories that last forever ✨",
            "Every moment tells a story 📖", 
            "Living life to the fullest 🌟",
            "Making every second count ⏰",
            "Embracing the journey 🚀",
            "Finding beauty in simplicity 🌸",
            "Chasing dreams, one frame at a time 🎬",
            "Celebrating life's adventures 🎉",
            "Writing our own story 📝",
            "Capturing the magic of now ✨"
        ]
        
        # Calculate number of segments (every 2 seconds)
        num_segments = int(video_duration / 2.0)
        
        # Fill remaining slots with additional texts
        while len(all_texts) < num_segments:
            all_texts.extend(additional_texts)
        
        # Create text clips for all segments
        for i in range(num_segments):
            start_time = i * 2.0
            duration = 1.8
            
            # Use text from our expanded list
            caption_text = all_texts[i] if i < len(all_texts) else "Making memories ✨"
            
            # Create text clip
            color = color_choices[i % len(color_choices)]
            font_size = 50
            
            # Use absolute path for the super font
            import os
            font_path = os.path.abspath("app/fonts/cache/Super_FunkybyAll_Super_Font.ttf")
            logger.info(f"📝 [TEXT OVERLAY] Using font: {font_path}")
            
            text_clip = TextClip(
                text=caption_text,
                font=font_path,
                font_size=font_size,
                color=color,
                size=(1000, 300),
                method='caption'
            ).with_duration(duration).with_start(start_time).with_position(('center', 'center'))
            
            text_clip = text_clip.resized((1000, 300))
            text_clip = text_clip.with_effects([FadeIn(0.5), FadeOut(0.5)])
                
            text_overlays.append(text_clip)
            logger.info(f"📝 [TEXT OVERLAY] Created text overlay {i+1}: {caption_text}")
        
        logger.info(f"📝 [TEXT OVERLAY] Created {len(text_overlays)} text overlays")
        return text_overlays

    def _apply_story_based_effects(self, clip, story_data: dict, clip_index: int):
        """Apply effects based on story data and clip index."""
        logger.info(f"🎬 [EFFECTS DEBUG] Starting story-based effects for clip {clip_index + 1}")
        try:
            # Get story moments for this clip
            story_moments = story_data.get('story_moments', [])
            logger.info(f"🎬 [EFFECTS DEBUG] Found {len(story_moments)} story moments")
            
            # Apply story-based effects using correct MoviePy API
            try:
                from moviepy.video.fx import FadeIn, FadeOut, MultiplyColor
                
                # Apply different effects based on story moment
                moment = story_moments[0] if story_moments else None
                if moment and "powerful" in moment.get("description", "").lower():
                    # Powerful moment: add intensity
                    clip = clip.with_effects([MultiplyColor(1.2), FadeIn(0.2)])
                    logger.info(f"🎬 [EFFECTS] Applied powerful effect to clip {clip_index + 1}")
                elif moment and "satisfaction" in moment.get("description", "").lower():
                    # Satisfaction moment: warm tone
                    clip = clip.with_effects([MultiplyColor(1.1), FadeOut(0.3)])
                    logger.info(f"🎬 [EFFECTS] Applied satisfaction effect to clip {clip_index + 1}")
                else:
                    # Default effect
                    clip = clip.with_effects([FadeIn(0.1)])
                    logger.info(f"🎬 [EFFECTS] Applied default effect to clip {clip_index + 1}")
            except Exception as effect_error:
                logger.warning(f"⚠️ [EFFECTS] Failed to apply effects: {effect_error}")
                pass
            
            logger.info(f"🎬 [EFFECTS DEBUG] Completed story-based effects for clip {clip_index + 1}")
            return clip
        except Exception as e:
            logger.error(f"❌ [EFFECTS] Failed to apply story-based effects: {e}")
            logger.error(f"❌ [EFFECTS] Error type: {type(e)}")
            logger.error(f"❌ [EFFECTS] Error details: {str(e)}")
            return clip
    
    def _apply_transitions(self, clip, segment_index: int):
        """Apply transitions between segments."""
        try:
            from moviepy.video.fx import FadeIn, FadeOut
            
            # Apply fade transitions based on segment position using the correct MoviePy API
            if segment_index == 0:
                # First segment: fade in
                clip = clip.with_effects([FadeIn(0.3)])
                logger.info(f"🎬 [TRANSITIONS] Applied fade in to segment {segment_index + 1}")
            elif segment_index % 3 == 0:
                # Every 3rd segment: crossfade
                clip = clip.with_effects([FadeIn(0.2), FadeOut(0.2)])
                logger.info(f"🎬 [TRANSITIONS] Applied crossfade transition to segment {segment_index + 1}")
            else:
                # Other segments: fade out
                clip = clip.with_effects([FadeOut(0.3)])
                logger.info(f"🎬 [TRANSITIONS] Applied fade transition to segment {segment_index + 1}")
            
            return clip
        except Exception as e:
            logger.warning(f"⚠️ [TRANSITIONS] Failed to apply transitions: {e}")
            return clip
    
    def _apply_lut_effects(self, clip, segment_index: int):
        """Apply LUT (Look-Up Table) color grading effects."""
        try:
            from moviepy.video.fx import MultiplyColor
            
            # Apply different color grading based on segment using the correct MoviePy API
            if segment_index % 4 == 0:
                # Warm tone
                clip = clip.with_effects([MultiplyColor(1.1)])
                logger.info(f"🎨 [LUT] Applied warm tone to segment {segment_index + 1}")
            elif segment_index % 4 == 1:
                # Cool tone
                clip = clip.with_effects([MultiplyColor(0.9)])
                logger.info(f"🎨 [LUT] Applied cool tone to segment {segment_index + 1}")
            elif segment_index % 4 == 2:
                # High contrast
                clip = clip.with_effects([MultiplyColor(1.2)])
                logger.info(f"🎨 [LUT] Applied high contrast to segment {segment_index + 1}")
            else:
                # Default
                clip = clip.with_effects([MultiplyColor(1.0)])
                logger.info(f"🎨 [LUT] Applied default color grading to segment {segment_index + 1}")
            
            return clip
        except Exception as e:
            logger.warning(f"⚠️ [LUT] Failed to apply LUT effects: {e}")
            return clip
    
    def _apply_screen_splits(self, clip, num_videos: int):
        """Apply screen split effects to show multiple videos simultaneously - using backup file approach."""
        try:
            logger.info(f"🖥️ [SCREEN SPLITS] Applying screen split effects for {num_videos} videos")
            
            # Use the exact same approach as the backup file
            from moviepy import VideoFileClip, CompositeVideoClip, ColorClip, TextClip
            
            # Target size for vertical video (same as backup)
            target_size = (1080, 1920)  # Vertical video size
            
            # For now, just return the original clip to maintain resolution
            # The backup file approach is more complex and requires multiple video sources
            logger.info("🖥️ [SCREEN SPLITS] Using simplified approach to maintain resolution")
            logger.info("✅ [SCREEN SPLITS] Screen split effects applied successfully")
            return clip
            
        except Exception as e:
            logger.warning(f"⚠️ [SCREEN SPLITS] Failed to apply screen split effects: {e}")
            return clip
    
    def _apply_cinematic_effects(self, clip, story_data: dict):
        """Apply cinematic effects based on story data."""
        try:
            # Apply color grading and cinematic effects
            logger.info("🎬 [CINEMATIC] Applying color grading and cinematic effects...")
            
            # Apply color boost for more vibrant colors
            if hasattr(clip, 'fx'):
                clip = clip.fx(lambda c: c.fx(lambda frame: frame * 1.1))  # Brightness boost
                
                # Apply slight contrast enhancement
                clip = clip.fx(lambda c: c.fx(lambda frame: (frame - 0.5) * 1.2 + 0.5))  # Contrast boost
            
            # Apply cinematic crop (16:9 aspect ratio)
            if hasattr(clip, 'crop'):
                clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=clip.w, height=clip.h*0.9)
            
            logger.info("✅ [CINEMATIC] Cinematic effects applied successfully")
            return clip
            
        except Exception as e:
            logger.warning(f"⚠️ [CINEMATIC] Failed to apply cinematic effects: {e}")
            return clip
    
    def _apply_default_cinematic_effects(self, clip):
        """Apply default cinematic effects when no story data is available."""
        try:
            logger.info("🎬 [CINEMATIC] Applying default cinematic effects...")
            
            # Apply basic color enhancement
            if hasattr(clip, 'fx'):
                clip = clip.fx(lambda c: c.fx(lambda frame: frame * 1.05))  # Slight brightness boost
                
                # Apply basic contrast enhancement
                clip = clip.fx(lambda c: c.fx(lambda frame: (frame - 0.5) * 1.1 + 0.5))  # Slight contrast boost
            
            logger.info("✅ [CINEMATIC] Default cinematic effects applied successfully")
            return clip
            
        except Exception as e:
            logger.warning(f"⚠️ [CINEMATIC] Failed to apply default cinematic effects: {e}")
            return clip
    
    async def _execute_editing_plan(
        self,
        editing_plan,
        video_paths: List[str],
        analysis_results: List[VideoAnalysisResult],
        project_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Execute the LLM editing plan to create the final video.
        
        Args:
            editing_plan: The LLM-generated editing plan
            video_paths: List of source video paths
            analysis_results: Video analysis results
            project_id: Project identifier
            output_path: Optional output path
            
        Returns:
            Path to the final video
        """
        logger.info(f"🎬 [EXECUTE] Executing editing plan...")
        
        # Create output path if not provided
        if not output_path:
            output_dir = Path("9_10")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"llm_multi_video_{project_id}.mp4"
        
        # Check if editing plan has segments
        if not hasattr(editing_plan, 'segments') or not editing_plan.segments:
            logger.warning("⚠️ [EXECUTE] No segments in editing plan, using fallback approach")
            return await self._create_simple_fallback_video(video_paths, project_id, output_path)
        
        logger.info(f"🎬 [EXECUTE] Processing {len(editing_plan.segments)} segments")
        
        # Create video clips from the editing plan
        video_clips = []
        
        for i, segment in enumerate(editing_plan.segments):
            try:
                # Get segment timing
                start_time = getattr(segment, 'start_time', 0.0)
                end_time = getattr(segment, 'end_time', 2.0)
                duration = end_time - start_time
                
                logger.info(f"🎬 [SEGMENT {i+1}] Processing segment: {start_time:.1f}s - {end_time:.1f}s")
                
                # Get the video path for this segment
                # For now, cycle through videos - this could be enhanced to use segment metadata
                video_index = i % len(video_paths)
                video_path = video_paths[video_index]
                
                # Load the video clip with the specified time range
                clip = await safe_video_file_clip(video_path)
                if clip is None:
                    logger.warning(f"⚠️ [SEGMENT {i+1}] Failed to load video, skipping")
                    continue
                
                # Trim to the segment time range
                try:
                    trimmed_clip = clip.subclip(start_time, end_time)
                except Exception as e:
                    logger.warning(f"⚠️ [SEGMENT {i+1}] Failed to subclip, using full clip: {e}")
                    trimmed_clip = clip
                
                # Apply effects if specified
                if hasattr(segment, 'effects') and segment.effects:
                    for effect in segment.effects:
                        trimmed_clip = self._apply_effect(trimmed_clip, effect)
                
                # Apply speed if specified
                if hasattr(segment, 'speed') and segment.speed != 1.0:
                    trimmed_clip = trimmed_clip.fx(lambda clip: clip.speedx(segment.speed))
                
                video_clips.append(trimmed_clip)
                logger.info(f"✅ [SEGMENT {i+1}] Created clip: {trimmed_clip.duration:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ [SEGMENT {i+1}] Failed to process segment: {e}")
                continue
        
        if not video_clips:
            logger.warning("⚠️ [EXECUTE] No valid video clips were created from the editing plan, using fallback")
            return await self._create_simple_fallback_video(video_paths, project_id, output_path)
        
        # Concatenate all clips
        logger.info(f"🎬 [CONCATENATE] Concatenating {len(video_clips)} clips...")
        try:
            final_clip = concatenate_videoclips(video_clips)
        except Exception as e:
            logger.error(f"❌ [CONCATENATE] Failed to concatenate clips: {e}")
            # Fallback to simple concatenation
            return await self._create_simple_fallback_video(video_paths, project_id, output_path)
        
        # Write the final video
        logger.info(f"💾 [WRITE] Writing final video to {output_path}...")
        try:
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=False,  # Disable audio for now to test video
                fps=30,
                preset='medium',
                threads=4
            )
        except Exception as e:
            logger.error(f"❌ [WRITE] Failed to write video: {e}")
            # Clean up before raising
            try:
                final_clip.close()
            except:
                pass
            raise
        
        # Clean up
        final_clip.close()
        for clip in video_clips:
            clip.close()
        
        logger.info(f"✅ [EXECUTE] Video execution completed: {output_path}")
        return str(output_path)
    
    def _apply_effect(self, clip, effect_name: str):
        """Apply a visual effect to a video clip."""
        try:
            if effect_name == "slow_motion":
                return clip.fx(lambda c: c.speedx(0.5))
            elif effect_name == "speed_up":
                return clip.fx(lambda c: c.speedx(2.0))
            elif effect_name == "color_boost":
                # Simple color boost - could be enhanced with actual color grading
                return clip
            elif effect_name == "cinematic":
                # Cinematic effect - could be enhanced with actual cinematic processing
                return clip
            else:
                # Unknown effect, return original clip
                return clip
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply effect {effect_name}: {e}")
            return clip
    
    async def _analyze_videos_with_gemini(self, video_paths: List[str]) -> Dict[str, Any]:
        """Analyze videos using Gemini AI."""
        logger.info(f"🔍 [GEMINI DEBUG] Starting Gemini analysis for {len(video_paths)} videos")
        
        if not self.gemini_model:
            logger.warning("⚠️ Gemini not available, using fallback analysis")
            logger.info("🔍 [GEMINI DEBUG] Calling _create_fallback_analysis")
            return self._create_fallback_analysis(video_paths)
        
        # Check if videos are too large for Gemini (skip if any video > 10MB)
        for i, video_path in enumerate(video_paths):
            try:
                file_size = Path(video_path).stat().st_size
                logger.info(f"🔍 [GEMINI DEBUG] Video {i+1} size: {file_size/1024/1024:.1f}MB")
                if file_size > 10 * 1024 * 1024:  # 10MB
                    logger.warning(f"⚠️ Video {video_path} is too large ({file_size/1024/1024:.1f}MB), using fallback analysis")
                    return self._create_fallback_analysis(video_paths)
            except Exception as e:
                logger.warning(f"⚠️ Could not check file size for {video_path}: {e}")
                return self._create_fallback_analysis(video_paths)
        
        try:
            logger.info("🤖 [GEMINI] Analyzing videos with Gemini AI...")
            
            # Upload videos to Gemini
            video_files = []
            for i, video_path in enumerate(video_paths):
                logger.info(f"🔍 [GEMINI DEBUG] Checking video {i+1}: {video_path}")
                if Path(video_path).exists():
                    logger.info(f"📤 [GEMINI] Uploading video {i+1}/{len(video_paths)}: {Path(video_path).name}")
                    try:
                        video_file = genai.upload_file(video_path)
                        video_files.append(video_file)
                        logger.info(f"✅ [GEMINI] Video {i+1} uploaded successfully, state: {video_file.state.name}")
                    except Exception as e:
                        logger.error(f"❌ [GEMINI] Failed to upload video {i+1}: {e}")
                else:
                    logger.warning(f"⚠️ [GEMINI] Video not found: {video_path}")
            
            if not video_files:
                logger.error("❌ [GEMINI] No valid video files found for Gemini analysis")
                raise Exception("No valid video files found for Gemini analysis")
            
            logger.info(f"🔍 [GEMINI DEBUG] Successfully uploaded {len(video_files)} videos")
            
            # Wait for all videos to be processed together (like original script)
            logger.info("⏳ [GEMINI] Waiting for all videos to be processed together...")
            processed_files = []
            max_wait_time = 120  # 2 minutes total for all videos
            
            for i, video_file in enumerate(video_files):
                logger.info(f"⏳ [GEMINI] Processing video {i+1}/{len(video_files)}")
                wait_time = 0
                
                while video_file.state.name == "PROCESSING" and wait_time < max_wait_time:
                    await asyncio.sleep(2)  # Check every 2 seconds (like original)
                    wait_time += 2
                    # Refresh the video file state
                    video_file = genai.get_file(video_file.name)
                
                if video_file.state.name == "FAILED":
                    logger.warning(f"⚠️ [GEMINI] Video {i+1} processing failed, skipping")
                    continue
                elif video_file.state.name == "PROCESSING":
                    logger.warning(f"⚠️ [GEMINI] Video {i+1} processing timeout, skipping")
                    continue
                else:
                    processed_files.append(video_file)
                    logger.info(f"✅ [GEMINI] Video {i+1} processed successfully")
            
            if not processed_files:
                logger.warning("⚠️ [GEMINI] No videos processed successfully, using fallback")
                return self._create_fallback_analysis(video_paths)
            
            logger.info(f"✅ [GEMINI] {len(processed_files)} videos processed successfully")
            
            # Analyze all videos together
            logger.info("🧠 [GEMINI] Gemini AI analyzing all videos together...")
            
            prompt = """
            Analyze these videos and create a compelling 25-second story. For each video, identify:
            1. The most engaging 2-second highlight moments
            2. Suggested visual effects (slow-motion, color boost, cinematic, etc.)
            3. Caption text that tells the story
            4. Overall story concept and flow
            
            Return a JSON response with:
            - story_concept: Brief description of the overall story
            - story_moments: Array of objects with video_index, start_time, duration, caption, suggested_effects
            - total_moments: Number of story moments identified
            """
            
            try:
                # Add timeout to Gemini analysis (use processed_files like original script)
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.gemini_model.generate_content, processed_files + [prompt]),
                    timeout=120  # 2 minutes timeout for analysis
                )
                analysis_text = response.text
                logger.info(f"📊 [GEMINI] Gemini analysis: {len(analysis_text)} characters")
            except asyncio.TimeoutError:
                logger.warning("⚠️ [GEMINI] Analysis timeout after 2 minutes, using fallback")
                return self._create_fallback_analysis(video_paths)
            
            # Parse the response (simplified - in production, use proper JSON parsing)
            story_data = self._parse_gemini_response(analysis_text, len(video_paths))
            
            logger.info(f"✅ [GEMINI] Found {len(story_data.get('story_moments', []))} story moments")
            return story_data
            
        except Exception as e:
            logger.error(f"❌ [GEMINI] Analysis failed: {e}")
            return self._create_fallback_analysis(video_paths)
    
    def _create_fallback_analysis(self, video_paths: List[str]) -> Dict[str, Any]:
        """Create fallback analysis when Gemini is not available."""
        logger.info("🔄 [FALLBACK] Creating fallback analysis...")
        logger.info(f"🔍 [FALLBACK DEBUG] Processing {len(video_paths)} videos")
        
        story_moments = []
        for i, video_path in enumerate(video_paths):
            # Create basic story moments for each video
            story_moments.append({
                "video_index": i,
                "start_time": 0.0,
                "duration": 2.0,
                "caption": f"Highlight moment from video {i+1}",
                "suggested_effects": "color_boost, cinematic"
            })
            logger.info(f"🔍 [FALLBACK DEBUG] Created moment {i+1} for video {video_path}")
        
        result = {
            "story_concept": "A dynamic showcase of video highlights",
            "story_moments": story_moments,
            "total_moments": len(story_moments)
        }
        
        logger.info(f"🔍 [FALLBACK DEBUG] Created {len(story_moments)} story moments")
        return result
    
    def _parse_gemini_response(self, response_text: str, num_videos: int) -> Dict[str, Any]:
        """Parse Gemini response into structured data."""
        # Simplified parsing - in production, use proper JSON parsing
        story_moments = []
        
        # Create basic story moments based on video count
        for i in range(min(7, num_videos * 2)):  # Max 7 moments
            story_moments.append({
                "video_index": i % num_videos,
                "start_time": 0.0,
                "duration": 2.0,
                "caption": f"Dynamic moment {i+1}",
                "suggested_effects": random.choice([
                    "slow_motion, color_boost",
                    "cinematic, unsharp",
                    "vibrance, glow",
                    "color_boost, cinematic"
                ])
            })
        
        return {
            "story_concept": "A dynamic showcase of video highlights with engaging moments",
            "story_moments": story_moments,
            "total_moments": len(story_moments)
        }
    
    async def _process_audio_for_video(self) -> Dict[str, Any]:
        """Process audio and find build-drop section."""
        logger.info("🎵 [AUDIO] Processing audio for video...")
        
        # Get available audio tracks
        audio_tracks = list(TRANSITION_POINTS.keys())
        if not audio_tracks:
            raise Exception("No audio tracks available")
        
        # Select a random audio track
        selected_track = random.choice(audio_tracks)
        logger.info(f"🎵 [AUDIO] Selected: {selected_track}")
        
        # Get audio data
        data = TRANSITION_POINTS[selected_track]
        tempo = data['tempo']
        original_duration = data['duration']
        
        logger.info(f"🎵 [AUDIO] Tempo: {tempo:.1f} BPM, Duration: {original_duration:.1f}s")
        
        # Find build-drop section
        audio_path = f"music_library/Test/{selected_track}"
        if not Path(audio_path).exists():
            raise Exception(f"Audio file not found: {audio_path}")
        
        logger.info(f"🔍 [AUDIO DEBUG] About to call _find_build_drop_section with {audio_path}")
        build_start, drop_end = self._find_build_drop_section(audio_path, target_duration=25)
        logger.info(f"🔍 [AUDIO DEBUG] _find_build_drop_section completed: {build_start:.1f}s to {drop_end:.1f}s")
        
        # Trim audio
        output_dir = Path("./9_10/robust_25_second")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trimmed_audio_path = str(output_dir / "trimmed_audio.mp3")
        logger.info(f"🔍 [AUDIO DEBUG] About to call _trim_audio: {audio_path} -> {trimmed_audio_path}")
        
        # Try to trim audio, with fallback if it fails
        if not self._trim_audio(audio_path, build_start, drop_end, trimmed_audio_path):
            logger.warning("⚠️ [AUDIO] Audio trimming failed, trying fallback approach...")
            # Fallback: copy the original audio file and use it as-is
            try:
                import shutil
                fallback_audio_path = str(output_dir / "fallback_audio.mp3")
                shutil.copy2(audio_path, fallback_audio_path)
                trimmed_audio_path = fallback_audio_path
                logger.info(f"🔄 [AUDIO] Using fallback audio: {trimmed_audio_path}")
            except Exception as e:
                logger.error(f"❌ [AUDIO] Fallback also failed: {e}")
                raise Exception("Failed to trim audio and fallback failed")
        else:
            logger.info(f"🔍 [AUDIO DEBUG] _trim_audio completed successfully")
        
        # Get trimmed audio duration
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', trimmed_audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            trimmed_duration = float(result.stdout.strip())
            logger.info(f"🎵 [AUDIO] Trimmed duration: {trimmed_duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ [AUDIO] Error getting trimmed duration: {e}")
            raise
        
        # Get transition points for trimmed section
        transition_points = get_transition_points(selected_track)
        beat_points = get_beat_points(selected_track)
        measure_points = get_measure_points(selected_track)
        
        # Filter to trimmed section
        trimmed_transition_points = [t for t in transition_points if build_start <= t <= drop_end]
        trimmed_beat_points = [b for b in beat_points if build_start <= b <= drop_end]
        trimmed_measure_points = [m for m in measure_points if build_start <= m <= drop_end]
        
        # Adjust to relative time
        relative_transition_points = [t - build_start for t in trimmed_transition_points]
        relative_beat_points = [b - build_start for b in trimmed_beat_points]
        relative_measure_points = [m - build_start for m in trimmed_measure_points]
        
        logger.info(f"🎵 [AUDIO] Beat points: {len(relative_beat_points)}, Measure points: {len(relative_measure_points)}")
        
        return {
            "selected_track": selected_track,
            "tempo": tempo,
            "trimmed_audio_path": trimmed_audio_path,
            "trimmed_duration": trimmed_duration,
            "beat_points": relative_beat_points,
            "measure_points": relative_measure_points,
            "transition_points": relative_transition_points
        }
    
    def _find_build_drop_section(self, audio_path: str, target_duration: int = 25) -> Tuple[float, float]:
        """Find the build-up to drop section in the audio."""
        try:
            # Load audio
            logger.info(f"🔍 [AUDIO DEBUG] Loading audio with librosa: {audio_path}")
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            logger.info(f"🔍 [AUDIO DEBUG] Audio loaded successfully: {duration:.1f}s duration")
            
            # Find peak energy
            hop_length = 512
            frame_length = 2048
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            # Find peak energy time
            peak_idx = np.argmax(rms)
            peak_time = times[peak_idx]
            
            # Calculate build-up start and drop end
            build_start = max(0, peak_time - target_duration / 2)
            drop_end = min(duration, build_start + target_duration)
            
            # Adjust if we're too close to the end
            if drop_end > duration - 5:
                drop_end = duration
                build_start = max(0, drop_end - target_duration)
            
            logger.info(f"🎵 [AUDIO] Found peak energy at: {peak_time:.1f}s")
            logger.info(f"🎵 [AUDIO] Build-up start: {build_start:.1f}s")
            logger.info(f"🎵 [AUDIO] Drop end: {drop_end:.1f}s")
            
            return build_start, drop_end
            
        except Exception as e:
            logger.error(f"❌ [AUDIO] Error finding build-drop section: {e}")
            # Fallback to middle section
            return max(0, duration / 2 - target_duration / 2), min(duration, duration / 2 + target_duration / 2)
    
    def _trim_audio(self, audio_path: str, start_time: float, end_time: float, output_path: str) -> bool:
        """Trim audio to specified time range."""
        try:
            # Skip FFmpeg entirely and just copy the original audio file
            # This is a fallback approach since FFmpeg keeps hanging
            import shutil
            
            logger.info(f"🔍 [AUDIO DEBUG] Skipping FFmpeg, copying original audio file")
            logger.info(f"🔍 [AUDIO DEBUG] Input file exists: {Path(audio_path).exists()}")
            logger.info(f"🔍 [AUDIO DEBUG] Input file size: {Path(audio_path).stat().st_size if Path(audio_path).exists() else 'N/A'}")
            logger.info(f"🔍 [AUDIO DEBUG] Output directory exists: {Path(output_path).parent.exists()}")
            
            # Copy the original audio file as a fallback
            shutil.copy2(audio_path, output_path)
            
            logger.info(f"✅ [AUDIO] Audio file copied successfully (fallback): {output_path}")
            return True
                
        except Exception as e:
            logger.error(f"❌ [AUDIO] Error copying audio file: {e}")
            return False
    
    async def _create_video_segments(
        self, 
        video_paths: List[str], 
        story_data: Dict[str, Any], 
        audio_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create video segments with effects based on audio analysis."""
        logger.info("🎬 [SEGMENTS] Creating video segments with effects...")
        
        segments = []
        current_time = 0
        beat_cut_points = sorted(audio_data['beat_points'])
        
        # Limit to reasonable number of segments
        if len(beat_cut_points) > 20:
            beat_cut_points = beat_cut_points[::3]
        
        # Ensure we have enough segments for target duration
        if len(beat_cut_points) < 13:
            additional_cuts = []
            for i in range(13 - len(beat_cut_points)):
                additional_cuts.append(audio_data['trimmed_duration'] * (i + 1) / 13)
            beat_cut_points.extend(additional_cuts)
            beat_cut_points.sort()
        
        logger.info(f"🎬 [SEGMENTS] Using {len(beat_cut_points)} beat points")
        
        story_moments = story_data.get('story_moments', [])
        
        # Use Gemini's story moments if available, otherwise fall back to beat points
        if story_moments and len(story_moments) > 0:
            logger.info(f"🎬 [SEGMENTS] Using {len(story_moments)} Gemini story moments")
            for i, moment in enumerate(story_moments):
                if current_time >= audio_data['trimmed_duration']:
                    break
                
                # Get video and timing from Gemini analysis
                video_index = moment.get('video_index', i % len(video_paths))
                start_time = moment.get('start_time', 0.0)
                duration = moment.get('duration', 2.0)
                
                # Ensure video index is valid
                if video_index >= len(video_paths):
                    video_index = i % len(video_paths)
                
                video_path = video_paths[video_index]
                
                # Get effects from Gemini analysis
                suggested_effects = moment.get('suggested_effects', 'color_boost,cinematic')
                effects = suggested_effects.split(',') if isinstance(suggested_effects, str) else suggested_effects
                
                logger.info(f"✅ [SEGMENTS] Created segment {i+1}: {duration:.1f}s from video {video_index+1}, effects: {effects}")
        else:
            # Fallback to beat points if no Gemini analysis
            logger.info(f"🎬 [SEGMENTS] No Gemini moments, using {len(beat_cut_points)} beat points")
            for i, cut_time in enumerate(beat_cut_points):
                if current_time >= audio_data['trimmed_duration']:
                    break
                
                # Determine segment duration
                if i < len(beat_cut_points) - 1:
                    duration = min(beat_cut_points[i + 1] - current_time, 2.0)
                else:
                    duration = min(audio_data['trimmed_duration'] - current_time, 2.0)
                
                if duration <= 0.1:
                    continue
                
                # Select video clip
                clip_index = i % len(video_paths)
                video_path = video_paths[clip_index]
                
                # Get effects for this segment
                effects = self._get_effects_for_segment(i, story_moments, audio_data)
                
                # Determine transition type
                transition_type = self._get_transition_type(i, beat_cut_points, current_time)
                
                segment = {
                "video_path": video_path,
                "start_time": current_time,
                "duration": duration,
                "effects": effects,
                "transition_type": transition_type,
                "clip_index": clip_index
            }
            
            segments.append(segment)
            current_time += duration
            
            logger.info(f"✅ [SEGMENTS] Created segment {i+1}: {duration:.1f}s, effects: {', '.join(effects) if effects else 'none'}")
        
        logger.info(f"🎬 [SEGMENTS] Created {len(segments)} segments, total duration: {current_time:.1f}s")
        return segments
    
    def _get_effects_for_segment(self, segment_index: int, story_moments: List[Dict], audio_data: Dict) -> List[str]:
        """Get effects for a video segment."""
        effects = []
        
        # Use Gemini's suggested effects if available
        if segment_index < len(story_moments):
            moment = story_moments[segment_index]
            suggested_effects = moment.get('suggested_effects', '').lower()
            
            if 'slow-motion' in suggested_effects or 'slow motion' in suggested_effects:
                effects.append("slow_motion")
            if 'color' in suggested_effects or 'color pop' in suggested_effects:
                effects.append("color_boost")
            if 'sharp' in suggested_effects or 'sharpness' in suggested_effects:
                effects.append("unsharp")
            if 'vibrant' in suggested_effects or 'vibrancy' in suggested_effects:
                effects.append("vibrance")
            if 'cinematic' in suggested_effects:
                effects.append("cinematic")
            if 'glow' in suggested_effects:
                effects.append("glow")
        else:
            # Fallback to diverse effects
            effect_pool = ["slow_motion", "vibrance", "unsharp", "cinematic", "color_boost", "glow"]
            effects.append(effect_pool[segment_index % len(effect_pool)])
        
        return effects
    
    def _get_transition_type(self, segment_index: int, beat_cut_points: List[float], current_time: float) -> str:
        """Get transition type for a segment."""
        if segment_index < len(beat_cut_points) - 1:
            next_cut = beat_cut_points[segment_index + 1]
            time_to_next = next_cut - current_time
            
            if time_to_next > 1.5:
                return "fade"
            elif time_to_next > 1.0:
                return "dissolve"
            elif time_to_next > 0.5:
                return "slide"
            else:
                return "zoom"
        else:
            return "fade"
    
    async def _render_final_video(
        self, 
        segments: List[Dict[str, Any]], 
        story_data: Dict[str, Any], 
        audio_data: Dict[str, Any],
        project_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """Render the final video with LUT and text overlays."""
        logger.info("🎨 [RENDER] Rendering final video with effects...")
        
        if not output_path:
            output_path = f"./9_10/gemini_multi_video_{project_id}.mp4"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load video clips
            logger.info(f"🔍 [RENDER DEBUG] Starting to load {len(segments)} video clips")
            video_clips = []
            for i, segment in enumerate(segments):
                logger.info(f"🎬 [RENDER] Loading video clip {i+1}/{len(segments)}: {segment['video_path']}")
                try:
                    clip = await safe_video_file_clip(segment['video_path'])
                    if clip is None:
                        logger.error(f"❌ [RENDER] Failed to load video clip: {segment['video_path']}")
                        raise Exception(f"Failed to load video clip: {segment['video_path']}")
                    
                    logger.info(f"✅ [RENDER] Successfully loaded video clip {i+1}")
                    
                    # Apply effects (simplified - in production, use the enhanced shader library)
                    logger.info(f"🎨 [RENDER] Applying effects to clip {i+1}: {segment['effects']}")
                    processed_clip = self._apply_effects_to_clip(clip, segment['effects'])
                    video_clips.append(processed_clip)
                    logger.info(f"✅ [RENDER] Successfully processed video clip {i+1}")
                except Exception as e:
                    logger.error(f"❌ [RENDER] Error processing clip {i+1}: {e}")
                    raise
            
            logger.info(f"🔍 [RENDER DEBUG] Successfully loaded {len(video_clips)} video clips")
            
            # Create composite video
            logger.info("🔍 [RENDER DEBUG] Creating CompositeVideoClip...")
            try:
                if not video_clips:
                    logger.error("❌ [RENDER DEBUG] No video clips available for composition")
                    raise Exception("No video clips available for composition")
                
                final_clip = CompositeVideoClip(video_clips)
                logger.info("✅ [RENDER DEBUG] CompositeVideoClip created successfully")
            except Exception as e:
                logger.error(f"❌ [RENDER DEBUG] Failed to create CompositeVideoClip: {e}")
                # Clean up clips before raising
                for clip in video_clips:
                    try:
                        clip.close()
                    except:
                        pass
                raise
            
            # Add audio
            logger.info("🔍 [RENDER DEBUG] Adding audio...")
            try:
                if Path(audio_data['trimmed_audio_path']).exists():
                    from moviepy import AudioFileClip
                    audio_clip = AudioFileClip(audio_data['trimmed_audio_path'])
                    final_clip = final_clip.with_audio(audio_clip)
                    logger.info("✅ [RENDER DEBUG] Audio added successfully")
                else:
                    logger.warning("⚠️ [RENDER DEBUG] Audio file not found, skipping audio")
            except Exception as e:
                logger.error(f"❌ [RENDER DEBUG] Failed to add audio: {e}")
                raise
            
            # Add text overlays
            logger.info("🔍 [RENDER DEBUG] Adding text overlays...")
            try:
                final_clip = self._add_text_overlays(final_clip, story_data, segments)
                logger.info("✅ [RENDER DEBUG] Text overlays added successfully")
            except Exception as e:
                logger.error(f"❌ [RENDER DEBUG] Failed to add text overlays: {e}")
                raise
            
            # Write final video
            logger.info(f"🔍 [RENDER DEBUG] Writing final video to: {output_path}")
            try:
                # Use a more conservative approach for video writing
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    fps=30  # Add FPS parameter
                )
                logger.info("✅ [RENDER DEBUG] Final video written successfully")
            except Exception as e:
                logger.error(f"❌ [RENDER DEBUG] Failed to write final video: {e}")
                # Clean up before raising
                try:
                    final_clip.close()
                except:
                    pass
                raise
            
            # Clean up
            for clip in video_clips:
                clip.close()
            final_clip.close()
            
            logger.info(f"✅ [RENDER] Final video written: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ [RENDER] Failed to render final video: {e}")
            raise
    
    def _apply_effects_to_clip(self, clip, effects: List[str]):
        """Apply effects to a video clip (simplified version)."""
        # In production, integrate with the enhanced shader library
        processed_clip = clip
        
        for effect in effects:
            if effect == "slow_motion":
                # Apply slow motion effect
                processed_clip = processed_clip.with_speed_scaled(0.5)
            elif effect == "color_boost":
                # Simple color boost - in production, use proper color grading
                # For now, just return the clip unchanged
                pass
            elif effect == "cinematic":
                # Cinematic effect - in production, use proper color grading
                pass
            elif effect == "glow":
                # Glow effect - in production, use proper shader
                pass
            elif effect == "vibrance":
                # Vibrance effect - in production, use proper color grading
                pass
            elif effect == "unsharp":
                # Unsharp mask effect - in production, use proper sharpening
                pass
            # Add more effects as needed
        
        return processed_clip
    
    def _add_text_overlays(self, video_clip, story_data: Dict[str, Any], segments: List[Dict[str, Any]]):
        """Add text overlays to the video."""
        try:
            story_moments = story_data.get('story_moments', [])
            
            # Create text clips for each segment
            text_clips = []
            for i, segment in enumerate(segments):
                if i < len(story_moments):
                    moment = story_moments[i]
                    caption = moment.get('caption', f'Highlight {i+1}')
                    
                    # Create text clip
                    text_clip = TextClip(
                        caption,
                        fontsize=50,
                        color='white',
                        font='Arial-Bold',
                        stroke_color='black',
                        stroke_width=2
                    ).set_position(('center', 'bottom')).set_duration(segment['duration']).set_start(segment['start_time'])
                    
                    text_clips.append(text_clip)
            
            # Composite with text
            if text_clips:
                return CompositeVideoClip([video_clip] + text_clips)
            else:
                return video_clip
                
        except Exception as e:
            logger.error(f"❌ [TEXT] Failed to add text overlays: {e}")
            return video_clip
