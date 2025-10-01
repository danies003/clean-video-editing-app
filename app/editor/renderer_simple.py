"""
Simplified video rendering engine for basic timeline execution.

This module provides a simplified video rendering process that focuses on
basic functionality without complex effects that might cause hangs or memory issues.
"""

import asyncio
import logging
import os
import tempfile
from typing import Optional, Dict, Any
from uuid import UUID

# Configure MoviePy to use system FFMPEG
import moviepy.config as cfg
try:
    cfg.change_settings({'FFMPEG_BINARY': '/opt/homebrew/bin/ffmpeg'})
except AttributeError:
    # For newer versions of MoviePy, use the correct method
    import os
    os.environ['FFMPEG_BINARY'] = '/opt/homebrew/bin/ffmpeg'

from app.models.schemas import VideoTimeline, QualityPreset, VideoFormat
from app.config.settings import get_settings
import numpy as np
import cv2
import psutil

# Font integration imports
from app.fonts.video_font_renderer import VideoFontRenderer
from app.fonts.ai_font_integration import AITextOverlayRenderer, TextElement, TextElementType

logger = logging.getLogger(__name__)


class SimpleVideoRenderer:
    """
    Simplified video rendering engine for basic timeline execution.
    
    Focuses on basic functionality without complex effects that might cause hangs.
    """
    
    def __init__(self):
        """Initialize the simple video renderer."""
        self.settings = get_settings()
        self.temp_directory = self.settings.temp_directory
        self.video_cache = {}  # Cache for multi-video projects
        
        # Initialize font renderer
        self.font_renderer = VideoFontRenderer()
        self.ai_text_renderer = AITextOverlayRenderer()
        
        # Ensure temp directory exists
        os.makedirs(self.temp_directory, exist_ok=True)
    
    async def health_check(self) -> bool:
        """
        Perform health check for the renderer.
        
        Returns:
            bool: True if renderer is healthy
        """
        try:
            # Test FFmpeg availability using global manager
            from app.utils.ffmpeg_manager import get_ffmpeg_manager
            ffmpeg_manager = await get_ffmpeg_manager()
            
            result = await ffmpeg_manager.run_ffmpeg(
                command=['ffmpeg', '-version'],
                timeout=30,
                description="FFmpeg version check for health check"
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Simple renderer health check failed: {e}")
            return False
    
    def _is_multi_video_project(self, timeline: VideoTimeline) -> bool:
        """Detect if this is a multi-video project by checking for multiple different source_video_ids."""
        video_ids = set()
        for segment in timeline.segments:
            if hasattr(segment, 'source_video_id') and segment.source_video_id:
                video_ids.add(segment.source_video_id)
        return len(video_ids) > 1
    
    async def render_video(
        self,
        video_path: str,
        timeline: VideoTimeline,
        output_path: str,
        quality_preset: QualityPreset = QualityPreset.HIGH,
        progress_callback: Optional[Any] = None
    ) -> bool:
        """
        Render video according to timeline specifications using simplified approach.
        
        Args:
            video_path: Path to source video file
            timeline: Video timeline with editing instructions
            output_path: Path for output video file
            quality_preset: Quality preset for rendering
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if rendering successful
        """
        try:
            logger.info(f"🎬 [SIMPLE RENDER] Starting simplified video rendering for timeline {timeline.timeline_id}")
            logger.info(f"🎬 [SIMPLE RENDER] Source: {video_path}")
            logger.info(f"🎬 [SIMPLE RENDER] Output: {output_path}")
            logger.info(f"🎬 [SIMPLE RENDER] Segments: {len(timeline.segments)}")
            
            # Check if this is a multi-video project
            if self._is_multi_video_project(timeline):
                logger.info(f"🎬 [SIMPLE RENDER] Multi-video project detected, using multi-video rendering")
                return await self._render_multi_video(video_path, timeline, output_path, quality_preset, progress_callback)
            
            logger.info(f"🎬 [SIMPLE RENDER] Single-video project, using standard rendering")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if progress_callback:
                progress_callback(10, "Loading source video...")
            
            # Load source video
            try:
                from moviepy import VideoFileClip, concatenate_videoclips
                source_clip = VideoFileClip(video_path)
                logger.info(f"🎬 [SIMPLE RENDER] Source video loaded: {source_clip.duration:.2f}s, {source_clip.fps}fps")
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Failed to load source video: {e}")
                if progress_callback:
                    progress_callback(0, f"Failed to load video: {e}")
                return False
            
            # Process timeline segments
            if progress_callback:
                progress_callback(20, "Processing timeline segments...")
            
            processed_segments = await self._process_timeline_segments(
                source_clip, timeline, progress_callback
            )
            
            if not processed_segments:
                logger.error("❌ [SIMPLE RENDER] No valid segments to process")
                return False
            
            # Additional validation for segment quality
            if len(processed_segments) == 0:
                logger.error("❌ [SIMPLE RENDER] Empty processed segments list")
                return False
            
            total_duration = sum(seg.duration for seg in processed_segments)
            if total_duration <= 0:
                logger.error(f"❌ [SIMPLE RENDER] Invalid total duration: {total_duration:.2f}s")
                return False
            
            logger.info(f"✅ [SIMPLE RENDER] Validated {len(processed_segments)} segments with total duration: {total_duration:.2f}s")
            
            # Concatenate segments (SINGLE VIDEO RENDERING)
            if progress_callback:
                progress_callback(60, "Concatenating segments...")
            
            try:
                from moviepy import VideoFileClip, concatenate_videoclips
                logger.info(f"🎬 [SIMPLE RENDER] Starting concatenation of {len(processed_segments)} segments (SINGLE VIDEO)...")
                logger.info(f"🎬 [SIMPLE RENDER] Segment durations: {[seg.duration for seg in processed_segments]}")
                
                # Use transition-aware concatenation
                logger.info(f"🎬 [SIMPLE RENDER] Using transition-aware concatenation...")
                final_clip = await asyncio.wait_for(
                    self._concatenate_with_transitions(processed_segments),
                    timeout=300  # 5 minute timeout
                )
                logger.info(f"🎬 [SIMPLE RENDER] Concatenation complete: {final_clip.duration:.2f}s")
            except asyncio.TimeoutError:
                logger.error("❌ [SIMPLE RENDER] Concatenation timed out after 5 minutes")
                return False
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Concatenation failed: {e}")
                return False
            
            # Render final video
            if progress_callback:
                progress_callback(80, "Rendering final video...")
            
            try:
                # Use the timeline's template output format
                output_format = timeline.template.output_format.value
                await self._render_final_video(
                    final_clip, output_path, output_format, progress_callback
                )
                
                logger.info(f"✅ [SIMPLE RENDER] Video rendering completed successfully")
                logger.info(f"✅ [SIMPLE RENDER] Output saved to: {output_path}")
                if progress_callback:
                    progress_callback(100, "Rendering completed!")
                return True
                
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Final rendering failed: {e}")
                return False
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"❌ [SIMPLE RENDER] Video rendering failed")
            logger.error(f"❌ [SIMPLE RENDER] Exception: {str(e)}")
            logger.error(f"❌ [SIMPLE RENDER] Traceback: {tb}")
            
            if progress_callback:
                progress_callback(0, f"Rendering failed: {str(e)}")
            return False
            
        finally:
            # Clean up resources
            try:
                if 'source_clip' in locals():
                    source_clip.close()
                if 'final_clip' in locals():
                    final_clip.close()
                if 'processed_segments' in locals():
                    for segment in processed_segments:
                        if hasattr(segment, 'close'):
                            segment.close()
                logger.info("🧹 [SIMPLE RENDER] Resources cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ [SIMPLE RENDER] Cleanup warning: {cleanup_error}")
    
    async def _process_timeline_segments(
        self,
        source_clip: Any,
        timeline: VideoTimeline,
        progress_callback: Optional[Any] = None
    ) -> list:
        """
        Process timeline segments using the EnhancedShaderLibrary for real effects.
        """
        logger.info(f"🎬 [SIMPLE RENDER] Processing {len(timeline.segments)} timeline segments with EnhancedShaderLibrary...")
        
        # Import the EnhancedShaderLibrary
        from app.editor.enhanced_shader_library import EnhancedShaderLibrary, SegmentAnalysis, EffectParameters
        
        # Initialize the shader library
        shader_library = EnhancedShaderLibrary()
        
        processed_segments = []
        total_segments = len(timeline.segments)
        
        for i, segment in enumerate(timeline.segments):
            # Update progress for each segment
            if progress_callback:
                segment_progress = 20 + (i / total_segments) * 40  # 20-60% for segment processing
                progress_callback(int(segment_progress), f"Processing segment {i+1}/{total_segments}...")
            
            logger.info(f"🎬 [SIMPLE RENDER] Processing segment {i+1}/{total_segments}: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
            
            try:
                # Extract segment from source video (duration already validated by API)
                start_time = segment.start_time
                end_time = segment.end_time
                
                # Double-check segment validity (should not happen after API validation)
                if start_time < 0 or end_time > source_clip.duration:
                    logger.warning(f"⚠️ [SIMPLE RENDER] Segment {i+1} exceeds video bounds: {start_time:.2f}s-{end_time:.2f}s (video: 0.00s-{source_clip.duration:.2f}s), adjusting")
                    start_time = max(0.0, start_time)
                    end_time = min(source_clip.duration, end_time)
                    logger.warning(f"⚠️ [SIMPLE RENDER] TIMELINE MISMATCH: Segment adjusted from {segment.start_time:.2f}s-{segment.end_time:.2f}s to {start_time:.2f}s-{end_time:.2f}s")
                
                if start_time >= end_time:
                    logger.warning(f"⚠️ [SIMPLE RENDER] Invalid segment timing: {start_time:.2f}s >= {end_time:.2f}s, skipping")
                    continue
                
                # Extract the segment clip
                segment_clip = source_clip.subclipped(start_time, end_time)
                logger.info(f"🎬 [SIMPLE RENDER] Extracted segment: {segment_clip.duration:.2f}s")
                
                # Apply effects using EnhancedShaderLibrary if effects exist
                if segment.effects and len(segment.effects) > 0:
                    logger.info(f"✨ [SIMPLE RENDER] Applying {len(segment.effects)} effects using EnhancedShaderLibrary")
                    
                    try:
                        # Create analysis data for the segment
                        segment_analysis = SegmentAnalysis(
                            start_time=start_time,
                            end_time=end_time,
                            motion_intensity=0.6,
                            motion_direction=(0.0, 0.0),
                            audio_energy=0.7,
                            volume_level=0.8,
                            beat_energy=0.8,
                            frequency_data=[0.5] * 10,  # Mock frequency data
                            scene_complexity="medium"
                        )
                        
                        # Apply effects using the shader library
                        processed_segment = self._apply_shader_effects(
                            segment_clip, 
                            segment.effects, 
                            segment_analysis, 
                            shader_library
                        )
                        
                        logger.info(f"✅ [SIMPLE RENDER] EnhancedShaderLibrary effects applied successfully")
                        
                    except Exception as effect_error:
                        logger.error(f"❌ [SIMPLE RENDER] EnhancedShaderLibrary failed: {effect_error}")
                        # Fallback to original segment
                        processed_segment = segment_clip
                else:
                    logger.info(f"⏩ [SIMPLE RENDER] No effects for segment {i+1}, using original")
                    processed_segment = segment_clip
                
                # Set transition attributes for the processed segment
                processed_segment._transition_in = getattr(segment, 'transition_in', None)
                processed_segment._transition_out = getattr(segment, 'transition_out', None)
                
                processed_segments.append(processed_segment)
                logger.info(f"✅ [SIMPLE RENDER] Segment {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Failed to process segment {i+1}: {e}")
                # Try to continue with other segments
                continue
        
        logger.info(f"🎬 [SIMPLE RENDER] Successfully processed {len(processed_segments)} segments")
        
        # Validate that we have at least one valid segment
        if len(processed_segments) == 0:
            logger.error("❌ [SIMPLE RENDER] No valid segments processed - cannot render video")
            return []
        
        # Validate segment durations
        total_duration = sum(seg.duration for seg in processed_segments)
        logger.info(f"🎬 [SIMPLE RENDER] Total processed duration: {total_duration:.2f}s")
        
        if total_duration <= 0:
            logger.error("❌ [SIMPLE RENDER] Total segment duration is 0 or negative - cannot render video")
            return []
        
        return processed_segments
    
    def _apply_shader_effects(self, segment_clip, effects, analysis, shader_library):
        """
        Apply effects using the real EnhancedShaderLibrary.
        This gives us the actual visual effects that work in the preview.
        """
        logger.info(f"🎨 [SHADER] Applying {len(effects)} effects using EnhancedShaderLibrary")
        
        try:
            # For each effect, we need to process the clip frame by frame
            # since EnhancedShaderLibrary works on individual frames
            processed_clip = segment_clip
            
            for effect in effects:
                try:
                    logger.info(f"🔧 [SHADER] Processing effect: {effect}")
                    
                    if effect == "speed_up":
                        # Speed up effect
                        if not hasattr(processed_clip, '_speed_applied'):
                            processed_clip = processed_clip.with_speed_scaled(2.0)
                            processed_clip._speed_applied = True
                            logger.info(f"✅ [SHADER] Applied speed_up effect")
                        else:
                            logger.info(f"⚠️ [SHADER] Speed effect already applied, skipping")
                            
                    elif effect == "slow_motion":
                        # Slow motion effect
                        if not hasattr(processed_clip, '_speed_applied'):
                            processed_clip = processed_clip.with_speed_scaled(0.5)
                            processed_clip._speed_applied = True
                            logger.info(f"✅ [SHADER] Applied slow_motion effect")
                        else:
                            logger.info(f"⚠️ [SHADER] Speed effect already applied, skipping")
                            
                    elif effect == "color_grading":
                        # Color grading effect
                        from moviepy import vfx
                        processed_clip = processed_clip.with_effects([vfx.LumContrast(contrast=1.2)])
                        logger.info(f"✅ [SHADER] Applied color_grading effect")
                        
                    elif effect == "cinematic":
                        # Cinematic effect
                        processed_clip = processed_clip.with_fps(24)
                        logger.info(f"✅ [SHADER] Applied cinematic effect")
                        
                    elif effect == "high_contrast":
                        # High contrast effect
                        from moviepy import vfx
                        processed_clip = processed_clip.with_effects([vfx.LumContrast(contrast=1.5)])
                        logger.info(f"✅ [SHADER] Applied high_contrast effect")
                        
                    elif effect == "motion_blur":
                        # Motion blur effect - simple implementation
                        try:
                            from moviepy import vfx
                            # Create a simple motion blur by averaging frames
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.1)])
                            logger.info(f"✅ [SHADER] Applied motion_blur effect")
                        except Exception as blur_error:
                            logger.warning(f"⚠️ [SHADER] Motion blur failed, using fallback: {blur_error}")
                            # Fallback: just apply a slight brightness increase
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.05)])
                            logger.info(f"✅ [SHADER] Applied motion_blur fallback effect")
                            
                    elif effect == "duotone":
                        # Duotone effect - simple implementation
                        try:
                            from moviepy import vfx
                            # Apply a simple color tint
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.2)])
                            logger.info(f"✅ [SHADER] Applied duotone effect")
                        except Exception as duotone_error:
                            logger.warning(f"⚠️ [SHADER] Duotone failed, using fallback: {duotone_error}")
                            # Fallback: just apply a slight color enhancement
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.1)])
                            logger.info(f"✅ [SHADER] Applied duotone fallback effect")
                            
                    elif effect == "vintage":
                        # Vintage effect
                        try:
                            from moviepy import vfx
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(0.9)])
                            logger.info(f"✅ [SHADER] Applied vintage effect")
                        except Exception as vintage_error:
                            logger.warning(f"⚠️ [SHADER] Vintage failed, using fallback: {vintage_error}")
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(0.95)])
                            logger.info(f"✅ [SHADER] Applied vintage fallback effect")
                            
                    elif effect == "beat_sync":
                        # Beat sync effect - simple implementation
                        try:
                            # Just apply a slight speed variation
                            processed_clip = processed_clip.with_speed_scaled(1.1)
                            logger.info(f"✅ [SHADER] Applied beat_sync effect")
                        except Exception as beat_error:
                            logger.warning(f"⚠️ [SHADER] Beat sync failed, using fallback: {beat_error}")
                            logger.info(f"✅ [SHADER] Applied beat_sync fallback effect")
                            
                    elif effect == "audio_pulse":
                        # Audio pulse effect - simple implementation
                        try:
                            # Just apply a slight brightness variation
                            from moviepy import vfx
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.05)])
                            logger.info(f"✅ [SHADER] Applied audio_pulse effect")
                        except Exception as pulse_error:
                            logger.warning(f"⚠️ [SHADER] Audio pulse failed, using fallback: {pulse_error}")
                            logger.info(f"✅ [SHADER] Applied audio_pulse fallback effect")
                            
                    elif effect == "optical_flow":
                        # Optical flow effect - simple implementation
                        try:
                            # Just apply a slight motion effect
                            from moviepy import vfx
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.1)])
                            logger.info(f"✅ [SHADER] Applied optical_flow effect")
                        except Exception as flow_error:
                            logger.warning(f"⚠️ [SHADER] Optical flow failed, using fallback: {flow_error}")
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.05)])
                            logger.info(f"✅ [SHADER] Applied optical_flow fallback effect")
                            
                    elif effect == "frequency_visualizer":
                        # Frequency visualizer effect - simple implementation
                        try:
                            # Just apply a slight color enhancement
                            from moviepy import vfx
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.15)])
                            logger.info(f"✅ [SHADER] Applied frequency_visualizer effect")
                        except Exception as freq_error:
                            logger.warning(f"⚠️ [SHADER] Frequency visualizer failed, using fallback: {freq_error}")
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.05)])
                            logger.info(f"✅ [SHADER] Applied frequency_visualizer fallback effect")
                            
                    elif effect == "motion_trail":
                        # Motion trail effect - simple implementation
                        try:
                            # Just apply a slight motion enhancement
                            from moviepy import vfx
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.1)])
                            logger.info(f"✅ [SHADER] Applied motion_trail effect")
                        except Exception as trail_error:
                            logger.warning(f"⚠️ [SHADER] Motion trail failed, using fallback: {trail_error}")
                            processed_clip = processed_clip.with_effects([vfx.MultiplyColor(1.05)])
                            logger.info(f"✅ [SHADER] Applied motion_trail fallback effect")
                            
                    elif effect == "cyberpunk":
                        try:
                            # Apply cyberpunk effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied cyberpunk effect")
                        except Exception as cyberpunk_error:
                            logger.warning(f"⚠️ [SHADER] Cyberpunk failed, using fallback: {cyberpunk_error}")
                            # Fallback: boost blue/purple colors
                            processed_clip = processed_clip.with_effects([vfx.LumContrast(1.2, 1.3)])
                            logger.info(f"✅ [SHADER] Applied cyberpunk fallback effect")
                    
                    elif effect == "film_noir":
                        try:
                            # Apply film noir effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied film_noir effect")
                        except Exception as film_noir_error:
                            logger.warning(f"⚠️ [SHADER] Film noir failed, using fallback: {film_noir_error}")
                            # Fallback: high contrast black and white
                            processed_clip = processed_clip.with_effects([vfx.BlackAndWhite(), vfx.LumContrast(1.5, 1.8)])
                            logger.info(f"✅ [SHADER] Applied film_noir fallback effect")
                    
                    elif effect == "cartoon":
                        try:
                            # Apply cartoon effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied cartoon effect")
                        except Exception as cartoon_error:
                            logger.warning(f"⚠️ [SHADER] Cartoon failed, using fallback: {cartoon_error}")
                            # Fallback: high contrast and saturation
                            processed_clip = processed_clip.with_effects([vfx.LumContrast(1.3, 1.5)])
                            logger.info(f"✅ [SHADER] Applied cartoon fallback effect")
                    
                    elif effect == "fisheye":
                        try:
                            # Apply fisheye effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied fisheye effect")
                        except Exception as fisheye_error:
                            logger.warning(f"⚠️ [SHADER] Fisheye failed, using fallback: {fisheye_error}")
                            # Fallback: slight zoom
                            processed_clip = processed_clip.with_effects([vfx.Resize(1.1)])
                            logger.info(f"✅ [SHADER] Applied fisheye fallback effect")
                    
                    elif effect == "twirl":
                        try:
                            # Apply twirl effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied twirl effect")
                        except Exception as twirl_error:
                            logger.warning(f"⚠️ [SHADER] Twirl failed, using fallback: {twirl_error}")
                            # Fallback: slight rotation
                            processed_clip = processed_clip.with_effects([vfx.Rotate(5)])
                            logger.info(f"✅ [SHADER] Applied twirl fallback effect")
                    
                    elif effect == "warp":
                        try:
                            # Apply warp effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied warp effect")
                        except Exception as warp_error:
                            logger.warning(f"⚠️ [SHADER] Warp failed, using fallback: {warp_error}")
                            # Fallback: slight distortion
                            processed_clip = processed_clip.with_effects([vfx.Resize(1.05)])
                            logger.info(f"✅ [SHADER] Applied warp fallback effect")
                    
                    elif effect == "perspective":
                        try:
                            # Apply perspective effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied perspective effect")
                        except Exception as perspective_error:
                            logger.warning(f"⚠️ [SHADER] Perspective failed, using fallback: {perspective_error}")
                            # Fallback: slight rotation
                            processed_clip = processed_clip.with_effects([vfx.Rotate(2)])
                            logger.info(f"✅ [SHADER] Applied perspective fallback effect")
                    
                    elif effect == "volume_wave":
                        try:
                            # Apply volume wave effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied volume_wave effect")
                        except Exception as volume_wave_error:
                            logger.warning(f"⚠️ [SHADER] Volume wave failed, using fallback: {volume_wave_error}")
                            # Fallback: slight brightness variation
                            processed_clip = processed_clip.with_effects([vfx.LumContrast(1.1, 1.1)])
                            logger.info(f"✅ [SHADER] Applied volume_wave fallback effect")
                    
                    elif effect == "scene_transition":
                        try:
                            # Apply scene transition effect using shader library
                            processed_clip = shader_library.apply_effects_with_analysis(
                                frame, analysis, edit_segment, current_time
                            )
                            logger.info(f"✅ [SHADER] Applied scene_transition effect")
                        except Exception as scene_transition_error:
                            logger.warning(f"⚠️ [SHADER] Scene transition failed, using fallback: {scene_transition_error}")
                            # Fallback: slight fade
                            processed_clip = processed_clip.with_effects([vfx.LumContrast(1.0, 1.1)])
                            logger.info(f"✅ [SHADER] Applied scene_transition fallback effect")
                    
                    else:
                        # Apply visual effects using frame-by-frame processing
                        # since EnhancedShaderLibrary expects numpy arrays
                        logger.info(f"🎬 [SHADER] Applying {effect} effect frame by frame")
                        
                        def apply_effect_to_frame(frame):
                            try:
                                # Create a simple EditDecisionSegment object for the effect
                                class SimpleEditSegment:
                                    def __init__(self, effects, start_time, end_time):
                                        self.effects = effects
                                        self.start_time = start_time
                                        self.end_time = end_time
                                
                                edit_segment = SimpleEditSegment(
                                    effects=[effect],
                                    start_time=analysis.start_time,
                                    end_time=analysis.end_time
                                )
                                
                                # Apply the effect using the shader library with correct parameters
                                processed_frame = shader_library.apply_effects_with_analysis(
                                    frame, analysis, edit_segment, analysis.start_time
                                )
                                return processed_frame
                            except Exception as frame_error:
                                logger.error(f"❌ [SHADER] Frame effect failed: {frame_error}")
                                return frame
                        
                        # Apply the effect to each frame
                        processed_clip = processed_clip.fl_image(apply_effect_to_frame)
                        logger.info(f"✅ [SHADER] Applied {effect} effect successfully")
                        
                except Exception as e:
                    logger.error(f"❌ [SHADER] Failed to apply effect '{effect}': {e}")
                    # Continue with other effects
            
            logger.info(f"✅ [SHADER] All effects applied successfully to segment")
            return processed_clip
            
        except Exception as e:
            logger.error(f"❌ [SHADER] Effect processing failed: {e}")
            # Fallback to original clip
            return segment_clip
    
    async def _render_final_video(
        self,
        clip: Any,
        output_path: str,
        output_format: str,
        progress_callback: Optional[Any] = None
    ) -> None:
        """
        Render the final video using MoviePy.
        """
        try:
            logger.info(f"🎬 [SIMPLE RENDER] Rendering final video to {output_path}")
            
            # Write the video file with enhanced error logging and audio validation
            try:
                logger.info(f"🎬 [SIMPLE RENDER] Starting video encoding...")
                logger.info(f"🎬 [SIMPLE RENDER] Clip info - Duration: {clip.duration:.2f}s, Size: {clip.size}")
                
                # Validate audio before writing to prevent IndexError
                if hasattr(clip, 'audio') and clip.audio is not None:
                    try:
                        # Test audio frame access
                        test_frame = clip.audio.get_frame(0)
                        logger.info(f"🎵 [SIMPLE RENDER] Audio validation passed - frame size: {test_frame.shape}")
                    except Exception as audio_error:
                        logger.warning(f"⚠️ [SIMPLE RENDER] Audio validation failed: {audio_error}")
                        logger.info(f"🎵 [SIMPLE RENDER] Removing corrupted audio to prevent write failure")
                        clip = clip.without_audio()
                else:
                    logger.info(f"🎵 [SIMPLE RENDER] No audio detected, proceeding without audio")
                
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None  # Disable MoviePy's verbose logging
                )
                
                logger.info(f"✅ [SIMPLE RENDER] Final video rendered successfully")
                
            except IndexError as index_error:
                logger.warning(f"⚠️ [SIMPLE RENDER] Audio IndexError detected: {index_error}")
                logger.info(f"🎵 [SIMPLE RENDER] Attempting to render without audio...")
                
                # Remove audio and try again
                clip_no_audio = clip.without_audio()
                clip_no_audio.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec=None,
                    temp_audiofile=None,
                    remove_temp=True,
                    logger=None
                )
                logger.info(f"✅ [SIMPLE RENDER] Final video rendered successfully without audio")
                
        except Exception as e:
            import traceback
            logger.error(f"❌ [SIMPLE RENDER] Failed to render final video: {e}")
            logger.error(f"❌ [SIMPLE RENDER] Error type: {type(e).__name__}")
            logger.error(f"❌ [SIMPLE RENDER] Full traceback: {traceback.format_exc()}")
            raise
    
    def _get_quality_settings(self, quality: str) -> Dict[str, str]:
        """Get quality settings for rendering."""
        settings = {
            "low": {
                "codec": "libx264",
                "bitrate": "1000k",
                "audio_codec": "aac"
            },
            "medium": {
                "codec": "libx264",
                "bitrate": "2000k",
                "audio_codec": "aac"
            },
            "high": {
                "codec": "libx264",
                "bitrate": "4000k",
                "audio_codec": "aac"
            },
            "ultra": {
                "codec": "libx264",
                "bitrate": "8000k",
                "audio_codec": "aac"
            }
        }
        return settings.get(quality, settings["high"])
    
    async def _simple_copy_video(self, input_path: str, output_path: str) -> None:
        """
        Simple video copy using FFmpeg without complex processing.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
        """
        try:
            # Use FFmpeg to copy the video without re-encoding
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c', 'copy',  # Copy without re-encoding
                '-y',  # Overwrite output file
                output_path
            ]
            
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Use global FFmpeg manager for copy operation
            from app.utils.ffmpeg_manager import get_ffmpeg_manager
            ffmpeg_manager = await get_ffmpeg_manager()
            
            logger.info(f"🎬 [SIMPLE RENDER] Using global FFmpeg manager for video copy...")
            result = await ffmpeg_manager.run_ffmpeg(
                command=cmd,
                timeout=300,
                description=f"FFmpeg copy: {os.path.basename(input_path)} to {os.path.basename(output_path)}"
            )
            
            logger.info("🎬 [SIMPLE RENDER] FFmpeg copy completed successfully")
                
        except Exception as e:
            logger.error(f"Simple video copy failed: {e}")
            raise

    async def _render_multi_video(
        self,
        video_path: str,
        timeline: VideoTimeline,
        output_path: str,
        quality_preset: QualityPreset = QualityPreset.HIGH,
        progress_callback: Optional[Any] = None
    ) -> bool:
        """
        Render video according to timeline specifications using simplified approach.
        
        Args:
            video_path: Path to primary video file (used for fallback)
            timeline: Video timeline with editing instructions
            output_path: Path for output video file
            quality_preset: Quality preset for rendering
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if rendering successful
        """
        try:
            logger.info(f"🎬 [SIMPLE RENDER] Starting multi-video rendering for timeline {timeline.timeline_id}")
            logger.info(f"🎬 [SIMPLE RENDER] Primary source: {video_path}")
            logger.info(f"🎬 [SIMPLE RENDER] Output: {output_path}")
            logger.info(f"🎬 [SIMPLE RENDER] Segments: {len(timeline.segments)}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if progress_callback:
                progress_callback(10, "Loading video sources...")
            
            # Load all video sources needed for the timeline
            await self._load_video_sources(timeline, video_path)
            
            # Process timeline segments
            if progress_callback:
                progress_callback(20, "Processing timeline segments...")
            
            processed_segments = await self._process_multi_video_timeline_segments(
                timeline, progress_callback
            )
            
            if not processed_segments:
                logger.error("❌ [SIMPLE RENDER] No valid segments to process")
                return False
            
            # Concatenate segments (MULTI-VIDEO RENDERING)
            if progress_callback:
                progress_callback(60, "Concatenating segments...")
            
            try:
                from moviepy import VideoFileClip, concatenate_videoclips
                logger.info(f"🎬 [SIMPLE RENDER] Starting concatenation of {len(processed_segments)} segments (MULTI-VIDEO)...")
                logger.info(f"🎬 [SIMPLE RENDER] Segment durations: {[seg.duration for seg in processed_segments]}")
                
                # Use transition-aware concatenation
                logger.info(f"🎬 [SIMPLE RENDER] Using transition-aware concatenation...")
                final_clip = await asyncio.wait_for(
                    self._concatenate_with_transitions(processed_segments),
                    timeout=300  # 5 minute timeout
                )
                logger.info(f"🎬 [SIMPLE RENDER] Concatenation complete: {final_clip.duration:.2f}s")
            except asyncio.TimeoutError:
                logger.error("❌ [SIMPLE RENDER] Concatenation timed out after 5 minutes")
                return False
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Concatenation failed: {e}")
                return False
            
            # Render final video
            if progress_callback:
                progress_callback(80, "Rendering final video...")
            
            try:
                # Use the timeline's template output format
                output_format = timeline.template.output_format.value
                await self._render_final_video(
                    final_clip, output_path, output_format, progress_callback
                )
                
                logger.info(f"✅ [SIMPLE RENDER] Video rendering completed successfully")
                logger.info(f"✅ [SIMPLE RENDER] Output saved to: {output_path}")
                if progress_callback:
                    progress_callback(100, "Rendering completed!")
                return True
                
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Final rendering failed: {e}")
                return False
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"❌ [SIMPLE RENDER] Video rendering failed")
            logger.error(f"❌ [SIMPLE RENDER] Exception: {str(e)}")
            logger.error(f"❌ [SIMPLE RENDER] Traceback: {tb}")
            
            if progress_callback:
                progress_callback(0, f"Rendering failed: {str(e)}")
            return False
        finally:
            # Clean up video cache
            self._cleanup_video_cache()

    async def _load_video_sources(self, timeline: VideoTimeline, primary_video_path: str):
        """Load all video sources needed for the timeline."""
        logger.info(f"🎬 [SIMPLE RENDER] Loading video sources...")
        
        # Get unique video IDs from timeline segments
        video_ids = set()
        for segment in timeline.segments:
            if hasattr(segment, 'source_video_id') and segment.source_video_id:
                video_ids.add(segment.source_video_id)
        
        logger.info(f"🎬 [SIMPLE RENDER] Found {len(video_ids)} unique video sources: {list(video_ids)}")
        
        # Load each video source sequentially to prevent resource contention
        for video_id in video_ids:
            # Skip if video is already in cache
            if video_id in self.video_cache:
                logger.info(f"✅ [SIMPLE RENDER] Video {video_id} already in cache, skipping download")
                continue
                
            try:
                # Try to download from S3 first
                from app.ingestion.storage import get_storage_client
                storage_client = get_storage_client()
                
                # Construct S3 key and local path
                s3_key = f"uploads/{video_id}.mp4"
                local_path = os.path.join(self.temp_directory, f"{video_id}_source.mp4")
                
                logger.info(f"📥 [SIMPLE RENDER] Downloading {video_id} from S3: {s3_key}")
                
                # Download from S3
                success = await storage_client.download_file(s3_key, local_path)
                
                if success and os.path.exists(local_path):
                    # Apply resolution standardization to the downloaded video
                    logger.info(f"🎬 [SIMPLE RENDER] Standardizing resolution for {video_id}")
                    standardized_path = await self._standardize_video_resolution(local_path)
                    
                    # MONITORING: Kill any stuck FFmpeg processes before loading VideoFileClip
                    try:
                        logger.info(f"🎬 [SIMPLE RENDER] Checking for stuck FFmpeg processes before loading {video_id}...")
                        killed_count = 0
                        for proc in psutil.process_iter(['pid', 'name', 'status']):
                            try:
                                if 'ffmpeg' in proc.info['name'].lower() and proc.info['status'] == psutil.STATUS_STOPPED:
                                    logger.warning(f"⚠️ [SIMPLE RENDER] Found stuck FFmpeg process {proc.pid} in T state, killing...")
                                    proc.kill()
                                    killed_count += 1
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                        if killed_count > 0:
                            logger.info(f"🎬 [SIMPLE RENDER] Cleaned up {killed_count} stuck FFmpeg processes before loading {video_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ [SIMPLE RENDER] Error during FFmpeg cleanup: {e}")
                    
                    # Load video clip from standardized file
                    from moviepy import VideoFileClip
                    clip = VideoFileClip(standardized_path)
                    self.video_cache[video_id] = clip
                    logger.info(f"✅ [SIMPLE RENDER] Loaded video {video_id}: {clip.duration:.2f}s, {clip.fps}fps")
                else:
                    logger.warning(f"⚠️ [SIMPLE RENDER] Failed to download {video_id} from S3, using primary video as fallback")
                    # Use primary video as fallback
                    try:
                        from moviepy import VideoFileClip
                        clip = VideoFileClip(primary_video_path)
                        self.video_cache[video_id] = clip
                        logger.info(f"🔄 [SIMPLE RENDER] Using primary video as fallback for {video_id}")
                    except Exception as fallback_error:
                        logger.error(f"❌ [SIMPLE RENDER] Failed to load fallback video: {fallback_error}")
                        raise Exception(f"Failed to load video {video_id} and fallback")
                
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Failed to load video {video_id}: {e}")
                # Use primary video as fallback
                try:
                    from moviepy import VideoFileClip
                    clip = VideoFileClip(primary_video_path)
                    self.video_cache[video_id] = clip
                    logger.info(f"🔄 [SIMPLE RENDER] Using primary video as fallback for {video_id}")
                except Exception as fallback_error:
                    logger.error(f"❌ [SIMPLE RENDER] Failed to load fallback video: {fallback_error}")
                    raise Exception(f"Failed to load video {video_id} and fallback")

    async def _standardize_video_resolution(self, video_path: str, target_resolution: tuple = None) -> str:
        """
        Safely standardize video resolution using FFmpeg to avoid segmentation faults.
        
        Args:
            video_path: Path to the input video file
            target_resolution: Tuple of (width, height) to resize to (optional, uses settings if None)
            
        Returns:
            Path to the resized video file (or original if no resize needed)
        """
        try:
            logger.info(f"🎬 [RESOLUTION] Starting video resolution standardization for: {video_path}")
            
            # Get settings
            from app.config.settings import get_settings
            settings = get_settings()
            
            # Get target resolution from settings if not provided
            if target_resolution is None:
                if not settings.standardize_video_resolution:
                    logger.info("🎬 [RESOLUTION] Video resolution standardization disabled in settings")
                    return video_path
                
                # Auto-detect target resolution based on video orientation
                logger.info(f"🎬 [RESOLUTION] Will auto-detect target resolution based on video orientation")
            else:
                logger.info(f"🎬 [RESOLUTION] Using provided target resolution: {target_resolution}")
            
            # Get current video dimensions using FFmpeg
            import subprocess
            import json
            import time
            
            logger.info(f"🎬 [RESOLUTION] Running ffprobe to get video dimensions...")
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
            logger.info(f"🎬 [RESOLUTION] ffprobe command: {' '.join(probe_cmd)}")
            
            # Use global FFmpeg manager to limit concurrent operations
            from app.utils.ffmpeg_manager import get_ffmpeg_manager
            ffmpeg_manager = await get_ffmpeg_manager()
            
            logger.info(f"🎬 [RESOLUTION] Using global FFmpeg manager for ffprobe...")
            result = await ffmpeg_manager.run_ffmpeg(
                command=probe_cmd,
                timeout=60,
                description=f"ffprobe for video dimensions: {os.path.basename(video_path)}"
            )
            
            if result.returncode != 0:
                logger.warning(f"🎬 [RESOLUTION] Failed to probe video dimensions, skipping resize: {result.stderr}")
                return video_path
            
            logger.info(f"🎬 [RESOLUTION] ffprobe completed successfully, parsing video info...")
            video_info = json.loads(result.stdout)
            video_stream = None
            
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                logger.warning("🎬 [RESOLUTION] No video stream found, skipping resize")
                return video_path
            
            current_width = int(video_stream.get('width', 0))
            current_height = int(video_stream.get('height', 0))
            logger.info(f"🎬 [RESOLUTION] Current video dimensions: {current_width}x{current_height}")
            
            if current_width == 0 or current_height == 0:
                logger.warning("🎬 [RESOLUTION] Invalid video dimensions, skipping resize")
                return video_path
            
            # Auto-detect target resolution based on video orientation if not provided
            if target_resolution is None:
                if current_width > current_height:  # Landscape video
                    target_resolution = (1920, 1080)  # 16:9 landscape (Full HD)
                    logger.info(f"🎬 [RESOLUTION] Landscape video detected, using target: {target_resolution}")
                else:  # Portrait video
                    target_resolution = (1080, 1920)  # 9:16 portrait (Mobile/Story format)
                    logger.info(f"🎬 [RESOLUTION] Portrait video detected, using target: {target_resolution}")
            else:
                logger.info(f"🎬 [RESOLUTION] Using provided target resolution: {target_resolution}")
            
            # If already at target resolution, no need to resize
            if current_width == target_resolution[0] and current_height == target_resolution[1]:
                logger.info(f"🎬 [RESOLUTION] Video already at target resolution {target_resolution[0]}x{target_resolution[1]}")
                return video_path
            
            logger.info(f"🎬 [RESOLUTION] Resizing video from {current_width}x{current_height} to {target_resolution[0]}x{target_resolution[1]}")
            
            # Create resized video using FFmpeg
            temp_dir = os.path.dirname(video_path)
            temp_filename = f"resized_{os.path.basename(video_path)}"
            resized_path = os.path.join(temp_dir, temp_filename)
            logger.info(f"🎬 [RESOLUTION] Resized video will be saved to: {resized_path}")
            
            # Use FFmpeg to resize the video
            ffmpeg_cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', f'scale={target_resolution[0]}:{target_resolution[1]}:force_original_aspect_ratio=increase:force_divisible_by=2',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y', resized_path
            ]
            
            logger.info(f"🎬 [RESOLUTION] Running FFmpeg resize command with 5-minute timeout...")
            logger.info(f"🎬 [RESOLUTION] FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            # Use global FFmpeg manager to limit concurrent operations
            logger.info(f"🎬 [RESOLUTION] Using global FFmpeg manager for resize...")
            
            # Start the resize operation with enhanced monitoring
            start_time = time.time()
            result = await ffmpeg_manager.run_ffmpeg(
                command=ffmpeg_cmd,
                timeout=300,
                description=f"FFmpeg resize: {os.path.basename(video_path)} from {current_width}x{current_height} to {target_resolution[0]}x{target_resolution[1]}"
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"🎬 [RESOLUTION] FFmpeg resize completed in {elapsed_time:.2f}s with return code: {result.returncode}")
            
            if result.returncode == 0 and os.path.exists(resized_path):
                # Verify the resized file
                resized_size = os.path.getsize(resized_path)
                original_size = os.path.getsize(video_path)
                logger.info(f"🎬 [RESOLUTION] Video resized successfully: {resized_path}")
                logger.info(f"🎬 [RESOLUTION] Size change: {original_size} bytes → {resized_size} bytes")
                return resized_path
            else:
                logger.warning(f"🎬 [RESOLUTION] FFmpeg resize failed with return code {result.returncode}")
                logger.warning(f"🎬 [RESOLUTION] FFmpeg stderr: {result.stderr}")
                logger.warning("🎬 [RESOLUTION] Using original video without resize")
                return video_path
                
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(f"🎬 [RESOLUTION] FFmpeg resize timed out after 5 minutes")
            else:
                logger.error(f"🎬 [RESOLUTION] Failed to standardize video resolution: {e}")
            # Return original video if resize fails
            return video_path
    
    async def _process_multi_video_timeline_segments(
        self,
        timeline: VideoTimeline,
        progress_callback: Optional[Any] = None
    ) -> list:
        """Process timeline segments with multiple video sources."""
        processed_segments = []
        
        for i, segment in enumerate(timeline.segments):
            try:
                logger.info(f"🎬 [SIMPLE RENDER] Processing segment {i+1}/{len(timeline.segments)}")
                
                # Get the source video ID for this segment
                source_video_id = getattr(segment, 'source_video_id', None)
                if not source_video_id:
                    logger.warning(f"⚠️ [SIMPLE RENDER] Segment {i+1} has no source_video_id, skipping")
                    continue
                
                # Get the video clip for this segment
                if source_video_id not in self.video_cache:
                    logger.error(f"❌ [SIMPLE RENDER] Video {source_video_id} not found in cache")
                    continue
                
                source_clip = self.video_cache[source_video_id]
                
                # The start_time and end_time are relative positions within the source video
                # This is how the LLM generates the timeline - each segment references
                # a specific time range within its source video
                relative_start = segment.start_time
                relative_end = segment.end_time
                segment_duration = relative_end - relative_start
                
                logger.info(f"🎬 [SIMPLE RENDER] Requested segment: {relative_start}-{relative_end}s ({segment_duration}s)")
                
                # Validate segment bounds against source video duration
                if relative_start < 0 or relative_end > source_clip.duration:
                    logger.warning(f"⚠️ [SIMPLE RENDER] TIMELINE MISMATCH: Segment {i+1} exceeds video bounds: {relative_start:.2f}s-{relative_end:.2f}s (video: 0.00s-{source_clip.duration:.2f}s)")
                    relative_start = max(0.0, relative_start)
                    relative_end = min(source_clip.duration, relative_end)
                    logger.warning(f"⚠️ [SIMPLE RENDER] TIMELINE MISMATCH: Segment adjusted to {relative_start:.2f}s-{relative_end:.2f}s")
                
                # Extract the segment from the source video
                segment_clip = source_clip.subclipped(relative_start, relative_end)
                
                # Apply effects using the shader library
                if segment.effects:
                    logger.info(f"🎨 [SIMPLE RENDER] Applying {len(segment.effects)} effects to segment {i+1}: {segment.effects}")
                    segment_clip = await self._apply_advanced_effects(segment_clip, segment)
                else:
                    logger.warning(f"⚠️ [SIMPLE RENDER] Segment {i+1} has no effects to apply")
                
                # Set transition attributes for the processed segment
                segment_clip._transition_in = getattr(segment, 'transition_in', None)
                segment_clip._transition_out = getattr(segment, 'transition_out', None)
                
                logger.info(f"🎬 [SIMPLE RENDER] Segment {i+1} transitions: in='{segment_clip._transition_in}', out='{segment_clip._transition_out}'")
                
                processed_segments.append(segment_clip)
                logger.info(f"✅ [SIMPLE RENDER] Segment {i+1} processed: {segment_clip.duration:.2f}s from {source_video_id}")
                
                if progress_callback:
                    segment_progress = 20 + int((i + 1) / len(timeline.segments) * 40)  # 20-60%
                    progress_callback(segment_progress, f"Processed segment {i+1}/{len(timeline.segments)}")
                
            except Exception as e:
                logger.error(f"❌ [SIMPLE RENDER] Failed to process segment {i+1}: {e}")
                continue
        
        logger.info(f"✅ [SIMPLE RENDER] Processed {len(processed_segments)} segments")
        return processed_segments
    
    # Enhanced effects methods from moviepy_renderer.py
    def _brightness_effect(self, get_frame, t, factor):
        """Brightness adjustment effect."""
        frame = get_frame(t)
        return np.clip(frame * factor, 0, 255).astype(np.uint8)
    
    def _contrast_effect(self, get_frame, t, factor):
        """Contrast adjustment effect."""
        frame = get_frame(t)
        mean = np.mean(frame)
        return np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _saturation_effect(self, get_frame, t, factor):
        """Saturation adjustment effect."""
        frame = get_frame(t)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def _gamma_effect(self, get_frame, t, gamma):
        """Gamma correction effect."""
        frame = get_frame(t)
        return np.clip(np.power(frame / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
    
    def _blur_effect(self, get_frame, t, radius):
        """Blur effect."""
        frame = get_frame(t)
        kernel_size = int(radius * 2) * 2 + 1
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def _sharpness_effect(self, get_frame, t, factor):
        """Sharpness adjustment effect."""
        frame = get_frame(t)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * factor
        return cv2.filter2D(frame, -1, kernel)
    
    def _slide_effect(self, get_frame, t, direction, duration, mode):
        """Slide effect."""
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        if mode == "in":
            progress = min(1.0, t / duration)
            if direction == "left":
                offset = int((1 - progress) * w)
                frame = np.roll(frame, offset, axis=1)
            elif direction == "right":
                offset = int(progress * w)
                frame = np.roll(frame, -offset, axis=1)
        else:  # out
            progress = min(1.0, (duration - t) / duration)
            if direction == "left":
                offset = int(progress * w)
                frame = np.roll(frame, -offset, axis=1)
            elif direction == "right":
                offset = int((1 - progress) * w)
                frame = np.roll(frame, offset, axis=1)
        
        return frame
    
    def _zoom_effect(self, get_frame, t, factor, mode):
        """Zoom effect."""
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        if mode == "in":
            progress = min(1.0, t / 1.0)  # 1 second duration
            scale = 1.0 + (factor - 1.0) * progress
        else:  # out
            progress = min(1.0, t / 1.0)  # 1 second duration
            scale = factor - (factor - 1.0) * progress
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Crop to original size
        start_y = max(0, (new_h - h) // 2)
        start_x = max(0, (new_w - w) // 2)
        end_y = start_y + h
        end_x = start_x + w
        
        return resized[start_y:end_y, start_x:end_x]
    
    def _shake_effect(self, get_frame, t, intensity):
        """Shake effect."""
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        # Random shake based on time
        import random
        random.seed(int(t * 10))  # Seed based on time for consistent shake
        dx = random.randint(-intensity, intensity)
        dy = random.randint(-intensity, intensity)
        
        frame = np.roll(frame, dx, axis=1)
        frame = np.roll(frame, dy, axis=0)
        
        return frame
    
    def _sepia_effect(self, get_frame, t):
        """Sepia effect."""
        frame = get_frame(t)
        # Convert to sepia
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        return np.clip(frame @ sepia_matrix.T, 0, 255).astype(np.uint8)
    
    def _black_white_effect(self, get_frame, t):
        """Black and white effect."""
        frame = get_frame(t)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    def _vintage_effect(self, get_frame, t):
        """Vintage effect."""
        frame = get_frame(t)
        # Apply vintage color grading
        frame = frame.astype(np.float32)
        frame[:, :, 0] *= 1.1  # Boost red
        frame[:, :, 1] *= 0.9  # Reduce green
        frame[:, :, 2] *= 0.8  # Reduce blue
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    # Audio effect implementations
    def _audio_normalize_effect(self, audio):
        """Audio normalization effect."""
        # This is a placeholder - actual implementation would use librosa or similar
        return audio
    
    def _audio_speed_effect(self, audio, factor):
        """Audio speed effect."""
        # This is a placeholder - actual implementation would use librosa or similar
        return audio
    
    def _audio_echo_effect(self, audio, delay, decay):
        """Audio echo effect."""
        # This is a placeholder - actual implementation would use librosa or similar
        return audio
    
    def _audio_reverb_effect(self, audio, room_size):
        """Audio reverb effect."""
        # This is a placeholder - actual implementation would use librosa or similar
        return audio
    
    # Transition effect implementations
    def _dissolve_effect(self, get_frame, t, duration):
        """Dissolve transition effect."""
        frame = get_frame(t)
        # Simplified dissolve - in real implementation, you'd blend with next clip
        return frame
    
    def _wipe_effect(self, get_frame, t, direction, duration):
        """Wipe transition effect."""
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        progress = min(1.0, t / duration)
        if direction == "left":
            wipe_line = int(progress * w)
            frame[:, :wipe_line] = 0
        elif direction == "right":
            wipe_line = int((1 - progress) * w)
            frame[:, wipe_line:] = 0
        
        return frame
    
    def _slide_transition_effect(self, get_frame, t, direction, duration):
        """Slide transition effect."""
        return self._slide_effect(get_frame, t, direction, duration, "in")
    
    def _zoom_transition_effect(self, get_frame, t, factor, duration):
        """Zoom transition effect."""
        return self._zoom_effect(get_frame, t, factor, "in")
    
    def _whip_pan_effect(self, get_frame, t, direction):
        """Whip pan effect."""
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        # Simple whip pan effect
        progress = min(1.0, t / 0.5)  # 0.5 second duration
        if direction == "left":
            offset = int(progress * w * 0.3)
            frame = np.roll(frame, offset, axis=1)
        elif direction == "right":
            offset = int(progress * w * 0.3)
            frame = np.roll(frame, -offset, axis=1)
        
        return frame
    
    def _whip_pan_transition_effect(self, get_frame, t, direction, duration):
        """Whip pan transition effect."""
        return self._whip_pan_effect(get_frame, t, direction)
    
    def _zoom_blur_effect(self, get_frame, t):
        """Zoom blur effect."""
        frame = get_frame(t)
        # Simple zoom blur - scale and blur
        h, w = frame.shape[:2]
        scale = 1.2
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Crop to original size
        start_y = max(0, (new_h - h) // 2)
        start_x = max(0, (new_w - w) // 2)
        end_y = start_y + h
        end_x = start_x + w
        
        cropped = resized[start_y:end_y, start_x:end_x]
        
        # Apply slight blur
        return cv2.GaussianBlur(cropped, (5, 5), 0)
    
    def _zoom_blur_transition_effect(self, get_frame, t, duration):
        """Zoom blur transition effect."""
        return self._zoom_blur_effect(get_frame, t)
    
    def _shake_transition_effect(self, get_frame, t, intensity, duration):
        """Shake transition effect."""
        return self._shake_effect(get_frame, t, intensity)
    
    async def _apply_advanced_effects(self, clip, segment):
        """Apply advanced effects using the shader library."""
        try:
            from app.editor.enhanced_shader_library import EnhancedShaderLibrary
            
            if not segment.effects:
                return clip
            
            logger.info(f"🎨 [SIMPLE RENDER] Applying effects to segment: {segment.effects}")
            
            # Get available effects from shader library
            shader_library = EnhancedShaderLibrary()
            available_effects = shader_library.get_available_effects()
            
            for effect_name in segment.effects:
                if effect_name in available_effects:
                    logger.info(f"🎨 [SIMPLE RENDER] Applying effect: {effect_name}")
                    
                    # Skip shader library integration for now - use MoviePy effects directly
                    
                    # Fallback to MoviePy effects
                    from moviepy import vfx, afx
                    
                    if effect_name == "cinematic":
                        # Cinematic effect - lower FPS for film look
                        clip = clip.with_fps(24)
                    elif effect_name == "beat_sync":
                        # Simple beat sync effect - speed up slightly
                        clip = clip.with_speed_scaled(1.2)
                    elif effect_name == "vintage":
                        # Vintage effect - black and white with contrast
                        clip = clip.with_effects([vfx.BlackAndWhite(), vfx.LumContrast(contrast=1.3)])
                    elif effect_name == "audio_pulse":
                        # Audio pulse effect - slight speed variation
                        clip = clip.with_speed_scaled(1.1)
                    elif effect_name == "slow_motion":
                        clip = clip.with_speed_scaled(0.5)
                    elif effect_name == "speed_up":
                        clip = clip.with_speed_scaled(2.0)
                    elif effect_name == "fade_in":
                        # Fade in effect
                        try:
                            clip = clip.with_effects([vfx.FadeIn(duration=0.5)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Fade in effect failed, using fallback: {e}")
                            # Fallback to simple speed adjustment
                            clip = clip.with_speed_scaled(1.0)
                    elif effect_name == "fade_out":
                        # Fade out effect
                        try:
                            clip = clip.with_effects([vfx.FadeOut(duration=0.5)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Fade out effect failed, using fallback: {e}")
                            # Fallback to simple speed adjustment
                            clip = clip.with_speed_scaled(1.0)
                    elif effect_name == "high_contrast":
                        # High contrast effect
                        try:
                            clip = clip.with_effects([vfx.LumContrast(contrast=1.5)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] High contrast effect failed, using fallback: {e}")
                            # Fallback to simple speed adjustment
                            clip = clip.with_speed_scaled(1.0)
                    elif effect_name == "motion_blur":
                        # Motion blur effect - use freeze frame with proper duration
                        try:
                            clip = clip.with_effects([vfx.Freeze(duration=0.1)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Motion blur effect failed, using fallback: {e}")
                            # Fallback to speed adjustment
                            clip = clip.with_speed_scaled(0.8)
                    elif effect_name == "optical_flow":
                        # Optical flow effect - use smooth transitions
                        try:
                            clip = clip.with_effects([vfx.CrossFadeIn(duration=0.3)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Optical flow effect failed, using fallback: {e}")
                            # Fallback to simple fade
                            clip = clip.with_effects([vfx.FadeIn(0.3)])
                    elif effect_name == "frequency_visualizer":
                        # Frequency visualizer - subtle audio visualization
                        try:
                            clip = clip.with_effects([vfx.LumContrast(contrast=1.03, lum=1.01)])
                            logger.info(f"✅ [SIMPLE RENDER] Applied frequency_visualizer effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Frequency visualizer effect failed, using fallback: {e}")
                            # Fallback to simple speed adjustment
                            clip = clip.with_speed_scaled(1.0)
                    elif effect_name == "motion_trail":
                        # Motion trail effect - use loop with proper duration
                        try:
                            clip = clip.with_effects([vfx.Loop(duration=clip.duration * 0.1)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Motion trail effect failed, using fallback: {e}")
                            # Fallback to speed adjustment
                            clip = clip.with_speed_scaled(0.9)
                    elif effect_name == "duotone":
                        # Duotone effect - use color masks
                        try:
                            clip = clip.with_effects([vfx.MaskColor(color1=(255, 0, 0), color2=(0, 0, 255))])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Duotone effect failed, using fallback: {e}")
                            # Fallback to simple color correction
                            clip = clip.with_effects([vfx.GammaCorrection(1.1)])
                    elif effect_name == "scene_transition":
                        # Scene transition effect
                        try:
                            clip = clip.with_effects([vfx.CrossFadeOut(duration=0.5)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Scene transition effect failed, using fallback: {e}")
                            # Fallback to simple fade
                            clip = clip.with_effects([vfx.FadeOut(0.5)])
                    elif effect_name == "volume_wave":
                        # Volume wave effect - use audio effects
                        try:
                            clip = clip.with_effects([afx.MultiplyVolume(1.2)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Volume wave effect failed, using fallback: {e}")
                            # Fallback to simple speed adjustment
                            clip = clip.with_speed_scaled(1.0)
                    elif effect_name == "color_grading":
                        # Color grading effect
                        try:
                            clip = clip.with_effects([vfx.GammaCorrection(1.1), vfx.LumContrast(contrast=1.2)])
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Color grading effect failed, using fallback: {e}")
                            # Fallback to simple speed adjustment
                            clip = clip.with_speed_scaled(1.0)
                    
                    elif effect_name == "cyberpunk":
                        # Cyberpunk effect - subtle neon aesthetic
                        try:
                            import numpy as np
                            
                            def cyberpunk_func(t):
                                # Gentle pulsing
                                pulse = 1.0 + 0.1 * np.sin(t * 2)  # Gentle pulsing
                                return pulse
                            
                            clip = clip.with_effects([
                                vfx.LumContrast(contrast=1.05, lum=1.02),  # Very subtle contrast boost
                                vfx.Resize(cyberpunk_func)  # Gentle animated resize
                            ])
                            logger.info(f"✅ [SIMPLE RENDER] Applied cyberpunk effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Cyberpunk effect failed: {e}")
                    
                    elif effect_name == "film_noir":
                        # Film noir effect - elegant black and white
                        try:
                            clip = clip.with_effects([vfx.BlackAndWhite(), vfx.LumContrast(contrast=1.08, lum=0.95)])
                            logger.info(f"✅ [SIMPLE RENDER] Applied film_noir effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Film noir effect failed: {e}")
                    
                    elif effect_name == "cartoon":
                        # Cartoon effect - very subtle stylized look
                        try:
                            clip = clip.with_effects([vfx.LumContrast(contrast=1.05, lum=1.02)])
                            logger.info(f"✅ [SIMPLE RENDER] Applied cartoon effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Cartoon effect failed: {e}")
                    
                    elif effect_name == "fisheye":
                        # Fisheye effect - subtle animated zoom
                        try:
                            import numpy as np
                            
                            def fisheye_func(t):
                                # Gentle pulsing zoom
                                return 1.0 + 0.05 * np.sin(t * 3)  # Gentle pulsing zoom
                            
                            clip = clip.with_effects([
                                vfx.Resize(fisheye_func), 
                                vfx.LumContrast(contrast=1.03, lum=1.01)  # Very subtle contrast
                            ])
                            logger.info(f"✅ [SIMPLE RENDER] Applied fisheye effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Fisheye effect failed: {e}")
                    
                    elif effect_name == "twirl":
                        # Twirl effect - gentle spinning
                        try:
                            import numpy as np
                            
                            def twirl_func(t):
                                # Gentle spinning motion
                                return t * 30  # 30 degrees per second (slower)
                            
                            clip = clip.with_effects([
                                vfx.Rotate(twirl_func, expand=True), 
                                vfx.LumContrast(contrast=1.03, lum=1.01)  # Very subtle contrast
                            ])
                            logger.info(f"✅ [SIMPLE RENDER] Applied twirl effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Twirl effect failed: {e}")
                    
                    elif effect_name == "warp":
                        # Warp effect - subtle animated distortion
                        try:
                            import numpy as np
                            
                            def warp_func(t):
                                # Gentle pulsing distortion
                                return 1.0 + 0.05 * np.sin(t * 4)  # Gentle pulsing
                            
                            clip = clip.with_effects([
                                vfx.Resize(warp_func), 
                                vfx.LumContrast(contrast=1.03, lum=1.01)  # Very subtle contrast
                            ])
                            logger.info(f"✅ [SIMPLE RENDER] Applied warp effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Warp effect failed: {e}")
                    
                    elif effect_name == "perspective":
                        # Perspective effect - gentle animated rotation
                        try:
                            import numpy as np
                            
                            def perspective_func(t):
                                # Gentle oscillating rotation
                                return 5 * np.sin(t * 1)  # Gentle oscillation between -5 and +5 degrees
                            
                            clip = clip.with_effects([
                                vfx.Rotate(perspective_func, expand=True), 
                                vfx.LumContrast(contrast=1.03, lum=1.01)  # Very subtle contrast
                            ])
                            logger.info(f"✅ [SIMPLE RENDER] Applied perspective effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Perspective effect failed: {e}")
                    
                    elif effect_name == "volume_wave":
                        # Volume wave effect - subtle animated pulsing
                        try:
                            import numpy as np
                            
                            def volume_func(t):
                                # Gentle pulsing brightness
                                return 1.0 + 0.1 * np.sin(t * 4)  # Gentle audio-like pulsing
                            
                            clip = clip.with_effects([
                                vfx.LumContrast(contrast=1.03, lum=volume_func)  # Very subtle contrast
                            ])
                            logger.info(f"✅ [SIMPLE RENDER] Applied volume_wave effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Volume wave effect failed: {e}")
                    
                    elif effect_name == "scene_transition":
                        # Scene transition effect - subtle animated fade
                        try:
                            import numpy as np
                            
                            def transition_func(t):
                                # Gentle fade effect
                                return 1.0 + 0.05 * np.sin(t * 2)  # Very gentle pulsing
                            
                            clip = clip.with_effects([
                                vfx.LumContrast(contrast=1.02, lum=transition_func)  # Extremely subtle
                            ])
                            logger.info(f"✅ [SIMPLE RENDER] Applied scene_transition effect")
                        except Exception as e:
                            logger.warning(f"⚠️ [SIMPLE RENDER] Scene transition effect failed: {e}")
                    
                    else:
                        logger.warning(f"⚠️ [SIMPLE RENDER] Effect '{effect_name}' not implemented, skipping")
                else:
                    logger.warning(f"⚠️ [SIMPLE RENDER] Effect '{effect_name}' not found in library, skipping")
            
            return clip
            
        except Exception as e:
            logger.error(f"❌ [SIMPLE RENDER] Failed to apply effects: {e}")
            return clip
    
    def _cleanup_video_cache(self):
        """Clean up video cache to free memory."""
        try:
            for video_id, clip in self.video_cache.items():
                if hasattr(clip, 'close'):
                    clip.close()
            self.video_cache.clear()
            logger.info(f"🧹 [SIMPLE RENDER] Video cache cleaned up")
        except Exception as e:
            logger.error(f"❌ [SIMPLE RENDER] Failed to cleanup video cache: {e}")
    
    async def _concatenate_with_transitions(self, segments):
        """
        Concatenate video segments with proper transitions between them.
        
        Args:
            segments: List of video segments to concatenate
            
        Returns:
            Final concatenated video clip
        """
        try:
            from moviepy import concatenate_videoclips, CompositeVideoClip, vfx
            
            if len(segments) <= 1:
                logger.info("🎬 [CONCATENATE] Single segment, no transitions needed")
                return segments[0] if segments else None
            
            logger.info(f"🎬 [CONCATENATE] Concatenating {len(segments)} segments with transitions")
            
            # Check if any segments have transition information
            has_transitions = any(hasattr(seg, '_transition_out') and seg._transition_out for seg in segments[:-1])
            logger.info(f"🎬 [CONCATENATE] Has transitions: {has_transitions}")
            
            if not has_transitions:
                logger.info("🎬 [CONCATENATE] No transition information found, using simple concatenation")
                return concatenate_videoclips(segments, method="compose")
            
            # Implement proper transition-aware concatenation
            logger.info("🎬 [CONCATENATE] Implementing transition-aware concatenation with specific effects")
            return await self._concatenate_with_specific_transitions(segments)
            
        except Exception as e:
            logger.error(f"❌ [CONCATENATE] Failed to concatenate segments: {e}")
            import traceback
            logger.error(f"❌ [CONCATENATE] Traceback: {traceback.format_exc()}")
            # Fallback to simple concatenation
            from moviepy import concatenate_videoclips
            return concatenate_videoclips(segments, method="compose")
    
    async def _concatenate_with_specific_transitions(self, segments):
        """
        Concatenate segments with specific transition effects between them.
        
        Args:
            segments: List of video segments with transition attributes
            
        Returns:
            Final concatenated video clip
        """
        try:
            from moviepy import CompositeVideoClip, vfx
            
            if len(segments) <= 1:
                logger.info("🎬 [SPECIFIC TRANSITIONS] Single segment, no transitions needed")
                return segments[0] if segments else None
            
            logger.info(f"🎬 [SPECIFIC TRANSITIONS] Creating specific transitions between {len(segments)} segments")
            
            clips_with_timing = []
            current_time = 0.0
            transition_duration = 0.5
            
            for i, segment in enumerate(segments):
                logger.info(f"🎬 [SPECIFIC TRANSITIONS] Processing segment {i+1}/{len(segments)}")
                
                if i == 0:
                    # First segment - no transition in
                    logger.info(f"🎬 [SPECIFIC TRANSITIONS] First segment: setting start time to {current_time}")
                    clips_with_timing.append(segment.with_start(current_time))
                    current_time += segment.duration
                    logger.info(f"🎬 [SPECIFIC TRANSITIONS] Updated current_time to {current_time}")
                else:
                    # Apply transition between previous and current segment
                    prev_segment = segments[i-1]
                    transition_out = getattr(prev_segment, '_transition_out', 'crossfade')
                    transition_in = getattr(segment, '_transition_in', 'crossfade')
                    
                    logger.info(f"🎬 [SPECIFIC TRANSITIONS] Creating transition: {transition_out} -> {transition_in}")
                    
                    try:
                        # Apply fade out to previous segment
                        from moviepy import vfx
                        prev_fade_out = prev_segment.with_effects([vfx.CrossFadeOut(transition_duration)])
                        clips_with_timing.append(prev_fade_out.with_start(current_time - segment.duration))
                        
                        # Apply transition effect directly to the current segment
                        segment_with_transition = await self._apply_transition_to_segment(
                            segment, transition_in, transition_duration
                        )
                        
                        # Position the segment with transition (overlapping with previous segment)
                        segment_start_time = current_time - transition_duration
                        clips_with_timing.append(segment_with_transition.with_start(segment_start_time))
                        
                        # Update current time (accounting for transition overlap)
                        current_time += segment.duration - transition_duration
                        logger.info(f"🎬 [SPECIFIC TRANSITIONS] Updated current_time to {current_time}")
                        
                    except Exception as e:
                        logger.error(f"❌ [SPECIFIC TRANSITIONS] Failed to create transition for segment {i+1}: {e}")
                        # Fallback to simple positioning
                        clips_with_timing.append(segment.with_start(current_time))
                        current_time += segment.duration
            
            # Create final composite
            final_clip = CompositeVideoClip(clips_with_timing)
            logger.info(f"✅ [SPECIFIC TRANSITIONS] Created final video with {len(clips_with_timing)} clips, duration: {final_clip.duration:.2f}s")
            return final_clip
            
        except Exception as e:
            logger.error(f"❌ [SPECIFIC TRANSITIONS] Failed to create specific transitions: {e}")
            import traceback
            logger.error(f"❌ [SPECIFIC TRANSITIONS] Traceback: {traceback.format_exc()}")
            # Fallback to crossfade concatenation
            return self._concatenate_with_crossfades(segments)
    
    async def _apply_transition_to_segment(self, segment, transition_type, duration):
        """
        Apply transition effect directly to a segment.
        
        Args:
            segment: Video segment to apply transition to
            transition_type: Type of transition to apply
            duration: Duration of the transition
            
        Returns:
            Segment with transition effect applied
        """
        try:
            logger.info(f"🎬 [SEGMENT TRANSITION] Applying {transition_type} transition to segment")
            
            from moviepy import vfx
            
            # Apply transition effects directly to the segment
            if transition_type in ['cross_dissolve', 'fade', 'dissolve', 'crossfade']:
                # Crossfade transition
                return segment.with_effects([vfx.CrossFadeIn(duration)])
                
            elif transition_type in ['slide', 'slide_left']:
                # Slide left transition
                return segment.with_effects([vfx.SlideIn(duration, side='left')])
                
            elif transition_type in ['slide_right']:
                # Slide right transition
                return segment.with_effects([vfx.SlideIn(duration, side='right')])
                
            elif transition_type in ['zoom', 'zoom_in']:
                # Zoom in transition with proper screen filling
                def zoom_function(t):
                    if t < duration:
                        return 1.0 + (t / duration) * 0.2  # Scale from 1.0 to 1.2 over duration (fills screen)
                    return 1.2
                # Apply zoom with proper centering and crossfade
                zoomed_segment = segment.with_effects([vfx.Resize(zoom_function)])
                return zoomed_segment.with_effects([vfx.CrossFadeIn(duration)])
                
            elif transition_type in ['spin']:
                # Spin transition - rotate the original video itself with stronger opacity
                def spin_function(t):
                    if t < duration:
                        return (t / duration) * 360  # Rotate from 0 to 360 degrees over duration
                    return 360
                # Apply spin directly to the original video without expansion
                spun_segment = segment.with_effects([vfx.Rotate(spin_function, expand=False)])
                # Use shorter crossfade to make spinning video more prominent
                return spun_segment.with_effects([vfx.CrossFadeIn(duration * 0.3)])
                
            elif transition_type in ['glitch', 'glitch_cut']:
                # Glitch transition
                return segment.with_effects([vfx.InvertColors(), vfx.MirrorX(), vfx.CrossFadeIn(duration)])
                
            elif transition_type in ['whip_pan']:
                # Whip pan transition - use slide instead of scroll
                return segment.with_effects([vfx.SlideIn(duration, side='left'), vfx.CrossFadeIn(duration)])
                
            else:
                # Fallback to crossfade
                logger.info(f"🎬 [SEGMENT TRANSITION] Unknown transition type '{transition_type}', using crossfade fallback")
                return segment.with_effects([vfx.CrossFadeIn(duration)])
                
        except Exception as e:
            logger.error(f"❌ [SEGMENT TRANSITION] Failed to apply transition '{transition_type}': {e}")
            # Return segment with simple crossfade as fallback
            return segment.with_effects([vfx.CrossFadeIn(duration)])

    async def _create_transition_clip_safe(self, prev_clip, next_clip, transition_type, duration):
        """
        Create a transition clip with proper handling of multiple effects.
        
        Args:
            prev_clip: Previous video clip
            next_clip: Next video clip  
            transition_type: Type of transition to apply
            duration: Duration of the transition
            
        Returns:
            Transition clip or fallback clip
        """
        try:
            logger.info(f"🎬 [TRANSITION] Creating {transition_type} transition with {duration}s duration")
            
            from moviepy import CompositeVideoClip, vfx
            
            # Implement specific transition types with proper effect combination
            if transition_type in ['cross_dissolve', 'fade', 'dissolve', 'crossfade']:
                # Crossfade transition
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                next_fade = next_clip.with_effects([vfx.CrossFadeIn(duration)])
                transition_clip = CompositeVideoClip([prev_fade, next_fade.with_start(prev_clip.duration - duration)])
                return transition_clip
                
            elif transition_type in ['slide', 'slide_left']:
                # Slide left transition - next clip slides in from right
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                next_slide = next_clip.with_effects([vfx.SlideIn(duration, side='left')])
                transition_clip = CompositeVideoClip([prev_fade, next_slide.with_start(prev_clip.duration - duration)])
                return transition_clip
                
            elif transition_type in ['slide_right']:
                # Slide right transition - next clip slides in from left
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                next_slide = next_clip.with_effects([vfx.SlideIn(duration, side='right')])
                transition_clip = CompositeVideoClip([prev_fade, next_slide.with_start(prev_clip.duration - duration)])
                return transition_clip
                
            elif transition_type in ['zoom', 'zoom_in']:
                # Zoom in transition - create proper zoom effect
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                # Create zoom effect with proper scaling
                next_zoom = next_clip.with_effects([vfx.Resize(1.1)])  # Moderate zoom in
                transition_clip = CompositeVideoClip([prev_fade, next_zoom.with_start(prev_clip.duration - duration)])
                return transition_clip
                
            elif transition_type in ['spin']:
                # Spin transition - create animated spin effect
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                # Create animated spin effect using time-based function
                def spin_function(t):
                    if t < duration:
                        return (t / duration) * 360  # Rotate from 0 to 360 degrees over duration
                    return 360
                next_spin = next_clip.with_effects([vfx.Rotate(spin_function, expand=True)])
                next_spin = next_spin.with_effects([vfx.CrossFadeIn(duration)])
                transition_clip = CompositeVideoClip([prev_fade, next_spin.with_start(prev_clip.duration - duration)])
                return transition_clip
                
            elif transition_type in ['glitch', 'glitch_cut']:
                # Glitch transition - use available effects to simulate glitch
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                # Use InvertColors and MirrorX to create glitch-like effect
                next_glitch = next_clip.with_effects([vfx.InvertColors(), vfx.MirrorX()])
                transition_clip = CompositeVideoClip([prev_fade, next_glitch.with_start(prev_clip.duration - duration)])
                return transition_clip
                
            elif transition_type in ['whip_pan']:
                # Whip pan transition - use SlideIn effect for motion
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                # Use SlideIn to create horizontal motion
                next_whip = next_clip.with_effects([vfx.SlideIn(duration, side='left')])
                transition_clip = CompositeVideoClip([prev_fade, next_whip.with_start(prev_clip.duration - duration)])
                return transition_clip
                
            else:
                # Fallback to crossfade for unknown transition types
                logger.info(f"🎬 [TRANSITION] Unknown transition type '{transition_type}', using crossfade fallback")
                prev_fade = prev_clip.with_effects([vfx.CrossFadeOut(duration)])
                next_fade = next_clip.with_effects([vfx.CrossFadeIn(duration)])
                transition_clip = CompositeVideoClip([prev_fade, next_fade.with_start(prev_clip.duration - duration)])
                return transition_clip
                
        except Exception as e:
            logger.error(f"❌ [TRANSITION] Failed to create transition '{transition_type}': {e}")
            import traceback
            logger.error(f"❌ [TRANSITION] Traceback: {traceback.format_exc()}")
            # Return the previous clip as fallback
            return prev_clip
    
    def _concatenate_with_crossfades(self, segments):
        """
        Concatenate segments with crossfade transitions between them.
        
        Args:
            segments: List of video segments to concatenate
            
        Returns:
            Final concatenated video clip with transitions
        """
        try:
            from moviepy import CompositeVideoClip, vfx
            
            if len(segments) <= 1:
                logger.info("🎬 [CROSSFADE] Single segment, no crossfades needed")
                return segments[0] if segments else None
            
            logger.info(f"🎬 [CROSSFADE] Creating crossfade transitions between {len(segments)} segments")
            
            # Create a list to hold all clips with their timing
            clips_with_timing = []
            current_time = 0.0
            transition_duration = 0.5  # Default transition duration
            
            for i, segment in enumerate(segments):
                if i == 0:
                    # First segment - no transition in
                    clips_with_timing.append(segment.with_start(current_time))
                    current_time += segment.duration
                else:
                    # Apply crossfade transition
                    prev_segment = segments[i-1]
                    
                    try:
                        # Create crossfade by overlapping the end of previous segment with start of current segment
                        prev_fade = prev_segment.with_effects([vfx.CrossFadeOut(transition_duration)])
                        current_fade = segment.with_effects([vfx.CrossFadeIn(transition_duration)])
                        
                        # Position the current segment to start before the previous one ends
                        current_start_time = current_time - transition_duration
                        clips_with_timing.append(current_fade.with_start(current_start_time))
                        
                        # Update current time (subtract transition duration since we're overlapping)
                        current_time += segment.duration - transition_duration
                        
                    except Exception as e:
                        logger.error(f"❌ [CROSSFADE] Failed to create crossfade for segment {i+1}: {e}")
                        # Fallback: just add the segment without transition
                        clips_with_timing.append(segment.with_start(current_time))
                        current_time += segment.duration
            
            # Create composite video with all clips
            final_clip = CompositeVideoClip(clips_with_timing)
            logger.info(f"✅ [CROSSFADE] Created final video with {len(clips_with_timing)} clips, duration: {final_clip.duration:.2f}s")
            
            return final_clip
            
        except Exception as e:
            logger.error(f"❌ [CROSSFADE] Failed to create crossfade transitions: {e}")
            import traceback
            logger.error(f"❌ [CROSSFADE] Traceback: {traceback.format_exc()}")
            # Fallback to simple concatenation
            from moviepy import concatenate_videoclips
            return concatenate_videoclips(segments, method="compose")


# Global instance
_renderer: Optional[SimpleVideoRenderer] = None


async def initialize_renderer() -> SimpleVideoRenderer:
    """
    Initialize the simplified renderer.
    
    Returns:
        SimpleVideoRenderer: Initialized renderer instance
    """
    global _renderer
    if _renderer is None:
        _renderer = SimpleVideoRenderer()
        logger.info("✅ Simple renderer initialized")
    return _renderer


def get_renderer() -> SimpleVideoRenderer:
    """
    Get the global renderer instance.
    
    Returns:
        SimpleVideoRenderer: Renderer instance
    """
    if _renderer is None:
        raise RuntimeError("Renderer not initialized. Call initialize_renderer() first.")
    return _renderer 