"""
Music Integration System

This module integrates the music library with the video editing pipeline,
providing automatic music selection and audio mixing capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import subprocess
import tempfile
import os

from app.music.library import MusicLibrary, MusicTrack, MusicGenre, MusicMood, get_music_library
from app.models.schemas import VideoAnalysisResult, TimelineSegment

logger = logging.getLogger(__name__)


class MusicIntegrationEngine:
    """
    Music integration engine for video editing.
    
    Handles automatic music selection, audio mixing, and integration
    with the video editing pipeline.
    """
    
    def __init__(self, music_library: Optional[MusicLibrary] = None):
        """Initialize the music integration engine."""
        self.music_library = music_library or get_music_library()
        self.temp_directory = Path(tempfile.gettempdir()) / "video_editing_music"
        self.temp_directory.mkdir(exist_ok=True)
        
        logger.info("Music integration engine initialized")
    
    def analyze_video_content(self, analysis_results: List[VideoAnalysisResult]) -> Dict[str, Any]:
        """
        Analyze video content to determine appropriate music characteristics.
        
        Args:
            analysis_results: List of video analysis results
            
        Returns:
            Content analysis with music recommendations
        """
        if not analysis_results:
            return {
                "mood": MusicMood.CALM,
                "genre": MusicGenre.CINEMATIC,
                "energy_level": "medium",
                "confidence": 0.5
            }
        
        # Analyze overall content characteristics
        total_duration = sum(result.duration for result in analysis_results)
        avg_motion = sum(
            sum(result.motion_analysis.motion_levels) / len(result.motion_analysis.motion_levels)
            for result in analysis_results
            if result.motion_analysis.motion_levels
        ) / len(analysis_results) if analysis_results else 0
        
        avg_volume = sum(
            sum(result.audio_analysis.volume_levels) / len(result.audio_analysis.volume_levels)
            for result in analysis_results
            if result.audio_analysis.volume_levels
        ) / len(analysis_results) if analysis_results else 0
        
        # Determine mood based on content analysis
        mood = self._determine_mood(avg_motion, avg_volume, total_duration)
        genre = self._determine_genre(avg_motion, avg_volume, total_duration)
        energy_level = self._determine_energy_level(avg_motion, avg_volume)
        
        # Calculate confidence based on analysis quality
        confidence = min(1.0, len(analysis_results) * 0.2 + 0.3)
        
        return {
            "mood": mood,
            "genre": genre,
            "energy_level": energy_level,
            "confidence": confidence,
            "total_duration": total_duration,
            "avg_motion": avg_motion,
            "avg_volume": avg_volume
        }
    
    def _determine_mood(self, motion: float, volume: float, duration: float) -> MusicMood:
        """Determine music mood based on content analysis."""
        # High motion and volume = exciting
        if motion > 0.7 and volume > 0.6:
            return MusicMood.EXCITING
        # Low motion and volume = calm
        elif motion < 0.3 and volume < 0.4:
            return MusicMood.CALM
        # Medium values = happy
        elif 0.4 <= motion <= 0.6 and 0.4 <= volume <= 0.6:
            return MusicMood.HAPPY
        # Very low values = mysterious
        elif motion < 0.2 and volume < 0.2:
            return MusicMood.MYSTERIOUS
        # High motion but low volume = dramatic
        elif motion > 0.6 and volume < 0.4:
            return MusicMood.DRAMATIC
        # Default to calm
        else:
            return MusicMood.CALM
    
    def _determine_genre(self, motion: float, volume: float, duration: float) -> MusicGenre:
        """Determine music genre based on content analysis."""
        # Short, high-energy content = electronic/dance
        if duration < 60 and motion > 0.6 and volume > 0.5:
            return MusicGenre.ELECTRONIC
        # Long, dramatic content = cinematic
        elif duration > 180 and motion > 0.5:
            return MusicGenre.CINEMATIC
        # Corporate/professional content = corporate
        elif motion < 0.4 and volume < 0.5:
            return MusicGenre.CORPORATE
        # Ambient for very calm content
        elif motion < 0.3 and volume < 0.3:
            return MusicGenre.AMBIENT
        # Default to cinematic
        else:
            return MusicGenre.CINEMATIC
    
    def _determine_energy_level(self, motion: float, volume: float) -> str:
        """Determine energy level based on content analysis."""
        energy_score = (motion + volume) / 2
        
        if energy_score > 0.7:
            return "high"
        elif energy_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def select_background_music(
        self,
        video_duration: float,
        content_analysis: Optional[Dict[str, Any]] = None,
        preferred_genre: Optional[MusicGenre] = None,
        preferred_mood: Optional[MusicMood] = None
    ) -> Optional[MusicTrack]:
        """
        Select background music for a video.
        
        Args:
            video_duration: Duration of the video in seconds
            content_analysis: Content analysis results
            preferred_genre: Preferred music genre
            preferred_mood: Preferred music mood
            
        Returns:
            Selected music track or None
        """
        try:
            # Use content analysis if available
            if content_analysis:
                genre = preferred_genre or content_analysis.get("genre")
                mood = preferred_mood or content_analysis.get("mood")
            else:
                genre = preferred_genre
                mood = preferred_mood
            
            # Get music suggestions
            tracks = self.music_library.get_tracks_for_video(
                video_duration=video_duration,
                content_mood=mood,
                preferred_genre=genre
            )
            
            if not tracks:
                logger.warning("No suitable music tracks found")
                return None
            
            # Select the best match (first in sorted list)
            selected_track = tracks[0]
            logger.info(f"Selected music: {selected_track.title} by {selected_track.artist}")
            
            return selected_track
            
        except Exception as e:
            logger.error(f"Failed to select background music: {e}")
            return None
    
    def create_audio_mix_plan(
        self,
        segments: List[TimelineSegment],
        background_music: MusicTrack,
        mix_strategy: str = "background"
    ) -> Dict[str, Any]:
        """
        Create an audio mixing plan for video segments with background music.
        
        Args:
            segments: Timeline segments
            background_music: Selected background music track
            mix_strategy: Mixing strategy (background, overlay, replace)
            
        Returns:
            Audio mixing plan
        """
        total_duration = sum(segment.duration for segment in segments)
        
        # Calculate music timing
        music_start = 0
        music_duration = min(background_music.duration, total_duration)
        
        # Create mixing plan
        mix_plan = {
            "background_music": {
                "track_id": background_music.id,
                "file_path": background_music.file_path,
                "start_time": music_start,
                "duration": music_duration,
                "volume": 0.3 if mix_strategy == "background" else 0.7,
                "fade_in": 2.0,
                "fade_out": 2.0
            },
            "segments": [],
            "strategy": mix_strategy,
            "total_duration": total_duration
        }
        
        # Process each segment
        current_time = 0
        for segment in segments:
            segment_audio = {
                "segment_id": str(segment.id),
                "start_time": current_time,
                "duration": segment.duration,
                "volume": 0.8 if mix_strategy == "background" else 0.4,
                "fade_in": 0.5,
                "fade_out": 0.5,
                "mix_with_music": mix_strategy != "replace"
            }
            
            mix_plan["segments"].append(segment_audio)
            current_time += segment.duration
        
        return mix_plan
    
    def apply_audio_mix(
        self,
        video_file: str,
        mix_plan: Dict[str, Any],
        output_file: str
    ) -> bool:
        """
        Apply audio mixing to create the final video with background music.
        
        Args:
            video_file: Input video file path
            mix_plan: Audio mixing plan
            output_file: Output video file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            background_music = mix_plan["background_music"]
            music_file = background_music["file_path"]
            
            # Create temporary audio file for mixing
            temp_audio = self.temp_directory / "mixed_audio.wav"
            
            # Build FFmpeg command for audio mixing
            cmd = [
                "ffmpeg",
                "-i", video_file,
                "-i", music_file,
                "-filter_complex",
                f"[0:a]volume={background_music['volume']}[v0];"
                f"[1:a]volume={background_music['volume']},"
                f"afade=t=in:st={background_music['start_time']}:d={background_music['fade_in']},"
                f"afade=t=out:st={background_music['start_time'] + background_music['duration'] - background_music['fade_out']}:d={background_music['fade_out']}[m0];"
                f"[v0][m0]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-y",
                str(temp_audio)
            ]
            
            # Execute FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg audio mixing failed: {result.stderr}")
                return False
            
            # Move temporary file to output
            os.rename(str(temp_audio), output_file)
            
            logger.info(f"Audio mixing completed: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply audio mix: {e}")
            return False
    
    def create_music_preview(
        self,
        music_track: MusicTrack,
        duration: float = 30.0,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a preview of a music track.
        
        Args:
            music_track: Music track to preview
            duration: Preview duration in seconds
            output_file: Output file path (optional)
            
        Returns:
            Path to preview file or None
        """
        try:
            if not output_file:
                output_file = str(self.temp_directory / f"preview_{music_track.id}.mp3")
            
            # Create preview using FFmpeg
            cmd = [
                "ffmpeg",
                "-i", music_track.file_path,
                "-t", str(duration),
                "-c:a", "mp3",
                "-b:a", "128k",
                "-y",
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to create music preview: {result.stderr}")
                return None
            
            logger.info(f"Music preview created: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to create music preview: {e}")
            return None
    
    def get_music_recommendations(
        self,
        video_duration: float,
        content_analysis: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get music recommendations for a video.
        
        Args:
            video_duration: Duration of the video in seconds
            content_analysis: Content analysis results
            limit: Maximum number of recommendations
            
        Returns:
            List of music recommendations with metadata
        """
        try:
            # Select background music
            selected_music = self.select_background_music(
                video_duration=video_duration,
                content_analysis=content_analysis
            )
            
            if not selected_music:
                return []
            
            # Get additional recommendations
            genre = content_analysis.get("genre") if content_analysis else None
            mood = content_analysis.get("mood") if content_analysis else None
            
            tracks = self.music_library.get_tracks_for_video(
                video_duration=video_duration,
                content_mood=mood,
                preferred_genre=genre
            )
            
            # Create recommendations
            recommendations = []
            for i, track in enumerate(tracks[:limit]):
                recommendation = {
                    "track": track.to_dict(),
                    "match_score": 1.0 - (i * 0.1),  # Decreasing match score
                    "reason": self._get_recommendation_reason(track, content_analysis),
                    "preview_url": f"/api/v1/music/preview/{track.id}"
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get music recommendations: {e}")
            return []
    
    def _get_recommendation_reason(
        self,
        track: MusicTrack,
        content_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """Get a human-readable reason for the recommendation."""
        reasons = []
        
        if content_analysis:
            if track.mood == content_analysis.get("mood"):
                reasons.append("matches content mood")
            if track.genre == content_analysis.get("genre"):
                reasons.append("matches content genre")
        
        # Duration match
        if abs(track.duration - content_analysis.get("total_duration", 0)) < 30:
            reasons.append("duration matches video")
        
        # BPM match for energetic content
        if content_analysis and content_analysis.get("energy_level") == "high" and track.bpm and track.bpm > 120:
            reasons.append("high energy BPM")
        
        return ", ".join(reasons) if reasons else "general match"


# Global music integration engine instance
_music_integration: Optional[MusicIntegrationEngine] = None


def get_music_integration() -> MusicIntegrationEngine:
    """Get the global music integration engine instance."""
    global _music_integration
    if _music_integration is None:
        _music_integration = MusicIntegrationEngine()
    return _music_integration

