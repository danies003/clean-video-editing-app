"""
Timeline builder for video editing automation.

This module creates video editing timelines based on analysis results
and user-defined templates, determining cut points, transitions, and effects.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from app.models.schemas import (
    VideoTimeline, TimelineSegment, VideoAnalysisResult,
    EditingTemplate, TemplateType
)

logger = logging.getLogger(__name__)


class TimelineBuilder:
    """
    Builds video editing timelines from analysis results and templates.
    
    Determines optimal cut points, transitions, and effects based on
    beat detection, motion analysis, and template configurations.
    """
    
    def __init__(self):
        """Initialize the timeline builder."""
        pass
    
    async def build_timeline(
        self,
        video_id: UUID,
        analysis_result: VideoAnalysisResult,
        template: EditingTemplate,
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> VideoTimeline:
        """
        Build a complete video timeline for editing.
        
        Args:
            video_id: Video identifier
            analysis_result: Video analysis results
            template: Editing template to apply
            custom_settings: Optional custom settings override
            
        Returns:
            VideoTimeline: Complete timeline for video editing
        """
        try:
            logger.info(f"Building timeline for video {video_id}")
            
            # Determine cut points based on template type
            cut_points = await self._determine_cut_points(
                analysis_result, template, custom_settings
            )
            
            # Create timeline segments
            segments = await self._create_segments(
                video_id, cut_points, template, analysis_result
            )
            
            # Calculate total duration
            total_duration = sum(
                segment.end_time - segment.start_time for segment in segments
            )
            
            # Create timeline
            timeline = VideoTimeline(
                video_id=video_id,
                template=template,
                segments=segments,
                total_duration=total_duration
            )
            
            logger.info(f"Timeline built for video {video_id}: {len(segments)} segments")
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to build timeline for video {video_id}: {e}")
            raise
    
    async def build_timeline_from_enhanced_plan(
        self,
        video_id: UUID,
        analysis_result: VideoAnalysisResult,
        enhanced_plan: Dict[str, Any],
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> VideoTimeline:
        """
        Build a complete video timeline from an enhanced LLM plan with segment-specific effects.
        
        Args:
            video_id: Video identifier
            analysis_result: Video analysis results
            enhanced_plan: Enhanced LLM editing plan with segment-specific effects
            custom_settings: Optional custom settings override
            
        Returns:
            VideoTimeline: Complete timeline for video editing
        """
        try:
            logger.info(f"Building enhanced timeline for video {video_id}")
            
            # Extract segments from enhanced plan
            enhanced_segments = enhanced_plan.get("segments", [])
            
            # Create a minimal template for compatibility
            from app.models.schemas import EditingTemplate, TemplateType
            template = EditingTemplate(
                template_id=uuid4(),
                name=f"Enhanced LLM Edit {datetime.utcnow().isoformat()}",
                template_type=TemplateType.BEAT_MATCH,
                description="Generated from enhanced LLM plan",
                effects=[],  # Effects are now segment-specific
                transition_duration=0.5,
                cut_sensitivity=0.7,
                beat_sync_threshold=0.1
            )
            
            # Create timeline segments with segment-specific effects
            segments = []
            for i, enhanced_segment in enumerate(enhanced_segments):
                # Handle both field name formats for compatibility
                start_time = float(enhanced_segment.get("start_time", enhanced_segment.get("start", 0)))
                end_time = float(enhanced_segment.get("end_time", enhanced_segment.get("end", 0)))
                
                # Get segment-specific effects from enhanced plan (handle both field names)
                segment_effects = enhanced_segment.get("effects", enhanced_segment.get("tags", []))
                
                # Use LLM transitions if available, otherwise fall back to template
                transition_in = enhanced_segment.get("transition_in")
                transition_out = enhanced_segment.get("transition_out")

                # If LLM didn't provide transitions, use template defaults
                if transition_in is None:
                    transition_in = await self._get_transition_in(i, template)
                if transition_out is None:
                    transition_out = await self._get_transition_out(i, len(enhanced_segments) - 1, template)
                
                segment = TimelineSegment(
                    start_time=start_time,
                    end_time=end_time,
                    source_video_id=video_id,
                    effects=segment_effects,  # Use segment-specific effects
                    transition_in=transition_in,
                    transition_out=transition_out
                )
                segments.append(segment)
            
            # Calculate total duration
            total_duration = sum(
                segment.end_time - segment.start_time for segment in segments
            )
            
            # Create timeline
            timeline = VideoTimeline(
                video_id=video_id,
                template=template,
                segments=segments,
                total_duration=total_duration
            )
            
            logger.info(f"Enhanced timeline built for video {video_id}: {len(segments)} segments with segment-specific effects")
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to build enhanced timeline for video {video_id}: {e}")
            raise
    
    async def _determine_cut_points(
        self,
        analysis_result: VideoAnalysisResult,
        template: EditingTemplate,
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Determine optimal cut points based on analysis and template.
        
        Args:
            analysis_result: Video analysis results
            template: Editing template
            custom_settings: Optional custom settings
            
        Returns:
            List[float]: List of cut point timestamps
        """
        cut_points = []
        
        # Get template settings
        cut_sensitivity = template.cut_sensitivity
        beat_sync_threshold = template.beat_sync_threshold
        
        # Override with custom settings if provided
        if custom_settings:
            cut_sensitivity = custom_settings.get('cut_sensitivity', cut_sensitivity)
            beat_sync_threshold = custom_settings.get('beat_sync_threshold', beat_sync_threshold)
        
        # Check if we have meaningful analysis data
        has_audio = (len(analysis_result.beat_detection.timestamps) > 0 or 
                    len(analysis_result.audio_analysis.audio_peaks) > 0)
        has_motion = len(analysis_result.motion_analysis.motion_spikes) > 0
        has_scenes = len(analysis_result.motion_analysis.scene_changes) > 0
        
        # If no meaningful data, use fallback cut points
        if not (has_audio or has_motion or has_scenes):
            logger.warning("No meaningful analysis data found, using fallback cut points for no-audio video")
            return self._create_fallback_cut_points(analysis_result.duration, template)
        
        # Add beat-based cut points
        if template.template_type in [TemplateType.BEAT_MATCH, TemplateType.FAST_PACED]:
            beat_cuts = await self._get_beat_cuts(
                analysis_result.beat_detection.timestamps,
                beat_sync_threshold
            )
            cut_points.extend(beat_cuts)
        
        # Add motion-based cut points
        # Use motion cuts for all template types when audio data is not available
        if (template.template_type in [TemplateType.CINEMATIC, TemplateType.TRANSITION_HEAVY] or 
            (not has_audio and has_motion)):
            motion_cuts = await self._get_motion_cuts(
                analysis_result.motion_analysis.motion_spikes,
                analysis_result.motion_analysis.motion_intensities,
                cut_sensitivity
            )
            cut_points.extend(motion_cuts)
        
        # Add scene change cut points
        scene_cuts = await self._get_scene_cuts(
            analysis_result.motion_analysis.scene_changes,
            analysis_result.motion_analysis.scene_confidence
        )
        cut_points.extend(scene_cuts)
        
        # Add audio-based cut points
        audio_cuts = await self._get_audio_cuts(
            analysis_result.audio_analysis.audio_peaks,
            cut_sensitivity
        )
        cut_points.extend(audio_cuts)
        
        # Remove duplicates and sort
        cut_points = sorted(list(set(cut_points)))
        
        # Filter cut points based on minimum segment duration
        min_segment_duration = template.transition_duration * 2
        filtered_cuts = [0.0]  # Always start at beginning
        
        for cut_point in cut_points:
            if cut_point - filtered_cuts[-1] >= min_segment_duration:
                filtered_cuts.append(cut_point)
        
        # Add end point if not already present, ensuring it doesn't exceed video duration
        video_duration = analysis_result.duration
        if filtered_cuts[-1] < video_duration:
            # Ensure the end point doesn't exceed video duration by more than 0.1 seconds
            end_point = min(video_duration, filtered_cuts[-1] + 0.1)
            filtered_cuts.append(end_point)
        
        logger.info(f"Determined {len(filtered_cuts)} cut points")
        return filtered_cuts
    
    async def _get_beat_cuts(
        self,
        beat_timestamps: List[float],
        beat_sync_threshold: float
    ) -> List[float]:
        """
        Get cut points based on beat detection.
        
        Args:
            beat_timestamps: Beat timestamps
            beat_sync_threshold: Beat synchronization threshold
            
        Returns:
            List[float]: Beat-based cut points
        """
        cuts = []
        
        for beat_time in beat_timestamps:
            # Apply threshold to avoid too many cuts
            if len(cuts) == 0 or beat_time - cuts[-1] >= beat_sync_threshold:
                cuts.append(beat_time)
        
        return cuts
    
    async def _get_motion_cuts(
        self,
        motion_spikes: List[float],
        motion_intensities: List[float],
        cut_sensitivity: float
    ) -> List[float]:
        """
        Get cut points based on motion analysis.
        
        Args:
            motion_spikes: Motion spike timestamps
            motion_intensities: Motion intensity values
            cut_sensitivity: Cut sensitivity threshold
            
        Returns:
            List[float]: Motion-based cut points
        """
        cuts = []
        
        for i, spike_time in enumerate(motion_spikes):
            if i < len(motion_intensities):
                intensity = motion_intensities[i]
                if intensity > cut_sensitivity:
                    cuts.append(spike_time)
        
        return cuts
    
    async def _get_scene_cuts(
        self,
        scene_changes: List[float],
        scene_confidence: List[float]
    ) -> List[float]:
        """
        Get cut points based on scene changes.
        
        Args:
            scene_changes: Scene change timestamps
            scene_confidence: Scene change confidence scores
            
        Returns:
            List[float]: Scene change cut points
        """
        cuts = []
        
        for i, scene_time in enumerate(scene_changes):
            if i < len(scene_confidence):
                confidence = scene_confidence[i]
                if confidence > 0.7:  # High confidence threshold
                    cuts.append(scene_time)
        
        return cuts
    
    async def _get_audio_cuts(
        self,
        audio_peaks: List[float],
        cut_sensitivity: float
    ) -> List[float]:
        """
        Get cut points based on audio peaks.
        
        Args:
            audio_peaks: Audio peak timestamps
            cut_sensitivity: Cut sensitivity threshold
            
        Returns:
            List[float]: Audio-based cut points
        """
        # Use audio peaks as potential cut points
        # Apply spacing to avoid too many cuts
        cuts = []
        min_spacing = 1.0  # Minimum 1 second between cuts
        
        for peak_time in audio_peaks:
            if len(cuts) == 0 or peak_time - cuts[-1] >= min_spacing:
                cuts.append(peak_time)
        
        return cuts
    
    def _create_fallback_cut_points(self, duration: float, template: EditingTemplate) -> List[float]:
        """
        Create fallback cut points for videos without meaningful analysis data.
        
        Args:
            duration: Video duration in seconds
            template: Editing template
            
        Returns:
            List[float]: Fallback cut points
        """
        # Create evenly spaced cut points based on video duration
        if duration <= 5.0:
            # Short videos: 2-3 segments
            segment_count = 2
        elif duration <= 20.0:  # Increased from 15.0 to accommodate 17s videos
            # Medium videos: 3-5 segments  
            segment_count = 3
        else:
            # Long videos: 5-8 segments
            segment_count = 5
        
        # Calculate segment duration
        segment_duration = duration / segment_count
        
        # Create cut points
        cut_points = [0.0]  # Always start at beginning
        for i in range(1, segment_count):
            cut_point = i * segment_duration
            cut_points.append(cut_point)
        
        # Add end point
        cut_points.append(duration)
        
        logger.info(f"Created {len(cut_points)} fallback cut points for {duration:.2f}s video")
        return cut_points
    
    async def _create_segments(
        self,
        video_id: UUID,
        cut_points: List[float],
        template: EditingTemplate,
        analysis_result: VideoAnalysisResult
    ) -> List[TimelineSegment]:
        """
        Create timeline segments from cut points.
        
        Args:
            video_id: Video identifier
            cut_points: List of cut point timestamps
            template: Editing template
            analysis_result: Video analysis results
            
        Returns:
            List[TimelineSegment]: Timeline segments
        """
        segments = []
        
        for i in range(len(cut_points) - 1):
            start_time = cut_points[i]
            end_time = cut_points[i + 1]
            
            # Determine effects for this segment
            effects = await self._get_segment_effects(
                start_time, end_time, template, analysis_result
            )
            
            # Determine transitions
            transition_in = await self._get_transition_in(i, template)
            transition_out = await self._get_transition_out(i, len(cut_points) - 2, template)
            
            segment = TimelineSegment(
                start_time=start_time,
                end_time=end_time,
                source_video_id=video_id,
                effects=effects,
                transition_in=transition_in,
                transition_out=transition_out
            )
            
            segments.append(segment)
        
        return segments
    
    async def _get_segment_effects(
        self,
        start_time: float,
        end_time: float,
        template: EditingTemplate,
        analysis_result: VideoAnalysisResult
    ) -> List[str]:
        """
        Determine effects for a timeline segment.
        
        Args:
            start_time: Segment start time
            end_time: Segment end time
            template: Editing template
            analysis_result: Video analysis results
            
        Returns:
            List[str]: List of effects to apply
        """
        effects = []
        
        # Add template effects
        effects.extend(template.effects)
        
        # Add motion-based effects
        segment_duration = end_time - start_time
        if segment_duration < 1.0:  # Short segments
            effects.append("speed_up")
        elif segment_duration > 5.0:  # Long segments
            effects.append("slow_motion")
        
        # Add audio-based effects
        if start_time < analysis_result.duration:
            # Check if segment contains audio peaks
            for peak_time in analysis_result.audio_analysis.audio_peaks:
                if start_time <= peak_time <= end_time:
                    effects.append("audio_emphasis")
                    break
        
        return list(set(effects))  # Remove duplicates
    
    async def _get_transition_in(self, segment_index: int, template: EditingTemplate) -> Optional[str]:
        """
        Get transition effect for segment start.
        
        Args:
            segment_index: Index of the segment
            template: Editing template
            
        Returns:
            Optional[str]: Transition effect name
        """
        if segment_index == 0:
            return None  # No transition for first segment
        
        # Select transition based on template
        transitions = ["fade_in", "slide_in", "zoom_in", "crossfade"]
        transition_index = segment_index % len(transitions)
        return transitions[transition_index]
    
    async def _get_transition_out(self, segment_index: int, total_segments: int, template: EditingTemplate) -> Optional[str]:
        """
        Get transition effect for segment end.
        
        Args:
            segment_index: Index of the segment
            total_segments: Total number of segments
            template: Editing template
            
        Returns:
            Optional[str]: Transition effect name
        """
        if segment_index == total_segments:
            return None  # No transition for last segment
        
        # Select transition based on template
        transitions = ["fade_out", "slide_out", "zoom_out", "crossfade"]
        transition_index = segment_index % len(transitions)
        return transitions[transition_index]


# Global timeline builder instance
_timeline_builder: Optional[TimelineBuilder] = None


async def initialize_timeline_builder() -> TimelineBuilder:
    """
    Initialize the global timeline builder.
    
    Returns:
        TimelineBuilder: Initialized timeline builder
    """
    global _timeline_builder
    
    _timeline_builder = TimelineBuilder()
    logger.info("Timeline builder initialized successfully")
    return _timeline_builder


def get_timeline_builder() -> TimelineBuilder:
    """
    Get the global timeline builder instance.
    
    Returns:
        TimelineBuilder: Timeline builder instance
        
    Raises:
        RuntimeError: If timeline builder is not initialized
    """
    if _timeline_builder is None:
        raise RuntimeError("Timeline builder not initialized. Call initialize_timeline_builder first.")
    
    return _timeline_builder 