"""
AutoTransition Integration

This module provides integration with AutoTransition for intelligent
transition and effect recommendations. Falls back to rule-based
recommendations when AutoTransition is not available.
"""

import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import VideoAnalysisResult, EditStyle
from app.editor.transition_detector import TransitionPoint

logger = logging.getLogger(__name__)


class AutoTransitionIntegration:
    """
    Integration with AutoTransition for intelligent transition and effect recommendations.
    
    Provides fallback recommendations when AutoTransition is not available.
    """
    
    def __init__(self):
        """Initialize the AutoTransition integration."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Check if AutoTransition is available
        try:
            import autotransition
            self.autotransition_available = True
            self.logger.info("✅ AutoTransition integration available")
        except ImportError:
            self.autotransition_available = False
            self.logger.warning("AutoTransition not available - using fallback recommendations")
    
    def recommend_visual_effects(
        self, 
        start_time: float, 
        end_time: float, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[str]:
        """
        Recommend visual effects for a segment based on content analysis.
        
        Args:
            start_time: Segment start time
            end_time: Segment end time
            analysis_result: Video analysis results
            style: Editing style
            
        Returns:
            List of recommended effect names
        """
        if not self.autotransition_available:
            return self._get_fallback_effects(start_time, end_time, analysis_result, style)
        return self._get_autotransition_effects(start_time, end_time, analysis_result, style)
    
    def recommend_transition_effects(
        self, 
        transition_point: TransitionPoint, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[str]:
        """
        Recommend transition effects for a transition point.
        
        Args:
            transition_point: The transition point
            analysis_result: Video analysis results
            style: Editing style
            
        Returns:
            List of recommended transition names
        """
        if not self.autotransition_available:
            return self._get_fallback_transitions(transition_point, analysis_result, style)
        return self._get_autotransition_transitions(transition_point, analysis_result, style)
    
    def _get_autotransition_effects(
        self, 
        start_time: float, 
        end_time: float, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[str]:
        """Get effects from AutoTransition (placeholder for future implementation)."""
        # Placeholder for actual AutoTransition integration
        return self._get_fallback_effects(start_time, end_time, analysis_result, style)
    
    def _get_autotransition_transitions(
        self, 
        transition_point: TransitionPoint, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[str]:
        """Get transitions from AutoTransition (placeholder for future implementation)."""
        # Placeholder for actual AutoTransition integration
        return self._get_fallback_transitions(transition_point, analysis_result, style)
    
    def _get_fallback_effects(
        self, 
        start_time: float, 
        end_time: float, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[str]:
        """Get fallback effects based on content analysis and style."""
        effects = []
        segment_duration = end_time - start_time
        
        # Style-based effects
        if style == EditStyle.TIKTOK:
            effects.extend(["beat_sync", "motion_blur", "cinematic"])
        elif style == EditStyle.YOUTUBE:
            effects.extend(["color_grading", "high_contrast", "motion_blur"])
        elif style == EditStyle.CINEMATIC:
            effects.extend(["cinematic", "color_grading", "motion_blur"])
        
        # Content-based effects
        if self._has_audio_peaks(start_time, end_time, analysis_result):
            effects.append("beat_sync")
        
        if self._has_motion_activity(start_time, end_time, analysis_result):
            effects.append("motion_blur")
        
        if self._is_scene_boundary(start_time, end_time, analysis_result):
            effects.append("scene_transition")
        
        # Duration-based effects
        if segment_duration < 1.0:
            effects.append("fast_pace")
        elif segment_duration > 3.0:
            effects.append("slow_motion")
        
        return list(set(effects))  # Remove duplicates
    
    def _get_fallback_transitions(
        self, 
        transition_point: TransitionPoint, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[str]:
        """Get fallback transitions based on transition point and style."""
        transitions = []
        
        # Style-based transitions
        if style == EditStyle.TIKTOK:
            transitions.extend(["whip_pan", "zoom", "slide"])
        elif style == EditStyle.YOUTUBE:
            transitions.extend(["cross_dissolve", "slide", "zoom"])
        elif style == EditStyle.CINEMATIC:
            transitions.extend(["cross_dissolve", "fade", "slide"])
        
        # Intensity-based transitions
        if transition_point.intensity > 0.8:
            transitions.extend(["whip_pan", "spin", "glitch"])
        elif transition_point.intensity > 0.5:
            transitions.extend(["zoom", "slide"])
        else:
            transitions.extend(["cross_dissolve", "fade"])
        
        return list(set(transitions))  # Remove duplicates
    
    def _has_audio_peaks(
        self, 
        start_time: float, 
        end_time: float, 
        analysis_result: VideoAnalysisResult
    ) -> bool:
        """Check if segment contains audio peaks."""
        if not analysis_result.beat_detection or not analysis_result.beat_detection.timestamps:
            return False
        
        peaks_in_segment = [
            peak for peak in analysis_result.beat_detection.timestamps
            if start_time <= peak <= end_time
        ]
        return len(peaks_in_segment) > 0
    
    def _has_motion_activity(
        self, 
        start_time: float, 
        end_time: float, 
        analysis_result: VideoAnalysisResult
    ) -> bool:
        """Check if segment has high motion activity."""
        if not analysis_result.motion_analysis or not analysis_result.motion_analysis.motion_spikes:
            return False
        
        motion_in_segment = [
            spike for spike in analysis_result.motion_analysis.motion_spikes
            if start_time <= spike <= end_time
        ]
        return len(motion_in_segment) > 0
    
    def _is_scene_boundary(
        self, 
        start_time: float, 
        end_time: float, 
        analysis_result: VideoAnalysisResult
    ) -> bool:
        """Check if segment contains a scene change."""
        if not analysis_result.motion_analysis or not analysis_result.motion_analysis.scene_changes:
            return False
        
        scene_changes_in_segment = [
            change for change in analysis_result.motion_analysis.scene_changes
            if start_time <= change <= end_time
        ]
        return len(scene_changes_in_segment) > 0 