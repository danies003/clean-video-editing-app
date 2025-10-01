"""
Smart Transition Detection System

Combines analysis-based detection with ML-based scoring to find optimal transition points
and reduce over-editing by selecting only the most natural and effective transitions.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from app.models.schemas import VideoAnalysisResult, EditStyle

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Types of transitions that can be detected"""
    SCENE_CHANGE = "scene_change"
    MAJOR_MOTION = "major_motion"
    BEAT_SYNC = "beat_sync"
    AUDIO_SILENCE = "audio_silence"
    VOLUME_CHANGE = "volume_change"
    COMPOSITE = "composite"  # Multiple factors combined


@dataclass
class TransitionPoint:
    """A detected transition point with metadata"""
    timestamp: float
    transition_type: TransitionType
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    context: Dict  # Additional context data


@dataclass
class TransitionSegment:
    """A segment defined by transition points"""
    start_time: float
    end_time: float
    start_transition: Optional[TransitionPoint]
    end_transition: Optional[TransitionPoint]
    duration: float
    importance_score: float


class SmartTransitionDetector:
    """
    Smart transition detection that combines multiple analysis signals
    to find optimal transition points and reduce over-editing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Style-specific parameters
        self.style_configs = {
            EditStyle.TIKTOK: {
                "max_segments": None,  # Let LLM decide based on content
                "min_segment_duration": 0.5,  # Allow very short segments
                "max_segment_duration": None,  # No maximum duration limit
                "motion_threshold": 0.1,  # Very low threshold to catch all motion
                "beat_sync_interval": 1,  # Every beat
                "scene_change_weight": 1.0,  # Full weight to scene changes
                "motion_weight": 1.0,  # Full weight to motion
                "beat_weight": 1.0,  # Full weight to beats
                "audio_weight": 1.0  # Full weight to audio
            },
            EditStyle.YOUTUBE: {
                "max_segments": None,  # Let LLM decide based on content
                "min_segment_duration": 0.5,  # Allow very short segments
                "max_segment_duration": None,  # No maximum duration limit
                "motion_threshold": 0.1,  # Very low threshold to catch all motion
                "beat_sync_interval": 1,  # Every beat
                "scene_change_weight": 1.0,  # Full weight to scene changes
                "motion_weight": 1.0,  # Full weight to motion
                "beat_weight": 1.0,  # Full weight to beats
                "audio_weight": 1.0  # Full weight to audio
            },
            EditStyle.CINEMATIC: {
                "max_segments": None,  # Let LLM decide based on content
                "min_segment_duration": 0.5,  # Allow very short segments
                "max_segment_duration": None,  # No maximum duration limit
                "motion_threshold": 0.1,  # Very low threshold to catch all motion
                "beat_sync_interval": 1,  # Every beat
                "scene_change_weight": 1.0,  # Full weight to scene changes
                "motion_weight": 1.0,  # Full weight to motion
                "beat_weight": 1.0,  # Full weight to beats
                "audio_weight": 1.0  # Full weight to audio
            }
        }
    
    def find_optimal_transitions(
        self, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[TransitionPoint]:
        """
        Find optimal transition points using smart detection.
        
        Args:
            analysis_result: Video analysis results
            style: Editing style (affects detection parameters)
            
        Returns:
            List of optimal transition points
        """
        self.logger.info(f"🔍 Finding optimal transitions for {style.value} style")
        
        # Get style-specific configuration
        config = self.style_configs[style]
        
        # Step 1: Find all potential transition points
        all_transitions = self._find_all_transition_candidates(analysis_result, config)
        
        # Step 2: Score each transition point
        scored_transitions = self._score_transition_points(all_transitions, analysis_result, config)
        
        # Step 3: Select optimal transitions based on style and video duration
        optimal_transitions = self._select_optimal_transitions(
            scored_transitions, analysis_result.duration, config
        )
        
        # Step 4: Validate and adjust timing
        final_transitions = self._validate_transition_timing(optimal_transitions, config)
        
        self.logger.info(f"✅ Found {len(final_transitions)} optimal transitions")
        return final_transitions
    
    def create_segments_from_transitions(
        self, 
        transitions: List[TransitionPoint], 
        video_duration: float
    ) -> List[TransitionSegment]:
        """
        Create segments from transition points.
        
        Args:
            transitions: List of transition points
            video_duration: Total video duration
            
        Returns:
            List of transition segments
        """
        segments = []
        
        # Add start of video
        all_timestamps = [0.0] + [t.timestamp for t in transitions] + [video_duration]
        
        # Validate and sort timestamps to ensure they're in ascending order
        all_timestamps = sorted(list(set(all_timestamps)))  # Remove duplicates and sort
        
        # Ensure we have valid start and end points
        if all_timestamps[0] != 0.0:
            all_timestamps.insert(0, 0.0)
        if all_timestamps[-1] != video_duration:
            all_timestamps.append(video_duration)
        
        self.logger.info(f"📊 Creating segments from timestamps: {all_timestamps}")
        
        for i in range(len(all_timestamps) - 1):
            start_time = all_timestamps[i]
            end_time = all_timestamps[i + 1]
            
            # Find transitions at start and end
            start_transition = next((t for t in transitions if abs(t.timestamp - start_time) < 0.1), None)
            end_transition = next((t for t in transitions if abs(t.timestamp - end_time) < 0.1), None)
            
            segment = TransitionSegment(
                start_time=start_time,
                end_time=end_time,
                start_transition=start_transition,
                end_transition=end_transition,
                duration=end_time - start_time,
                importance_score=self._calculate_segment_importance(start_transition, end_transition)
            )
            
            self.logger.info(f"📊 Created segment {i+1}: {start_time:.2f}s-{end_time:.2f}s (duration: {segment.duration:.2f}s)")
            segments.append(segment)
        
        self.logger.info(f"✅ Created {len(segments)} valid segments")
        return segments
    
    def _find_all_transition_candidates(
        self, 
        analysis_result: VideoAnalysisResult, 
        config: Dict
    ) -> List[TransitionPoint]:
        """Find all potential transition candidates from analysis data."""
        candidates = []
        
        # Scene changes (highest priority)
        if analysis_result.motion_analysis and analysis_result.motion_analysis.scene_changes:
            for timestamp in analysis_result.motion_analysis.scene_changes:
                candidates.append(TransitionPoint(
                    timestamp=timestamp,
                    transition_type=TransitionType.SCENE_CHANGE,
                    intensity=1.0,
                    confidence=0.95,
                    reasoning="Scene change detected",
                    context={"source": "scene_detection"}
                ))
        
        # Major motion spikes
        if analysis_result.motion_analysis and analysis_result.motion_analysis.motion_spikes:
            motion_threshold = config["motion_threshold"]
            motion_intensities = analysis_result.motion_analysis.motion_intensities
            for i, timestamp in enumerate(analysis_result.motion_analysis.motion_spikes):
                if i < len(motion_intensities):
                    intensity = motion_intensities[i]
                    if intensity > motion_threshold:
                        candidates.append(TransitionPoint(
                            timestamp=timestamp,
                            transition_type=TransitionType.MAJOR_MOTION,
                            intensity=min(intensity, 1.0),
                            confidence=0.8,
                            reasoning=f"Major motion spike (intensity: {intensity:.2f})",
                            context={"source": "motion_analysis", "intensity": intensity}
                        ))
        
        # Beat synchronization (but not every beat)
        if analysis_result.beat_detection and analysis_result.beat_detection.timestamps:
            beat_interval = config["beat_sync_interval"]
            beat_timestamps = analysis_result.beat_detection.timestamps[::beat_interval]
            
            for timestamp in beat_timestamps:
                candidates.append(TransitionPoint(
                    timestamp=timestamp,
                    transition_type=TransitionType.BEAT_SYNC,
                    intensity=0.6,
                    confidence=0.7,
                    reasoning=f"Beat synchronization (BPM: {analysis_result.beat_detection.bpm})",
                    context={"source": "beat_detection", "bpm": analysis_result.beat_detection.bpm}
                ))
        
        # Audio silence periods
        if analysis_result.audio_analysis and analysis_result.audio_analysis.silence_periods:
            for silence_period in analysis_result.audio_analysis.silence_periods:
                start_time, end_time = silence_period
                silence_duration = end_time - start_time
                # Use the middle of the silence period
                timestamp = start_time + silence_duration / 2
                
                candidates.append(TransitionPoint(
                    timestamp=timestamp,
                    transition_type=TransitionType.AUDIO_SILENCE,
                    intensity=0.7,
                    confidence=0.75,
                    reasoning="Audio silence period",
                    context={"source": "audio_analysis", "silence_duration": silence_duration}
                ))
        
        # Volume changes
        if analysis_result.audio_analysis and analysis_result.audio_analysis.volume_levels:
            volume_changes = self._detect_volume_changes(analysis_result.audio_analysis.volume_levels)
            for timestamp in volume_changes:
                candidates.append(TransitionPoint(
                    timestamp=timestamp,
                    transition_type=TransitionType.VOLUME_CHANGE,
                    intensity=0.5,
                    confidence=0.6,
                    reasoning="Significant volume change",
                    context={"source": "audio_analysis"}
                ))
        
        # Sort by timestamp
        candidates.sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Found {len(candidates)} transition candidates")
        return candidates
    
    def _score_transition_points(
        self, 
        candidates: List[TransitionPoint], 
        analysis_result: VideoAnalysisResult, 
        config: Dict
    ) -> List[Tuple[TransitionPoint, float]]:
        """Score transition points based on multiple factors."""
        scored = []
        
        for candidate in candidates:
            # Base score from transition type
            base_score = candidate.intensity * candidate.confidence
            
            # Style-specific weighting
            if candidate.transition_type == TransitionType.SCENE_CHANGE:
                weighted_score = base_score * config["scene_change_weight"]
            elif candidate.transition_type == TransitionType.MAJOR_MOTION:
                weighted_score = base_score * config["motion_weight"]
            elif candidate.transition_type == TransitionType.BEAT_SYNC:
                weighted_score = base_score * config["beat_weight"]
            elif candidate.transition_type in [TransitionType.AUDIO_SILENCE, TransitionType.VOLUME_CHANGE]:
                weighted_score = base_score * config["audio_weight"]
            else:
                weighted_score = base_score
            
            # Additional context scoring
            context_score = self._calculate_context_score(candidate, analysis_result)
            final_score = (weighted_score + context_score) / 2
            
            scored.append((candidate, final_score))
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def _select_optimal_transitions(
        self, 
        scored_transitions: List[Tuple[TransitionPoint, float]], 
        video_duration: float, 
        config: Dict
    ) -> List[TransitionPoint]:
        """Select optimal transitions based on video duration and style."""
        max_segments = config["max_segments"]
        min_duration = config["min_segment_duration"]
        
        selected = []
        last_timestamp = 0.0
        
        # If max_segments is None, let the LLM decide based on content
        if max_segments is None:
            # Let LLM decide everything - no restrictions
            for candidate, score in scored_transitions:
                # Only check minimum distance to avoid overlapping segments
                if candidate.timestamp - last_timestamp < min_duration:
                    continue
                
                # Accept all transitions that meet minimum distance requirement
                selected.append(candidate)
                last_timestamp = candidate.timestamp
            
            return selected
        
        # Original logic for when max_segments is specified
        # Calculate target segment duration
        target_segment_duration = video_duration / max_segments
        
        # First pass: select high-quality transitions
        for candidate, score in scored_transitions:
            # Check minimum distance from last transition
            if candidate.timestamp - last_timestamp < min_duration:
                continue
            
            # Check if this would create reasonable segment duration
            if candidate.timestamp - last_timestamp > target_segment_duration * 2:
                continue
            
            selected.append(candidate)
            last_timestamp = candidate.timestamp
        
        # Second pass: ensure good distribution across the video
        if len(selected) < max_segments:
            # Find gaps in the video that need transitions
            all_timestamps = [0.0] + [t.timestamp for t in selected] + [video_duration]
            gaps = []
            
            for i in range(len(all_timestamps) - 1):
                gap_start = all_timestamps[i]
                gap_end = all_timestamps[i + 1]
                gap_duration = gap_end - gap_start
                
                if gap_duration > target_segment_duration * 1.5:
                    # This gap is too large, find a transition point in it
                    target_time = gap_start + gap_duration / 2
                    
                    # Find the best candidate in this time range
                    best_candidate = None
                    best_score = 0
                    
                    for candidate, score in scored_transitions:
                        if (gap_start < candidate.timestamp < gap_end and 
                            candidate.timestamp - gap_start >= min_duration and
                            gap_end - candidate.timestamp >= min_duration):
                            if score > best_score:
                                best_candidate = candidate
                                best_score = score
                    
                    if best_candidate:
                        selected.append(best_candidate)
                        selected.sort(key=lambda x: x.timestamp)
        
        # Limit to max_segments
        selected = selected[:max_segments]
        
        return selected
    
    def _validate_transition_timing(
        self, 
        transitions: List[TransitionPoint], 
        config: Dict
    ) -> List[TransitionPoint]:
        """Validate and adjust transition timing."""
        validated = []
        min_duration = config["min_segment_duration"]
        
        for i, transition in enumerate(transitions):
            # Check minimum distance from previous transition
            if i > 0:
                prev_timestamp = validated[-1].timestamp
                if transition.timestamp - prev_timestamp < min_duration:
                    # Skip this transition if too close
                    continue
            
            validated.append(transition)
        
        return validated
    
    def _detect_volume_changes(self, volume_levels: List[float]) -> List[float]:
        """Detect significant volume changes."""
        if len(volume_levels) < 2:
            return []
        
        # Calculate volume differences
        volume_diffs = np.diff(volume_levels)
        
        # Find significant changes (above threshold)
        threshold = np.std(volume_diffs) * 1.5
        significant_changes = np.where(np.abs(volume_diffs) > threshold)[0]
        
        # Convert to timestamps (assuming uniform time distribution)
        timestamps = []
        for idx in significant_changes:
            timestamp = (idx / len(volume_levels)) * 100  # Assuming 100 time points
            timestamps.append(timestamp)
        
        return timestamps
    
    def _calculate_context_score(
        self, 
        transition: TransitionPoint, 
        analysis_result: VideoAnalysisResult
    ) -> float:
        """Calculate additional context score for a transition."""
        score = 0.0
        
        # Check if transition aligns with multiple signals
        alignment_count = 0
        
        # Check motion alignment
        if analysis_result.motion_analysis and analysis_result.motion_analysis.motion_spikes:
            for i, timestamp in enumerate(analysis_result.motion_analysis.motion_spikes):
                if i < len(analysis_result.motion_analysis.motion_intensities):
                    intensity = analysis_result.motion_analysis.motion_intensities[i]
                    if abs(timestamp - transition.timestamp) < 0.5:  # Within 0.5 seconds
                        alignment_count += 1
                        score += intensity * 0.2
        
        # Check beat alignment
        if analysis_result.beat_detection and analysis_result.beat_detection.timestamps:
            for timestamp in analysis_result.beat_detection.timestamps:
                if abs(timestamp - transition.timestamp) < 0.3:  # Within 0.3 seconds
                    alignment_count += 1
                    score += 0.1
        
        # Bonus for multiple alignments
        if alignment_count >= 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_segment_importance(
        self, 
        start_transition: Optional[TransitionPoint], 
        end_transition: Optional[TransitionPoint]
    ) -> float:
        """Calculate importance score for a segment."""
        importance = 0.5  # Base importance
        
        if start_transition:
            importance += start_transition.intensity * 0.3
        
        if end_transition:
            importance += end_transition.intensity * 0.3
        
        return min(importance, 1.0)

    def select_transitions_with_llm(
        self,
        candidates: List[TransitionPoint],
        analysis_result: 'VideoAnalysisResult',
        llm_client: Any,
        style: 'EditStyle',
        max_segments: int = 5
    ) -> List['TransitionPoint']:
        """
        Use the LLM to select the optimal transitions from all candidates based on content analysis.
        Args:
            candidates: List of all candidate TransitionPoint objects
            analysis_result: VideoAnalysisResult for context
            llm_client: LLM client with a call_llm(prompt) method
            style: EditStyle (e.g., TIKTOK, CINEMATIC)
            max_segments: Maximum number of segments to select
        Returns:
            List[TransitionPoint]: Selected transitions as decided by the LLM
        """
        import json
        import logging
        logger = logging.getLogger(__name__)
        if not candidates:
            return []
        # Prepare a summary of candidates for the LLM
        candidate_list = [
            {
                "timestamp": round(tp.timestamp, 2),
                "type": tp.transition_type.value,
                "intensity": round(tp.intensity, 2),
                "confidence": round(tp.confidence, 2),
                "reasoning": tp.reasoning
            }
            for tp in candidates
        ]
        # Prepare a summary of the analysis
        analysis_summary = {
            "duration": analysis_result.duration,
            "bpm": getattr(analysis_result.beat_detection, 'bpm', None),
            "num_beats": len(getattr(analysis_result.beat_detection, 'timestamps', [])),
            "num_motion_spikes": len(getattr(analysis_result.motion_analysis, 'motion_spikes', [])),
            "num_scene_changes": len(getattr(analysis_result.motion_analysis, 'scene_changes', [])),
        }
        # Build the LLM prompt
        prompt = f"""
You are an expert video editor. Given the following video analysis and candidate transitions, select the optimal {max_segments} transitions for a {style.value} style edit. Avoid over-editing (no flashing). Only select transitions that make sense for pacing and content.

Video Analysis:
{json.dumps(analysis_summary, indent=2)}

Transition Candidates:
{json.dumps(candidate_list, indent=2)}

Return a JSON list of the selected transition timestamps (in seconds, sorted ascending), and a short reasoning for your selection. Example:
{{
  "selected": [1.0, 3.5, 7.2],
  "reasoning": "Selected transitions at natural scene changes and strong beats, spaced for good pacing."
}}
"""
        # Call the LLM
        response = llm_client.call_llm(prompt)
        try:
            result = json.loads(response)
            selected_timestamps = result.get("selected", [])
            # Map back to TransitionPoint objects
            selected = []
            for ts in selected_timestamps:
                # Find the closest candidate
                closest = min(candidates, key=lambda tp: abs(tp.timestamp - ts))
                if closest not in selected:
                    selected.append(closest)
            # Sort by timestamp
            selected.sort(key=lambda tp: tp.timestamp)
            logger.info(f"[LLM_TRANSITIONS] LLM selected {len(selected)} transitions: {[tp.timestamp for tp in selected]}")
            return selected
        except Exception as e:
            logger.error(f"[LLM_TRANSITIONS] Failed to parse LLM response: {e}, response: {response}")
            # Fallback: return top-N by confidence
            return sorted(candidates, key=lambda tp: tp.confidence, reverse=True)[:max_segments]


# Global instance
_transition_detector: Optional[SmartTransitionDetector] = None


def get_transition_detector() -> SmartTransitionDetector:
    """Get the global transition detector instance."""
    global _transition_detector
    if _transition_detector is None:
        _transition_detector = SmartTransitionDetector()
    return _transition_detector 