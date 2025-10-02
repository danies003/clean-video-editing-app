"""
Enhanced LLM Editor with Smart Transition Detection

Combines smart transition detection with AutoTransition ML models
to create more natural and selective editing plans.
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from app.editor.llm_editor import LLMEditor, EditingDecision, EditingPlan, LLMProvider, MoviePyRenderingPlan
from app.editor.transition_detector import SmartTransitionDetector, TransitionPoint, TransitionSegment
from app.editor.autotransition_integration import AutoTransitionIntegration
from app.editor.dynamic_effects_engine import get_dynamic_effects_engine
from app.models.schemas import VideoAnalysisResult, EditStyle, CrossVideoAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEditingDecision:
    """Enhanced editing decision with transition metadata"""
    start_time: float
    end_time: float
    effects: List[str]  # Changed from effect: str to effects: List[str]
    intensity: float
    reasoning: str
    transition: str = "cross_dissolve"  # Add transition field
    transition_point: Optional[TransitionPoint] = None
    segment_importance: float = 0.5
    recommended_transitions: Optional[List[str]] = None
    recommended_effects: Optional[List[str]] = None
    speed_factor: float = 1.0  # Add speed_factor for speed effects
    
    def __post_init__(self):
        if self.recommended_transitions is None:
            self.recommended_transitions = []
        if self.recommended_effects is None:
            self.recommended_effects = []


@dataclass
class EnhancedEditingPlan(EditingPlan):
    """Enhanced editing plan with smart transition detection"""
    transition_points: List[TransitionPoint]
    transition_segments: List[TransitionSegment]
    smart_detection_metadata: Dict[str, Any]


class EnhancedLLMEditor(LLMEditor):
    """
    Enhanced LLM editor that uses smart transition detection
    and AutoTransition integration for better editing plans.
    """
    
    def __init__(self, provider: LLMProvider = LLMProvider.GEMINI):
        super().__init__(provider)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize LLM client for enhanced effects generation
        self.llm_client = self._initialize_llm_client(provider)
        
        # Initialize smart transition detection
        self.transition_detector = SmartTransitionDetector()
        
        # Initialize AutoTransition integration (optional)
        try:
            self.autotransition_integration = AutoTransitionIntegration()
        except ImportError:
            self.autotransition_integration = None
            self.logger.warning("AutoTransition not available, continuing without it")
        
        # Initialize dynamic effects engine
        self.dynamic_effects_engine = get_dynamic_effects_engine()
        
        self.logger.info("🚀 Enhanced LLM Editor initialized with smart transition detection and dynamic effects")
        self.logger.info("🎬 [ENHANCED_LLM] Will generate dynamic effect code during planning phase")
    
    def _initialize_llm_client(self, provider: LLMProvider):
        """Initialize LLM client for enhanced effects generation."""
        try:
            if provider == LLMProvider.GEMINI:
                import google.generativeai as genai
                import os
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    self.logger.warning("⚠️ [ENHANCED_LLM] Gemini API key not found, continuing without Gemini client")
                    return None
                
                genai.configure(api_key=api_key)
                self.logger.info("✅ [ENHANCED_LLM] Gemini client initialized successfully")
                return genai
            elif provider == LLMProvider.OPENAI:
                from openai import OpenAI
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self.logger.warning("⚠️ [ENHANCED_LLM] OpenAI API key not found, continuing without OpenAI client")
                    return None
                
                client = OpenAI(api_key=api_key)
                self.logger.info("✅ [ENHANCED_LLM] OpenAI client initialized successfully")
                return client
            else:
                self.logger.warning(f"⚠️ [ENHANCED_LLM] Provider {provider} not supported, continuing without LLM client")
                return None
        except Exception as e:
            self.logger.warning(f"⚠️ [ENHANCED_LLM] Failed to initialize LLM client: {e}, continuing without it")
            return None
    
    async def generate_editing_plan(
        self, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle,
        target_duration: Optional[float] = None,
        multi_video_context: Optional[Dict[str, Any]] = None
    ) -> EnhancedEditingPlan:
        """Generate an enhanced editing plan with smart transition detection."""
        logger.info(f"🎬 [ENHANCED_LLM] Generating enhanced {style.value} editing plan...")
        
        # Check if this is a multi-video editing request
        if multi_video_context:
            logger.info("🎬 [MULTI-VIDEO] Generating multi-video editing plan with cross-video analysis...")
            return self._generate_multi_video_editing_plan(
                analysis_result=analysis_result,
                style=style,
                target_duration=target_duration,
                multi_video_context=multi_video_context
            )
        
        # Step 1: Generate LLM timeline assignment for single video
        logger.info(f"🧠 [ENHANCED_LLM] Generating LLM timeline assignment for single video...")
        
        # Create single video context
        single_video_context = {
            "video_ids": [str(analysis_result.video_id)],
            "total_videos": 1,
            "cross_analysis_completed": False,
            "similarity_matrix": {},
            "cross_video_segments": []
        }
        
        # Generate LLM timeline assignment
        timeline_assignment = await self._generate_multi_video_timeline_assignment(
            analysis_result=analysis_result,
            style=style,
            target_duration=target_duration,
            multi_video_context=single_video_context
        )
        
        # Convert timeline assignment to editing plan
        enhanced_plan = self._convert_timeline_assignment_to_plan(
            timeline_assignment=timeline_assignment,
            analysis_result=analysis_result,
            style=style
        )
        
        logger.info(f"✅ [ENHANCED_LLM] Generated enhanced plan with {len(enhanced_plan.segments)} segments using LLM timeline assignment")
        return enhanced_plan
    
    def _detect_optimal_transitions(
        self, 
        analysis_result: VideoAnalysisResult, 
        style: EditStyle
    ) -> List[TransitionPoint]:
        """
        Detect optimal transition points using enhanced analysis data.
        
        Args:
            analysis_result: Video analysis results including scene and content analysis
            style: Editing style
            
        Returns:
            List of optimal transition points
        """
        try:
            self.logger.info("🔍 [ENHANCED_LLM] Detecting optimal transitions with enhanced analysis")
            
            # Get enhanced analysis data
            enhanced_metadata = analysis_result.analysis_metadata
            scene_analysis = enhanced_metadata.get("scene_analysis", {})
            content_analysis = enhanced_metadata.get("content_analysis", {})
            
            # Collect all transition points from different sources
            all_transition_points = []
            
            # 1. Scene-based transitions from PySceneDetect
            scene_transitions = scene_analysis.get("transition_points", [])
            for transition in scene_transitions:
                transition_point = TransitionPoint(
                    time=transition["time"],
                    confidence=transition["confidence"],
                    type="scene_boundary",
                    metadata={
                        "scene_id": transition.get("scene_id"),
                        "duration": transition.get("duration"),
                        "content_analysis": transition.get("content_analysis", {})
                    }
                )
                all_transition_points.append(transition_point)
            
            # 2. Content-based transitions from MViTv2
            content_transitions = content_analysis.get("content_transitions", [])
            for transition in content_transitions:
                transition_point = TransitionPoint(
                    time=transition["time"],
                    confidence=transition["confidence"],
                    type="content_based",
                    metadata={
                        "reasoning": transition.get("reasoning"),
                        "content_analysis": transition.get("content_analysis", {})
                    }
                )
                all_transition_points.append(transition_point)
            
            # 3. Traditional motion and audio-based transitions
            motion_analysis = analysis_result.motion_analysis
            audio_analysis = analysis_result.audio_analysis
            
            # Motion-based transitions
            for i, motion_score in enumerate(motion_analysis.motion_spikes):
                if motion_score > 0.7:  # High motion threshold
                    time = (i / len(motion_analysis.motion_spikes)) * analysis_result.duration
                    transition_point = TransitionPoint(
                        time=time,
                        confidence=motion_score,
                        type="motion_based",
                        metadata={"motion_score": motion_score}
                    )
                    all_transition_points.append(transition_point)
            
            # Audio-based transitions (beat sync)
            for beat_time in analysis_result.beat_detection.timestamps:
                transition_point = TransitionPoint(
                    time=beat_time,
                    confidence=0.8,  # High confidence for beats
                    type="audio_based",
                    metadata={"beat_sync": True}
                )
                all_transition_points.append(transition_point)
            
            # Sort by confidence and remove duplicates
            all_transition_points.sort(key=lambda x: x.confidence, reverse=True)
            
            # Remove transitions that are too close together (within 0.5 seconds)
            filtered_transitions = []
            for transition in all_transition_points:
                # Check if this transition is too close to existing ones
                too_close = False
                for existing in filtered_transitions:
                    if abs(transition.time - existing.time) < 0.5:
                        too_close = True
                        break
                
                if not too_close:
                    filtered_transitions.append(transition)
            
            # Limit to top transitions based on style
            max_transitions = self._get_max_transitions_for_style(style)
            optimal_transitions = filtered_transitions[:max_transitions]
            
            self.logger.info(f"✅ [ENHANCED_LLM] Found {len(optimal_transitions)} optimal transitions from {len(all_transition_points)} candidates")
            self.logger.info(f"📊 [ENHANCED_LLM] Transition sources: Scene={len(scene_transitions)}, Content={len(content_transitions)}, Motion={len([t for t in all_transition_points if t.type == 'motion_based'])}, Audio={len([t for t in all_transition_points if t.type == 'audio_based'])}")
            
            return optimal_transitions
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Error detecting optimal transitions: {e}")
            return []
    
    def _get_max_transitions_for_style(self, style: EditStyle) -> int:
        """Get maximum number of transitions based on editing style"""
        style_limits = {
            EditStyle.TIKTOK: 8,
            EditStyle.INSTAGRAM: 6,
            EditStyle.YOUTUBE: 10,
            EditStyle.CINEMATIC: 4,
            EditStyle.DRAMATIC: 12
        }
        return style_limits.get(style, 8)
    
    def _create_transition_segments(
        self, 
        transition_points: List[TransitionPoint], 
        analysis_result: VideoAnalysisResult,
        style: EditStyle
    ) -> List[TransitionSegment]:
        """Create transition segments from detected transition points."""
        logger.info(f"📊 [ENHANCED_LLM] Creating transition segments...")
        
        # Use the smart transition detector to create segments
        segments = self.transition_detector.create_segments_from_transitions(
            transitions=transition_points,
            video_duration=analysis_result.duration
        )
        
        logger.info(f"📊 [ENHANCED_LLM] Created {len(segments)} transition segments")
        
        # Log segment details for debugging
        for i, segment in enumerate(segments):
            logger.info(f"📊 [ENHANCED_LLM] Segment {i+1}: {segment.start_time:.2f}s-{segment.end_time:.2f}s (importance: {segment.importance_score:.2f})")
        
        return segments
    
    def generate_moviepy_rendering_plan(
        self,
        analysis_result: VideoAnalysisResult,
        style: EditStyle,
        source_video_path: str,
        output_path: str,
        target_duration: Optional[float] = None,
        output_format: str = "mp4",
        output_quality: str = "high"
    ) -> MoviePyRenderingPlan:
        """
        Generate a MoviePy rendering plan from the enhanced editing plan.
        
        Args:
            analysis_result: Video analysis results
            style: Editing style
            source_video_path: Path to source video
            output_path: Path for output video
            target_duration: Target duration (optional)
            output_format: Output format
            output_quality: Output quality
            
        Returns:
            MoviePyRenderingPlan: Complete rendering plan
        """
        from app.models.schemas import MoviePyRenderingPlan, MoviePySegment, MoviePyEffect, MoviePyTransition
        
        # Generate the enhanced editing plan
        enhanced_plan = self.generate_editing_plan(analysis_result, style, target_duration)
        
        # Convert segments to MoviePySegment objects
        segments = []
        for segment_data in enhanced_plan.segments:
            # Convert effects, excluding speed effects which are handled separately
            effects = []
            speed_factor = 1.0  # Default speed
            
            # Handle EnhancedEditingDecision objects
            if hasattr(segment_data, 'effects'):
                # This is an EnhancedEditingDecision object
                effects_list = segment_data.effects
                start_time = segment_data.start_time
                end_time = segment_data.end_time
                transition_in = getattr(segment_data, 'transition_in', None)
                transition_out = getattr(segment_data, 'transition_out', None)
            else:
                # This is a dictionary (fallback)
                effects_list = segment_data.get("effects", [])
                start_time = segment_data.get("start_time", 0.0)
                end_time = segment_data.get("end_time", 0.0)
                transition_in = segment_data.get("transition_in", None)
                transition_out = segment_data.get("transition_out", None)
            
            for effect_name in effects_list:
                # Handle speed effects - include them in effects list for frontend
                if effect_name == "speed_up":
                    speed_factor = 2.0  # 2x speed
                    # Also add to effects list for frontend compatibility
                    effect = MoviePyEffect(
                        effect_type=effect_name,
                        parameters={"speed_factor": speed_factor},
                        intensity=1.0
                    )
                    effects.append(effect)
                elif effect_name == "slow_motion":
                    speed_factor = 0.5  # 0.5x speed
                    # Also add to effects list for frontend compatibility
                    effect = MoviePyEffect(
                        effect_type=effect_name,
                        parameters={"speed_factor": speed_factor},
                        intensity=1.0
                    )
                    effects.append(effect)
                elif effect_name == "speed":
                    # Get speed_factor from effect parameters if available
                    if hasattr(segment_data, 'speed_factor'):
                        speed_factor = segment_data.speed_factor
                    else:
                        speed_factor = segment_data.get("speed_factor", 1.0)
                    # Also add to effects list for frontend compatibility
                    effect = MoviePyEffect(
                        effect_type=effect_name,
                        parameters={"speed_factor": speed_factor},
                        intensity=1.0
                    )
                    effects.append(effect)
                else:
                    # Regular visual effect
                    effect = MoviePyEffect(
                        effect_type=effect_name,
                        parameters={},
                        intensity=1.0
                    )
                    effects.append(effect)
            
            # Convert transitions
            moviepy_transition_in = None
            if transition_in:
                moviepy_transition_in = MoviePyTransition(
                    transition_type=transition_in,
                    duration=0.5,
                    parameters={},
                    easing="linear"
                )
            
            moviepy_transition_out = None
            if transition_out:
                moviepy_transition_out = MoviePyTransition(
                    transition_type=transition_out,
                    duration=0.5,
                    parameters={},
                    easing="linear"
                )
            
            # Create MoviePySegment with proper speed handling
            segment = MoviePySegment(
                start_time=float(start_time),
                end_time=float(end_time),
                speed=speed_factor,  # Use the calculated speed factor
                volume=1.0,
                effects=effects,  # Only visual effects, not speed effects
                audio_effects=[],
                transition_in=moviepy_transition_in,
                transition_out=moviepy_transition_out,
                crop=None,
                resize=None,
                rotation=0.0,
                brightness=1.0,
                contrast=1.0,
                saturation=1.0,
                gamma=1.0,
                blur=0.0,
                sharpness=1.0
            )
            segments.append(segment)
        
        # Create MoviePyRenderingPlan
        rendering_plan = MoviePyRenderingPlan(
            video_id=analysis_result.video_id,
            source_video_path=source_video_path,
            output_path=output_path,
            target_duration=float(enhanced_plan.target_duration),
            output_format=output_format,
            output_quality=output_quality,
            fps=analysis_result.fps,
            resolution=analysis_result.resolution,
            segments=segments,
            global_effects=[],
            audio_settings={},
            color_settings={},
            style=style.value,  # Add missing style field
            confidence=enhanced_plan.confidence,  # Add missing confidence field
            reasoning=enhanced_plan.reasoning,  # Add missing reasoning field
            metadata={
                "style": enhanced_plan.style,
                "reasoning": enhanced_plan.reasoning,
                "confidence": enhanced_plan.confidence,
                "enhanced_plan": True
            }
        )
        
        logger.info(f"✅ [ENHANCED_LLM] Generated MoviePy rendering plan with {len(segments)} segments")
        return rendering_plan
    
    def _create_editing_plan_structure(
        self, 
        decisions: List[EnhancedEditingDecision], 
        analysis_result: VideoAnalysisResult,
        style: EditStyle
    ) -> Dict[str, Any]:
        """Create the editing plan structure from enhanced decisions."""
        logger.info(f"📋 [ENHANCED_LLM] Creating editing plan structure...")
        
        # Convert decisions to segments format
        segments = []
        for i, decision in enumerate(decisions):
            # Use the AI-generated transition from the decision
            transition_out = decision.transition
            if not transition_out:
                # Fallback to style-appropriate default only if AI didn't provide one
                if style == EditStyle.TIKTOK:
                    transition_out = "whip_pan"
                elif style == EditStyle.YOUTUBE:
                    transition_out = "cross_dissolve"
                elif style == EditStyle.CINEMATIC:
                    transition_out = "fade_out"
                else:
                    transition_out = "cross_dissolve"

            # Set transition_in from previous segment's transition_out, or use style-appropriate default
            transition_in = None
            if i > 0 and segments:
                # Use the previous segment's transition_out as this segment's transition_in
                transition_in = segments[-1].get("transition_out")
            else:
                # For the first segment, use style-appropriate default
                if style == EditStyle.TIKTOK:
                    transition_in = "fade_in"
                elif style == EditStyle.YOUTUBE:
                    transition_in = "cross_dissolve"
                elif style == EditStyle.CINEMATIC:
                    transition_in = "fade_in"
                else:
                    transition_in = "cross_dissolve"

            segment = {
                "start_time": decision.start_time,
                "end_time": decision.end_time,
                "transition_in": transition_in,
                "transition_out": transition_out,
                "transition_duration": "0.5",
                "effects": decision.effects,  # Use the full effects list
                "speed": str(decision.intensity)
            }
            segments.append(segment)
        
        # Generate reasoning for the plan
        reasoning = self._generate_reasoning(
            segments=[],  # We don't have TransitionSegment objects here
            enhanced_transitions=[],  # We don't have enhanced transitions here
            style=style
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence([])
        
        # Create smart detection metadata
        smart_detection_metadata = {
            "total_candidates": len(decisions),
            "selected_transitions": len([d for d in decisions if d.transition_point]),
            "average_score": sum(d.segment_importance for d in decisions) / len(decisions) if decisions else 0.0,
            "style_config": {
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
        
        # Create the editing plan
        editing_plan = {
            "video_id": analysis_result.video_id,
            "style": style.value,
            "segments": segments,
            "notes": f"Enhanced LLM plan: {reasoning}",
            "edit_scale": "1.0",
            "smart_detection_metadata": smart_detection_metadata
        }
        
        logger.info(f"📋 [ENHANCED_LLM] Created editing plan with {len(segments)} segments")
        return editing_plan
    
    def _generate_effects_for_segments(
        self, 
        segments: List[TransitionSegment], 
        analysis_result: VideoAnalysisResult,
        style: EditStyle,
        enhanced_transitions: List[tuple]
    ) -> List[EnhancedEditingDecision]:
        """Generate effects for all segments using a single comprehensive LLM call."""
        logger.info(f"🎯 [ENHANCED_LLM] Generating comprehensive effects plan for {len(segments)} segments...")
        
        # Get all available effects and transitions
        dramatic_effects = self._load_dramatic_effects()
        
        # Create a comprehensive prompt for all segments
        segments_info = []
        for i, segment in enumerate(segments):
            segment_context = self._get_segment_context(segment, analysis_result)
            segments_info.append({
                "segment_id": i + 1,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "importance": segment.importance_score,
                "context": segment_context,
                "start_transition": segment.start_transition.transition_type.value if segment.start_transition else 'None',
                "end_transition": segment.end_transition.transition_type.value if segment.end_transition else 'None'
            })
        
        # Create segments text
        segments_text = ""
        for seg in segments_info:
            segments_text += f"""
Segment {seg['segment_id']}:
- Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s)
- Importance: {seg['importance']:.2f}
- Context: {seg['context']}
- Start transition: {seg['start_transition']}
- End transition: {seg['end_transition']}
"""
        
        # Create comprehensive prompt
        prompt = f"""
You are an expert video editor creating a comprehensive editing plan for a {analysis_result.duration:.2f}s video with {len(segments)} segments. 

VIDEO CONTEXT:
- Style: {style.value}
- Duration: {analysis_result.duration:.2f}s
- FPS: {analysis_result.fps}
- Resolution: {analysis_result.resolution[0]}x{analysis_result.resolution[1]}

CRITICAL TIMING CONSTRAINTS:
- Video duration is EXACTLY {analysis_result.duration:.2f} seconds
- NO segment can start before 0.0s or end after {analysis_result.duration:.2f}s
- Each segment must have a valid start_time and end_time within the video duration
- Segments must be in chronological order (start_time < end_time)
- Minimum segment duration: 0.5 seconds
- Maximum segment duration: {analysis_result.duration:.2f} seconds

AVAILABLE EFFECTS: {dramatic_effects}
AVAILABLE TRANSITIONS: ['cross_dissolve', 'slide', 'zoom', 'whip_pan', 'spin', 'glitch']

SEGMENTS TO EDIT:
{segments_text}

EDITING GUIDELINES:
- Vary effects between segments for visual interest
- Use different transitions to avoid repetition
- Consider segment importance and context
- Layer 2-4 effects per segment for rich visuals
- Match effects to segment timing and motion
- Use beat_sync when beats align with segment timing
- Use motion_blur for high motion segments
- Use frequency_visualizer for audio-reactive segments

TIMING VALIDATION RULES:
- Each segment's start_time must be >= 0.0 and < video duration
- Each segment's end_time must be > start_time and <= video duration
- Segments must not overlap or have gaps
- Total duration of all segments should equal video duration
- Use beat timestamps and motion spikes to create natural segment boundaries
- DO NOT create uniform 0.5-second segments - use actual content analysis!

DETAILED ANALYSIS DATA (Use this to create intelligent segment boundaries):

CHUNKING INSTRUCTIONS:
- Use beat timestamps as natural segment boundaries for rhythm-based editing
- Use motion spikes to create dynamic segments for action sequences
- Use scene changes to create narrative segments
- Use audio peaks for audio-reactive segments
- Vary segment durations based on content importance
- Some segments can be longer (1-2 seconds) for important moments
- Some segments can be shorter (0.5-1 second) for quick transitions
- Create segments based on ACTUAL content analysis, not uniform timing

INTELLIGENT CHUNKING RULES:
- Beat-aligned segments: Use beat timestamps as segment boundaries
- Motion-based segments: Use motion spikes to create dynamic cuts
- Scene-based segments: Use scene changes for narrative flow
- Audio-reactive segments: Use audio peaks for sound-sync editing
- Content-aware segments: Vary duration based on content importance
- Style-appropriate segments: Match chunking to editing style (fast for TikTok, cinematic for YouTube)

"""
        
        # Add detailed beat analysis
        if hasattr(analysis_result, 'beat_detection') and analysis_result.beat_detection:
            prompt += f"- Total beats in video: {len(analysis_result.beat_detection.timestamps)}\n"
            prompt += f"- Beat timestamps: {analysis_result.beat_detection.timestamps}\n"
            if hasattr(analysis_result.beat_detection, 'tempo'):
                prompt += f"- Detected tempo: {analysis_result.beat_detection.tempo:.1f} BPM\n"
        
        # Add detailed motion analysis
        if hasattr(analysis_result, 'motion_analysis') and analysis_result.motion_analysis:
            prompt += f"- Total motion spikes in video: {len(analysis_result.motion_analysis.motion_spikes)}\n"
            prompt += f"- Motion spike timestamps: {analysis_result.motion_analysis.motion_spikes}\n"
            prompt += f"- Total scene changes in video: {len(analysis_result.motion_analysis.scene_changes)}\n"
            prompt += f"- Scene change timestamps: {analysis_result.motion_analysis.scene_changes}\n"
            
            if hasattr(analysis_result.motion_analysis, 'motion_intensity'):
                avg_motion = sum(analysis_result.motion_analysis.motion_intensity) / len(analysis_result.motion_analysis.motion_intensity) if analysis_result.motion_analysis.motion_intensity else 0
                prompt += f"- Average motion intensity: {avg_motion:.3f}\n"
        
        # Add detailed audio analysis
        if hasattr(analysis_result, 'audio_analysis') and analysis_result.audio_analysis:
            prompt += f"- Audio levels: {analysis_result.audio_analysis.volume_levels}\n"
            if hasattr(analysis_result.audio_analysis, 'frequency_analysis'):
                prompt += f"- Frequency data available: {len(analysis_result.audio_analysis.frequency_analysis) if analysis_result.audio_analysis.frequency_analysis else 0} frequency bands\n"
            if hasattr(analysis_result.audio_analysis, 'audio_peaks'):
                audio_peaks_in_segment = [
                    peak for peak in analysis_result.audio_analysis.audio_peaks 
                    if segment.start_time <= peak <= segment.end_time
                ]
                prompt += f"- Audio peak timestamps in segment: {audio_peaks_in_segment}\n"

        prompt += f"""

SEGMENT CREATION INSTRUCTIONS:
- Create segments based on ACTUAL content analysis, not uniform timing
- Use beat timestamps as natural segment boundaries
- Use motion spikes to create dynamic segments
- Use scene changes to create narrative segments
- Use audio peaks for audio-reactive segments
- Vary segment durations based on content importance
- Some segments can be longer (1-2 seconds) for important moments
- Some segments can be shorter (0.5-1 second) for quick transitions

Return ONLY a JSON object with this structure:
{{
    "segments": [
        {{
            "segment_id": 1,
            "start_time": [use actual beat/motion/audio timestamp],
            "end_time": [use next beat/motion/audio timestamp],
            "effects": ["effect1", "effect2", "effect3"],
            "transition": "transition_name",
            "intensity": 1.5,
            "reasoning": "explanation for this segment"
        }},
        ...
    ]
}}

IMPORTANT: 
- Use the analysis data above to create intelligent segment boundaries
- Do NOT create uniform 0.5-second segments
- Ensure all start_time and end_time values are within 0.0 to {analysis_result.duration:.2f} seconds!
- Create segments based on actual beats, motion, and audio content!
"""
        
        # Get comprehensive LLM response
        llm_response = self._call_llm(prompt)
        
        try:
            # Parse response
            response_data = json.loads(llm_response)
            segments_data = response_data.get("segments", [])
            
            # Convert to EnhancedEditingDecision objects using LLM's intelligent timing
            editing_decisions = []
            for i, segment_data in enumerate(segments_data):
                # Use LLM's timing directly, not constrained by dummy segments
                decision = EnhancedEditingDecision(
                    start_time=segment_data.get("start_time", 0.0),
                    end_time=segment_data.get("end_time", analysis_result.duration),
                    effects=segment_data.get("effects", []),
                    intensity=segment_data.get("intensity", 1.0),
                    reasoning=segment_data.get("reasoning", ""),
                    transition=segment_data.get("transition", "cross_dissolve"),
                    segment_importance=0.8  # Default importance for LLM-generated segments
                )
                editing_decisions.append(decision)
            
            logger.info(f"✅ [ENHANCED_LLM] Generated comprehensive plan with {len(editing_decisions)} decisions")
            return editing_decisions
            
        except Exception as e:
            logger.error(f"❌ [ENHANCED_LLM] Failed to parse comprehensive LLM response: {e}")
                    # If no decisions generated, raise an error
        raise Exception("Failed to generate LLM editing decisions")
    
    def _get_llm_effect_decision(
        self, 
        segment: TransitionSegment,
        analysis_result: VideoAnalysisResult,
        style: EditStyle,
        recommended_effects: List[str],
        recommended_transitions: List[str]
    ) -> Dict[str, Any]:
        """
        Get LLM-based effect decision with enhanced analysis data.
        
        Args:
            segment: Transition segment
            analysis_result: Video analysis results including scene and content analysis
            style: Editing style
            recommended_effects: Recommended effects from AutoTransition
            recommended_transitions: Recommended transitions from AutoTransition
            
        Returns:
            Dict containing LLM decision
        """
        try:
            # Get enhanced analysis data
            enhanced_metadata = analysis_result.analysis_metadata
            scene_analysis = enhanced_metadata.get("scene_analysis", {})
            content_analysis = enhanced_metadata.get("content_analysis", {})
            
            # Get scene context for this segment
            segment_scenes = []
            for scene in scene_analysis.get("scenes", []):
                if scene["start_time"] <= segment.start_time <= scene["end_time"]:
                    segment_scenes.append(scene)
            
            # Get content analysis for this segment
            segment_content = content_analysis.get("content_analysis", {})
            
            # Build enhanced context
            scene_context = ""
            if segment_scenes:
                scene = segment_scenes[0]  # Use first matching scene
                scene_context = f"""
Scene Analysis:
- Scene Type: {scene.get('content_analysis', {}).get('scene_type', 'unknown')}
- Scene Duration: {scene.get('duration', 0):.2f}s
- Scene Confidence: {scene.get('confidence', 0):.2f}
- Brightness: {scene.get('content_analysis', {}).get('avg_brightness', 0):.1f}
- Contrast: {scene.get('content_analysis', {}).get('avg_contrast', 0):.1f}
- Motion: {scene.get('content_analysis', {}).get('avg_motion', 0):.1f}
"""
            
            content_context = ""
            if segment_content:
                content_context = f"""
Content Analysis:
- Overall Scene Type: {segment_content.get('overall_scene_type', 'unknown')}
- Action Level: {segment_content.get('avg_action_level', 0):.2f}
- Content Complexity: {segment_content.get('avg_complexity', 0):.2f}
- Analysis Confidence: {segment_content.get('avg_confidence', 0):.2f}
- Analysis Method: {segment_content.get('analysis_method', 'unknown')}
"""
            
            # Build comprehensive prompt with enhanced data
            prompt = f"""You are an expert video editor creating a {style.value} style video edit.

VIDEO ANALYSIS DATA:
Duration: {analysis_result.duration:.2f} seconds
FPS: {analysis_result.fps}
Resolution: {analysis_result.resolution}

AUDIO ANALYSIS:
- Tempo: {analysis_result.beat_detection.bpm:.1f} BPM
- Beat Timestamps: {analysis_result.beat_detection.timestamps[:5]}... (showing first 5)
- Audio Peaks: {len(analysis_result.audio_analysis.audio_peaks)} peaks detected
- Volume Levels: {len(analysis_result.audio_analysis.volume_levels)} levels calculated

MOTION ANALYSIS:
- Motion Spikes: {len(analysis_result.motion_analysis.motion_spikes)} spikes detected
- Motion Intensities: {len(analysis_result.motion_analysis.motion_intensities)} intensity values
- Scene Changes: {len(analysis_result.motion_analysis.scene_changes)} changes detected

{scene_context}
{content_context}

CURRENT SEGMENT:
- Start Time: {segment.start_time:.2f}s
- End Time: {segment.end_time:.2f}s
- Duration: {segment.end_time - segment.start_time:.2f}s
- Importance Score: {segment.importance_score:.2f}
- Transition Type: {segment.transition_type}

RECOMMENDED EFFECTS: {recommended_effects}
RECOMMENDED TRANSITIONS: {recommended_transitions}

AVAILABLE EFFECTS: beat_sync, motion_blur, color_grading, cinematic, vintage, cyberpunk, glitch, audio_pulse, frequency_visualizer, volume_wave, fisheye, twirl, warp, perspective, duotone, invert, high_contrast, film_noir, cartoon

AVAILABLE TRANSITIONS: cross_dissolve, slide, zoom, whip_pan, spin, fade_in, fade_out

TASK: Create an intelligent editing decision for this segment based on the enhanced analysis data.

GUIDELINES:
- Use scene type to choose appropriate effects (e.g., action scenes get motion_blur, dark scenes get cinematic)
- Use content complexity to determine effect intensity
- Use action level to choose dynamic vs static effects
- Use brightness/contrast data to choose color effects
- Sync with beats when timing aligns
- Use recommended effects/transitions when appropriate
- Consider scene duration for pacing decisions

Return ONLY a JSON object with this structure:
{{
    "segment_id": 1,
    "start_time": {segment.start_time:.2f},
    "end_time": {segment.end_time:.2f},
    "effects": ["effect1", "effect2", "effect3"],
    "transition": "transition_name",
    "intensity": 1.5,
    "reasoning": "detailed explanation using scene and content analysis data"
}}
"""
        
            try:
                # Get LLM response
                llm_response = self._call_llm(prompt)
                
                try:
                    # Parse response
                    response_data = json.loads(llm_response)
                    
                    # Validate response structure
                    if "effects" not in response_data or "transition" not in response_data:
                        raise ValueError("Invalid response structure")
                    
                    self.logger.info(f"✅ [ENHANCED_LLM] Generated effect decision using enhanced analysis")
                    return response_data
                    
                except Exception as e:
                    self.logger.error(f"❌ [ENHANCED_LLM] Failed to parse LLM response: {e}")
                    # If LLM fails, raise an error
                    raise Exception("Failed to parse LLM effect decision response")
                
            except Exception as e:
                self.logger.error(f"❌ [ENHANCED_LLM] Failed to parse LLM response: {e}")
                # If LLM fails, raise an error
                raise Exception("Failed to parse LLM effect decision response")
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Error in LLM effect decision: {e}")
            # If LLM fails, raise an error
            raise Exception("Failed to generate LLM effect decision")

    def _load_dramatic_effects(self) -> List[str]:
        """Load ALL effects from the enhanced shader library for LLM access."""
        try:
            from app.editor.enhanced_shader_library import EnhancedShaderLibrary, EffectType
            shader_library = EnhancedShaderLibrary()
            all_effects = shader_library.get_available_effects()
            
            # Return ALL effects from the enhanced shader library
            # This gives the LLM access to all 27 effects for intelligent selection
            self.logger.info(f"✅ [ENHANCED_LLM] Loaded {len(all_effects)} effects from enhanced shader library")
            return all_effects
            
        except ImportError as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Could not import enhanced shader library: {e}")
            # If shader library fails, raise an error
            raise Exception("Failed to import enhanced shader library")

    def _get_dynamic_available_effects(self) -> Dict[str, List[str]]:
        """Dynamically get available effects from the shader library."""
        try:
            from app.editor.enhanced_shader_library import EnhancedShaderLibrary
            shader_library = EnhancedShaderLibrary()
            available_effects = shader_library.get_available_effects()
            
            # Categorize effects based on their names
            categorized_effects = {
                "Visual Effects": [],
                "Audio Reactive Effects": [],
                "Motion Based Effects": [],
                "Style Transfer Effects": [],
                "Geometric Effects": [],
                "Color Effects": [],
                "Speed Effects": [],
                "Transition Effects": []
            }
            
            # Define effect categories based on naming patterns
            for effect in available_effects:
                if effect in ["beat_sync", "frequency_visualizer", "volume_wave", "audio_pulse"]:
                    categorized_effects["Audio Reactive Effects"].append(effect)
                elif effect in ["motion_blur", "optical_flow", "scene_transition", "motion_trail"]:
                    categorized_effects["Motion Based Effects"].append(effect)
                elif effect in ["cinematic", "vintage", "cyberpunk", "film_noir", "cartoon"]:
                    categorized_effects["Style Transfer Effects"].append(effect)
                elif effect in ["fisheye", "twirl", "warp", "perspective"]:
                    categorized_effects["Geometric Effects"].append(effect)
                elif effect in ["color_grading", "duotone", "invert", "high_contrast"]:
                    categorized_effects["Color Effects"].append(effect)
                elif effect in ["speed_up", "slow_motion", "speed"]:
                    categorized_effects["Speed Effects"].append(effect)
                elif effect in ["cross_dissolve", "slide", "zoom", "whip_pan", "spin", "glitch"]:
                    categorized_effects["Transition Effects"].append(effect)
                else:
                    categorized_effects["Visual Effects"].append(effect)
            
            return categorized_effects
            
        except Exception as e:
            logger.error(f"Could not load dynamic effects from shader library: {e}")
            # If shader library fails, raise an error
            raise Exception("Failed to load effects from shader library")

    def _build_dynamic_effects_prompt(self) -> str:
        """Build dynamic effects prompt from shader library."""
        categorized_effects = self._get_dynamic_available_effects()
        
        prompt_sections = []
        
        for category, effects in categorized_effects.items():
            if effects:  # Only include categories that have effects
                effects_list = "\n".join([f"- {effect}" for effect in effects])
                prompt_sections.append(f"{category}:\n{effects_list}")
        
        return "\n\n".join(prompt_sections)

    def _get_segment_context(self, segment: TransitionSegment, analysis_result: VideoAnalysisResult) -> str:
        """Get context information for a segment to help with effect/transition selection."""
        context_parts = []
        
        # Check if segment contains beats
        if hasattr(analysis_result, 'beat_detection') and analysis_result.beat_detection:
            beats_in_segment = [
                beat for beat in analysis_result.beat_detection.timestamps 
                if segment.start_time <= beat <= segment.end_time
            ]
            if beats_in_segment:
                context_parts.append(f"Contains {len(beats_in_segment)} beats")
        
        # Check if segment contains motion spikes
        if hasattr(analysis_result, 'motion_analysis') and analysis_result.motion_analysis:
            motion_in_segment = [
                spike for spike in analysis_result.motion_analysis.motion_spikes 
                if segment.start_time <= spike <= segment.end_time
            ]
            if motion_in_segment:
                context_parts.append(f"High motion activity")
        
        # Check if segment contains scene changes
        if hasattr(analysis_result, 'motion_analysis') and analysis_result.motion_analysis:
            scene_changes_in_segment = [
                change for change in analysis_result.motion_analysis.scene_changes 
                if segment.start_time <= change <= segment.end_time
            ]
            if scene_changes_in_segment:
                context_parts.append(f"Scene change detected")
        
        # Check segment position in video
        video_midpoint = analysis_result.duration / 2
        if segment.start_time < video_midpoint * 0.3:
            context_parts.append("Early in video")
        elif segment.start_time > video_midpoint * 1.7:
            context_parts.append("Late in video")
        else:
            context_parts.append("Middle section")
        
        # Check segment duration
        if segment.duration < 1.5:
            context_parts.append("Short segment")
        elif segment.duration > 3.0:
            context_parts.append("Long segment")
        else:
            context_parts.append("Medium segment")
        
        return ", ".join(context_parts) if context_parts else "Standard segment"
    
    def _extract_transitions(self, decisions: List[EnhancedEditingDecision]) -> List[str]:
        """Extract unique transitions from decisions."""
        transitions = set()
        for decision in decisions:
            if decision.recommended_transitions:
                transitions.update(decision.recommended_transitions)
        return list(transitions)
    
    def _extract_effects(self, decisions: List[EnhancedEditingDecision]) -> List[str]:
        """Extract unique effects from decisions."""
        effects = set()
        for decision in decisions:
            effects.update(decision.effects)  # Changed from decision.effect to decision.effects
            if decision.recommended_effects:
                effects.update(decision.recommended_effects)
        return list(effects)
    
    def _generate_reasoning(
        self, 
        segments: List[TransitionSegment], 
        enhanced_transitions: List[tuple],
        style: EditStyle
    ) -> str:
        """Generate reasoning for the enhanced editing plan."""
        total_segments = len(segments)
        avg_importance = sum(s.importance_score for s in segments) / total_segments if segments else 0
        avg_transition_score = sum(score for _, score in enhanced_transitions) / len(enhanced_transitions) if enhanced_transitions else 0
        
        reasoning = f"""
Enhanced {style.value} editing plan using smart transition detection:
- {total_segments} segments created from {len(enhanced_transitions)} optimal transitions
- Average segment importance: {avg_importance:.2f}
- Average transition score: {avg_transition_score:.2f}
- Smart detection reduced over-editing by selecting only natural transition points
- AutoTransition integration provided professional effect recommendations
- Each segment optimized for style-specific pacing and visual impact
"""
        
        return reasoning.strip()
    
    def _calculate_confidence(self, enhanced_transitions: List[tuple]) -> float:
        """Calculate confidence score for the enhanced editing plan."""
        if not enhanced_transitions:
            return 0.5
        
        # Calculate average confidence from transition points
        total_confidence = sum(transition[2] for transition in enhanced_transitions)
        return total_confidence / len(enhanced_transitions)

    def _get_analysis_duration(self, analysis) -> float:
        """Safely get duration from analysis result (handles both objects and dicts)."""
        if hasattr(analysis, 'duration'):
            return analysis.duration
        else:
            return analysis.get('duration', 0.0)
    
    def _generate_multi_video_editing_plan(
        self,
        analysis_result: VideoAnalysisResult,
        style: EditStyle,
        target_duration: Optional[float],
        multi_video_context: Dict[str, Any]
    ) -> EnhancedEditingPlan:
        """
        Generate enhanced editing plan for multi-video projects.
        
        Args:
            analysis_result: Video analysis result
            style: Editing style
            target_duration: Optional target duration
            multi_video_context: Multi-video context information
            
        Returns:
            EnhancedEditingPlan: Enhanced editing plan
        """
        try:
            self.logger.info("🎬 [ENHANCED_LLM] Generating multi-video editing plan")
            
            # Extract multi-video context
            video_analyses = multi_video_context.get('video_analyses', [])
            cross_analysis = multi_video_context.get('cross_analysis')
            project_id = multi_video_context.get('project_id')
            
            # Generate LLM timeline assignment
            timeline_assignment = self._generate_llm_timeline_assignment(
                video_analyses, cross_analysis, style, target_duration
            )
            
            # Convert to enhanced editing plan
            enhanced_plan = self._convert_timeline_assignment_to_plan(
                timeline_assignment, analysis_result, style
            )
            
            decisions = enhanced_plan.smart_detection_metadata.get('decisions', [])
            self.logger.info(f"✅ [ENHANCED_LLM] Multi-video editing plan generated with {len(decisions)} decisions")
            return enhanced_plan
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Failed to generate multi-video editing plan: {e}")
            raise

    def _generate_llm_timeline_assignment(
        self,
        video_analyses: List[VideoAnalysisResult],
        cross_analysis: Optional[CrossVideoAnalysisResult],
        style: EditStyle,
        target_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate LLM timeline assignment for multi-video project using existing structure.
        
        Args:
            video_analyses: List of video analysis results
            cross_analysis: Optional cross-video analysis result
            style: Editing style
            target_duration: Optional target duration
            
        Returns:
            Dict[str, Any]: LLM-generated timeline assignment
        """
        try:
            self.logger.info("🎬 [ENHANCED_LLM] Generating LLM timeline assignment for multi-video project")
            
            # Create simple prompt for LLM
            prompt = self._create_simple_timeline_prompt(video_analyses, cross_analysis, style, target_duration)
            
            # Call LLM
            response = self._call_llm_for_timeline_simple(prompt)
            
            # Parse LLM response
            timeline_assignment = self._parse_simple_timeline_response(response)
            
            self.logger.info(f"✅ [ENHANCED_LLM] LLM timeline assignment generated with {len(timeline_assignment.get('segments', []))} segments")
            return timeline_assignment
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Failed to generate LLM timeline assignment: {e}")
            # If LLM fails, raise an error
            raise Exception("Failed to generate LLM timeline assignment")

    def _create_simple_timeline_prompt(
        self,
        video_analyses: List[VideoAnalysisResult],
        cross_analysis: Optional[CrossVideoAnalysisResult],
        style: EditStyle,
        target_duration: Optional[float] = None
    ) -> str:
        """Create comprehensive multi-video timeline prompt with rearrange functionality and advanced effects."""
        # Get video IDs from analyses
        video_ids = [str(analysis.video_id) for analysis in video_analyses]
        
        # Get dynamic effects from shader library (like single video workflow)
        try:
            from app.editor.enhanced_shader_library import EnhancedShaderLibrary
            shader_library = EnhancedShaderLibrary()
            available_effects = shader_library.get_available_effects()
            
            # Categorize effects based on their names (like single video workflow)
            categorized_effects = {
                "Visual Effects": [],
                "Audio Reactive Effects": [],
                "Motion Based Effects": [],
                "Style Transfer Effects": [],
                "Geometric Effects": [],
                "Color Effects": [],
                "Speed Effects": [],
                "Transition Effects": []
            }
            
            # Define effect categories based on naming patterns
            for effect in available_effects:
                if effect in ["beat_sync", "frequency_visualizer", "volume_wave", "audio_pulse"]:
                    categorized_effects["Audio Reactive Effects"].append(effect)
                elif effect in ["motion_blur", "optical_flow", "scene_transition", "motion_trail"]:
                    categorized_effects["Motion Based Effects"].append(effect)
                elif effect in ["cinematic", "vintage", "cyberpunk", "film_noir", "cartoon"]:
                    categorized_effects["Style Transfer Effects"].append(effect)
                elif effect in ["fisheye", "twirl", "warp", "perspective"]:
                    categorized_effects["Geometric Effects"].append(effect)
                elif effect in ["color_grading", "duotone", "invert", "high_contrast"]:
                    categorized_effects["Color Effects"].append(effect)
                elif effect in ["speed_up", "slow_motion", "speed"]:
                    categorized_effects["Speed Effects"].append(effect)
                elif effect in ["cross_dissolve", "slide", "zoom", "whip_pan", "spin", "glitch"]:
                    categorized_effects["Transition Effects"].append(effect)
                else:
                    categorized_effects["Visual Effects"].append(effect)
            
            # Build categorized effects prompt
            effects_prompt_sections = []
            for category, effects in categorized_effects.items():
                if effects:
                    effects_list = "\n".join([f"- {effect}" for effect in effects])
                    effects_prompt_sections.append(f"{category}:\n{effects_list}")
            
            effects_prompt = "\n\n".join(effects_prompt_sections)
            
        except Exception as e:
            # If dynamic effects loading fails, raise an error instead of using fallback
            self.logger.error(f"Failed to load dynamic effects: {e}")
            raise RuntimeError(f"Dynamic effects loading failed: {e}")
        
        # Dynamic effects capabilities (loaded from shader library)
        advanced_effects_capabilities = f"""
ADVANCED EFFECTS CAPABILITIES:

The following effects are available from the shader library and can be used in segments:

{effects_prompt}

Each effect is designed to enhance specific aspects of video content:
- Audio reactive effects respond to beat detection and volume changes
- Motion effects enhance movement and transitions
- Style effects apply artistic filters and color grading
- Geometric effects create visual distortions and transformations
- Speed effects control playback timing
- Transition effects create smooth segment connections
"""
        
        # Calculate total available duration FIRST (before using it in style instructions)
        total_available_duration = sum(analysis.duration for analysis in video_analyses)
        
        # Style-specific instructions with advanced effects (from single video workflow)
        style_instructions = {
            EditStyle.TIKTOK: f"""
TikTok Style Guidelines:
- Fast-paced, attention-grabbing cuts
- Use beat-synchronized cuts when beats are detected
- Emphasize high-motion moments
- Keep segments short relative to each video's duration (use 10-40% of each video's length)
- Use dynamic transitions (whip_pan, zoom, spin)
- Apply vibrant color grading (color_grading, high_contrast, cyberpunk)
- Use visual effects for impact (glitch, motion_blur, audio_pulse)
- CRITICAL: Duration MUST be at least 90% of total available content ({total_available_duration * 0.9:.1f} seconds minimum)

Recommended Effects for TikTok:
- Color Grading: color_grading, high_contrast, cyberpunk
- Visual Effects: glitch, motion_blur, audio_pulse
- Transitions: whip_pan, zoom, spin, glitch
- Speed: speed_up, slow_motion
""",
            EditStyle.YOUTUBE: f"""
YouTube Style Guidelines:
- Balanced pacing with clear narrative flow
- Use scene changes for natural breaks
- Emphasize important moments
- Mix fast and slow segments
- Use smooth transitions (cross_dissolve, slide, zoom)
- Apply professional color grading (cinematic, color_grading, high_contrast)
- Use subtle visual effects (vintage, film_noir, cartoon)
- CRITICAL: Duration MUST be at least 90% of total available content ({total_available_duration * 0.9:.1f} seconds minimum)

Recommended Effects for YouTube:
- Color Grading: cinematic, color_grading, high_contrast
- Visual Effects: vintage, film_noir, cartoon
- Transitions: cross_dissolve, slide, zoom, whip_pan
- Speed: speed, slow_motion, speed_up
""",
            EditStyle.CINEMATIC: f"""
Cinematic Style Guidelines:
- Slow, deliberate pacing
- Emphasize dramatic moments
- Use long takes for emotional impact
- Smooth, elegant transitions
- Focus on visual storytelling
- Apply cinematic color grading (cinematic, color_grading, high_contrast)
- Use atmospheric visual effects (vintage, film_noir, cartoon)
- CRITICAL: Duration MUST be at least 90% of total available content ({total_available_duration * 0.9:.1f} seconds minimum)

Recommended Effects for Cinematic:
- Color Grading: cinematic, color_grading, high_contrast, vintage
- Visual Effects: vintage, film_noir, cartoon, glitch
- Transitions: cross_dissolve, slide, zoom, whip_pan
- Speed: slow_motion, speed, speed_up
"""
        }
        
        # Build comprehensive video analysis context with detailed time-based data
        video_contexts = []
        for i, analysis in enumerate(video_analyses):
            context = f"Video {i+1} (ID: {analysis.video_id}): Duration {analysis.duration:.1f}s"
            
            # Add detailed motion analysis with timestamps
            if hasattr(analysis, 'motion_analysis') and analysis.motion_analysis:
                motion = analysis.motion_analysis
                if hasattr(motion, 'motion_score'):
                    context += f", Motion Score: {motion.motion_score:.2f}"
                if hasattr(motion, 'motion_spikes') and motion.motion_spikes:
                    context += f", Motion Spikes: {motion.motion_spikes}"
                if hasattr(motion, 'scene_changes') and motion.scene_changes:
                    context += f", Scene Changes: {motion.scene_changes}"
                if hasattr(motion, 'activity_level'):
                    context += f", Activity Level: {motion.activity_level}"
            
            # Add detailed audio analysis with timestamps
            if hasattr(analysis, 'audio_analysis') and analysis.audio_analysis:
                audio = analysis.audio_analysis
                if hasattr(audio, 'audio_peaks') and audio.audio_peaks:
                    context += f", Audio Peaks: {audio.audio_peaks}"
                if hasattr(audio, 'silence_periods') and audio.silence_periods:
                    context += f", Silence Periods: {audio.silence_periods}"
                if hasattr(audio, 'volume_levels') and len(audio.volume_levels) > 0:
                    context += f", Volume Range: {min(audio.volume_levels):.2f}-{max(audio.volume_levels):.2f}"
            
            # Add detailed beat detection with timestamps
            if hasattr(analysis, 'beat_detection') and analysis.beat_detection:
                beats = analysis.beat_detection
                if hasattr(beats, 'timestamps') and beats.timestamps:
                    context += f", Beat Timestamps: {beats.timestamps}"
                if hasattr(beats, 'bpm'):
                    context += f", BPM: {beats.bpm}"
                if hasattr(beats, 'energy_levels') and beats.energy_levels:
                    context += f", Energy Levels: {beats.energy_levels}"
            
            video_contexts.append(context)
        
        # Build cross-analysis context if available
        cross_analysis_context = ""
        if cross_analysis:
            cross_analysis_context = f"""
        CROSS-VIDEO ANALYSIS:
        - Similarity Matrix: {cross_analysis.similarity_matrix if hasattr(cross_analysis, 'similarity_matrix') else 'N/A'}
        - Cross-video Segments: {len(cross_analysis.cross_video_segments) if hasattr(cross_analysis, 'cross_video_segments') else 0}
        - Analysis Method: {cross_analysis.metadata.get('analysis_method', 'N/A') if hasattr(cross_analysis, 'metadata') else 'N/A'}
        """
        
        prompt = f"""
TASK: Create a multi-video sequence by rearranging segments from different videos for optimal narrative flow using advanced effects and color grading.

🚨 CRITICAL INSTRUCTION - READ THIS FIRST:
The start_time and end_time values you generate MUST be RELATIVE to each individual video's duration, NOT absolute timeline positions.

EXAMPLES OF CORRECT RELATIVE TIMELINE:
- Segment 1: start_time: 0.0, end_time: 2.0 (uses Video 1, 0-2s of Video 1)
- Segment 2: start_time: 1.0, end_time: 3.5 (uses Video 2, 1-3.5s of Video 2)  
- Segment 3: start_time: 0.5, end_time: 2.0 (uses Video 3, 0.5-2s of Video 3)
- Segment 4: start_time: 3.0, end_time: 5.0 (uses Video 1, 3-5s of Video 1)
- Segment 5: start_time: 0.0, end_time: 2.0 (uses Video 2, 0-2s of Video 2)
- Segment 6: start_time: 2.0, end_time: 4.5 (uses Video 3, 2-4.5s of Video 3)

❌ WRONG: Absolute timeline positions
- Segment 1: start_time: 0.0, end_time: 2.0 (absolute timeline)
- Segment 2: start_time: 2.0, end_time: 4.5 (absolute timeline)
- Segment 3: start_time: 4.5, end_time: 6.0 (absolute timeline)

✅ CORRECT: Relative positions within each video
- Segment 1: start_time: 0.0, end_time: 2.0 (relative to Video 1's duration)
- Segment 2: start_time: 1.0, end_time: 3.5 (relative to Video 2's duration)
- Segment 3: start_time: 0.5, end_time: 2.0 (relative to Video 3's duration)

🎬 TIMELINE POSITION REQUIREMENTS:
- start_time and end_time are RELATIVE to each individual video's duration
- Each segment's start_time and end_time must be within the video's actual duration
- For a 5-second video, use start_time: 0-5, end_time: 0-5
- For a 10-second video, use start_time: 0-10, end_time: 0-10
- NEVER use time ranges that exceed the video's actual duration

MULTI-VIDEO CONTEXT:
- Number of videos: {len(video_analyses)}
- Target duration: {target_duration if target_duration else 'Let content determine optimal length'}
- Total available content: {total_available_duration:.1f} seconds
- Editing style: {style.value}
- Video IDs: {video_ids}

DURATION GUIDANCE:
- Total available content: {total_available_duration:.1f} seconds
- TARGET DURATION: {target_duration if target_duration else total_available_duration * 0.9:.1f} seconds
- IMPORTANT: The final video duration will be the sum of all segment durations (segments are concatenated sequentially)
- Each segment duration = end_time - start_time (relative to its source video)
- Total final duration = sum of all segment durations
- CRITICAL: Ensure the sum of all segment durations equals the target duration
- CRITICAL: You MUST use segments from ALL {len(video_analyses)} videos
- CRITICAL: Create enough segments to reach the target duration
- CRITICAL: Each segment's end_time MUST NOT exceed its source video's duration

🚨 REMEMBER: start_time and end_time are RELATIVE to each individual video's duration, never exceed the video's actual duration!
"""
        
        # Dynamic effects capabilities (loaded from shader library)
        advanced_effects_capabilities = f"""
ADVANCED EFFECTS CAPABILITIES:

The following effects are available from the shader library and can be used in segments:

{effects_prompt}

Each effect is designed to enhance specific aspects of video content:
- Audio reactive effects respond to beat detection and volume changes
- Motion effects enhance movement and transitions
- Style effects apply artistic filters and color grading
- Geometric effects create visual distortions and transformations
- Speed effects control playback timing
- Transition effects create smooth segment connections
"""
        
        # Style-specific instructions with advanced effects (from single video workflow)
        style_instructions = {
            EditStyle.TIKTOK: f"""
TikTok Style Guidelines:
- Fast-paced, attention-grabbing cuts
- Use beat-synchronized cuts when beats are detected
- Emphasize high-motion moments
- Keep segments short relative to each video's duration (use 10-40% of each video's length)
- Use dynamic transitions (whip_pan, zoom, spin)
- Apply vibrant color grading (color_grading, high_contrast)
- Use visual effects for impact (glitch, cinematic, vintage)
- CRITICAL: Duration MUST be at least 90% of total available content ({total_available_duration * 0.9:.1f} seconds minimum)

Recommended Effects for TikTok:
- Color Grading: color_grading, high_contrast, duotone
- Visual Effects: glitch, cinematic, vintage
- Transitions: whip_pan, zoom, spin, glitch
- Speed: speed_up, speed
""",
            EditStyle.YOUTUBE: f"""
YouTube Style Guidelines:
- Balanced pacing with clear narrative flow
- Use scene changes for natural breaks
- Emphasize important moments
- Mix fast and slow segments
- Use smooth transitions (crossfade, dissolve, match_cut)
- Apply professional color grading (cinematic, warm, high_contrast)
- Use subtle visual effects (film_grain, vignette)
- CRITICAL: Duration MUST be at least 90% of total available content ({total_available_duration * 0.9:.1f} seconds minimum)

Recommended Effects for YouTube:
- Color Grading: cinematic, warm, high_contrast
- Visual Effects: film_grain, vignette, dramatic_lighting
- Transitions: crossfade, dissolve, match_cut, slide
- Speed: normal, variable_speed, slow_motion
""",
            EditStyle.CINEMATIC: f"""
Cinematic Style Guidelines:
- Slow, deliberate pacing
- Emphasize dramatic moments
- Use long takes for emotional impact
- Smooth, elegant transitions
- Focus on visual storytelling
- Apply cinematic color grading (cinematic, moody, dramatic)
- Use atmospheric visual effects (vintage_film, bleach_bypass)
- CRITICAL: Duration MUST be at least 90% of total available content ({total_available_duration * 0.9:.1f} seconds minimum)

Recommended Effects for Cinematic:
- Color Grading: cinematic, moody, dramatic, vintage
- Visual Effects: vintage_film, bleach_bypass, film_grain, vignette
- Transitions: crossfade, dissolve, fade, morph
- Speed: slow_motion, normal, freeze_frame
"""
        }
        
        # Build comprehensive video analysis context with detailed time-based data
        video_contexts = []
        for i, analysis in enumerate(video_analyses):
            context = f"Video {i+1} (ID: {analysis.video_id}): Duration {analysis.duration:.1f}s"
            
            # Add detailed motion analysis with timestamps
            if hasattr(analysis, 'motion_analysis') and analysis.motion_analysis:
                motion = analysis.motion_analysis
                if hasattr(motion, 'motion_score'):
                    context += f", Motion Score: {motion.motion_score:.2f}"
                if hasattr(motion, 'motion_spikes') and motion.motion_spikes:
                    context += f", Motion Spikes: {motion.motion_spikes}"
                if hasattr(motion, 'scene_changes') and motion.scene_changes:
                    context += f", Scene Changes: {motion.scene_changes}"
                if hasattr(motion, 'activity_level'):
                    context += f", Activity Level: {motion.activity_level}"
            
            # Add detailed audio analysis with timestamps
            if hasattr(analysis, 'audio_analysis') and analysis.audio_analysis:
                audio = analysis.audio_analysis
                if hasattr(audio, 'audio_peaks') and audio.audio_peaks:
                    context += f", Audio Peaks: {audio.audio_peaks}"
                if hasattr(audio, 'silence_periods') and audio.silence_periods:
                    context += f", Silence Periods: {audio.silence_periods}"
                if hasattr(audio, 'volume_levels') and len(audio.volume_levels) > 0:
                    context += f", Volume Range: {min(audio.volume_levels):.2f}-{max(audio.volume_levels):.2f}"
            
            # Add detailed beat detection with timestamps
            if hasattr(analysis, 'beat_detection') and analysis.beat_detection:
                beats = analysis.beat_detection
                if hasattr(beats, 'timestamps') and beats.timestamps:
                    context += f", Beat Timestamps: {beats.timestamps}"
                if hasattr(beats, 'bpm'):
                    context += f", BPM: {beats.bpm}"
                if hasattr(beats, 'energy_levels') and beats.energy_levels:
                    context += f", Energy Levels: {beats.energy_levels}"
            
            video_contexts.append(context)
        
        # Calculate total available duration
        total_available_duration = sum(analysis.duration for analysis in video_analyses)
        
        # Build cross-analysis context if available
        cross_analysis_context = ""
        if cross_analysis:
            cross_analysis_context = f"""
        CROSS-VIDEO ANALYSIS:
        - Similarity Matrix: {cross_analysis.similarity_matrix if hasattr(cross_analysis, 'similarity_matrix') else 'N/A'}
        - Cross-video Segments: {len(cross_analysis.cross_video_segments) if hasattr(cross_analysis, 'cross_video_segments') else 0}
        - Analysis Method: {cross_analysis.metadata.get('analysis_method', 'N/A') if hasattr(cross_analysis, 'metadata') else 'N/A'}
        """
        
        prompt = f"""
        TASK: Create a multi-video sequence by rearranging segments from different videos for optimal narrative flow using advanced effects and color grading.
        
        ⚠️ CRITICAL INSTRUCTION - READ THIS FIRST ⚠️
        The start_time and end_time values you generate MUST be RELATIVE to each individual video's duration, NOT absolute timeline positions.
        - If Video 1 is 10 seconds long, use start_time: 0-10, end_time: 0-10
        - If Video 2 is 5 seconds long, use start_time: 0-5, end_time: 0-5
        - NEVER use values like start_time: 10, end_time: 13 (these are absolute positions)
        - ALWAYS use values like start_time: 2, end_time: 4 (these are relative to the video)
        - CRITICAL: Each segment's end_time MUST NOT exceed the video's actual duration
        - CRITICAL: If a video is 5 seconds long, NEVER use end_time > 5.0

MULTI-VIDEO CONTEXT:
        - Number of videos: {len(video_analyses)}
        - Target duration: {target_duration if target_duration else 'Let content determine optimal length'}
        - Total available content: {total_available_duration:.1f} seconds
- Editing style: {style.value}
        - Video IDs: {video_ids}
        
        DURATION GUIDANCE:
        - Total available content: {total_available_duration:.1f} seconds
        - MINIMUM DURATION REQUIREMENT: The final video MUST be at least 90% of the total available content duration ({total_available_duration * 0.9:.1f} seconds)
        - If no target duration specified: Create engaging content that feels complete and satisfying
        - Aim for natural pacing that keeps viewers engaged
        - Consider the content's narrative arc and emotional flow
        - Balance between showing the best moments and maintaining viewer interest
        - DO NOT exceed the total available content duration of {total_available_duration:.1f} seconds
        - CRITICAL: Ensure the total duration of all segments equals at least 90% of available content
        - CRITICAL: You MUST use segments from ALL {len(video_analyses)} videos to meet the duration requirement
        - CRITICAL: Create at least 8-12 segments to meet the 90% duration requirement
        
        ⚠️ SPEED EFFECTS DURATION IMPACT:
        - beat_sync (1.2x speed): reduces duration by 17% (duration/1.2)
        - audio_pulse (1.1x speed): reduces duration by 9% (duration/1.1)  
        - speed_up (2.0x speed): reduces duration by 50% (duration/2.0)
        - slow_motion (0.5x speed): increases duration by 100% (duration/0.5)
        - When using speed effects, account for their duration impact in your target duration calculation
        - Example: If you want 10s final duration and use speed_up (2.0x), plan for 20s of original content
        - CRITICAL: Multiple speed effects on the same segment are MULTIPLIED (e.g., beat_sync + audio_pulse = 1.2 × 1.1 = 1.32x total speed)
        - CRITICAL: Calculate final segment duration as: original_duration / (speed_effect_1 × speed_effect_2 × ...)
        
        ⚠️ RELATIVE TIME INSTRUCTION - REPEATED ⚠️
        The start_time and end_time in your response should be RELATIVE to each video's duration, not absolute timeline positions.
        - For Video 1 (10s): Use 0-10s range
        - For Video 2 (10s): Use 0-10s range  
        - For Video 3 (5s): Use 0-5s range
        - Do NOT use absolute timeline positions like 10-13s or 17-25s

VIDEO ANALYSIS DATA:
        {chr(10).join(video_contexts)}
        
        {cross_analysis_context}
        
        {advanced_effects_capabilities}
        
        AVAILABLE EFFECTS (use these exact names):
        {effects_prompt}

        {style_instructions.get(style, '')}

        INTELLIGENT TIME CHUNK SELECTION INSTRUCTIONS:
        1. ANALYZE EACH VIDEO'S CONTENT: Use the provided analysis data to identify the best moments
        2. SELECT SPECIFIC TIME CHUNKS: Choose 2-8 second segments based on:
           - Motion spikes: Use segments with high motion activity
           - Audio peaks: Use segments with strong audio moments
           - Scene changes: Use segments around scene transitions
           - Beat timestamps: Use segments aligned with musical beats
           - Energy levels: Use segments with high energy for dynamic parts
        3. ENSURE VALID TIME RANGES: All start_time and end_time must be within the video's duration
        4. CREATE CONTENT-AWARE SEGMENTS: Each segment should represent the best moment from that video
        5. VARY SEGMENT LENGTHS: Use 2-8 seconds based on content intensity
        6. CONSIDER NARRATIVE FLOW: Arrange segments to create engaging story progression
        7. RESPECT VIDEO DURATIONS: Never create segments that exceed the actual video duration
        
        CONTENT-BASED SEGMENT SELECTION:
        - Use Motion Spikes: Select segments around high motion timestamps for dynamic scenes
        - Use Audio Peaks: Select segments around audio peak timestamps for rhythmic moments
        - Use Scene Changes: Select segments around scene change timestamps for narrative transitions
        - Use Beat Timestamps: Select segments aligned with beat timestamps for musical sync
        - Use Energy Levels: Select segments with high energy levels for engaging content

AUDIO SEQUENCING:
        - Audio will follow the rearranged sequence for optimal flow
        - Visual follows the same rearranged sequence for consistency
        - Each segment should specify which video to use for content
        
        SEGMENT REARRANGEMENT STRATEGY:
        - High motion segments → Use motion_blur, optical_flow effects
        - Beat-synchronized segments → Use beat_sync, audio_pulse effects  
        - Scene transitions → Use scene_transition, cross_dissolve effects
        - Style segments → Use cinematic, vintage, cyberpunk effects
        - Geometric segments → Use fisheye, twirl, warp effects
        - Color segments → Use color_grading, duotone, high_contrast effects

        RESPONSE FORMAT:
        Return a JSON array of segments with this exact structure:
        [
          {{
            "video_id": "79d07b2e-42ed-4459-8a1a-c5b55899f503",
            "start_time": 1.5,
            "end_time": 3.5,
            "effects": ["cinematic", "color_grading"],
            "transition_in": "crossfade",
            "transition_out": "slide",
            "segment_purpose": "dynamic_opening"
          }}
        ]

🎬 TRANSITION VARIETY REQUIREMENT:
- Use DIFFERENT transitions for each segment to create visual variety
- Available transitions: crossfade, slide, zoom, whip_pan, spin, glitch, dissolve, fade
- DO NOT use the same transition for all segments
- Match transitions to segment content and style

🚨 CRITICAL: start_time and end_time are RELATIVE to each individual video's duration!
- Each segment's start_time and end_time must be within the video's actual duration
- For a 5-second video, use start_time: 0-5, end_time: 0-5
- For a 10-second video, use start_time: 0-10, end_time: 0-10
- NEVER use time ranges that exceed the video's actual duration

🎨 MANDATORY EFFECTS REQUIREMENT:
- EVERY segment MUST have at least 1-3 effects from the available effects list
- NEVER return empty effects arrays
- Choose effects based on the segment's content and purpose:
  * High motion segments → Use motion_blur, optical_flow, speed_up
  * Beat-synchronized segments → Use beat_sync, audio_pulse, frequency_visualizer
  * Scene transitions → Use scene_transition, cross_dissolve, slide
  * Style segments → Use cinematic, vintage, cyberpunk, film_noir
  * Geometric segments → Use fisheye, twirl, warp, perspective
  * Color segments → Use color_grading, duotone, high_contrast, invert
  * Speed segments → Use speed_up, slow_motion, speed
  * Transition segments → Use cross_dissolve, slide, zoom, whip_pan, spin, glitch

EFFECTS EXAMPLES:
- Opening segment: ["cinematic", "cross_dissolve", "color_grading"]
- High motion segment: ["motion_blur", "speed_up", "high_contrast"]
- Beat sync segment: ["beat_sync", "audio_pulse", "frequency_visualizer"]
- Style segment: ["vintage", "film_noir", "cartoon"]
- Transition segment: ["whip_pan", "zoom", "glitch"]
- Closing segment: ["cross_dissolve", "cinematic", "high_contrast"]

❌ WRONG: "effects": []
✅ CORRECT: "effects": ["cinematic", "cross_dissolve"]

DURATION REQUIREMENT: The sum of all segment durations (end_time - start_time) MUST equal at least 90% of the total available content duration ({total_available_duration * 0.9:.1f} seconds).
        """
        return prompt
        
    def _call_llm_for_timeline_simple(self, prompt: str) -> str:
        """Call LLM for simple timeline generation."""
        try:
            self.logger.info("🤖 [ENHANCED_LLM] Calling LLM for timeline generation")
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video editor. Provide JSON responses."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            llm_response = response.choices[0].message.content
            self.logger.info(f"✅ [ENHANCED_LLM] LLM response received: {len(llm_response)} characters")
            return llm_response
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Failed to call LLM: {e}")
            raise

    def _parse_simple_timeline_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into simple timeline assignment."""
        try:
            import json
            import re
            
            # Extract JSON from response (handle both objects and arrays)
            json_match = re.search(r'(\[.*\]|\{.*\})', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # If data is an array, wrap it in the expected format
            if isinstance(data, list):
                return {
                    "segments": data,
                    "estimated_duration": max([seg.get('end_time', 0) for seg in data]) if data else 0,
                    "overall_strategy": "LLM-generated multi-video timeline",
                    "confidence_score": 0.8
                }
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Failed to parse LLM response: {e}")
            # If LLM fails, raise an error
            raise Exception("Failed to generate LLM timeline assignment")

    def _convert_timeline_assignment_to_plan(
        self,
        timeline_assignment: Dict[str, Any],
        analysis_result: VideoAnalysisResult,
        style: EditStyle
    ) -> EnhancedEditingPlan:
        """
        Convert LLM timeline assignment to enhanced editing plan using existing structure.
        
        Args:
            timeline_assignment: LLM timeline assignment
            analysis_result: Video analysis result
            style: Editing style
            
        Returns:
            EnhancedEditingPlan: Enhanced editing plan
        """
        try:
            decisions = []
            segments = []
            transitions = []
            effects = []
            
            for segment_data in timeline_assignment.get('segments', []):
                # Create enhanced editing decision
                decision = EnhancedEditingDecision(
                    start_time=segment_data.get('start_time', 0.0),
                    end_time=segment_data.get('end_time', 0.0),
                    effects=segment_data.get('effects', []),
                    intensity=segment_data.get('confidence_score', 0.5),
                    reasoning=segment_data.get('llm_reasoning', ''),
                    transition=segment_data.get('transition_out', 'crossfade'),
                    segment_importance=segment_data.get('confidence_score', 0.5),
                    recommended_effects=segment_data.get('effects', []),
                    recommended_transitions=[segment_data.get('transition_in', ''), segment_data.get('transition_out', '')]
                )
                decisions.append(decision)
                
                # Create timeline segment for saving
                from app.models.schemas import TimelineSegment
                from uuid import UUID
                
                # Create enhanced reasoning with analysis basis
                enhanced_reasoning = segment_data.get('reasoning', segment_data.get('llm_reasoning', ''))
                if segment_data.get('analysis_basis'):
                    enhanced_reasoning += f" [Analysis: {segment_data.get('analysis_basis')}]"
                if segment_data.get('content_quality'):
                    enhanced_reasoning += f" [Quality: {segment_data.get('content_quality')}]"
                if segment_data.get('segment_purpose'):
                    enhanced_reasoning += f" [Purpose: {segment_data.get('segment_purpose')}]"
                
                # Create enhanced segment tags
                segment_tags = segment_data.get('segment_tags', [])
                if segment_data.get('analysis_basis'):
                    segment_tags.append(f"analysis:{segment_data.get('analysis_basis')}")
                if segment_data.get('content_quality'):
                    segment_tags.append(f"quality:{segment_data.get('content_quality')}")
                if segment_data.get('segment_purpose'):
                    segment_tags.append(f"purpose:{segment_data.get('segment_purpose')}")
                
                # Log effects processing for debugging
                segment_effects = segment_data.get('effects', [])
                self.logger.info(f"🎨 [ENHANCED_LLM] Segment {len(segments)+1} effects: {segment_effects}")
                
                segment = TimelineSegment(
                    start_time=segment_data.get('start_time', 0.0),
                    end_time=segment_data.get('end_time', 0.0),
                    source_video_id=UUID(segment_data.get('video_id', segment_data.get('source_video_id', str(analysis_result.video_id)))),
                    effects=segment_data.get('effects', []),  # Fixed: use 'effects' (plural) instead of 'effect' (singular)
                    transition_in=segment_data.get('transition_in', 'crossfade'),
                    transition_out=segment_data.get('transition_out', 'crossfade'),
                    effectCustomizations=segment_data.get('effectCustomizations', {}),
                    audio_start_time=segment_data.get('audio_start_time'),
                    audio_end_time=segment_data.get('audio_end_time'),
                    segment_order=segment_data.get('segment_order', len(segments)),
                    llm_reasoning=enhanced_reasoning,
                    confidence_score=segment_data.get('intensity', segment_data.get('confidence_score', 0.5)),
                    segment_tags=segment_tags
                )
                segments.append(segment)
                
                # Extract transitions and effects
                if segment_data.get('transition_in'):
                    transitions.append(segment_data.get('transition_in'))
                if segment_data.get('transition_out'):
                    transitions.append(segment_data.get('transition_out'))
                if segment_data.get('effects'):
                    effects.extend(segment_data.get('effects'))
            
            # Create enhanced editing plan with all required fields
            plan = EnhancedEditingPlan(
                style=style.value,
                target_duration=timeline_assignment.get('estimated_duration', 0.0),
                segments=segments,  # Now populated with timeline segments
                transitions=list(set(transitions)),  # Remove duplicates
                effects=list(set(effects)),  # Remove duplicates
                reasoning=timeline_assignment.get('overall_strategy', ''),
                confidence=timeline_assignment.get('confidence_score', 0.5),
                transition_points=[],
                transition_segments=[],
                smart_detection_metadata={
                    'decisions': [{
                        'start_time': d.start_time,
                        'end_time': d.end_time,
                        'effects': d.effects,
                        'intensity': d.intensity,
                        'reasoning': d.reasoning,
                        'transition': d.transition,
                        'segment_importance': d.segment_importance,
                        'recommended_effects': d.recommended_effects,
                        'recommended_transitions': d.recommended_transitions,
                        'speed_factor': d.speed_factor
                    } for d in decisions],
                    'overall_strategy': timeline_assignment.get('overall_strategy', ''),
                    'narrative_flow': timeline_assignment.get('narrative_flow', ''),
                    'style_justification': timeline_assignment.get('style_justification', ''),
                    'confidence_score': timeline_assignment.get('confidence_score', 0.5),
                    'estimated_duration': timeline_assignment.get('estimated_duration', 0.0)
                }
            )
            
            # Calculate the actual total duration from segments
            actual_total_duration = sum(
                segment.end_time - segment.start_time for segment in segments
            )
            
            # Update the target_duration to reflect the actual calculated duration
            plan.target_duration = actual_total_duration
            
            self.logger.info(f"🎬 [ENHANCED_LLM] Calculated actual total duration: {actual_total_duration:.2f}s from {len(segments)} segments")
            
            return plan
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Failed to convert timeline assignment to plan: {e}")
            raise

    async def _generate_multi_video_timeline_assignment(
        self,
        analysis_result: VideoAnalysisResult,
        style: EditStyle,
        target_duration: Optional[float],
        multi_video_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate LLM timeline assignment for multi-video projects.
        
        Args:
            analysis_result: Video analysis result
            style: Editing style
            target_duration: Optional target duration
            multi_video_context: Multi-video context information
            
        Returns:
            Dict containing timeline assignment details
        """
        try:
            self.logger.info("🎬 [ENHANCED_LLM] Generating LLM timeline assignment")
            
            # Create video analyses list for LLM
            video_analyses = [analysis_result]
            
            # Generate LLM timeline assignment using real LLM
            timeline_assignment = self._generate_llm_timeline_assignment(
                video_analyses=video_analyses,
                cross_analysis=None,
                style=style,
                target_duration=target_duration
            )
            
            self.logger.info(f"✅ [ENHANCED_LLM] Generated timeline assignment with {len(timeline_assignment.get('segments', []))} segments")
            return timeline_assignment
            
        except Exception as e:
            self.logger.error(f"❌ [ENHANCED_LLM] Error generating timeline assignment: {e}")
            raise


# Factory function
def create_enhanced_llm_editor(provider: str = "openai") -> EnhancedLLMEditor:
    """Create an enhanced LLM editor instance."""
    llm_provider = LLMProvider(provider.lower())
    return EnhancedLLMEditor(llm_provider) 

# Global instance for API usage
_enhanced_llm_editor_instance = None

def get_enhanced_llm_editor() -> EnhancedLLMEditor:
    """Get the global enhanced LLM editor instance."""
    global _enhanced_llm_editor_instance
    if _enhanced_llm_editor_instance is None:
        _enhanced_llm_editor_instance = EnhancedLLMEditor()
    return _enhanced_llm_editor_instance 