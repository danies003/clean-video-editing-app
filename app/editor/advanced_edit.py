import random
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID, uuid4

from app.models.schemas import (
    VideoAnalysisResult, 
    EditDecisionMap, 
    EditDecisionSegment,
    EditingTemplate,
    TemplateType,
    QualityPreset,
    EditStyle
)

from app.editor.llm_editor import create_llm_editor, EditingPlan

def _calculate_adaptive_transitions(segments_data, edit_scale):
    """
    Calculate which segments should use crossfades vs advanced transitions.
    Returns two sets: crossfade_segments, advanced_segments
    """
    total_segments = len(segments_data)
    
    # Add randomization to transition distribution
    random.seed(uuid4().int)  # Use random seed for variation
    
    # Calculate transition distribution with randomization
    crossfade_ratio = 0.4 + (random.random() - 0.5) * 0.2  # 30-50% crossfades
    advanced_ratio = 0.3 + (random.random() - 0.5) * 0.2   # 20-40% advanced
    
    num_crossfades = max(1, int(total_segments * crossfade_ratio))
    num_advanced = max(1, int(total_segments * advanced_ratio))
    
    # Randomly select segments for each transition type
    all_indices = list(range(total_segments))
    random.shuffle(all_indices)
    
    crossfade_segments = set(all_indices[:num_crossfades])
    advanced_segments = set(all_indices[num_crossfades:num_crossfades + num_advanced])
    
    return crossfade_segments, advanced_segments

def _select_advanced_transition_type(segment_data, transition_idx):
    """
    Select an advanced transition type based on segment characteristics.
    """
    start = segment_data['start']
    motion_times = segment_data.get('motion_times', [])
    motion_intensities = segment_data.get('motion_intensities', [])
    beat_times = segment_data.get('beat_times', [])
    scene_times = segment_data.get('scene_times', [])
    energy_level = segment_data.get('energy_level', 0.0)
    
    # Add randomization to transition selection
    random.seed(uuid4().int + transition_idx)  # Different seed for each transition
    
    # Calculate motion strength from motion data
    motion_strength = 0.0
    if motion_times and motion_intensities:
        nearby_motions = [intensity for time, intensity in zip(motion_times, motion_intensities) 
                         if abs(time - start) < 0.5]
        motion_strength = max(nearby_motions) / 30.0 if nearby_motions else 0.0
    
    # Check for scene changes nearby (wider detection)
    scene_nearby = len(scene_times) > 0
    
    # Check for beat alignment
    beat_nearby = len(beat_times) > 0
    
    # Add randomization to thresholds
    energy_threshold = 0.2 + (random.random() - 0.5) * 0.1  # 0.15-0.25
    motion_threshold = 0.3 + (random.random() - 0.5) * 0.1  # 0.25-0.35
    high_motion_threshold = 0.4 + (random.random() - 0.5) * 0.1  # 0.35-0.45
    high_energy_threshold = 0.25 + (random.random() - 0.5) * 0.1  # 0.20-0.30
    
    # More sensitive thresholds for transition selection with randomization
    if energy_level > energy_threshold and motion_strength > motion_threshold:  # High energy + motion = shake
        return "shake"
    elif motion_strength > high_motion_threshold:  # High motion = whip pan
        return "whip_pan"
    elif energy_level > high_energy_threshold:  # High energy = zoom blur
        return "zoom_blur"
    elif scene_nearby:  # Scene changes get match_cut
        return "match_cut"
    elif beat_nearby:  # Beat-aligned segments get whip_pan for rhythm
        return "whip_pan"
    else:
        # Randomize default transition
        default_transitions = ["match_cut", "crossfade", "fade"]
        return random.choice(default_transitions)

def generate_llm_edit_plan(
    video_id: UUID,
    analysis: VideoAnalysisResult,
    style: str = "tiktok",
    edit_scale: float = 0.5,
    target_duration: Optional[float] = None
) -> EditDecisionMap:
    """
    Generate an intelligent editing plan using LLM analysis.
    
    Args:
        video_id: UUID of the video to edit
        analysis: Video analysis results from the analysis engine
        style: Editing style preference ("tiktok", "youtube", "cinematic")
        edit_scale: Editing intensity from 0.0 (minimal) to 1.0 (maximum)
        target_duration: Target duration in seconds (optional)
    """
    # Create LLM editor
    llm_editor = create_llm_editor("openai")  # Use OpenAI for real LLM integration
    
    # Convert style string to EditStyle enum
    try:
        edit_style = EditStyle(style)
    except ValueError:
        edit_style = EditStyle.TIKTOK  # Default to TikTok
    
    # Generate LLM editing plan
    llm_plan = llm_editor.generate_editing_plan(
        analysis_result=analysis,
        style=edit_style,
        target_duration=target_duration
    )
    
    # Convert LLM plan to EditDecisionMap
    segments = []
    for llm_segment in llm_plan.segments:
        # Handle transitions - LLM returns complex objects, extract transition type
        transition = "cut"  # default
        if llm_plan.transitions:
            if isinstance(llm_plan.transitions[0], dict):
                # Extract transition type from complex object
                transition = llm_plan.transitions[0].get("type", "cut")
            else:
                # Simple string list
                transition = random.choice(llm_plan.transitions)
        
        segment = EditDecisionSegment(
            start=llm_segment.start_time,
            end=llm_segment.end_time,
            transition=transition,
            transition_duration=0.5,
            tags=[llm_segment.effect],
            speed=llm_segment.intensity
        )
        segments.append(segment)
    
    # Create EditDecisionMap
    edit_plan = EditDecisionMap(
        video_id=video_id,
        style=llm_plan.style,
        segments=segments,
        notes=f"LLM-generated plan: {llm_plan.reasoning} (confidence: {llm_plan.confidence:.2f})",
        edit_scale=edit_scale
    )
    
    return edit_plan


def generate_advanced_edit_plan(
    video_id: UUID,
    analysis: VideoAnalysisResult,
    style: str = "tiktok",
    edit_scale: float = 0.5
) -> EditDecisionMap:
    """
    Generate an advanced editing plan using AI-driven content analysis.
    
    Args:
        video_id: UUID of the video to edit
        analysis: Video analysis results from the analysis engine
        style: Editing style preference ("tiktok", "youtube", "cinematic")
        edit_scale: Editing intensity from 0.0 (minimal) to 1.0 (maximum)
    """
    
    # Set random seed based on video_id and current time for unique results
    # Use microsecond precision for better uniqueness
    seed_value = hash(str(video_id) + str(datetime.utcnow().timestamp()) + str(datetime.utcnow().microsecond))
    random.seed(seed_value)
    
    # Extract analysis data
    beats = analysis.beat_detection.timestamps
    energies = analysis.beat_detection.energy_levels
    motions = analysis.motion_analysis.motion_spikes
    scenes = analysis.motion_analysis.scene_changes
    duration = analysis.duration
    
    # Style-specific parameters with MUCH more randomization
    base_style_params = {
        "tiktok": {"min_segment": 0.3, "max_segment": 2.0, "cuts_per_minute": 30},
        "youtube": {"min_segment": 0.5, "max_segment": 3.0, "cuts_per_minute": 20},
        "cinematic": {"min_segment": 1.0, "max_segment": 5.0, "cuts_per_minute": 12}
    }
    
    base_params = base_style_params.get(style, base_style_params["tiktok"])
    
    # Add MUCH more randomization to style parameters (±50% variation)
    style_params = {
        "min_segment": base_params["min_segment"] * (0.5 + random.random() * 1.0),  # ±50% variation
        "max_segment": base_params["max_segment"] * (0.5 + random.random() * 1.0),  # ±50% variation
        "cuts_per_minute": base_params["cuts_per_minute"] * (0.4 + random.random() * 1.2)  # ±60% variation
    }
    
    # Calculate cut density based on edit_scale and style with MUCH more randomization
    target_cuts = int((style_params["cuts_per_minute"] / 60) * duration * edit_scale)
    target_cuts = max(3, min(target_cuts, int(duration / style_params["min_segment"])))
    
    # Add MUCH more randomization to target cuts (±50% variation)
    target_cuts = int(target_cuts * (0.5 + random.random() * 1.0))
    target_cuts = max(2, min(target_cuts, int(duration / style_params["min_segment"])))
    
    # Generate cut points with MUCH more randomization
    cut_points = [0.0]  # Always start at 0
    
    # Collect all potential cut points with scores
    candidates = []
    
    # Add beats with high scores and MUCH more randomization
    for i, beat in enumerate(beats):
        if 0.5 < beat < duration - 0.5:  # Avoid cuts too close to start/end
            score = energies[i] if i < len(energies) else 0.5
            # Add MUCH more randomization to beat scores (±0.5 variation)
            score += (random.random() - 0.5) * 1.0
            # Add random chance to skip some beats entirely
            if random.random() > 0.3:  # 70% chance to include beat
                candidates.append((beat, score + 0.3))
    
    # Add motion spikes with MUCH more randomization
    for motion in motions:
        if 0.5 < motion < duration - 0.5:
            # Add larger random offset to motion timing (±0.3s variation)
            adjusted_motion = motion + (random.random() - 0.5) * 0.6
            if 0.5 < adjusted_motion < duration - 0.5:
                # Add random chance to skip some motions
                if random.random() > 0.4:  # 60% chance to include motion
                    candidates.append((adjusted_motion, 0.6 + (random.random() - 0.5) * 0.3))
    
    # Add scene changes with MUCH more randomization
    for scene in scenes:
        if 0.5 < scene < duration - 0.5:
            # Add larger random offset to scene timing (±0.3s variation)
            adjusted_scene = scene + (random.random() - 0.5) * 0.6
            if 0.5 < adjusted_scene < duration - 0.5:
                # Add random chance to skip some scenes
                if random.random() > 0.5:  # 50% chance to include scene
                    candidates.append((adjusted_scene, 0.5 + (random.random() - 0.5) * 0.3))
    
    # Add completely random cut points for more variety
    num_random_cuts = random.randint(2, 6)
    for _ in range(num_random_cuts):
        random_time = 0.5 + random.random() * (duration - 1.0)
        candidates.append((random_time, random.random() * 0.5))
    
    # Sort by score and select best cuts
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for cut_time, score in candidates:
        if len(cut_points) >= target_cuts:
            break
        # Ensure minimum distance between cuts with some flexibility
        min_distance = style_params["min_segment"] * (0.7 + random.random() * 0.6)  # ±30% flexibility
        if all(abs(cut_time - existing) > min_distance for existing in cut_points):
            cut_points.append(cut_time)
    
    # Always end at duration
    cut_points.append(duration)
    cut_points = sorted(list(set(cut_points)))
    
    # Create segments data for adaptive transition analysis
    segments_data = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        
        # Convert to new format with proper segment data structure
        segment_data = {
            'start': start,
            'end': end,
            'motion_times': [m for m in motions if start <= m <= end],
            'motion_intensities': [25.0 + (random.random() - 0.5) * 20.0 for _ in [m for m in motions if start <= m <= end]],  # More randomization
            'beat_times': [b for b in beats if start <= b <= end],
            'scene_times': [s for s in scenes if start <= s <= end],
            'energy_level': max([e + (random.random() - 0.5) * 0.3 for j, e in enumerate(energies) if j < len(energies) and beats[j] >= start and beats[j] < end] or [0.1])
        }
        segments_data.append(segment_data)
    
    # Calculate adaptive transitions
    crossfade_segments, advanced_segments = _calculate_adaptive_transitions(segments_data, edit_scale)
    
    # Build segments with adaptive transitions
    segments = []
    transition_idx = 0
    
    for i, segment_data in enumerate(segments_data):
        start = segment_data['start']
        end = segment_data['end']
        
        # Get transition from adaptive system with more randomization
        if i in advanced_segments:
            transition = _select_advanced_transition_type(segment_data, transition_idx)
        elif i in crossfade_segments:
            transition = "crossfade"
        else:
            # Add random chance for different transitions
            if random.random() > 0.7:  # 30% chance for random transition
                transition = random.choice(["match_cut", "crossfade", "fade", "none"])
            else:
                transition = "none"
        
        transition_idx += 1
        energy = max([e for j, e in enumerate(energies) if beats[j] >= start and beats[j] < end] or [0])

        # Recalculate segment properties for tags with MUCH more randomization
        is_high_motion = any(abs(m - start) < 0.3 for m in motions)  # Increased tolerance
        is_on_beat = any(abs(b - start) < 0.2 for b in beats)  # Increased tolerance
        is_scene = any(abs(s - start) < 0.3 for s in scenes)  # Increased tolerance

        tags = []
        # Add MUCH more randomization to tag selection
        if is_high_motion and is_on_beat and edit_scale > 0.2:  # Lowered threshold
            if random.random() > 0.1:  # 90% chance to add highlight tag
                tags.append("highlight")
        if is_scene:
            if random.random() > 0.1:  # 90% chance to add scene tag
                tags.append("scene")
        if energy < 0.2 and edit_scale > 0.5:  # Lowered threshold
            if random.random() > 0.2:  # 80% chance to add skip tag
                tags.append("skip")

        # Speed effects based on energy and edit_scale with MUCH more randomization
        speed = 1.0
        if edit_scale > 0.4:  # Lowered threshold
            if energy > 0.6:  # Lowered threshold
                speed = 1.1 + (random.random() - 0.5) * 0.4  # 0.9-1.3 variation
                if random.random() > 0.1:  # 90% chance to add fast tag
                    tags.append("fast")
            elif energy < 0.3:  # Increased threshold
                speed = 0.7 + (random.random() - 0.5) * 0.4  # 0.5-0.9 variation
                if random.random() > 0.1:  # 90% chance to add slow-mo tag
                    tags.append("slow-mo")

        segment = EditDecisionSegment(
            start=start,
            end=end,
            transition=transition,
            transition_duration=0.3 + (random.random() - 0.5) * 0.4,  # 0.1-0.7s variation
            tags=tags,
            speed=speed
        )
        segments.append(segment)
    
    return EditDecisionMap(
        video_id=video_id,
        style=style,
        segments=segments,
        notes=f"Generated by adaptive editing algorithm with aggressive randomization: content-aware transition distribution; edit_scale={edit_scale}; {len(crossfade_segments)} crossfades, {len(advanced_segments)} advanced transitions; seed={seed_value}.",
        edit_scale=edit_scale
    )

def edit_decision_map_to_template(plan: EditDecisionMap, name: str = "") -> EditingTemplate:
    """
    Convert an EditDecisionMap to an EditingTemplate for use in rendering and registration.
    """
    # Extract all unique effects from plan segments
    all_effects = []
    for segment in plan.segments:
        if segment.tags:
            all_effects.extend(segment.tags)
    
    # Remove duplicates and keep unique effects
    unique_effects = list(set(all_effects))
    
    return EditingTemplate(
        template_id=uuid4(),
        name=name if name else f"Advanced Edit {datetime.utcnow().isoformat()} ({plan.style})",
        description=plan.notes or "Generated by advanced editing algorithm.",
        template_type=TemplateType.BEAT_MATCH,  # or map style to type
        quality_preset=QualityPreset.HIGH,
        effects=unique_effects,  # ✅ Use the advanced effects from plan segments
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    ) 