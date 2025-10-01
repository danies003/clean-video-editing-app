"""
AI Font Integration for Video Editor
Integrates AI-powered font selection with video editing pipeline
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from app.fonts.font_source_manager import HybridFontManager, FontData, FontSource
from app.analyzer.engine import VideoAnalysisResult

logger = logging.getLogger(__name__)

class VideoContentType(Enum):
    CORPORATE = "corporate"
    CREATIVE = "creative"
    TECH = "tech"
    LIFESTYLE = "lifestyle"
    LUXURY = "luxury"
    MINIMALIST = "minimalist"
    VINTAGE = "vintage"
    CYBERPUNK = "cyberpunk"
    NATURE = "nature"
    GAMING = "gaming"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SOCIAL = "social"
    FASHION = "fashion"

class VideoMood(Enum):
    ENERGETIC = "energetic"
    CALM = "calm"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    DRAMATIC = "dramatic"
    MINIMAL = "minimal"
    VINTAGE = "vintage"
    FUTURISTIC = "futuristic"

class TextElementType(Enum):
    TITLE_SEQUENCE = "title_sequence"
    SUBTITLES = "subtitles"
    CAPTIONS = "captions"
    LOWER_THIRD = "lower_third"
    END_CREDITS = "end_credits"
    CALL_TO_ACTION = "call_to_action"
    LOGO = "logo"
    WATERMARK = "watermark"

@dataclass
class TextElement:
    text: str
    element_type: TextElementType
    start_time: float
    end_time: float
    position: Tuple[int, int]
    size: int
    color: str
    font_name: Optional[str] = None
    font_style_tags: List[str] = None

class AIVideoFontAnalyzer:
    """AI analyzer for video content to determine font requirements"""
    
    def __init__(self):
        self.font_manager = HybridFontManager()
    
    async def analyze_video_for_fonts(self, video_analysis: VideoAnalysisResult) -> Dict:
        """Analyze video content to determine font requirements"""
        
        # Extract video characteristics
        content_type = await self._determine_content_type(video_analysis)
        mood = await self._determine_mood(video_analysis)
        complexity = await self._determine_complexity(video_analysis)
        
        # Determine text element requirements
        text_elements = await self._determine_text_elements(video_analysis)
        
        return {
            'content_type': content_type,
            'mood': mood,
            'complexity': complexity,
            'text_elements': text_elements,
            'duration': video_analysis.duration,
            'resolution': video_analysis.resolution
        }
    
    async def _determine_content_type(self, video_analysis: VideoAnalysisResult) -> VideoContentType:
        """Determine video content type using AI analysis"""
        # This would use the video analysis results to determine content type
        # For now, return a default based on available data
        
        if hasattr(video_analysis, 'scene_types') and video_analysis.scene_types:
            # Analyze scene types to determine content
            if any('corporate' in scene.lower() for scene in video_analysis.scene_types):
                return VideoContentType.CORPORATE
            elif any('creative' in scene.lower() for scene in video_analysis.scene_types):
                return VideoContentType.CREATIVE
            elif any('tech' in scene.lower() for scene in video_analysis.scene_types):
                return VideoContentType.TECH
        
        # Default fallback
        return VideoContentType.CORPORATE
    
    async def _determine_mood(self, video_analysis: VideoAnalysisResult) -> VideoMood:
        """Determine video mood using AI analysis"""
        # This would analyze video characteristics to determine mood
        # For now, return a default
        
        if hasattr(video_analysis, 'action_levels') and video_analysis.action_levels:
            avg_action = sum(video_analysis.action_levels) / len(video_analysis.action_levels)
            if avg_action > 0.7:
                return VideoMood.ENERGETIC
            elif avg_action < 0.3:
                return VideoMood.CALM
        
        return VideoMood.PROFESSIONAL
    
    async def _determine_complexity(self, video_analysis: VideoAnalysisResult) -> str:
        """Determine video complexity level"""
        if hasattr(video_analysis, 'content_complexity') and video_analysis.content_complexity:
            avg_complexity = sum(video_analysis.content_complexity) / len(video_analysis.content_complexity)
            if avg_complexity > 0.7:
                return "high"
            elif avg_complexity < 0.3:
                return "low"
        
        return "medium"
    
    async def _determine_text_elements(self, video_analysis: VideoAnalysisResult) -> List[TextElement]:
        """Determine what text elements are needed for the video"""
        # This would analyze video content to determine text requirements
        # For now, return common text elements
        
        text_elements = []
        
        # Add title sequence if video is long enough
        if video_analysis.duration > 30:
            text_elements.append(TextElement(
                text="Video Title",
                element_type=TextElementType.TITLE_SEQUENCE,
                start_time=0,
                end_time=3,
                position=(540, 200),
                size=72,
                color="white"
            ))
        
        # Add subtitles for longer videos
        if video_analysis.duration > 60:
            text_elements.append(TextElement(
                text="Subtitle Text",
                element_type=TextElementType.SUBTITLES,
                start_time=5,
                end_time=video_analysis.duration - 5,
                position=(540, 800),
                size=24,
                color="white"
            ))
        
        # Add call to action for promotional content
        if video_analysis.duration > 10:
            text_elements.append(TextElement(
                text="Learn More",
                element_type=TextElementType.CALL_TO_ACTION,
                start_time=video_analysis.duration - 5,
                end_time=video_analysis.duration,
                position=(540, 900),
                size=36,
                color="yellow"
            ))
        
        return text_elements

class AITextOverlayRenderer:
    """AI-powered text overlay renderer with intelligent font selection"""
    
    def __init__(self):
        self.font_manager = HybridFontManager()
        self.analyzer = AIVideoFontAnalyzer()
    
    async def render_text_overlays(self, video_analysis: VideoAnalysisResult, 
                                 text_elements: List[TextElement]) -> List[Dict]:
        """Render text overlays with AI-selected fonts"""
        
        # Analyze video for font requirements
        font_requirements = await self.analyzer.analyze_video_for_fonts(video_analysis)
        
        # Get AI-recommended fonts
        recommended_fonts = await self.font_manager.get_ai_recommended_fonts(font_requirements)
        
        # Assign fonts to text elements
        rendered_overlays = []
        
        for element in text_elements:
            # Select best font for this text element
            selected_font = await self._select_font_for_element(element, recommended_fonts, font_requirements)
            
            # Render text overlay
            overlay = await self._render_text_element(element, selected_font)
            rendered_overlays.append(overlay)
        
        return rendered_overlays
    
    async def _select_font_for_element(self, element: TextElement, 
                                     recommended_fonts: List[FontData], 
                                     font_requirements: Dict) -> FontData:
        """Select the best font for a specific text element"""
        
        # Filter fonts based on element type
        element_fonts = self._filter_fonts_for_element_type(element.element_type, recommended_fonts)
        
        # Rank fonts based on element requirements
        ranked_fonts = self._rank_fonts_for_element(element, element_fonts, font_requirements)
        
        # Return the best font
        return ranked_fonts[0] if ranked_fonts else recommended_fonts[0]
    
    def _filter_fonts_for_element_type(self, element_type: TextElementType, 
                                     fonts: List[FontData]) -> List[FontData]:
        """Filter fonts based on text element type"""
        
        # Define font preferences for each element type
        element_preferences = {
            TextElementType.TITLE_SEQUENCE: ["display", "bold", "stylized"],
            TextElementType.SUBTITLES: ["sans-serif", "readable", "clean"],
            TextElementType.CAPTIONS: ["sans-serif", "readable", "small"],
            TextElementType.LOWER_THIRD: ["sans-serif", "modern", "clean"],
            TextElementType.END_CREDITS: ["serif", "readable", "classic"],
            TextElementType.CALL_TO_ACTION: ["display", "bold", "attention-grabbing"],
            TextElementType.LOGO: ["display", "stylized", "unique"],
            TextElementType.WATERMARK: ["thin", "subtle", "clean"]
        }
        
        preferred_tags = element_preferences.get(element_type, [])
        
        # Filter fonts that match preferred tags
        filtered_fonts = []
        for font in fonts:
            if any(tag in font.style_tags for tag in preferred_tags):
                filtered_fonts.append(font)
        
        return filtered_fonts if filtered_fonts else fonts
    
    def _rank_fonts_for_element(self, element: TextElement, fonts: List[FontData], 
                              font_requirements: Dict) -> List[FontData]:
        """Rank fonts based on element requirements"""
        
        # Calculate score for each font
        for font in fonts:
            score = 0.0
            
            # Base AI confidence
            score += font.ai_confidence * 0.4
            
            # Popularity score
            score += font.popularity_score * 0.2
            
            # Size appropriateness
            if element.size > 48:  # Large text
                if "display" in font.style_tags or "bold" in font.style_tags:
                    score += 0.2
            else:  # Small text
                if "readable" in font.style_tags or "clean" in font.style_tags:
                    score += 0.2
            
            # Color contrast consideration
            if element.color.lower() in ["white", "yellow", "light"]:
                if "bold" in font.style_tags or "display" in font.style_tags:
                    score += 0.1
            
            # Update font score
            font.ai_confidence = score
        
        # Sort by score
        return sorted(fonts, key=lambda x: x.ai_confidence, reverse=True)
    
    async def _render_text_element(self, element: TextElement, font: FontData) -> Dict:
        """Render a text element with selected font"""
        
        # This would integrate with MoviePy or your video rendering system
        return {
            'text': element.text,
            'font_name': font.name,
            'font_source': font.source.value,
            'start_time': element.start_time,
            'end_time': element.end_time,
            'position': element.position,
            'size': element.size,
            'color': element.color,
            'ai_confidence': font.ai_confidence
        }

# Example usage
async def main():
    """Example of using AI font integration"""
    
    # Create AI text overlay renderer
    renderer = AITextOverlayRenderer()
    
    # Example video analysis (this would come from your video analyzer)
    video_analysis = VideoAnalysisResult(
        duration=120,
        resolution=(1920, 1080),
        scene_types=["corporate", "professional"],
        action_levels=[0.3, 0.4, 0.2],
        content_complexity=[0.6, 0.7, 0.5]
    )
    
    # Example text elements
    text_elements = [
        TextElement(
            text="Welcome to Our Company",
            element_type=TextElementType.TITLE_SEQUENCE,
            start_time=0,
            end_time=3,
            position=(960, 200),
            size=72,
            color="white"
        ),
        TextElement(
            text="Learn more about our services",
            element_type=TextElementType.CALL_TO_ACTION,
            start_time=115,
            end_time=120,
            position=(960, 900),
            size=36,
            color="yellow"
        )
    ]
    
    # Render text overlays with AI-selected fonts
    overlays = await renderer.render_text_overlays(video_analysis, text_elements)
    
    print("AI-Generated Text Overlays:")
    for overlay in overlays:
        print(f"- {overlay['text']} using {overlay['font_name']} (AI Score: {overlay['ai_confidence']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
