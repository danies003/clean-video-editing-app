"""
Video Font Renderer
Integrates AI-selected fonts with video rendering pipeline
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import shutil

from moviepy import TextClip, CompositeVideoClip
from app.fonts.font_source_manager import HybridFontManager, FontData
from app.fonts.ai_font_integration import AITextOverlayRenderer, TextElement, TextElementType

logger = logging.getLogger(__name__)

class VideoFontRenderer:
    """Renders text overlays with AI-selected fonts in videos"""
    
    def __init__(self, font_cache_dir: str = "app/fonts/cache"):
        self.font_cache_dir = Path(font_cache_dir)
        self.font_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.font_manager = HybridFontManager(str(self.font_cache_dir))
        self.ai_renderer = AITextOverlayRenderer()
        
        # Font file cache
        self.font_files = {}
    
    async def add_text_overlays_to_video(self, video_clip, text_elements: List[TextElement], 
                                       video_analysis: Dict) -> CompositeVideoClip:
        """Add AI-selected text overlays to video"""
        
        # Get AI-recommended fonts for text elements
        font_requirements = await self.ai_renderer.analyzer.analyze_video_for_fonts(video_analysis)
        recommended_fonts = await self.font_manager.get_ai_recommended_fonts(font_requirements)
        
        # Create text clips with AI-selected fonts
        text_clips = []
        
        for element in text_elements:
            # Select best font for this element
            selected_font = await self._select_font_for_element(element, recommended_fonts, font_requirements)
            
            # Download font if needed
            font_path = await self._ensure_font_available(selected_font)
            
            # Create text clip
            text_clip = await self._create_text_clip(element, font_path, selected_font)
            if text_clip:
                text_clips.append(text_clip)
        
        # Composite video with text overlays
        if text_clips:
            final_video = CompositeVideoClip([video_clip] + text_clips)
            logger.info(f"Added {len(text_clips)} text overlays to video")
            return final_video
        else:
            logger.warning("No text clips created")
            return video_clip
    
    async def _select_font_for_element(self, element: TextElement, 
                                     recommended_fonts: List[FontData], 
                                     font_requirements: Dict) -> FontData:
        """Select the best font for a text element"""
        
        # Filter fonts based on element type
        element_fonts = self._filter_fonts_for_element_type(element.element_type, recommended_fonts)
        
        if not element_fonts:
            element_fonts = recommended_fonts
        
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
    
    async def _ensure_font_available(self, font: FontData) -> str:
        """Ensure font file is available locally"""
        
        # Check if font is already cached
        font_key = f"{font.name}_{font.source.value}"
        if font_key in self.font_files:
            return self.font_files[font_key]
        
        # Download font if needed
        font_path = await self._download_font(font)
        if font_path:
            self.font_files[font_key] = font_path
            return font_path
        
        # Fallback to system font
        logger.warning(f"Could not load font {font.name}, using system font")
        return None
    
    async def _download_font(self, font: FontData) -> Optional[str]:
        """Download font file"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(font.download_url) as response:
                    if response.status == 200:
                        # Save font file
                        font_filename = f"{font.name.replace(' ', '_')}.ttf"
                        font_path = self.font_cache_dir / font_filename
                        
                        with open(font_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        logger.info(f"Downloaded font {font.name} to {font_path}")
                        return str(font_path)
                    else:
                        logger.error(f"Failed to download font {font.name}: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error downloading font {font.name}: {e}")
            return None
    
    async def _create_text_clip(self, element: TextElement, font_path: str, font: FontData) -> Optional[TextClip]:
        """Create MoviePy TextClip with selected font"""
        try:
            # Create text clip
            text_clip = TextClip(
                element.text,
                font=font_path if font_path else None,
                fontsize=element.size,
                color=element.color,
                method='caption',
                size=(1920, None)  # Fit to video width
            ).set_start(element.start_time).set_end(element.end_time)
            
            # Set position
            text_clip = text_clip.set_position(element.position)
            
            # Add font info to clip for debugging
            text_clip.font_info = {
                'name': font.name,
                'source': font.source.value,
                'ai_confidence': font.ai_confidence
            }
            
            logger.info(f"Created text clip for '{element.text}' with font {font.name}")
            return text_clip
            
        except Exception as e:
            logger.error(f"Error creating text clip for '{element.text}': {e}")
            return None

class FontPreviewGenerator:
    """Generates font previews for UI"""
    
    def __init__(self, font_cache_dir: str = "app/fonts/cache"):
        self.font_cache_dir = Path(font_cache_dir)
        self.preview_cache = {}
    
    async def generate_font_preview(self, font: FontData, text: str = "AaBbCc", 
                                  size: int = 48) -> str:
        """Generate font preview image"""
        
        # Check cache first
        cache_key = f"{font.name}_{text}_{size}"
        if cache_key in self.preview_cache:
            return self.preview_cache[cache_key]
        
        try:
            # Create preview using MoviePy
            preview_clip = TextClip(
                text,
                font=font.name,  # Try system font first
                fontsize=size,
                color='black',
                method='caption',
                size=(300, 100)
            )
            
            # Save preview
            preview_path = self.font_cache_dir / f"preview_{cache_key}.png"
            preview_clip.save_frame(str(preview_path), t=0)
            
            self.preview_cache[cache_key] = str(preview_path)
            return str(preview_path)
            
        except Exception as e:
            logger.error(f"Error generating preview for font {font.name}: {e}")
            return None

# Example usage
async def main():
    """Example of using the video font renderer"""
    
    # Create font renderer
    renderer = VideoFontRenderer()
    
    # Example text elements
    text_elements = [
        TextElement(
            text="Welcome to Our Video",
            element_type=TextElementType.TITLE_SEQUENCE,
            start_time=0,
            end_time=3,
            position=(960, 200),
            size=72,
            color="white"
        ),
        TextElement(
            text="Learn More",
            element_type=TextElementType.CALL_TO_ACTION,
            start_time=10,
            end_time=15,
            position=(960, 900),
            size=36,
            color="yellow"
        )
    ]
    
    # Example video analysis
    video_analysis = {
        'content_type': 'corporate',
        'mood': 'professional',
        'complexity': 'medium',
        'duration': 30
    }
    
    # This would be used with your actual video clip
    # video_with_text = await renderer.add_text_overlays_to_video(video_clip, text_elements, video_analysis)
    
    print("Font renderer ready for video integration")

if __name__ == "__main__":
    asyncio.run(main())



