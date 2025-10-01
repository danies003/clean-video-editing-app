"""
Hybrid Font Source Manager
Combines Google Fonts, Fontesk, and AI-powered font selection
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FontSource(Enum):
    GOOGLE_FONTS = "google_fonts"
    FONTESK = "fontesk"
    SYSTEM = "system"
    CUSTOM = "custom"

@dataclass
class FontData:
    name: str
    source: FontSource
    category: str
    style_tags: List[str]
    download_url: str
    file_size: int
    preview_url: Optional[str] = None
    ai_confidence: float = 0.0
    popularity_score: float = 0.0

class AIStyleCategories:
    """AI-powered font style categorization"""
    
    # Video content-based categories
    CONTENT_CATEGORIES = {
        "corporate": ["professional", "clean", "modern", "sans-serif"],
        "creative": ["artistic", "handwritten", "script", "decorative"],
        "tech": ["futuristic", "digital", "monospace", "geometric"],
        "lifestyle": ["friendly", "rounded", "casual", "humanist"],
        "luxury": ["elegant", "serif", "classic", "sophisticated"],
        "minimalist": ["clean", "thin", "geometric", "modern"],
        "vintage": ["retro", "serif", "classic", "antique"],
        "cyberpunk": ["futuristic", "digital", "neon", "tech"],
        "nature": ["organic", "rounded", "natural", "flowing"],
        "gaming": ["bold", "display", "futuristic", "tech"],
        "education": ["readable", "clean", "sans-serif", "friendly"],
        "entertainment": ["playful", "display", "creative", "bold"],
        "news": ["serif", "readable", "professional", "classic"],
        "social": ["casual", "friendly", "modern", "rounded"],
        "fashion": ["elegant", "sophisticated", "serif", "stylish"]
    }
    
    # Mood-based categories
    MOOD_CATEGORIES = {
        "energetic": ["bold", "display", "futuristic", "dynamic"],
        "calm": ["serif", "elegant", "soft", "rounded"],
        "professional": ["sans-serif", "clean", "modern", "readable"],
        "playful": ["handwritten", "rounded", "casual", "fun"],
        "dramatic": ["display", "bold", "serif", "stylized"],
        "minimal": ["thin", "clean", "geometric", "modern"],
        "vintage": ["serif", "classic", "retro", "antique"],
        "futuristic": ["sans-serif", "geometric", "tech", "modern"]
    }
    
    # Video type categories
    VIDEO_TYPE_CATEGORIES = {
        "title_sequence": ["display", "bold", "stylized", "dramatic"],
        "subtitles": ["sans-serif", "readable", "clean", "modern"],
        "captions": ["sans-serif", "readable", "small", "clean"],
        "lower_third": ["sans-serif", "modern", "clean", "professional"],
        "end_credits": ["serif", "readable", "classic", "elegant"],
        "call_to_action": ["display", "bold", "attention-grabbing", "modern"],
        "logo": ["display", "stylized", "unique", "memorable"],
        "watermark": ["thin", "subtle", "clean", "minimal"]
    }

class GoogleFontsSource:
    """Google Fonts integration with AI categorization"""
    
    def __init__(self):
        self.api_base = "https://fonts.googleapis.com/css2"
        self.cdn_base = "https://fonts.gstatic.com"
        self.fonts_cache = {}
        self.categories = AIStyleCategories()
    
    async def get_fonts_by_ai_category(self, content_type: str, mood: str, video_type: str) -> List[FontData]:
        """Get fonts based on AI-determined categories"""
        # Combine categories for AI selection
        style_tags = []
        style_tags.extend(self.categories.CONTENT_CATEGORIES.get(content_type, []))
        style_tags.extend(self.categories.MOOD_CATEGORIES.get(mood, []))
        style_tags.extend(self.categories.VIDEO_TYPE_CATEGORIES.get(video_type, []))
        
        # Get Google Fonts that match these tags
        return await self._search_fonts_by_tags(style_tags)
    
    async def _search_fonts_by_tags(self, tags: List[str]) -> List[FontData]:
        """Search Google Fonts by style tags"""
        # This would integrate with Google Fonts API
        # For now, return mock data
        return [
            FontData(
                name="Roboto",
                source=FontSource.GOOGLE_FONTS,
                category="sans-serif",
                style_tags=["modern", "clean", "readable"],
                download_url="https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxK.woff2",
                file_size=25000,
                ai_confidence=0.95,
                popularity_score=0.9
            ),
            FontData(
                name="Open Sans",
                source=FontSource.GOOGLE_FONTS,
                category="sans-serif",
                style_tags=["friendly", "readable", "modern"],
                download_url="https://fonts.gstatic.com/s/opensans/v18/mem8YaGs126MiZpBA-UFVZ0b.woff2",
                file_size=30000,
                ai_confidence=0.88,
                popularity_score=0.85
            )
        ]

class FonteskSource:
    """Fontesk integration with AI categorization"""
    
    def __init__(self):
        self.base_url = "https://fontesk.com"
        self.fonts_cache = {}
        self.categories = AIStyleCategories()
    
    async def get_fonts_by_ai_category(self, content_type: str, mood: str, video_type: str) -> List[FontData]:
        """Get Fontesk fonts based on AI-determined categories"""
        # Combine categories for AI selection
        style_tags = []
        style_tags.extend(self.categories.CONTENT_CATEGORIES.get(content_type, []))
        style_tags.extend(self.categories.MOOD_CATEGORIES.get(mood, []))
        style_tags.extend(self.categories.VIDEO_TYPE_CATEGORIES.get(video_type, []))
        
        # Search Fontesk fonts that match these tags
        return await self._search_fontesk_fonts(style_tags)
    
    async def _search_fontesk_fonts(self, tags: List[str]) -> List[FontData]:
        """Search Fontesk fonts by style tags"""
        # This would integrate with your fontesk_scraper.py
        # For now, return mock data
        return [
            FontData(
                name="Cyberpunk Future",
                source=FontSource.FONTESK,
                category="display",
                style_tags=["futuristic", "tech", "cyberpunk", "digital"],
                download_url="https://fontesk.com/download/cyberpunk-future",
                file_size=45000,
                ai_confidence=0.92,
                popularity_score=0.7
            ),
            FontData(
                name="Vintage Script",
                source=FontSource.FONTESK,
                category="script",
                style_tags=["vintage", "handwritten", "elegant", "classic"],
                download_url="https://fontesk.com/download/vintage-script",
                file_size=35000,
                ai_confidence=0.89,
                popularity_score=0.6
            )
        ]

class AIFontSelector:
    """AI-powered font selection based on video content analysis"""
    
    def __init__(self):
        self.google_fonts = GoogleFontsSource()
        self.fontesk = FonteskSource()
        self.categories = AIStyleCategories()
    
    async def select_fonts_for_video(self, video_analysis: Dict) -> List[FontData]:
        """AI selects fonts based on video content analysis"""
        
        # Extract video characteristics
        content_type = video_analysis.get('content_type', 'general')
        mood = video_analysis.get('mood', 'neutral')
        video_type = video_analysis.get('video_type', 'general')
        duration = video_analysis.get('duration', 0)
        complexity = video_analysis.get('complexity', 'medium')
        
        # Get fonts from both sources
        google_fonts = await self.google_fonts.get_fonts_by_ai_category(content_type, mood, video_type)
        fontesk_fonts = await self.fontesk.get_fonts_by_ai_category(content_type, mood, video_type)
        
        # Combine and rank fonts
        all_fonts = google_fonts + fontesk_fonts
        ranked_fonts = self._rank_fonts_by_ai_score(all_fonts, video_analysis)
        
        return ranked_fonts[:10]  # Return top 10 recommendations
    
    def _rank_fonts_by_ai_score(self, fonts: List[FontData], video_analysis: Dict) -> List[FontData]:
        """Rank fonts based on AI confidence and video characteristics"""
        
        for font in fonts:
            # Calculate AI score based on multiple factors
            ai_score = 0.0
            
            # Base confidence score
            ai_score += font.ai_confidence * 0.4
            
            # Popularity score
            ai_score += font.popularity_score * 0.3
            
            # Style matching score
            style_match = self._calculate_style_match(font, video_analysis)
            ai_score += style_match * 0.3
            
            # Add AI score to font data
            font.ai_confidence = ai_score
        
        # Sort by AI score
        return sorted(fonts, key=lambda x: x.ai_confidence, reverse=True)
    
    def _calculate_style_match(self, font: FontData, video_analysis: Dict) -> float:
        """Calculate how well font style matches video characteristics"""
        # This would use more sophisticated AI matching
        # For now, return a simple score
        return 0.8

class HybridFontManager:
    """Main font manager combining all sources with AI selection"""
    
    def __init__(self, cache_dir: str = "app/fonts/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.google_fonts = GoogleFontsSource()
        self.fontesk = FonteskSource()
        self.ai_selector = AIFontSelector()
        
        self.font_cache = {}
        self.preview_cache = {}
    
    async def get_ai_recommended_fonts(self, video_analysis: Dict) -> List[FontData]:
        """Get AI-recommended fonts for video"""
        return await self.ai_selector.select_fonts_for_video(video_analysis)
    
    async def get_font_by_name(self, font_name: str, source: FontSource = None) -> Optional[FontData]:
        """Get specific font by name"""
        # Check cache first
        if font_name in self.font_cache:
            return self.font_cache[font_name]
        
        # Try different sources
        sources = [source] if source else [FontSource.GOOGLE_FONTS, FontSource.FONTESK, FontSource.SYSTEM]
        
        for source_type in sources:
            try:
                if source_type == FontSource.GOOGLE_FONTS:
                    font = await self._get_google_font(font_name)
                elif source_type == FontSource.FONTESK:
                    font = await self._get_fontesk_font(font_name)
                elif source_type == FontSource.SYSTEM:
                    font = await self._get_system_font(font_name)
                
                if font:
                    self.font_cache[font_name] = font
                    return font
            except Exception as e:
                logger.warning(f"Failed to get font {font_name} from {source_type}: {e}")
                continue
        
        return None
    
    async def _get_google_font(self, font_name: str) -> Optional[FontData]:
        """Get font from Google Fonts"""
        # Implementation would call Google Fonts API
        return None
    
    async def _get_fontesk_font(self, font_name: str) -> Optional[FontData]:
        """Get font from Fontesk"""
        # Implementation would use your fontesk_scraper.py
        return None
    
    async def _get_system_font(self, font_name: str) -> Optional[FontData]:
        """Get system font"""
        # Implementation would check system fonts
        return None

# Example usage
async def main():
    """Example of using the AI font system"""
    font_manager = HybridFontManager()
    
    # Example video analysis
    video_analysis = {
        'content_type': 'corporate',
        'mood': 'professional',
        'video_type': 'title_sequence',
        'duration': 30,
        'complexity': 'high'
    }
    
    # Get AI-recommended fonts
    recommended_fonts = await font_manager.get_ai_recommended_fonts(video_analysis)
    
    print("AI Recommended Fonts:")
    for font in recommended_fonts:
        print(f"- {font.name} ({font.source.value}) - AI Score: {font.ai_confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())



