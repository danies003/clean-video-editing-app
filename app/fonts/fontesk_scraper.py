"""
Fontesk Font Scraper
Scrapes all commercial-free fonts from Fontesk and categorizes them for AI selection
"""

import asyncio
import aiohttp
import json
import logging
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import re
import time

logger = logging.getLogger(__name__)

@dataclass
class FonteskFont:
    name: str
    category: str
    tags: List[str]
    download_url: str
    preview_url: str
    file_size: int
    license: str
    designer: str
    description: str
    ai_style_tags: List[str] = None

class FonteskScraper:
    """Scraper for Fontesk commercial-free fonts"""
    
    def __init__(self, base_url: str = "https://fontesk.com/license/free-for-commercial-use"):
        self.base_url = base_url
        self.fonts_data = []
        self.session = None
        self.total_pages = 292  # Based on your information
        
    async def scrape_all_fonts(self) -> List[FonteskFont]:
        """Scrape all fonts from all pages"""
        logger.info(f"Starting to scrape {self.total_pages} pages from Fontesk")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Create tasks for all pages
            tasks = []
            for page in range(1, self.total_pages + 1):
                task = self.scrape_page(page)
                tasks.append(task)
            
            # Process pages in batches to avoid overwhelming the server
            batch_size = 10
            all_fonts = []
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
                
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        all_fonts.extend(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Error in batch: {result}")
                
                # Add delay between batches to be respectful
                await asyncio.sleep(1)
            
            self.fonts_data = all_fonts
            logger.info(f"Scraped {len(all_fonts)} fonts total")
            
            # Save to file
            await self.save_fonts_data()
            
            return all_fonts
    
    async def scrape_page(self, page_num: int) -> List[FonteskFont]:
        """Scrape a single page"""
        url = f"{self.base_url}/page/{page_num}/"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch page {page_num}: {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find font cards - look for articles containing font information
                font_cards = soup.find_all('article')
                
                fonts = []
                for card in font_cards:
                    try:
                        font_data = await self.extract_font_data(card, page_num)
                        if font_data:
                            fonts.append(font_data)
                    except Exception as e:
                        logger.error(f"Error extracting font data from page {page_num}: {e}")
                        continue
                
                logger.info(f"Page {page_num}: Found {len(fonts)} fonts")
                return fonts
                
        except Exception as e:
            logger.error(f"Error scraping page {page_num}: {e}")
            return []
    
    async def extract_font_data(self, card, page_num: int) -> Optional[FonteskFont]:
        """Extract font data from a font card"""
        try:
            # Extract font name - look for text that looks like font names
            name_elem = card.find('h3') or card.find('h2') or card.find('h1')
            if not name_elem:
                # Look for any text that might be a font name
                text_elements = card.find_all(['span', 'div', 'p'])
                for elem in text_elements:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 3 and len(text) < 50 and 'Font' in text:
                        name = text
                        break
                else:
                    return None
            else:
                name = name_elem.get_text(strip=True)
            
            # Extract category
            category_elem = card.find('.category') or card.find('.font-category')
            category = category_elem.get_text(strip=True) if category_elem else "Unknown"
            
            # Extract tags
            tags_elem = card.find('.tags') or card.find('.font-tags')
            tags = []
            if tags_elem:
                tag_links = tags_elem.find_all('a') or tags_elem.find_all('span')
                tags = [tag.get_text(strip=True) for tag in tag_links]
            
            # Extract download URL - try multiple selectors
            download_url = ""
            download_selectors = [
                'a[class*="download"]',
                'a[href*="download"]', 
                'a[href*=".zip"]',
                'a[href*=".ttf"]',
                'a[href*=".otf"]',
                '.download-link',
                '.font-download',
                'a[title*="download"]',
                'a[title*="Download"]'
            ]
            
            for selector in download_selectors:
                download_elem = card.select_one(selector)
                if download_elem and download_elem.get('href'):
                    download_url = download_elem['href']
                    # Make sure it's a full URL
                    if download_url.startswith('/'):
                        download_url = f"https://fontesk.com{download_url}"
                    logger.info(f"Found download URL for {name}: {download_url}")
                    break
            
            if not download_url:
                logger.warning(f"No download URL found for {name}")
            
            # Extract preview URL
            preview_elem = card.find('img') or card.find('.preview')
            preview_url = preview_elem['src'] if preview_elem else ""
            
            # Estimate file size based on font characteristics
            file_size = self.estimate_file_size(name, category, tags)
            
            # Extract license
            license_elem = card.find('.license') or card.find('.font-license')
            license_text = license_elem.get_text(strip=True) if license_elem else "Free for commercial use"
            
            # Extract designer
            designer_elem = card.find('.designer') or card.find('.author')
            designer = designer_elem.get_text(strip=True) if designer_elem else "Unknown"
            
            # Extract description
            desc_elem = card.find('.description') or card.find('.font-desc')
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # Generate AI style tags
            ai_style_tags = self.generate_ai_style_tags(name, category, tags, description)
            
            return FonteskFont(
                name=name,
                category=category,
                tags=tags,
                download_url=download_url,
                preview_url=preview_url,
                file_size=file_size,
                license=license_text,
                designer=designer,
                description=description,
                ai_style_tags=ai_style_tags
            )
            
        except Exception as e:
            logger.error(f"Error extracting font data: {e}")
            return None
    
    def estimate_file_size(self, name: str, category: str, tags: List[str]) -> int:
        """Estimate font file size based on characteristics"""
        base_size = 25000  # 25KB base
        
        # Adjust based on category
        category_multipliers = {
            'display': 1.5,
            'script': 1.3,
            'handwritten': 1.2,
            'decorative': 1.4,
            'sans-serif': 1.0,
            'serif': 1.1
        }
        
        multiplier = category_multipliers.get(category.lower(), 1.0)
        
        # Adjust based on tags
        if 'bold' in ' '.join(tags).lower():
            multiplier += 0.2
        if 'italic' in ' '.join(tags).lower():
            multiplier += 0.1
        if 'variable' in ' '.join(tags).lower():
            multiplier += 0.5
        
        return int(base_size * multiplier)
    
    def generate_ai_style_tags(self, name: str, category: str, tags: List[str], description: str) -> List[str]:
        """Generate AI style tags for font categorization"""
        ai_tags = []
        
        # Add category-based tags
        category_mapping = {
            'sans-serif': ['modern', 'clean', 'readable'],
            'serif': ['classic', 'elegant', 'traditional'],
            'script': ['handwritten', 'elegant', 'artistic'],
            'display': ['bold', 'attention-grabbing', 'stylized'],
            'handwritten': ['casual', 'personal', 'artistic'],
            'decorative': ['ornate', 'stylized', 'artistic']
        }
        
        ai_tags.extend(category_mapping.get(category.lower(), []))
        
        # Add tags from existing tags
        tag_mapping = {
            'bold': ['bold', 'strong'],
            'italic': ['elegant', 'stylish'],
            'thin': ['minimal', 'clean'],
            'rounded': ['friendly', 'soft'],
            'geometric': ['modern', 'tech'],
            'vintage': ['retro', 'classic'],
            'futuristic': ['tech', 'modern'],
            'elegant': ['sophisticated', 'luxury'],
            'casual': ['friendly', 'approachable'],
            'professional': ['clean', 'readable']
        }
        
        for tag in tags:
            tag_lower = tag.lower()
            for key, values in tag_mapping.items():
                if key in tag_lower:
                    ai_tags.extend(values)
        
        # Add tags based on name analysis
        name_lower = name.lower()
        if 'cyber' in name_lower or 'tech' in name_lower:
            ai_tags.extend(['futuristic', 'tech', 'digital'])
        if 'vintage' in name_lower or 'retro' in name_lower:
            ai_tags.extend(['vintage', 'retro', 'classic'])
        if 'elegant' in name_lower or 'luxury' in name_lower:
            ai_tags.extend(['elegant', 'luxury', 'sophisticated'])
        
        # Remove duplicates and return
        return list(set(ai_tags))
    
    async def save_fonts_data(self):
        """Save scraped fonts data to JSON file"""
        output_file = Path("app/fonts/fontesk_fonts.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        fonts_dict = [asdict(font) for font in self.fonts_data]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fonts_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(fonts_dict)} fonts to {output_file}")
    
    async def load_fonts_data(self) -> List[FonteskFont]:
        """Load fonts data from JSON file"""
        output_file = Path("app/fonts/fontesk_fonts.json")
        
        if not output_file.exists():
            logger.warning("Fonts data file not found. Run scraping first.")
            return []
        
        with open(output_file, 'r', encoding='utf-8') as f:
            fonts_dict = json.load(f)
        
        fonts = [FonteskFont(**font_data) for font_data in fonts_dict]
        logger.info(f"Loaded {len(fonts)} fonts from {output_file}")
        
        return fonts

# Example usage
async def main():
    """Example of using the Fontesk scraper"""
    scraper = FonteskScraper()
    
    # Scrape all fonts
    fonts = await scraper.scrape_all_fonts()
    
    print(f"Scraped {len(fonts)} fonts")
    
    # Show some examples
    for i, font in enumerate(fonts[:5]):
        print(f"\n{i+1}. {font.name}")
        print(f"   Category: {font.category}")
        print(f"   Tags: {', '.join(font.tags)}")
        print(f"   AI Tags: {', '.join(font.ai_style_tags)}")
        print(f"   Size: {font.file_size} bytes")

if __name__ == "__main__":
    asyncio.run(main())
