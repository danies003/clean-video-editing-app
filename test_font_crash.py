#!/usr/bin/env python3
"""
Test script to isolate the font renderer crash
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_font_renderer():
    """Test font renderer initialization"""
    try:
        print("Testing VideoFontRenderer initialization...")
        from app.fonts.video_font_renderer import VideoFontRenderer
        renderer = VideoFontRenderer()
        print("✅ VideoFontRenderer initialized successfully")
        return True
    except Exception as e:
        print(f"❌ VideoFontRenderer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing font renderer components...")
    print("=" * 50)
    
    success = test_font_renderer()
    
    if success:
        print("\n✅ Font renderer initialized successfully!")
    else:
        print("\n❌ Font renderer failed to initialize!")
        sys.exit(1)
