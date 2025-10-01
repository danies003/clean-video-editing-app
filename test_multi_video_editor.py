#!/usr/bin/env python3
"""
Test script to isolate the MultiVideoEditor crash
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multi_video_editor():
    """Test MultiVideoEditor initialization"""
    try:
        print("Testing MultiVideoEditor initialization...")
        from app.editor.multi_video_editor import MultiVideoEditor
        editor = MultiVideoEditor()
        print("✅ MultiVideoEditor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ MultiVideoEditor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing MultiVideoEditor...")
    print("=" * 50)
    
    success = test_multi_video_editor()
    
    if success:
        print("\n✅ MultiVideoEditor initialized successfully!")
    else:
        print("\n❌ MultiVideoEditor failed to initialize!")
        sys.exit(1)
