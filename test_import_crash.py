#!/usr/bin/env python3
"""
Test script to isolate the import crash in job context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import_in_job_context():
    """Test imports that happen in job processing"""
    try:
        print("Testing imports in job processing context...")
        
        # Test the exact import that happens in the job
        from app.editor.multi_video_editor import MultiVideoEditor
        print("✅ MultiVideoEditor import successful")
        
        # Test creating an instance
        editor = MultiVideoEditor()
        print("✅ MultiVideoEditor instantiation successful")
        
        return True
    except Exception as e:
        print(f"❌ Import/instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing job context imports...")
    print("=" * 50)
    
    success = test_import_in_job_context()
    
    if success:
        print("\n✅ All imports successful!")
    else:
        print("\n❌ Import failed!")
        sys.exit(1)
