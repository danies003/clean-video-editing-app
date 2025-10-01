"""
Simple Video Renderer - Consolidated rendering engine.

This module provides the main video rendering functionality using the simplified
SimpleVideoRenderer that handles both single and multi-video projects reliably.
"""

from app.editor.renderer_simple import SimpleVideoRenderer, initialize_renderer, get_renderer

# Re-export the simple renderer as the main renderer
__all__ = ['SimpleVideoRenderer', 'initialize_renderer', 'get_renderer'] 