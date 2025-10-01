"""
Scene Detection Module using PySceneDetect
Provides scene-aware analysis for video editing decisions
"""

import logging
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from scenedetect import detect, ContentDetector, ThresholdDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.video_manager import VideoManager
from scenedetect.frame_timecode import FrameTimecode

logger = logging.getLogger(__name__)


class SceneDetector:
    """Enhanced scene detection using PySceneDetect with multiple detection methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_scenes(self, video_path: str, method: str = "content") -> List[Dict]:
        """
        Detect scenes in video using specified method
        
        Args:
            video_path: Path to video file
            method: Detection method ("content", "threshold", "adaptive")
            
        Returns:
            List of scene dictionaries with timing and metadata
        """
        try:
            self.logger.info(f"🔍 [SCENE_DETECT] Starting scene detection with method: {method}")
            
            # Initialize video manager
            video_manager = VideoManager([video_path])
            stats_manager = StatsManager()
            scene_manager = SceneManager(stats_manager)
            
            # Add detector based on method
            if method == "content":
                scene_manager.add_detector(ContentDetector(threshold=27.0))
            elif method == "threshold":
                scene_manager.add_detector(ThresholdDetector(threshold=12))
            elif method == "adaptive":
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3))
            else:
                scene_manager.add_detector(ContentDetector(threshold=27.0))
            
            # Start video manager
            video_manager.set_duration()
            video_manager.set_fps()
            
            # Start scene detection
            scene_manager.detect_scenes(frame_source=video_manager)
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            
            # Convert to our format
            scenes = []
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                scene_data = {
                    "scene_id": i + 1,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "start_frame": scene[0].get_frames(),
                    "end_frame": scene[1].get_frames(),
                    "method": method,
                    "confidence": self._calculate_scene_confidence(scene, stats_manager)
                }
                scenes.append(scene_data)
            
            self.logger.info(f"✅ [SCENE_DETECT] Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            self.logger.error(f"❌ [SCENE_DETECT] Error detecting scenes: {str(e)}")
            return []
    
    def _calculate_scene_confidence(self, scene: Tuple, stats_manager: StatsManager) -> float:
        """Calculate confidence score for scene detection"""
        try:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            
            # Get content values for frames around scene boundary
            content_values = []
            for frame_num in range(max(0, start_frame - 5), min(end_frame + 5, 1000)):
                if stats_manager.master_stats.get(frame_num):
                    content_values.append(stats_manager.master_stats[frame_num].content_val)
            
            if content_values:
                # Calculate variance as confidence indicator
                variance = np.var(content_values)
                confidence = min(1.0, variance / 100.0)  # Normalize to 0-1
                return confidence
            
            return 0.5  # Default confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ [SCENE_DETECT] Error calculating confidence: {str(e)}")
            return 0.5
    
    def analyze_scene_content(self, video_path: str, scene: Dict) -> Dict:
        """
        Analyze content within a specific scene
        
        Args:
            video_path: Path to video file
            scene: Scene dictionary with timing info
            
        Returns:
            Scene content analysis
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Sample frames from scene
            start_frame = int(scene["start_time"] * cap.get(cv2.CAP_PROP_FPS))
            end_frame = int(scene["end_time"] * cap.get(cv2.CAP_PROP_FPS))
            
            frame_count = end_frame - start_frame
            sample_interval = max(1, frame_count // 10)  # Sample 10 frames
            
            brightness_values = []
            contrast_values = []
            motion_values = []
            
            prev_frame = None
            
            for i in range(start_frame, end_frame, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Brightness analysis
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Contrast analysis
                contrast = np.std(gray)
                contrast_values.append(contrast)
                
                # Motion analysis
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    motion = np.mean(diff)
                    motion_values.append(motion)
                
                prev_frame = gray
            
            cap.release()
            
            # Calculate scene characteristics
            analysis = {
                "avg_brightness": np.mean(brightness_values) if brightness_values else 0,
                "avg_contrast": np.mean(contrast_values) if contrast_values else 0,
                "avg_motion": np.mean(motion_values) if motion_values else 0,
                "brightness_variance": np.var(brightness_values) if brightness_values else 0,
                "contrast_variance": np.var(contrast_values) if contrast_values else 0,
                "motion_variance": np.var(motion_values) if motion_values else 0,
                "scene_type": self._classify_scene_type(brightness_values, contrast_values, motion_values),
                "sample_frames": len(brightness_values)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ [SCENE_DETECT] Error analyzing scene content: {str(e)}")
            return {}
    
    def _classify_scene_type(self, brightness: List, contrast: List, motion: List) -> str:
        """Classify scene type based on visual characteristics"""
        try:
            avg_brightness = np.mean(brightness) if brightness else 0
            avg_contrast = np.mean(contrast) if contrast else 0
            avg_motion = np.mean(motion) if motion else 0
            
            # Scene type classification logic
            if avg_motion > 20:
                if avg_brightness > 150:
                    return "action_bright"
                else:
                    return "action_dark"
            elif avg_contrast > 50:
                if avg_brightness > 150:
                    return "high_contrast_bright"
                else:
                    return "high_contrast_dark"
            elif avg_brightness > 150:
                return "bright_static"
            elif avg_brightness < 50:
                return "dark_static"
            else:
                return "normal_static"
                
        except Exception as e:
            self.logger.warning(f"⚠️ [SCENE_DETECT] Error classifying scene: {str(e)}")
            return "unknown"
    
    def get_optimal_transition_points(self, scenes: List[Dict], 
                                    min_duration: float = 1.0,
                                    max_duration: float = 20.0) -> List[Dict]:  # Increased from 10.0 to accommodate longer videos
        """
        Get optimal transition points based on scene boundaries
        
        Args:
            scenes: List of detected scenes
            min_duration: Minimum segment duration
            max_duration: Maximum segment duration
            
        Returns:
            List of optimal transition points
        """
        transition_points = []
        
        for scene in scenes:
            duration = scene["duration"]
            
            # Check if scene duration is within acceptable range
            if min_duration <= duration <= max_duration:
                transition_point = {
                    "time": scene["end_time"],
                    "confidence": scene["confidence"],
                    "scene_id": scene["scene_id"],
                    "duration": duration,
                    "type": "scene_boundary",
                    "content_analysis": scene.get("content_analysis", {})
                }
                transition_points.append(transition_point)
        
        # Sort by confidence
        transition_points.sort(key=lambda x: x["confidence"], reverse=True)
        
        return transition_points


def create_scene_detector() -> SceneDetector:
    """Factory function to create scene detector instance"""
    return SceneDetector() 