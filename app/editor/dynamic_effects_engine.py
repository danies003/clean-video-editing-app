"""
Dynamic Effects Engine

This module provides LLM-generated effects and fallback effects
for the enhanced video editing system.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable
from app.editor.llm_editor import LLMEditor, LLMProvider

logger = logging.getLogger(__name__)


class DynamicEffectsEngine:
    """
    Engine for generating dynamic effects using LLM and fallback mechanisms.
    """
    
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        """Initialize the dynamic effects engine."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm_editor = LLMEditor(provider=llm_provider)
        
        # Fallback effects for when LLM generation fails
        self.fallback_effects = {
            "beat_sync": self._create_beat_sync_effect,
            "motion_blur": self._create_motion_blur_effect,
            "cinematic": self._create_cinematic_effect,
            "color_grading": self._create_color_grading_effect,
            "high_contrast": self._create_high_contrast_effect,
            "glitch": self._create_glitch_effect,
            "vintage": self._create_vintage_effect,
            "neon": self._create_neon_effect,
            "dramatic": self._create_dramatic_effect,
            "fast_pace": self._create_fast_pace_effect,
            "slow_motion": self._create_slow_motion_effect,
            "scene_transition": self._create_scene_transition_effect
        }
    
    def generate_effect(self, effect_name: str, **kwargs) -> Optional[Callable]:
        """
        Generate an effect function using LLM or fallback.
        
        Args:
            effect_name: Name of the effect to generate
            **kwargs: Additional parameters for the effect
            
        Returns:
            Effect function or None if generation fails
        """
        try:
            # Try LLM generation first
            effect_func = self._generate_llm_effect(effect_name, **kwargs)
            if effect_func:
                return effect_func
            
            # Fallback to predefined effects
            if effect_name in self.fallback_effects:
                self.logger.info(f"✅ [DYNAMIC_EFFECTS] Using fallback effect: {effect_name}")
                return self.fallback_effects[effect_name](**kwargs)
            
            self.logger.warning(f"⚠️ [DYNAMIC_EFFECTS] Effect not found: {effect_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ [DYNAMIC_EFFECTS] Error generating effect {effect_name}: {e}")
            return None
    
    def _generate_llm_effect(self, effect_name: str, **kwargs) -> Optional[Callable]:
        """
        Generate an effect using LLM.
        
        Args:
            effect_name: Name of the effect to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated effect function or None
        """
        start_time = time.time()
        
        try:
            # Create LLM prompt for effect generation
            prompt = self._create_effect_prompt(effect_name, **kwargs)
            
            # Generate effect code using LLM
            response = self.llm_editor._call_openai(prompt)
            
            # Extract and validate the generated code
            effect_func = self._extract_and_validate_effect(response, effect_name)
            
            if effect_func:
                generation_time = time.time() - start_time
                self.logger.info(f"⏱️ LLM generation took {generation_time:.2f}s for: {effect_name}")
                return effect_func
            
        except Exception as e:
            self.logger.warning(f"⚠️ [DYNAMIC_EFFECTS] LLM generation failed for {effect_name}: {e}")
        
        return None
    
    def _create_effect_prompt(self, effect_name: str, **kwargs) -> str:
        """Create a prompt for LLM effect generation."""
        return f"""
You are an expert video effects programmer. Create a Python function for the "{effect_name}" effect.

Requirements:
- Function name: {effect_name}_effect
- Parameters: frame (numpy array), intensity (float), time (float)
- Return: Modified frame as numpy array
- Use OpenCV (cv2) and NumPy for image processing
- Intensity should control effect strength (0.0 to 1.0)
- Time parameter can be used for animated effects
- Ensure the function is safe and handles edge cases

Effect description:
{self._get_effect_description(effect_name)}

Additional parameters: {kwargs}

Return ONLY the Python function code, no explanations or markdown formatting.
"""
    
    def _get_effect_description(self, effect_name: str) -> str:
        """Get description for an effect."""
        descriptions = {
            "beat_sync": "Synchronize visual effects with audio beats, creating pulsing or rhythmic changes",
            "motion_blur": "Add motion blur effect to simulate camera movement or fast motion",
            "cinematic": "Apply cinematic color grading and visual style",
            "color_grading": "Adjust colors for artistic or technical purposes",
            "high_contrast": "Increase contrast and dramatic visual impact",
            "glitch": "Create digital glitch effects with artifacts and distortions",
            "vintage": "Apply vintage film look with color shifts and grain",
            "neon": "Add neon glow effects and bright color highlights",
            "dramatic": "Create dramatic lighting and contrast effects",
            "fast_pace": "Speed up visual elements for fast-paced editing",
            "slow_motion": "Slow down visual elements for dramatic effect",
            "scene_transition": "Create smooth transitions between scenes"
        }
        return descriptions.get(effect_name, f"Create a {effect_name} visual effect")
    
    def _extract_and_validate_effect(self, response: str, effect_name: str) -> Optional[Callable]:
        """
        Extract and validate the generated effect function.
        
        Args:
            response: LLM response containing the effect code
            effect_name: Name of the effect
            
        Returns:
            Validated effect function or None
        """
        try:
            # Extract code from response
            code = self._extract_code_from_response(response)
            if not code:
                return None
            
            # Create a namespace for execution
            namespace = {}
            
            # Execute the code to create the function
            exec(code, namespace)
            
            # Get the function
            func_name = f"{effect_name}_effect"
            if func_name not in namespace:
                self.logger.warning(f"⚠️ [DYNAMIC_EFFECTS] Function {func_name} not found in generated code")
                return None
            
            effect_func = namespace[func_name]
            
            # Validate the function
            if self._validate_effect_function(effect_func, effect_name):
                self.logger.info(f"✅ [DYNAMIC_EFFECTS] Successfully created effect function for: {effect_name}")
                return effect_func
            
        except Exception as e:
            self.logger.error(f"❌ [DYNAMIC_EFFECTS] Error extracting effect: {e}")
        
        return None
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                data = json.loads(response)
                return data.get('code', '')
            
            # Otherwise, extract code blocks
            lines = response.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                
                if in_code_block:
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines)
            
            # If no code blocks, return the whole response
            return response.strip()
            
        except Exception as e:
            self.logger.warning(f"⚠️ [DYNAMIC_EFFECTS] Error extracting code: {e}")
            return response.strip()
    
    def _validate_effect_function(self, func: Callable, effect_name: str) -> bool:
        """
        Validate that the generated effect function works correctly.
        
        Args:
            func: The effect function to validate
            effect_name: Name of the effect
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            import numpy as np
            import cv2
            
            # Create a test frame
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Test the function
            result = func(test_frame, 0.5, 0.0)
            
            # Check that the result is valid
            if result is None:
                return False
            
            if not isinstance(result, np.ndarray):
                return False
            
            if result.shape != test_frame.shape:
                return False
            
            if result.dtype != np.uint8:
                return False
            
            self.logger.info(f"✅ Enhanced validation passed for: {effect_name}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ [DYNAMIC_EFFECTS] Validation failed for {effect_name}: {e}")
            return False
    
    # Fallback effect implementations
    def _create_beat_sync_effect(self, **kwargs):
        """Create a beat sync effect."""
        def beat_sync_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Create a pulsing effect based on time
            pulse = np.sin(time * 2 * np.pi) * 0.5 + 0.5
            pulse_intensity = intensity * pulse
            
            # Apply brightness adjustment
            brightened = cv2.convertScaleAbs(frame, alpha=1 + pulse_intensity, beta=pulse_intensity * 50)
            
            return brightened
        
        return beat_sync_effect
    
    def _create_motion_blur_effect(self, **kwargs):
        """Create a motion blur effect."""
        def motion_blur_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Create motion blur kernel
            kernel_size = int(intensity * 15) + 1
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1
            kernel = kernel / kernel_size
            
            # Apply motion blur
            blurred = cv2.filter2D(frame, -1, kernel)
            
            return blurred
        
        return motion_blur_effect
    
    def _create_cinematic_effect(self, **kwargs):
        """Create a cinematic effect."""
        def cinematic_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Apply cinematic color grading
            # Increase contrast and saturation
            contrast = 1 + intensity * 0.5
            saturation = 1 + intensity * 0.3
            
            # Convert to HSV for saturation adjustment
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Apply contrast
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
            
            return frame
        
        return cinematic_effect
    
    def _create_color_grading_effect(self, **kwargs):
        """Create a color grading effect."""
        def color_grading_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Apply warm color grading
            frame = frame.astype(np.float32)
            frame[:, :, 0] *= 1.1  # Increase blue slightly
            frame[:, :, 2] *= 1.2  # Increase red more
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            return frame
        
        return color_grading_effect
    
    def _create_high_contrast_effect(self, **kwargs):
        """Create a high contrast effect."""
        def high_contrast_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Convert to grayscale for contrast calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Blend with original based on intensity
            blended = cv2.addWeighted(frame, 1 - intensity, gray_3channel, intensity, 0)
            
            return blended
        
        return high_contrast_effect
    
    def _create_glitch_effect(self, **kwargs):
        """Create a glitch effect."""
        def glitch_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Create random glitch artifacts
            if np.random.random() < intensity * 0.3:
                # Random horizontal shift
                shift = int(intensity * 20)
                frame = np.roll(frame, shift, axis=1)
            
            if np.random.random() < intensity * 0.2:
                # Random color channel shift
                frame = np.roll(frame, int(intensity * 10), axis=2)
            
            return frame
        
        return glitch_effect
    
    def _create_vintage_effect(self, **kwargs):
        """Create a vintage effect."""
        def vintage_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Apply sepia tone
            frame = frame.astype(np.float32)
            frame[:, :, 0] *= 0.393  # Blue
            frame[:, :, 1] *= 0.769  # Green
            frame[:, :, 2] *= 0.189  # Red
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            return frame
        
        return vintage_effect
    
    def _create_neon_effect(self, **kwargs):
        """Create a neon effect."""
        def neon_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Create neon glow
            glow = cv2.GaussianBlur(frame, (15, 15), 0)
            glow = cv2.addWeighted(glow, intensity, frame, 1 - intensity, 0)
            
            return glow
        
        return neon_effect
    
    def _create_dramatic_effect(self, **kwargs):
        """Create a dramatic effect."""
        def dramatic_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Apply dramatic lighting
            frame = cv2.convertScaleAbs(frame, alpha=1 + intensity * 0.5, beta=-intensity * 30)
            
            return frame
        
        return dramatic_effect
    
    def _create_fast_pace_effect(self, **kwargs):
        """Create a fast pace effect."""
        def fast_pace_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Apply sharpening for fast pace
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(frame, -1, kernel)
            
            # Blend with original
            result = cv2.addWeighted(frame, 1 - intensity, sharpened, intensity, 0)
            
            return result
        
        return fast_pace_effect
    
    def _create_slow_motion_effect(self, **kwargs):
        """Create a slow motion effect."""
        def slow_motion_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Apply slight blur for slow motion feel
            blurred = cv2.GaussianBlur(frame, (5, 5), intensity * 2)
            
            # Blend with original
            result = cv2.addWeighted(frame, 1 - intensity, blurred, intensity, 0)
            
            return result
        
        return slow_motion_effect
    
    def _create_scene_transition_effect(self, **kwargs):
        """Create a scene transition effect."""
        def scene_transition_effect(frame, intensity, time):
            import cv2
            import numpy as np
            
            # Create fade effect
            fade = int(intensity * 255)
            frame = cv2.addWeighted(frame, 1 - intensity, np.zeros_like(frame), intensity, fade)
            
            return frame
        
        return scene_transition_effect


# Global instance
_dynamic_effects_engine: Optional[DynamicEffectsEngine] = None


def get_dynamic_effects_engine() -> DynamicEffectsEngine:
    """Get the global dynamic effects engine instance."""
    global _dynamic_effects_engine
    if _dynamic_effects_engine is None:
        _dynamic_effects_engine = DynamicEffectsEngine()
    return _dynamic_effects_engine 