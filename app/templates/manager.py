"""
Template management system for video editing automation.

This module provides template creation, storage, retrieval, and management
for user-defined video editing rules and configurations.
"""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime

import boto3
import json
from app.config.settings import get_settings

from app.models.schemas import EditingTemplate, TemplateType, VideoFormat, QualityPreset

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Manages video editing templates and their configurations.
    
    Provides CRUD operations for templates and template-based
    editing rule management.
    """
    
    def __init__(self):
        """Initialize the template manager."""
        self.templates: Dict[UUID, EditingTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default editing templates."""
        default_templates = [
            EditingTemplate(
                template_id=uuid4(),
                name="Beat Match",
                template_type=TemplateType.BEAT_MATCH,
                description="Synchronizes cuts with detected beats for music videos",
                transition_duration=0.3,
                cut_sensitivity=0.8,
                beat_sync_threshold=0.05,
                effects=["fade_in", "fade_out"],
                output_format=VideoFormat.MP4,
                quality_preset=QualityPreset.HIGH
            ),
            EditingTemplate(
                template_id=uuid4(),
                name="Cinematic",
                template_type=TemplateType.CINEMATIC,
                description="Creates cinematic cuts with longer shots and smooth transitions",
                transition_duration=1.0,
                cut_sensitivity=0.6,
                beat_sync_threshold=0.2,
                effects=["crossfade", "color_grade"],
                color_grading={
                    "contrast": 1.2,
                    "saturation": 0.9,
                    "brightness": 1.1
                },
                audio_fade_in=0.5,
                audio_fade_out=0.5,
                output_format=VideoFormat.MP4,
                quality_preset=QualityPreset.HIGH
            ),
            EditingTemplate(
                template_id=uuid4(),
                name="Fast Paced",
                template_type=TemplateType.FAST_PACED,
                description="Creates fast-paced edits with quick cuts and transitions",
                transition_duration=0.2,
                cut_sensitivity=0.9,
                beat_sync_threshold=0.03,
                effects=["quick_cut", "zoom_transition"],
                output_format=VideoFormat.MP4,
                quality_preset=QualityPreset.MEDIUM
            ),
            EditingTemplate(
                template_id=uuid4(),
                name="Slow Motion",
                template_type=TemplateType.SLOW_MOTION,
                description="Emphasizes slow motion effects with dramatic timing",
                transition_duration=1.5,
                cut_sensitivity=0.4,
                beat_sync_threshold=0.3,
                effects=["slow_motion", "dramatic_fade"],
                output_format=VideoFormat.MP4,
                quality_preset=QualityPreset.ULTRA
            ),
            EditingTemplate(
                template_id=uuid4(),
                name="Transition Heavy",
                template_type=TemplateType.TRANSITION_HEAVY,
                description="Uses many creative transitions between scenes",
                transition_duration=0.8,
                cut_sensitivity=0.7,
                beat_sync_threshold=0.15,
                effects=["slide_transition", "zoom_transition", "fade_transition"],
                output_format=VideoFormat.MP4,
                quality_preset=QualityPreset.HIGH
            ),
            EditingTemplate(
                template_id=uuid4(),
                name="Minimal",
                template_type=TemplateType.MINIMAL,
                description="Clean, minimal editing with subtle transitions",
                transition_duration=0.4,
                cut_sensitivity=0.5,
                beat_sync_threshold=0.25,
                effects=["simple_fade"],
                output_format=VideoFormat.MP4,
                quality_preset=QualityPreset.HIGH
            )
        ]
        
        for template in default_templates:
            self.templates[template.template_id] = template
        
        logger.info(f"Initialized {len(default_templates)} default templates")
    
    async def create_template(self, template: EditingTemplate) -> EditingTemplate:
        """
        Create a new editing template.
        
        Args:
            template: Template configuration
            
        Returns:
            EditingTemplate: Created template with generated ID
        """
        try:
            # Generate new ID if not provided
            if not template.template_id:
                template.template_id = uuid4()
            
            # Set timestamps
            template.created_at = datetime.utcnow()
            template.updated_at = datetime.utcnow()
            
            # Store template
            self.templates[template.template_id] = template
            
            # Persist to S3
            settings = get_settings()
            s3_kwargs = {
                'aws_access_key_id': settings.aws_access_key_id,
                'aws_secret_access_key': settings.aws_secret_access_key,
                'region_name': settings.aws_region
            }
            if settings.s3_endpoint_url:
                s3_kwargs['endpoint_url'] = settings.s3_endpoint_url
            s3 = boto3.client('s3', **s3_kwargs)
            s3_key = f"templates/{template.template_id}.json"
            s3.put_object(
                Bucket=settings.s3_bucket_name,
                Key=s3_key,
                Body=json.dumps(template.model_dump(), default=str),
                ContentType="application/json"
            )
            logger.info(f"Created template: {template.name} ({template.template_id}) and uploaded to S3")
            return template
            
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise
    
    async def get_template(self, template_id: UUID) -> Optional[EditingTemplate]:
        """
        Retrieve a template by ID.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Optional[EditingTemplate]: Template if found, None otherwise
        """
        # Try in-memory first
        template = self.templates.get(template_id)
        if template:
            return template
        # Try S3
        try:
            settings = get_settings()
            s3_kwargs = {
                'aws_access_key_id': settings.aws_access_key_id,
                'aws_secret_access_key': settings.aws_secret_access_key,
                'region_name': settings.aws_region
            }
            if settings.s3_endpoint_url:
                s3_kwargs['endpoint_url'] = settings.s3_endpoint_url
            s3 = boto3.client('s3', **s3_kwargs)
            s3_key = f"templates/{template_id}.json"
            response = s3.get_object(Bucket=settings.s3_bucket_name, Key=s3_key)
            template_data = json.loads(response['Body'].read())
            template = EditingTemplate(**template_data)
            # Optionally cache in memory
            self.templates[template_id] = template
            logger.info(f"Loaded template {template_id} from S3")
            return template
        except Exception as e:
            logger.error(f"Template {template_id} not found in memory or S3: {e}")
            return None
    
    async def list_templates(
        self,
        template_type: Optional[TemplateType] = None
    ) -> List[EditingTemplate]:
        """
        List available templates with optional filtering.
        
        Args:
            template_type: Optional template type filter
            
        Returns:
            List[EditingTemplate]: List of matching templates
        """
        templates = list(self.templates.values())
        
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
        
        # Sort by name
        templates.sort(key=lambda x: x.name)
        return templates
    
    async def update_template(
        self,
        template_id: UUID,
        updated_template: EditingTemplate
    ) -> Optional[EditingTemplate]:
        """
        Update an existing template.
        
        Args:
            template_id: Template identifier
            updated_template: Updated template configuration
            
        Returns:
            Optional[EditingTemplate]: Updated template if found, None otherwise
        """
        try:
            if template_id not in self.templates:
                return None
            
            # Preserve original ID and timestamps
            updated_template.template_id = template_id
            updated_template.created_at = self.templates[template_id].created_at
            updated_template.updated_at = datetime.utcnow()
            
            # Update template
            self.templates[template_id] = updated_template
            
            logger.info(f"Updated template: {updated_template.name} ({template_id})")
            return updated_template
            
        except Exception as e:
            logger.error(f"Failed to update template {template_id}: {e}")
            raise
    
    async def delete_template(self, template_id: UUID) -> bool:
        """
        Delete a template.
        
        Args:
            template_id: Template identifier
            
        Returns:
            bool: True if template was deleted, False if not found
        """
        try:
            if template_id in self.templates:
                template_name = self.templates[template_id].name
                del self.templates[template_id]
                logger.info(f"Deleted template: {template_name} ({template_id})")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            raise
    
    async def get_template_by_type(self, template_type: TemplateType) -> Optional[EditingTemplate]:
        """
        Get the default template for a specific type.
        
        Args:
            template_type: Template type
            
        Returns:
            Optional[EditingTemplate]: Default template for the type
        """
        templates = await self.list_templates(template_type=template_type)
        return templates[0] if templates else None
    
    async def get_default_template(self, style: str = "cinematic") -> Optional[EditingTemplate]:
        """
        Get a default template for a given style (template_type as string).
        """
        from app.models.schemas import TemplateType
        # Try to match style to TemplateType
        try:
            template_type = TemplateType(style.upper())
        except Exception:
            # Fallback: try to match by name
            template_type = None
            for t in TemplateType:
                if t.value == style.lower() or t.name.lower() == style.lower():
                    template_type = t
                    break
        if template_type:
            template = await self.get_template_by_type(template_type)
            if template:
                return template
        # Fallback: return any available template
        templates = await self.list_templates()
        return templates[0] if templates else None
    
    async def validate_template(self, template: EditingTemplate) -> bool:
        """
        Validate template configuration.
        
        Args:
            template: Template to validate
            
        Returns:
            bool: True if template is valid
        """
        try:
            # Check required fields
            if not template.name or not template.description:
                return False
            
            # Validate numeric ranges
            if not (0.1 <= template.transition_duration <= 5.0):
                return False
            
            if not (0.1 <= template.cut_sensitivity <= 1.0):
                return False
            
            if not (0.01 <= template.beat_sync_threshold <= 1.0):
                return False
            
            if not (0.0 <= template.audio_fade_in <= 5.0):
                return False
            
            if not (0.0 <= template.audio_fade_out <= 5.0):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False
    
    async def get_template_config(self, template_id: UUID) -> Dict[str, Any]:
        """
        Get template configuration as a dictionary.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Dict[str, Any]: Template configuration
        """
        template = await self.get_template(template_id)
        if not template:
            return {}
        
        return {
            'template_id': str(template.template_id),
            'name': template.name,
            'template_type': template.template_type.value,
            'transition_duration': template.transition_duration,
            'cut_sensitivity': template.cut_sensitivity,
            'beat_sync_threshold': template.beat_sync_threshold,
            'effects': template.effects,
            'color_grading': template.color_grading,
            'audio_fade_in': template.audio_fade_in,
            'audio_fade_out': template.audio_fade_out,
            'output_format': template.output_format.value,
            'quality_preset': template.quality_preset.value
        }


# Global template manager instance
_template_manager: Optional[TemplateManager] = None


async def initialize_template_manager() -> TemplateManager:
    """
    Initialize the global template manager.
    
    Returns:
        TemplateManager: Initialized template manager
    """
    global _template_manager
    
    _template_manager = TemplateManager()
    logger.info("Template manager initialized successfully")
    return _template_manager


def get_template_manager() -> TemplateManager:
    """
    Get the global template manager instance.
    
    Returns:
        TemplateManager: Template manager instance
        
    Raises:
        RuntimeError: If template manager is not initialized
    """
    if _template_manager is None:
        raise RuntimeError("Template manager not initialized. Call initialize_template_manager first.")
    
    return _template_manager 