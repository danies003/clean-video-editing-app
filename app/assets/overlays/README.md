# Video Overlays Directory

This directory stores overlay assets for video editing:

## Structure

```
app/assets/overlays/
├── graphics/           # Static graphics and logos
├── animations/         # Animated overlays (GIFs, MP4s)
├── borders/           # Video borders and frames
├── watermarks/        # Watermark templates
├── transitions/       # Transition overlay effects
└── templates/         # Overlay templates
```

## Usage

Overlays can be accessed programmatically through the asset manager:

```python
from app.assets.manager import AssetManager

asset_manager = AssetManager()
overlay_path = asset_manager.get_overlay("graphics/logo.png")
```

## Supported Formats

- **Images**: PNG, JPG, SVG
- **Videos**: MP4, MOV, AVI
- **Animations**: GIF, APNG
- **Vectors**: SVG, AI
