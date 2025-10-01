"""
Music Library Manager

This module provides management capabilities for the music library,
including bulk operations, library maintenance, and metadata management.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid
from datetime import datetime

from app.music.library import MusicLibrary, MusicTrack, MusicGenre, MusicMood, get_music_library

logger = logging.getLogger(__name__)


class MusicLibraryManager:
    """
    Music library manager for bulk operations and maintenance.
    
    Provides capabilities for importing music, managing metadata,
    and maintaining library integrity.
    """
    
    def __init__(self, music_library: Optional[MusicLibrary] = None):
        """Initialize the music library manager."""
        self.music_library = music_library or get_music_library()
        self.music_directory = self.music_library.music_directory
        
        logger.info("Music library manager initialized")
    
    def import_music_directory(
        self,
        source_directory: str,
        auto_detect_metadata: bool = True,
        default_genre: MusicGenre = MusicGenre.CINEMATIC,
        default_mood: MusicMood = MusicMood.CALM
    ) -> Dict[str, Any]:
        """
        Import music files from a directory.
        
        Args:
            source_directory: Directory containing music files
            auto_detect_metadata: Whether to auto-detect metadata
            default_genre: Default genre for imported tracks
            default_mood: Default mood for imported tracks
            
        Returns:
            Import results summary
        """
        source_path = Path(source_directory)
        if not source_path.exists():
            raise ValueError(f"Source directory does not exist: {source_directory}")
        
        imported_tracks = []
        failed_imports = []
        
        # Supported audio formats
        audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
        
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(source_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(audio_files)} audio files to import")
        
        for audio_file in audio_files:
            try:
                # Get relative path for storage
                relative_path = audio_file.relative_to(source_path)
                target_path = self.music_directory / relative_path
                
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file if not already in music directory
                if not target_path.exists():
                    import shutil
                    shutil.copy2(audio_file, target_path)
                
                # Create music track
                track = self._create_track_from_file(
                    file_path=str(target_path.relative_to(self.music_directory)),
                    auto_detect_metadata=auto_detect_metadata,
                    default_genre=default_genre,
                    default_mood=default_mood
                )
                
                # Add to library
                success = self.music_library.add_track(track)
                if success:
                    imported_tracks.append(track)
                    logger.info(f"Imported: {track.title}")
                else:
                    failed_imports.append(str(audio_file))
                    
            except Exception as e:
                logger.error(f"Failed to import {audio_file}: {e}")
                failed_imports.append(str(audio_file))
        
        return {
            "imported_count": len(imported_tracks),
            "failed_count": len(failed_imports),
            "imported_tracks": [track.to_dict() for track in imported_tracks],
            "failed_files": failed_imports
        }
    
    def _create_track_from_file(
        self,
        file_path: str,
        auto_detect_metadata: bool = True,
        default_genre: MusicGenre = MusicGenre.CINEMATIC,
        default_mood: MusicMood = MusicMood.CALM
    ) -> MusicTrack:
        """Create a music track from a file."""
        full_path = self.music_directory / file_path
        
        # Get basic file info
        file_name = Path(file_path).stem
        file_size = full_path.stat().st_size
        
        # Auto-detect metadata if enabled
        if auto_detect_metadata:
            metadata = self._detect_audio_metadata(str(full_path))
        else:
            metadata = {}
        
        # Create track
        track = MusicTrack(
            id=str(uuid.uuid4()),
            title=metadata.get("title", file_name),
            artist=metadata.get("artist", "Unknown Artist"),
            genre=MusicGenre(metadata.get("genre", default_genre.value)),
            mood=MusicMood(metadata.get("mood", default_mood.value)),
            duration=metadata.get("duration", 0.0),
            file_path=file_path,
            bpm=metadata.get("bpm"),
            key=metadata.get("key"),
            tags=metadata.get("tags", []),
            description=metadata.get("description", ""),
            is_royalty_free=metadata.get("is_royalty_free", True),
            license_type=metadata.get("license_type", "royalty_free")
        )
        
        return track
    
    def _detect_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Detect audio metadata using FFmpeg."""
        try:
            import subprocess
            
            # Get basic audio info
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to detect metadata for {file_path}")
                return {}
            
            data = json.loads(result.stdout)
            format_info = data.get("format", {})
            stream_info = data.get("streams", [{}])[0]
            
            # Extract metadata
            metadata = {
                "title": format_info.get("tags", {}).get("title", ""),
                "artist": format_info.get("tags", {}).get("artist", ""),
                "duration": float(format_info.get("duration", 0)),
                "bpm": self._extract_bpm(format_info.get("tags", {})),
                "key": self._extract_key(format_info.get("tags", {})),
                "tags": self._extract_tags(format_info.get("tags", {})),
                "is_royalty_free": True,  # Default assumption
                "license_type": "royalty_free"
            }
            
            # Clean up empty values
            metadata = {k: v for k, v in metadata.items() if v}
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to detect metadata for {file_path}: {e}")
            return {}
    
    def _extract_bpm(self, tags: Dict[str, str]) -> Optional[int]:
        """Extract BPM from tags."""
        bpm_keys = ["BPM", "bpm", "TBPM", "Tempo"]
        for key in bpm_keys:
            if key in tags:
                try:
                    return int(float(tags[key]))
                except (ValueError, TypeError):
                    continue
        return None
    
    def _extract_key(self, tags: Dict[str, str]) -> Optional[str]:
        """Extract musical key from tags."""
        key_keys = ["KEY", "key", "TKEY", "Musical Key"]
        for key in key_keys:
            if key in tags:
                return tags[key]
        return None
    
    def _extract_tags(self, tags: Dict[str, str]) -> List[str]:
        """Extract tags from metadata."""
        tag_keys = ["genre", "GENRE", "style", "STYLE", "mood", "MOOD"]
        extracted_tags = []
        
        for key in tag_keys:
            if key in tags:
                # Split comma-separated values
                values = [v.strip() for v in tags[key].split(",")]
                extracted_tags.extend(values)
        
        return list(set(extracted_tags))  # Remove duplicates
    
    def export_library(self, output_file: str) -> bool:
        """
        Export the music library to a file.
        
        Args:
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                "library_info": {
                    "export_date": datetime.now().isoformat(),
                    "total_tracks": len(self.music_library.tracks),
                    "version": "1.0"
                },
                "tracks": [track.to_dict() for track in self.music_library.tracks.values()]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Library exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export library: {e}")
            return False
    
    def import_library(self, import_file: str) -> Dict[str, Any]:
        """
        Import a music library from a file.
        
        Args:
            import_file: Import file path
            
        Returns:
            Import results summary
        """
        try:
            with open(import_file, 'r') as f:
                data = json.load(f)
            
            imported_count = 0
            failed_count = 0
            
            for track_data in data.get("tracks", []):
                try:
                    track = MusicTrack.from_dict(track_data)
                    success = self.music_library.add_track(track)
                    if success:
                        imported_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to import track: {e}")
                    failed_count += 1
            
            return {
                "imported_count": imported_count,
                "failed_count": failed_count,
                "total_tracks": len(data.get("tracks", []))
            }
            
        except Exception as e:
            logger.error(f"Failed to import library: {e}")
            return {"imported_count": 0, "failed_count": 0, "error": str(e)}
    
    def cleanup_library(self) -> Dict[str, Any]:
        """
        Clean up the music library by removing orphaned files and fixing metadata.
        
        Returns:
            Cleanup results summary
        """
        cleanup_results = {
            "orphaned_files": [],
            "missing_files": [],
            "fixed_metadata": [],
            "removed_tracks": []
        }
        
        # Check for orphaned files (files not in library)
        for file_path in self.music_directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}:
                relative_path = str(file_path.relative_to(self.music_directory))
                if not any(track.file_path == relative_path for track in self.music_library.tracks.values()):
                    cleanup_results["orphaned_files"].append(relative_path)
        
        # Check for missing files (tracks referencing non-existent files)
        tracks_to_remove = []
        for track_id, track in self.music_library.tracks.items():
            full_path = self.music_directory / track.file_path
            if not full_path.exists():
                cleanup_results["missing_files"].append(track.file_path)
                tracks_to_remove.append(track_id)
        
        # Remove tracks with missing files
        for track_id in tracks_to_remove:
            self.music_library.remove_track(track_id)
            cleanup_results["removed_tracks"].append(track_id)
        
        # Fix metadata issues
        for track in self.music_library.tracks.values():
            fixed = False
            
            # Fix empty titles
            if not track.title or track.title == "Unknown":
                track.title = Path(track.file_path).stem
                fixed = True
            
            # Fix empty artists
            if not track.artist or track.artist == "Unknown Artist":
                track.artist = "Unknown Artist"
                fixed = True
            
            if fixed:
                cleanup_results["fixed_metadata"].append(track.id)
        
        # Save changes
        if cleanup_results["fixed_metadata"] or cleanup_results["removed_tracks"]:
            self.music_library._save_library()
        
        return cleanup_results
    
    def get_library_health(self) -> Dict[str, Any]:
        """
        Get library health information.
        
        Returns:
            Library health summary
        """
        total_tracks = len(self.music_library.tracks)
        missing_files = 0
        invalid_metadata = 0
        
        for track in self.music_library.tracks.values():
            # Check if file exists
            full_path = self.music_directory / track.file_path
            if not full_path.exists():
                missing_files += 1
            
            # Check metadata validity
            if not track.title or not track.artist or track.duration <= 0:
                invalid_metadata += 1
        
        health_score = max(0, 100 - (missing_files + invalid_metadata) * 10)
        
        return {
            "total_tracks": total_tracks,
            "missing_files": missing_files,
            "invalid_metadata": invalid_metadata,
            "health_score": health_score,
            "status": "healthy" if health_score > 80 else "needs_attention" if health_score > 60 else "unhealthy"
        }
    
    def generate_library_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive library report.
        
        Returns:
            Library report
        """
        stats = self.music_library.get_library_stats()
        health = self.get_library_health()
        
        # Genre distribution
        genre_distribution = {}
        for track in self.music_library.tracks.values():
            genre = track.genre.value
            genre_distribution[genre] = genre_distribution.get(genre, 0) + 1
        
        # Mood distribution
        mood_distribution = {}
        for track in self.music_library.tracks.values():
            mood = track.mood.value
            mood_distribution[mood] = mood_distribution.get(mood, 0) + 1
        
        # Duration analysis
        durations = [track.duration for track in self.music_library.tracks.values()]
        duration_stats = {
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
            "avg": sum(durations) / len(durations) if durations else 0
        }
        
        return {
            "library_stats": stats,
            "health": health,
            "genre_distribution": genre_distribution,
            "mood_distribution": mood_distribution,
            "duration_stats": duration_stats,
            "generated_at": datetime.now().isoformat()
        }


# Global music library manager instance
_music_manager: Optional[MusicLibraryManager] = None


def get_music_manager() -> MusicLibraryManager:
    """Get the global music library manager instance."""
    global _music_manager
    if _music_manager is None:
        _music_manager = MusicLibraryManager()
    return _music_manager

