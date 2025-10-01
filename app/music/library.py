"""
Background Music Library System

This module provides a comprehensive music library system for the video editing tool,
including music categorization, selection, and integration with the editing pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class MusicGenre(Enum):
    """Music genres available in the library."""
    # Upbeat & Energetic
    ELECTRONIC = "electronic"
    DANCE = "dance"
    POP = "pop"
    ROCK = "rock"
    HIP_HOP = "hip_hop"
    
    # Calm & Ambient
    AMBIENT = "ambient"
    CINEMATIC = "cinematic"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    LOUNGE = "lounge"
    
    # Emotional
    DRAMATIC = "dramatic"
    ROMANTIC = "romantic"
    MELANCHOLIC = "melancholic"
    INSPIRATIONAL = "inspirational"
    
    # Corporate & Professional
    CORPORATE = "corporate"
    NEWS = "news"
    DOCUMENTARY = "documentary"
    
    # Fun & Playful
    COMEDY = "comedy"
    GAMING = "gaming"
    KIDS = "kids"
    UPBEAT = "upbeat"


class MusicMood(Enum):
    """Music moods for content matching."""
    HAPPY = "happy"
    SAD = "sad"
    EXCITING = "exciting"
    CALM = "calm"
    DRAMATIC = "dramatic"
    ROMANTIC = "romantic"
    MYSTERIOUS = "mysterious"
    HEROIC = "heroic"
    NOSTALGIC = "nostalgic"
    ENERGETIC = "energetic"


@dataclass
class MusicTrack:
    """Represents a music track in the library."""
    id: str
    title: str
    artist: str
    genre: MusicGenre
    mood: MusicMood
    duration: float  # in seconds
    file_path: str
    bpm: Optional[int] = None
    key: Optional[str] = None  # Musical key (C, D, E, etc.)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_royalty_free: bool = True
    license_type: str = "royalty_free"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre.value,
            "mood": self.mood.value,
            "duration": self.duration,
            "file_path": self.file_path,
            "bpm": self.bpm,
            "key": self.key,
            "tags": self.tags,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "is_royalty_free": self.is_royalty_free,
            "license_type": self.license_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MusicTrack':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            artist=data["artist"],
            genre=MusicGenre(data["genre"]),
            mood=MusicMood(data["mood"]),
            duration=data["duration"],
            file_path=data["file_path"],
            bpm=data.get("bpm"),
            key=data.get("key"),
            tags=data.get("tags", []),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            is_royalty_free=data.get("is_royalty_free", True),
            license_type=data.get("license_type", "royalty_free")
        )


class MusicLibrary:
    """
    Background music library system for video editing.
    
    Provides music categorization, selection, and integration capabilities.
    """
    
    def __init__(self, music_directory: str = "music_library"):
        """Initialize the music library."""
        self.music_directory = Path(music_directory)
        self.music_directory.mkdir(exist_ok=True)
        
        self.tracks: Dict[str, MusicTrack] = {}
        self.metadata_file = self.music_directory / "library_metadata.json"
        
        # Load existing library
        self._load_library()
        
        logger.info(f"Music library initialized with {len(self.tracks)} tracks")
    
    def _load_library(self):
        """Load music library from metadata file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for track_data in data.get("tracks", []):
                        track = MusicTrack.from_dict(track_data)
                        self.tracks[track.id] = track
                logger.info(f"Loaded {len(self.tracks)} tracks from library")
            except Exception as e:
                logger.error(f"Failed to load music library: {e}")
        else:
            # Create default library with sample tracks
            self._create_default_library()
    
    def _save_library(self):
        """Save music library to metadata file."""
        try:
            data = {
                "tracks": [track.to_dict() for track in self.tracks.values()],
                "last_updated": datetime.now().isoformat()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Music library saved successfully")
        except Exception as e:
            logger.error(f"Failed to save music library: {e}")
    
    def _create_default_library(self):
        """Create a default music library with sample tracks."""
        sample_tracks = [
            {
                "title": "Upbeat Electronic",
                "artist": "AI Generated",
                "genre": MusicGenre.ELECTRONIC,
                "mood": MusicMood.ENERGETIC,
                "duration": 120.0,
                "bpm": 128,
                "key": "C",
                "tags": ["electronic", "dance", "upbeat"],
                "description": "High-energy electronic track perfect for action sequences"
            },
            {
                "title": "Cinematic Drama",
                "artist": "AI Generated",
                "genre": MusicGenre.CINEMATIC,
                "mood": MusicMood.DRAMATIC,
                "duration": 180.0,
                "bpm": 90,
                "key": "Dm",
                "tags": ["cinematic", "dramatic", "orchestral"],
                "description": "Epic cinematic track for dramatic moments"
            },
            {
                "title": "Calm Ambient",
                "artist": "AI Generated",
                "genre": MusicGenre.AMBIENT,
                "mood": MusicMood.CALM,
                "duration": 300.0,
                "bpm": 60,
                "key": "Am",
                "tags": ["ambient", "calm", "relaxing"],
                "description": "Peaceful ambient track for calm scenes"
            },
            {
                "title": "Corporate Professional",
                "artist": "AI Generated",
                "genre": MusicGenre.CORPORATE,
                "mood": MusicMood.CALM,
                "duration": 150.0,
                "bpm": 100,
                "key": "C",
                "tags": ["corporate", "professional", "business"],
                "description": "Professional corporate music for business content"
            },
            {
                "title": "Romantic Piano",
                "artist": "AI Generated",
                "genre": MusicGenre.CLASSICAL,
                "mood": MusicMood.ROMANTIC,
                "duration": 240.0,
                "bpm": 70,
                "key": "F",
                "tags": ["romantic", "piano", "classical"],
                "description": "Beautiful romantic piano piece"
            }
        ]
        
        for track_data in sample_tracks:
            track = MusicTrack(
                id=str(uuid.uuid4()),
                file_path=f"sample_{track_data['title'].lower().replace(' ', '_')}.mp3",
                **track_data
            )
            self.tracks[track.id] = track
        
        self._save_library()
        logger.info("Created default music library with sample tracks")
    
    def add_track(self, track: MusicTrack) -> bool:
        """Add a new track to the library."""
        try:
            self.tracks[track.id] = track
            self._save_library()
            logger.info(f"Added track: {track.title} by {track.artist}")
            return True
        except Exception as e:
            logger.error(f"Failed to add track: {e}")
            return False
    
    def remove_track(self, track_id: str) -> bool:
        """Remove a track from the library."""
        try:
            if track_id in self.tracks:
                del self.tracks[track_id]
                self._save_library()
                logger.info(f"Removed track: {track_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove track: {e}")
            return False
    
    def get_track(self, track_id: str) -> Optional[MusicTrack]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)
    
    def search_tracks(
        self,
        genre: Optional[MusicGenre] = None,
        mood: Optional[MusicMood] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        bpm_range: Optional[Tuple[int, int]] = None,
        tags: Optional[List[str]] = None,
        is_royalty_free: Optional[bool] = None
    ) -> List[MusicTrack]:
        """Search for tracks based on criteria."""
        results = []
        
        for track in self.tracks.values():
            # Filter by genre
            if genre and track.genre != genre:
                continue
            
            # Filter by mood
            if mood and track.mood != mood:
                continue
            
            # Filter by duration
            if min_duration and track.duration < min_duration:
                continue
            if max_duration and track.duration > max_duration:
                continue
            
            # Filter by BPM
            if bpm_range and track.bpm:
                if not (bpm_range[0] <= track.bpm <= bpm_range[1]):
                    continue
            
            # Filter by tags
            if tags and not any(tag.lower() in [t.lower() for t in track.tags] for tag in tags):
                continue
            
            # Filter by royalty free
            if is_royalty_free is not None and track.is_royalty_free != is_royalty_free:
                continue
            
            results.append(track)
        
        return results
    
    def get_tracks_by_genre(self, genre: MusicGenre) -> List[MusicTrack]:
        """Get all tracks of a specific genre."""
        return self.search_tracks(genre=genre)
    
    def get_tracks_by_mood(self, mood: MusicMood) -> List[MusicTrack]:
        """Get all tracks of a specific mood."""
        return self.search_tracks(mood=mood)
    
    def get_random_track(self, **filters) -> Optional[MusicTrack]:
        """Get a random track matching the given filters."""
        import random
        results = self.search_tracks(**filters)
        return random.choice(results) if results else None
    
    def get_tracks_for_video(
        self,
        video_duration: float,
        content_mood: Optional[MusicMood] = None,
        preferred_genre: Optional[MusicGenre] = None
    ) -> List[MusicTrack]:
        """
        Get suitable tracks for a video based on duration and content.
        
        Args:
            video_duration: Duration of the video in seconds
            content_mood: Preferred mood based on video content
            preferred_genre: Preferred music genre
            
        Returns:
            List of suitable tracks
        """
        # Search for tracks that are close to the video duration
        duration_tolerance = 0.2  # 20% tolerance
        min_duration = video_duration * (1 - duration_tolerance)
        max_duration = video_duration * (1 + duration_tolerance)
        
        # Search for tracks
        tracks = self.search_tracks(
            genre=preferred_genre,
            mood=content_mood,
            min_duration=min_duration,
            max_duration=max_duration,
            is_royalty_free=True
        )
        
        # If no tracks found with duration tolerance, expand search
        if not tracks:
            tracks = self.search_tracks(
                genre=preferred_genre,
                mood=content_mood,
                is_royalty_free=True
            )
        
        # Sort by duration difference from target
        tracks.sort(key=lambda t: abs(t.duration - video_duration))
        
        return tracks[:10]  # Return top 10 matches
    
    def get_all_tracks(self) -> List[MusicTrack]:
        """Get all tracks in the library."""
        return list(self.tracks.values())
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        if not self.tracks:
            return {"total_tracks": 0}
        
        genres = {}
        moods = {}
        total_duration = 0
        
        for track in self.tracks.values():
            # Count genres
            genre = track.genre.value
            genres[genre] = genres.get(genre, 0) + 1
            
            # Count moods
            mood = track.mood.value
            moods[mood] = moods.get(mood, 0) + 1
            
            # Sum duration
            total_duration += track.duration
        
        return {
            "total_tracks": len(self.tracks),
            "total_duration": total_duration,
            "genres": genres,
            "moods": moods,
            "average_duration": total_duration / len(self.tracks) if self.tracks else 0
        }


# Global music library instance
_music_library: Optional[MusicLibrary] = None


def get_music_library() -> MusicLibrary:
    """Get the global music library instance."""
    global _music_library
    if _music_library is None:
        _music_library = MusicLibrary()
    return _music_library


def create_music_track(
    title: str,
    artist: str,
    genre: MusicGenre,
    mood: MusicMood,
    duration: float,
    file_path: str,
    **kwargs
) -> MusicTrack:
    """Create a new music track."""
    return MusicTrack(
        id=str(uuid.uuid4()),
        title=title,
        artist=artist,
        genre=genre,
        mood=mood,
        duration=duration,
        file_path=file_path,
        **kwargs
    )

