"""
Music Library API Routes

This module provides API endpoints for the background music library system.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from app.music.library import (
    MusicLibrary, MusicTrack, MusicGenre, MusicMood,
    get_music_library, create_music_track
)

router = APIRouter(prefix="/api/v1/music", tags=["music"])


class MusicTrackResponse(BaseModel):
    """Response model for music track."""
    id: str
    title: str
    artist: str
    genre: str
    mood: str
    duration: float
    file_path: str
    bpm: Optional[int] = None
    key: Optional[str] = None
    tags: List[str] = []
    description: str = ""
    is_royalty_free: bool = True
    license_type: str = "royalty_free"


class MusicSearchRequest(BaseModel):
    """Request model for music search."""
    genre: Optional[str] = None
    mood: Optional[str] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    bpm_range: Optional[List[int]] = None
    tags: Optional[List[str]] = None
    is_royalty_free: Optional[bool] = None


class VideoMusicRequest(BaseModel):
    """Request model for video music selection."""
    video_duration: float
    content_mood: Optional[str] = None
    preferred_genre: Optional[str] = None


@router.get("/tracks", response_model=List[MusicTrackResponse])
async def get_all_tracks(
    library: MusicLibrary = Depends(get_music_library)
):
    """Get all tracks in the music library."""
    try:
        tracks = library.get_all_tracks()
        return [MusicTrackResponse(**track.to_dict()) for track in tracks]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tracks: {str(e)}")


@router.get("/tracks/{track_id}", response_model=MusicTrackResponse)
async def get_track(
    track_id: str,
    library: MusicLibrary = Depends(get_music_library)
):
    """Get a specific track by ID."""
    try:
        track = library.get_track(track_id)
        if not track:
            raise HTTPException(status_code=404, detail="Track not found")
        return MusicTrackResponse(**track.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get track: {str(e)}")


@router.post("/search", response_model=List[MusicTrackResponse])
async def search_tracks(
    request: MusicSearchRequest,
    library: MusicLibrary = Depends(get_music_library)
):
    """Search for tracks based on criteria."""
    try:
        # Convert string enums to enum objects
        genre = MusicGenre(request.genre) if request.genre else None
        mood = MusicMood(request.mood) if request.mood else None
        
        # Convert BPM range
        bpm_range = None
        if request.bpm_range and len(request.bpm_range) == 2:
            bpm_range = (request.bpm_range[0], request.bpm_range[1])
        
        tracks = library.search_tracks(
            genre=genre,
            mood=mood,
            min_duration=request.min_duration,
            max_duration=request.max_duration,
            bpm_range=bpm_range,
            tags=request.tags,
            is_royalty_free=request.is_royalty_free
        )
        
        return [MusicTrackResponse(**track.to_dict()) for track in tracks]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/genres", response_model=List[str])
async def get_genres():
    """Get all available music genres."""
    return [genre.value for genre in MusicGenre]


@router.get("/moods", response_model=List[str])
async def get_moods():
    """Get all available music moods."""
    return [mood.value for mood in MusicMood]


@router.get("/tracks/genre/{genre}", response_model=List[MusicTrackResponse])
async def get_tracks_by_genre(
    genre: str,
    library: MusicLibrary = Depends(get_music_library)
):
    """Get tracks by genre."""
    try:
        genre_enum = MusicGenre(genre)
        tracks = library.get_tracks_by_genre(genre_enum)
        return [MusicTrackResponse(**track.to_dict()) for track in tracks]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid genre")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tracks: {str(e)}")


@router.get("/tracks/mood/{mood}", response_model=List[MusicTrackResponse])
async def get_tracks_by_mood(
    mood: str,
    library: MusicLibrary = Depends(get_music_library)
):
    """Get tracks by mood."""
    try:
        mood_enum = MusicMood(mood)
        tracks = library.get_tracks_by_mood(mood_enum)
        return [MusicTrackResponse(**track.to_dict()) for track in tracks]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid mood")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tracks: {str(e)}")


@router.post("/video-suggestions", response_model=List[MusicTrackResponse])
async def get_video_music_suggestions(
    request: VideoMusicRequest,
    library: MusicLibrary = Depends(get_music_library)
):
    """Get music suggestions for a video based on duration and content."""
    try:
        # Convert string enums to enum objects
        content_mood = MusicMood(request.content_mood) if request.content_mood else None
        preferred_genre = MusicGenre(request.preferred_genre) if request.preferred_genre else None
        
        tracks = library.get_tracks_for_video(
            video_duration=request.video_duration,
            content_mood=content_mood,
            preferred_genre=preferred_genre
        )
        
        return [MusicTrackResponse(**track.to_dict()) for track in tracks]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/random", response_model=MusicTrackResponse)
async def get_random_track(
    genre: Optional[str] = Query(None, description="Filter by genre"),
    mood: Optional[str] = Query(None, description="Filter by mood"),
    library: MusicLibrary = Depends(get_music_library)
):
    """Get a random track matching the given filters."""
    try:
        # Convert string enums to enum objects
        genre_enum = MusicGenre(genre) if genre else None
        mood_enum = MusicMood(mood) if mood else None
        
        track = library.get_random_track(genre=genre_enum, mood=mood_enum)
        if not track:
            raise HTTPException(status_code=404, detail="No tracks found matching criteria")
        
        return MusicTrackResponse(**track.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get random track: {str(e)}")


@router.get("/stats", response_model=Dict[str, Any])
async def get_library_stats(
    library: MusicLibrary = Depends(get_music_library)
):
    """Get music library statistics."""
    try:
        return library.get_library_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/tracks", response_model=MusicTrackResponse)
async def add_track(
    title: str,
    artist: str,
    genre: str,
    mood: str,
    duration: float,
    file_path: str,
    bpm: Optional[int] = None,
    key: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: str = "",
    is_royalty_free: bool = True,
    license_type: str = "royalty_free",
    library: MusicLibrary = Depends(get_music_library)
):
    """Add a new track to the library."""
    try:
        # Convert string enums to enum objects
        genre_enum = MusicGenre(genre)
        mood_enum = MusicMood(mood)
        
        track = create_music_track(
            title=title,
            artist=artist,
            genre=genre_enum,
            mood=mood_enum,
            duration=duration,
            file_path=file_path,
            bpm=bpm,
            key=key,
            tags=tags or [],
            description=description,
            is_royalty_free=is_royalty_free,
            license_type=license_type
        )
        
        success = library.add_track(track)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add track")
        
        return MusicTrackResponse(**track.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add track: {str(e)}")


@router.delete("/tracks/{track_id}")
async def remove_track(
    track_id: str,
    library: MusicLibrary = Depends(get_music_library)
):
    """Remove a track from the library."""
    try:
        success = library.remove_track(track_id)
        if not success:
            raise HTTPException(status_code=404, detail="Track not found")
        return {"message": "Track removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove track: {str(e)}")

