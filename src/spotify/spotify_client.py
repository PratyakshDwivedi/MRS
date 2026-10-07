"""Optional Spotify enrichment.

The recommender works fully offline from the local clustered dataset. This
client is an *optional* enhancement: when valid Spotify API credentials are
present in the environment, it can fetch live metadata (album art, preview
URLs, canonical links) for recommended tracks.

Credentials are read from environment variables and never hard-coded:

    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET

If credentials are missing or the API errors, methods degrade gracefully and
return ``None`` instead of raising, so the app keeps working.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

try:  # spotipy is optional at runtime
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    _SPOTIPY_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _SPOTIPY_AVAILABLE = False


class SpotifyClient:
    """Thin wrapper around spotipy using the Client Credentials flow."""

    def __init__(self) -> None:
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        """True when spotipy is installed and credentials are configured."""
        return (
            _SPOTIPY_AVAILABLE
            and bool(os.getenv("SPOTIFY_CLIENT_ID"))
            and bool(os.getenv("SPOTIFY_CLIENT_SECRET"))
        )

    def _get_client(self) -> Any | None:
        if not self.available:
            return None
        if self._client is None:
            auth = SpotifyClientCredentials(
                client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            )
            self._client = spotipy.Spotify(client_credentials_manager=auth)
        return self._client

    def enrich_track(self, track_id: str) -> dict[str, Any] | None:
        """Return live metadata for a track id, or None on any failure."""
        client = self._get_client()
        if client is None or not track_id:
            return None
        try:
            track = client.track(track_id)
        except Exception:  # pragma: no cover - network/credential errors
            return None
        images = track.get("album", {}).get("images", [])
        return {
            "name": track.get("name"),
            "preview_url": track.get("preview_url"),
            "external_url": track.get("external_urls", {}).get("spotify"),
            "album_image": images[0]["url"] if images else None,
        }


@lru_cache(maxsize=1)
def get_spotify_client() -> SpotifyClient:
    """Return a process-wide Spotify client instance."""
    return SpotifyClient()
