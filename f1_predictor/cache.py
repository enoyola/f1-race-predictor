"""
Data caching module for F1 Predictor.

Provides local caching of API responses with TTL validation to minimize
API calls and improve performance.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class DataCache:
    """
    Local cache for F1 data with TTL support.

    Stores data as JSON files with timestamp metadata for expiration validation.
    """

    def __init__(self, cache_dir: str = ".f1_cache"):
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Failed to create cache directory '{self.cache_dir}': {e}. Caching disabled.")

    def _get_cache_path(self, key: str) -> Path:
        safe_key = key.replace('/', '_').replace(':', '_')
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str, ignore_ttl: bool = False) -> Optional[Any]:
        """
        Retrieve cached data if valid.

        Args:
            key: Cache key identifier
            ignore_ttl: If True, return data even if expired (for fallback scenarios)

        Returns:
            Cached data if valid, None if not found, expired, or corrupted
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)

            if 'data' not in cache_entry or 'expires_at' not in cache_entry:
                return None

            if not ignore_ttl:
                expires_at = datetime.fromisoformat(cache_entry['expires_at'])
                if datetime.now() > expires_at:
                    return None

            return cache_entry['data']

        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning(f"Corrupted cache file for key '{key}': {e}")
            try:
                if cache_path.exists():
                    cache_path.unlink()
            except OSError:
                pass
            return None

    def set(self, key: str, data: Any, ttl: int = 3600) -> None:
        """Store data in cache with an expiration timestamp (ttl in seconds)."""
        cache_path = self._get_cache_path(key)
        expires_at = datetime.now() + timedelta(seconds=ttl)
        cache_entry = {
            'data': data,
            'cached_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat(),
            'ttl': ttl
        }
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=1)
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to write cache for key '{key}': {e}")

    def is_valid(self, key: str) -> bool:
        """Check if cached data exists and is still valid."""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return False
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            if 'expires_at' not in cache_entry:
                return False
            return datetime.now() <= datetime.fromisoformat(cache_entry['expires_at'])
        except (json.JSONDecodeError, ValueError, OSError):
            return False

    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Remove cache files. With a prefix, only keys starting with it are removed.

        Returns:
            Number of files removed
        """
        if not self.cache_dir.exists():
            return 0
        removed = 0
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                if prefix and not cache_file.stem.startswith(prefix):
                    continue
                cache_file.unlink()
                removed += 1
        except OSError as e:
            logger.warning(f"Failed to clear cache: {e}")
        return removed
