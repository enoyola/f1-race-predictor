"""
Data caching module for F1 Predictor.

Provides local caching of API responses with TTL validation to minimize
API calls and improve performance.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any


class DataCache:
    """
    Local cache for F1 data with TTL support.
    
    Stores data as JSON files with timestamp metadata for expiration validation.
    """
    
    def __init__(self, cache_dir: str = ".f1_cache"):
        """
        Initialize cache with configurable directory.
        
        Args:
            cache_dir: Directory path for cache storage (default: .f1_cache)
        """
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Warning: Failed to create cache directory '{self.cache_dir}': {e}")
            print("Caching will be disabled.")
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get file path for cache key.
        
        Args:
            key: Cache key identifier
            
        Returns:
            Path object for cache file
        """
        # Sanitize key to create valid filename
        safe_key = key.replace('/', '_').replace(':', '_')
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str, ignore_ttl: bool = False) -> Optional[dict]:
        """
        Retrieve cached data if valid.
        
        Args:
            key: Cache key identifier
            ignore_ttl: If True, return data even if expired (for fallback scenarios)
            
        Returns:
            Cached data dictionary if valid, None if not found or corrupted
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            # Validate cache entry structure
            if 'data' not in cache_entry or 'expires_at' not in cache_entry:
                return None
            
            # Check if cache is still valid (unless ignoring TTL)
            if not ignore_ttl:
                expires_at = datetime.fromisoformat(cache_entry['expires_at'])
                if datetime.now() > expires_at:
                    # Cache expired, but don't remove if we might need it as fallback
                    return None
            
            return cache_entry['data']
            
        except (json.JSONDecodeError, ValueError, OSError) as e:
            # Invalid cache file, log and remove it
            print(f"Warning: Corrupted cache file for key '{key}': {e}")
            try:
                if cache_path.exists():
                    cache_path.unlink()
            except OSError:
                pass  # Ignore errors during cleanup
            return None
    
    def set(self, key: str, data: Any, ttl: int = 3600) -> None:
        """
        Store data in cache with expiration timestamp.
        
        Args:
            key: Cache key identifier
            data: Data to cache (must be JSON serializable)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
        """
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
                json.dump(cache_entry, f, indent=2)
        except (OSError, TypeError) as e:
            # Log error but don't fail - caching is optional
            print(f"Warning: Failed to write cache for key '{key}': {e}")
    
    def is_valid(self, key: str) -> bool:
        """
        Check if cached data exists and is still valid.
        
        Args:
            key: Cache key identifier
            
        Returns:
            True if cache exists and hasn't expired, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            if 'expires_at' not in cache_entry:
                return False
            
            expires_at = datetime.fromisoformat(cache_entry['expires_at'])
            return datetime.now() <= expires_at
            
        except (json.JSONDecodeError, ValueError, OSError):
            return False
    
    def clear(self) -> None:
        """
        Clear all cached data by removing all cache files.
        """
        if not self.cache_dir.exists():
            return
        
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                cache_file.unlink()
        except OSError as e:
            print(f"Warning: Failed to clear cache: {e}")
