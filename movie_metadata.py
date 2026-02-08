"""
Movie metadata fetching and caching using TMDB API
Provides rich movie data including genres, directors, cast, keywords, and similar movies
"""
import os
import json
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import requests
from pathlib import Path

from config import TMDB_API_KEY, TMDB_BASE_URL, CACHE_DIR, CACHE_EXPIRY_DAYS


class TMDBClient:
    """Client for fetching movie metadata from TMDB API with caching"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDB client
        
        Args:
            api_key: TMDB API key (defaults to config value)
        """
        self.api_key = api_key or TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.cache_dir = Path(CACHE_DIR) / "tmdb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.enabled = bool(self.api_key)
        
    def is_enabled(self) -> bool:
        """Check if TMDB API is enabled"""
        return self.enabled
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Search for a movie by title and optionally year
        
        Args:
            title: Movie title to search for
            year: Optional year to narrow search
            
        Returns:
            Movie data dictionary or None if not found
        """
        if not self.enabled:
            return None
        
        # Check cache first
        cache_key = f"search_{title}_{year or 'any'}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            params = {
                "api_key": self.api_key,
                "query": title,
                "language": "en-US",
                "page": 1,
                "include_adult": False
            }
            
            if year:
                params["year"] = year
            
            response = self.session.get(
                f"{self.base_url}/search/movie",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return None
            
            # Return the first (best) match
            movie = results[0]
            
            # Cache the result
            self._save_to_cache(cache_key, movie)
            
            return movie
            
        except Exception as e:
            print(f"Error searching movie '{title}': {e}")
            return None
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get detailed information about a movie
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            Detailed movie data including genres, runtime, etc.
        """
        if not self.enabled:
            return None
        
        # Check cache
        cache_key = f"details_{movie_id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            params = {
                "api_key": self.api_key,
                "language": "en-US",
                "append_to_response": "keywords"
            }
            
            response = self.session.get(
                f"{self.base_url}/movie/{movie_id}",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            self._save_to_cache(cache_key, data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching movie details for ID {movie_id}: {e}")
            return None
    
    def get_credits(self, movie_id: int) -> Optional[Dict]:
        """
        Get cast and crew information for a movie
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            Dictionary with cast and crew lists
        """
        if not self.enabled:
            return None
        
        # Check cache
        cache_key = f"credits_{movie_id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            params = {
                "api_key": self.api_key,
                "language": "en-US"
            }
            
            response = self.session.get(
                f"{self.base_url}/movie/{movie_id}/credits",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            self._save_to_cache(cache_key, data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching credits for movie ID {movie_id}: {e}")
            return None
    
    def get_similar_movies(self, movie_id: int, limit: int = 10) -> List[Dict]:
        """
        Get movies similar to the given movie
        
        Args:
            movie_id: TMDB movie ID
            limit: Maximum number of similar movies to return
            
        Returns:
            List of similar movie dictionaries
        """
        if not self.enabled:
            return []
        
        # Check cache
        cache_key = f"similar_{movie_id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached[:limit]
        
        try:
            params = {
                "api_key": self.api_key,
                "language": "en-US",
                "page": 1
            }
            
            response = self.session.get(
                f"{self.base_url}/movie/{movie_id}/similar",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # Cache the result
            self._save_to_cache(cache_key, results)
            
            return results[:limit]
            
        except Exception as e:
            print(f"Error fetching similar movies for ID {movie_id}: {e}")
            return []
    
    def get_movie_metadata(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Get comprehensive metadata for a movie (one-stop method)
        
        Args:
            title: Movie title
            year: Optional year
            
        Returns:
            Dictionary with all metadata combined
        """
        if not self.enabled:
            return None
        
        # Search for the movie
        movie = self.search_movie(title, year)
        if not movie:
            return None
        
        movie_id = movie["id"]
        
        # Get detailed info
        details = self.get_movie_details(movie_id)
        credits = self.get_credits(movie_id)
        similar = self.get_similar_movies(movie_id, limit=5)
        
        # Combine everything
        metadata = {
            "tmdb_id": movie_id,
            "title": movie.get("title", title),
            "year": self._extract_year(movie.get("release_date")),
            "overview": movie.get("overview", ""),
            "genres": [],
            "directors": [],
            "cast": [],
            "keywords": [],
            "similar_movies": [],
            "vote_average": movie.get("vote_average", 0),
            "vote_count": movie.get("vote_count", 0),
            "popularity": movie.get("popularity", 0),
            "runtime": None,
            "budget": None
        }
        
        # Add details if available
        if details:
            metadata["genres"] = [g["name"] for g in details.get("genres", [])]
            metadata["runtime"] = details.get("runtime")
            metadata["budget"] = details.get("budget")
            
            # Extract keywords
            keywords_data = details.get("keywords", {})
            if isinstance(keywords_data, dict):
                metadata["keywords"] = [k["name"] for k in keywords_data.get("keywords", [])]
        
        # Add credits if available
        if credits:
            # Get directors
            crew = credits.get("crew", [])
            metadata["directors"] = [
                person["name"] for person in crew 
                if person.get("job") == "Director"
            ]
            
            # Get top cast (first 10)
            cast = credits.get("cast", [])
            metadata["cast"] = [
                person["name"] for person in cast[:10]
            ]
        
        # Add similar movies
        metadata["similar_movies"] = [
            {
                "title": m.get("title", ""),
                "year": self._extract_year(m.get("release_date")),
                "id": m.get("id")
            }
            for m in similar
        ]
        
        return metadata
    
    def _extract_year(self, release_date: Optional[str]) -> Optional[int]:
        """Extract year from release date string"""
        if not release_date:
            return None
        try:
            return int(release_date.split("-")[0])
        except (ValueError, IndexError):
            return None
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key"""
        # Create safe filename
        safe_key = "".join(c if c.isalnum() or c in "_-" else "_" for c in cache_key)
        return self.cache_dir / f"{safe_key}.json"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Retrieve data from cache if not expired"""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check if cache is expired
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            expiry = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)
            
            if mtime < expiry:
                # Cache expired, delete it
                cache_path.unlink()
                return None
            
            # Read cache
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error reading cache for {cache_key}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Union[Dict, List[Dict]]) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache for {cache_key}: {e}")
