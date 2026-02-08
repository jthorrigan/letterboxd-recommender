"""
Data processing module for Letterboxd data
Handles CSV import, cleaning, and preprocessing with TMDB enrichment
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

from movie_metadata import TMDBClient


class DataProcessor:
    """Process and clean Letterboxd data with TMDB enrichment"""
    
    def __init__(self, tmdb_client: Optional[TMDBClient] = None):
        self.ratings_df = None
        self.watchlist_df = None
        self.watched_df = None
        self.user_movies = None
        self.tmdb_client = tmdb_client or TMDBClient()
        
    def process_csv_files(self, ratings_file, watchlist_file=None, watched_file=None) -> Dict:
        """
        Process uploaded Letterboxd CSV files
        
        Args:
            ratings_file: Path or file object for ratings.csv
            watchlist_file: Optional path or file object for watchlist.csv
            watched_file: Optional path or file object for watched.csv
            
        Returns:
            Dictionary with processed data and statistics
        """
        try:
            # Read ratings file
            self.ratings_df = pd.read_csv(ratings_file)
            
            # Clean and process ratings
            self.ratings_df = self._clean_ratings_data(self.ratings_df)
            
            # Read optional files
            if watchlist_file is not None:
                self.watchlist_df = pd.read_csv(watchlist_file)
                self.watchlist_df = self._clean_movie_data(self.watchlist_df)
                
            if watched_file is not None:
                self.watched_df = pd.read_csv(watched_file)
                self.watched_df = self._clean_movie_data(self.watched_df)
            
            # Create unified movie dataset
            self.user_movies = self._create_unified_dataset()
            
            # Calculate statistics
            stats = self._calculate_statistics()
            
            return {
                'success': True,
                'stats': stats,
                'ratings': self.ratings_df,
                'user_movies': self.user_movies
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _clean_ratings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize ratings data"""
        # Common Letterboxd column names
        column_mapping = {
            'Name': 'title',
            'Year': 'year',
            'Rating': 'rating',
            'Date': 'date',
            'Letterboxd URI': 'uri',
            'Tags': 'tags',
            'Watched Date': 'watched_date',
            'Review': 'review',
            'Rewatch': 'rewatch'
        }
        
        # Rename columns if they exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Extract year from title if not in separate column
        if 'year' not in df.columns and 'title' in df.columns:
            df['year'] = df['title'].apply(self._extract_year)
            df['title'] = df['title'].apply(self._clean_title)
        
        # Convert rating to numeric (Letterboxd uses 0.5 to 5.0 scale)
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'watched_date' in df.columns:
            df['watched_date'] = pd.to_datetime(df['watched_date'], errors='coerce')
            
        # Handle missing values
        df = df.dropna(subset=['title'])
        
        return df
    
    def _clean_movie_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean movie data from watchlist or watched files"""
        column_mapping = {
            'Name': 'title',
            'Year': 'year',
            'Letterboxd URI': 'uri',
            'Date': 'date'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        if 'year' not in df.columns and 'title' in df.columns:
            df['year'] = df['title'].apply(self._extract_year)
            df['title'] = df['title'].apply(self._clean_title)
        
        return df
    
    def _extract_year(self, title: str) -> Optional[int]:
        """Extract year from movie title"""
        try:
            match = re.search(r'\((\d{4})\)', str(title))
            if match:
                return int(match.group(1))
        except:
            pass
        return None
    
    def _clean_title(self, title: str) -> str:
        """Remove year from title"""
        try:
            # Remove year in parentheses
            cleaned = re.sub(r'\s*\(\d{4}\)\s*$', '', str(title))
            return cleaned.strip()
        except:
            return str(title)
    
    def _create_unified_dataset(self) -> pd.DataFrame:
        """Create a unified dataset combining all available data"""
        if self.ratings_df is None:
            return None
        
        # Start with ratings
        movies = self.ratings_df.copy()
        
        # Add watched status
        movies['watched'] = True
        
        # Add watchlist items not yet watched
        if self.watchlist_df is not None:
            watchlist = self.watchlist_df.copy()
            watchlist['watched'] = False
            watchlist['in_watchlist'] = True
            
            # Merge without duplicates
            movies = pd.concat([movies, watchlist], ignore_index=True)
            movies = movies.drop_duplicates(subset=['title', 'year'], keep='first')
        
        return movies
    
    def _calculate_statistics(self) -> Dict:
        """Calculate viewing statistics"""
        if self.ratings_df is None:
            return {}
        
        stats = {
            'total_ratings': len(self.ratings_df),
            'unique_movies': self.ratings_df['title'].nunique(),
            'average_rating': self.ratings_df['rating'].mean() if 'rating' in self.ratings_df.columns else None,
            'rating_std': self.ratings_df['rating'].std() if 'rating' in self.ratings_df.columns else None,
            'date_range': None
        }
        
        # Date range
        if 'date' in self.ratings_df.columns:
            dates = self.ratings_df['date'].dropna()
            if len(dates) > 0:
                stats['date_range'] = {
                    'earliest': dates.min(),
                    'latest': dates.max()
                }
        
        # Rating distribution
        if 'rating' in self.ratings_df.columns:
            stats['rating_distribution'] = self.ratings_df['rating'].value_counts().to_dict()
        
        # Top rated movies
        if 'rating' in self.ratings_df.columns and 'title' in self.ratings_df.columns:
            top_rated = self.ratings_df.nlargest(10, 'rating')[['title', 'year', 'rating']]
            stats['top_rated'] = top_rated.to_dict('records')
        
        return stats
    
    def get_user_ratings(self) -> pd.DataFrame:
        """Get user ratings dataframe"""
        return self.ratings_df
    
    def get_user_movies(self) -> pd.DataFrame:
        """Get unified movies dataframe"""
        return self.user_movies
    
    def get_highly_rated_movies(self, threshold: float = 4.0) -> pd.DataFrame:
        """Get movies rated above threshold"""
        if self.ratings_df is None or 'rating' not in self.ratings_df.columns:
            return pd.DataFrame()
        
        # Filter safely to avoid ambiguous truth value errors
        try:
            result = self.ratings_df[self.ratings_df['rating'] >= threshold]
            return result
        except (KeyError, ValueError, TypeError):
            return pd.DataFrame()
    
    def get_movies_by_year_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Get movies within year range"""
        if self.user_movies is None or 'year' not in self.user_movies.columns:
            return pd.DataFrame()
        
        # Filter safely to avoid ambiguous truth value errors
        try:
            result = self.user_movies[
                (self.user_movies['year'] >= start_year) & 
                (self.user_movies['year'] <= end_year)
            ]
            return result
        except (KeyError, ValueError, TypeError):
            return pd.DataFrame()
    
    def format_stats_text(self, stats: Dict) -> str:
        """Format statistics as readable text"""
        if not stats:
            return "No statistics available"
        
        text = f"📊 **Statistics**\n\n"
        text += f"- Total ratings: {stats.get('total_ratings', 0)}\n"
        text += f"- Unique movies: {stats.get('unique_movies', 0)}\n"
        
        if stats.get('average_rating'):
            text += f"- Average rating: {stats['average_rating']:.2f} / 5.0\n"
            text += f"- Rating std dev: {stats.get('rating_std', 0):.2f}\n"
        
        if stats.get('date_range'):
            earliest = stats['date_range']['earliest']
            latest = stats['date_range']['latest']
            if pd.notna(earliest) and pd.notna(latest):
                text += f"- Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}\n"
        
        return text
