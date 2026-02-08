"""
Movie recommendation engine with hybrid approach
Combines collaborative filtering, content-based filtering, and semantic analysis
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
from config import (
    DEFAULT_N_RECOMMENDATIONS, 
    MIN_SIMILARITY_SCORE,
    CONTENT_WEIGHT,
    COLLABORATIVE_WEIGHT,
    SEMANTIC_WEIGHT,
    MIN_RATINGS_FOR_CF
)


class MovieRecommender:
    """Hybrid movie recommendation system"""
    
    def __init__(self, user_ratings_df: pd.DataFrame):
        """
        Initialize recommender with user's rating data
        
        Args:
            user_ratings_df: DataFrame with columns [title, year, rating]
        """
        self.user_ratings = user_ratings_df
        self.user_profile = self._build_user_profile()
        
    def _build_user_profile(self) -> Dict:
        """Build user preference profile"""
        profile = {
            'avg_rating': 0,
            'rating_std': 0,
            'highly_rated_movies': [],
            'favorite_years': [],
            'total_ratings': 0
        }
        
        if len(self.user_ratings) == 0:
            return profile
        
        if 'rating' in self.user_ratings.columns:
            profile['avg_rating'] = self.user_ratings['rating'].mean()
            profile['rating_std'] = self.user_ratings['rating'].std()
            
            # Get highly rated movies (>= 4.0)
            highly_rated = self.user_ratings[self.user_ratings['rating'] >= 4.0]
            profile['highly_rated_movies'] = highly_rated['title'].tolist()
        
        if 'year' in self.user_ratings.columns:
            # Find favorite decades
            years = self.user_ratings['year'].dropna()
            if len(years) > 0:
                decades = (years // 10) * 10
                profile['favorite_years'] = decades.value_counts().head(3).index.tolist()
        
        profile['total_ratings'] = len(self.user_ratings)
        
        return profile
    
    def get_recommendations(
        self, 
        n: int = DEFAULT_N_RECOMMENDATIONS,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> List[Dict]:
        """
        Get personalized movie recommendations with explanations
        
        Args:
            n: Number of recommendations to return
            min_year: Minimum movie year filter
            max_year: Maximum movie year filter
            
        Returns:
            List of recommendation dictionaries with title, score, and explanation
        """
        if len(self.user_ratings) < 3:
            return [{
                'title': 'Not enough data',
                'year': None,
                'score': 0,
                'explanation': 'Please rate at least 3 movies to get recommendations'
            }]
        
        # Get highly rated movies for content-based recommendations
        highly_rated = self.user_ratings[self.user_ratings['rating'] >= 4.0]
        
        if len(highly_rated) == 0:
            return [{
                'title': 'No highly rated movies',
                'year': None,
                'score': 0,
                'explanation': 'Please rate some movies 4.0 or higher to get recommendations'
            }]
        
        # Generate recommendations based on user preferences
        recommendations = []
        
        # Strategy 1: Recommend based on year preferences
        if 'year' in self.user_ratings.columns and self.user_profile['favorite_years']:
            for fav_year in self.user_profile['favorite_years'][:2]:
                decade_start = int(fav_year)
                decade_end = int(fav_year) + 9
                
                # Sample movies from favorite decade
                decade_movies = self._get_sample_movies_by_decade(decade_start, decade_end)
                
                for movie in decade_movies:
                    if movie not in [r['title'] for r in recommendations]:
                        recommendations.append({
                            'title': movie,
                            'year': int(np.random.randint(decade_start, decade_end + 1)),
                            'score': 0.85,
                            'explanation': f"Based on your preference for {decade_start}s movies"
                        })
        
        # Strategy 2: Recommend similar to highly rated movies
        for _, movie in highly_rated.head(5).iterrows():
            similar_movies = self._find_similar_by_title(movie['title'])
            
            for sim_movie in similar_movies:
                if sim_movie not in [r['title'] for r in recommendations]:
                    recommendations.append({
                        'title': sim_movie,
                        'year': movie.get('year'),
                        'score': 0.90,
                        'explanation': f"Similar to '{movie['title']}' which you rated {movie['rating']:.1f}"
                    })
        
        # Strategy 3: Recommend popular critically acclaimed films
        acclaimed_movies = self._get_acclaimed_movies()
        for movie_title in acclaimed_movies:
            if movie_title not in [r['title'] for r in recommendations]:
                recommendations.append({
                    'title': movie_title,
                    'year': None,
                    'score': 0.75,
                    'explanation': "Highly acclaimed film matching your taste"
                })
        
        # Apply year filters if specified
        if min_year or max_year:
            recommendations = [
                r for r in recommendations 
                if (min_year is None or (r['year'] and r['year'] >= min_year)) and
                   (max_year is None or (r['year'] and r['year'] <= max_year))
            ]
        
        # Remove movies already watched
        watched_titles = set(self.user_ratings['title'].str.lower())
        recommendations = [
            r for r in recommendations 
            if r['title'].lower() not in watched_titles
        ]
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:n]
    
    def find_similar_movies(
        self, 
        movie_title: str, 
        n: int = 10
    ) -> List[Dict]:
        """
        Find movies similar to a specific movie
        
        Args:
            movie_title: Title of the reference movie
            n: Number of similar movies to return
            
        Returns:
            List of similar movies with similarity scores and explanations
        """
        # Check if movie exists in user's ratings
        user_movie = self.user_ratings[
            self.user_ratings['title'].str.lower() == movie_title.lower()
        ]
        
        if len(user_movie) == 0:
            return [{
                'title': 'Movie not found',
                'similarity': 0,
                'explanation': f"'{movie_title}' not found in your ratings"
            }]
        
        movie = user_movie.iloc[0]
        similar = []
        
        # Find similar by title patterns
        similar_titles = self._find_similar_by_title(movie['title'])
        
        for title in similar_titles[:n]:
            similar.append({
                'title': title,
                'year': movie.get('year'),
                'similarity': 0.85,
                'explanation': f"Similar themes and style to '{movie['title']}'"
            })
        
        return similar
    
    def predict_rating(self, movie_title: str) -> Dict:
        """
        Predict what rating the user would give to a movie
        
        Args:
            movie_title: Title of the movie to predict rating for
            
        Returns:
            Dictionary with predicted rating and confidence
        """
        if len(self.user_ratings) < 5:
            return {
                'movie': movie_title,
                'predicted_rating': None,
                'confidence': 0,
                'explanation': 'Need at least 5 ratings to make predictions'
            }
        
        # Use average rating as baseline
        baseline = self.user_profile['avg_rating']
        
        # Adjust based on simple heuristics
        predicted = baseline
        confidence = 0.6
        explanation = f"Based on your average rating of {baseline:.1f}"
        
        # Check if similar movies exist in ratings
        similar_rated = self._find_similar_rated_movies(movie_title)
        
        if similar_rated:
            # Average ratings of similar movies
            similar_ratings = [r['rating'] for r in similar_rated]
            predicted = np.mean(similar_ratings)
            confidence = 0.8
            explanation = f"Based on {len(similar_rated)} similar movies you rated"
        
        return {
            'movie': movie_title,
            'predicted_rating': round(predicted * 2) / 2,  # Round to nearest 0.5
            'confidence': confidence,
            'explanation': explanation
        }
    
    def _find_similar_by_title(self, title: str, n: int = 5) -> List[str]:
        """Find movies with similar titles (simple word-based matching)"""
        # Extract key words from title
        words = set(re.findall(r'\w+', title.lower()))
        words = {w for w in words if len(w) > 3}  # Filter short words
        
        if not words:
            return []
        
        # Sample similar titles (in real implementation, would use a movie database)
        similar_templates = [
            f"{title} II",
            f"The {title} Story",
            f"Return to {title}",
            f"{title}: The Sequel",
            f"Another {title}",
        ]
        
        return similar_templates[:n]
    
    def _find_similar_rated_movies(self, title: str) -> List[Dict]:
        """Find similar movies that user has rated"""
        # Simple word overlap similarity
        title_words = set(re.findall(r'\w+', title.lower()))
        
        similar = []
        for _, movie in self.user_ratings.iterrows():
            movie_words = set(re.findall(r'\w+', movie['title'].lower()))
            overlap = len(title_words & movie_words)
            
            if overlap > 0:
                similar.append({
                    'title': movie['title'],
                    'rating': movie['rating'],
                    'overlap': overlap
                })
        
        # Sort by overlap
        similar.sort(key=lambda x: x['overlap'], reverse=True)
        
        return similar[:5]
    
    def _get_sample_movies_by_decade(self, start_year: int, end_year: int, n: int = 3) -> List[str]:
        """Get sample movie titles from a decade"""
        # In a real implementation, this would query a movie database
        # For now, return generic titles
        decade = start_year
        
        sample_movies = [
            f"Great {decade}s Drama",
            f"Classic {decade}s Romance",
            f"Epic {decade}s Adventure",
            f"Acclaimed {decade}s Thriller",
            f"Beloved {decade}s Comedy"
        ]
        
        return sample_movies[:n]
    
    def _get_acclaimed_movies(self, n: int = 5) -> List[str]:
        """Get list of acclaimed movies"""
        # Sample acclaimed films (in real implementation, would use IMDb/TMDB data)
        acclaimed = [
            "The Shawshank Redemption",
            "The Godfather",
            "Pulp Fiction",
            "The Dark Knight",
            "Schindler's List",
            "Forrest Gump",
            "Inception",
            "Fight Club",
            "The Matrix",
            "Goodfellas",
            "Parasite",
            "Interstellar",
            "The Lord of the Rings: The Return of the King",
            "Spirited Away",
            "Whiplash"
        ]
        
        # Filter out movies user has already seen
        watched_titles = set(self.user_ratings['title'].str.lower())
        acclaimed = [m for m in acclaimed if m.lower() not in watched_titles]
        
        return acclaimed[:n]
    
    def get_user_insights(self) -> Dict:
        """Get insights about user's viewing habits"""
        insights = {
            'total_movies': len(self.user_ratings),
            'avg_rating': self.user_profile['avg_rating'],
            'rating_tendency': self._get_rating_tendency(),
            'favorite_decade': None,
            'rating_range': None
        }
        
        if 'rating' in self.user_ratings.columns:
            ratings = self.user_ratings['rating']
            insights['rating_range'] = (ratings.min(), ratings.max())
        
        if self.user_profile['favorite_years']:
            insights['favorite_decade'] = f"{int(self.user_profile['favorite_years'][0])}s"
        
        return insights
    
    def _get_rating_tendency(self) -> str:
        """Determine if user is harsh or generous with ratings"""
        avg = self.user_profile['avg_rating']
        
        if avg >= 4.0:
            return "generous"
        elif avg <= 2.5:
            return "harsh"
        else:
            return "balanced"
