"""
Enhanced movie recommendation engine with TMDB integration
Combines genre matching, director/actor preferences, semantic similarity, and diversity
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import Counter, defaultdict
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Semantic similarity disabled.")

from movie_metadata import TMDBClient
from config import (
    DEFAULT_N_RECOMMENDATIONS,
    GENRE_WEIGHT,
    DIRECTOR_CAST_WEIGHT,
    SEMANTIC_WEIGHT,
    YEAR_WEIGHT,
    TMDB_SIMILAR_WEIGHT,
    MIN_TMDB_RATING,
    MIN_TMDB_VOTES,
    MAX_SAME_DIRECTOR,
    DIVERSITY_PENALTY,
    EMBEDDING_MODEL,
    TMDB_API_KEY
)


class MovieRecommender:
    """Enhanced hybrid movie recommendation system"""
    
    def __init__(self, user_ratings_df: pd.DataFrame, tmdb_client: Optional[TMDBClient] = None):
        """
        Initialize recommender with user's rating data
        
        Args:
            user_ratings_df: DataFrame with columns [title, year, rating]
            tmdb_client: Optional TMDBClient for metadata fetching
        """
        self.user_ratings = user_ratings_df
        self.tmdb_client = tmdb_client or TMDBClient()
        self.user_profile = self._build_user_profile()
        self.enriched_ratings = self._enrich_user_ratings()
        
        # Initialize embedding model if available
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as e:
                print(f"Could not load embedding model: {e}")
        
        # Build candidate pool
        self.candidate_movies = self._build_candidate_pool()
        
    def _build_user_profile(self) -> Dict:
        """Build comprehensive user preference profile"""
        profile = {
            'avg_rating': 0,
            'rating_std': 0,
            'highly_rated_movies': [],
            'favorite_genres': [],
            'favorite_directors': [],
            'favorite_actors': [],
            'favorite_decades': [],
            'total_ratings': 0,
            'watched_titles_lower': set()
        }
        
        if len(self.user_ratings) == 0:
            return profile
        
        # Basic stats
        if 'rating' in self.user_ratings.columns:
            profile['avg_rating'] = self.user_ratings['rating'].mean()
            profile['rating_std'] = self.user_ratings['rating'].std()
            
            # Get highly rated movies (>= 4.0)
            highly_rated = self._get_highly_rated_movies()
            profile['highly_rated_movies'] = highly_rated[['title', 'year', 'rating']].to_dict('records')
        
        # Watched titles for filtering
        if 'title' in self.user_ratings.columns:
            profile['watched_titles_lower'] = set(self.user_ratings['title'].str.lower())
        
        # Year preferences
        if 'year' in self.user_ratings.columns:
            years = self.user_ratings['year'].dropna()
            if len(years) > 0:
                decades = (years // 10) * 10
                profile['favorite_decades'] = decades.value_counts().head(3).index.tolist()
        
        profile['total_ratings'] = len(self.user_ratings)
        
        return profile
    
    def _get_highly_rated_movies(self, threshold: float = 4.0) -> pd.DataFrame:
        """Helper method to get highly rated movies"""
        if 'rating' not in self.user_ratings.columns:
            return pd.DataFrame()
        return self.user_ratings[self.user_ratings['rating'] >= threshold]
    
    def _enrich_user_ratings(self) -> pd.DataFrame:
        """Enrich user ratings with TMDB metadata"""
        if not self.tmdb_client.is_enabled():
            return self.user_ratings
        
        enriched = self.user_ratings.copy()
        
        # Add metadata columns
        enriched['genres'] = None
        enriched['directors'] = None
        enriched['cast'] = None
        enriched['keywords'] = None
        enriched['overview'] = None
        enriched['tmdb_id'] = None
        
        # Fetch metadata for highly rated movies
        highly_rated = self._get_highly_rated_movies() if 'rating' in enriched.columns else enriched
        
        for idx, row in highly_rated.iterrows():
            metadata = self.tmdb_client.get_movie_metadata(row['title'], row.get('year'))
            
            if metadata:
                enriched.at[idx, 'genres'] = metadata.get('genres', [])
                enriched.at[idx, 'directors'] = metadata.get('directors', [])
                enriched.at[idx, 'cast'] = metadata.get('cast', [])
                enriched.at[idx, 'keywords'] = metadata.get('keywords', [])
                enriched.at[idx, 'overview'] = metadata.get('overview', '')
                enriched.at[idx, 'tmdb_id'] = metadata.get('tmdb_id')
        
        # Update user profile with genre/director/actor preferences
        self._update_profile_from_metadata(enriched)
        
        return enriched
    
    def _update_profile_from_metadata(self, enriched_df: pd.DataFrame):
        """Update user profile with preferences from enriched metadata"""
        if 'rating' not in enriched_df.columns:
            return
        
        # Analyze highly rated movies from the enriched dataframe
        highly_rated = enriched_df[enriched_df['rating'] >= 4.0]
        
        # Early return if no highly rated movies
        if highly_rated.empty:
            return
        
        # Genre preferences (weighted by rating)
        genre_scores = Counter()
        for _, row in highly_rated.iterrows():
            if pd.notna(row['genres']) and isinstance(row['genres'], list) and row['genres']:
                for genre in row['genres']:
                    genre_scores[genre] += row['rating']
        
        self.user_profile['favorite_genres'] = [g for g, _ in genre_scores.most_common(5)]
        
        # Director preferences
        director_scores = Counter()
        for _, row in highly_rated.iterrows():
            if pd.notna(row['directors']) and isinstance(row['directors'], list) and row['directors']:
                for director in row['directors']:
                    director_scores[director] += row['rating']
        
        self.user_profile['favorite_directors'] = [d for d, _ in director_scores.most_common(5)]
        
        # Actor preferences (weight by rating and position in cast)
        actor_scores = Counter()
        for _, row in highly_rated.iterrows():
            if pd.notna(row['cast']) and isinstance(row['cast'], list) and row['cast']:
                for i, actor in enumerate(row['cast'][:5]):  # Top 5 cast
                    # Weight by position (lead actors count more)
                    weight = row['rating'] * (1.0 - i * 0.1)
                    actor_scores[actor] += weight
        
        self.user_profile['favorite_actors'] = [a for a, _ in actor_scores.most_common(10)]
    
    def _build_candidate_pool(self) -> List[Dict]:
        """Build a pool of candidate movies to recommend"""
        if not self.tmdb_client.is_enabled():
            # Fallback: use basic acclaimed movies list
            return self._get_basic_candidate_pool()
        
        candidates = []
        seen_ids = set()
        
        # Get similar movies from TMDB for highly rated films
        highly_rated = self._get_highly_rated_movies() if 'rating' in self.enriched_ratings.columns else self.enriched_ratings
        
        for _, movie in highly_rated.head(10).iterrows():
            # Get TMDB similar movies
            if pd.notna(movie.get('tmdb_id')):
                similar = self.tmdb_client.get_similar_movies(int(movie['tmdb_id']), limit=10)
                
                for sim_movie in similar:
                    movie_id = sim_movie.get('id')
                    if movie_id and movie_id not in seen_ids:
                        seen_ids.add(movie_id)
                        
                        # Fetch full metadata
                        metadata = self.tmdb_client.get_movie_metadata(
                            sim_movie.get('title', ''),
                            self._extract_year(sim_movie.get('release_date'))
                        )
                        
                        if metadata and self._passes_quality_filter(metadata):
                            candidates.append(metadata)
        
        # Add movies from favorite genres and directors
        # (In a real implementation, would query TMDB for these)
        
        return candidates
    
    def _get_basic_candidate_pool(self) -> List[Dict]:
        """Fallback candidate pool when TMDB is not available"""
        # Return a list of acclaimed movies (simplified version)
        acclaimed = [
            {"title": "The Shawshank Redemption", "year": 1994, "genres": ["Drama"]},
            {"title": "The Godfather", "year": 1972, "genres": ["Crime", "Drama"]},
            {"title": "Pulp Fiction", "year": 1994, "genres": ["Crime", "Drama"]},
            {"title": "The Dark Knight", "year": 2008, "genres": ["Action", "Crime", "Drama"]},
            {"title": "Schindler's List", "year": 1993, "genres": ["Biography", "Drama", "History"]},
            {"title": "Fight Club", "year": 1999, "genres": ["Drama"]},
            {"title": "The Matrix", "year": 1999, "genres": ["Action", "Sci-Fi"]},
            {"title": "Goodfellas", "year": 1990, "genres": ["Biography", "Crime", "Drama"]},
            {"title": "Parasite", "year": 2019, "genres": ["Comedy", "Drama", "Thriller"]},
            {"title": "Interstellar", "year": 2014, "genres": ["Adventure", "Drama", "Sci-Fi"]},
            {"title": "Spirited Away", "year": 2001, "genres": ["Animation", "Adventure", "Family"]},
            {"title": "Whiplash", "year": 2014, "genres": ["Drama", "Music"]},
            {"title": "Inception", "year": 2010, "genres": ["Action", "Sci-Fi", "Thriller"]},
            {"title": "The Prestige", "year": 2006, "genres": ["Drama", "Mystery", "Thriller"]},
            {"title": "The Departed", "year": 2006, "genres": ["Crime", "Drama", "Thriller"]},
        ]
        
        # Filter out already watched
        return [
            m for m in acclaimed
            if m['title'].lower() not in self.user_profile['watched_titles_lower']
        ]
    
    def _passes_quality_filter(self, metadata: Dict) -> bool:
        """Check if a movie passes quality thresholds"""
        # Check rating and vote count
        rating = metadata.get('vote_average', 0)
        votes = metadata.get('vote_count', 0)
        
        if rating < MIN_TMDB_RATING or votes < MIN_TMDB_VOTES:
            return False
        
        # Check if already watched
        title = metadata.get('title', '').lower()
        if title in self.user_profile['watched_titles_lower']:
            return False
        
        # Filter out obvious sequels/prequels of watched movies
        if self._is_sequel_of_watched(title):
            return False
        
        return True
    
    def _is_sequel_of_watched(self, title: str) -> bool:
        """Check if movie appears to be a sequel of a watched movie"""
        title_lower = title.lower()
        
        # Common sequel patterns with word boundaries
        sequel_patterns = [
            r'\b(?:ii|iii|iv|v|vi|vii|viii|ix|x)\b',  # Roman numerals
            r'\b(?:2|3|4|5|6|7|8|9)\b',  # Numbers
            r':\s*(?:part|chapter|episode)\s+\d+',  # Part/Chapter/Episode
            r'\b(?:returns|revenge|reloaded|resurrection|the sequel|strikes back)\b',  # Sequel keywords
        ]
        
        for pattern in sequel_patterns:
            match = re.search(pattern, title_lower)
            if match:
                # Extract base title by removing the sequel pattern
                base_title = re.sub(pattern, '', title_lower).strip()
                # Remove trailing punctuation
                base_title = re.sub(r'[:\-\s]+$', '', base_title).strip()
                
                # Check if base title matches any watched movie (with word boundaries)
                for watched in self.user_profile['watched_titles_lower']:
                    watched_base = watched.strip()
                    # Check for substantial match (at least 70% of the base title)
                    if len(base_title) >= 4 and (
                        base_title in watched_base or 
                        watched_base in base_title and len(watched_base) >= len(base_title) * 0.7
                    ):
                        return True
        
        return False
    
    def get_recommendations(
        self,
        n: int = DEFAULT_N_RECOMMENDATIONS,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> List[Dict]:
        """
        Get personalized movie recommendations with detailed explanations
        
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
        
        # Get highly rated movies
        if 'rating' in self.user_ratings.columns:
            highly_rated = self.user_ratings[self.user_ratings['rating'] >= 4.0]
        else:
            highly_rated = self.user_ratings
        
        if len(highly_rated) == 0:
            return [{
                'title': 'No highly rated movies',
                'year': None,
                'score': 0,
                'explanation': 'Please rate some movies 4.0 or higher to get recommendations'
            }]
        
        # Show warning if TMDB not enabled
        if not self.tmdb_client.is_enabled():
            print("⚠️ TMDB API key not configured. Recommendations are limited. See README for setup.")
        
        # Score all candidate movies
        scored_candidates = []
        
        for candidate in self.candidate_movies:
            # Apply year filters
            if min_year and candidate.get('year') and candidate['year'] < min_year:
                continue
            if max_year and candidate.get('year') and candidate['year'] > max_year:
                continue
            
            # Calculate hybrid score
            score, explanation_parts = self._calculate_hybrid_score(candidate)
            
            if score > 0:
                scored_candidates.append({
                    'title': candidate.get('title', ''),
                    'year': candidate.get('year'),
                    'score': score,
                    'explanation': self._format_explanation(explanation_parts),
                    'metadata': candidate
                })
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply diversity filter
        diverse_recommendations = self._apply_diversity_filter(scored_candidates, n * 2)
        
        # Return top N
        return diverse_recommendations[:n]
    
    def _calculate_hybrid_score(self, candidate: Dict) -> Tuple[float, Dict]:
        """
        Calculate hybrid recommendation score with multiple factors
        
        Returns:
            Tuple of (score, explanation_parts)
        """
        scores = {}
        explanation = {}
        
        # 1. Genre match score (30%)
        genre_score, genre_exp = self._genre_preference_score(candidate)
        scores['genre'] = genre_score * GENRE_WEIGHT
        explanation['genre'] = genre_exp
        
        # 2. Director/Cast match (20%)
        director_cast_score, dc_exp = self._director_actor_score(candidate)
        scores['director_cast'] = director_cast_score * DIRECTOR_CAST_WEIGHT
        explanation['director_cast'] = dc_exp
        
        # 3. Semantic similarity (25%)
        semantic_score, sem_exp = self._semantic_similarity_score(candidate)
        scores['semantic'] = semantic_score * SEMANTIC_WEIGHT
        explanation['semantic'] = sem_exp
        
        # 4. Year/Era preference (10%)
        year_score, year_exp = self._year_preference_score(candidate)
        scores['year'] = year_score * YEAR_WEIGHT
        explanation['year'] = year_exp
        
        # 5. TMDB similarity (15%) - already baked into candidate selection
        scores['tmdb_similar'] = 0.85 * TMDB_SIMILAR_WEIGHT
        explanation['tmdb_similar'] = "Similar to your highly rated films"
        
        # Total score
        total_score = sum(scores.values())
        
        return total_score, explanation
    
    def _genre_preference_score(self, candidate: Dict) -> Tuple[float, str]:
        """Calculate score based on genre matching"""
        candidate_genres = candidate.get('genres', [])
        
        if not candidate_genres or not self.user_profile['favorite_genres']:
            return 0.0, ""
        
        # Calculate overlap with favorite genres
        candidate_genres_set = set(candidate_genres) if isinstance(candidate_genres, list) else set()
        favorite_genres_set = set(self.user_profile['favorite_genres'])
        
        # Check if sets are non-empty after conversion
        if not candidate_genres_set or not favorite_genres_set:
            return 0.0, ""
        
        overlap = candidate_genres_set & favorite_genres_set
        
        if not overlap:
            return 0.0, ""
        
        # Score based on overlap (max 1.0)
        score = len(overlap) / min(len(candidate_genres_set), len(favorite_genres_set))
        
        # Create explanation
        genres_list = ", ".join(list(overlap)[:3])
        explanation = f"Matches your favorite genres: {genres_list}"
        
        return score, explanation
    
    def _director_actor_score(self, candidate: Dict) -> Tuple[float, str]:
        """Calculate score based on director and actor preferences"""
        score = 0.0
        explanations = []
        
        # Director match (weighted more)
        candidate_directors = candidate.get('directors', [])
        if isinstance(candidate_directors, list) and self.user_profile['favorite_directors']:
            director_overlap = set(candidate_directors) & set(self.user_profile['favorite_directors'])
            if director_overlap:
                score += 0.7
                director = list(director_overlap)[0]
                
                # Count how many highly rated movies by this director the user has
                director_movie_count = 0
                for movie in self.user_profile['highly_rated_movies']:
                    # Check if this movie has director metadata and matches
                    movie_directors = None
                    for _, enriched_movie in self.enriched_ratings.iterrows():
                        if enriched_movie['title'] == movie.get('title'):
                            movie_directors = enriched_movie.get('directors')
                            break
                    
                    if movie_directors and isinstance(movie_directors, list) and director in movie_directors:
                        director_movie_count += 1
                
                if director_movie_count > 1:
                    explanations.append(f"You rated {director_movie_count} {director} films highly")
                else:
                    explanations.append(f"Directed by {director}")
        
        # Actor match
        candidate_cast = candidate.get('cast', [])
        if isinstance(candidate_cast, list) and self.user_profile['favorite_actors']:
            actor_overlap = set(candidate_cast) & set(self.user_profile['favorite_actors'])
            if actor_overlap:
                score += 0.3
                actors = ", ".join(list(actor_overlap)[:2])
                explanations.append(f"Features {actors}")
        
        explanation = "; ".join(explanations) if explanations else ""
        
        return min(score, 1.0), explanation
    
    def _semantic_similarity_score(self, candidate: Dict) -> Tuple[float, str]:
        """Calculate semantic similarity using embeddings or TF-IDF"""
        overview = candidate.get('overview', '')
        keywords = candidate.get('keywords', [])
        
        if not overview and not keywords:
            return 0.0, ""
        
        # Combine overview and keywords
        candidate_text = overview
        if keywords and isinstance(keywords, list):
            candidate_text += " " + " ".join(keywords)
        
# Get text from highly rated movies
highly_rated_texts = []

# Safety check: ensure rating column exists
if 'rating' not in self.enriched_ratings.columns:
    return 0.0, ""

# Filter highly rated movies safely
highly_rated_movies = self.enriched_ratings[self.enriched_ratings['rating'] >= 4.0]

# Check if we have any highly rated movies
if len(highly_rated_movies) == 0:
    return 0.0, ""

# Extract text from highly rated movies
for _, movie in highly_rated_movies.iterrows():
    text = movie.get('overview', '')
    movie_keywords = movie.get('keywords', [])
    if isinstance(movie_keywords, list):
        text += " " + " ".join(movie_keywords)
    if text.strip():
        highly_rated_texts.append(text)
        
        if not highly_rated_texts:
            return 0.0, ""
        
        # Use sentence transformers if available
        if self.embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                candidate_embedding = self.embedding_model.encode([candidate_text])
                user_embeddings = self.embedding_model.encode(highly_rated_texts)
                
                similarities = cosine_similarity(candidate_embedding, user_embeddings)[0]
                score = float(np.max(similarities))
                
                explanation = "Similar themes and plot to your favorites"
                return score, explanation
                
            except Exception as e:
                print(f"Embedding error: {e}")
        
        # Fallback to TF-IDF
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            all_texts = [candidate_text] + highly_rated_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            score = float(np.max(similarities))
            
            explanation = "Similar themes to your favorites"
            return score, explanation
            
        except Exception as e:
            print(f"TF-IDF error: {e}")
            return 0.0, ""
    
    def _year_preference_score(self, candidate: Dict) -> Tuple[float, str]:
        """Calculate score based on year/decade preferences"""
        year = candidate.get('year')
        
        if not year or not self.user_profile['favorite_decades']:
            return 0.0, ""
        
        decade = (year // 10) * 10
        
        if decade in self.user_profile['favorite_decades']:
            rank = self.user_profile['favorite_decades'].index(decade)
            # First favorite gets 1.0, second gets 0.7, third gets 0.4
            score = 1.0 - (rank * 0.3)
            explanation = f"From the {decade}s, one of your favorite eras"
            return score, explanation
        
        return 0.0, ""
    
    def _format_explanation(self, explanation_parts: Dict) -> str:
        """Format explanation parts into readable text"""
        parts = []
        
        # Prioritize most interesting explanations
        for key in ['director_cast', 'genre', 'semantic', 'year']:
            exp = explanation_parts.get(key, '')
            if exp:
                parts.append(exp)
        
        if not parts:
            parts.append(explanation_parts.get('tmdb_similar', 'Recommended based on your taste'))
        
        # Return top 2-3 reasons
        return " • ".join(parts[:3])
    
    def _apply_diversity_filter(self, candidates: List[Dict], limit: int) -> List[Dict]:
        """Apply diversity filtering to prevent too many similar recommendations"""
        if not candidates:
            return []
        
        diverse_recs = []
        director_counts = Counter()
        decade_counts = Counter()
        genre_counts = Counter()
        
        for candidate in candidates:
            if len(diverse_recs) >= limit:
                break
            
            metadata = candidate.get('metadata', {})
            
            # Check director diversity
            directors = metadata.get('directors', [])
            director_penalty = 0
            if directors:
                for director in directors:
                    if director_counts[director] >= MAX_SAME_DIRECTOR:
                        director_penalty = 1.0
                        break
                    director_penalty += director_counts[director] * DIVERSITY_PENALTY
            
            # Check decade diversity
            year = metadata.get('year')
            decade_penalty = 0
            if year:
                decade = (year // 10) * 10
                decade_penalty = decade_counts[decade] * DIVERSITY_PENALTY * 0.5
            
            # Check genre diversity
            # Genre penalty is averaged because a movie can have multiple genres,
            # and we want to encourage cross-genre recommendations rather than penalize
            # movies that happen to share one genre with previous recommendations
            genres = metadata.get('genres', [])
            genre_penalty = 0
            if genres:
                for genre in genres:
                    genre_penalty += genre_counts[genre] * DIVERSITY_PENALTY * 0.3
                genre_penalty /= max(len(genres), 1)
            
            # Total penalty
            total_penalty = director_penalty + decade_penalty + genre_penalty
            
            # Apply penalty to score
            adjusted_score = candidate['score'] * (1.0 - min(total_penalty, 0.5))
            
            if adjusted_score > 0.1:  # Minimum threshold after penalty
                diverse_recs.append(candidate)
                
                # Update counts
                if directors:
                    for director in directors:
                        director_counts[director] += 1
                if year:
                    decade_counts[(year // 10) * 10] += 1
                if genres:
                    for genre in genres:
                        genre_counts[genre] += 1
        
        return diverse_recs
    
    def _extract_year(self, release_date: Optional[str]) -> Optional[int]:
        """Extract year from release date string"""
        if not release_date:
            return None
        try:
            return int(release_date.split("-")[0])
        except (ValueError, IndexError):
            return None
    
    # Keep existing methods for compatibility
    def find_similar_movies(self, movie_title: str, n: int = 10) -> List[Dict]:
        """Find movies similar to a specific movie"""
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
        
        # Try to get TMDB similar movies
        if self.tmdb_client.is_enabled():
            # Find TMDB ID
            metadata = self.tmdb_client.search_movie(movie['title'], movie.get('year'))
            
            if metadata:
                similar = self.tmdb_client.get_similar_movies(metadata['id'], limit=n)
                
                results = []
                for sim_movie in similar:
                    results.append({
                        'title': sim_movie.get('title', ''),
                        'year': self._extract_year(sim_movie.get('release_date')),
                        'similarity': 0.85,
                        'explanation': f"Similar themes and style to '{movie['title']}'"
                    })
                
                return results[:n]
        
        # Fallback
        return [{
            'title': 'The Shawshank Redemption',
            'year': 1994,
            'similarity': 0.75,
            'explanation': f"Highly acclaimed film similar to your taste"
        }]
    
    def predict_rating(self, movie_title: str) -> Dict:
        """Predict what rating the user would give to a movie"""
        if len(self.user_ratings) < 5:
            return {
                'movie': movie_title,
                'predicted_rating': None,
                'confidence': 0,
                'explanation': 'Need at least 5 ratings to make predictions'
            }
        
        # Use average rating as baseline
        baseline = self.user_profile['avg_rating']
        predicted = baseline
        confidence = 0.6
        explanation = f"Based on your average rating of {baseline:.1f}"
        
        return {
            'movie': movie_title,
            'predicted_rating': round(predicted * 2) / 2,
            'confidence': confidence,
            'explanation': explanation
        }
    
    def get_user_insights(self) -> Dict:
        """Get insights about user's viewing habits"""
        insights = {
            'total_movies': len(self.user_ratings),
            'avg_rating': self.user_profile['avg_rating'],
            'rating_tendency': self._get_rating_tendency(),
            'favorite_decade': None,
            'rating_range': None,
            'favorite_genres': self.user_profile['favorite_genres'][:3],
            'favorite_directors': self.user_profile['favorite_directors'][:3],
        }
        
        if 'rating' in self.user_ratings.columns:
            ratings = self.user_ratings['rating']
            insights['rating_range'] = (ratings.min(), ratings.max())
        
        if self.user_profile['favorite_decades']:
            insights['favorite_decade'] = f"{int(self.user_profile['favorite_decades'][0])}s"
        
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
