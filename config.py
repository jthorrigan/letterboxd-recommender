"""
Configuration file for Letterboxd Recommender App
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App Configuration
APP_TITLE = "🎬 Letterboxd Movie Recommender"
APP_DESCRIPTION = """
Get personalized movie recommendations based on your Letterboxd watching history!
Upload your Letterboxd data or enter your username to get started.
"""

# Scraping Configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_TIMEOUT = 10
RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Recommendation Configuration
DEFAULT_N_RECOMMENDATIONS = 10
MIN_SIMILARITY_SCORE = 0.1
MAX_RECOMMENDATIONS = 50

# Collaborative Filtering Configuration
MIN_RATINGS_FOR_CF = 10
SVD_COMPONENTS = 20

# Content-Based Filtering Configuration
CONTENT_WEIGHT = 0.4
COLLABORATIVE_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.3

# Visualization Configuration
PLOT_HEIGHT = 500
PLOT_TEMPLATE = "plotly_white"

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
CACHE_EXPIRY_DAYS = 30

# Recommendation Weights (must sum to 1.0)
GENRE_WEIGHT = 0.30
DIRECTOR_CAST_WEIGHT = 0.20
SEMANTIC_WEIGHT = 0.25
YEAR_WEIGHT = 0.10
TMDB_SIMILAR_WEIGHT = 0.15

# Validate weights sum to 1.0
_TOTAL_WEIGHT = GENRE_WEIGHT + DIRECTOR_CAST_WEIGHT + SEMANTIC_WEIGHT + YEAR_WEIGHT + TMDB_SIMILAR_WEIGHT
if not (0.99 <= _TOTAL_WEIGHT <= 1.01):  # Allow small floating point error
    raise ValueError(f"Recommendation weights must sum to 1.0, got {_TOTAL_WEIGHT}")

# Quality Filters
MIN_TMDB_RATING = 6.0
MIN_TMDB_VOTES = 100
MAX_SAME_DIRECTOR = 2
DIVERSITY_PENALTY = 0.15

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality sentence transformer

# OMDb API Configuration (optional - for additional metadata)
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

# Data paths
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)
