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

# API Keys (optional - for enhanced metadata)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

# Data paths
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)
