"""
Letterboxd profile scraping module
Scrapes public Letterboxd profiles to get ratings and watchlist data
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import Dict, List, Optional
import re
from config import USER_AGENT, REQUEST_TIMEOUT, RATE_LIMIT_DELAY


class LetterboxdScraper:
    """Scrape public Letterboxd profiles"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.base_url = "https://letterboxd.com"
        
    def scrape_user_profile(self, username: str) -> Dict:
        """
        Scrape a user's public Letterboxd profile
        
        Args:
            username: Letterboxd username
            
        Returns:
            Dictionary with scraped data and status
        """
        try:
            # Check if profile exists
            profile_url = f"{self.base_url}/{username}/"
            response = self._make_request(profile_url)
            
            if response is None or response.status_code != 200:
                return {
                    'success': False,
                    'error': 'Profile not found or is private'
                }
            
            # Scrape films (ratings and diary)
            films_data = self._scrape_films(username)
            
            # Scrape watchlist
            watchlist_data = self._scrape_watchlist(username)
            
            # Create dataframes
            ratings_df = self._create_ratings_dataframe(films_data)
            watchlist_df = self._create_watchlist_dataframe(watchlist_data)
            
            # Calculate statistics
            stats = {
                'username': username,
                'total_ratings': len(ratings_df),
                'watchlist_count': len(watchlist_df),
                'average_rating': ratings_df['rating'].mean() if len(ratings_df) > 0 and 'rating' in ratings_df.columns else None
            }
            
            return {
                'success': True,
                'ratings': ratings_df,
                'watchlist': watchlist_df,
                'stats': stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with rate limiting"""
        try:
            time.sleep(RATE_LIMIT_DELAY)
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            return response
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def _scrape_films(self, username: str, max_pages: int = 20) -> List[Dict]:
        """Scrape user's rated films"""
        films = []
        page = 1
        
        while page <= max_pages:
            url = f"{self.base_url}/{username}/films/page/{page}/"
            response = self._make_request(url)
            
            if response is None or response.status_code != 200:
                break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find film list items
            film_items = soup.find_all('li', class_='poster-container')
            
            if not film_items:
                break
            
            for item in film_items:
                film_data = self._parse_film_item(item)
                if film_data:
                    films.append(film_data)
            
            page += 1
            
            # Check if there are more pages
            pagination = soup.find('div', class_='pagination')
            if pagination:
                next_link = pagination.find('a', class_='next')
                if not next_link:
                    break
            else:
                break
        
        return films
    
    def _parse_film_item(self, item) -> Optional[Dict]:
        """Parse individual film item"""
        try:
            # Get film data from poster div
            poster = item.find('div', class_='film-poster')
            if not poster:
                return None
            
            # Title and year
            title = poster.get('data-film-name', '')
            year = poster.get('data-film-release-year', '')
            
            # Rating (stars)
            rating_span = item.find('span', class_='rating')
            rating = None
            if rating_span:
                # Letterboxd uses rated-N classes where N is rating * 2
                rating_class = [c for c in rating_span.get('class', []) if c.startswith('rated-')]
                if rating_class:
                    try:
                        rating_value = int(rating_class[0].replace('rated-', ''))
                        rating = rating_value / 2.0  # Convert to 0.5-5.0 scale
                    except:
                        pass
            
            # Film slug/URI
            film_link = poster.find('a')
            uri = ''
            if film_link:
                uri = film_link.get('href', '')
            
            return {
                'title': title,
                'year': int(year) if year else None,
                'rating': rating,
                'uri': uri
            }
            
        except Exception as e:
            print(f"Error parsing film item: {e}")
            return None
    
    def _scrape_watchlist(self, username: str, max_pages: int = 10) -> List[Dict]:
        """Scrape user's watchlist"""
        watchlist = []
        page = 1
        
        while page <= max_pages:
            url = f"{self.base_url}/{username}/watchlist/page/{page}/"
            response = self._make_request(url)
            
            if response is None or response.status_code != 200:
                break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find film list items
            film_items = soup.find_all('li', class_='poster-container')
            
            if not film_items:
                break
            
            for item in film_items:
                film_data = self._parse_watchlist_item(item)
                if film_data:
                    watchlist.append(film_data)
            
            page += 1
            
            # Check if there are more pages
            pagination = soup.find('div', class_='pagination')
            if pagination:
                next_link = pagination.find('a', class_='next')
                if not next_link:
                    break
            else:
                break
        
        return watchlist
    
    def _parse_watchlist_item(self, item) -> Optional[Dict]:
        """Parse individual watchlist item"""
        try:
            poster = item.find('div', class_='film-poster')
            if not poster:
                return None
            
            title = poster.get('data-film-name', '')
            year = poster.get('data-film-release-year', '')
            
            film_link = poster.find('a')
            uri = ''
            if film_link:
                uri = film_link.get('href', '')
            
            return {
                'title': title,
                'year': int(year) if year else None,
                'uri': uri
            }
            
        except Exception as e:
            print(f"Error parsing watchlist item: {e}")
            return None
    
    def _create_ratings_dataframe(self, films: List[Dict]) -> pd.DataFrame:
        """Create ratings dataframe from scraped data"""
        if not films:
            return pd.DataFrame(columns=['title', 'year', 'rating', 'uri'])
        
        df = pd.DataFrame(films)
        
        # Filter out films without ratings
        if 'rating' in df.columns:
            df = df[df['rating'].notna()]
        
        return df
    
    def _create_watchlist_dataframe(self, watchlist: List[Dict]) -> pd.DataFrame:
        """Create watchlist dataframe from scraped data"""
        if not watchlist:
            return pd.DataFrame(columns=['title', 'year', 'uri'])
        
        df = pd.DataFrame(watchlist)
        return df
