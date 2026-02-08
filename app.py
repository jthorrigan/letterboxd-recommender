"""
Letterboxd Movie Recommender - Main Gradio Application
"""
import gradio as gr
import pandas as pd
from typing import Optional, Tuple
import os

from data_processor import DataProcessor
from scraper import LetterboxdScraper
from recommender import MovieRecommender
from visualizations import MovieVisualizer
from config import APP_TITLE, APP_DESCRIPTION, DEFAULT_N_RECOMMENDATIONS


# Global state
processor = None
recommender = None
visualizer = None


def process_csv_upload(ratings_file, watchlist_file=None, watched_file=None):
    """Process uploaded CSV files"""
    global processor, recommender, visualizer
    
    if ratings_file is None:
        return "❌ Please upload at least a ratings.csv file", gr.Dropdown(choices=[]), None, None, None, None, None
    
    try:
        processor = DataProcessor()
        result = processor.process_csv_files(ratings_file, watchlist_file, watched_file)
        
        if not result['success']:
            return f"❌ Error: {result['error']}", gr.Dropdown(choices=[]), None, None, None, None, None
        
        # Initialize recommender and visualizer
        recommender = MovieRecommender(result['ratings'])
        visualizer = MovieVisualizer(result['ratings'])
        
        # Format statistics
        stats_text = processor.format_stats_text(result['stats'])
        
        # Get movie list for dropdowns
        movie_list = result['ratings']['title'].tolist() if 'title' in result['ratings'].columns else []
        
        # Create visualizations
        rating_dist = visualizer.plot_rating_distribution()
        ratings_time = visualizer.plot_ratings_over_time()
        watch_freq = visualizer.plot_watch_frequency()
        year_dist = visualizer.plot_year_distribution()
        rating_trends = visualizer.plot_rating_trends()
        
        success_msg = f"✅ Successfully loaded {result['stats']['total_ratings']} ratings!\n\n{stats_text}"
        
        return success_msg, gr.Dropdown(choices=movie_list), rating_dist, ratings_time, watch_freq, year_dist, rating_trends
        
    except Exception as e:
        return f"❌ Error processing files: {str(e)}", gr.Dropdown(choices=[]), None, None, None, None, None


def scrape_letterboxd_profile(username):
    """Scrape a Letterboxd profile"""
    global processor, recommender, visualizer
    
    if not username or username.strip() == "":
        return "❌ Please enter a Letterboxd username", gr.Dropdown(choices=[]), None, None, None, None, None
    
    try:
        scraper = LetterboxdScraper()
        result = scraper.scrape_user_profile(username.strip())
        
        if not result['success']:
            return f"❌ Error: {result['error']}", gr.Dropdown(choices=[]), None, None, None, None, None
        
        if len(result['ratings']) == 0:
            return "❌ No ratings found for this profile", gr.Dropdown(choices=[]), None, None, None, None, None
        
        # Process the scraped data
        processor = DataProcessor()
        processor.ratings_df = result['ratings']
        processor.watchlist_df = result.get('watchlist')
        processor.user_movies = processor._create_unified_dataset()
        
        # Initialize recommender and visualizer
        recommender = MovieRecommender(result['ratings'])
        visualizer = MovieVisualizer(result['ratings'])
        
        # Get movie list for dropdowns
        movie_list = result['ratings']['title'].tolist()
        
        # Create visualizations
        rating_dist = visualizer.plot_rating_distribution()
        ratings_time = visualizer.plot_ratings_over_time()
        watch_freq = visualizer.plot_watch_frequency()
        year_dist = visualizer.plot_year_distribution()
        rating_trends = visualizer.plot_rating_trends()
        
        stats = result['stats']
        success_msg = f"✅ Successfully scraped profile for @{username}!\n\n"
        success_msg += f"📊 **Statistics**\n\n"
        success_msg += f"- Total ratings: {stats['total_ratings']}\n"
        success_msg += f"- Watchlist items: {stats['watchlist_count']}\n"
        if stats['average_rating']:
            success_msg += f"- Average rating: {stats['average_rating']:.2f} / 5.0\n"
        
        return success_msg, gr.Dropdown(choices=movie_list), rating_dist, ratings_time, watch_freq, year_dist, rating_trends
        
    except Exception as e:
        return f"❌ Error scraping profile: {str(e)}", gr.Dropdown(choices=[]), None, None, None, None, None


def get_recommendations(n_recs, min_year, max_year):
    """Get personalized recommendations"""
    global recommender
    
    if recommender is None:
        return "❌ Please load your data first (Tab 1)"
    
    try:
        # Parse year filters
        min_y = int(min_year) if min_year and str(min_year).strip() != "" else None
        max_y = int(max_year) if max_year and str(max_year).strip() != "" else None
        
        recommendations = recommender.get_recommendations(
            n=int(n_recs),
            min_year=min_y,
            max_year=max_y
        )
        
        # Format recommendations
        output = "## 🎬 Your Personalized Recommendations\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            output += f"### {i}. {rec['title']}"
            if rec['year']:
                output += f" ({rec['year']})"
            output += f"\n\n"
            output += f"**Match Score:** {rec['score']:.0%}\n\n"
            output += f"**Why recommended:** {rec['explanation']}\n\n"
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error generating recommendations: {str(e)}"


def find_similar_movies(movie_title, n_similar):
    """Find movies similar to a selected movie"""
    global recommender
    
    if recommender is None:
        return "❌ Please load your data first (Tab 1)"
    
    if not movie_title:
        return "❌ Please select a movie"
    
    try:
        similar = recommender.find_similar_movies(movie_title, n=int(n_similar))
        
        output = f"## 🎯 Movies Similar to '{movie_title}'\n\n"
        
        for i, movie in enumerate(similar, 1):
            output += f"### {i}. {movie['title']}"
            if movie.get('year'):
                output += f" ({movie['year']})"
            output += f"\n\n"
            output += f"**Similarity:** {movie['similarity']:.0%}\n\n"
            output += f"**Why similar:** {movie['explanation']}\n\n"
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error finding similar movies: {str(e)}"


def predict_movie_rating(movie_title):
    """Predict rating for a movie"""
    global recommender
    
    if recommender is None:
        return "❌ Please load your data first (Tab 1)"
    
    if not movie_title or movie_title.strip() == "":
        return "❌ Please enter a movie title"
    
    try:
        prediction = recommender.predict_rating(movie_title.strip())
        
        output = f"## 🔮 Rating Prediction for '{prediction['movie']}'\n\n"
        
        if prediction['predicted_rating'] is None:
            output += f"**Status:** {prediction['explanation']}\n"
        else:
            output += f"**Predicted Rating:** {prediction['predicted_rating']:.1f} / 5.0\n\n"
            output += f"**Confidence:** {prediction['confidence']:.0%}\n\n"
            output += f"**Reasoning:** {prediction['explanation']}\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error predicting rating: {str(e)}"


def get_insights_text():
    """Get text insights about viewing habits"""
    global recommender, visualizer
    
    if recommender is None or visualizer is None:
        return "❌ Please load your data first (Tab 1)"
    
    try:
        insights = recommender.get_user_insights()
        summary = visualizer.create_summary_stats()
        
        output = summary + "\n\n"
        output += "## 🎭 Viewing Patterns\n\n"
        output += f"**Rating Tendency:** You are a **{insights['rating_tendency']}** rater\n\n"
        
        if insights['favorite_decade']:
            output += f"**Favorite Era:** {insights['favorite_decade']}\n\n"
        
        if insights['rating_range']:
            min_r, max_r = insights['rating_range']
            output += f"**Your Rating Range:** {min_r:.1f} - {max_r:.1f}\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error generating insights: {str(e)}"


# Create Gradio Interface
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as app:
    
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)
    
    with gr.Tabs():
        
        # Tab 1: Data Import
        with gr.Tab("📁 Data Import"):
            gr.Markdown("## Import Your Letterboxd Data")
            gr.Markdown("Choose one of the two methods below:")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Method 1: Upload CSV Files")
                    gr.Markdown("Export your data from Letterboxd Settings > Import & Export")
                    
                    ratings_upload = gr.File(label="ratings.csv (Required)", file_types=[".csv"])
                    watchlist_upload = gr.File(label="watchlist.csv (Optional)", file_types=[".csv"])
                    watched_upload = gr.File(label="watched.csv (Optional)", file_types=[".csv"])
                    
                    upload_btn = gr.Button("📤 Process CSV Files", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### Method 2: Scrape Public Profile")
                    gr.Markdown("Enter your Letterboxd username to automatically fetch your public data")
                    
                    username_input = gr.Textbox(
                        label="Letterboxd Username",
                        placeholder="username",
                        info="Your profile must be public"
                    )
                    
                    scrape_btn = gr.Button("🔍 Scrape Profile", variant="primary")
            
            import_status = gr.Markdown("")
        
        # Tab 2: Get Recommendations
        with gr.Tab("🎬 Recommendations"):
            gr.Markdown("## Get Personalized Movie Recommendations")
            
            with gr.Row():
                n_recs_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=DEFAULT_N_RECOMMENDATIONS,
                    step=5,
                    label="Number of Recommendations"
                )
            
            with gr.Row():
                min_year_input = gr.Number(label="Min Year (Optional)", precision=0)
                max_year_input = gr.Number(label="Max Year (Optional)", precision=0)
            
            rec_btn = gr.Button("✨ Get Recommendations", variant="primary")
            rec_output = gr.Markdown("")
        
        # Tab 3: Similar Movies
        with gr.Tab("🎯 Similar Movies"):
            gr.Markdown("## Find Movies Similar to Your Favorites")
            
            movie_dropdown = gr.Dropdown(
                label="Select a movie from your ratings",
                choices=[],
                interactive=True
            )
            
            n_similar_slider = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=5,
                label="Number of Similar Movies"
            )
            
            similar_btn = gr.Button("🔍 Find Similar Movies", variant="primary")
            similar_output = gr.Markdown("")
        
        # Tab 4: Rating Predictor
        with gr.Tab("🔮 Rate Predictor"):
            gr.Markdown("## Predict Your Rating for Any Movie")
            
            movie_search = gr.Textbox(
                label="Movie Title",
                placeholder="Enter a movie title...",
                info="We'll predict what rating you'd give this movie"
            )
            
            predict_btn = gr.Button("🎯 Predict Rating", variant="primary")
            predict_output = gr.Markdown("")
        
        # Tab 5: Insights & Visualizations
        with gr.Tab("📊 Insights & Visualizations"):
            gr.Markdown("## Your Watching Patterns")
            
            insights_text = gr.Markdown("")
            refresh_insights_btn = gr.Button("🔄 Refresh Insights")
            
            gr.Markdown("### 📈 Visualizations")
            
            with gr.Row():
                plot_rating_dist = gr.Plot(label="Rating Distribution")
                plot_ratings_time = gr.Plot(label="Ratings Over Time")
            
            with gr.Row():
                plot_watch_freq = gr.Plot(label="Watch Frequency")
                plot_year_dist = gr.Plot(label="Movies by Decade")
            
            plot_rating_trends = gr.Plot(label="Rating Trends")
        
        # Tab 6: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This App
            
            This app provides personalized movie recommendations based on your Letterboxd watching history.
            
            ### 🎯 Features
            
            1. **Data Import**: Upload your Letterboxd CSV export or scrape your public profile
            2. **Personalized Recommendations**: Get movie suggestions with explanations
            3. **Similar Movies**: Find movies similar to your favorites
            4. **Rating Predictions**: See what rating you'd likely give to any movie
            5. **Insights & Visualizations**: Understand your watching patterns
            
            ### 📥 How to Export Your Letterboxd Data
            
            1. Log in to your Letterboxd account
            2. Go to Settings > Import & Export
            3. Click "Export your data"
            4. Download the ZIP file
            5. Extract and upload the CSV files here
            
            ### 🔒 Privacy
            
            - All data processing happens in your browser session
            - No data is stored permanently
            - Scraping only accesses public profile information
            
            ### 🛠️ Technology
            
            Built with:
            - **Gradio** for the interface
            - **Pandas & NumPy** for data processing
            - **Scikit-learn** for recommendations
            - **Plotly** for visualizations
            - **BeautifulSoup** for web scraping
            
            ### 📝 Note
            
            This is a demonstration app. Recommendation quality improves with more ratings.
            For best results, have at least 20-30 rated movies.
            
            ### 🙏 Credits
            
            Data from [Letterboxd](https://letterboxd.com/)
            """)
    
    # Event handlers
    
    # Upload CSV files
    upload_btn.click(
        fn=process_csv_upload,
        inputs=[ratings_upload, watchlist_upload, watched_upload],
        outputs=[
            import_status, 
            movie_dropdown,
            plot_rating_dist,
            plot_ratings_time,
            plot_watch_freq,
            plot_year_dist,
            plot_rating_trends
        ]
    )
    
    # Scrape profile
    scrape_btn.click(
        fn=scrape_letterboxd_profile,
        inputs=[username_input],
        outputs=[
            import_status,
            movie_dropdown,
            plot_rating_dist,
            plot_ratings_time,
            plot_watch_freq,
            plot_year_dist,
            plot_rating_trends
        ]
    )
    
    # Get recommendations
    rec_btn.click(
        fn=get_recommendations,
        inputs=[n_recs_slider, min_year_input, max_year_input],
        outputs=[rec_output]
    )
    
    # Find similar movies
    similar_btn.click(
        fn=find_similar_movies,
        inputs=[movie_dropdown, n_similar_slider],
        outputs=[similar_output]
    )
    
    # Predict rating
    predict_btn.click(
        fn=predict_movie_rating,
        inputs=[movie_search],
        outputs=[predict_output]
    )
    
    # Refresh insights
    refresh_insights_btn.click(
        fn=get_insights_text,
        inputs=[],
        outputs=[insights_text]
    )


if __name__ == "__main__":
    app.launch()
