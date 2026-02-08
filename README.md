# 🎬 Letterboxd Movie Recommender

A comprehensive Gradio-based movie recommendation application that analyzes your Letterboxd watching history to provide personalized movie recommendations.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

### 🎯 Core Functionality

1. **Dual Data Import Methods**
   - 📤 **CSV Upload**: Upload your official Letterboxd data export
   - 🔍 **Username Scraping**: Automatically fetch public profile data

2. **Hybrid Recommendation System**
   - 🤝 Collaborative filtering based on similar users
   - 📊 Content-based filtering using movie attributes
   - 🧠 Semantic analysis of your reviews and preferences
   - 💡 **Every recommendation includes an explanation** showing why it's suggested

3. **Personalized Recommendations**
   - Get top N movie suggestions based on your taste
   - Filter by genre, year range, and more
   - See match scores and detailed reasoning

4. **Similar Movie Finder**
   - Select any movie from your favorites
   - Discover similar films with similarity scores
   - Understand why movies are similar

5. **Rating Predictor**
   - Predict what rating you'd give to any movie
   - See confidence levels for predictions
   - Based on your rating patterns and preferences

6. **Interactive Visualizations**
   - 📊 Rating distribution histogram
   - 📈 Ratings over time with trends
   - 📅 Watch frequency timeline
   - 🎭 Movies by decade analysis
   - 📉 Rating trends (getting harsher/more generous?)

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jthorrigan/letterboxd-recommender.git
   cd letterboxd-recommender
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Open in browser**
   - The app will launch at `http://localhost:7860`

## 📥 Getting Your Letterboxd Data

### Method 1: CSV Export (Recommended)

1. Log in to your [Letterboxd](https://letterboxd.com) account
2. Navigate to **Settings** > **Import & Export**
3. Click **"Export your data"**
4. Wait for the email with your data export
5. Download and extract the ZIP file
6. Upload the CSV files to the app:
   - `ratings.csv` (required)
   - `watchlist.csv` (optional)
   - `watched.csv` (optional)

### Method 2: Username Scraping

1. Ensure your Letterboxd profile is **public**
2. Enter your username in the app (without @ symbol)
3. Click "Scrape Profile"
4. Wait for data to be fetched

**Note**: Scraping respects rate limits and only accesses public data.

## 🎮 How to Use

### Tab 1: Data Import
- Choose between CSV upload or username scraping
- Wait for data to process
- View import statistics

### Tab 2: Recommendations
- Set number of recommendations (5-50)
- Apply optional year filters
- Get personalized suggestions with explanations

### Tab 3: Similar Movies
- Select a movie from your rated films
- Choose number of similar movies
- Discover films with similar themes and styles

### Tab 4: Rating Predictor
- Enter any movie title
- See predicted rating and confidence level
- Understand the reasoning behind the prediction

### Tab 5: Insights & Visualizations
- View your watching statistics
- Explore interactive charts
- Understand your rating patterns and preferences

## 🏗️ Project Structure

```
letterboxd-recommender/
├── app.py                  # Main Gradio application
├── recommender.py          # Hybrid recommendation engine
├── scraper.py             # Letterboxd profile scraping
├── data_processor.py      # CSV processing and data cleaning
├── visualizations.py      # Interactive chart generation
├── config.py              # Configuration and constants
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── .gitattributes        # Git LFS configuration
└── README.md             # This file
```

## 🔧 Configuration

Edit `config.py` to customize:

- Number of recommendations
- Similarity thresholds
- Collaborative filtering parameters
- Content-based weights
- Visualization settings
- API keys (optional, for enhanced metadata)

### Optional: Movie API Integration

For enhanced movie metadata and posters:

1. Get free API keys:
   - [TMDB API](https://www.themoviedb.org/settings/api)
   - [OMDb API](http://www.omdbapi.com/apikey.aspx)

2. Create a `.env` file:
   ```
   TMDB_API_KEY=your_key_here
   OMDB_API_KEY=your_key_here
   ```

## 🌐 Deploy to Hugging Face Spaces

1. **Create a new Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Set visibility to Public or Private

2. **Upload files**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   
   # Copy all project files
   cp /path/to/letterboxd-recommender/* .
   
   git add .
   git commit -m "Initial commit"
   git push
   ```

3. **Space will build automatically**
   - Check the "Building" status
   - Once complete, your app will be live!

## 🛠️ Technology Stack

- **Frontend**: [Gradio](https://gradio.app/) 4.19+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, SciPy
- **Visualizations**: Plotly
- **Web Scraping**: BeautifulSoup4, Requests
- **Environment**: Python 3.8+

## 📊 Recommendation Algorithm

The app uses a **hybrid approach** combining:

1. **Content-Based Filtering** (40% weight)
   - Analyzes movie attributes: genres, years, directors
   - Finds patterns in your highly-rated films
   - Recommends based on feature similarity

2. **Collaborative Filtering** (30% weight)
   - Uses matrix factorization (SVD)
   - Finds users with similar taste
   - Suggests their favorites you haven't seen

3. **Semantic Analysis** (30% weight)
   - Analyzes text from reviews (if available)
   - Identifies themes and preferences
   - Uses word embeddings for similarity

**Result**: Each recommendation shows:
- Match score (0-100%)
- Explanation of why it's recommended
- Relevant metadata

## 🔒 Privacy & Ethics

### Privacy
- ✅ All data processing is local/session-based
- ✅ No data is stored permanently on servers
- ✅ No tracking or analytics
- ✅ You control your data

### Web Scraping Ethics
- ✅ Only public profile data is accessed
- ✅ Rate limiting prevents server overload
- ✅ Respects robots.txt
- ✅ User-agent identification
- ⚠️ Use responsibly and respect Letterboxd's terms

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions

- Integration with TMDB/OMDb APIs for movie posters and metadata
- More sophisticated recommendation algorithms
- Support for additional data sources
- Enhanced visualizations
- Mobile-responsive design improvements
- Additional filtering options
- Export recommendations to CSV/PDF

## 📝 Requirements

**Minimum**:
- Python 3.8+
- 3+ rated movies for basic recommendations

**Recommended**:
- Python 3.10+
- 20+ rated movies for quality recommendations
- Public Letterboxd profile (for scraping method)

## 🐛 Troubleshooting

### "Profile not found" error
- Ensure your profile is public
- Check username spelling (no @ symbol)
- Try exporting CSV instead

### "Need more ratings" message
- Rate at least 5-10 movies for basic functionality
- 20+ ratings recommended for quality results

### CSV upload fails
- Ensure you're using official Letterboxd export
- Check file is not corrupted
- Try re-exporting from Letterboxd

### Scraping is slow
- This is normal - respects rate limits
- Large profiles take longer
- Consider CSV upload for faster results

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Letterboxd](https://letterboxd.com/) for the amazing platform and data
- [Gradio](https://gradio.app/) for the excellent UI framework
- [Hugging Face](https://huggingface.co/) for Spaces hosting
- The open-source community for the tools and libraries

## 📧 Contact

Questions? Suggestions? Feel free to:
- Open an issue
- Submit a pull request
- Reach out on GitHub

---

**Disclaimer**: This is an unofficial tool and is not affiliated with, endorsed by, or connected to Letterboxd. All movie data and ratings belong to their respective owners.