#!/usr/bin/env python3
"""
Demo script showing the Letterboxd Recommender in action
"""
import pandas as pd
from data_processor import DataProcessor
from recommender import MovieRecommender
from visualizations import MovieVisualizer

def main():
    print("=" * 70)
    print("🎬 Letterboxd Movie Recommender - Demo")
    print("=" * 70)
    
    # Create sample data
    print("\n📁 Loading sample movie ratings...")
    data = {
        'Name': [
            'The Shawshank Redemption (1994)',
            'The Godfather (1972)',
            'Pulp Fiction (1994)',
            'The Dark Knight (2008)',
            'Forrest Gump (1994)',
            'Inception (2010)',
            'Fight Club (1999)',
            'The Matrix (1999)',
            'Goodfellas (1990)',
            'Parasite (2019)',
            'Interstellar (2014)',
            'Whiplash (2014)',
            'The Prestige (2006)',
            'Memento (2000)',
            'Spirited Away (2001)'
        ],
        'Rating': [5.0, 5.0, 4.5, 4.5, 4.0, 5.0, 4.0, 4.5, 4.5, 5.0, 4.0, 4.5, 4.0, 4.0, 4.5],
        'Date': [
            '2024-01-15', '2024-01-10', '2024-01-20', '2024-01-25', '2024-02-01',
            '2024-02-05', '2024-02-10', '2024-02-15', '2024-02-20', '2024-02-25',
            '2024-03-01', '2024-03-05', '2024-03-10', '2024-03-15', '2024-03-20'
        ]
    }
    df = pd.DataFrame(data)
    temp_file = '/tmp/demo_ratings.csv'
    df.to_csv(temp_file, index=False)
    
    # Process data
    processor = DataProcessor()
    result = processor.process_csv_files(temp_file)
    
    if not result['success']:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"✅ Loaded {result['stats']['total_ratings']} movie ratings")
    print(f"   Average rating: {result['stats']['average_rating']:.2f} / 5.0")
    
    # Initialize recommender
    print("\n🤖 Initializing recommendation engine...")
    recommender = MovieRecommender(result['ratings'])
    
    # Get recommendations
    print("\n✨ Generating personalized recommendations...")
    recommendations = recommender.get_recommendations(n=5)
    
    print("\n" + "=" * 70)
    print("🎬 YOUR PERSONALIZED RECOMMENDATIONS")
    print("=" * 70)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        if rec['year']:
            print(f"   Year: {rec['year']}")
        print(f"   Match Score: {rec['score']:.0%}")
        print(f"   📝 {rec['explanation']}")
    
    # Find similar movies
    print("\n" + "=" * 70)
    print("🎯 SIMILAR MOVIES TO 'INCEPTION'")
    print("=" * 70)
    
    similar = recommender.find_similar_movies('Inception', n=3)
    for i, movie in enumerate(similar, 1):
        print(f"\n{i}. {movie['title']}")
        print(f"   Similarity: {movie['similarity']:.0%}")
        print(f"   📝 {movie['explanation']}")
    
    # Predict rating
    print("\n" + "=" * 70)
    print("🔮 RATING PREDICTION")
    print("=" * 70)
    
    prediction = recommender.predict_rating("The Lord of the Rings")
    print(f"\nMovie: {prediction['movie']}")
    if prediction['predicted_rating']:
        print(f"Predicted Rating: {prediction['predicted_rating']:.1f} / 5.0")
        print(f"Confidence: {prediction['confidence']:.0%}")
    print(f"📝 {prediction['explanation']}")
    
    # Get insights
    print("\n" + "=" * 70)
    print("📊 YOUR VIEWING INSIGHTS")
    print("=" * 70)
    
    insights = recommender.get_user_insights()
    print(f"\nTotal Movies Rated: {insights['total_movies']}")
    print(f"Average Rating: {insights['avg_rating']:.2f} / 5.0")
    print(f"Rating Tendency: You are a {insights['rating_tendency']} rater")
    if insights['favorite_decade']:
        print(f"Favorite Era: {insights['favorite_decade']}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    visualizer = MovieVisualizer(result['ratings'])
    
    # Test plot creation
    fig = visualizer.plot_rating_distribution()
    print("   ✅ Rating distribution chart")
    
    fig = visualizer.plot_ratings_over_time()
    print("   ✅ Ratings over time chart")
    
    fig = visualizer.plot_year_distribution()
    print("   ✅ Movies by decade chart")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)
    print("\n💡 To use the full Gradio app, run: python app.py")
    print("   Then open http://localhost:7860 in your browser")
    print("=" * 70)


if __name__ == "__main__":
    main()
