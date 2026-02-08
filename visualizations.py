"""
Visualization module for movie watching patterns
Creates interactive charts using Plotly
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from config import PLOT_HEIGHT, PLOT_TEMPLATE


class MovieVisualizer:
    """Create visualizations for movie watching patterns"""
    
    def __init__(self, ratings_df: pd.DataFrame):
        self.ratings_df = ratings_df
        
    def plot_rating_distribution(self) -> go.Figure:
        """Create histogram of rating distribution"""
        if 'rating' not in self.ratings_df.columns:
            return self._create_empty_plot("No rating data available")
        
        ratings = self.ratings_df['rating'].dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ratings,
            nbinsx=10,
            name='Ratings',
            marker_color='#00c030'
        ))
        
        fig.update_layout(
            title="Rating Distribution",
            xaxis_title="Rating (0.5 - 5.0)",
            yaxis_title="Number of Movies",
            height=PLOT_HEIGHT,
            template=PLOT_TEMPLATE,
            showlegend=False
        )
        
        return fig
    
    def plot_ratings_over_time(self) -> go.Figure:
        """Plot ratings over time"""
        if 'date' not in self.ratings_df.columns or 'rating' not in self.ratings_df.columns:
            return self._create_empty_plot("No date or rating data available")
        
        df = self.ratings_df[['date', 'rating']].dropna()
        
        if len(df) == 0:
            return self._create_empty_plot("No valid date/rating data")
        
        df = df.sort_values('date')
        
        # Calculate rolling average
        df['rolling_avg'] = df['rating'].rolling(window=10, min_periods=1).mean()
        
        fig = go.Figure()
        
        # Individual ratings
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['rating'],
            mode='markers',
            name='Individual Ratings',
            marker=dict(size=6, color='#00c030', opacity=0.5)
        ))
        
        # Rolling average
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['rolling_avg'],
            mode='lines',
            name='10-Movie Average',
            line=dict(color='#ff8000', width=3)
        ))
        
        fig.update_layout(
            title="Ratings Over Time",
            xaxis_title="Date",
            yaxis_title="Rating",
            height=PLOT_HEIGHT,
            template=PLOT_TEMPLATE,
            hovermode='closest'
        )
        
        return fig
    
    def plot_watch_frequency(self) -> go.Figure:
        """Plot number of movies watched over time"""
        if 'date' not in self.ratings_df.columns:
            return self._create_empty_plot("No date data available")
        
        df = self.ratings_df[['date']].dropna()
        
        if len(df) == 0:
            return self._create_empty_plot("No valid date data")
        
        # Group by month
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_counts = df.groupby('year_month').size().reset_index(name='count')
        monthly_counts['date'] = monthly_counts['year_month'].dt.to_timestamp()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_counts['date'],
            y=monthly_counts['count'],
            name='Movies Watched',
            marker_color='#00c030'
        ))
        
        fig.update_layout(
            title="Movies Watched Per Month",
            xaxis_title="Date",
            yaxis_title="Number of Movies",
            height=PLOT_HEIGHT,
            template=PLOT_TEMPLATE,
            showlegend=False
        )
        
        return fig
    
    def plot_year_distribution(self) -> go.Figure:
        """Plot distribution of movie years watched"""
        if 'year' not in self.ratings_df.columns:
            return self._create_empty_plot("No year data available")
        
        years = self.ratings_df['year'].dropna()
        
        if len(years) == 0:
            return self._create_empty_plot("No valid year data")
        
        # Group into decades
        decade_counts = years.apply(lambda x: (x // 10) * 10).value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{int(d)}s" for d in decade_counts.index],
            y=decade_counts.values,
            name='Movies',
            marker_color='#00c030'
        ))
        
        fig.update_layout(
            title="Movies Watched by Decade",
            xaxis_title="Decade",
            yaxis_title="Number of Movies",
            height=PLOT_HEIGHT,
            template=PLOT_TEMPLATE,
            showlegend=False
        )
        
        return fig
    
    def plot_rating_trends(self) -> go.Figure:
        """Analyze if user is getting harsher or more generous over time"""
        if 'date' not in self.ratings_df.columns or 'rating' not in self.ratings_df.columns:
            return self._create_empty_plot("No date or rating data available")
        
        df = self.ratings_df[['date', 'rating']].dropna()
        
        if len(df) < 10:
            return self._create_empty_plot("Need at least 10 ratings for trend analysis")
        
        df = df.sort_values('date')
        
        # Split into periods
        n = len(df)
        period_size = max(n // 5, 10)  # At least 10 movies per period
        
        periods = []
        period_avgs = []
        
        for i in range(0, n, period_size):
            period_df = df.iloc[i:i+period_size]
            if len(period_df) > 0:
                periods.append(f"Period {len(periods)+1}")
                period_avgs.append(period_df['rating'].mean())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods,
            y=period_avgs,
            mode='lines+markers',
            name='Average Rating',
            line=dict(color='#00c030', width=3),
            marker=dict(size=10)
        ))
        
        # Add trend line
        if len(periods) > 1:
            import numpy as np
            x_numeric = list(range(len(periods)))
            z = np.polyfit(x_numeric, period_avgs, 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=periods,
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(color='#ff8000', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Rating Trends Over Time",
            xaxis_title="Time Period",
            yaxis_title="Average Rating",
            height=PLOT_HEIGHT,
            template=PLOT_TEMPLATE
        )
        
        return fig
    
    def create_summary_stats(self) -> str:
        """Create summary statistics text"""
        if len(self.ratings_df) == 0:
            return "No data available"
        
        stats = f"## 📊 Summary Statistics\n\n"
        stats += f"**Total Movies Rated:** {len(self.ratings_df)}\n\n"
        
        if 'rating' in self.ratings_df.columns:
            ratings = self.ratings_df['rating'].dropna()
            if len(ratings) > 0:
                stats += f"**Average Rating:** {ratings.mean():.2f} / 5.0\n\n"
                stats += f"**Most Common Rating:** {ratings.mode().iloc[0]:.1f}\n\n"
                stats += f"**Rating Range:** {ratings.min():.1f} - {ratings.max():.1f}\n\n"
        
        if 'year' in self.ratings_df.columns:
            years = self.ratings_df['year'].dropna()
            if len(years) > 0:
                stats += f"**Year Range:** {int(years.min())} - {int(years.max())}\n\n"
        
        if 'date' in self.ratings_df.columns:
            dates = self.ratings_df['date'].dropna()
            if len(dates) > 0:
                stats += f"**Watching Period:** {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}\n\n"
                days = (dates.max() - dates.min()).days
                if days > 0:
                    stats += f"**Movies per Day:** {len(self.ratings_df) / days:.2f}\n\n"
        
        return stats
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            height=PLOT_HEIGHT,
            template=PLOT_TEMPLATE,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
