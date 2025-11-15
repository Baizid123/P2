import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 2s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        color: white;
        transition: transform 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    .recommendation-section {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class MovieRecommendationSystem:
    def __init__(self):
        self.data = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.gemini_api_key = None
        
    def load_data(self):
        """Load and preprocess the movie dataset"""
        try:
            # Load the dataset
            self.data = pd.read_csv("movies.csv", low_memory=False)
            
            # Data cleaning steps from your notebook
            self.data = self.data.drop([
                'homepage', 'production_companies', 'production_countries',
                'spoken_languages', 'status', 'original_title', 'index', 'crew'], axis=1)
            
            # Handle missing values
            self.data.dropna(subset=['genres', 'keywords', 'overview', 'cast', 'director'], inplace=True)
            self.data['tagline'] = self.data['tagline'].fillna("")
            self.data['release_date'] = self.data['release_date'].fillna("NaT")
            self.data['runtime'] = pd.to_numeric(self.data['runtime'], errors='coerce')
            self.data['runtime'] = self.data['runtime'].fillna(self.data['runtime'].mean())
            
            # Convert data types
            self.data['release_date'] = pd.to_datetime(self.data['release_date'], errors='coerce')
            numeric_columns = ['vote_average', 'vote_count', 'runtime', 'budget', 'revenue', 'popularity']
            for col in numeric_columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            self.data['vote_count'] = pd.to_numeric(self.data['vote_count'], downcast='integer', errors='coerce')
            
            text_columns = ['title', 'genres', 'keywords', 'overview', 'tagline', 'cast', 'director', 'original_language']
            for col in text_columns:
                self.data[col] = self.data[col].astype(str)
            
            self.data.reset_index(drop=True, inplace=True)
            
            # Create additional features for analysis
            self.data['year'] = pd.to_datetime(self.data['release_date']).dt.year
            self.data['profit'] = self.data['revenue'] - self.data['budget']
            self.data['roi'] = (self.data['profit'] / self.data['budget']).replace([np.inf, -np.inf], np.nan)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def setup_gemini(self, api_key):
        """Setup Gemini API"""
        try:
            self.gemini_api_key = api_key
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Error setting up Gemini: {str(e)}")
            return False
    
    def create_similarity_matrix(self):
        """Create content-based similarity matrix"""
        try:
            # Combine features for content-based filtering
            self.data['combined_features'] = (
                self.data['genres'] + ' ' + 
                self.data['keywords'] + ' ' + 
                self.data['overview'] + ' ' + 
                self.data['cast'] + ' ' + 
                self.data['director']
            )
            
            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            self.tfidf_matrix = tfidf.fit_transform(self.data['combined_features'])
            
            # Compute cosine similarity
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            
            # Create indices mapping
            self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()
            
            return True
        except Exception as e:
            st.error(f"Error creating similarity matrix: {str(e)}")
            return False
    
    def get_content_recommendations(self, title, num_recommendations=10):
        """Get content-based recommendations"""
        try:
            idx = self.indices[title]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations+1]
            movie_indices = [i[0] for i in sim_scores]
            return self.data.iloc[movie_indices][['title', 'vote_average', 'genres', 'release_date', 'director', 'overview']]
        except:
            return None
    
    def get_gemini_recommendations(self, movie_title, num_recommendations=5):
        """Get AI-powered recommendations using Gemini"""
        try:
            if not self.gemini_api_key:
                return None
                
            movie_data = self.data[self.data['title'] == movie_title].iloc[0]
            
            prompt = f"""
            Based on the movie "{movie_title}" with the following details:
            - Genres: {movie_data['genres']}
            - Overview: {movie_data['overview']}
            - Director: {movie_data['director']}
            - Cast: {movie_data['cast'][:200]}  # Limit cast length
            - Rating: {movie_data['vote_average']}
            
            Recommend {num_recommendations} similar movies that a fan of "{movie_title}" would enjoy.
            Provide the recommendations in this exact format:
            
            **Movie Title 1**: Brief reasoning (1-2 sentences focusing on thematic similarities)
            **Movie Title 2**: Brief reasoning (1-2 sentences focusing on directorial style)
            **Movie Title 3**: Brief reasoning (1-2 sentences focusing on actor performances)
            **Movie Title 4**: Brief reasoning (1-2 sentences focusing on genre elements)
            **Movie Title 5**: Brief reasoning (1-2 sentences focusing on overall vibe)
            
            Make the recommendations diverse and insightful.
            """
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            return f"Error generating AI recommendations: {str(e)}"
    
    def create_visualizations(self):
        """Create comprehensive visualizations using only Plotly"""
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Dataset Overview", 
            "🎭 Genre Analysis", 
            "⭐ Rating Distribution",
            "💰 Financial Insights"
        ])
        
        with viz_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Movie count by year
                yearly_counts = self.data['year'].value_counts().sort_index()
                
                fig = px.line(
                    x=yearly_counts.index, 
                    y=yearly_counts.values,
                    title="🎬 Movies Released by Year",
                    labels={'x': 'Year', 'y': 'Number of Movies'},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Budget vs Revenue scatter plot
                sample_data = self.data.head(500).copy()
                sample_data = sample_data[sample_data['budget'] > 0]
                sample_data = sample_data[sample_data['revenue'] > 0]
                
                fig = px.scatter(
                    sample_data,
                    x='budget',
                    y='revenue',
                    color='vote_average',
                    size='popularity',
                    title="💰 Budget vs Revenue Analysis",
                    hover_data=['title'],
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Genre heatmap
                genres_list = []
                for genres in self.data['genres']:
                    genre_split = [genre.strip() for genre in genres.split() if genre.strip()]
                    genres_list.extend(genre_split)
                
                genre_counts = pd.Series(genres_list).value_counts().head(15)
                
                fig = px.bar(
                    x=genre_counts.values,
                    y=genre_counts.index,
                    orientation='h',
                    title="🏆 Top 15 Genres",
                    color=genre_counts.values,
                    color_continuous_scale='plasma',
                    labels={'x': 'Count', 'y': 'Genre'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Genre rating analysis
                genre_ratings = []
                for idx, row in self.data.iterrows():
                    genres = [genre.strip() for genre in row['genres'].split() if genre.strip()]
                    for genre in genres:
                        genre_ratings.append({'genre': genre, 'rating': row['vote_average']})
                
                genre_ratings_df = pd.DataFrame(genre_ratings)
                avg_ratings = genre_ratings_df.groupby('genre')['rating'].mean().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=avg_ratings.values,
                    y=avg_ratings.index,
                    orientation='h',
                    title="⭐ Top Rated Genres",
                    color=avg_ratings.values,
                    color_continuous_scale='viridis',
                    labels={'x': 'Average Rating', 'y': 'Genre'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution
                fig = px.histogram(
                    self.data,
                    x='vote_average',
                    nbins=20,
                    title="📈 Distribution of Movie Ratings",
                    color_discrete_sequence=['#4ECDC4'],
                    labels={'vote_average': 'Rating', 'count': 'Number of Movies'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Vote count distribution
                fig = px.histogram(
                    self.data,
                    x='vote_count',
                    nbins=50,
                    title="🗳️ Distribution of Vote Counts",
                    color_discrete_sequence=['#FF6B6B'],
                    labels={'vote_count': 'Vote Count', 'count': 'Number of Movies'}
                )
                fig.update_xaxes(type="log")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                # Profit analysis
                profitable_movies = self.data[self.data['profit'] > 0]
                fig = px.box(
                    profitable_movies,
                    y='profit',
                    title="💸 Movie Profit Distribution",
                    color_discrete_sequence=['#00CC96']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROI analysis
                valid_roi = self.data[self.data['roi'].notna() & (self.data['roi'] < 100)]  # Remove outliers
                fig = px.scatter(
                    valid_roi.head(200),
                    x='budget',
                    y='roi',
                    color='vote_average',
                    size='revenue',
                    title="📊 Return on Investment Analysis",
                    hover_data=['title'],
                    color_continuous_scale='rainbow'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    # Initialize the recommendation system
    recommender = MovieRecommendationSystem()
    
    # Header with animation
    st.markdown('<h1 class="main-header">🎬 CineMatch AI: Smart Movie Recommendations</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gemini API Key input
        gemini_api_key = st.text_input("Enter Gemini API Key:", type="password", 
                                      help="Get your API key from Google AI Studio")
        
        if gemini_api_key:
            if recommender.setup_gemini(gemini_api_key):
                st.success("✅ Gemini API configured!")
            else:
                st.error("❌ Failed to configure Gemini API")
        
        st.markdown("---")
        st.header("🎯 Recommendation Settings")
        num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        st.markdown("---")
        st.header("📈 Filter Options")
        min_rating = st.slider("Minimum Rating:", 0.0, 10.0, 7.0)
        min_votes = st.slider("Minimum Votes:", 0, 10000, 100)
        
        st.markdown("---")
        st.header("🚀 Quick Stats")
        if recommender.data is not None:
            st.metric("Total Movies", len(recommender.data))
            st.metric("Average Rating", f"{recommender.data['vote_average'].mean():.2f}")
            st.metric("Unique Directors", recommender.data['director'].nunique())
    
    # Load data with progress indicator
    if recommender.data is None:
        with st.spinner('🚀 Loading movie data...'):
            if recommender.load_data():
                st.success(f"✅ Loaded {len(recommender.data)} movies successfully!")
            else:
                st.error("❌ Failed to load data")
                return
    
    # Create similarity matrix
    if recommender.cosine_sim is None:
        with st.spinner('🔧 Building recommendation engine...'):
            if recommender.create_similarity_matrix():
                st.success("✅ Recommendation engine ready!")
            else:
                st.error("❌ Failed to build recommendation engine")
                return
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎭 Movie Explorer", 
        "🤖 AI Recommendations", 
        "📊 Analytics Dashboard",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("🔍 Movie Explorer & Recommendations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Movie selection
            selected_movie = st.selectbox(
                "Select a movie you like:",
                options=recommender.data['title'].sort_values().unique(),
                help="Choose a movie to get similar recommendations",
                key="movie_selector"
            )
            
            if selected_movie:
                movie_info = recommender.data[recommender.data['title'] == selected_movie].iloc[0]
                
                st.markdown("### 🎬 Selected Movie Info")
                st.markdown(f"""
                <div class="movie-card">
                    <h4>{movie_info['title']}</h4>
                    <p><strong>⭐ Rating:</strong> {movie_info['vote_average']}/10 ({movie_info['vote_count']} votes)</p>
                    <p><strong>🎭 Genres:</strong> {movie_info['genres']}</p>
                    <p><strong>📅 Release Date:</strong> {movie_info['release_date'].strftime('%Y-%m-%d') if pd.notna(movie_info['release_date']) else 'N/A'}</p>
                    <p><strong>🎬 Director:</strong> {movie_info['director']}</p>
                    <p><strong>💰 Budget:</strong> ${movie_info['budget']:,.0f}</p>
                    <p><strong>📝 Overview:</strong> {movie_info['overview'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if selected_movie:
                st.markdown("### 🎯 Content-Based Recommendations")
                
                # Get recommendations
                recommendations = recommender.get_content_recommendations(selected_movie, num_recommendations)
                
                if recommendations is not None:
                    # Display recommendations in a grid
                    cols = st.columns(2)
                    for idx, (_, movie) in enumerate(recommendations.iterrows()):
                        with cols[idx % 2]:
                            overview_preview = movie['overview'][:100] + "..." if len(movie['overview']) > 100 else movie['overview']
                            st.markdown(f"""
                            <div class="movie-card">
                                <h5>#{idx+1} {movie['title']}</h5>
                                <p>⭐ {movie['vote_average']}/10</p>
                                <p>🎭 {movie['genres'][:50]}...</p>
                                <p>🎬 {movie['director']}</p>
                                <p>📅 {movie['release_date'].strftime('%Y') if pd.notna(movie['release_date']) else 'N/A'}</p>
                                <p>📖 {overview_preview}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("Could not generate recommendations for this movie.")
    
    with tab2:
        st.header("🤖 AI-Powered Recommendations")
        
        if not gemini_api_key:
            st.warning("🔑 Please enter your Gemini API key in the sidebar to enable AI recommendations.")
        else:
            if selected_movie:
                with st.spinner('🧠 Generating AI recommendations using Gemini...'):
                    ai_recommendations = recommender.get_gemini_recommendations(selected_movie, 5)
                    
                    if ai_recommendations and "Error" not in ai_recommendations:
                        st.markdown("### 💡 Gemini AI Suggestions")
                        st.markdown(f"""
                        <div class="recommendation-section">
                            <h4>AI Recommendations for "{selected_movie}"</h4>
                            {ai_recommendations.replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>')}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate AI recommendations. Please check your API key.")
            
            # Advanced filtering section
            st.markdown("### 🎛️ Advanced Movie Discovery")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Extract unique genres
                all_genres = set()
                for genres in recommender.data['genres']:
                    genre_list = [genre.strip() for genre in genres.split() if genre.strip()]
                    all_genres.update(genre_list)
                
                selected_genres = st.multiselect(
                    "Filter by genres:",
                    options=sorted(list(all_genres)),
                    default=[],
                    key="genre_filter"
                )
            
            with col2:
                top_directors = recommender.data['director'].value_counts().head(20).index.tolist()
                selected_director = st.selectbox(
                    "Filter by director:",
                    options=["All Directors"] + top_directors,
                    key="director_filter"
                )
            
            with col3:
                year_range = st.slider(
                    "Release year range:",
                    min_value=int(recommender.data['year'].min()),
                    max_value=int(recommender.data['year'].max()),
                    value=(1990, 2020),
                    key="year_filter"
                )
            
            # Apply filters
            filtered_data = recommender.data.copy()
            
            if selected_genres:
                genre_filter = filtered_data['genres'].apply(
                    lambda x: any(genre in x for genre in selected_genres)
                )
                filtered_data = filtered_data[genre_filter]
            
            if selected_director != "All Directors":
                filtered_data = filtered_data[filtered_data['director'] == selected_director]
            
            filtered_data = filtered_data[
                (filtered_data['year'] >= year_range[0]) & 
                (filtered_data['year'] <= year_range[1]) &
                (filtered_data['vote_average'] >= min_rating) &
                (filtered_data['vote_count'] >= min_votes)
            ]
            
            st.markdown(f"### 📋 Filtered Results ({len(filtered_data)} movies)")
            
            # Display filtered results in a carousel-like format
            if len(filtered_data) > 0:
                # Show top movies from filtered results
                top_movies = filtered_data.nlargest(9, 'vote_average')
                
                display_cols = st.columns(3)
                for idx, (_, movie) in enumerate(top_movies.iterrows()):
                    with display_cols[idx % 3]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <h5>{movie['title']}</h5>
                            <p>⭐ {movie['vote_average']}/10 ({movie['vote_count']} votes)</p>
                            <p>🎭 {movie['genres'][:50]}...</p>
                            <p>🎬 {movie['director']}</p>
                            <p>📅 {movie['year']}</p>
                            <p>💰 ${movie['budget']:,.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🎭 No movies match your filters. Try adjusting your criteria.")
    
    with tab3:
        st.header("📊 Analytics Dashboard")
        recommender.create_visualizations()
        
        # Additional advanced analytics
        st.markdown("### 🔬 Advanced Movie Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Director analysis
            director_stats = recommender.data.groupby('director').agg({
                'vote_average': 'mean',
                'title': 'count',
                'revenue': 'mean',
                'profit': 'mean'
            }).round(2).sort_values('title', ascending=False).head(10)
            
            director_stats.columns = ['Avg Rating', 'Movie Count', 'Avg Revenue', 'Avg Profit']
            
            st.markdown("#### 🎬 Top Directors by Movie Count")
            st.dataframe(director_stats, use_container_width=True)
        
        with col2:
            # Language distribution
            language_counts = recommender.data['original_language'].value_counts().head(10)
            fig = px.pie(
                values=language_counts.values,
                names=language_counts.index,
                title="🌍 Top 10 Languages",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature engineering explanation
        st.markdown("### 🔧 Feature Engineering Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h5>🎭 Genre Analysis</h5>
                <p>Combined multiple genre tags to create comprehensive genre profiles for each movie</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h5>📝 Content Features</h5>
                <p>Integrated overview, cast, director, and keywords for semantic similarity analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h5>💰 Financial Metrics</h5>
                <p>Calculated profit and ROI to analyze commercial success patterns</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("ℹ️ About CineMatch AI")
        
        st.markdown("""
        ### 🎯 What is CineMatch AI?
        
        CineMatch AI is an intelligent movie recommendation system that combines:
        
        - **🤖 AI-Powered Insights**: Uses Google's Gemini AI for semantic understanding
        - **🎭 Content-Based Filtering**: Analyzes genres, cast, directors, and plots
        - **📊 Advanced Analytics**: Comprehensive data visualization and insights
        - **🎨 Interactive UI**: Beautiful, animated interface for seamless exploration
        
        ### 🔧 How It Works
        
        1. **Data Processing**: Cleans and processes movie metadata
        2. **Feature Engineering**: Combines multiple features for better recommendations
        3. **Similarity Analysis**: Uses cosine similarity for content-based filtering
        4. **AI Enhancement**: Gemini AI provides contextual recommendations
        5. **Visual Analytics**: Interactive plots and dashboards
        
        ### 📈 Key Features
        
        - **Smart Recommendations**: Both content-based and AI-powered suggestions
        - **Advanced Filtering**: Filter by genres, directors, ratings, and more
        - **Interactive Visualizations**: Comprehensive analytics dashboard
        - **Real-time Updates**: Dynamic recommendations based on user input
        - **Financial Analysis**: Budget, revenue, and ROI insights
        
        ### 🛠️ Technical Stack
        
        - **Backend**: Python, Pandas, Scikit-learn
        - **AI/ML**: Google Gemini API, TF-IDF, Cosine Similarity
        - **Visualization**: Plotly (no matplotlib dependency)
        - **Frontend**: Streamlit
        - **Data**: TMDB-style movie dataset
        
        ### 🎬 Get Started
        
        Choose a movie you like from the dropdown, and let CineMatch AI find your next favorite film!
        
        ### 🔑 API Setup
        
        To enable AI recommendations:
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Enjoy smart, contextual movie suggestions!
        """)

if __name__ == "__main__":
    main()