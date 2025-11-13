#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-13T07:45:56.190Z
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings

# Suppress warnings for cleaner Streamlit output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Interactive Movie Recommendation Dashboard")

# --- DATA PROCESSING AND MODELING FUNCTIONS (Cached for performance) ---

@st.cache_data
def load_and_preprocess_data(file_path="movies.csv"):
    """Loads the data and performs all cleaning/feature engineering steps from the notebook."""
    try:
        # Load data
        data = pd.read_csv(file_path, low_memory=False)
        st.success("✅ Dataset loaded successfully.")
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure 'movies.csv' is in the same directory.")
        return None

    # Create a copy and clean columns as per P2code.ipynb
    df = data.copy()

    # Drop unnecessary columns (Cell 4 logic)
    cols_to_drop = [
        'homepage', 'production_companies', 'production_countries',
        'spoken_languages', 'status', 'original_title', 'index', 'crew'
    ]
    df = df.drop(cols_to_drop, axis=1, errors='ignore')

    # Impute missing values (Cell 6 logic)
    for col in ['genres', 'keywords', 'overview', 'tagline', 'cast', 'director']:
        df[col] = df[col].fillna('')

    # Drop rows with single missing values in crucial columns (Cell 7 logic - single dropna)
    df.dropna(subset=['release_date', 'runtime'], inplace=True)

    # Convert features to suitable string format (Cell 7 logic)
    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        # Check if the column needs literal_eval (as might be needed for genres/keywords/cast)
        # Assuming the data is simplified string format post-cleaning, but handling lists if present
        try:
             # This part simplifies the complex parsing often found in notebooks
            df[feature] = df[feature].apply(lambda x: [i['name'] for i in literal_eval(x)] if isinstance(x, str) and '[' in x else x)
        except (ValueError, TypeError):
             # If not list-like string, continue as simple string
            pass

        df[feature] = df[feature].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        df[feature] = df[feature].apply(lambda x: x.lower().replace(" ", ""))

    # Create the 'combined_features' string for TF-IDF
    def create_soup(x):
        # Concatenate relevant features, replacing NaN with empty string
        return x['keywords'] + ' ' + x['cast'] + ' ' + x['genres'] + ' ' + x['director']

    df['combined_features'] = df.apply(create_soup, axis=1)

    # Filter data for modeling (requires non-zero revenue/budget and minimum vote count)
    df_modeling = df[(df['budget'] > 0) & (df['revenue'] > 0) & (df['vote_count'] >= 10)].copy()

    return df, df_modeling

# --- CORE RECOMMENDATION SYSTEM CLASSES ---

class ContentRecommender:
    """Manages the Content-Based Filtering using TF-IDF."""
    def __init__(self, df):
        self.df = df
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()

    def get_recommendations(self, title, n_recs=50):
        """Generates content-based recommendations."""
        if title not in self.indices:
            return pd.DataFrame({'title': [f"Movie '{title}' not found in database."], 'score': [0.0], 'description': [""]})

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the most similar movies (excluding itself)
        sim_scores = sim_scores[1:n_recs+1]
        movie_indices = [i[0] for i in sim_scores]
        recommendations = self.df[['title', 'overview']].iloc[movie_indices].copy()
        recommendations['content_score'] = [i[1] for i in sim_scores]
        recommendations.rename(columns={'overview': 'description'}, inplace=True)
        return recommendations.reset_index(drop=True)

class MockLLMQueryInterpreter:
    """Mocks the Generative AI component (LLM) to extract filtering criteria."""
    def __init__(self, df):
        # Create a set of all unique genres for keyword matching
        genres_list = df['genres'].apply(lambda x: re.split('(?=[A-Z])', x) if x else ['']).apply(lambda x: [i.lower() for i in x if i])
        self.all_genres = set(g for sub in genres_list for g in sub)

    def interpret_query(self, query):
        interpretation = {
            'genres': [],
            'rating_preference': None, # 'high' or 'low'
            'popularity_preference': None, # 'popular' or 'niche'
        }

        query_lower = query.lower()
        # 1. Genre extraction
        for genre in self.all_genres:
            if genre and genre in query_lower:
                interpretation['genres'].append(genre)

        # 2. Rating preference
        if any(word in query_lower for word in ['high rating', 'best rated', 'highly rated', 'critically acclaimed']):
            interpretation['rating_preference'] = 'high'
        elif any(word in query_lower for word in ['low rating', 'underrated', 'bad rating']):
            interpretation['rating_preference'] = 'low'

        # 3. Popularity preference
        if any(word in query_lower for word in ['popular', 'mainstream', 'widely seen', 'most watched']):
            interpretation['popularity_preference'] = 'popular'
        elif any(word in query_lower for word in ['niche', 'underground', 'less known', 'obscure']):
            interpretation['popularity_preference'] = 'niche'

        return interpretation

class HybridRecommender:
    """Combines Content-Based Filtering with criteria filtering based on LLM interpretation."""
    def __init__(self, df, content_recommender, mock_llm, pred_model=None):
        self.df = df
        self.content_recommender = content_recommender
        self.mock_llm = mock_llm
        self.pred_model = pred_model

    def get_hybrid_recommendations(self, movie_title, query, n_recs=10):
        # 1. Get initial content-based recommendations
        content_recs_all = self.content_recommender.get_recommendations(movie_title, n_recs=50)

        if 'not found' in content_recs_all['title'].iloc[0]:
            return content_recs_all

        # 2. Interpret the natural language query
        interpretation = self.mock_llm.interpret_query(query)

        # 3. Apply filters to initial candidates
        # Map back to the full dataset indices for filtering columns
        rec_titles = content_recs_all['title'].tolist()
        filtered_df = self.df[self.df['title'].isin(rec_titles)].copy()

        # Filtering based on LLM interpretation
        if interpretation.get('genres'):
            genre_filter = filtered_df['genres'].apply(lambda x: any(g in x for g in interpretation['genres']))
            filtered_df = filtered_df[genre_filter]

        if interpretation.get('rating_preference') == 'high':
            min_rating = self.df['vote_average'].quantile(0.75)
            filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]
        elif interpretation.get('rating_preference') == 'low':
            max_rating = self.df['vote_average'].quantile(0.25)
            filtered_df = filtered_df[filtered_df['vote_average'] <= max_rating]

        if interpretation.get('popularity_preference') == 'popular':
            min_pop = self.df['popularity'].median()
            filtered_df = filtered_df[filtered_df['popularity'] >= min_pop]
        elif interpretation.get('popularity_preference') == 'niche':
            max_pop = self.df['popularity'].median()
            filtered_df = filtered_df[filtered_df['popularity'] < max_pop]

        # 4. Final Scoring and Selection
        if filtered_df.empty:
            st.warning("⚠️ No movies matched both the content and filtering criteria. Showing only top content recommendations.")
            final_recs = content_recs_all.head(n_recs).copy()
            final_recs['final_score'] = final_recs['content_score']
        else:
            final_recs = filtered_df.merge(content_recs_all[['title', 'content_score']], on='title', how='left')

            # Simplified Hybrid Score (Content Score + Weighted Popularity)
            max_pop_all = self.df['popularity'].max()
            final_recs['normalized_popularity'] = final_recs['popularity'] / max_pop_all
            # Weighted average: 60% Content, 40% Popularity proxy for model score
            final_recs['final_score'] = (0.6 * final_recs['content_score']) + (0.4 * final_recs['normalized_popularity'])

            final_recs = final_recs.sort_values('final_score', ascending=False).head(n_recs)
            final_recs = final_recs[['title', 'final_score', 'overview']].rename(columns={'overview': 'description'})

        return final_recs

# --- MODEL TRAINING (Cached Resource) ---
@st.cache_resource
def train_classification_model(df_modeling):
    """Trains the classification models for the Evaluation page."""
    st.info("⚙️ Training classification models for evaluation metrics...")

    # Feature Engineering for Classification (Cell 16 logic)
    R = df_modeling['vote_average'].mean() # Overall mean rating
    V_m = df_modeling['vote_count'].median() # Overall median vote count

    df_modeling['recommendable'] = np.where(
        (df_modeling['vote_average'] >= R) & (df_modeling['vote_count'] >= V_m), 1, 0
    )

    # Feature and Target Selection (Cell 17 logic)
    numerical_features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
    categorical_features = ['original_language']
    all_features = numerical_features + categorical_features
    X = df_modeling[all_features].fillna(0)
    y = df_modeling['recommendable']

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing Pipeline (Cell 18 logic)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Model Pipelines (Cell 19 logic) - XGBoost is the primary model
    xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(random_state=42))])
    rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=42))])

    # Fit models
    xgb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return xgb_model, lr_model, rf_model, X_test, y_test, df_modeling

# --- DASHBOARD PAGES ---

def page_recommendation_engine(hybrid_recommender, df):
    st.title("🎬 Interactive Movie Recommendation Engine")
    st.markdown("""
    This hybrid recommender system combines **Content-Based Filtering** (using movie metadata like cast, director, genres, and keywords) with **Natural Language Query Filtering** (powered by a mock Generative AI interpreter) to provide context-aware movie suggestions.
    """)
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        movie_title = st.selectbox(
            "1. Select a Movie to Base Recommendations On:",
            options=[''] + sorted(df['title'].unique().tolist()),
            index=0
        )

    with col2:
        query = st.text_input(
            "2. Enter a Natural Language Query for Filtering:",
            placeholder="e.g., highly rated action movies with high budget"
        )

    st.markdown("---")

    if st.button("✨ Get Hybrid Recommendations", type="primary"):
        if not movie_title:
            st.warning("Please select a base movie.")
            return

        with st.spinner(f"Generating hybrid recommendations based on '{movie_title}' and query: '{query}'..."):
            recommendations = hybrid_recommender.get_hybrid_recommendations(
                movie_title, query, n_recs=10
            )

!pip install streamlit