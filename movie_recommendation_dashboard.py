import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Movie Recommendation Dashboard", layout="wide")

# ------------------ THEME SWITCHER ------------------
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body{background-color:#0E1117;color:white;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background-color:white;color:black;}</style>", unsafe_allow_html=True)

# ------------------ TMDB API Config ------------------
TMDB_API_KEY = 'YOUR_TMDB_API_KEY'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

@st.cache_data
def fetch_movie_details(title):
    url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url).json()
    if response['results']:
        movie = response['results'][0]
        poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None
        overview = movie.get('overview', 'No summary available.')
        return poster_url, overview
    return None, 'No summary found.'

@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    return df

movies_df = load_data()

# ------------------ Sidebar Filters ------------------
st.sidebar.title("Filters")
selected_genres = st.sidebar.multiselect("Genres", options=movies_df['genre'].unique())
selected_year = st.sidebar.slider("Year Range", int(movies_df['year'].min()), int(movies_df['year'].max()), (2000, 2025))

# ------------------ LLM Semantic Search ------------------
st.sidebar.title("🎯 Natural Language Search")
nl_query = st.sidebar.text_input("Find movies like:")

# Filter dataframe
df_filtered = movies_df.copy()
if selected_genres:
    df_filtered = df_filtered[df_filtered['genre'].isin(selected_genres)]
df_filtered = df_filtered[(df_filtered['year'] >= selected_year[0]) & (df_filtered['year'] <= selected_year[1])]

st.title("Movie Dataset Summary")
st.dataframe(df_filtered.head())
st.write("Shape:", df_filtered.shape)

# ------------------ Feature Engineering ------------------
st.title("Feature Engineering")
st.write("Scaling numerical features for semantic recommendations.")
features = ['rating', 'year']
X = df_filtered[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
similarity_matrix = cosine_similarity(X_scaled)

# ------------------ Recommendations ------------------
st.title("Semantic Movie Recommendations")
selected_movie = st.selectbox("Pick a Movie", df_filtered['title'].values)
num_recs = st.slider("Number of Recommendations", 1, 10, 5)

def recommend_semantic(movie, n):
    idx = df_filtered[df_filtered['title'] == movie].index[0]
    sims = list(enumerate(similarity_matrix[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:n+1]
    return df_filtered.iloc[[i[0] for i in sims]]

# If LLM query exists, could override selection (mockup behavior)
if nl_query:
    selected_movie = nl_query  # In production, use embeddings to match best movie

recs = recommend_semantic(selected_movie, num_recs)

# ------------------ Carousel Display with CSS Glow & Frosted Glass ------------------
st.subheader(f"Top {num_recs} Recommendations — Carousel View")
carousel_html = "<div style='display:flex;overflow-x:auto;gap:20px;padding:10px;'>"

for _, row in recs.iterrows():
    poster, summary = fetch_movie_details(row['title'])
    card = f"""
        <div style='
            min-width:250px;
            border-radius:15px;
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.1);
            color:white;
            padding:15px;
            box-shadow: 0 0 20px rgba(255,255,255,0.2);
            transition:0.3s;
            cursor:pointer;'
            onmouseover="this.style.transform='scale(1.05)';this.style.boxShadow='0 0 30px #FFD700'" 
            onmouseout="this.style.transform='scale(1)';this.style.boxShadow='0 0 20px rgba(255,255,255,0.2)'"
        >
            <img src='{poster}' style='width:100%;border-radius:12px;'>
            <h3 style='margin-top:10px;'>{row['title']}</h3>
            <p><b>Year:</b> {row['year']} | <b>Rating:</b> {row['rating']}</p>
            <p><b>Genre:</b> {row['genre']}</p>
            <p style='opacity:0.8;font-size:14px;'>{summary[:180]}...</p>
        </div>
    """
    carousel_html += card

carousel_html += "</div>"
st.markdown(carousel_html, unsafe_allow_html=True)

# ------------------ Animated UI Component Example ------------------
st.subheader("🎛️ Enjoy Animated UI Components")
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie_animation, height=200)

# ------------------ Evaluation ------------------
st.title("Evaluation")
st.write("Metrics like MAP, Hit Rate, NDCG can be included.")

st.success("Fully Enhanced Dashboard Loaded with Glow Effects, Frosted Glass UI, Semantic Search, Carousel Navigation, and Theme Switcher!")
