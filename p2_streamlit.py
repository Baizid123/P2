import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json, requests

# -------------------------------------------------
# BASIC CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Interactive Movie AI Dashboard",
    layout="wide",
    page_icon="🎬"
)

st.markdown("""
    <style>
    .stApp {background-color:#0b132b;color:white;}
    .metric {font-size:22px;font-weight:600;margin-top:10px}
    h1,h2,h3,h4 {color:#f7fafc;}
    </style>
""", unsafe_allow_html=True)

st.title("🎬 AI-Powered Movie Recommendation Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df["genres"] = df["genres"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["combined"] = (
        df["title"] + " " + df["genres"].str.replace("|"," ") + " " + df["overview"]
    )
    return df

movies = load_data()

# -------------------------------------------------
# TF-IDF
# -------------------------------------------------
@st.cache_data
def create_model(corpus):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(corpus)
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(matrix)
    return vectorizer, matrix, knn

vectorizer, tfidf_matrix, knn = create_model(movies["combined"].tolist())

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("🔧 Controls")

search_keyword = st.sidebar.text_input("Search Movies by Title")
selected_genres = st.sidebar.multiselect(
    "Filter by Genre",
    sorted(set("|".join(movies["genres"]).split("|")))
)

use_gemini = st.sidebar.checkbox("Enable Google Gemini AI")
gemini_api_key = None

if use_gemini:
    gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

k_count = st.sidebar.slider("Number of Recommendations", 3, 20, 6)

# -------------------------------------------------
# FILTER LOGIC
# -------------------------------------------------
filtered = movies.copy()

if search_keyword:
    filtered = filtered[filtered["title"].str.contains(search_keyword, case=False, na=False)]

if selected_genres:
    pattern = "|".join(selected_genres)
    filtered = filtered[filtered["genres"].str.contains(pattern)]

st.subheader("📄 Movie Dataset Overview")
st.write(f"Showing **{len(filtered)}** movies")
st.dataframe(filtered[["title","genres"]], use_container_width=True)

# -------------------------------------------------
# RECOMMENDATION SYSTEM
# -------------------------------------------------
st.subheader("🎯 Content-Based Recommendations")

selected_movie = st.selectbox("Choose a Movie:", movies["title"].tolist())

def recommend(movie_name, k=5):
    index = movies[movies["title"] == movie_name].index[0]
    distances, indices = knn.kneighbors(tfidf_matrix[index], n_neighbors=k+1)
    result = movies.iloc[indices[0][1:]][["title","genres"]]
    result["similarity"] = (1 - distances[0][1:]).round(3)
    return result

rec_df = recommend(selected_movie, k=k_count)
st.write(rec_df)

# -------------------------------------------------
# GEMINI ENHANCED RECOMMENDATIONS
# -------------------------------------------------
st.markdown("---")
st.subheader("🤖 AI Reasoned Recommendations (Optional)")

if use_gemini and gemini_api_key:
    prompt = f"""
    Based on the movie **{selected_movie}**, generate short reasoning for why each
    of the following movies is a good recommendation:

    {json.dumps(rec_df['title'].tolist())}

    Return ONLY JSON list format:
    [
        {{"title":"Movie", "reason":"Short reason"}}
    ]
    """

    if st.button("Generate AI Justification"):
        with st.spinner("Calling Gemini..."):
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                    headers={"Content-Type":"application/json",
                             "x-goog-api-key":gemini_api_key},
                    json={"contents":[{"parts":[{"text":prompt}]}]}
                )
                text = resp.json()["candidates"][0]["content"][0]["text"]
                start, end = text.find("["), text.rfind("]")+1
                parsed = json.loads(text[start:end])
                st.json(parsed)
            except:
                st.error("⚠ Gemini response unreadable. Try again.")

else:
    st.info("Enable Genie AI on sidebar to explain recommendation reasoning.")

# -------------------------------------------------
# VISUALS
# -------------------------------------------------
st.markdown("---")
st.subheader("📊 Genre Distribution")

genre_counts = (
    movies["genres"]
    .str.replace("|"," ")
    .str.split()
    .explode()
    .value_counts()
)

fig, ax = plt.subplots(figsize=(10,4))
genre_counts.head(15).plot(kind="bar", ax=ax)
st.pyplot(fig)

st.success("✅ Dashboard Ready — You can now deploy with `streamlit run P2_Final.py`")
