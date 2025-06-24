import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper function for preprocessing ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s,]', '', text)
    return text

# --- Load model and data ---
vectorizer = joblib.load("tfidf_vectorizer.joblib")
data = pd.read_csv("global_jobs_dataset_100k.csv")

# Combine relevant fields and preprocess
data['combined'] = (
    data['title'].astype(str) + ' ' +
    data['description'].astype(str) + ' ' +
    data['skills'].astype(str)
).apply(preprocess_text)
data['date_posted'] = pd.to_datetime(data['date_posted'])

# Sort data by most recent jobs first
data = data.sort_values(by='date_posted', ascending=False).reset_index(drop=True)

# Precompute TF-IDF matrix for all jobs
tfidf_matrix = vectorizer.transform(data['combined'])

def filter_jobs_by_keywords(user_input, top_n=50):
    user_input_processed = preprocess_text(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = data.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    results = results[results['similarity_score'] > 0]  # filter zero similarity

    # Show most recent jobs first among those with the same similarity score
    return results.sort_values(by=['similarity_score', 'date_posted'], ascending=[False, False])

# --- Streamlit UI ---
st.title("Job Recommender")
st.write("Enter skills or keywords to find matching job postings")

user_input = st.text_input("Enter keywords (comma-separated)", "")

if st.button("Search"):
    if user_input.strip() == "":
        st.warning("Please enter at least one keyword.")
    else:
        results = filter_jobs_by_keywords(user_input)
        if results.empty:
            st.info("No matching jobs found.")
        else:
            st.success(f"Found {len(results)} matching jobs! Showing top {min(len(results), 50)} results.")
            for _, row in results.head(50).iterrows():
                st.markdown(f"### {row['title']}")
                st.markdown(f"**Location:** {row['location']}")
                st.markdown(f"**Posted:** {row['date_posted'].date()}")
                st.markdown(f"**Similarity Score:** {row['similarity_score']:.3f}")
                st.markdown(f"**Skills:** {row['skills']}")
                st.markdown(f"**Description:** {row['description']}")
                st.markdown("---")