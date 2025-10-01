import joblib
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = joblib.load('vectorizer.joblib')
x = joblib.load('tfidf_matrix.joblib')
newdf= joblib.load('songdata.joblib')

def rec(song_name, top_n=5):
    song_name = song_name.strip().lower()
    newdf['song_lower'] = newdf['song'].str.lower()

    if song_name not in newdf['song_lower'].values:
        return f"'{song_name}' not found in dataset."

    idx = newdf[newdf['song_lower'] == song_name].index[0]
    query_vec = x[idx]

    # ✅ Compute similarity only for this one song
    sim_scores = cosine_similarity(query_vec, x).flatten()
    sim_scores[idx] = -1  # exclude the song itself

    top_indices = sim_scores.argsort()[::-1][:top_n]
    recommended = [newdf.iloc[i]['song'] for i in top_indices]
    return recommended


import streamlit as st

st.header('Song Recommendation')

user_input = st.selectbox('Select a song:', newdf['song'].sort_values().unique())

if st.button('Recommend'):
    res = rec(user_input)
    st.subheader('Results:')
    for i in res:
        st.write(i)
