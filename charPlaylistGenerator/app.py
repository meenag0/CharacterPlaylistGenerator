import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from ast import literal_eval
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import CLIENT_ID, CLIENT_SECRET

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load data and preprocess
@st.cache_data
def load_data():
    # Load movie and song data
    mov_df = pd.read_csv('movieEmotion.csv')
    song_df = pd.read_csv('songEmotion.csv')
    
    return mov_df, song_df

def convert_to_dict(emotion_str):
    return literal_eval(emotion_str)

# calculate emotion vector
def calculate_emotion_vector(emotion_dict):
    return np.array(list(emotion_dict.values()))

def convert_to_list(emotion_str):
    return literal_eval(emotion_str)


def regenerate_songs():
    random.seed(42)  # for reproducibility
    return top_200_songs.sample(n=15)[['song', 'artist']]


def get_spotify_data(song_name, artist_name):
    query = f"{song_name} {artist_name}"
    results = sp.search(q=query, type='track', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        return track['external_urls']['spotify'], track['album']['images'][0]['url']
    else:
        return None, None

# Main function to run the app
def main():
    st.title('cine.fm')
    
    # Load data
    mov_df, song_df = load_data()

    mov_df['Emotion Vector'] = mov_df['emotion_metrics'].apply(convert_to_dict).apply(calculate_emotion_vector)
    song_df['Emotion Vector'] = song_df['emotion_metrics'].apply(convert_to_dict).apply(calculate_emotion_vector)


    selected_movie = st.selectbox('Select a movie:', mov_df['Title'])

    selected_movie_vector = mov_df.loc[mov_df['Title'] == selected_movie, 'Emotion Vector'].values[0]

    # Convert the selected movie's emotion vector to a sparse matrix
    selected_movie_vector_sparse = csr_matrix(selected_movie_vector)

    # Convert emotion vectors of songs to a sparse matrix
    songs_sparse_matrix = csr_matrix(song_df['Emotion Vector'].to_list())

    # Calculate cosine similarity using sparse matrix operations
    cosine_similarities = cosine_similarity(selected_movie_vector_sparse, songs_sparse_matrix)

    # Extract top 200 similar songs
    top_200_indices = np.argsort(cosine_similarities[0])[::-1][:200]
    global top_200_songs
    top_200_songs = song_df.iloc[top_200_indices]

    # Display Refresh icon
    refresh_col, select_col = st.columns([1, 10])
    with refresh_col:
        if st.button("↻", help="Click to get a new set of songs", key="refresh"):
            rec_songs = regenerate_songs()

    # Display recommended songs as tiles
    with select_col:
        st.subheader('Recommended Songs:')
        rec_songs = regenerate_songs()  # Initially display random songs
        for index, row in rec_songs.iterrows():
            spotify_link, cover_url = get_spotify_data(row["song"], row["artist"])
            st.write(
                f"""
                <div style='display: flex; align-items: center; padding: 10px; border-bottom: 1px solid lightgray;'>
                    <img src='{cover_url}' style='width: 100px; height: 100px; object-fit: cover; border-radius: 5px;'>
                    <div style='margin-left: 20px;'>
                        <h3 style='margin: 0; padding: 0;'>{row["song"]}</h3>
                        <p style='margin: 0; padding: 0; font-size: 14px; color: gray;'>by {row["artist"]}</p>
                        <a href="{spotify_link}" target="_blank">Listen on Spotify</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == '__main__':
    main()