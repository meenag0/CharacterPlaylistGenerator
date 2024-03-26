import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from ast import literal_eval


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


# Main function to run the app
def main():
    st.title('Song Recommendation App')
    
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
    top_200_songs = song_df.iloc[top_200_indices]

    # Randomly select 15 songs from the top 200
    random.seed(42)  # for reproducibility, remove this line if you want different results each time
    rec_songs = top_200_songs.sample(n=15)[['song', 'artist']]

    # User input

    # Display recommended songs
    st.subheader('Recommended Songs:')
    st.write(rec_songs)

if __name__ == '__main__':
    main()