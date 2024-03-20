from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

class SongRecommender:
    def __init__(self, sp, df):
        self.sp = sp  # Spotify API client
        self.df = df
        self.music_features = ['danceability_mean', 'energy_mean', 'loudness_mean', 'valence_mean', 'tempo_mean']
        self.genre_feature = 'Associated Music Genres'
        self.model_content = None
        self.content_matrix = None

    def preprocess_content(self):
        # Preprocess content features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.df[self.music_features])
        genre_matrix = pd.get_dummies(self.df[self.genre_feature])

        # Concatenate content features and genre features
        self.content_matrix = pd.concat([pd.DataFrame(scaled_features), genre_matrix], axis=1)

    def train_content_model(self):
        # Train content-based model (NMF for feature extraction)
        self.model_content = NMF(n_components=10)  # Adjust the number of components as needed
        self.model_content.fit(self.content_matrix)

    def recommend(self, character_name, num_recommendations=5):
        # Find music features and associated genres for the given character
        character_data = self.df[self.df['character'] == character_name].iloc[0]
        danceability = character_data['danceability_mean']
        energy = character_data['energy_mean']
        loudness = character_data['loudness_mean']
        valence = character_data['valence_mean']
        tempo = character_data['tempo_mean']
        genres = character_data['Associated Music Genres']

        # Transform user input into latent features
        user_latent_features = self.model_content.transform([[danceability, energy, loudness, valence, tempo] + [0] * len(genres.split(','))])

        # Calculate cosine similarity between user input and songs in the dataset
        similarities = cosine_similarity(user_latent_features, self.model_content.components_)[0]

        # Get indices of top recommendations
        top_indices = similarities.argsort()[-num_recommendations:][::-1]

        # Get song IDs corresponding to the top indices
        song_ids = self.df.iloc[top_indices]['song_id'].tolist()

        # Get song details from Spotify API
        recommendations = []
        for song_id in song_ids:
            track_info = self.sp.track(song_id)
            song_name = track_info['name']
            artist_name = track_info['artists'][0]['name']
            recommendations.append(f"{song_name} by {artist_name}")

        return recommendations