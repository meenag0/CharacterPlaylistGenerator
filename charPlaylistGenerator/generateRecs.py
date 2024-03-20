# Import necessary libraries
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from songRecommender import SongRecommender  
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET


df = pd.read_csv('/Users/meenakshigopalakrishnan/CharacterPlaylistGenerator/charPlaylistGenerator/complete_df.csv')

# Initialize Spotipy client
client_credentials_manager = SpotifyClientCredentials(client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

song_recommender = SongRecommender(sp, df)

# Preprocess data and train model
song_recommender.preprocess_content()
song_recommender.train_content_model()

# Test recommendations for a character
character_name = 'Your_Character_Name'
num_recommendations = 5
recommendations = song_recommender.recommend(character_name, num_recommendations)

# Print recommendations
print(f'Recommendations for {character_name}:')
for i, recommendation in enumerate(recommendations, start=1):
    print(f'{i}. {recommendation}')

print(df.dtypes)
