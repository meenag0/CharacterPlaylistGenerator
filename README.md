# Cine.fm

This Python project generates personalized playlists based on movies and songs. It uses sentiment analysis to score song lyrics and movie descriptions and intelligently curates a unique playlist for each user.

## Features
- **Movie and Music Playlists**: Generates personalized playlists based on the scores obtained from lyrics and movie descriptions.
- **Sentiment Analysis**: Utilizes sentiment analysis to evaluate lyrics and descriptions for better playlist recommendations.

## Technology Stack
- **Language**: Python
- **Tools/Libraries**: Jupyter Notebook, Pandas, scikit-learn, Numpy, NLTK, Tensorflow, Streamlit, Spotipy, Genius API, OMNI API, Spotify API, Hugging Face Transformers

## Project Files
- **`app.py`**: Main application file handling the backend logic. Integrates with Spotify and OMDB APIs to fetch data, uses sentiment scores to generate playlists, and provides a user interface with Streamlit.
- **`moviePlaylist.ipynb`**: Manages and generates movie-based playlists. It likely includes interactive features for selecting movies and visualizations of movie data.
- **`sentAnalysis.ipynb`**: Analyzes the sentiment of lyrics and movie descriptions. Includes preprocessing of textual data, sentiment scoring, and visual interpretation of results.
