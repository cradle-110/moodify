import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pymongo import MongoClient
import requests
import time
import random

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="c23563670ff943438fdc616383e9f0ea",
    client_secret="08e420d130d94312a20123663db0ec25",
    redirect_uri="http://localhost:5000/callback",
    scope="user-library-read"
))

client = MongoClient("mongodb://mongo:example@localhost:27017/")
db = client.raw_data
saved_tracks = db.saved_tracks

results = []
limit = 50
offset = 0

def retry_with_backoff(func, max_retries=5, base_delay=1.0, jitter=True):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Function failed after {max_retries} retries: {e}")
                return None
            delay = (2 ** attempt) * base_delay
            if jitter:
                delay += random.uniform(0, 0.5)
            print(f"Retrying in {delay:.2f} seconds after error: {e}")
            time.sleep(delay)

while True:
    print(f"Fetching songs {offset} to {offset + limit}")
    batch = sp.current_user_saved_tracks(limit=limit, offset=offset)
    saved_tracks_batch = batch['items']

    # artist info of each track's first listed artist
    artist_infos = sp.artists(map(
        lambda track: track['track']['artists'][0]['id'],
        saved_tracks_batch
    ))
    for index, track in enumerate(saved_tracks_batch):
        album_art_url = track['track']['album']['images'][0]['url'] if track['track']['album']['images'] else None

        def fetch_album_art():
            if not album_art_url:
                raise ValueError("No album art URL found")
            response = requests.get(album_art_url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code} for {album_art_url}")
            return response.content

        album_art_data = retry_with_backoff(fetch_album_art) if album_art_url else None

        track['album_art_data'] = album_art_data
        track['artist_info'] = artist_infos[index]

    results.extend(saved_tracks_batch)

    if len(batch['items']) < limit:
        break
    offset += limit


# Save raw "SavedTrackObject" to mongodb
saved_tracks.drop()
saved_tracks.insert_many(results)
