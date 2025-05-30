import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pymongo import MongoClient
import requests

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="744c0194864b419da63bde5738eab3f5",
    client_secret="d44736a41bc2499980fc8db322e6f9f6",
    redirect_uri="http://localhost:3000/callback",
    scope="user-library-read"
))

client = MongoClient("mongodb://mongo:example@localhost:27017/")
db = client.raw_data
saved_tracks = db.saved_tracks

results = []
limit = 50
offset = 0

while True:
    print(f"Fetching songs {offset} to {offset + limit}")
    batch = sp.current_user_saved_tracks(limit=limit, offset=offset)
    saved_tracks_batch = batch['items']
    for track in saved_tracks_batch:
        # add album art
        album_art_url = track['track']['album']['images'][0]['url'] if track['track']['album']['images'] else None
        if album_art_url:
            response = requests.get(album_art_url)
            if response.status_code == 200:
                track['album_art_data'] = response.content
            else:
                track['album_art_data'] = None
        else:
            track['album_art_data'] = None
    results.extend(saved_tracks_batch)
    if len(batch['items']) < limit:
        break
    offset += limit

# Save raw "SavedTrackObject" to mongodb
saved_tracks.drop()
saved_tracks.insert_many(results)
