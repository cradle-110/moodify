from pymongo import MongoClient
from PIL import Image
from io import BytesIO

mongo = MongoClient("mongodb://mongo:example@localhost:27017/")
saved_tracks = mongo.raw_data.saved_tracks

def get_track_image(track_id):
    saved_track = saved_tracks.find_one({"track.id": track_id})
    return Image.open(BytesIO(saved_track['album_art_data'])).convert("RGB")

def get_track_document(track_id):
    saved_track = saved_tracks.find_one({"track.id": track_id})
    return Image.open(BytesIO(saved_track['document'])).convert("RGB")