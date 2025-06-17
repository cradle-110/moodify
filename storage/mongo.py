from pymongo import MongoClient

mongo = MongoClient("mongodb://mongo:example@localhost:27017/")
saved_tracks = mongo.raw_data.saved_tracks