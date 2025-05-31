import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import uuid
from pymongo import MongoClient
from io import BytesIO

client = MongoClient("mongodb://mongo:example@localhost:27017/")
db = client.raw_data
saved_tracks = db.saved_tracks

# Load CLIP model + processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# === Qdrant setup ===
qdrant = QdrantClient("http://localhost:6333")
collection_name = "clip_embeddings_separate"

# Create collection if not exists
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)  # Shared config
)

# Store embeddings
count = 0
for saved_track in saved_tracks.find():
    points = []
    
    if count % 100 == 0:
        print(f"on track {count}")

    text = f"{saved_track['track']['name']} by {saved_track['track']['artists'][0]['name']}"

    # Load image
    image = Image.open(BytesIO(saved_track['album_art_data'])).convert("RGB")

    # Prepare inputs
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Convert to list
    image_vector = outputs.image_embeds[0].cpu().numpy().tolist()
    text_vector = outputs.text_embeds[0].cpu().numpy().tolist()

    # Create image point
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=image_vector,
        payload={
            "type": "image",
            "track_id": saved_track['track']['id'],
            "track_name": saved_track["track"]["name"],
            "artist_name": saved_track["track"]["artists"][0]["name"]
        }
    ))

    # Create text point
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=text_vector,
        payload={
            "type": "text",
            "track_id": saved_track['track']['id'],
            "track_name": saved_track["track"]["name"],
            "artist_name": saved_track["track"]["artists"][0]["name"]
        }
    ))

    # Upload to Qdrant
    qdrant.upsert(collection_name=collection_name, points=points)
    count += 1

print(f"✅ Uploaded {len(saved_tracks) * 2} separate CLIP embeddings (text + image) to Qdrant")
