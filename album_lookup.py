from transformers import CLIPProcessor, CLIPModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest, Filter, PointStruct, FieldCondition, MatchValue
import numpy as np

# --- Setup ---
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Qdrant client ---
client = QdrantClient(host="localhost", port=6333)
collection_name = "clip_embeddings_separate"  # Or whatever you used when saving

# --- Your prompt ---
prompt = "manga"

# --- Create embedding ---
inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    outputs = model.get_text_features(**inputs)
embedding = outputs[0].cpu().numpy().tolist()  # Convert to list for Qdrant

# --- Perform search ---
search_result = client.search(
    collection_name=collection_name,
    query_vector=embedding,
    limit=5  # Return top 5 matches
)

# --- Print results ---
for result in search_result:
    print(f"Score: {result.score:.4f}")
    print(f"Payload: {result.payload}")
