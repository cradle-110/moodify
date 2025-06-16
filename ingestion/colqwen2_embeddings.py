import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from pymongo import MongoClient
from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import uuid
from io import BytesIO
import numpy as np

## https://colab.research.google.com/github/qdrant/examples/blob/master/pdf-retrieval-at-scale/ColPali_ColQwen2_Tutorial.ipynb#scrollTo=o-fbK8jiR21K

## load mongo client
mongo = MongoClient("mongodb://mongo:example@localhost:27017/")
saved_tracks = mongo.raw_data.saved_tracks

## load qdrant client
qdrant = QdrantClient("http://localhost:6333")
collection_name = "colqwen2_embeddings"

## load ColQwen2 model + processor
model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(model_name)

## create qdrant collection if not exists
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "original": 
            models.VectorParams( #switch off HNSW
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=0 #switching off HNSW
                    )
            ),
        "mean_pooling_columns": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            ),
        "mean_pooling_rows": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
    }
)

## pull all saved tracks from mongo
track_data = [x for x in saved_tracks.find().limit(100)]

## iterate in batches and create embeddings
def upload_batch(original_batch, pooled_by_rows_batch, pooled_by_columns_batch, payload_batch, collection_name):
    try:
        qdrant.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
                "mean_pooling_rows": pooled_by_rows_batch
            },
            payload=payload_batch,
        )
    except Exception as e:
        print(f"Error during upsert: {e}")

def get_patches(image_size, model_processor, model):
    return model_processor.get_n_patches(
        image_size, 
        model.spatial_merge_size
    )

def embed_and_mean_pool_batch(image_batch, model_processor, model):
    #embed
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device) 
        image_embeddings = model(**processed_images)

    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()
    
    #mean pooling
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []
    
    
    for image_embedding, tokenized_image, image in zip(image_embeddings, 
                                                       processed_images.input_ids, 
                                                       image_batch):
        x_patches, y_patches = get_patches(image.size, model_processor, model)
        # print(f"model divided this PDF page in {x_patches} rows and {y_patches} columns")

        image_tokens_mask = (tokenized_image == model_processor.image_token_id)
        
        image_tokens = image_embedding[image_tokens_mask].view(x_patches, y_patches, model.dim)
        pooled_by_rows = torch.mean(image_tokens, dim=0)
        pooled_by_columns = torch.mean(image_tokens, dim=1)

        image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_idxs[0].cpu().item()
        last_image_token_idx = image_token_idxs[-1].cpu().item()
        
        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1:]
        
        # print(f"There are {len(prefix_tokens)} prefix tokens and {len(postfix_tokens)} in a PDF page embedding")

        #adding back prefix and postfix special tokens
        pooled_by_rows = torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0).cpu().float().numpy().tolist()
        pooled_by_columns = torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0).cpu().float().numpy().tolist()
        
        pooled_by_rows_batch.append(pooled_by_rows)
        pooled_by_columns_batch.append(pooled_by_columns)


    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch

batch_size = 10
with tqdm(total=len(track_data), desc=f"Uploading progress of document embeds to \"{collection_name}\" collection") as pbar:
    for i in range(0, len(track_data), batch_size):
        batch = track_data[i : i + batch_size]
        image_batch = [Image.open(BytesIO(saved_track['document'])).convert("RGB") for saved_track in batch]
        current_batch_size = len(image_batch)
        try:
            original_batch, pooled_by_rows_batch, pooled_by_columns_batch = embed_and_mean_pool_batch(
                image_batch, 
                processor, 
                model
            )
        except Exception as e:
            print(f"Error during embed: {e}")
            continue
        try:
            upload_batch(
                np.asarray(original_batch, dtype=np.float32),
                np.asarray(pooled_by_rows_batch, dtype=np.float32),
                np.asarray(pooled_by_columns_batch, dtype=np.float32),
                [
                    {
                        "track_id": saved_track['track']['id'],
                        "track_name": saved_track["track"]["name"],
                    }
                    for saved_track in batch
                ],
                collection_name
            )
        except Exception as e:
            print(f"Error during upsert: {e}")
            continue
        # Update the progress bar
        pbar.update(current_batch_size)
print("Uploading complete!")
