from colpali_engine.models import ColQwen2, ColQwen2Processor
import torch
from transformers.utils.import_utils import is_flash_attn_2_available
from qdrant_client import QdrantClient, models

from storage.qdrant import qdrant

## load ColQwen2 model + processor
model_name = "vidore/colqwen2-v1.0"
collection_name = "colqwen2_embeddings"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(model_name)

def reranking_search_batch(prompt,
                           search_limit=20,
                           prefetch_limit=200):
    inputs_text = processor.process_queries([prompt]).to(model.device)
    with torch.no_grad():
        query = model(**inputs_text).cpu().float().numpy().squeeze()
    return qdrant.query_points(
        collection_name=collection_name,
        query=query,
        prefetch=[
            models.Prefetch(
                query=query,
                limit=prefetch_limit,
                using="mean_pooling_columns"
            ),
            models.Prefetch(
                query=query,
                limit=prefetch_limit,
                using="mean_pooling_rows"
            ),
        ],
        limit=search_limit,
        with_payload=True,
        with_vectors=False,
        using="original"
    )
