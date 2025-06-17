from colpali_engine.models import ColQwen2, ColQwen2Processor
import torch
from transformers.utils.import_utils import is_flash_attn_2_available

## load ColQwen2 model + processor
model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(model_name)