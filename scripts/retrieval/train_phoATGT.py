import os
import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from transformers import AutoTokenizer

# ==== Config ====
DATA_PATH   = "data/processed/retrieval/data_atgt.json"
OUTPUT_DIR  = "models/retrieval/phobert_atgt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 16
NUM_EPOCHS  = 3
LR          = 2e-5
MAX_SEQ_LEN = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Debug CUDA indexing issues

# ==== Special Tokens ====
special_tokens = (
    [f"[ART_{i}]" for i in range(1, 90)] +
    [f"[CL_{i}]"  for i in range(1, 11)] +
    [f"[PT_{c}]" for c in list("abcde")]
)

# ==== Load data ====
def load_pair_examples(path):
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)
    examples = []
    for rec in raw_data:
        anchor = str(rec.get("anchor", "")).strip()
        positive = str(rec.get("positive", "")).strip()
        if anchor and positive:
            examples.append(InputExample(texts=[anchor, positive]))
    print(f"✅ Loaded {len(examples)} training pairs")
    return examples

train_examples = load_pair_examples(DATA_PATH)
if not train_examples:
    raise ValueError("❌ No training data found.")

# ==== Load PhoBERT tokenizer and model ====
print("🔧 Loading PhoBERT tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# Khởi tạo transformer model (KHÔNG truyền tokenizer vào đây)
word_embedding_model = models.Transformer(
    "vinai/phobert-base",
    tokenizer_args={"additional_special_tokens": special_tokens},
    max_seq_length=MAX_SEQ_LEN
)

# Resize embedding để hỗ trợ special tokens
word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model],
    device=DEVICE
)

# ==== DataLoader & Loss ====
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# ==== Train ====
print("🚀 Training started...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=int(len(train_dataloader) * 0.1),
    optimizer_params={'lr': LR},
    output_path=OUTPUT_DIR,
    show_progress_bar=True
)

print(f"✅ Training complete. Model saved to {OUTPUT_DIR}")
