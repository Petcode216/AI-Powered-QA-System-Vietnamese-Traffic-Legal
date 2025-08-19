import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# ==== Config ====
DATA_PATH   = "data/processed/retrieval/data_168.json"
OUTPUT_DIR  = "models/retrieval/phobert-168"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 16
NUM_EPOCHS  = 3
LR          = 2e-5

# ==== Load data ====
def load_pair_examples(path):
    with open(path, encoding="utf8") as f:
        data = json.load(f)

    examples = []
    for rec in data:
        anchor = str(rec.get("anchor", "")).strip()
        positive = str(rec.get("positive", "")).strip()
        if anchor and positive:
            examples.append(InputExample(texts=[anchor, positive]))

    print(f"✅ Loaded {len(examples)} anchor-positive pairs")
    return examples

train_examples = load_pair_examples(DATA_PATH)
if len(train_examples) == 0:
    raise ValueError("❌ No training examples found. Check your input file.")

# ==== PhoBERT base with special tokens ====
special_tokens = (
    [f"[ART_{i}]" for i in range(1,90)] +
    [f"[CL_{i}]"  for i in range(1,11)] +
    [f"[PT_{c}]"  for c in list("abcde")] +
    [f"[AM_{i*100000}_{(i+1)*100000}]" for i in range(20)] +
    ["[AM_NA]"]
)

word_emb = models.Transformer(
    "vinai/phobert-base",
    tokenizer_args={"additional_special_tokens": special_tokens}
)
pooling = models.Pooling(word_emb.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
model = SentenceTransformer(modules=[word_emb, pooling], device=DEVICE)

# ✅ Resize embeddings để hỗ trợ special tokens
model[0].auto_model.resize_token_embeddings(len(model[0].tokenizer))

# ==== DataLoader & Loss ====
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

train_loss = losses.MultipleNegativesRankingLoss(model=model)

# ==== Train ====
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=int(len(train_dataloader) * 0.1),
    optimizer_params={'lr': LR},
    output_path=OUTPUT_DIR
)

print(f"✅ Model trained & saved to {OUTPUT_DIR}")
