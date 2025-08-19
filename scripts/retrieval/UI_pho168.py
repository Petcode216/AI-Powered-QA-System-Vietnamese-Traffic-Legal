import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ==== Config ====
MODEL_PATH   = "models/retrieval/phobert-168"
CORPUS_PATH  = "data/processed/retrieval/law_corpus.txt"  # Updated path
OUTPUT_PATH  = "data/processed/retrieval/top_law.txt"     # Output file for top 1 law
TOP_K        = 1                                                # Changed to top 1 for pipeline
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Load model ====
model = SentenceTransformer(MODEL_PATH, device=DEVICE)
print("✅ Model loaded.")

# ==== Load law corpus ====
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    law_clauses = [line.strip() for line in f if line.strip()]

# ==== Encode law corpus (can cache this if needed) ====
print("🔄 Encoding law corpus...")
law_embeddings = model.encode(law_clauses, convert_to_tensor=True, show_progress_bar=True)

# ==== Format conversion function ====
def normalize_law_format(law_text):
    parts = law_text.split()
    if not parts:
        return law_text
    article = parts[0].replace("[ART_", "Điều ").replace("]", "")
    clause = parts[1].replace("[CL_", "Khoản ").replace("]", "")
    point = parts[2].replace("[PT_", "Điểm ").replace("]", "") if len(parts) > 2 else ""
    amount = parts[3].replace("[AM_", "").replace("]", "").split("_")
    amount_str = f"Phạt tiền từ {int(amount[0]):,d} đồng đến {int(amount[1]):,d} đồng" if len(amount) == 2 else ""
    description = " ".join(parts[4:]) if len(parts) > 4 else "Không có mô tả"
    return f"{article}, {clause}, {point}, Nghị định 168/2024/NĐ-CP: {amount_str} {description}".strip()

# ==== Inference loop ====
print("🚦 Ready. Type your question (or 'exit' to quit):")
while True:
    question = input("\n📥 Câu hỏi: ").strip()
    if question.lower() == "exit":
        print("👋 Thoát.")
        break
    if not question:
        continue

    # Encode question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(question_embedding, law_embeddings)[0]

    # Top-1 result
    top_result = torch.topk(similarities, k=TOP_K)
    top_law = law_clauses[top_result[1][0]]
    top_score = top_result[0][0].item()

    # Normalize and save top 1 law
    normalized_law = normalize_law_format(top_law)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(normalized_law)

    print(f"\n📄 Top 1 law (Score: {top_score:.4f}):")
    print(f"{normalized_law}")