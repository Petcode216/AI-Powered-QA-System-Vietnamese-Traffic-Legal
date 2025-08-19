import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ==== Config ====
MODEL_PATH   = "models/retrieval/phobert_atgt"
CORPUS_PATH  = "data/processed/retrieval/law_corpus_atgt.txt"       # One law clause per line
TOP_K        = 1
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==== Load model ====
model = SentenceTransformer(MODEL_PATH, device=DEVICE)
print("✅ Model loaded.")

# ==== Load law corpus ====
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    law_clauses = [line.strip() for line in f if line.strip()]

# ==== Encode law corpus (can cache this if needed) ====
print("🔄 Encoding law corpus...")
law_embeddings = model.encode(law_clauses, convert_to_tensor=True, show_progress_bar=True)

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

    # Top-K results
    top_results = torch.topk(similarities, k=TOP_K)

    print("\n📄 Top kết quả liên quan nhất:")
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"\n[{score:.4f}] {law_clauses[idx]}")
