import torch
from sentence_transformers import SentenceTransformer, util
import json
import random
from tqdm import tqdm

# ==== Load Trained Model ====
model = SentenceTransformer("models/retrieval/phobert-168")

# ==== Load data ====
with open("data/processed/retrieval/data_168.json", encoding="utf-8") as f:
    all_data = json.load(f)

# ==== Prepare corpus and queries ====
corpus_texts = list({item["positive"].strip() for item in all_data})  # unique positive responses
queries = [item["anchor"].strip() for item in all_data]
true_answers = [item["positive"].strip() for item in all_data]

# ==== Encode ====
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)

# ==== Evaluation ====
def evaluate_hit_mrr_k(k=10):
    hits = 0
    mrr_total = 0

    for i, query_emb in enumerate(tqdm(query_embeddings, desc=f"Evaluating @k={k}")):
        scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        top_k_idx = torch.topk(scores, k=k).indices.tolist()

        # Find the true positive index in corpus
        try:
            true_idx = corpus_texts.index(true_answers[i])
        except ValueError:
            continue  # skip if true answer not in corpus

        if true_idx in top_k_idx:
            hits += 1
            rank = top_k_idx.index(true_idx) + 1  # ranks start from 1
            mrr_total += 1 / rank

    hit_at_k = hits / len(queries)
    mrr_at_k = mrr_total / len(queries)

    return hit_at_k, mrr_at_k

# ==== Run for multiple K ====
for k in [1, 3, 5, 10]:
    hit, mrr = evaluate_hit_mrr_k(k=k)
    print(f"🎯 Hit@{k}: {hit:.4f} | MRR@{k}: {mrr:.4f}")
