import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import streamlit as st

# ==== Config ====
RETRIEVAL_168_MODEL_PATH = "models/retrieval/phobert-168"
RETRIEVAL_ATGT_MODEL_PATH = "models/retrieval/phobert_atgt"
GENERATION_MODEL_PATH = "models/generation/vit5_finetuned"
CORPUS_168_PATH = "data/processed/retrieval/law_corpus_168.txt"
CORPUS_ATGT_PATH = "data/processed/retrieval/law_corpus_atgt.txt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOP_K = 1

# ==== Load models (run once) ====
@st.cache_resource
def load_models():
    print("✅ Loading 168 retrieval model...")
    retrieval_model_168 = SentenceTransformer(RETRIEVAL_168_MODEL_PATH, device=DEVICE)
    print("✅ Loading ATGT retrieval model...")
    retrieval_model_atgt = SentenceTransformer(RETRIEVAL_ATGT_MODEL_PATH, device=DEVICE)
    print("✅ Loading generation model...")
    tokenizer = T5Tokenizer.from_pretrained(GENERATION_MODEL_PATH)
    generation_model = T5ForConditionalGeneration.from_pretrained(GENERATION_MODEL_PATH).to(DEVICE)
    return retrieval_model_168, retrieval_model_atgt, tokenizer, generation_model

@st.cache_data
def load_and_encode_corpus():
    print("🔄 Encoding 168 law corpus...")
    with open(CORPUS_168_PATH, "r", encoding="utf-8") as f:
        law_clauses_168 = [line.strip() for line in f if line.strip()]
    law_embeddings_168 = retrieval_model_168.encode(law_clauses_168, convert_to_tensor=True, show_progress_bar=True)

    print("🔄 Encoding ATGT law corpus...")
    with open(CORPUS_ATGT_PATH, "r", encoding="utf-8") as f:
        law_clauses_atgt = [line.strip() for line in f if line.strip()]
    law_embeddings_atgt = retrieval_model_atgt.encode(law_clauses_atgt, convert_to_tensor=True, show_progress_bar=True)
    return law_clauses_168, law_embeddings_168, law_clauses_atgt, law_embeddings_atgt

# ==== Load resources ====
retrieval_model_168, retrieval_model_atgt, tokenizer, generation_model = load_models()
law_clauses_168, law_embeddings_168, law_clauses_atgt, law_embeddings_atgt = load_and_encode_corpus()

# ==== Format conversion function ====
def normalize_law_format(law_text, is_168=True):
    parts = law_text.split()
    if not parts:
        return law_text
    article = parts[0].replace("[ART_", "Điều ").replace("]", "")
    clause = parts[1].replace("[CL_", "Khoản ").replace("]", "")
    point = parts[2].replace("[PT_", "Điểm ").replace("]", "") if len(parts) > 2 else ""
    
    if is_168:
        amount = parts[3].replace("[AM_", "").replace("]", "").split("_") if len(parts) > 3 else []
        amount_str = f"Phạt tiền từ {int(amount[0]):,d} đồng đến {int(amount[1]):,d} đồng" if len(amount) == 2 else ""
        description = " ".join(parts[4:]) if len(parts) > 4 else "Không có mô tả"
        law_ref = "Nghị định 168/2024/NĐ-CP"
    else:
        amount_str = ""
        description = " ".join(parts[3:]) if len(parts) > 3 else "Không có mô tả"
        law_ref = "Luật Giao thông Đường bộ (ATGT)"

    return f"{article}, {clause}, {point}, {law_ref}: {amount_str} {description}".strip()

# ==== Pipeline function ====
def get_answer(query):
    # Select model and corpus based on query
    #add 'hình phạt', 'vi phạm' for more spread
    if 'phạt' in query.lower() or 'hình phạt' in query.lower() or 'vi phạm' in query.lower():
        model = retrieval_model_168
        embeddings = law_embeddings_168
        clauses = law_clauses_168
        is_168_flag = True
    else:
        model = retrieval_model_atgt
        embeddings = law_embeddings_atgt
        clauses = law_clauses_atgt
        is_168_flag = False

    question_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, embeddings)[0]
    top_result = torch.topk(similarities, k=TOP_K)
    top_law = clauses[top_result[1][0]]
    normalized_law = normalize_law_format(top_law, is_168_flag)
    st.text(f"🔍 Retrieved Law Context: {normalized_law}")

    input_text = f"Câu hỏi: {query}\nLuật: {normalized_law}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    output_ids = generation_model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# ==== Streamlit UI ====
st.title("Hệ thống Hỏi Đáp Luật Giao Thông")
query = st.text_input("📥 Nhập câu hỏi của bạn:", "")
if st.button("Gửi"):
    if query:
        with st.spinner("Đang xử lý..."):
            answer = get_answer(query)
        st.text_area("📄 Trả lời:", value=answer, height=100)
    else:
        st.warning("Vui lòng nhập câu hỏi!")