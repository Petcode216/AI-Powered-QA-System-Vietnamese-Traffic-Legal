import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model and tokenizer
model_dir = "models/generation/vit5_finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_answer(query):
    # Read the top 1 law from the retrieval output
    try:
        with open("data/processed/retrieval/top_law.txt", "r", encoding="utf-8") as f:
            law_context = f.read().strip()
    except FileNotFoundError:
        return "Xin lỗi, không tìm thấy luật áp dụng. Vui lòng chạy retrieval trước."

    # Construct input in the format used during training
    input_text = f"Câu hỏi: {query}\nLuật: {law_context}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# Inference loop
while True:
    query = input("\n📥 Câu hỏi: ").strip()
    if query.lower() == "exit":
        print("👋 Thoát.")
        break
    if not query:
        continue

    answer = get_answer(query)
    print(f"📄 Trả lời: {answer}")