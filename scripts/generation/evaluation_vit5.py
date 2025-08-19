import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset
from evaluate import load
import matplotlib.pyplot as plt

# Load the fine-tuned model and tokenizer
model_dir = "models/generation/vit5_finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load a subset of the dataset for evaluation (e.g., validation split)
with open('data/processed/generation/vit5_traffic_qa_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
dataset = Dataset.from_dict({
    'input': [item['input'] for item in data],
    'target': [item['target'] for item in data]
})
dataset = dataset.train_test_split(test_size=0.1, seed=42)['test']  # Use validation set

# Tokenize the dataset, preserving 'target'
def tokenize_function(examples):
    inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    return {**inputs, 'target': examples['target']}  # Preserve target

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'target'])  # Include target

# Generate predictions
predictions = []
references = []
model.eval()
with torch.no_grad():
    for item in tokenized_dataset:
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(device)
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=128, num_beams=5, early_stopping=True)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reference = item['target']  # Now accessible
        predictions.append(prediction)
        references.append(reference)

# Compute BLEU score
bleu = load("bleu")
bleu_results = bleu.compute(predictions=predictions, references=references)
bleu_score = bleu_results['bleu']
print(f"BLEU Score: {bleu_score}")

# Compute ROUGE score
rouge = load("rouge")
rouge_results = rouge.compute(predictions=predictions, references=references)
rouge1 = rouge_results['rouge1']
rouge2 = rouge_results['rouge2']
rougeL = rouge_results['rougeL']
print(f"ROUGE-1 Score: {rouge1}")
print(f"ROUGE-2 Score: {rouge2}")
print(f"ROUGE-L Score: {rougeL}")

# Visualization: Bar chart comparing BLEU and ROUGE scores
metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
scores = [bleu_score, rouge1, rouge2, rougeL]

plt.figure(figsize=(8, 5))
plt.bar(metrics, scores, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)  # Assuming scores are between 0 and 1
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.show()

# Optional: Print some examples
for i in range(min(5, len(predictions))):
    print(f"Input: {dataset[i]['input']}")
    print(f"Predicted: {predictions[i]}")
    print(f"Target: {references[i]}")
    print("---")