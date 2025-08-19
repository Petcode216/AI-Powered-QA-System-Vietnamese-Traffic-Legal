import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load the dataset
with open('data/vit5_traffic_qa_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_dict({
    'input': [item['input'] for item in data],  # Removed ['data']
    'target': [item['target'] for item in data]  # Removed ['data']
})

# Split into training and validation sets
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Load pre-trained viT5 model and tokenizer
model_name = "VietAI/vit5-base"  # Confirm the exact model name
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False) #Silence warning
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to NVIDIA GPU (cuda:0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512)
    outputs = tokenizer(examples['target'], padding='max_length', truncation=True, max_length=128)
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='models/generation/vit5_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,  # Enable mixed precision training
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./models/generation/vit5_finetuned/logs',
    logging_steps=10,
    eval_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('models/generation/vit5_finetuned')
tokenizer.save_pretrained('models/generation/vit5_finetuned')