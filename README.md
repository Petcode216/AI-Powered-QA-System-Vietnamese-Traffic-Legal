# 🚦 AI-powered Question Answering for Vietnamese Traffic Law Consultation

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)  
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Research--Capstone-success)  


This project develops an **AI-powered chatbot** that provides **legal consultation on Vietnamese traffic laws** using advanced Natural Language Processing (NLP).  
It integrates **PhoBERT** for legal text retrieval and **viT5** for natural answer generation, forming a **two-stage Question Answering (QA) pipeline**.  

The goal is to make complex legal information **accessible, accurate, and user-friendly**, helping the public understand traffic regulations and promoting road safety.

---

## 📌 Project Overview
- **Domain**: Vietnamese Traffic Law  
- **Models Used**:
  - **PhoBERT** (bi-encoder): Extracts and retrieves relevant law clauses.  
  - **viT5** (seq2seq): Generates natural, simplified Vietnamese answers.  
- **Interface**: A Streamlit-based chatbot application.  
- **Core Contribution**: A two-stage QA pipeline tailored for low-resource Vietnamese legal NLP.

---

## 🎯 Objectives
1. **Accurate & Contextual Responses**  
   Fine-tune PhoBERT and viT5 for precise retrieval and fluent answer generation.  

2. **Enhanced Public Understanding**  
   Translate complex legal jargon into concise, clear Vietnamese.  

3. **Efficiency for Legal Professionals**  
   Automate common inquiries, reducing workload for lawyers.  

---

## ⚖️ Problem Statement
Vietnamese traffic law is written in **complex, domain-specific legal language**, making it difficult for the general public to understand.  
Challenges include:
- Limited annotated datasets for Vietnamese law.  
- Nuances in Vietnamese language (tones, specialized terms).  
- Need for transforming technical clauses into **user-friendly answers**.  

---

## 🔧 System Architecture
### Two-stage QA Pipeline
1. **Retrieval (PhoBERT Bi-Encoder)**
   - Encodes queries & legal texts into embeddings.
   - Retrieves the most relevant law snippet.
   - Supports dual models:
     - `phobert-traffic` (general traffic law)
     - `phobert-168` (fine-related queries)

2. **Generation (viT5 Seq2Seq)**
   - Input: user query + retrieved law context.  
   - Output: fluent Vietnamese answer grounded in legal text.  

3. **UI Integration**
   - Built with **Streamlit**.
   - Displays both generated answer + referenced law (Điều, Khoản, Điểm).

---

## 📊 Results
### Retrieval (PhoBERT)
| Model         | Hit@1 | Hit@3 | Hit@5 |
|---------------|-------|-------|-------|
| PhoBERT-168   | 0.8279 | 0.9827 | 0.9872 |
| PhoBERT-ATGT  | 0.5791 | 0.9351 | 0.9492 |

### Generation (viT5)
| Metric   | Score |
|----------|-------|
| BLEU     | 0.825 |
| ROUGE-1  | 0.913 |
| ROUGE-2  | 0.874 |
| ROUGE-L  | 0.890 |

✔️ Generated answers are legally accurate, fluent, and easy to understand.  

---

## 🛠️ Tech Stack
- **Language**: Python 3.11  
- **Frameworks**: PyTorch, Hugging Face Transformers  
- **Models**: PhoBERT, VinAI/viT5  
- **Interface**: Streamlit  
- **Data Handling**: pandas, NumPy, JSON  
- **Evaluation**: BLEU, ROUGE, Hit@K, MRR@K  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/traffic-law-chatbot.git
   cd traffic-law-chatbot

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run training:
    ```bash
    a. Retrival
    python scripts/train_pho168c.py
    python scripts/train_phoATGT.py

    b.Generation
    python scripts/train_vit5.py

4. Launch the chatbot
    ```bash
    streamlit run app.py

---

⚠️ Limitations

Restricted to Vietnamese Road Traffic Law.

Does not provide personalized legal advice.

Accuracy depends on dataset quality.

Currently supports Vietnamese only.

---

🔮 Future Work

Extend coverage to other legal domains (criminal, civil, labor law).

Enhance query routing with intent classification.

Deploy cross-platform (Zalo, Messenger, web).

Explore reinforcement learning from human feedback (RLHF) for answer optimization.

Semi-automated legal data updates for law changes.

---

📖 References

PhoBERT: Pre-trained Language Models for Vietnamese

ViT5: Vietnamese Text-to-Text Transformer

VNLawBERT: Vietnamese Legal Answer Selection (Chau et al., 2020)

ViLQA: Legal Question Answering in Vietnamese (ntphuc149/ViLQA)
