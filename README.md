# ASAG — Automatic Short Answer Grading

Automatic grading of student short answers using fine-tuned BERT + BiLSTM + Multi-Head Attention.

Built as part of NLP course project. Reimplemented from [wuhan-1222/ASAG](https://github.com/wuhan-1222/ASAG) using HuggingFace Transformers + PyTorch.

---

## Model Architecture
```
Input: [CLS] Reference Answer [SEP] Student Answer [SEP]
         ↓
    BERT (bert-base-uncased)
    Layers 0-7 → Frozen
    Layers 8-11 → Fine-tuned
         ↓
    BiLSTM (2 layers, 256 units, bidirectional)
         ↓
    Multi-Head Attention (4 heads, 512 dim)
         ↓
    Mean + Max Pooling
         ↓
    Dropout(0.3) + Linear Classifier
         ↓
    11 Score Classes (0.0 to 5.0)
```

---

## Dataset

- **NorthTexas Dataset** — expand.txt (~3800 samples)
- **Format:** question | reference_answer | student_answer | score
- **Score classes:** 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
- **Split:** 80% train / 20% test

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy (within-1-tolerance) | 0.81129 |
| Pearson Correlation | 0.77122 |
| MAE | 0.60744 |
| RMSE | 1.54412 |
| Macro F1 | 0.67202 |
| Weighted F1 | 0.80764 |

---

## Training Progress

| Epoch | Loss | Accuracy | Pearson |
|-------|------|----------|---------|
| 1 | 2.3193 | 0.5358 | 0.4283 |
| 4 | 1.5357 | 0.7865 | 0.7656 |
| 8 | 0.9427 | 0.8044 | 0.7708 |
| 10 | 0.7997 | 0.8113 | 0.7712 |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| HuggingFace Transformers | BERT model and tokenizer |
| bert-base-uncased | Pre-trained language model |
| scikit-learn | Evaluation metrics |
| Google Colab T4 GPU | Training environment |

---

## How to Run

### Google Colab (Recommended)
1. Open `ASAG_notebook.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order

### Local Machine
```bash
git clone https://github.com/YOUR_USERNAME/ASAG-NLP.git
cd ASAG-NLP
pip install -r requirements.txt
python model.py
```

---

## Key Differences from Original Repo

| Aspect | Original | Ours |
|--------|----------|------|
| Framework | bert4keras + TensorFlow | HuggingFace + PyTorch |
| BERT layers | All 12 fine-tuned | 8 frozen + 4 fine-tuned |
| Capsule Network | Yes | Removed (NaN bug) |
| Optimizer | Adam flat lr | AdamW + warmup scheduler |
| Gradient clipping | None | max_norm=1.0 |
| Pooling | Max only | Mean + Max |

---

## Project Structure
```
ASAG-NLP/
├── model.py              # Complete model code
├── requirements.txt      # Dependencies
├── README.md             # This file
└── ASAG_notebook.ipynb   # Google Colab notebook
```
