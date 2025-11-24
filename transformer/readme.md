# Transformer Implementation from Scratch
- A PyTorch implementation of the Transformer model from the paper
  - [Attention is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - [Paper Review](https://velog.io/@smsm8898/Paper-Review-Attention-is-all-you-need)

- This project implements the complete architecture and trains it on the WMT14 English-German translation task
  - [WMT14 Dataset](https://huggingface.co/datasets/wmt/wmt14)

## 🚀 Quick Start
- Caution: You need to check where the dataset(tokenizer) will be download and preprocess
1. Prepare WMT14 Dataset(BPE Tokenizer)
```
run preprocess.ipynb
```
2. Train (Base)Transformer
```bash
python -m transformer.train 
or 
./run.sh base
```


## 🎯 Project Goals
This project aims to:

- Deeply understand the Transformer architecture by implementing it from scratch
- Reproduce the original paper's results on machine translation
Experiment with training techniques and hyperparameters
- Build a foundation for studying modern NLP models (BERT, GPT, T5, etc.)


## 📋 Progress
- [x] Review transformer paper
- [x] Train BPE tokenizer
- [x] Implement transformer architecture
- [] Train (tiny) transformer on WMT14 English-German
  - In MPS, it takes too much time to train Transformer from scratch


## 📊 Dataset
**WMT14 English-German Translation**
- `dataset.py`
- Training: ~4.5M sentence pairs
- Validation: ~3K sentence pairs
- Test: ~3K sentence pairs
- Tokenization: SentencePiece (BPE with 32K vocabulary)


## 🏗️ Architecture
**Implementation Details**
- `model.py`
- Multi-Head Attention: Scaled dot-product attention with 8 heads
- Positional Encoding: Sinusoidal position embeddings
- Encoder: 6 identical layers with self-attention and feed-forward networks
- Decoder: 6 identical layers with masked self-attention, cross-attention, and feed-forward networks
- Layer Normalization: Applied after each sub-layer with residual connections
- Label Smoothing: ε = 0.1 for regularization

## ❓
- [] BPE
- [] LabelSmoothing
- [] LayerNorm
- [] Dropout
- [] noam learning rate scheduler
- [] Hardware: precision, gradient_clip