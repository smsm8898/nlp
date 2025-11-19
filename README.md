## NLP
- Understand Natural Language Process from `Transfomers` to `LLM`
- Implement architeure from scratch
- Pretrain models on benchmark dataset

## Setup
```bash
git clone https://github.com/smsm8898/nlp.git
cd nlp
uv sync
```

## Dataset
- transformer: 

## Progress
- [x] Transfomers
- [x] BERT
- [x] GPT-1
- [x] GPT-2

## 주요 논문 타임라인

| Year | Paper | Model | Key Contribution | Technical Innovation | Challenge/Task |
|------|-------|-------|------------------|---------------------|----------------|
| 2017 | Attention is All You Need | Transformer | Self-attention으로 RNN/CNN 완전 대체 | Multi-head attention, Positional encoding | Machine Translation (WMT) |
| 2018 | BERT | BERT | 양방향 사전학습으로 NLU 혁신 | Masked Language Modeling, Next Sentence Prediction | GLUE, SQuAD |
| 2018 | Improving Language Understanding by Generative Pre-Training | GPT-1 | 생성적 사전학습 + 파인튜닝 프레임워크 | Transformer decoder로 언어 모델링 | Multiple NLU tasks |
| 2019 | Language Models are Unsupervised Multitask Learners | GPT-2 | 대규모 모델의 zero-shot 능력 입증 | 스케일업 (1.5B params), Byte-level BPE | Zero-shot task transfer |
| 2019 | ALBERT | ALBERT | 파라미터 효율적 BERT | Factorized embedding, Cross-layer sharing | GLUE, SQuAD, RACE |
| 2019 | XLNet | XLNet | BERT의 마스킹 단점 개선 | Permutation Language Modeling, Two-stream attention | GLUE, SQuAD, RACE |
| 2019 | RoBERTa | RoBERTa | BERT 학습 최적화 전략 | Dynamic masking, 더 큰 배치/데이터 | GLUE, SQuAD |
| 2020 | Language Models are Few-Shot Learners | GPT-3 | Few-shot learning 패러다임 제시 | 175B params, In-context learning | Few-shot generalization |
| 2020 | Exploring the Limits of Transfer Learning | T5 | 모든 NLP 태스크를 Text-to-Text로 통합 | Unified text-to-text format, C4 dataset | Multi-task learning |
| 2020 | Longformer | Longformer | 긴 문서 처리 가능 | Sparse attention patterns (sliding window + global) | Long document understanding |
| 2020 | Reformer | Reformer | 메모리 효율적 Transformer | Locality-sensitive hashing, Reversible layers | Long sequence efficiency |
| 2020 | DeBERTa | DeBERTa | Attention 메커니즘 개선 | Disentangled attention, Enhanced mask decoder | SuperGLUE |
| 2021 | Switch Transformers | Switch Transformer | 희소 모델로 효율적 스케일업 | Mixture of Experts, Sparse routing | Efficient scaling to 1T+ params |
| 2021 | LoRA | LoRA | 효율적 파인튜닝 방법 | Low-rank adaptation matrices | Parameter-efficient fine-tuning |
| 2022 | Training language models to follow instructions | InstructGPT | 인간 선호도 정렬 | RLHF (Reinforcement Learning from Human Feedback) | Instruction following |
| 2022 | Scaling Instruction-Finetuned Language Models | FLAN-T5 | instruction tuning의 효과 입증 | Multi-task instruction finetuning | Zero-shot task generalization |
| 2022 | Constitutional AI | Claude | AI 안전성 개선 | Self-critique and revision, RLAIF | Harmless and helpful AI |
| 2023 | LLaMA | LLaMA | 오픈소스 고성능 LLM | 7B-65B models, 효율적 학습 | Open research foundation |
| 2023 | GPT-4 Technical Report | GPT-4 | 멀티모달 LLM | Image + text input, 향상된 reasoning | Multimodal understanding |
| 2024 | Mixtral of Experts | Mixtral | 오픈소스 MoE 모델 | 8x7B Sparse MoE architecture | Efficient open-source LLM |


