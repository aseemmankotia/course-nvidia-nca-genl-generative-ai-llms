# NCA-GENL Exam Simulation - Quick Reference

## Exam Format
| Parameter | Value |
|-----------|-------|
| Questions | 50-51 multiple-choice |
| Duration | 60 minutes |
| Format | Online remote-proctored |
| Passing Score | ~70% (~36 correct) |
| Time per Question | 1.2 min average (aim for 1 min) |

## Time Management Strategy
1. **First Pass (40 min):** Answer all questions you know immediately (~1 min each)
2. **Flag Hard Questions:** If >90 seconds, flag and skip (come back later)
3. **Second Pass (15 min):** Tackle flagged questions with banked time
4. **Final Review (5 min):** Review flagged, check for misreads

**Goal:** Bank 10-15 minutes by answering easy questions quickly

## Wrong Answer Elimination
1. Cross out obviously wrong (violates requirements, wrong service category)
2. Eliminate over-complicated solutions when simple option exists
3. Eliminate options that miss stated requirements
4. Choose from remaining using cost/simplicity/best-practices criteria

## Common Exam Traps
| Trap | How to Avoid |
|------|-------------|
| Decoder-only vs encoder-decoder | Decoder for generation, encoder-decoder for translation |
| RLHF vs supervised fine-tuning | RLHF = 3-step process with reward model; SFT = direct demo learning |
| Temperature vs top-p | Temperature = standard diversity control; top-p = alternative sampling |
| NeMo vs NIM | NeMo = training framework; NIM = deployment microservices |
| BLEU vs ROUGE | BLEU = translation (precision); ROUGE = summarization (recall) |
| ASR vs TTS | ASR = speech→text; TTS = text→speech |
| Overfitting vs underfitting | Train↓ Val↑ = overfit; Train↑ Val↑ = underfit |

## Domain Quick Review
### Generative AI Fundamentals (20-25%)
- Transformer architecture: self-attention, encoder/decoder types
- Autoregressive vs autoencoding models
- Pre-training vs fine-tuning phases

### NVIDIA AI Platform (20-25%)
- NeMo Framework: training, guardrails, customization
- NIM: deployment, inference optimization
- Riva: speech (ASR/TTS)
- NeMo Retriever: RAG at scale

### Model Training (20-25%)
- RLHF sequence: pre-train → SFT → reward model → PPO
- Overfitting signals: train/val loss divergence
- Hyperparameters: learning rate, batch size, epochs

### Inference & Deployment (15-20%)
- Temperature: 0.1-0.3 (focused) to 0.8-1.0 (creative)
- Quantization: INT8/INT4 for efficiency
- Batching strategies for throughput

### Evaluation (10-15%)
- BLEU: n-gram precision (translation)
- ROUGE: recall-based (summarization)
- Perplexity: language model quality

## Mental Preparation
- **Read Carefully:** Watch for NOT, BEST, MOST, LEAST
- **Trust Your Prep:** You've done the labs, you know this
- **Don't Second-Guess:** First instinct usually correct
- **Stay Calm:** 60 min is enough if you don't panic

## Test Day Checklist
- [ ] Government ID ready
- [ ] Quiet room, clean desk
- [ ] Stable internet connection
- [ ] Close all other programs
- [ ] Bathroom break before starting
- [ ] Water bottle (clear, no label)

**Remember:** This exam tests practical knowledge. If you've done the labs and understand the concepts, you're ready.
