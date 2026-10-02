# Transformer Architecture & LLMs Cheat Sheet

## Core Architecture Components

| Component | Purpose | Exam Gotcha |
|-----------|---------|-------------|
| Self-Attention | Compute relationships between all token pairs in sequence | Without scaling by sqrt(d_k), softmax saturates (large dot products) |
| Multi-Head Attention | Parallel attention with different learned patterns | Each head dimension = embedding_dim ÷ num_heads |
| Positional Encoding | Inject sequence order information into embeddings | Without it, model is permutation-invariant (order-blind) |
| Feed-Forward Network | Apply non-linear transformations per-token independently | Typically 4× larger than embedding dimension |

## Attention Math (Heavily Tested)

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V

Where:
- Q (query): [batch, seq_len, d_k] - what token is looking for
- K (key): [batch, seq_len, d_k] - what token offers for matching  
- V (value): [batch, seq_len, d_v] - actual content to retrieve
- d_k: key dimension (must scale to prevent saturation)
```

**Multi-head example:** 12 heads, embedding_dim=768  
→ Each head operates on 768÷12 = 64 dimensions  
→ Outputs concatenated → 12×64 = 768 → final projection

## Architecture Patterns

| Architecture | Components | Use Case | Pretraining Objective |
|--------------|------------|----------|----------------------|
| Encoder-only (BERT) | Encoder stack only | Classification, embeddings, understanding | MLM (Masked Language Modeling) |
| Decoder-only (GPT) | Decoder stack only | Text generation, continuation | CLM (Causal Language Modeling) |
| Encoder-Decoder (T5) | Both stacks + cross-attention | Translation, summarization, Q&A | Span corruption (masked spans) |

**Exam decision tree:**  
- Need to generate text from scratch or continue text? → Decoder-only  
- Input differs from output (translate, summarize with context)? → Encoder-decoder  
- Classification or embedding task? → Encoder-only

## Context Windows & Limits

| Concept | Definition | Calculation |
|---------|------------|-------------|
| Context Window | Max total tokens (prompt + generation) | Fixed by architecture |
| Available Generation | Tokens left for new generation | context_window - prompt_length |
| Max Sequence Length | Longest input/output sequence supported | Determined by positional encoding scheme |
| Autoregressive Generation | Generate one token at a time | Each step uses previous tokens as context |

**Common exam scenario:**  
Context window = 4096, prompt = 4000 tokens  
→ Can generate: 4096 - 4000 = **96 tokens max**  
→ If prompt = 4100 tokens, rightmost 4096 kept (truncate left 4)

## Tokenization Methods

| Method | How It Works | Vocab Size | Handles Unknown |
|--------|--------------|------------|----------------|
| BPE (Byte-Pair Encoding) | Merge most frequent character pairs iteratively | 30k-50k typical | Decomposes to chars/bytes |
| WordPiece | Merge based on likelihood maximization | 30k-50k typical | Decomposes with ## prefix |
| SentencePiece | Treats text as raw byte stream (language-agnostic) | Configurable | Always decomposes (no OOV) |

**Exam tip:** All subword tokenizers compress text → fewer tokens than characters.  
Example: 1000 characters → ~250 tokens (4 chars/token avg) is normal.

## Pretraining Objectives

| Objective | How It Works | Model Type |
|-----------|--------------|------------|
| CLM (Causal LM) | Predict next token given previous tokens (left-to-right) | Decoder-only (GPT) |
| MLM (Masked LM) | Predict randomly masked tokens using bidirectional context | Encoder-only (BERT) |
| Span Corruption | Mask contiguous spans, predict them (T5 variant) | Encoder-decoder |
| Denoising | Corrupt input (shuffle, mask, delete), reconstruct original | Encoder-decoder (BART) |

**Key difference tested:**  
- CLM sees only past context (causal mask prevents seeing future)  
- MLM sees full bidirectional context (past + future)

## Common Exam Traps

1. **Positional Encoding Required:** Models without it cannot learn sequence order. Not optional.

2. **Context Window is Total:** Includes prompt + generation. Not separate budgets.

3. **Head Dimension Calculation:** Always embedding_dim ÷ num_heads. Not a fixed constant.

4. **Attention Scaling:** Divides by sqrt(d_k), not multiplies. Prevents saturation.

5. **Architecture Selection:** Encoder-decoder when input ≠ output context. Decoder-only for open-ended generation.

6. **Tokenization Compression:** Subword methods produce fewer tokens than characters (compression). More tokens than characters suggests character-level or broken tokenizer.

7. **Autoregressive = Sequential:** One token per step. Not parallel generation of full sequence.

8. **Truncation Direction:** Right-truncation (keep recent context) is standard for generation. Left-truncation loses immediate context.
