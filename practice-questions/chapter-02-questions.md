## Chapter 2: Transformer Architecture and Large Language Models — Practice Questions

**Q1.** A company is building a question-answering system that must read a document context and generate a precise answer based on that context. The system should not generate answers beyond what is explicitly stated in the provided text. Which transformer architecture BEST fits this requirement?

A) Encoder-decoder architecture (T5-style) with context as encoder input and question as decoder prompt
B) Decoder-only architecture (GPT-style) with prompt engineering to include context
C) Encoder-only architecture (BERT-style) with classification head
D) Multi-head attention only without positional encoding to focus on context relationships

<details>
<summary>Answer</summary>

**Correct: A**

Encoder-decoder architectures like T5 are specifically designed for tasks where input context (encoder) differs from output generation (decoder). The encoder processes the full document context bidirectionally, while the decoder generates the answer autoregressively. This prevents hallucination beyond the provided context better than decoder-only models.

**Why the others are wrong:**
- Decoder-only models like GPT can work but are more prone to hallucination and generating information not present in context, as they process everything left-to-right without dedicated bidirectional context encoding
- Encoder-only models like BERT output embeddings or classifications, not generative text answers. They cannot produce free-form text responses
- Positional encoding is mandatory in all transformer variants to maintain token order information. Without it, the model cannot understand sequence structure. Multi-head attention alone is not an architecture

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** During inference, a GPT-based model with a context window of 8,192 tokens receives a prompt containing 8,000 tokens. How many new tokens can the model generate in a single forward pass?

A) 192 tokens (context window minus prompt length)
B) 8,192 tokens (the full context window)
C) Unlimited tokens through multiple forward passes
D) 1 token, then the context must be truncated to continue generation

<details>
<summary>Answer</summary>

**Correct: A**

Context window defines the TOTAL number of tokens (prompt + generated) the model can process. With 8,192 max and 8,000 already used by prompt, only 192 tokens remain available for generation. Exam tests this arithmetic.

**Why the others are wrong:**
- 8,192 is the total capacity, not available generation space. The prompt already occupies 8,000 tokens
- While sliding window approaches exist in production systems, the question asks about a single forward pass. Each token generated must fit within the 8,192 limit including the original prompt
- After generating 192 tokens (reaching 8,192 total), the model must truncate earlier tokens to continue. It cannot generate even 1 more token without removing something from context

</details>

---

**Q3.** A research team is comparing attention mechanisms. In multi-head attention with 12 heads and embedding dimension of 768, what is the dimension of each attention head's query, key, and value projections?

A) 768 dimensions (same as embedding dimension)
B) 12 dimensions (number of heads)
C) 32 dimensions (standard head dimension regardless of total embedding size)
D) 64 dimensions (768 ÷ 12 heads)

<details>
<summary>Answer</summary>

**Correct: D**

In multi-head attention, the embedding dimension is split equally across heads. Each head operates on 768÷12=64 dimensions independently, then outputs are concatenated back to 768 dimensions. This parallelization allows different heads to learn different attention patterns. Exam commonly tests this calculation.

**Why the others are wrong:**
- 768 would be the dimension if using single-head attention. Multi-head attention explicitly divides the embedding space across heads
- 12 is the number of parallel heads, not the dimension of each head's projections
- Head dimension is always embedding_dim ÷ num_heads. There is no fixed 'standard' dimension of 32 regardless of model size

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** An ML engineer notices that when training a transformer from scratch without positional encoding, the model achieves only 23% accuracy on a sequence classification task, far below the expected 89%. What is the MOST likely cause?

A) Learning rate is too high causing gradient explosion
B) Multi-head attention is computing incorrect attention scores
C) The dataset is too small for transformer pretraining
D) The model cannot distinguish token order, treating 'dog bites man' identically to 'man bites dog'

<details>
<summary>Answer</summary>

**Correct: D**

Without positional encoding, self-attention is permutation-invariant—it produces identical outputs regardless of input order. The model cannot learn that sequence order matters. Positional encoding injects order information into token embeddings. This is a heavily tested gotcha on the exam.

**Why the others are wrong:**
- High learning rate would cause training instability or divergence, not specifically low accuracy while still training. The 23% suggests the model is learning something but missing critical information
- Attention score computation is determined by the architecture and would fail consistently, not produce partial accuracy. The issue is missing information (position), not broken computation
- Dataset size affects generalization and overfitting, not the fundamental ability to learn sequence order. Even with small data, a model with positional encoding would learn that order matters

</details>

---

**Q5.** A company needs to pretrain a foundation model for code generation tasks. Which pretraining objective is MOST appropriate for this use case?

A) Masked Language Modeling (MLM) to predict randomly masked tokens
B) Contrastive learning to learn code embeddings
C) Causal Language Modeling (CLM) to predict next tokens autoregressively
D) Seq2seq translation from comments to code

<details>
<summary>Answer</summary>

**Correct: C**

Code generation is autoregressive: given partial code, generate the next token/line. Causal Language Modeling (CLM) trains the model to predict next tokens given previous context, exactly matching code generation inference. GPT models use CLM. Decoder-only + CLM is the standard for generation tasks.

**Why the others are wrong:**
- MLM (used by BERT) predicts randomly masked tokens using bidirectional context. This is optimal for understanding tasks (classification, QA) but not for left-to-right generation. The model would see 'future' tokens during training that aren't available during generation
- Contrastive learning creates embeddings but doesn't train the model to generate sequences token-by-token. It's useful for retrieval/search, not generation
- Direct translation training requires parallel comment-code pairs and is supervised fine-tuning, not pretraining. Pretraining should work on raw unlabeled code using self-supervised objectives

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** A tokenizer uses Byte-Pair Encoding (BPE) with a vocabulary size of 50,000. An input text of 1,000 characters is tokenized into 312 tokens. What does this indicate about the tokenizer's behavior?

A) The tokenizer is broken—it should produce exactly 1,000 tokens for 1,000 characters
B) The vocabulary size is too small, causing excessive token splitting
C) The text contains 688 out-of-vocabulary (OOV) characters that were dropped
D) The tokenizer merged common character sequences into subword units, reducing token count below character count

<details>
<summary>Answer</summary>

**Correct: D**

BPE learns to merge frequent character/subword sequences into single tokens. Common words or subwords become single tokens, resulting in fewer tokens than characters. 312 tokens from 1,000 characters (3.2 chars/token avg) is normal BPE behavior. Exam tests understanding that subword tokenization compresses text.

**Why the others are wrong:**
- One character per token is character-level tokenization, not BPE. BPE specifically merges sequences to create subword units, reducing token count
- Smaller vocabulary would increase token count (more splits), not decrease it. The 50,000 vocabulary is reasonably large and enables good compression through learned merges
- BPE doesn't drop unknown characters. It falls back to character-level or byte-level encoding for unseen sequences, ensuring no information loss. 312 tokens represent all 1,000 characters

</details>

---

**Q7.** During attention computation, query matrix Q has shape [batch_size, seq_len, 64], key matrix K has shape [batch_size, seq_len, 64]. The attention scores are computed as QK^T. Before applying softmax, what operation is typically applied to the attention scores and why?

A) Multiply by sqrt(64) to amplify small attention values
B) Add positional encoding to inject sequence order
C) Divide by sqrt(64) to prevent softmax saturation from large dot products
D) Apply layer normalization to stabilize training

<details>
<summary>Answer</summary>

**Correct: C**

Scaled dot-product attention divides by sqrt(d_k) where d_k is key dimension (64). Without scaling, dot products grow large in magnitude (proportional to d_k), pushing softmax into regions with tiny gradients. Division by sqrt(64)=8 keeps scores in a range where softmax has useful gradients. This is fundamental scaled attention math that the exam tests.

**Why the others are wrong:**
- Multiplying by sqrt(d_k) would make the problem worse, creating even larger values that saturate softmax more severely
- Positional encoding is added to input embeddings before attention, not to attention scores. Attention scores represent relationships between tokens, not absolute positions
- Layer normalization is applied to attention outputs or residual connections, not to raw attention scores before softmax. The scaling is specifically a mathematical correction for dot product magnitude

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A production system uses a transformer model with maximum sequence length of 4,096 tokens. A user submits a prompt that tokenizes to 4,100 tokens. What typically happens during inference?

A) The most recent 4,096 tokens are kept and the earliest 4 tokens are truncated
B) The model automatically extends its context window to accommodate 4,100 tokens
C) The first 4,096 tokens are processed and the remaining 4 tokens are ignored
D) Inference fails with an error requiring the user to shorten the prompt

<details>
<summary>Answer</summary>

**Correct: A**

Standard practice is right-truncation: keep the most recent tokens (nearest to generation point) and drop the earliest tokens. This preserves immediate context for generation. The model processes the rightmost 4,096 tokens. Some systems show warnings, but inference continues. Exam tests practical deployment behavior.

**Why the others are wrong:**
- Context window is fixed by the model architecture (positional encodings, attention matrices). It cannot be dynamically extended without architectural changes or techniques like RoPE/ALiBi which must be designed in from training
- Left-truncation (keeping first 4,096, dropping last 4) would lose the most recent context closest to where generation begins, making generated text incoherent with the end of the prompt
- Most production systems truncate automatically with a warning rather than hard-failing. Hard errors are poor UX. The exam tests that systems handle this gracefully by truncating, not crashing

</details>

---

