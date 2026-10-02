# Data Preprocessing & Feature Engineering Cheat Sheet

## Text Cleaning Priority Order
1. **Remove HTML/XML tags** → prevents tokenizing markup
2. **Remove URLs** → noise, don't generalize
3. **Handle emojis** → remove or convert to text (model-dependent)
4. **Lowercase** → reduces vocab size (unless case matters)
5. **Unicode normalization** → NFKC for ML, NFC for fidelity
6. **Remove extra whitespace** → normalize to single spaces

**Don't apply**: Stemming/lemmatization (transformers learn morphology)

---

## Unicode Normalization Forms
| Form | Use Case | Effect | Example |
|------|----------|--------|--------|
| **NFC** | Formal documents, preserving fidelity | Composes characters | é (2 chars) → é (1 char) |
| **NFD** | Linguistic analysis | Separates base + diacritics | é → e + ´ |
| **NFKC** | **NLP/ML** (recommended) | Compatibility + composition | Ａ→A, ₂→2, é→é |
| **NFKD** | Rarely used | Compatibility + decomposition | Combines NFKC + NFD |

**Exam rule**: NFKC for ML (reduces vocab), NFC for human text

---

## Tokenization Comparison
| Type | Vocab Size | Sequence Length | [UNK] Rate | Best For |
|------|------------|----------------|------------|----------|
| **Word-level** | 50K-100K+ | Shortest (1x) | High (rare words) | Simple tasks, known vocab |
| **BPE/WordPiece** | 32K-50K | Medium (1.5-2x) | Zero | **Most LLMs** (GPT, BERT) |
| **Character-level** | 256-500 | Longest (10-20x) | Zero | Noisy text, typos |

**Exam pick**: BPE for unknown domains (no [UNK], reasonable length)

---

## Handling Imbalanced Data
| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Class weighting** | Default (any ratio) | No data loss, fast | Must support in loss function |
| **Oversampling (SMOTE)** | Minority <500 samples | Creates synthetic data | Risk overfitting minority |
| **Undersampling** | Majority >1M, minority >10K | Faster training | Discards data |

**Weight formula**: `class_weight = total_samples / (n_classes × class_samples)`

**Example**: 10K pos, 1K neg → neg_weight=10, pos_weight=1

---

## Sequence Length Handling
| Scenario | Strategy | Details |
|----------|----------|--------|
| 95% fit, 5% exceed | **Sliding window chunking** | 128-256 token overlap |
| All exceed max | Truncate + summarization | Keep first 256 + last 256 tokens |
| All fit | Pad to max in batch | Use attention masks |

**Overlap calculation**: For 512 window, use 128 overlap (25%)

---

## Data Augmentation Quality Ranking
1. **Back-translation** (best diversity, preserves meaning)
   - Translate to 3+ languages → back to source
   - 2-4% accuracy gain in low-resource settings
2. **Contextual replacement** (BERT MLM)
   - Replace 15% max of tokens
   - Higher quality than synonyms
3. **Synonym replacement** (WordNet)
   - 10-15% replacement rate
   - Fast but lower quality
4. **Random deletion** (worst)
   - 10% deletion rate
   - Use only for robustness testing

---

## Noisy Label Detection
| Method | Approach | Accuracy |
|--------|----------|----------|
| **Ensemble disagreement** | Train 5 models, flag when all disagree | 70-80% |
| **Confident learning** | Use predicted probabilities (cleanlab) | 75-85% |
| **Manual review** | Domain expert review | 100% (slow) |

**Process**: Flag → Review → Relabel → Retrain

---

## EDA Checklist Before Fine-tuning
- [ ] **Class distribution** → imbalance >3:1 needs weighting
- [ ] **Sequence length** → 95th percentile determines truncation
- [ ] **Vocab size** → guides tokenizer choice
- [ ] **Token frequency** → top 50 words, check for noise
- [ ] **Duplicates** → identify inconsistent labels

---

## Visualization Tools
| Visualization | Library | Use Case |
|---------------|---------|----------|
| **t-SNE** | sklearn | Embedding clusters (perplexity=30) |
| **UMAP** | umap-learn | Faster than t-SNE, large datasets |
| **Word cloud** | wordcloud | Quick vocab overview |
| **Seaborn heatmap** | seaborn | Confusion matrix, correlations |

**t-SNE params**: perplexity=30 (default), iterations=1000+

---

## Attention Masks & Padding
```python
# Input: ["Hello world", "Hi"]
# After padding to max_length=4:
# tokens:    [["Hello", "world", "<pad>", "<pad>"], ["Hi", "<pad>", "<pad>", "<pad>"]]
# attention: [[1, 1, 0, 0], [1, 0, 0, 0]]
```
**Rule**: 1 = real token (attend), 0 = padding (ignore)

---

## Memory Scaling (Transformers)
| Sequence Length | Attention Memory | Total Memory (approx) |
|-----------------|------------------|----------------------|
| 512 tokens | 1x | 1x |
| 1024 tokens | 4x | ~2x |
| 2048 tokens | 16x | ~4-8x |

**Formula**: Attention = O(n²), doubles length = 4x memory
