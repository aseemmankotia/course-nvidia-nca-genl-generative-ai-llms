## Chapter 5: Data Preprocessing, Feature Engineering, Analysis and Visualisation — Practice Questions

**Q1.** A company is fine-tuning a BERT model for sentiment analysis on customer reviews. The dataset contains 50,000 positive reviews and 5,000 negative reviews. During training, the model achieves 92% accuracy but fails to identify most negative reviews. Which approach would BEST address this issue?

A) Increase the learning rate and train for more epochs to improve overall accuracy
B) Remove 45,000 positive reviews to create a balanced dataset of 5,000 samples per class
C) Apply class weighting with weight=10 for negative class and weight=1 for positive class during loss calculation
D) Use data augmentation to generate synthetic positive reviews until both classes have 50,000 samples

<details>
<summary>Answer</summary>

**Correct: C**

Class weighting (weight inversely proportional to class frequency) is the most effective approach for imbalanced datasets in LLM fine-tuning. With a 10:1 imbalance, weighting the negative class 10x higher forces the model to pay more attention to minority class errors without discarding valuable training data. This maintains the full dataset size while addressing the imbalance during loss calculation.

**Why the others are wrong:**
- Higher learning rate and more epochs will likely make the problem worse - the model will continue to optimize for the majority class since that's what minimizes overall loss. Accuracy is a misleading metric for imbalanced data.
- Undersampling to 5,000 samples per class throws away 90% of the data (45,000 positive reviews). This drastically reduces training data and model performance. Only viable when you have massive datasets and the minority class is still large enough.
- Augmenting the majority class makes the imbalance worse. You would augment the minority class (negative reviews) using techniques like back-translation, synonym replacement, or paraphrasing - but class weighting is still more effective and doesn't risk introducing noise.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A data scientist is preparing text data for training a custom GPT-2 model. The raw dataset contains user-generated content with HTML tags, URLs, emojis, and inconsistent casing. Which preprocessing sequence is MOST appropriate before tokenization?

A) Convert to lowercase → Remove HTML tags → Keep URLs and emojis for context → Apply stemming
B) Remove HTML tags → Convert to lowercase → Remove URLs → Remove emojis → Apply Unicode normalization (NFKC)
C) Remove HTML tags → Keep original casing → Remove URLs → Convert emojis to text descriptions → Apply Unicode normalization (NFC)
D) Apply lemmatization → Remove HTML tags → Remove URLs → Convert to lowercase → Remove all non-ASCII characters

<details>
<summary>Answer</summary>

**Correct: B**

This sequence follows best practices for modern LLM preprocessing: (1) Remove HTML tags first to avoid tokenizing markup; (2) Lowercase to reduce vocabulary size and improve generalization; (3) Remove URLs which are noise and don't generalize; (4) Remove emojis unless specifically needed (GPT-2 tokenizer handles them poorly); (5) NFKC normalization combines characters and applies compatibility decomposition, ensuring consistent Unicode representation. This order prevents creating orphaned tokens and reduces vocabulary bloat.

**Why the others are wrong:**
- Keeping URLs and emojis adds noise - URLs don't generalize across documents and emojis fragment into multiple tokens in GPT-2's BPE tokenizer. Stemming is outdated for transformer models which learn contextual representations.
- Keeping original casing doubles vocabulary size ("The" vs "the") without benefit for most tasks. Converting emojis to text ("😊" → "smiling face") changes token count unpredictably. NFC normalization alone doesn't handle compatibility characters as well as NFKC.
- Lemmatization before cleaning means you're lemmatizing HTML tags and URLs, wasting computation. Removing all non-ASCII eliminates valid multilingual content. Lemmatization is also unnecessary for subword tokenizers which handle morphological variations.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** An ML engineer is comparing tokenization strategies for a domain-specific language model. The vocabulary size is capped at 32,000 tokens. The corpus contains many technical terms like "antidisestablishmentarianism" and compound words. Which tokenization approach provides the BEST trade-off between vocabulary size and representation quality?

A) Subword tokenization using Byte-Pair Encoding (BPE) trained on the corpus to learn 32,000 merge operations
B) Word-level tokenization with vocabulary limited to the 32,000 most frequent words, treating rare words as [UNK] tokens
C) Character-level tokenization using only 256 characters including special tokens, requiring no vocabulary limit
D) Hybrid approach: word-level for common words (top 20,000) and character-level for remaining tokens

<details>
<summary>Answer</summary>

**Correct: A**

BPE subword tokenization is optimal for the constraints: (1) It learns frequent subwords from the corpus, so technical terms like 'antidisestablishmentarianism' become ['anti', 'dis', 'establish', 'ment', 'arian', 'ism'] instead of [UNK]; (2) Vocabulary stays exactly at 32,000 by controlling merge operations; (3) No out-of-vocabulary words - any word can be decomposed into subwords; (4) Sequence lengths are reasonable (longer than word-level, much shorter than character-level). BPE is used by GPT-2/3/4, RoBERTa for this reason.

**Why the others are wrong:**
- Word-level with 32K vocab will have massive [UNK] problems with technical terms, compounds, and rare words. Every [UNK] loses semantic information. Domain-specific corpora have many rare but important terms that won't make the top 32K.
- Character-level creates extremely long sequences (10-20x longer than word-level), making training prohibitively expensive. A 512-token sentence becomes 2,000+ characters. Models struggle to learn long-range dependencies at character level. Only viable for very short sequences or specific tasks.
- Hybrid approaches add complexity without clear benefits. You need separate vocabularies and rules for when to switch between modes. The 20K/12K split is arbitrary. BPE naturally handles this - common words often stay as single tokens, rare words split into subwords.

</details>

---

**Q4.** A team is analyzing token distributions for a fine-tuning dataset. They discover that 95% of sequences are under 256 tokens, but 5% range from 512 to 2,048 tokens. The model's maximum sequence length is 512 tokens. Given memory constraints, which strategy is MOST appropriate?

A) Truncate all sequences to 512 tokens, discarding content beyond that point, and use the full dataset
B) Filter out all sequences longer than 512 tokens, training only on the 95% of sequences that fit within the limit
C) Split sequences longer than 512 into chunks with 128-token sliding window overlap, creating multiple training samples per long document
D) Use gradient checkpointing and mixed precision training to fit 2,048-token sequences into memory

<details>
<summary>Answer</summary>

**Correct: C**

Sliding window chunking preserves all information from long sequences while fitting memory constraints: A 1,024-token document becomes 3 chunks ([0:512], [384:896], [768:1280]) with 128-token overlap. Overlap ensures context isn't lost at boundaries - critical for tasks like question answering. This increases training data (5% of sequences become 15-20% of samples) and preserves all content. The model learns from complete information rather than truncated data.

**Why the others are wrong:**
- Truncation loses potentially important information in the 5% of long sequences. If those sequences contain critical examples (detailed technical documentation, complex reasoning), the model never learns from that content. Simple but wasteful.
- Filtering removes 5% of data entirely. If long sequences are important (e.g., longer documents might be higher quality or contain more examples), you're biasing the dataset toward shorter, potentially simpler content. Only acceptable if long sequences are noise.
- Increasing max sequence length to 2,048 quadratically increases memory (self-attention is O(n²)). Even with gradient checkpointing and FP16, fitting 2,048 tokens requires 16x more memory than 512 tokens for attention alone. This violates the stated memory constraints and drastically slows training.

</details>

---

**Q5.** During EDA for a text classification dataset, a data scientist discovers that 30% of training labels are inconsistent when the same text appears multiple times with different labels. The test set is cleanly labeled by experts. Which approach would BEST improve model performance?

A) Apply label smoothing with epsilon=0.3 during training to account for the 30% noise rate in the loss function
B) Increase dropout rate to 0.5 and reduce learning rate to make the model more robust to noisy labels
C) Remove all duplicate texts from the training set, keeping only the first occurrence of each unique text
D) Use ensemble disagreement to identify mislabeled samples: train 5 models, flag samples where all 5 disagree with the label, manually review and relabel flagged samples

<details>
<summary>Answer</summary>

**Correct: D**

Ensemble disagreement is highly effective for identifying mislabeled data: When 5 independently trained models all predict class A but the label is B, the label is likely wrong. This leverages model wisdom to find annotation errors. Manually reviewing only flagged samples (targeted effort) is efficient. Studies show this catches 70-80% of label errors with minimal false positives. After relabeling, retrain on clean data for best performance. This directly fixes the root cause (bad labels) rather than working around it.

**Why the others are wrong:**
- Label smoothing (soft targets like [0.7, 0.3] instead of [1.0, 0.0]) is designed for model calibration, not noisy labels. It assumes labels are fundamentally correct but shouldn't be overfit. With 30% inconsistent labels, you're still training on wrong information 30% of the time. The model learns incorrect patterns.
- Higher dropout and lower learning rate make training slower and might reduce overfitting, but don't address the core issue - the model still sees wrong labels 30% of the time. Robustness techniques help with random noise, not systematic labeling inconsistencies. Performance will still suffer.
- Removing duplicates doesn't solve the problem - you keep one instance with a potentially wrong label and discard others that might be correctly labeled. You also lose valuable training data. The issue isn't duplication, it's inconsistent labeling across duplicates.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** A researcher is visualizing the embedding space of a fine-tuned BERT model to understand how well it separates three classes of scientific abstracts. The embedding dimension is 768. Which dimensionality reduction technique is MOST appropriate for creating a 2D visualization that preserves local cluster structure?

A) t-SNE with perplexity=30 and 1,000 iterations, optimizing for local neighborhood preservation
B) Principal Component Analysis (PCA) to project 768 dimensions to 2 dimensions, preserving maximum variance
C) Random projection using Johnson-Lindenstrauss lemma to reduce to 2 dimensions while preserving distances
D) Autoencoder with 768-256-64-2 architecture trained to reconstruct the original embeddings

<details>
<summary>Answer</summary>

**Correct: A**

t-SNE is specifically designed for visualizing high-dimensional data by preserving local structure: (1) Perplexity=30 is standard for datasets with hundreds to thousands of samples, balancing local vs global structure; (2) It excels at revealing clusters and separations between classes; (3) 1,000+ iterations ensure convergence; (4) Unlike PCA, t-SNE uses non-linear transformations to capture complex relationships. For understanding class separation in embedding space, t-SNE is the gold standard. UMAP is a faster alternative with similar properties.

**Why the others are wrong:**
- PCA is linear and preserves global variance, not local cluster structure. In high dimensions (768→2), PCA often creates overlapping blobs because it can't capture non-linear relationships. First two PCs might explain only 15-20% of variance. PCA is better for initial exploration or preprocessing before t-SNE.
- Random projection preserves pairwise distances approximately but isn't optimized for visualization. Results are hard to interpret - no guarantee of cluster separation. Used more for dimensionality reduction before ML algorithms (preprocessing) than for human-interpretable visualization.
- Autoencoders can learn non-linear mappings but require careful architecture design and training. The 2-dimensional bottleneck is too narrow - reconstruction loss will be high, meaning the 2D representation loses significant information. t-SNE is purpose-built for this task and requires no training.

</details>

---

**Q7.** A data preprocessing pipeline for a multilingual NLP model must handle text in English, Arabic, Japanese, and emoji-heavy social media content. The text contains mixed scripts, zero-width characters, and combining diacritics. Which Unicode normalization form should be applied?

A) NFC (Canonical Composition) to combine base characters with diacritics into single code points where possible
B) NFD (Canonical Decomposition) to separate all base characters and combining marks into separate code points
C) NFKC (Compatibility Composition) to apply compatibility mappings and combine characters into canonical forms
D) NFKD (Compatibility Decomposition) to decompose compatibility characters while preserving all distinctions

<details>
<summary>Answer</summary>

**Correct: C**

NFKC is most appropriate for NLP preprocessing of mixed-script social media: (1) Applies compatibility mappings that convert visually similar characters to canonical forms (e.g., full-width 'Ａ' → 'A', subscript '₂' → '2'); (2) Combines base + diacritics into single code points, reducing token fragmentation; (3) Handles zero-width joiners/non-joiners consistently; (4) Creates more uniform representation across scripts. This reduces vocabulary size and improves tokenization consistency. NFKC is used by most modern NLP libraries (spaCy, Transformers) for this reason.

**Why the others are wrong:**
- NFC preserves compatibility distinctions, meaning 'Ａ' (full-width) and 'A' (normal) remain different characters. In social media, these are semantically identical but create separate tokens. NFC is safer for text where distinctions matter (scholarly texts) but not optimal for noisy social media.
- NFD explodes character counts by separating all diacritics (é → e + ´). In Arabic and Japanese, this creates many combining marks. Tokenizers treat these separately, fragmenting words into many tokens. Makes sequences longer and vocabulary larger. NFD is used mainly for specialized linguistic analysis.
- NFKD combines compatibility mappings with decomposition, giving the worst of both worlds: applies aggressive normalization (losing some distinctions) then explodes character count with decomposition. Results in long sequences of basic characters + combining marks. No practical advantage for ML.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A company is implementing data augmentation for a small fine-tuning dataset (2,000 samples) to improve a text classification model. Which augmentation technique would provide the MOST semantic diversity while maintaining label correctness?

A) Back-translation: translate samples to French, German, and Spanish, then back to English using different MT models
B) Synonym replacement: randomly replace 15% of words with WordNet synonyms, generating 3 variants per sample
C) Random deletion: randomly remove 10% of words from each sample, generating 5 variants per sample
D) Contextual word replacement: use a masked language model (BERT) to replace 20% of words with contextually appropriate alternatives

<details>
<summary>Answer</summary>

**Correct: A**

Back-translation through multiple languages provides highest semantic diversity while preserving meaning: (1) Each translation path (EN→FR→EN, EN→DE→EN, EN→ES→EN) produces different paraphrases with same meaning; (2) Neural MT models use different words/structures, creating natural variation; (3) Label correctness is maintained because core meaning is preserved; (4) Generates genuinely new text patterns the model hasn't seen. This method has shown 2-4% accuracy improvements in low-resource settings. Using multiple target languages increases diversity.

**Why the others are wrong:**
- Synonym replacement often creates unnatural or semantically incorrect text. WordNet synonyms ignore context ("good food" → "beneficial food", "sound sleep" → "healthy sleep"). With 15% replacement rate, many sentences become awkward. Simple but lower quality augmentation - better than nothing but not optimal.
- Random deletion breaks grammar and often changes meaning. Removing 10% of words (every 10th word roughly) can delete critical words like negations ("not good" → "good"). Generated samples are lower quality and may have incorrect labels. Only useful for robustness testing, not training data augmentation.
- Contextual replacement (MLM) is better than synonym replacement but 20% is too aggressive - risks changing meaning. Even BERT makes context errors. Generates less diversity than back-translation because it's replacing individual words within the same sentence structure rather than fully paraphrasing.

</details>

---

