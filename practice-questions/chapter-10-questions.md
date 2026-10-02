## Chapter 10: Full NCA-GENL Exam Simulation and Test-Taking Strategy — Practice Questions

**Q1.** A company is developing a customer service chatbot that must maintain conversation context, generate human-like responses, and handle multi-turn dialogues. The solution must use NVIDIA AI Enterprise components and minimize custom training. Which architecture BEST meets these requirements?

A) Deploy Llama 3.1 70B using NVIDIA NIM with GPU-accelerated inference and use prompt engineering for conversation flow
B) Fine-tune a BERT model using NVIDIA NeMo Framework and deploy with Triton Inference Server
C) Use NVIDIA Riva for speech-to-text, then deploy a custom seq2seq model with TensorRT optimization
D) Implement a retrieval-augmented generation (RAG) pipeline using encoder-only models with FAISS indexing

<details>
<summary>Answer</summary>

**Correct: A**

Llama 3.1 is a decoder-only autoregressive model optimized for dialogue and conversation. NIM provides production-ready deployment with GPU acceleration. This requires no custom training - just prompt engineering - which meets the minimize training requirement. Decoder-only models excel at multi-turn dialogue.

**Why the others are wrong:**
- BERT is an encoder-only model designed for understanding, not generation. It cannot generate conversational responses and would require significant custom architecture to produce dialogue.
- Riva is specifically for speech AI (ASR/TTS), not text-based chatbots. A custom seq2seq model would require extensive training, violating the minimize training requirement.
- RAG with encoder-only models is for retrieval and understanding, not dialogue generation. Encoder-only models (like BERT) cannot generate conversational text - they output embeddings, not human-like responses.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An AI team needs to evaluate their machine translation model's output quality. They want a metric that measures precision of n-grams between generated translations and reference translations. Which metric should they use?

A) BLEU score to measure n-gram precision with brevity penalty
B) ROUGE-L score to measure longest common subsequence overlap
C) Perplexity to measure model uncertainty on the validation set
D) F1 score to balance precision and recall of word matches

<details>
<summary>Answer</summary>

**Correct: A**

BLEU (Bilingual Evaluation Understudy) specifically measures n-gram precision between generated and reference translations, with a brevity penalty to prevent gaming the metric with short outputs. It's the standard metric for machine translation evaluation.

**Why the others are wrong:**
- ROUGE-L measures recall (not precision) of the longest common subsequence. ROUGE is designed for summarization tasks, not translation, and focuses on recall rather than precision.
- Perplexity measures how well a probability model predicts a sample - it's a language modeling metric, not a translation quality metric. It doesn't compare generated output to references.
- F1 score is a general classification metric. While it could theoretically be adapted, it's not standard for translation and doesn't capture n-gram overlap or sequential quality that BLEU measures.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q3.** A developer is fine-tuning a large language model and notices the training loss decreases steadily but validation loss starts increasing after epoch 5. Training continues through epoch 20. What is the MOST likely issue?

A) The learning rate is too high and causing gradient explosion during later epochs
B) The model architecture is too small to capture patterns in the dataset
C) The model is overfitting to the training data and requires early stopping or regularization
D) The training data contains insufficient examples for the model to generalize

<details>
<summary>Answer</summary>

**Correct: C**

Decreasing training loss with increasing validation loss is the textbook definition of overfitting. The model is memorizing training data rather than learning generalizable patterns. Early stopping (halting at epoch 5) or regularization techniques would prevent this.

**Why the others are wrong:**
- High learning rate causes erratic loss curves (both training and validation would fluctuate or explode). Here training loss decreases steadily, indicating stable gradients.
- A too-small model would show high loss on both training and validation sets (underfitting). Here training loss is decreasing, showing the model has sufficient capacity to learn the training set.
- Insufficient training data typically causes both training and validation loss to remain high. The fact that training loss decreases shows there's enough data to fit - the issue is the model is fitting too well to the training set specifically.

</details>

---

**Q4.** A company needs to generate diverse creative content and wants to control randomness in model outputs. They want higher randomness for brainstorming ideas but lower randomness for factual summaries. Which sampling parameter should they adjust?

A) Increase top-p (nucleus sampling) value for more diversity, decrease for more focused outputs
B) Increase temperature for more diversity, decrease for more deterministic outputs
C) Increase the number of beams in beam search for more diverse outputs
D) Adjust the learning rate during inference to control output variability

<details>
<summary>Answer</summary>

**Correct: B**

Temperature directly controls randomness in sampling. Higher temperature (e.g., 0.8-1.0) increases randomness and creativity. Lower temperature (e.g., 0.1-0.3) makes outputs more deterministic and focused. This is the primary parameter for controlling diversity vs consistency in generation tasks.

**Why the others are wrong:**
- Top-p does affect diversity, but temperature is the more direct and commonly used parameter for this purpose. Top-p controls the cumulative probability mass to sample from, while temperature scales the entire probability distribution. Temperature is the standard answer for diversity control.
- Beam search increases the number of hypotheses considered but aims for higher-probability sequences, not diversity. More beams typically lead to safer, more generic outputs - the opposite of creative brainstorming.
- Learning rate is a training parameter, not an inference parameter. It cannot be adjusted during inference and does not control output variability - that's determined by sampling parameters.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** An organization is deploying a speech recognition system for a call center using NVIDIA Riva. The system must transcribe customer calls in real-time with low latency. Which Riva component should they use?

A) Riva TTS (Text-to-Speech) service with neural vocoder
B) Riva NLP service with intent classification models
C) Riva NMT (Neural Machine Translation) service for multilingual support
D) Riva ASR (Automatic Speech Recognition) service with streaming API

<details>
<summary>Answer</summary>

**Correct: D**

Riva ASR (Automatic Speech Recognition) is specifically designed for speech-to-text transcription. The streaming API enables real-time, low-latency transcription of audio - exactly what's needed for live call center transcription.

**Why the others are wrong:**
- TTS (Text-to-Speech) converts text to audio, not audio to text. This is the opposite direction from what's needed for transcription.
- NLP services handle text understanding tasks like intent classification, but they require text input. They can't directly transcribe audio - you'd need ASR first to convert speech to text.
- NMT translates text from one language to another. It doesn't perform speech recognition. You'd need ASR to transcribe first, then optionally NMT to translate.

</details>

---

**Q6.** A team is implementing Reinforcement Learning from Human Feedback (RLHF) to align a language model with human preferences. Which sequence of steps correctly describes the RLHF process?

A) Train reward model on human preferences → Pre-train base model → Use PPO to optimize policy → Supervised fine-tuning on demonstrations
B) Pre-train base model → Use PPO to optimize policy → Train reward model → Supervised fine-tuning on demonstrations
C) Supervised fine-tuning on demonstrations → Train reward model → Pre-train base model → Use PPO to optimize policy
D) Pre-train base model → Supervised fine-tuning on demonstrations → Train reward model on human preferences → Use PPO to optimize policy against reward model

<details>
<summary>Answer</summary>

**Correct: D**

The correct RLHF sequence is: (1) Start with a pre-trained base model, (2) Supervised fine-tuning (SFT) on human demonstrations to teach desired behavior format, (3) Train a reward model on human preference comparisons, (4) Use RL (typically PPO) to optimize the SFT model against the reward model. This is the standard InstructGPT/ChatGPT approach.

**Why the others are wrong:**
- Training the reward model before pre-training makes no sense - you need a model to generate outputs for humans to rank. Pre-training must come first to create the foundational model.
- PPO requires a reward model to provide rewards for optimization. You can't optimize the policy before having a reward model to guide it. The reward model must be trained before RL.
- Pre-training must come first to create the base model. SFT should happen after pre-training but before reward modeling, as you need SFT outputs to collect human preferences for the reward model.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** A company is building a RAG (Retrieval-Augmented Generation) system that must search through 10 million technical documents. Which combination of components is MOST cost-effective for this scale?

A) Use NVIDIA NIM to deploy a large language model and perform brute-force text matching across all documents
B) Use NVIDIA NeMo Retriever with vector database for embedding storage and semantic search
C) Use NVIDIA Riva ASR to convert documents to speech, then use TTS for retrieval
D) Use NVIDIA Picasso to generate images of document pages and use computer vision for search

<details>
<summary>Answer</summary>

**Correct: B**

NeMo Retriever is specifically designed for RAG at scale. It handles embedding generation and semantic search over large document collections efficiently using vector databases. This is the purpose-built solution for searching millions of documents in a RAG pipeline.

**Why the others are wrong:**
- Brute-force text matching across 10 million documents with an LLM would be prohibitively expensive and slow. RAG requires efficient retrieval (finding relevant docs fast) before generation. NIM is for inference deployment, not document search.
- Riva ASR/TTS are for speech applications, not document search. Converting text documents to speech and back to search them makes no technical or economic sense.
- Picasso is for generative AI in visual media (images, video). Converting text documents to images to search them is inefficient and not what Picasso is designed for. Text search should use text embeddings.

</details>

---

**Q8.** During an NCA-GENL exam, a candidate encounters a question asking which NVIDIA component is 'MOST appropriate' for a scenario. The candidate identifies two options that could both work technically. What is the BEST test-taking strategy?

A) Flag the question as ambiguous and skip it until all other questions are answered
B) Choose the option that uses the most advanced or newest NVIDIA technology
C) Choose the option that is most cost-effective and aligns with NVIDIA AI Enterprise best practices
D) Select the first option identified as technically correct to save time

<details>
<summary>Answer</summary>

**Correct: C**

When multiple options are technically valid, exam questions test your ability to identify the MOST appropriate solution considering factors like cost-effectiveness, simplicity, alignment with best practices, and meeting stated requirements without over-engineering. NVIDIA exams favor practical, enterprise-ready solutions over complex ones.

**Why the others are wrong:**
- While flagging complex questions is good strategy, this wastes the analysis already done. If you've narrowed to two options, use decision criteria (cost, simplicity, best practices) to choose rather than skipping entirely.
- Newest or most advanced technology is often not the best answer. Exams test judgment in selecting appropriate tools - sometimes a simpler, more established solution is correct. Over-engineering is a common trap.
- Choosing the first technically correct option without comparing alternatives risks missing the MOST appropriate answer. Exam questions specifically use 'MOST' and 'BEST' to test your ability to differentiate between multiple valid options.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

