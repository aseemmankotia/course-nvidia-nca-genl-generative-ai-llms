## Chapter 7: LLM Integration and Deployment—Part 1: RAG, APIs, and Python Libraries — Practice Questions

**Q1.** A financial services company needs to build a chatbot that answers questions about their internal policy documents. The documents contain proprietary procedures not available in public datasets. The chatbot must cite specific sections from these documents. Which approach BEST meets these requirements?

A) Fine-tune a large language model on the policy documents and deploy it using NVIDIA Triton Inference Server
B) Implement a RAG system with embeddings of policy documents stored in a vector database, retrieving relevant chunks before generation
C) Use prompt engineering with few-shot examples from the policy documents in the system prompt
D) Deploy a pre-trained model via OpenAI API and include entire policy documents in each API call context

<details>
<summary>Answer</summary>

**Correct: B**

RAG is specifically designed for grounding LLM responses in private/domain-specific data. It retrieves relevant document chunks via vector search, then passes them to the LLM as context. This reduces hallucination, enables citation of sources, and doesn't require retraining the model. Perfect for proprietary documents.

**Why the others are wrong:**
- Fine-tuning requires significant compute resources, doesn't guarantee the model will cite sources correctly, and makes updating documents difficult (requires retraining). RAG is more flexible for document-based Q&A.
- Few-shot examples in prompts don't scale to large policy document sets and won't retrieve the most relevant sections dynamically. The model may still hallucinate without actual document content as context.
- Including entire policy documents in each API call is cost-prohibitive (token costs), hits context length limits, and doesn't selectively retrieve only relevant sections. RAG optimizes by retrieving only pertinent chunks.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A data scientist needs to load a pre-trained BERT model for sequence classification and run inference on a batch of customer reviews. Which Hugging Face Transformers code accomplishes this with the LEAST custom code?

A) from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('bert-base-uncased'); tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'); inputs = tokenizer(reviews); outputs = model(**inputs)
B) from transformers import BertForSequenceClassification; model = BertForSequenceClassification.from_pretrained('bert-base-uncased'); model.eval(); model.forward(reviews)
C) import torch; model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased'); model.classify(reviews)
D) from transformers import pipeline; classifier = pipeline('sentiment-analysis'); results = classifier(reviews)

<details>
<summary>Answer</summary>

**Correct: D**

pipeline() is the highest-level Transformers API, handling tokenizer loading, model loading, tokenization, inference, and post-processing automatically. 'sentiment-analysis' pipeline uses a pre-trained classification model and returns human-readable results with minimal code.

**Why the others are wrong:**
- AutoModel loads the base model without a task-specific head. For classification, need AutoModelForSequenceClassification. Also requires manual tokenization, batching, and post-processing. More code than pipeline().
- BertForSequenceClassification.forward() requires pre-tokenized tensor inputs, not raw text. Missing tokenization step entirely. Also need to manually handle output logits to get predictions.
- torch.hub is not the standard way to load Transformers models. The API shown is incorrect—Transformers models don't have a .classify() method. This code would fail.

</details>

---

**Q3.** An ML engineer is comparing vector databases for a RAG system handling 50 million embeddings with real-time queries (<100ms latency required). The system must support semantic search with cosine similarity. Which vector database feature is MOST critical for meeting the latency requirement?

A) Support for exact k-NN search using brute-force distance computation
B) Approximate nearest neighbor (ANN) search using HNSW or IVF indexing
C) Built-in embedding model integration for automatic vectorization
D) Multi-tenancy support for isolated namespaces per user

<details>
<summary>Answer</summary>

**Correct: B**

At 50M embeddings, exact k-NN is computationally prohibitive for <100ms latency. ANN algorithms like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File) achieve sub-100ms query times by trading slight accuracy for massive speed gains. Essential for production RAG at scale.

**Why the others are wrong:**
- Exact k-NN computes distance to ALL vectors—infeasible at 50M scale for real-time queries. Would take seconds or minutes, not <100ms. Only viable for small datasets (<100K vectors).
- Embedding model integration is convenient but not critical for latency. Pre-computing embeddings offline and storing them achieves the same result. Latency is determined by search algorithm, not vectorization method.
- Multi-tenancy is an operational feature for organization/security, not performance. Doesn't impact query latency. All major vector DBs (Pinecone, Weaviate, Milvus) support this regardless of search speed.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A developer uses LangChain to build a conversational agent that remembers previous exchanges within a session. Which LangChain component provides this capability?

A) SequentialChain with output parsers for structured data extraction
B) ConversationChain with ConversationBufferMemory or ConversationSummaryMemory
C) RetrievalQA chain with VectorStoreRetriever for context injection
D) Agent with Tool selection using ReAct prompting framework

<details>
<summary>Answer</summary>

**Correct: B**

LangChain Memory components (ConversationBufferMemory, ConversationSummaryMemory) explicitly store conversation history and inject it into prompts. ConversationChain automatically uses Memory to maintain context across turns. This is the standard LangChain pattern for stateful conversations.

**Why the others are wrong:**
- SequentialChain runs multiple chains in sequence, passing outputs as inputs. Output parsers structure responses (JSON, lists). Neither provides conversation memory—they're for workflow orchestration, not state management.
- RetrievalQA retrieves external documents, not conversation history. It's for RAG workflows (grounding in knowledge base), not remembering what the user said 3 messages ago.
- Agents select tools/actions based on reasoning. ReAct is a prompting technique for step-by-step reasoning. Neither inherently stores conversation history—Agents can use Memory, but the Agent itself isn't the memory component.

</details>

---

**Q5.** A startup is deciding between self-hosting LLaMA-2-7B on GPU instances versus using OpenAI GPT-3.5-turbo API for a customer support chatbot handling 1 million requests/month with average 500 input tokens and 150 output tokens per request. Which factor is the PRIMARY cost consideration?

A) Self-hosting requires upfront GPU purchase costs that exceed 12 months of API fees
B) Self-hosting requires hiring ML engineers for model maintenance, increasing operational costs beyond compute
C) API calls incur network latency costs that reduce throughput and require additional infrastructure
D) API pricing scales linearly with token usage while self-hosted costs are mostly fixed after initial setup

<details>
<summary>Answer</summary>

**Correct: D**

This is the fundamental cost trade-off: API pricing (e.g., $0.0015/1K tokens for GPT-3.5-turbo) scales with usage—1M requests × 650 tokens ≈ $975/month and grows linearly. Self-hosted GPU costs (e.g., $1.20/hour for A10G) are ~$900/month fixed, regardless of request volume. At scale, self-hosted becomes cheaper; at low/variable volume, API wins. Exam tests this trade-off.

**Why the others are wrong:**
- Renting cloud GPUs (AWS, GCP, Lambda Labs) eliminates upfront purchase costs. The comparison is rental GPU cost vs API cost, not purchase vs subscription. Most companies rent GPUs, not buy hardware.
- While maintenance is real, the PRIMARY cost driver is compute/tokens. A small team can maintain self-hosted models with automation. The exam focuses on direct cost comparison ($/request), not organizational overhead.
- Network latency is a performance consideration, not a cost factor. API latency is ~500ms-2s; self-hosted is ~50-200ms. Doesn't significantly impact infrastructure costs—it's about speed/user experience, not expenses.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** An application uses the Hugging Face Transformers library to generate text with GPT-2. The developer needs to prevent repetitive output and ensure diverse completions. Which generation parameter should be adjusted?

A) Set temperature=0.1 to make outputs more deterministic and consistent
B) Set top_k=50 and top_p=0.9 to sample from a filtered distribution of likely tokens
C) Set num_beams=5 to use beam search for finding the highest probability sequence
D) Set max_length=512 to allow longer context for more varied responses

<details>
<summary>Answer</summary>

**Correct: B**

Repetitive output comes from greedy/deterministic sampling. top_k and top_p (nucleus sampling) introduce controlled randomness by sampling from the top K tokens or top cumulative probability p. This prevents the model from always picking the single highest-probability token (which causes loops). Standard technique for diverse, creative generation.

**Why the others are wrong:**
- temperature=0.1 makes outputs MORE deterministic (closer to greedy), not less. Low temperature reduces diversity and can increase repetition. Need higher temperature (0.7-1.0) or sampling parameters for diversity.
- Beam search finds high-probability sequences but is still deterministic for a given beam width. It reduces randomness and can produce repetitive patterns, especially in open-ended generation. Not the right tool for diversity.
- max_length controls output length, not diversity. Longer outputs don't inherently prevent repetition—a repetitive model will just repeat for longer. Need sampling parameters to change token selection behavior.

</details>

---

**Q7.** A company builds a RAG system where user queries are embedded using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) and searched against a FAISS index of 10 million document chunks. Queries return irrelevant results despite documents containing correct information. Which action is MOST likely to improve retrieval accuracy?

A) Re-embed documents using a domain-specific or larger embedding model (e.g., all-mpnet-base-v2 with 768 dimensions)
B) Switch from FAISS IndexFlatL2 to IndexIVFFlat with 1000 clusters to improve search speed
C) Increase the number of retrieved chunks (k) from 5 to 20 to ensure relevant documents are included
D) Implement query expansion by generating multiple paraphrases of the user query before embedding

<details>
<summary>Answer</summary>

**Correct: A**

Retrieval quality depends heavily on embedding model quality. all-MiniLM-L6-v2 is fast but less accurate than larger models. If documents are domain-specific (legal, medical, technical), a generic embedding model may not capture semantic nuance. Using a better/domain-adapted model (all-mpnet-base-v2, domain-fine-tuned embeddings) directly improves vector similarity and retrieval relevance.

**Why the others are wrong:**
- IndexIVFFlat is for speed, not accuracy. It uses approximate search which may slightly reduce accuracy compared to IndexFlatL2 (exact search). Won't improve retrieval quality—may even slightly hurt it. This is the opposite of what's needed.
- Increasing k retrieves more chunks but doesn't fix the core issue: the wrong documents are ranking highly. If the top 5 are all irrelevant, top 20 will also be mostly irrelevant. Need better embeddings to rank correct documents higher, not just retrieve more.
- Query expansion can help but is a secondary technique. If embeddings are poor quality, expanding queries won't fix the fundamental semantic mismatch. Better embeddings have a larger direct impact. Exam favors root cause fixes over workarounds.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A developer uses LangChain to create a chain that first retrieves relevant documents from a vector store, then generates an answer using an LLM. Which LangChain chain type is specifically designed for this pattern?

A) LLMChain with a custom prompt template that includes retrieved document text
B) RetrievalQA or ConversationalRetrievalChain with a VectorStoreRetriever
C) TransformChain that pre-processes input before passing to an LLM
D) SimpleSequentialChain with separate retrieval and generation steps

<details>
<summary>Answer</summary>

**Correct: B**

RetrievalQA and ConversationalRetrievalChain are purpose-built for RAG. They encapsulate: 1) retrieval from a vector store using a Retriever, 2) prompt construction with retrieved docs, 3) LLM generation with context. This is the standard LangChain abstraction for question-answering over documents. ConversationalRetrievalChain adds conversation memory.

**Why the others are wrong:**
- LLMChain is the base chain for LLM calls with prompts. Could manually build RAG with it, but requires writing custom retrieval logic and prompt engineering. RetrievalQA is the higher-level, purpose-built abstraction that handles this automatically.
- TransformChain applies a function to inputs/outputs (data transformation). Doesn't handle retrieval or LLM calls—it's for preprocessing/post-processing in a chain pipeline. Not designed for RAG pattern.
- SimpleSequentialChain runs chains in sequence, passing outputs forward. Could implement RAG manually, but RetrievalQA is the idiomatic LangChain way. Exam tests knowledge of built-in RAG chains, not DIY approaches.

</details>

---

