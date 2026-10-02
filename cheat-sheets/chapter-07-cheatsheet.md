# Chapter 7 Cheat Sheet: LLM Integration and Deployment (RAG, APIs, Libraries)

## RAG (Retrieval-Augmented Generation)

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| RAG | Combine retrieval + generation | Private/domain data, reduce hallucination, cite sources |
| Fine-tuning | Adapt model weights | Change style/format/domain permanently |
| Prompt engineering | In-context learning | Few examples, no data storage needed |

**RAG Architecture**: Query → Embed → Vector Search → Retrieve Top-K Chunks → Inject into Prompt → LLM Generate

## Vector Databases & Search

| Database | Type | Best For |
|----------|------|----------|
| FAISS | Library (local) | Prototyping, <10M vectors |
| Pinecone | Managed service | Production, managed infrastructure |
| Weaviate | Self-hosted/managed | GraphQL queries, hybrid search |
| Milvus | Self-hosted | Large scale, on-prem |

**Search Algorithms**:
- **Exact k-NN** (IndexFlatL2): 100% recall, O(n) time—only for <100K vectors
- **ANN** (IndexIVFFlat, HNSW): 90-99% recall, <100ms at millions of vectors
- **Exam Rule**: Production RAG >100K vectors → MUST use ANN indexing

## Hugging Face Transformers

**Load Model (3 Methods)**:
```python
# Method 1: Pipeline (highest level)
from transformers import pipeline
classifier = pipeline('sentiment-analysis')

# Method 2: AutoModel (task-specific)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Method 3: Specific architecture
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```

**Inference Methods**:
- `pipeline()`: Simplest—handles everything automatically
- `model.generate()`: Text generation with control (temp, top_k, top_p)
- `model(**inputs)`: Raw forward pass, manual post-processing

**Generation Parameters**:
| Parameter | Effect | Values |
|-----------|--------|--------|
| temperature | Randomness | 0.1 (deterministic) to 2.0 (creative) |
| top_k | Limit sampling pool | 50 (typical), higher = more diverse |
| top_p | Nucleus sampling | 0.9 (typical), cumulative probability cutoff |
| num_beams | Beam search width | 1 (greedy), 5-10 (better quality, slower) |
| do_sample | Enable sampling | True (diverse), False (greedy/deterministic) |

## LangChain Components

**Chains**:
- **LLMChain**: Single LLM call + prompt template (base building block)
- **RetrievalQA**: RAG pattern—retrieve docs from vector store + answer with LLM
- **ConversationalRetrievalChain**: RetrievalQA + conversation memory
- **SequentialChain**: Run chains in sequence, output → next input

**Memory Types**:
- **ConversationBufferMemory**: Stores entire conversation verbatim (grows with length)
- **ConversationSummaryMemory**: Summarizes conversation (saves tokens)
- **ConversationBufferWindowMemory**: Keeps last K messages only

**Agents**: LLM decides which tools to use dynamically (ReAct reasoning pattern)

**Typical RAG Setup**:
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings())
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vectorstore.as_retriever())
result = qa_chain.run("Your question here")
```

## API vs Self-Hosted Comparison

| Factor | API (OpenAI, Anthropic, Cohere) | Self-Hosted (LLaMA, Mistral) |
|--------|----------------------------------|-------------------------------|
| Cost structure | $/token (linear with usage) | Fixed GPU rental |
| Upfront cost | $0 | GPU setup (~$0-5000) |
| Scaling | Automatic, pay per use | Manual GPU provisioning |
| Latency | 500ms-2s (network) | 50-200ms (local) |
| Control | Limited (model version, params) | Full (quantization, serving) |
| Privacy | Data sent to provider | Data stays on your infra |
| Maintenance | Zero | Model updates, GPU management |

**Exam Decision Rule**:
- **Low/variable volume (<500K requests/month)**: API cheaper and simpler
- **High/predictable volume (>1M requests/month)**: Self-hosted becomes cost-effective
- **Strict data privacy**: Self-hosted required
- **Rapid prototyping**: API for speed

## Key Exam Traps

1. **RAG vs Fine-tuning**: RAG for grounding in external data; fine-tuning for style/format. Don't fine-tune when data changes frequently.
2. **Exact vs ANN Search**: Exact only viable <100K vectors. Production RAG at scale REQUIRES ANN.
3. **Memory in LangChain**: ConversationChain needs explicit Memory component—Agents and basic chains don't have memory by default.
4. **Pipeline simplicity**: Don't use model.generate() when pipeline() suffices. Exam favors simplest correct approach.
5. **Cost scaling**: API cost grows with usage; self-hosted cost mostly fixed. Calculate crossover point.
6. **Embedding model quality**: #1 factor in RAG accuracy. Better model > tuning k or search parameters.
7. **Generation diversity**: Low temperature + greedy = repetitive. Need sampling (top_k/top_p) + moderate temp for diversity.
