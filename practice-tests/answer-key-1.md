# Answer Key — Practice Test 1

**Q1. Correct: C**

Gradient boosting models excel at tabular data with mixed feature types, provide probability outputs, and SHAP (SHapley Additive exPlanations) values offer mathematically rigorous, per-prediction explanations that satisfy regulatory requirements. This combination delivers both high accuracy and interpretability.

---

**Q2. Correct: B**

The divergence between training and validation performance (94% vs 78%) combined with increasing validation loss while training loss decreases is the classic signature of overfitting. Dropout prevents co-adaptation of neurons and early stopping would have halted training at epoch 15 when validation performance peaked.

---

**Q3. Correct: D**

After first conv: 64×64×32. After first pool: 32×32×32. After second conv: 32×32×64. After second pool: 16×16×64 = 16,384 features. Dense layer parameters = (input_features × output_neurons) + bias = (16,384 × 128) + 128 = 2,097,152 + 128 = 2,097,280.

---

**Q4. Correct: B**

Sigmoid activation applied to each output independently produces values between 0 and 1 for each class without constraining them to sum to 1. This enables multi-label classification where multiple outputs can simultaneously be positive. Use binary cross-entropy loss for training.

---

**Q5. Correct: D**

The gradient magnitude of 0.0001 in early time steps indicates vanishing gradient problem, which is endemic to standard RNNs processing long sequences. LSTMs and GRUs use gating mechanisms (forget gates, input gates) that maintain a constant error carousel, allowing gradients to flow across many time steps without vanishing.

---

**Q6. Correct: C**

Recall measures the proportion of actual fraud cases correctly identified (true positives / [true positives + false negatives]). When missing fraud (false negative) costs 10× more than false alarms (false positive), maximizing recall is the priority. Model B catches 88% of fraud cases vs Model A's 60%, making it worth the increase in false alarms.

---

**Q7. Correct: C**

Self-attention allows transformers to compute direct relationships between any two tokens in O(1) steps, regardless of their distance in the sequence. This eliminates the vanishing gradient problem that prevents RNNs from learning long-range dependencies beyond a few hundred tokens.

---

**Q8. Correct: D**

Low temperature (0.2) sharpens the probability distribution, making the model strongly favor the highest-probability tokens. This produces more deterministic, conservative outputs that stick closely to patterns seen during training, reducing randomness.

---

**Q9. Correct: C**

Tokenizers have fixed vocabularies created during pre-training. If fine-tuning data contains hospital-specific terms or abbreviations not in the original vocabulary, they get split into subword tokens or marked as unknown, losing semantic meaning. Different hospitals' terminology may tokenize completely differently, breaking generalization.

---

**Q10. Correct: A**

KV-cache stores computed attention keys and values from previous tokens in autoregressive generation, eliminating redundant computation for each new token. This directly reduces per-token latency by avoiding recomputation of the entire attention mechanism for the conversation history.

---

**Q11. Correct: C**

When deeper layers show lower activation diversity and repetitive attention patterns, they are not learning distinct features beyond middle layers. This suggests the model has more depth than needed for the task complexity, and removing layers could reduce computational cost without sacrificing performance - a common finding in transformer analysis.

---

**Q12. Correct: B**

Flash Attention specifically addresses the memory bottleneck of standard attention by fusing operations and recomputing attention scores during backprop rather than storing the O(n²) attention matrix. For 4,096 tokens with batch size 8, standard attention stores massive intermediate activations; Flash Attention reduces peak memory significantly while maintaining exact attention computation.

---

**Q13. Correct: A**

RoPE (Rotary Position Embeddings) enables better extrapolation to longer sequences than the model was trained on by encoding relative positional information through rotation matrices. This is specifically effective for maintaining coherence in longer contexts (200+ tokens) without the quadratic memory cost of extended context windows. Models like CodeLlama and GPT-NeoX use RoPE precisely for improved long-range dependencies in code generation tasks.

---

**Q14. Correct: A**

Few-shot learning with examples and explanations teaches the model the logical patterns needed for accurate SQL generation. Including reasoning helps the model understand WHY certain JOINs or clauses are used, directly addressing the logical correctness issue.

---

**Q15. Correct: A**

Structured output modes (like OpenAI's JSON mode with schema or Anthropic's tool use) enforce the output format at the model level, guaranteeing the response matches the specified JSON schema. This is the most reliable method for consistent formatting.

---

**Q16. Correct: A**

Chain-of-thought prompting forces the model to explicitly ground its reasoning in the actual code elements first, creating a reference list it must then use for documentation. This intermediate step significantly reduces fabrication by making the model's reasoning explicit and checkable.

---

**Q17. Correct: C**

RAG retrieves only the relevant policy information for each specific query, minimizing context size while ensuring accuracy. This is the standard pattern for handling large knowledge bases - it's more flexible than pre-routing and doesn't lose information like compression.

---

**Q18. Correct: D**

Providing explicit definitions and examples directly addresses the two stated problems: missing details (examples show what to look for) and term confusion (definitions clarify distinctions). This is targeted few-shot learning with domain-specific guidance.

---

**Q19. Correct: D**

A deterministic filter provides guaranteed enforcement - it cannot be bypassed by prompt injection or model mistakes. While not pure prompt engineering, it's the most ROBUST safeguard. For critical compliance requirements, non-LLM guardrails are more reliable than prompting techniques.

---

**Q20. Correct: D**

Human-in-the-loop workflows are critical for high-stakes decisions like credit approval. This approach ensures human oversight, maintains accountability, creates an audit trail, and allows humans to catch potential bias or errors before they impact customers. This aligns with responsible AI principles requiring human agency and oversight in consequential decisions.

---

**Q21. Correct: C**

This defense-in-depth approach addresses the problem at multiple levels: output filtering provides immediate protection, retraining on sanitized data eliminates the root cause (PII in training data), and regular adversarial testing ensures ongoing security. This comprehensive strategy follows responsible AI safety principles by combining preventive, detective, and corrective controls.

---

**Q22. Correct: B**

Constitutional AI is a technique developed by Anthropic that trains models to self-critique and self-revise outputs based on a defined set of principles (the 'constitution'). This allows the model to identify and correct potentially harmful responses through a two-phase process: supervised learning and reinforcement learning from AI feedback (RLAIF), reducing the need for extensive human labeling of harmful content.

---

**Q23. Correct: A**

The root cause is insufficient representation of darker skin tones in the training data (8% vs likely 30-40% of actual population). Collecting real additional data from underrepresented groups and retraining with balanced sampling directly addresses the data imbalance that causes the performance disparity. This is the gold standard approach for mitigating algorithmic bias caused by training data imbalance.

---

**Q24. Correct: B**

Reward hacking occurs when a model exploits misspecified or incomplete reward functions to achieve high scores without truly aligned behavior. A reward model that overemphasizes narrow metrics (brevity, speed) will cause the AI to optimize for those at the expense of critical factors like security and maintainability—exactly the definition of reward hacking. The model might generate fast, short code with vulnerabilities because that's what gets rewarded.

---

**Q25. Correct: D**

Cyclical features using sine/cosine transformations preserve the periodic nature of temporal data (e.g., December is close to January, Sunday wraps to Monday). This is critical for seasonal patterns where the model needs to understand that month 12 and month 1 are adjacent in the cycle, not 11 months apart.

---

**Q26. Correct: D**

StandardScaler (z-score normalization) is the primary solution for features with different scales in gradient-based models. It transforms each feature to have mean=0 and standard deviation=1, ensuring all features contribute equally to gradient calculations and distance metrics. This directly addresses the slow convergence issue caused by scale differences.

---

**Q27. Correct: D**

KNN imputation is optimal here because: (1) the feature has high correlation with target (r=0.72), making it critical to preserve; (2) KNN uses similar records to estimate missing values, preserving local patterns and relationships; (3) MCAR assumption is met, making neighbor-based imputation valid; (4) business requires all data points, ruling out deletion; (5) KNN respects feature correlations better than simple statistical measures.

---

**Q28. Correct: B**

Faceted violin plots optimally handle this scenario: (1) facets (small multiples) separate product_category and region without overlap; (2) violin plots show full distribution shape of satisfaction scores, revealing multimodality and density; (3) hue encoding adds support_tier as a third dimension within each facet; (4) executive-friendly by showing patterns at a glance; (5) scales well to large datasets (50k records). This combines distribution visualization with multi-categorical comparison.

---

**Q29. Correct: B**

This approach is correct because: (1) customer_id is a unique identifier with no predictive value for segmentation patterns—it must be dropped to prevent data leakage and overfitting; (2) log transformation addresses the extreme right skew by compressing large values while expanding small values; (3) StandardScaler after log transformation normalizes the now-symmetric distribution. This pipeline handles both the identifier problem and the skewness effectively for clustering algorithms.

---

**Q30. Correct: C**

This pattern demonstrates successful feature engineering. The raw dates (last_purchase_date, current_date) have low predictive value individually because models struggle with absolute date values. The derived feature 'days_since_last_purchase' extracts the meaningful temporal relationship (recency) that actually drives predictions. High feature importance (0.45) indicates it captures a true pattern in the data—recency is often a strong predictor in customer behavior models. This is the goal of feature engineering: transform raw data into representations that expose predictive patterns.

---

**Q31. Correct: A**

Semantic similarity using embeddings (e.g., sentence-transformers, OpenAI embeddings) captures meaning and intent rather than exact wording. This is ideal for evaluating support responses where different phrasings can convey the same helpful information. Cosine similarity on embeddings is the industry standard for semantic evaluation in 2026.

---

**Q32. Correct: B**

A 0.4 percentage point conversion difference (4.2% vs 3.8%) is relatively small, while 15% higher satisfaction is substantial. Research consistently shows customer satisfaction drives retention, repeat purchases, and lifetime value. Version B's creative, varied descriptions (higher temperature) likely build more trust and engagement, leading to better long-term revenue despite slightly lower immediate conversion.

---

**Q33. Correct: A**

Natural Language Inference (entailment checking) directly tests whether each claim in the generated summary is supported by the source document. NLI models can classify each sentence as entailed, neutral, or contradicted by the source, explicitly catching hallucinations where the model invents clinical facts. This is the gold standard for factual consistency evaluation in 2026.

---

**Q34. Correct: A**

CoT provides 7 percentage points improvement over direct prompting (78% vs 71%) at modest additional cost, while ReAct's extra 7 points (85% vs 78%) costs 3.2x more. At 50k daily queries (1.5M monthly), ReAct would exceed the $15k budget. CoT offers the best accuracy-per-dollar and fits within budget constraints while delivering substantial improvement over baseline.

---

**Q35. Correct: C**

71% recall means the system misses 29% of unsafe content—a significant safety risk. The business explicitly prioritizes user protection over false positive friction. Lowering the threshold increases recall (catches more unsafe content) at the cost of precision (more false positives). High precision (94%) indicates room to trade precision for recall while maintaining acceptable accuracy.

---

**Q36. Correct: A**

Triton Inference Server with TensorRT-LLM provides the lowest latency for LLM inference through kernel fusion, quantization (FP8/INT8), and in-flight batching. Dynamic batching optimizes throughput for concurrent requests. A100 GPUs provide sufficient memory bandwidth for 70B models with TensorRT-LLM optimizations.

---

**Q37. Correct: B**

NeMo Retriever embedding models (typically E5-large or similar) require ~2-4GB. FAISS GPU indices for efficient vector search require additional memory (~8-12GB for index structures). Batch processing buffers and intermediate tensors need ~8-10GB. Total: 24GB minimum (A10/A30/A100 24GB tier).

---

**Q38. Correct: A**

NCCL (NVIDIA Collective Communications Library) timeout specifically indicates network communication failure between GPUs. With tensor_model_parallel=4 and pipeline_model_parallel=2, this is an 8-GPU distributed job requiring high-bandwidth inter-GPU communication. InfiniBand or NVLink issues are the primary cause of NCCL timeouts.

---

**Q39. Correct: B**

FP8 quantization on Hopper (H100/H200) with Transformer Engine provides 2-4x speedup over FP16 while maintaining >99% quality. Hopper's hardware FP8 Tensor Cores and Transformer Engine's per-tensor scaling maintain accuracy. This is the best performance-quality tradeoff for production LLM deployment.

---

**Q40. Correct: D**

Python backend in Triton allows loading separate model instances with different LoRA adapters. Setting instance_group count creates isolated model instances, each with dedicated GPU memory. Each instance can load base model + specific LoRA adapter. This enables true multi-tenancy with isolation. Triton routes requests to appropriate instances.

---

**Q41. Correct: C**

Mixtral 8x7B has ~47B parameters. At FP16 (2 bytes), this is ~94GB. At 4-bit GPTQ (0.5 bytes), this is ~24GB for weights + ~10-12GB for activations/KV cache = ~35GB total, fitting in 40GB A100. TensorRT-LLM has optimized GPTQ kernels. 4-bit is the sweet spot for this GPU.

---

**Q42. Correct: B**

Triton Inference Server with TensorRT-LLM provides the best combination for production LLM deployment with strict latency requirements. TensorRT-LLM optimizes inference performance (reducing latency by 3-8x compared to native PyTorch), while Triton's dynamic batching efficiently handles concurrent requests. This combination is specifically designed for production workloads requiring sub-200ms latency at scale.

---

**Q43. Correct: D**

NeMo Retriever provides dedicated embedding and re-ranking microservices that are specifically optimized for RAG pipelines. These microservices are deployable as NVIDIA Inference Microservices (NIMs), providing production-ready APIs. The embedding service generates vector representations while the re-ranking service improves retrieval precision—both designed to work together in the NeMo Retriever architecture.

---

**Q44. Correct: C**

PagedAttention directly addresses KV-cache memory management by treating the cache as virtual memory with paged blocks. This eliminates memory fragmentation and allows dynamic allocation, typically reducing KV-cache memory waste by 60-80%. This is the most direct solution for increasing batch size when KV-cache memory is the bottleneck. vLLM pioneered this approach specifically to solve the KV-cache memory problem.

---

