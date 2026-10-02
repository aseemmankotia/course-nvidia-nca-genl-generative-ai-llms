# Answer Key — Practice Test 2

**Q1. Correct: D**

The symptoms clearly indicate overfitting: training loss decreasing while validation loss plateaus/increases, and a 16-point accuracy gap between training and validation. Dropout (randomly dropping units during training) and L2 regularization (penalizing large weights) are the standard techniques to reduce overfitting by preventing the model from memorizing training data.

---

**Q2. Correct: A**

ReLU and LeakyReLU solve vanishing gradients because their derivatives are either 1 (for positive inputs) or a small constant (LeakyReLU for negative inputs), allowing gradients to flow backward through many layers without exponentially decreasing. ReLU has become the standard for deep networks specifically because it eliminates the vanishing gradient problem that plagues sigmoid and tanh in deep architectures.

---

**Q3. Correct: B**

Random forests often exhibit diminishing returns after a certain number of trees—accuracy plateaus while inference time scales linearly with tree count. Reducing from 200 to 40-50 trees (a 75% reduction) would reduce inference time proportionally while likely maintaining near-identical accuracy since the model is already showing minimal overfitting (0.4% gap) and random forests are robust to tree count reduction after the initial plateau. This directly addresses the speed bottleneck.

---

**Q4. Correct: D**

This addresses class imbalance at both data and loss levels. SMOTE creates synthetic examples of the minority class to balance training. Focal loss or class-weighted loss penalizes misclassification of the minority class more heavily (e.g., weight=19 for defective, weight=1 for non-defective), forcing the model to prioritize learning defect patterns rather than just predicting the majority class. This combination is the standard solution for severe class imbalance in neural networks.

---

**Q5. Correct: A**

This accurately describes max pooling's dual purpose: (1) downsampling spatial dimensions (256→128→64→32 after three 2×2 poolings with stride 2), reducing computation and memory by 75% per layer, and (2) providing translation invariance—if a feature moves slightly within a pooled region, the maximum is still captured. 'Retaining most prominent features' is correct because max pooling keeps the strongest activation in each region, which typically represents detected features.

---

**Q6. Correct: C**

LoRA and adapter-based methods are specifically designed for this scenario: limited data and compute for large pre-trained models. LoRA trains low-rank matrices (typically reducing trainable parameters from 250M to 1-2M) that are applied to attention layers, achieving 95-98% of full fine-tuning performance while being 3-10x faster and using far less memory. This addresses the compute constraint and reduces overfitting (the 0.92 training loss vs 1.84 validation loss gap) by drastically limiting trainable parameters. This is the current standard approach (2024-2026) for fine-tuning large language models with limited resources.

---

**Q7. Correct: C**

ALiBi removes absolute positional embeddings and uses linear biases in attention scores, allowing the model to extrapolate to sequences longer than those seen during training without architectural changes or retraining. This maintains computational efficiency while enabling better length generalization.

---

**Q8. Correct: D**

Expert collapse in MoE models typically occurs when the auxiliary load balancing loss is too weak. This loss term encourages equal distribution of inputs across experts. When set too low, the router learns to favor certain experts, creating a feedback loop where popular experts get better training signal and become even more preferred.

---

**Q9. Correct: D**

RAG systems store knowledge externally in retrievable databases rather than encoded in model weights. This allows complete and verifiable deletion of specific information by removing entries from the vector database, while the base model remains unchanged. This is the only architecture that provides true 'machine unlearning' capability.

---

**Q10. Correct: B**

Language-specific adapter layers add trainable parameters for each language while keeping the core GQA mechanism intact. This allows low-resource languages to have dedicated representational capacity without increasing the memory costs of the attention mechanism itself. Adapters have proven effective for multilingual models where different languages have varying complexity.

---

**Q11. Correct: A**

Flash attention optimizes memory during the forward pass by avoiding materialization of the full attention matrix. However, the backward pass still requires attention matrices. Without gradient checkpointing, all intermediate activations must be stored. Flash attention with gradient checkpointing recomputes attention during backward pass, trading computation for memory—this combination is essential for long sequences.

---

**Q12. Correct: B**

Clinical notes have structured sections (diagnosis, history, medications, etc.) where section headers contain critical organizational information. Longformer's hybrid approach uses efficient local attention for detailed context while designated global tokens (section headers) can attend to and be attended by all tokens, enabling long-range dependencies. This matches the document structure and information flow of clinical notes.

---

**Q13. Correct: D**

RAG is the most practical solution for hallucination reduction with limited compute. It doesn't require model retraining, instead augmenting inference with retrieved factual context. The model uses retrieved medical documents to ground its responses, directly addressing hallucination without changing learned weights. This is a standard production approach for reducing hallucination in specialized domains.

---

**Q14. Correct: A**

Chain-of-thought with explicit NULL handling and structured output (JSON schema) directly addresses hallucination by forcing the model to acknowledge missing data rather than fabricate it. Structured output constraints in modern LLMs (2026) enforce schema compliance and type validation, preventing numerical hallucinations.

---

**Q15. Correct: B**

Hierarchical prompt design optimizes for the common case (85% of requests) by keeping them under context limits while handling edge cases separately. This directly addresses the performance issue by preventing most requests from hitting the expensive maximum context processing path, reducing both latency and cost.

---

**Q16. Correct: C**

Few-shot example selection bias is the root cause. When examples heavily weight common conditions, the model learns to pattern-match toward those exemplars. Rare conditions lack representation, causing the model to force-fit them into similar common categories. This is a classic prompt engineering failure in imbalanced domains.

---

**Q17. Correct: A**

Micro-task decomposition isolates each problematic clause with its specific critique, eliminating information overload (4650 total tokens) and ensuring focused attention on each issue. This architectural pattern prevents critique dilution in large contexts and makes validation explicit per clause—directly addressing the root cause of ignored critiques.

---

**Q18. Correct: C**

Architectural separation (two-model system) provides guaranteed policy accuracy by routing policy content through a specialized model optimized for factual accuracy and source citation, while preserving the persona model's empathy benefits for non-policy interactions. This enforces the compliance boundary at the system level rather than relying on prompt instructions.

---

**Q19. Correct: B**

Token optimization of Strategy B (reducing 1240→900 tokens via compression) maintains accuracy above 75% threshold while minimizing cost. At 900 tokens × 2M problems × $0.02/1K = $36,000/month, this provides the best cost-effectiveness meeting requirements. Few-shot examples can be compressed (removing verbose steps, using math notation) without losing pattern-matching value.

---

**Q20. Correct: A**

Collecting real data from underrepresented populations and retraining with stratified sampling directly addresses the root cause of bias (unrepresentative training data). This approach maintains scientific validity, ensures the model learns genuine patterns from diverse populations, and validates performance across subgroups before deployment—core responsible AI practices for fairness.

---

**Q21. Correct: C**

Prompt injection requires defense-in-depth because no single layer is foolproof. This approach combines: (1) input validation to catch obvious attacks, (2) access controls at retrieval time (the critical security boundary), (3) output filtering as a last line of defense, and (4) monitoring to detect novel attacks. Enforcing permissions at retrieval time ensures the RAG system never accesses unauthorized documents regardless of prompt manipulation.

---

**Q22. Correct: A**

Citation-based RAG provides verifiable traceability by explicitly linking generated claims to specific source documents. The system retrieves identifiable sources, instructs the model to cite them, and validates citations—enabling auditors to check that recommendations are grounded in approved information sources. This creates a clear audit trail meeting regulatory explainability standards for financial advice.

---

**Q23. Correct: D**

Structured access through gated APIs with use-case review provides a balance between research access and risk mitigation for models with dual-use capabilities. This allows monitoring for misuse, enables revocation if abuse occurs, maintains researcher access for safety research, and provides time to develop defenses before wider release. This aligns with current responsible AI release frameworks for high-risk capabilities.

---

**Q24. Correct: C**

This systematic analysis addresses potential root causes: unrepresentative training data, labeling bias where legitimate discussions of discrimination were mislabeled as violations, and cultural context issues where language patterns in minority communities were misunderstood. High overall accuracy can mask serious subgroup performance disparities. This diagnostic approach identifies specific failure modes enabling targeted fixes.

---

**Q25. Correct: C**

Data leakage is the most critical issue. When SMOTE is applied before train/test split, synthetic samples in training data may be created using information from test samples, leading to artificially inflated validation metrics that don't translate to production. SMOTE must always be applied only to training data after splitting to prevent leakage.

---

**Q26. Correct: D**

The dramatic accuracy drop from backtesting to production is a classic symptom of forward-looking bias (data leakage in time series). When calculating rolling statistics on historical data without proper shifting, the feature at time t includes information from time t+1, t+2, etc. In production, future data isn't available, causing the performance gap. Using .shift(1) or equivalent ensures each rolling feature uses only data available at prediction time.

---

**Q27. Correct: C**

Providing multiple representations allows tree-based models like gradient boosting to automatically learn the most useful representation for different regions of the data. The original amount preserves true values, log transformation handles magnitude scaling, and the binary flag captures a threshold effect. Gradient boosting models handle mixed-scale features well and benefit from this flexibility. This approach is robust and commonly used in practice for features with complex distributions.

---

**Q28. Correct: B**

This solution directly addresses the business need: identifying specific transactions for review. The histogram with color-coded thresholds provides quick visual assessment of confidence distribution, while the filtered table enables actionable review by showing specific transaction IDs that need attention. This combines analytical visualization with operational utility, making it actionable not just informative.

---

**Q29. Correct: A**

L2 regularization penalizes coefficients based on their squared magnitude. When features have vastly different scales, regularization disproportionately affects features with larger numerical ranges because their coefficients would naturally be smaller (a coefficient of 0.00001 for 'age_x_income' might represent the same predictive contribution as a coefficient of 0.1 for 'age'). The squared penalty amplifies this disparity. StandardScaler equalizes feature scales so regularization applies fairly across all features, allowing 'age_x_income' to demonstrate its true predictive value.

---

**Q30. Correct: A**

The symptom—good validation performance but poor production performance across different regions—strongly indicates data leakage. When target encoding is applied before splitting, test set categories are encoded using statistics that include test set target values. This artificially inflates validation performance. In production with truly unseen data, this leaked information isn't available, causing the performance drop. Correct implementation computes encoding statistics only from training data and applies those statistics to validation/test/production data.

---

**Q31. Correct: D**

Strategy B demonstrates better generalization despite lower validation accuracy. In production AI systems, especially for critical applications like medical diagnosis, consistent performance across different data distributions is more valuable than peak performance on a single validation set. The exam tests understanding that overfitting to validation data and poor generalization are more dangerous than slightly lower accuracy metrics.

---

**Q32. Correct: B**

The exam tests recognition that proxy metrics (CTR) can diverge from business metrics (purchases). A 3-day test period is insufficient to capture user behavior patterns, and the actual business outcome is negative. The test duration concern combined with the wrong optimization target (CTR vs. purchases) is the critical evaluation flaw that should halt deployment.

---

**Q33. Correct: D**

The exam tests understanding that evaluation metrics must align with real-world task requirements. For legal document analysis, semantic correctness matters more than exact text span matching. Reporting both metrics with clear context allows stakeholders to understand true system capabilities while maintaining transparency about measurement approaches. This demonstrates mature evaluation practice.

---

**Q34. Correct: B**

The exam tests comprehensive evaluation and mitigation strategy for adversarial robustness. The correct approach includes creating persistent evaluation infrastructure (adversarial benchmark), improving the model (augmented training), establishing ongoing testing (continuous red-teaming), and production monitoring. This addresses both immediate and future adversarial threats systematically.

---

**Q35. Correct: C**

The exam tests efficient evaluation design for preference-based assessments. Pairwise comparisons provide clearer preference signals and require fewer total evaluations for significance. LLM-as-judge approaches (with careful validation) can scale human-like judgments while maintaining quality for well-defined tasks like financial summary assessment. This balances rigor with practical evaluation costs.

---

**Q36. Correct: A**

L40S GPUs provide optimal price-performance for 70B models with 48GB memory each. Tensor parallelism degree 4 distributes the model across 4 GPUs efficiently. Inflight batching (continuous batching) dynamically schedules requests to maximize throughput while meeting latency SLAs. KV cache management significantly reduces latency for multi-turn conversations and maintains the 2-second P95 target with 500 concurrent users.

---

**Q37. Correct: C**

Triton Inference Server (the foundation of NIM) natively supports multi-model serving with dynamic loading/unloading based on demand. The shared GPU memory pool with dynamic allocation maximizes utilization across varying workloads. Model ensembles in Triton allow different preprocessing pipelines to be chained with the model inference, handling the different tokenization schemes efficiently. This approach minimizes idle GPU memory and allows automatic scaling per model.

---

**Q38. Correct: B**

Inconsistent retrieval quality despite normalized embeddings indicates the initial dense retrieval is not differentiating well between relevant and semi-relevant documents. NeMo Reranker uses a cross-encoder model that evaluates query-document pairs jointly (versus embedding them separately), providing much more accurate relevance scoring. This two-stage retrieval (fast dense retrieval + precise reranking) is the standard architecture for production RAG systems and directly addresses the described symptom of inconsistent precision.

---

**Q39. Correct: D**

NeMo Framework with Megatron-Core provides native support for 3D parallelism (tensor, pipeline, and data parallelism) required for this configuration. Transformer Engine accelerates transformer training with FP8 precision on Hopper/Ampere architectures. NCCL handles multi-node communication. CUDA Toolkit provides the base layer. The FP8 recipe optimizes training speed while maintaining model quality. This is the complete stack for large-scale transformer training.

---

**Q40. Correct: D**

The hybrid approach addresses the key constraint: 1000 streams × 30 FPS = 30,000 inferences/second is computationally prohibitive for full LLM analysis. Content moderation has high class imbalance—most frames are policy-compliant. Edge-based initial filtering using lightweight Maxine models dramatically reduces the load requiring expensive LLM inference (typically 80-90% reduction). This maximizes cost-efficiency while maintaining comprehensive coverage. The two-stage approach is the industry-standard pattern for high-volume moderation.

---

**Q41. Correct: B**

NIM includes TensorRT-LLM optimizations by default—there's no performance sacrifice versus custom deployment for standard models like Mistral 7B. NIM's container architecture supports multiple model versions and can be deployed on-premises with NVIDIA AI Enterprise, satisfying security requirements. NIM provides production-grade observability, automatic scaling, and health checks out-of-box, significantly reducing operational complexity. For A/B testing, NIM supports multiple concurrent model deployments with traffic routing. Time-to-production is weeks faster than custom TensorRT-LLM builds.

---

**Q42. Correct: A**

This solution provides true hardware-level isolation using MIG instances (preventing cross-tenant data leakage at the GPU memory level), while dynamic batching with properly sized batches optimizes throughput. Separate containerized Triton instances per tenant ensures complete process isolation, and CUDA_VISIBLE_DEVICES correctly maps containers to specific MIG slices. This architecture guarantees both the security and performance SLAs required for multi-tenant financial services.

---

**Q43. Correct: B**

This directly addresses the identified bottleneck (attention mechanism and KV cache memory bandwidth). Flash Attention 2 reduces attention complexity from O(N²) to O(N), paged attention eliminates memory fragmentation for variable-length sequences, and INT8 KV cache quantization cuts memory bandwidth requirements by 50% while maintaining quality. On H100, this combination specifically targets the attention and KV cache read bottleneck identified in profiling. TensorRT-LLM's --use_paged_kv_cache is designed exactly for this optimization.

---

**Q44. Correct: C**

The bottleneck is vector search at 180ms (90% of the target SLA). NVIDIA RAFT CAGRA provides GPU-accelerated graph-based ANN search with 3-10x speedup over FAISS IVF for this scale, easily bringing search latency to 20-60ms. Storing embeddings in GPU memory eliminates PCIe transfer overhead. Query batching for concurrent requests leverages GPU parallelism without adding per-query latency. This directly addresses the bottleneck with NVIDIA's purpose-built solution for GPU-accelerated vector search, which is specifically designed for NeMo Retriever integration.

---

