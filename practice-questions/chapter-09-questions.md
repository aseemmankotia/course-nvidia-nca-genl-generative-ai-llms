## Chapter 9: Putting It All Together—Real-World LLM Application Architecture — Practice Questions

**Q1.** A startup is building a customer support chatbot that must respond to 10,000 queries per day with <500ms latency. They have 5,000 labeled support tickets. The engineering team has 2 weeks to launch. Which approach BEST meets these requirements?

A) Fine-tune a 70B parameter model on the 5,000 tickets and deploy on 8×A100 GPUs
B) Train a custom 175B model from scratch using all available compute resources
C) Fine-tune GPT-4 on the tickets and deploy via OpenAI API with caching enabled
D) Use prompt engineering with a hosted 7B model API and implement RAG over the ticket database

<details>
<summary>Answer</summary>

**Correct: D**

Prompt engineering + RAG with a smaller hosted model meets latency requirements (<500ms), doesn't require extensive fine-tuning time (2 week constraint), and 5,000 samples is marginal for fine-tuning but perfect for RAG knowledge base. Hosted API eliminates infrastructure complexity.

**Why the others are wrong:**
- 70B model requires extensive GPU infrastructure ($50k+/month), fine-tuning 5,000 samples takes >2 weeks including data prep and validation, and inference latency will exceed 500ms even on 8×A100s
- Training from scratch requires months and millions of samples, not 5,000 tickets and 2 weeks. This violates both time and data constraints completely
- Cannot fine-tune GPT-4 (closed-source), and even if using fine-tuning API for GPT-3.5/4, 2-week timeline is tight for data prep, fine-tuning, and validation. Latency unpredictable on shared API

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An e-commerce company fine-tuned a 13B model for product recommendations. After 3 months in production, accuracy dropped from 87% to 72%. Logs show user preference patterns shifted significantly. What is the MOST cost-effective solution to restore performance?

A) Collect 6 months of new interaction data and completely retrain the base model from scratch
B) Implement prompt engineering with few-shot examples of new preference patterns while collecting data for future fine-tuning
C) Immediately fine-tune on the last 30 days of data using the existing model as the base
D) Switch to a larger 70B model which is more robust to distribution shifts

<details>
<summary>Answer</summary>

**Correct: B**

Prompt engineering with few-shot examples provides immediate improvement (days not weeks), zero fine-tuning cost, and allows parallel data collection for future fine-tuning if needed. This is the classic 'quick fix while preparing long-term solution' pattern.

**Why the others are wrong:**
- Retraining from scratch discards 3 months of learned weights, costs 10-50× more than incremental fine-tuning, and takes weeks to months. Wasteful when existing model is a good starting point
- 30 days of data may not be sufficient for robust fine-tuning and risks overfitting to very recent patterns. Need to validate data quantity and quality first before committing compute resources
- Larger models cost 5-10× more for inference (more GPU memory, slower throughput) and don't solve fundamental data drift issue. Model size ≠ adaptation to new distributions

</details>

---

**Q3.** A healthcare company needs to process 1 million patient records daily using an LLM for clinical note summarization. Average note is 2,000 tokens input, 200 tokens output. Current cost is $8,000/day using GPT-4 API at $0.03/1k input tokens and $0.06/1k output tokens. Which optimization reduces cost by at least 60% while maintaining quality?

A) Switch to batch inference with a self-hosted Llama 2 70B model on 4×A100 GPUs ($12k/month rental)
B) Fine-tune a 7B model on 10,000 notes and self-host on 1×A100 GPU ($2k/month rental)
C) Implement semantic caching to avoid re-processing similar notes (estimated 70% cache hit rate)
D) Use GPT-3.5-turbo instead of GPT-4 ($0.001/1k input, $0.002/1k output)

<details>
<summary>Answer</summary>

**Correct: B**

Current cost: 1M × (2k × 0.03/1k + 0.2k × 0.06/1k) = $60k + $12k = $72k/day = $2.16M/month. Fine-tuned 7B model: $2k rental + one-time fine-tuning ~$500 = ~$2.5k/month (~99% reduction). Quality maintained because task-specific fine-tuning on clinical notes works well with smaller models.

**Why the others are wrong:**
- 70B model rental: $12k/month is good savings BUT 70B model on 4×A100 will struggle with 1M daily inferences (throughput bottleneck). Need ~8-16 GPUs for this volume, raising cost to $24-48k/month (~35% reduction, not 60%)
- Caching math: 70% cache hit means 30% still processed = 0.3 × $72k/day = $21.6k/day = $648k/month (~70% reduction). Close, but exam asks for solution that 'reduces cost by at least 60% while maintaining quality'—caching is valid but fine-tuning option is more cost-effective and ALWAYS works regardless of cache hit variance
- GPT-3.5 cost: 1M × (2k × 0.001/1k + 0.2k × 0.002/1k) = $2k + $0.4k = $2.4k/day = $72k/month. This IS ~97% reduction BUT exam specifies 'while maintaining quality'—GPT-3.5 has significantly lower clinical summarization quality than GPT-4, likely unacceptable for healthcare

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A legal tech company deployed an LLM for contract analysis. Users report that the model performs well on standard contracts but fails on specialized agreements (IP licensing, M&A). They have 500 labeled examples of specialized contracts. What combination of techniques is MOST effective?

A) Fine-tune on all 500 examples and deprecate the original model entirely
B) Implement RAG with a vector database of specialized contracts + prompt engineering with domain-specific instructions
C) Fine-tune on 400 examples, reserve 100 for validation, then use RAG for edge cases not covered in training
D) Use prompt engineering only with detailed few-shot examples from the 500 contracts

<details>
<summary>Answer</summary>

**Correct: C**

Hybrid approach leverages fine-tuning for consistent specialized terminology and reasoning patterns (400 examples is sufficient for task-specific fine-tuning), proper train/val split ensures quality, and RAG handles the long-tail edge cases that don't appear frequently enough to learn via fine-tuning. This is the exam's preferred 'best of all worlds' pattern.

**Why the others are wrong:**
- Fine-tuning on all 500 without validation set risks overfitting and provides no way to measure performance. Also loses the base model's general legal knowledge—specialized contracts still use common legal language
- RAG + prompting works but doesn't teach the model specialized legal reasoning patterns. RAG retrieves similar examples but model still needs to 'understand' how to apply them. 500 labeled examples suggest fine-tuning is viable and would outperform pure RAG
- Few-shot prompting with 500 examples hits context length limits (most models <32k tokens). Cannot fit enough examples to cover all specialized patterns. Also more expensive (larger input contexts) and less consistent than fine-tuning

</details>

---

**Q5.** A social media company is deploying an LLM-based content moderation system that must process 50,000 posts/second during peak hours. Each post averages 100 tokens. P99 latency must be <100ms. Which architecture BEST meets these requirements?

A) Deploy multiple smaller 7B models across 64×A100 GPUs with model replication and load balancing
B) Deploy a 70B model across 32×A100 GPUs using tensor parallelism with dynamic batching
C) Use a distilled 1.3B model on 8×A100 GPUs with aggressive batching (batch size 512) and speculative decoding
D) Use a mixture-of-experts 175B model with 16 active experts on 64×H100 GPUs

<details>
<summary>Answer</summary>

**Correct: A**

50k posts/second × 100 tokens = 5M tokens/second throughput. 7B models on A100 achieve ~2000 tokens/sec/GPU with low batch sizes (needed for <100ms latency). 64 GPUs × 2000 = 128k tokens/sec... wait, need ~2500 tokens/sec/GPU. With model replication (multiple copies of 7B model), each GPU handles independent requests enabling parallel processing. P99 <100ms achievable because no cross-GPU communication (unlike tensor parallelism). Math: 64 GPUs, each running independent 7B model copy, ~800-1000 posts/sec/GPU × 64 = 51k-64k posts/sec.

**Why the others are wrong:**
- 70B model with tensor parallelism has high inter-GPU communication overhead (50-100ms just for communication). Cannot achieve P99 <100ms. Tensor parallelism optimizes for large batches, not low latency
- 1.3B distilled model might hit throughput BUT batch size 512 means P99 latency will be ~500-1000ms (waiting for batch to fill + processing). Speculative decoding helps but cannot overcome batching latency. Batching trades latency for throughput—wrong trade-off here
- MoE 175B model has complex routing overhead, higher latency variance (P99 will exceed 100ms), and 16 active experts still means large memory footprint. MoE optimizes for model capacity, not latency

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** An LLM application for financial analysis has been in production for 6 months. The team wants to measure success and iterate. Current metrics: 78% user satisfaction, 65% task completion rate, $15k/month infrastructure cost. Users complain about 'inconsistent output quality'. What is the BEST next step?

A) Immediately begin collecting human feedback (thumbs up/down) and fine-tune monthly on all negative examples
B) Switch to a larger model (13B → 70B) to improve consistency, accepting 3× cost increase
C) Add comprehensive logging of input/output pairs, implement RLHF with reward model trained on user ratings
D) Implement A/B testing comparing current prompt templates vs improved templates with structured output formats

<details>
<summary>Answer</summary>

**Correct: D**

A/B testing is the gold standard for measuring improvement impact with real users. 'Inconsistent output quality' suggests prompt engineering issue (structured outputs often fix this). A/B test provides clear metrics (does satisfaction/completion improve?) before committing resources. Low cost, fast to implement (days), measurable results.

**Why the others are wrong:**
- Fine-tuning on negative examples monthly is dangerous—models learn 'what not to do' poorly from negative examples alone. Need positive examples too. Also, monthly retraining without understanding root cause (inconsistent prompts? data quality? model limitation?) is wasteful
- Larger model doesn't guarantee consistency and 3× cost increase ($45k/month) without evidence it solves the problem is premature optimization. Exam penalizes 'throw resources at problem without diagnosis'
- RLHF is expensive (requires reward model training, PPO training, weeks of iteration) and overkill when simpler prompt engineering might solve inconsistency. RLHF is for complex preference learning, not output format consistency

</details>

---

**Q7.** A research team fine-tuned a 7B model for medical Q&A using 10,000 doctor-patient conversations. Evaluation on held-out test set shows 92% accuracy, but clinical deployment reveals 68% accuracy on real patient queries. What is the MOST likely cause and solution?

A) Test set was too small; solution is to collect 50,000 more conversations and retrain with larger validation set
B) Distribution shift between training conversations and real queries; solution is to implement continuous learning with daily retraining on production data
C) Training conversations lack diversity compared to real queries; solution is to augment prompts with few-shot examples from production and monitor performance by query type
D) Overfitting on training data despite good test accuracy; solution is to add regularization and reduce model size to 3B parameters

<details>
<summary>Answer</summary>

**Correct: C**

High test accuracy (92%) but low production accuracy (68%) indicates distribution mismatch, not overfitting (test accuracy would be low too). Training conversations likely cover common scenarios but real queries include edge cases, different phrasings, diverse patient populations. Prompt augmentation is fast, cheap, and allows targeted improvement by query type. Monitoring by type identifies which distributions need more training data.

**Why the others are wrong:**
- Test set size isn't the issue—92% test accuracy suggests test set is representative of training data, but training data isn't representative of production. Collecting more similar conversations won't help; need diverse data
- Daily retraining is extremely expensive (compute + human labeling of production data) and risky (production data may include errors, adversarial inputs, privacy issues). Need validation before incorporating production data into training
- Overfitting shows low test accuracy. Here test accuracy is high (92%), so model isn't overfitting training data. Reducing model size would likely hurt both test and production accuracy

</details>

---

**Q8.** A gaming company uses an LLM to generate NPC dialogue. They need to reduce inference cost by 50% without sacrificing dialogue quality. Current setup: 13B model on 2×A100 GPUs, 1 million requests/day, average 150 input tokens and 80 output tokens. Which combination of optimizations achieves the cost target?

A) Quantize model to 4-bit precision + implement KV cache optimization + reduce batch size from 32 to 8
B) Implement prompt caching for common game scenarios + quantize to 8-bit + use mixed-precision inference
C) Switch to a 7B model + quantize to 4-bit + implement speculative decoding with a 1B draft model
D) Move to CPU inference with 32-core servers + implement aggressive batching (batch size 256)

<details>
<summary>Answer</summary>

**Correct: B**

Gaming dialogue has repeating scenarios (NPC types, locations, quests), making prompt caching highly effective (30-50% cache hit rate typical). 8-bit quantization provides ~2× speedup with minimal quality loss. Mixed-precision inference (FP16/BF16 computation) adds another 1.3-1.5× speedup. Combined: ~2.5-3× throughput increase on same hardware = ~60-70% cost reduction. Quality preserved because 8-bit quantization and mixed-precision maintain model capabilities for dialogue generation.

**Why the others are wrong:**
- 4-bit quantization risks quality loss for dialogue generation (creative text). Reducing batch size from 32 to 8 INCREASES cost per request (lower GPU utilization). KV cache optimization helps but reducing batch size negates the benefit. Net result: <50% cost reduction or quality degradation
- 7B model + 4-bit quantization + speculative decoding could work BUT speculative decoding adds complexity (draft model training/tuning), and 4-bit risks dialogue quality. Also, switching models requires revalidation. Exam prefers solutions that optimize existing setup before changing models
- CPU inference for LLMs is 10-50× slower than GPU. Even with aggressive batching, cannot match GPU throughput. Would need 20-40 CPU servers to match 2×A100, costing more not less

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

