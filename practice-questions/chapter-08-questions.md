## Chapter 8: LLM Integration and Deployment—Part 2: The NVIDIA Software Stack — Practice Questions

**Q1.** A financial services company needs to deploy a fine-tuned LLaMA 70B model for real-time fraud detection queries. They require sub-200ms latency with support for variable batch sizes during peak and off-peak hours. Which NVIDIA solution BEST meets these requirements?

A) Deploy using NVIDIA NeMo framework with custom inference scripts on A100 GPUs
B) Run the model on Triton Inference Server without TensorRT-LLM optimizations
C) Use NVIDIA NIM containers with TensorRT-LLM optimizations and automatic batching
D) Use standard PyTorch inference with CUDA kernels on H100 GPUs

<details>
<summary>Answer</summary>

**Correct: C**

NIM containers provide pre-optimized deployment with TensorRT-LLM built-in, automatic batching for variable workloads, and GPU memory optimizations specifically designed for low-latency LLM inference. NIM is purpose-built for production LLM deployment with minimal configuration.

**Why the others are wrong:**
- NeMo is designed for training and fine-tuning, not production inference optimization. Custom inference scripts lack the built-in optimizations and batching logic that NIM provides.
- While Triton supports model serving, without TensorRT-LLM optimizations you miss quantization, kernel fusion, and FP8 support that dramatically reduce latency. Triton alone doesn't include LLM-specific optimizations.
- Standard PyTorch inference lacks the transformer-specific optimizations (quantization, kernel fusion, memory optimization) that TensorRT-LLM provides. Even on H100s, you won't achieve sub-200ms without these optimizations.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An ML team is building a custom GPT-style model from scratch for a specialized medical domain. They need to handle data preprocessing, implement prompt learning techniques, and apply RLHF for safety alignment. Which NVIDIA framework provides the MOST comprehensive solution for this entire workflow?

A) NVIDIA NIM with pre-built medical models
B) Triton Inference Server with custom backends
C) TensorRT-LLM with custom CUDA kernels
D) NVIDIA NeMo with its modular training pipeline

<details>
<summary>Answer</summary>

**Correct: D**

NeMo is specifically designed as an end-to-end framework covering the entire LLM lifecycle: data preprocessing modules, model training infrastructure, prompt learning (P-tuning, prompt tuning), and RLHF implementation. Its modular design supports all stages from data to deployment.

**Why the others are wrong:**
- NIM is for deploying pre-trained models, not building custom models from scratch. It provides inference containers, not training infrastructure or RLHF capabilities.
- Triton is an inference server, not a training framework. It doesn't provide data preprocessing, model training, prompt learning, or RLHF—only model serving capabilities.
- TensorRT-LLM is an optimization library for inference, not a training framework. It handles model optimization and deployment, but doesn't provide training, RLHF, or prompt learning capabilities.

</details>

---

**Q3.** A company runs multiple LLM variants (7B, 13B, 70B parameters) and needs to serve them concurrently with dynamic batching, version control, and health monitoring. Requests vary from 1 to 64 batch sizes. Which deployment architecture is MOST appropriate?

A) Use Triton Inference Server with dynamic batching and concurrent model execution
B) Deploy each model in separate NIM containers with manual load balancing
C) Run all models in a single NeMo training container
D) Deploy using TensorRT-LLM with custom batching logic

<details>
<summary>Answer</summary>

**Correct: A**

Triton Inference Server is specifically designed for multi-model serving with built-in dynamic batching (automatically groups requests), concurrent model execution (serves multiple models simultaneously), model versioning, and health check endpoints. It handles variable batch sizes efficiently without manual intervention.

**Why the others are wrong:**
- While NIM containers can deploy models, running separate containers requires manual orchestration for load balancing, lacks unified dynamic batching across models, and doesn't provide centralized version control or health monitoring that Triton offers natively.
- NeMo is a training framework, not designed for production inference serving. It lacks the inference-optimized batching, multi-model serving, and production monitoring features needed for this use case.
- TensorRT-LLM optimizes individual models but doesn't provide multi-model serving infrastructure, dynamic batching across models, version control, or health monitoring. You'd need to build serving infrastructure yourself.

</details>

---

**Q4.** During LLM deployment, an engineer observes 45ms latency per token on CPU inference versus 3ms on A100 GPU. The CFO asks for justification of GPU costs. Which metric provides the STRONGEST business case for GPU deployment?

A) Cost per 1000 tokens: GPU $0.002 vs CPU $0.015 due to throughput differences
B) GPU latency is 15x faster, improving user experience
C) GPUs support larger batch sizes than CPUs
D) Tensor Cores provide FP16 acceleration unavailable on CPUs

<details>
<summary>Answer</summary>

**Correct: A**

Cost per token is the business metric that matters to finance. GPUs achieve higher throughput (tokens/second/dollar) despite higher hardware costs. A $15/hour A100 serving 10,000 tokens/sec costs $0.0015 per 1000 tokens, while a $2/hour CPU serving 500 tokens/sec costs $0.004 per 1000 tokens. The exam tests understanding that throughput efficiency drives TCO.

**Why the others are wrong:**
- While latency improvement is real, it's a technical metric. CFOs care about cost efficiency, not milliseconds. This doesn't address the cost justification question directly.
- Batch size capability is a technical feature, not a cost metric. It doesn't translate to dollars saved or revenue generated, which is what financial stakeholders need.
- Technical acceleration features (Tensor Cores, FP16) are implementation details. They don't directly answer the cost question in business terms that justify the investment.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A research team fine-tuned a Mistral-7B model using NeMo and now needs to deploy it with 4-bit quantization and kernel fusion for production. Which combination of NVIDIA tools is required?

A) NeMo for export, then NIM container deployment with built-in optimizations
B) TensorRT-LLM alone handles fine-tuning and deployment
C) NIM containers automatically retrain and optimize any uploaded model
D) NeMo for training, TensorRT-LLM for optimization, Triton for serving

<details>
<summary>Answer</summary>

**Correct: D**

NeMo handles the fine-tuning (already done), TensorRT-LLM provides quantization (4-bit INT4/FP8), kernel fusion, and transformer-specific optimizations, and Triton serves the optimized model with batching and monitoring. This is the standard three-stage workflow: train → optimize → serve.

**Why the others are wrong:**
- NIM containers provide pre-optimized models but don't perform custom quantization or kernel fusion on user-provided fine-tuned models. NIM is for deploying already-optimized models, not for creating optimizations from scratch.
- TensorRT-LLM is an optimization library for inference, not a training framework. It cannot fine-tune models—that's NeMo's role. It only optimizes already-trained models.
- NIM containers deploy pre-built optimized models. They don't retrain or apply custom optimizations to uploaded models. NIM serves what you give it; it doesn't transform models.

</details>

---

**Q6.** An application serves a mixture of small (128 tokens) and large (2048 tokens) inference requests. Without intervention, large requests cause latency spikes for small requests. Which Triton Inference Server feature BEST addresses this issue?

A) Model versioning to separate small and large request handlers
B) Concurrent model execution across multiple GPU instances
C) Dynamic batching with configurable timeout and batch size limits
D) Health check endpoints to route requests to healthy servers

<details>
<summary>Answer</summary>

**Correct: C**

Dynamic batching groups requests intelligently based on configured timeouts and batch size constraints. It prevents large requests from blocking small ones by forming batches quickly when small requests arrive, while still batching large requests separately. The max_queue_delay_microseconds parameter controls this trade-off.

**Why the others are wrong:**
- Model versioning is for deploying different model versions (v1, v2), not for handling different request sizes. It doesn't solve the batching or latency spike issue.
- Concurrent model execution runs multiple models simultaneously, not multiple instances of the same model with different batching strategies. It doesn't address within-model request prioritization.
- Health checks monitor server availability and route around failures. They don't control batching behavior or prevent head-of-line blocking from large requests.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q7.** A company needs to deploy LLMs across 8 NVIDIA H100 GPUs with model parallelism and efficient memory usage. Which NVIDIA technology provides multi-GPU and multi-node parallelism specifically for LLM inference?

A) NeMo's Megatron-LM for training-time parallelism
B) TensorRT-LLM with tensor parallelism and pipeline parallelism support
C) Triton's model ensemble feature
D) CUDA unified memory for automatic GPU memory management

<details>
<summary>Answer</summary>

**Correct: B**

TensorRT-LLM includes built-in support for tensor parallelism (splitting model layers across GPUs) and pipeline parallelism (distributing layers across GPUs) specifically optimized for inference workloads. It handles multi-GPU and multi-node deployment with efficient communication patterns for LLM serving.

**Why the others are wrong:**
- Megatron-LM in NeMo is designed for distributed training parallelism, not inference parallelism. While the concepts are similar, NeMo focuses on training workflows, not production inference deployment.
- Triton's ensemble feature chains multiple models in a pipeline (e.g., preprocessing → inference → postprocessing), not parallelism across GPUs for a single model. It's about model composition, not distributed execution.
- CUDA unified memory provides a programming abstraction but doesn't implement model parallelism strategies. It's a low-level memory management feature, not a parallelism framework for LLM inference.

</details>

---

**Q8.** An engineering team observes that their LLM inference uses only 30% of available GPU memory but still experiences out-of-memory errors during traffic spikes. Which NIM feature would MOST likely resolve this issue?

A) NIM's automatic batching adjusts batch sizes based on available memory
B) NIM's model versioning automatically scales to more GPUs
C) NIM's KV cache optimization and memory pooling reduce fragmentation
D) NIM's health checks prevent routing to memory-constrained instances

<details>
<summary>Answer</summary>

**Correct: C**

OOM errors with low average memory usage indicate fragmentation. NIM includes KV cache optimization (efficient storage of key-value pairs in transformer attention) and memory pooling to reduce fragmentation. These techniques ensure memory is used efficiently, preventing spikes from causing OOM even when overall usage is low.

**Why the others are wrong:**
- While NIM does automatic batching, the issue is memory fragmentation, not batch size. Adjusting batch sizes won't solve fragmentation-related OOM errors when average utilization is only 30%.
- Model versioning is for deploying different model versions, not for scaling infrastructure. NIM doesn't automatically provision more GPUs—that's an orchestration layer responsibility (Kubernetes, etc.).
- Health checks detect and route around failures but don't prevent memory issues. They react to problems rather than solving the underlying fragmentation that causes OOM during spikes.

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

