# Chapter 8: NVIDIA Software Stack Cheat Sheet

## Framework Comparison

| Tool | Purpose | Key Features | When to Use |
|------|---------|--------------|-------------|
| **NVIDIA NeMo** | Train/fine-tune LLMs | Data preprocessing, model training, prompt learning, RLHF | Building custom models from scratch, domain-specific training |
| **NVIDIA NIM** | Deploy optimized LLMs | Pre-built containers, auto-batching, GPU memory optimization | Fast production deployment, minimal config |
| **TensorRT-LLM** | Optimize inference | Quantization (INT4/8, FP8), kernel fusion, multi-GPU parallelism | Custom optimization pipelines, maximum performance |
| **Triton Inference Server** | Serve multiple models | Dynamic batching, concurrent execution, versioning, health checks | Multi-model deployments, production monitoring |

## NeMo Modular Pipeline

1. **Data Preprocessing**: Tokenization, data cleaning, format conversion
2. **Model Training**: Distributed training, checkpointing, mixed precision
3. **Prompt Learning**: P-tuning, prompt tuning, adapter methods
4. **RLHF**: Reward modeling, PPO training, safety alignment

## NIM Key Benefits

- **Reduced Latency**: TensorRT-LLM optimizations built-in
- **GPU Memory Optimization**: KV cache optimization, memory pooling
- **Automatic Batching**: Adapts to variable workloads automatically
- **Pre-configured**: Minimal setup for production deployment

## Triton Production Features

| Feature | Description | Exam Relevance |
|---------|-------------|----------------|
| Dynamic Batching | Auto-groups requests, configurable timeout | Prevents head-of-line blocking |
| Concurrent Model Execution | Serves multiple models simultaneously | Multi-model use cases |
| Model Versioning | A/B testing, canary deployments | Production rollout strategies |
| Health Checks | Liveness/readiness endpoints | Orchestration integration |

## TensorRT-LLM Optimizations

### Quantization Support
- **INT4**: 4-bit integer (up to 4x memory reduction)
- **INT8**: 8-bit integer (2x memory reduction)
- **FP8**: 8-bit float (H100 Tensor Cores, best quality/performance)

### Parallelism Strategies
- **Tensor Parallelism**: Split model layers across GPUs (horizontal)
- **Pipeline Parallelism**: Distribute layers across GPUs (vertical)
- **Multi-node**: Scale across multiple machines

### Performance Features
- Kernel fusion: Combines operations to reduce memory transfers
- CUDA graph optimization: Reduces kernel launch overhead
- PagedAttention: Efficient KV cache management

## CPU vs GPU Inference

| Metric | CPU (typical) | GPU (A100 typical) | Ratio |
|--------|---------------|---------------------|-------|
| Latency/token | 40-50ms | 2-4ms | 10-20x faster |
| Throughput | 500 tokens/sec | 10,000+ tokens/sec | 20x higher |
| Cost/1000 tokens | $0.004-0.015 | $0.001-0.003 | 3-5x cheaper |

**Exam Note**: GPU cost efficiency comes from throughput, not hardware price.

## Deployment Decision Tree

```
Need to build custom LLM from scratch?
├─ YES → NVIDIA NeMo (training + fine-tuning)
└─ NO → Already have trained model
    ├─ Need custom optimizations? 
    │   └─ YES → TensorRT-LLM (quantization, fusion)
    └─ Ready to deploy?
        ├─ Single model, fast setup → NIM containers
        └─ Multiple models, production features → Triton
```

## Common Exam Traps

1. **NeMo vs NIM**: NeMo = training, NIM = deployment containers
2. **NIM vs TensorRT-LLM**: NIM = pre-optimized, TensorRT-LLM = DIY optimization
3. **Triton vs NIM**: Triton = multi-model + monitoring, NIM = single-model simplicity
4. **Cost justification**: Throughput (tokens/sec/dollar), not latency (milliseconds)
5. **OOM with low utilization**: Fragmentation → KV cache optimization in NIM

## Quick Commands Reference

### NeMo Training
```bash
# Fine-tune with NeMo
python scripts/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
  --config-path=conf --config-name=megatron_gpt_config
```

### TensorRT-LLM Build
```bash
# Convert and optimize model
trtllm-build --checkpoint_dir ./model \
  --output_dir ./trt_engine \
  --gemm_plugin auto \
  --max_batch_size 64
```

### Triton Model Config
```python
name: "llama_model"
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
}
```

## Memory & Performance Numbers (Exam-Relevant)

- **KV Cache**: ~1.5GB per 1000 tokens for 7B model
- **Dynamic batching timeout**: 100-5000 microseconds typical
- **FP8 vs FP16**: ~2x memory reduction, <1% accuracy loss
- **Tensor parallelism**: Linear scaling up to 8 GPUs, communication overhead beyond
