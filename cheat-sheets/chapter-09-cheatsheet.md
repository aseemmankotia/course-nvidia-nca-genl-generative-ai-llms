# Chapter 9 Cheat Sheet: Real-World LLM Application Architecture

## Decision Matrix: Fine-Tuning vs Prompt Engineering vs RAG

| Criterion | Prompt Engineering | RAG | Fine-Tuning |
|-----------|-------------------|-----|-------------|
| **Data needed** | 0-10 examples | Knowledge base | >1,000 labeled examples |
| **Time to deploy** | Hours-days | Days-weeks | Weeks-months |
| **Cost** | Low (context tokens) | Medium (embeddings + storage) | High (compute + data) |
| **Use case** | Quick iteration, few-shot | Dynamic knowledge, freshness | Task-specific behavior, consistency |
| **Quality ceiling** | 60-80% | 70-85% | 85-95% |
| **Exam tip** | Default choice for <500 examples | Best for factual/retrieval tasks | Use when task patterns must be learned |

## Model Size Trade-Offs (Single A100 GPU)

| Model Size | Memory | Latency (P50) | Throughput | Cost (/1M tokens) | Use Case |
|------------|--------|---------------|------------|-------------------|----------|
| **1.3B (distilled)** | 3GB | 20ms | 5,000 tok/sec | $0.50 | Classification, extraction |
| **7B** | 14GB | 50ms | 2,000 tok/sec | $1-2 | General tasks, low latency |
| **13B** | 26GB | 100ms | 1,000 tok/sec | $2-4 | Complex reasoning |
| **70B** | 140GB (4-8 GPUs) | 500ms+ | 200 tok/sec | $10-20 | Highest quality, not latency-sensitive |

**Exam rule**: For P99 <100ms → use ≤7B. For P99 <500ms → use ≤13B.

## Cost Optimization Techniques (Ranked by ROI)

1. **Semantic Caching**: 30-70% cost reduction, <1 week to implement
   - Use for: Repetitive queries (support, FAQ)
   - Implementation: Embed queries, cosine similarity >0.9 → cache hit

2. **Quantization**: 2-4× speedup, <1% quality loss (8-bit) or 5-10% (4-bit)
   - 8-bit: Safe for all tasks
   - 4-bit: Use for non-creative tasks, requires calibration

3. **Batch Inference**: 3-5× throughput, but +500ms latency
   - Use for: Offline processing, non-interactive
   - Avoid for: User-facing, <1sec SLA

4. **Model Distillation**: 5-10× speedup, 10-20% quality drop
   - 70B → 7B distilled can match 70B on specific tasks

5. **Prompt Optimization**: Reduce input tokens 20-50%
   - Remove examples that don't improve performance
   - Use shorter instructions, abbreviations

## Latency Optimization Stack

```
Target P99 <100ms:
├─ Use 7B or smaller model
├─ 8-bit quantization (2× speedup)
├─ Batch size = 1-4 (minimize queue time)
├─ GPU acceleration (A100 or H100)
└─ Optional: Speculative decoding (2-3× speedup)

Target P99 <500ms:
├─ Use 13B or smaller model  
├─ 4-bit or 8-bit quantization
├─ Batch size = 8-16
└─ KV cache optimization
```

## Production Monitoring Checklist

**Core Metrics** (log every request):
- ✅ Latency: P50, P95, P99
- ✅ Cost: Input tokens, output tokens, $ per request
- ✅ Task success: Completion rate, accuracy (if measurable)
- ✅ Errors: Rate, types (timeout, OOM, API errors)

**Quality Metrics** (sample or user-triggered):
- ✅ User feedback: Thumbs up/down, ratings
- ✅ Output quality score: Internal eval model or human review
- ✅ Distribution shift: Track query/output embedding distributions

**Business Metrics**:
- ✅ User retention, session length
- ✅ Conversion rate (if applicable)

## Iterating on LLM Applications: Decision Tree

```
Performance issue detected → 

├─ Quality problem?
│  ├─ Inconsistent outputs → Structured prompts, few-shot examples
│  ├─ Factual errors → Add RAG
│  ├─ Task-specific failures → Fine-tune if >1k examples
│  └─ Distribution shift → Collect new data, A/B test prompt updates
│
├─ Latency problem?
│  ├─ P99 >500ms → Reduce batch size, use smaller model, quantize
│  ├─ Throughput bottleneck → Add GPU replicas, increase batch size
│  └─ Cold start issues → Keep models warm, use serverless with min instances
│
└─ Cost problem?
   ├─ High token usage → Implement caching, optimize prompts
   ├─ High compute cost → Quantize, use smaller model, batch processing
   └─ Scaling cost → Autoscale GPUs, use spot instances
```

## Hybrid Architecture Pattern (Exam Favorite)

**When to use**: Task has consistent patterns + evolving knowledge

**Setup**:
1. Fine-tune base model on task-specific data (teaches task structure)
2. Add RAG for knowledge retrieval (handles facts, updates)
3. Use prompt engineering for edge cases (few-shot examples for rare scenarios)

**Example**: Legal contract analysis
- Fine-tuned on 5k contracts → learns legal reasoning patterns
- RAG over clause library → retrieves relevant precedents
- Prompts include jurisdiction-specific rules → handles edge cases

**Exam trap**: Don't use all three for simple tasks. Prompt engineering alone often sufficient.

## A/B Testing LLM Changes

**Always A/B test**:
- New prompt templates
- Different models (7B vs 13B)
- RAG vs no RAG
- Fine-tuning experiments

**Metrics to track** (7-14 days minimum):
- Task success rate (primary metric)
- User satisfaction (thumbs up/down)
- Latency P50/P99
- Cost per successful request

**Rollout**: 5% → 25% → 50% → 100% if metrics improve

## Exam Traps to Avoid

❌ "Use the largest model for best quality" → Overfitting, cost, latency issues  
❌ "Fine-tune on all available data" → Need train/val/test split  
❌ "Production accuracy dropped → retrain immediately" → Diagnose first (prompt? data? distribution shift?)  
❌ "Batch size 256 for low latency" → Batching trades latency for throughput  
❌ "4-bit quantization always safe" → Can degrade creative/generative tasks
