# Experimentation and Evaluation Cheat Sheet

## Evaluation Metrics Comparison

| Metric | Use Case | Range | Direction | Key Focus |
|--------|----------|-------|-----------|----------|
| **Perplexity** | Language modeling | 1 to ∞ | Lower is better | Next-token prediction quality |
| **BLEU** | Translation/generation | 0.0 - 1.0 | Higher is better | N-gram precision |
| **ROUGE-1** | Summarization | 0.0 - 1.0 | Higher is better | Unigram recall |
| **ROUGE-2** | Summarization | 0.0 - 1.0 | Higher is better | Bigram recall |
| **ROUGE-L** | Summarization | 0.0 - 1.0 | Higher is better | Longest common subsequence |
| **Human eval** | Open-ended tasks | 1-5 or 1-7 | Higher is better | Fluency, coherence, relevance |

## Perplexity Quick Reference

```
Perplexity = exp(cross-entropy loss)
         = exp(-1/N × Σ log P(word|context))

Lower perplexity = model less "surprised" by data = BETTER

Typical ranges (task-dependent):
- Excellent: < 20
- Good: 20-50  
- Moderate: 50-100
- Poor: > 100
```

**Exam trap:** Students often think higher perplexity is better (confusing with accuracy %). Remember: perplexity measures surprise/confusion.

## BLEU vs ROUGE Decision Matrix

| Task | Metric | Why |
|------|--------|-----|
| Machine translation | BLEU | Precision-focused; measures how much generated text matches reference |
| Document summarization | ROUGE | Recall-focused; measures how much reference content is captured |
| Text generation (creative) | BLEU or Human | BLEU for factual, human for creativity |
| Code generation | BLEU + exact match | N-gram overlap + syntax correctness |

**Key difference:** BLEU = precision (generated → reference). ROUGE = recall (reference → generated).

## Experimental Design Framework

### A/B Testing Checklist
1. **Hypothesis:** Null (no difference) vs Alternative (difference exists)
2. **Traffic split:** Usually 50/50; minimum 10% for treatment
3. **Sample size:** Depends on baseline rate and minimum detectable effect
4. **Duration:** Long enough to capture weekly cycles (7+ days typical)
5. **Metric:** Primary metric + guardrail metrics (latency, cost)
6. **Significance:** p < 0.05 typical threshold
7. **Effect size:** Is improvement practically meaningful?

### Ablation Study Template
```
Baseline system: Component A + B + C = 85% accuracy
Remove A: B + C = 78% accuracy → A contributes 7pp
Remove B: A + C = 81% accuracy → B contributes 4pp  
Remove C: A + B = 70% accuracy → C contributes 15pp

Conclusion: C is most critical component
```

**Exam note:** Ablation removes ONE component at a time to isolate impact.

## Standard Benchmarks

| Benchmark | Tasks | Model Type | Interpretation |
|-----------|-------|------------|----------------|
| **GLUE** | 9 language understanding | Any LLM | General NLU capability |
| **SuperGLUE** | 8 harder tasks | Any LLM | Advanced reasoning |
| **MMLU** | 57 subjects (STEM to humanities) | Any LLM | Broad knowledge coverage |
| **HumanEval** | 164 coding problems | Code models | Programming ability |

**MMLU exam fact:** Scores ≥ 70% indicate strong general knowledge; ≥ 85% is state-of-the-art territory (as of 2026).

## LLM-as-Judge Best Practices

**When to use:**
- Open-ended generation (chatbots, creative writing)
- No clear reference answer
- Human eval too expensive/slow

**Setup:**
```
Judge model: GPT-4 or Claude-3-Opus (stronger than model being evaluated)
Rating scale: 1-5 for helpfulness, coherence, factuality  
Prompt template: "Rate the following response on [dimension]: [response]"
```

**Risks:**
- Style bias (favors outputs similar to judge's training)
- Position bias (favors first/last option in comparisons)  
- Length bias (may favor longer responses)

**Validation:** Run periodic human eval (10-20% sample) to verify judge alignment.

## Statistical Significance Traps

**p-value < 0.05 means:**
✅ Results unlikely due to random chance  
✅ Statistical significance achieved

**p-value < 0.05 does NOT mean:**
❌ Large/meaningful improvement  
❌ Ready for production deployment  
❌ Will sustain over time

**Exam scenario:** Given p=0.03 but only 1% improvement → statistically significant but may not be practically significant.

## Human Evaluation Scales

**Typical dimensions (1-5 scale):**
- **Fluency:** Grammar, readability, naturalness
- **Coherence:** Logical flow, stays on topic  
- **Relevance:** Answers the question asked
- **Factuality:** Information is accurate
- **Helpfulness:** Useful for user's goal

**Inter-annotator agreement:** κ (kappa) > 0.6 is acceptable, > 0.8 is excellent.

**Exam note:** Human eval is gold standard but expensive; use for final validation or small samples.

## Quick Decision Tree

```
Evaluating a model?
├─ Language modeling → Perplexity (lower better)
├─ Translation → BLEU (higher better)  
├─ Summarization → ROUGE (higher better)
├─ Open-ended generation → LLM-as-judge or human eval
└─ General capability → Benchmark (MMLU, GLUE)

Designing experiment?
├─ Comparing two versions → A/B test
├─ Measuring component value → Ablation study  
└─ Baseline performance → Benchmark against standard datasets
```
