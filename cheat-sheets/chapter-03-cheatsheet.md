# Prompt Engineering Cheat Sheet — NCA-GENL Exam

## Prompting Strategies (Exam Weight: ~13%)

| Strategy | When to Use | Example Phrase | Accuracy Impact |
|----------|-------------|----------------|------------------|
| **Zero-shot** | Task is straightforward, model likely knows it | "Classify this review as positive/negative" | Good for common tasks |
| **Few-shot** | Task is nuanced, specific format needed | Include 1-5 examples before task | +10-30% accuracy |
| **Chain-of-thought** | Complex reasoning, math, analysis | "Let's think step by step" | +20-40% on reasoning |

## Temperature & Sampling Parameters

| Parameter | Range | Effect | Use Case |
|-----------|-------|--------|----------|
| **Temperature** | 0 | Deterministic (same input → same output) | Customer service, data extraction |
| | 0.3-0.4 | Mostly consistent, slight variation | Code generation, factual Q&A |
| | 0.7-0.9 | Creative but coherent | Content writing, brainstorming |
| | 1.0+ | Highly random (risky) | Experimental only |
| **Top-p** | 0.1 | Very focused (top 10% probability) | Technical accuracy needed |
| | 0.9 | Balanced diversity | General use |
| | 1.0 | All tokens considered | Maximum creativity |
| **Max_tokens** | 50-100 | Short responses | Summaries, titles |
| | 500-1000 | Medium content | Paragraphs, explanations |
| | 2000+ | Long form | Articles, detailed analysis |

## Prompt Types

| Type | Purpose | Priority |
|------|---------|----------|
| **System Prompt** | Sets role, behavior, constraints | Lower (user can override) |
| **User Prompt** | Specific task/question | Higher (takes precedence) |
| **Prompt Template** | Reusable structure with variables | Consistency across tasks |

## Structured Output Techniques

```
Best Practice for JSON Output:
1. Specify exact schema: {"field": "type"}
2. Set temperature=0 for consistency
3. Include one example if complex
4. Request validation in prompt
```

## Common Exam Traps

| Trap | Wrong Assumption | Right Answer |
|------|------------------|-------------|
| "Need consistency" | Increase max_tokens | Set temperature=0 |
| "Need creativity" | Use few-shot examples | Increase temperature to 0.7-0.9 |
| "Output too short" | Use chain-of-thought | Increase max_tokens |
| "Need reasoning" | Lower temperature | Add "Let's think step by step" |
| "Complex task" | Fine-tune immediately | Try few-shot first |

## Few-Shot Guidelines

- **Optimal**: 1-5 examples
- **Quality > Quantity**: 3 great examples beat 10 mediocre ones
- **Format**: Show input → output pairs
- **Diversity**: Cover edge cases in examples
- **Context cost**: Each example uses token budget

## Temperature Decision Tree

```
Need same output every time? → Temperature = 0
Need slight variation but accurate? → Temperature = 0.3-0.4
Need creative but coherent? → Temperature = 0.7-0.9
Experimenting with wild ideas? → Temperature = 1.0+
```

## Exam Formulas to Memorize

- **Zero-shot accuracy** < **Few-shot accuracy** < **Fine-tuned accuracy**
- **Temperature 0** = deterministic (1 possible output)
- **Temperature 1** = full probability distribution
- **Top-p 0.9** = most common setting (90th percentile)
- **Chain-of-thought** = +20-40% on reasoning tasks (research-backed)

## Red Flags (Common Mistakes)

1. ❌ Using temperature > 1.5 in production
2. ❌ Putting entire documents in prompt (use RAG instead)
3. ❌ Expecting perfect JSON without schema in prompt
4. ❌ Using few-shot for simple tasks (wastes context)
5. ❌ Conflicting system vs user prompts
6. ❌ Vague instructions ("make it good")
7. ❌ Not setting max_tokens (unbounded cost)

## Quick Reference: What Controls What

- **Consistency**: Temperature (lower = more consistent)
- **Creativity**: Temperature (higher = more creative)
- **Output length**: max_tokens
- **Token selection diversity**: top-p
- **Output quality**: Prompt clarity + examples
- **Reasoning quality**: Chain-of-thought
- **Format compliance**: Prompt template + temperature=0
