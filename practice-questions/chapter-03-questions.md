## Chapter 3: Prompt Engineering — Practice Questions

**Q1.** A developer needs to build a sentiment classifier that can categorize customer reviews as positive, negative, or neutral. The team has no labeled training data but needs immediate results. Which prompting approach BEST meets these requirements?

A) Few-shot prompting with 3-5 example reviews for each sentiment category
B) Zero-shot prompting with clear instruction to classify sentiment
C) Chain-of-thought prompting asking the model to explain its reasoning
D) Fine-tuning a custom model on unlabeled review data

<details>
<summary>Answer</summary>

**Correct: B**

Zero-shot prompting works well for straightforward classification tasks that the model already understands from pre-training. Sentiment analysis is a common task that doesn't require examples when instructions are clear.

**Why the others are wrong:**
- Few-shot would improve accuracy but violates the 'no labeled data' constraint and is unnecessary for simple sentiment tasks
- Chain-of-thought adds overhead without benefit for simple classification; CoT is for complex reasoning, not basic categorization
- Fine-tuning requires labeled training data and infrastructure, contradicting the 'immediate results' requirement

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An application generates product descriptions for an e-commerce site. The business requires descriptions to be exactly 50 words and follow a strict JSON format: {"title": "", "features": [], "price_range": ""}. Initial outputs are inconsistent in length and format. What is the MOST effective solution?

A) Increase max_tokens to 500 to ensure the model has room for the full response
B) Use few-shot prompting with 5 examples of correctly formatted descriptions
C) Set temperature to 0 and use a prompt template specifying the exact JSON structure with field descriptions
D) Set top_p to 0.9 to increase output diversity and creativity

<details>
<summary>Answer</summary>

**Correct: C**

Temperature=0 ensures deterministic output for consistency. Prompt templates with explicit structure (JSON schema) guide the model to produce formatted output reliably. This addresses both the format and consistency issues.

**Why the others are wrong:**
- Increasing max_tokens doesn't control output length or format; it only sets the ceiling, not the structure or word count
- Few-shot would help but temperature=0 + template is more direct and doesn't consume context window with examples; examples alone don't guarantee exact word count
- Higher top_p increases randomness, making consistency worse; the problem requires determinism, not creativity

</details>

---

**Q3.** A medical diagnosis assistant must analyze patient symptoms and provide detailed reasoning for its suggested diagnoses. Initial prompts return only final diagnoses without explanation. Which modification will BEST improve the model's reasoning transparency?

A) Increase temperature from 0.3 to 1.2 to generate more creative diagnostic possibilities
B) Add 'Let's think step by step' to the prompt and request intermediate reasoning before the conclusion
C) Provide few-shot examples of correct diagnoses without showing reasoning steps
D) Reduce max_tokens to force the model to be more concise in its output

<details>
<summary>Answer</summary>

**Correct: B**

Chain-of-thought prompting (e.g., 'Let's think step by step') explicitly instructs the model to show its reasoning process. This technique significantly improves reasoning quality and transparency for complex analytical tasks.

**Why the others are wrong:**
- Higher temperature increases randomness and potentially introduces hallucinations in medical context; reasoning needs consistency, not creativity
- Examples without reasoning steps don't teach the model to show its work; the problem is lack of explanation, not final answer quality
- Reducing max_tokens would truncate explanations, making the transparency problem worse

</details>

---

**Q4.** A customer service chatbot must handle 10,000+ daily conversations. The team notices responses are sometimes inconsistent for identical questions. The current configuration uses temperature=0.8 and top_p=0.95. Which change will MOST effectively improve response consistency?

A) Change top_p to 1.0 while keeping temperature at 0.8
B) Increase max_tokens from 256 to 1024 to allow fuller responses
C) Switch from zero-shot to few-shot prompting with 3 example conversations
D) Change temperature to 0 while keeping top_p at 0.95

<details>
<summary>Answer</summary>

**Correct: D**

Temperature=0 makes the model deterministic, selecting the highest probability token every time. This ensures identical inputs produce identical outputs. Top_p is less critical when temperature=0.

**Why the others are wrong:**
- Top_p=1.0 considers all tokens regardless of probability, increasing randomness; doesn't address consistency with temperature still at 0.8
- Max_tokens affects length, not consistency; identical questions would still get inconsistent answers
- Few-shot helps quality but doesn't guarantee consistency; temperature=0.8 still allows randomness in token selection

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A legal document summarizer needs to process 20-page contracts and extract key clauses. Early tests show the model produces summaries that sometimes miss critical information or hallucinate clauses. The team has 10 correctly summarized contracts. What combination provides the BEST approach?

A) Use zero-shot prompting with temperature=1.5 to maximize creative summarization
B) Fine-tune a model on the 10 contracts and use temperature=0
C) Use chain-of-thought prompting alone without examples, setting top_p=0.1
D) Use the 10 contracts as few-shot examples with chain-of-thought prompting asking the model to identify clauses step-by-step before summarizing

<details>
<summary>Answer</summary>

**Correct: D**

Few-shot examples show the model what good summaries look like. Chain-of-thought improves reasoning for complex extraction tasks, reducing hallucinations. This combination leverages both techniques for a complex analytical task.

**Why the others are wrong:**
- Temperature=1.5 increases hallucination risk; zero-shot misses the opportunity to use the 10 good examples; creativity is not the goal for legal accuracy
- 10 samples are insufficient for fine-tuning, which typically needs thousands; this wastes the examples and adds infrastructure complexity
- Chain-of-thought helps but ignoring the 10 examples wastes valuable in-context learning; top_p=0.1 alone doesn't prevent hallucinations

</details>

---

**Q6.** A code generation tool must produce Python functions that are correct but can vary in implementation approach (different algorithms are acceptable). The current setting uses temperature=0 and developers complain about repetitive solutions. Which parameter change BEST addresses this while maintaining code correctness?

A) Set temperature to 2.0 to maximize creative coding approaches
B) Increase temperature to 0.7 while keeping top_p at default to allow varied but reasonable implementations
C) Decrease top_p to 0.1 to focus on only the most likely tokens
D) Increase max_tokens to 2048 to allow longer, more detailed implementations

<details>
<summary>Answer</summary>

**Correct: B**

Temperature=0.7 introduces controlled randomness that allows different valid implementations while avoiding wild variations. This is the sweet spot for creative but coherent code generation.

**Why the others are wrong:**
- Temperature=2.0 is too high and likely produces syntactically incorrect or nonsensical code; exceeds typical 0-1 range for production use
- Lower top_p reduces diversity, making the repetition problem worse; need more variation, not less
- Max_tokens affects length, not implementation diversity; doesn't address the repetitive solutions issue

</details>

---

**Q7.** An AI tutoring system needs to explain complex math problems to students. The system prompt defines the AI as a patient tutor, while user prompts contain the specific problems. Initial results show the AI sometimes ignores the patient tutor instruction and provides terse answers. What is the MOST likely cause?

A) Temperature is set too low at 0.2, preventing creative explanations
B) The user prompt is overriding the system prompt behavior; system prompts provide general context but user prompts often take precedence
C) The max_tokens limit of 150 is too restrictive for detailed explanations
D) Few-shot examples are needed to show the AI how to be patient

<details>
<summary>Answer</summary>

**Correct: B**

System prompts set overall behavior but user prompts are more immediately influential. If user prompts implicitly request concise answers (e.g., 'solve this quickly'), they can override the system's patient tutor instruction.

**Why the others are wrong:**
- Low temperature affects randomness, not instruction-following; temperature=0.2 is reasonable for educational content
- 150 tokens may be restrictive but doesn't explain why instructions are ignored; the problem is behavioral, not length
- Few-shot could reinforce behavior but the root cause is prompt hierarchy; system vs user prompt conflict is the primary issue

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A research team needs to analyze scientific papers and generate hypotheses for future experiments. The task requires creative leaps based on existing data. Current configuration: temperature=0, top_p=0.9, max_tokens=512. Which single parameter change will MOST effectively enable creative hypothesis generation?

A) Increase temperature to 0.9 to allow the model to explore less probable but potentially innovative connections
B) Decrease top_p to 0.5 to focus on high-probability tokens
C) Increase max_tokens to 2048 to allow longer hypotheses
D) Keep all parameters the same but add 'Let's think step by step' to the prompt

<details>
<summary>Answer</summary>

**Correct: A**

Temperature directly controls randomness and creativity. Temperature=0 is deterministic (always picks highest probability token). Increasing to 0.9 allows exploration of less obvious connections needed for creative hypothesis generation.

**Why the others are wrong:**
- Lower top_p reduces creativity by excluding lower-probability tokens; opposite of what's needed
- Max_tokens affects output length, not creativity; longer doesn't mean more creative
- Chain-of-thought improves reasoning but doesn't enable creative leaps; temperature=0 would still pick the most obvious next tokens

</details>

---

