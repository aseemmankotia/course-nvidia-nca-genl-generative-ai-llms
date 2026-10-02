## Chapter 6: Experimentation and Evaluation — Practice Questions

**Q1.** A company is evaluating two language models for a chatbot. Model A achieves a perplexity of 45 on the validation set, while Model B achieves a perplexity of 78. The team wants to choose the model that better predicts the next token in conversations. Which model should they select?

A) Model A, because lower perplexity indicates the model is less surprised by the validation data
B) Model B, because higher perplexity indicates better prediction accuracy
C) Model B, because perplexity above 70 indicates production-ready quality
D) Model A, but only if the perplexity difference exceeds 50 points for statistical significance

<details>
<summary>Answer</summary>

**Correct: A**

Perplexity measures how well a model predicts the next token - lower values mean the model is less 'perplexed' (surprised) by the data. Model A's perplexity of 45 is better than Model B's 78.

**Why the others are wrong:**
- Higher perplexity means worse prediction - the model is MORE surprised by the data
- There is no industry standard threshold like '70' for perplexity - it's relative to the task and dataset
- The 50-point threshold is fabricated - statistical significance depends on variance and sample size, not a fixed perplexity gap

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** A research team is comparing machine translation outputs. They run BLEU score evaluation and get: System X = 0.42, System Y = 0.38. The team lead asks why BLEU was chosen over ROUGE. What is the correct explanation?

A) BLEU scores above 0.40 indicate human-level translation quality, making it the standard metric
B) ROUGE would give identical scores since both measure n-gram overlap
C) ROUGE is designed for summarization tasks with recall focus, while BLEU measures n-gram precision for translation quality
D) BLEU is the only metric that works with non-English language pairs

<details>
<summary>Answer</summary>

**Correct: C**

BLEU (Bilingual Evaluation Understudy) focuses on precision of n-gram matches and is designed for translation/generation tasks. ROUGE focuses on recall and is designed for summarization evaluation.

**Why the others are wrong:**
- BLEU 0.40 does not indicate human-level quality - the scale is 0-1 but interpretation varies by task and language pair; human parity typically requires much higher scores
- ROUGE emphasizes recall (how much reference content is captured) while BLEU emphasizes precision (how much generated content matches reference) - they give different scores
- Both BLEU and ROUGE work with any language pair - neither is language-specific

</details>

---

**Q3.** An ML team runs an A/B test comparing two prompts for a customer service bot. Prompt A (control) gets 10,000 requests, Prompt B (treatment) gets 10,000 requests. Prompt B shows 8% higher customer satisfaction with p-value = 0.03. The team lead wants to deploy Prompt B immediately. What should the team consider before deployment?

A) Verify the effect size is meaningful for business impact, check for novelty effects, and monitor for longer duration before full rollout
B) Deploy immediately - p-value below 0.05 confirms statistical significance
C) Reject the results - A/B tests require at least 50,000 requests per variant for validity
D) Re-run the test with equal traffic split for exactly 7 days to meet industry standards

<details>
<summary>Answer</summary>

**Correct: A**

Statistical significance (p < 0.05) doesn't guarantee practical significance. Need to verify: (1) effect size matters for business, (2) no novelty effects biasing results, (3) sustained performance over time. Best practice is gradual rollout.

**Why the others are wrong:**
- p < 0.05 shows statistical significance but doesn't account for practical significance, novelty effects, or long-term performance - deployment decisions need more context
- 10,000 requests per variant is typically sufficient for detecting meaningful effects - the 50,000 threshold is fabricated; sample size depends on baseline rate and minimum detectable effect
- The 7-day duration and equal split are arbitrary - test duration depends on traffic patterns and cycle effects; the traffic was already equally split at 10k each

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q4.** A team performs an ablation study on their RAG (Retrieval-Augmented Generation) pipeline. The full system achieves 78% accuracy. Removing the re-ranking component drops accuracy to 71%. Removing the query expansion drops it to 62%. Which component has the larger impact?

A) Re-ranking component, because 78 - 71 = 7 percentage points
B) Both have equal impact since 7% + 16% = 23% total degradation
C) Query expansion, because removing it causes a 16 percentage point drop (78 - 62)
D) Cannot determine without running both components removed simultaneously

<details>
<summary>Answer</summary>

**Correct: C**

Ablation studies measure impact by removing one component at a time. Query expansion removal causes 78% → 62% = 16pp drop, larger than re-ranking's 7pp drop. The larger drop indicates greater component importance.

**Why the others are wrong:**
- 7 percentage points is the re-ranking impact, but query expansion's 16pp drop is larger
- The drops don't add to measure total degradation - ablation studies remove components independently to isolate individual impact
- Ablation studies specifically test components one at a time - removing both simultaneously would measure combined effect, not individual impact

</details>

---

**Q5.** A company evaluates their summarization model on news articles. They get ROUGE-1 = 0.45, ROUGE-2 = 0.22, ROUGE-L = 0.41. A competitor's model scores ROUGE-1 = 0.43, ROUGE-2 = 0.28, ROUGE-L = 0.40. The team asks which model is better. What is the correct assessment?

A) Their model is better because ROUGE-1 and ROUGE-L are higher, and ROUGE-1 is the most important metric
B) The competitor's model is better because ROUGE-2 captures bigram overlap which indicates better semantic coherence
C) The models have different strengths - higher ROUGE-1/L suggests better recall of key terms, higher ROUGE-2 suggests better phrase-level quality; choose based on use case requirements
D) Calculate the average: (0.45+0.22+0.41)/3 = 0.36 vs (0.43+0.28+0.40)/3 = 0.37 - competitor wins

<details>
<summary>Answer</summary>

**Correct: C**

ROUGE-1 measures unigram overlap (individual words), ROUGE-2 measures bigram overlap (two-word phrases), ROUGE-L measures longest common subsequence. Each captures different quality aspects. Choice depends on whether word recall or phrase coherence matters more for the application.

**Why the others are wrong:**
- No single ROUGE metric is definitively 'most important' - the relative importance depends on the summarization task requirements
- Higher ROUGE-2 indicates better bigram matching but doesn't automatically make it 'better' - unigram coverage (ROUGE-1) and sequence quality (ROUGE-L) also matter
- Averaging ROUGE scores is not standard practice - each metric measures different linguistic properties and shouldn't be combined into a single number

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q6.** An organization implements LLM-as-judge to evaluate open-ended chatbot responses. They use GPT-4 to score outputs from their fine-tuned Llama model on a 1-5 scale for helpfulness. Initial results show 87% agreement with human ratings. What is a key limitation they should monitor?

A) LLM judges cannot evaluate responses longer than 512 tokens due to attention limitations
B) LLM-as-judge is only valid for multiple-choice questions, not open-ended generation
C) The judge model may favor responses stylistically similar to its own training data, introducing systematic bias toward certain response patterns
D) 87% agreement is below the 95% threshold required for production deployment

<details>
<summary>Answer</summary>

**Correct: C**

LLM-as-judge can exhibit bias toward outputs that match its training distribution or style. For example, GPT-4 might prefer responses that sound like its own outputs. This systematic bias should be monitored through periodic human evaluation comparisons.

**Why the others are wrong:**
- Modern LLMs can evaluate much longer contexts - GPT-4 supports 128K tokens, and the 512 limit is fabricated
- LLM-as-judge is specifically designed for open-ended generation where automated metrics like BLEU/ROUGE are insufficient - it's not limited to multiple-choice
- There is no industry standard 95% agreement threshold - 87% agreement is actually quite good; acceptable agreement depends on task difficulty and cost-benefit tradeoffs

</details>

---

**Q7.** A research lab benchmarks their new model on MMLU (Massive Multitask Language Understanding). They achieve 71.2% accuracy. The current state-of-the-art scores 86.4%. The team wants to understand what this gap means. Which interpretation is most accurate?

A) Their model failed benchmarking - any score below 80% on MMLU indicates the model is not production-ready
B) They should switch to GLUE instead because MMLU is designed only for models with 100B+ parameters
C) The gap is irrelevant - MMLU scores don't correlate with real-world task performance
D) MMLU tests knowledge across 57 subjects; the 15.2pp gap suggests their model lags in broad knowledge coverage but the score itself doesn't determine production-readiness - that depends on the specific use case

<details>
<summary>Answer</summary>

**Correct: D**

MMLU measures multitask knowledge across 57 subjects. While 71.2% shows the model has learned substantial knowledge, the gap to 86.4% indicates room for improvement. However, production-readiness depends on whether the specific application requires state-of-the-art MMLU performance or if 71.2% suffices for the use case.

**Why the others are wrong:**
- There is no universal 80% threshold for production-readiness on MMLU - deployment decisions depend on task requirements; some applications may not need MMLU-style broad knowledge at all
- MMLU works for any model size - it's not restricted to 100B+ parameter models; GLUE and MMLU test different capabilities (GLUE focuses on language understanding tasks, MMLU on knowledge)
- MMLU scores do correlate with certain real-world capabilities (knowledge-intensive tasks) - benchmarks provide standardized comparison points even if not perfectly predictive of all applications

</details>

---

**Q8.** A team designs an experiment to test whether adding a context window of 8K tokens improves summarization quality versus their baseline 2K context. They plan to use hypothesis testing. Which setup follows proper experimental design?

A) Null hypothesis: 8K context improves quality. Alternative: no difference. Run 100 examples and compare mean ROUGE scores
B) Null hypothesis: no difference between 2K and 8K context. Alternative: 8K performs differently. Use paired samples (same articles with both contexts), measure ROUGE-L, test for statistical significance
C) Run both models on different datasets to avoid data leakage, then compare perplexity scores
D) Hypothesis testing doesn't apply to NLP tasks - simply deploy both and use A/B testing instead

<details>
<summary>Answer</summary>

**Correct: B**

Proper hypothesis testing: (1) null hypothesis assumes no difference, (2) alternative hypothesis is what you're testing (8K is different), (3) paired samples control for article difficulty, (4) appropriate metric (ROUGE-L for summarization), (5) statistical test for significance.

**Why the others are wrong:**
- Null hypothesis is incorrectly stated - should be 'no difference' as the default assumption to be tested against; also, different example sets for each variant introduces uncontrolled variance
- Using different datasets introduces confounding variables - can't tell if differences are from context size or dataset difficulty; also perplexity measures prediction quality, not summarization quality
- Hypothesis testing absolutely applies to NLP - it's the foundation for determining if observed differences are statistically significant vs random chance; A/B testing is a form of hypothesis testing in production

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

