# Practice Test 2: NVIDIA NCA-GENL: Generative AI & LLM Certification Prep

- **Questions:** 44
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-2.md`. Track results in `score-tracker.md`.

---

**Q1.** A financial services company is training a deep neural network to detect fraudulent transactions. After 50 epochs, the training loss continues to decrease steadily to 0.02, but the validation loss has plateaued at 0.18 for the last 20 epochs while showing slight upward drift. The model achieves 94% accuracy on training data but only 78% on validation data. Which action would MOST effectively address this issue?

A) Increase the learning rate to help the model escape the local minimum and continue improving on validation data
B) Add more hidden layers to the network to increase model capacity and improve feature extraction
C) Train for additional epochs since the training loss is still decreasing and the model hasn't converged
D) Implement dropout layers with a rate of 0.3-0.5 and add L2 regularization to the dense layers

---

**Q2.** An AI research team is comparing activation functions for a 12-layer convolutional neural network used for medical image classification. During training, they observe that layers 7-12 have gradients approaching zero (10^-7 to 10^-8 range), causing these layers to learn extremely slowly. Which activation function replacement would BEST resolve this vanishing gradient problem?

A) Replace tanh or sigmoid activations with ReLU (Rectified Linear Unit) or LeakyReLU
B) Replace sigmoid activations with tanh since tanh has a larger output range (-1 to 1) compared to sigmoid (0 to 1)
C) Replace all activations with softmax to ensure gradients are properly normalized across all layers
D) Keep sigmoid but reduce the learning rate by a factor of 10 to allow gradients to accumulate over more iterations

---

**Q3.** A manufacturing company deploys a predictive maintenance model using a random forest ensemble with 200 decision trees. Each tree is trained on the full dataset of 50,000 samples with all 35 features. During inference, predictions are highly accurate but too slow for real-time monitoring (850ms per prediction vs required 100ms). The model shows training accuracy of 98.2% and test accuracy of 97.8%. Which optimization would provide the BEST speed improvement while maintaining model performance?

A) Convert the random forest to a single optimized decision tree using the most important features identified by feature importance scores
B) Reduce the number of trees from 200 to 40-50 and test whether accuracy remains above 97% threshold
C) Implement bagging by training each tree on a random 70% subset of data and using a random subset of features at each split (standard random forest configuration)
D) Replace the random forest with a gradient boosting model which trains trees sequentially and typically requires fewer trees

---

**Q4.** A data scientist is training a binary classification neural network to identify defective products from sensor data. The dataset contains 95,000 non-defective samples and 5,000 defective samples (95:5 imbalance). After training with binary cross-entropy loss, the model achieves 95.2% accuracy but predicts 'non-defective' for almost all inputs, missing 94% of actual defects. Which combination of techniques would MOST effectively improve defect detection?

A) Increase the decision threshold from 0.5 to 0.8 so the model only predicts 'defective' when more confident
B) Switch to a multi-class classification approach by creating subcategories of defects to give the model more signal
C) Add more hidden layers and neurons to increase model capacity so it can learn the minority class patterns better
D) Apply SMOTE (Synthetic Minority Over-sampling) to balance the dataset AND use focal loss or class-weighted loss instead of standard binary cross-entropy

---

**Q5.** An autonomous vehicle company is training a convolutional neural network to detect pedestrians. The model uses three convolutional layers with kernel sizes of 5×5, 3×3, and 3×3 respectively, each followed by 2×2 max pooling. Input images are 256×256×3. After the final pooling layer, the spatial dimensions are 30×30 with 128 feature maps. Which statement BEST explains the function of the max pooling layers in this architecture?

A) Max pooling reduces spatial dimensions by 50% per layer, providing translation invariance and reducing computational cost for subsequent layers while retaining the most prominent features in each pooled region
B) Max pooling applies a learned filter that selects the maximum activation, similar to how convolutional filters learn features, making the network deeper and more expressive
C) Max pooling normalizes the activations in each 2×2 region to prevent exploding gradients during backpropagation through deep networks
D) Max pooling increases the receptive field of later layers by combining adjacent pixels, allowing the network to focus on larger objects like full pedestrians rather than small details

---

**Q6.** A healthcare AI team is fine-tuning a pre-trained transformer model (250M parameters) for medical report summarization. They have 12,000 labeled medical reports and limited compute budget. After 5 epochs of full model fine-tuning, validation loss is 1.84 and training loss is 0.92, but generated summaries are incoherent. Training takes 40 hours per epoch on their hardware. Which approach would MOST efficiently improve results within their constraints?

A) Continue full fine-tuning for 20 more epochs since transformer models require extensive training to converge properly on domain-specific tasks
B) Reduce the model size to 50M parameters by removing transformer layers, which will speed up training and reduce overfitting
C) Switch to LoRA (Low-Rank Adaptation) or adapter layers, which freeze the base model and train only small inserted modules, reducing trainable parameters by 99% while maintaining quality
D) Implement curriculum learning by starting with simpler, shorter medical reports and gradually increasing complexity over epochs

---

**Q7.** A research team is fine-tuning a decoder-only transformer model for code generation. During training, they notice the model performs well on short code snippets but struggles with generating complete functions exceeding 2048 tokens. The team wants to maintain computational efficiency while enabling the model to handle longer contexts. Which architectural modification would BEST address this requirement?

A) Implement sliding window attention with overlapping segments of 512 tokens
B) Increase the number of attention heads from 12 to 24 while keeping embedding dimensions constant
C) Replace standard attention with ALiBi (Attention with Linear Biases) positional encoding
D) Add additional encoder layers to create an encoder-decoder architecture

---

**Q8.** An AI startup is deploying a mixture-of-experts (MoE) language model with 8 expert networks and a gating mechanism. During inference, they observe that only 2 of the 8 experts are consistently activated for 90% of inputs, while the other 6 experts remain underutilized. This is causing suboptimal model performance. What is the MOST likely cause of this expert collapse?

A) The router network is using softmax activation instead of sigmoid, preventing multiple expert selection
B) The expert networks have insufficient parameter capacity compared to the routing network
C) The training data contains insufficient diversity, causing mode collapse in the gating function
D) The load balancing loss coefficient is set too low, failing to encourage uniform expert utilization

---

**Q9.** A financial services company needs to deploy a large language model for contract analysis. Their compliance requirements mandate that they must be able to completely remove all knowledge about specific individuals from the model after processing. They are evaluating different architectures. Which approach would BEST meet this requirement?

A) Implement differential privacy during training with a privacy budget epsilon value below 1.0
B) Deploy a fine-tuned model with periodic retraining from scratch excluding removed individuals
C) Use a mixture-of-experts architecture where individual experts can be retrained independently
D) Use a retrieval-augmented generation (RAG) system with contracts stored in a vector database that supports hard deletion

---

**Q10.** During the development of a multilingual translation model, engineers implement grouped-query attention (GQA) to reduce memory consumption. The model has 48 attention heads and they configure 8 key-value head groups. After deployment, they observe that translation quality for low-resource languages has degraded compared to the baseline multi-head attention model, while high-resource languages maintain quality. What is the MOST effective solution to address this issue?

A) Increase the number of key-value head groups from 8 to 16 to provide more representational capacity
B) Implement language-specific adapter layers that operate after the GQA mechanism
C) Apply higher dropout rates specifically to the query projections for high-resource languages during training
D) Switch to multi-query attention (MQA) with a single key-value head shared across all queries

---

**Q11.** A machine learning team is implementing flash attention for their transformer model training pipeline. They need to process sequences of 16,384 tokens with a batch size of 32 on GPUs with 40GB of memory. The standard attention implementation causes out-of-memory errors. After implementing flash attention, they still experience memory issues during backward pass. What is the MOST likely configuration error?

A) The gradient checkpointing is disabled, requiring storage of all attention matrices for backpropagation
B) The block size for tiled computation is set too large, causing intermediate activation tensors to exceed SRAM capacity
C) Flash attention is enabled only for the forward pass but not configured for the backward pass recomputation
D) The attention dropout rate is too high, requiring storage of dropout masks for all attention layers

---

**Q12.** A healthcare AI company is evaluating whether to use sparse attention patterns for their clinical notes processing model. Their notes average 32,768 tokens with critical information distributed throughout the document. They are comparing different sparse attention patterns: local attention (fixed window), stride attention (dilated), and global attention (dedicated global tokens). Which pattern would provide the BEST balance between computational efficiency and the ability to capture long-range dependencies in clinical documentation?

A) Local attention with a window size of 1024 tokens to capture paragraph-level context efficiently
B) Longformer-style combined pattern: local attention for all tokens plus global attention for section headers
C) Stride attention with stride-4 pattern to reduce computations while maintaining document-wide receptive field
D) Random sparse attention where each token attends to a random subset of 256 tokens across the sequence

---

**Q13.** A research team is fine-tuning a decoder-only transformer model with 7 billion parameters for a specialized medical diagnosis task. During training, they observe that the model generates coherent text but frequently hallucinates medical facts not present in their training corpus. They have limited compute resources and cannot afford to retrain from scratch. Which approach would MOST effectively reduce hallucination while preserving the model's existing medical knowledge?

A) Increase the temperature parameter during inference from 0.7 to 1.2 to introduce more randomness and reduce overconfident incorrect predictions
B) Add more attention heads to the existing transformer layers and continue pre-training on general medical literature for 10,000 additional steps
C) Apply aggressive dropout (p=0.5) to all attention layers during fine-tuning and reduce the learning rate by a factor of 100
D) Implement retrieval-augmented generation (RAG) by connecting the model to a verified medical knowledge base and modifying the inference pipeline to ground responses in retrieved documents

---

**Q14.** A financial services company is building a fraud detection system that analyzes transaction narratives. The LLM must extract structured data (amount, merchant, location) from unstructured text while maintaining zero hallucination tolerance for numerical values. Initial tests show the model occasionally invents amounts when none are explicitly stated. Which prompt engineering technique BEST addresses this requirement?

A) Use chain-of-thought prompting with explicit instructions to output 'NULL' for missing fields, combined with structured output formatting (JSON schema) and validation constraints
B) Implement few-shot learning with 10 examples showing correct extraction patterns and use temperature=0 to ensure deterministic outputs
C) Apply retrieval-augmented generation (RAG) to cross-reference all extracted amounts against a transaction database before returning results
D) Increase the max_tokens parameter to 2048 and add a system prompt stating 'You are a precise financial analyst who never invents data'

---

**Q15.** An e-commerce platform uses an LLM to generate product descriptions from specifications. During load testing, the team discovers response times vary dramatically (200ms to 8000ms) for similar input lengths. Analysis shows 15% of requests trigger the model's maximum context window, causing performance degradation. The specification data averages 3200 tokens. Which optimization provides the MOST efficient solution?

A) Implement prompt compression by removing all whitespace and unnecessary formatting, reducing average token count by 20-25%
B) Redesign prompts using a hierarchical approach: send critical specifications in the main prompt and optional details via a follow-up completion if needed, processing 85% of requests with shorter context
C) Switch to a larger context window model (128k tokens) to eliminate truncation issues entirely
D) Use batch processing to group similar product types together, allowing the model to reuse cached context across multiple descriptions

---

**Q16.** A healthcare application uses prompt chaining to process patient intake forms: (1) extract symptoms, (2) map to ICD-10 codes, (3) suggest diagnostic tests. During audit, 8% of chains produce incorrect ICD-10 codes despite correct symptom extraction. The symptom-to-code mapping prompt uses 50 few-shot examples. Further investigation reveals errors cluster around rare conditions (< 1% prevalence). What is the ROOT CAUSE of this failure pattern?

A) The model's training data cutoff predates recent ICD-10 code updates, causing it to use deprecated codes for rare conditions
B) Chain-of-thought reasoning is not being used in the mapping step, preventing the model from explaining its code selection logic
C) The few-shot examples exhibit selection bias toward common conditions, causing the model to overgeneralize patterns and misclassify rare conditions as similar common ones
D) The temperature parameter is set too high (> 0.7), introducing randomness that disproportionately affects rare condition classification

---

**Q17.** A legal tech company builds a contract analysis system using constitutional AI principles to ensure outputs never suggest illegal clauses. The system uses a two-stage prompt: (1) constitutional critique evaluating legality, (2) revision incorporating critique. Testing reveals the revision stage sometimes ignores critique warnings, proceeding with problematic clauses. Token analysis shows critique outputs average 450 tokens, while revision prompts receive full contract text (4200 tokens) plus critique. Which architectural change BEST ensures critique adherence?

A) Split revision into micro-tasks: process one flagged clause at a time with its specific critique, then reassemble the contract programmatically
B) Increase the revision prompt's emphasis on critique importance by adding 'CRITICAL: You must address all legal concerns raised in the critique' in bold system instructions
C) Implement a retrieval mechanism that extracts only flagged problematic clauses and their critiques (avg. 280 tokens) instead of passing the full critique text
D) Restructure to use function calling where the revision model must explicitly invoke a validate_legality() function that returns boolean approval before generating each clause

---

**Q18.** A customer service chatbot uses persona prompting with the instruction: 'You are a friendly, empathetic support agent named Alex.' A/B testing reveals 22% higher customer satisfaction versus generic prompts, but compliance review flags 40 conversations where 'Alex' disclosed personal opinions on company policies rather than stating official policy. The chatbot uses RAG with a policy knowledge base. Which prompt modification MOST effectively maintains the satisfaction benefit while ensuring policy compliance?

A) Enhance persona with explicit boundaries: 'You are Alex, a support agent. Be friendly and empathetic. When discussing policies, always preface with "According to our policy" and cite policy sections. Never share personal opinions on policies.'
B) Remove the persona entirely and replace with: 'You are an AI assistant. Always be accurate and cite company policies exactly. Maintain professional tone.'
C) Implement a two-model system: a persona model handles emotional tone, while a separate policy-specialized model generates all policy-related content that gets inserted verbatim
D) Add few-shot examples showing 'Alex' correctly handling policy questions with empathy while citing sources, ensuring examples demonstrate the boundary between empathy and policy accuracy

---

**Q19.** A research team evaluates prompt strategies for mathematical reasoning tasks. Strategy A uses zero-shot chain-of-thought ('Let's think step by step'). Strategy B uses few-shot with 8 worked examples. Strategy C uses least-to-most prompting (decompose problem, solve subproblems). Testing on 500 problems shows: Strategy A: 68% accuracy, avg 850 tokens/response. Strategy B: 79% accuracy, avg 1240 tokens/response. Strategy C: 82% accuracy, avg 2100 tokens/response. API costs are $0.02/1K tokens (input+output combined). The production system will process 2M problems/month. Which strategy optimizes for cost-effectiveness at ≥75% accuracy?

A) Strategy B: 79% accuracy meets the threshold with moderate token usage (1240 tokens), balancing performance and cost at $49,600/month
B) Strategy B with token optimization: compress few-shot examples using abbreviated notation and remove verbose explanations, targeting 900 tokens while maintaining 76-77% accuracy at approximately $36,000/month
C) Strategy C: The 3% accuracy improvement (82% vs 79%) justifies the higher token cost ($84,000/month) for mathematical reasoning where correctness is critical
D) Hybrid approach: Use Strategy A (850 tokens) for all problems, then retry failures with Strategy C (2100 tokens), achieving estimated 78% accuracy at $44,200/month

---

**Q20.** A healthcare AI system for diagnosing rare diseases was trained on a dataset where 95% of cases were from urban hospitals in developed countries. During deployment in rural clinics across multiple continents, the system shows significantly lower accuracy for patients from underrepresented populations. The development team wants to address this issue while maintaining model performance on the original dataset. Which approach BEST aligns with responsible AI principles for handling dataset bias?

A) Collect additional labeled data from underrepresented populations, retrain with stratified sampling to ensure balanced representation, and validate performance across all subgroups before deployment
B) Implement post-processing calibration to adjust confidence scores differently for underrepresented demographic groups
C) Add a disclaimer to the system warning users about potential accuracy limitations for certain populations while continuing current deployment
D) Apply synthetic data generation techniques to create artificial examples of underrepresented cases and add them to the training set

---

**Q21.** An enterprise deploys a generative AI assistant that has access to internal company documents through RAG (Retrieval-Augmented Generation). Security testing reveals that users can craft prompts that cause the system to leak confidential salary information from HR documents, despite initial guardrails. The AI team identifies this as a prompt injection vulnerability. Which defense strategy provides the MOST comprehensive protection against this class of attacks?

A) Implement a separate LLM-based classifier that screens all user inputs and blocks requests containing suspicious patterns before they reach the main model
B) Fine-tune the base model on examples of prompt injection attempts labeled as malicious to teach it to refuse such requests
C) Use a multi-layered approach: input validation with pattern blocking, strict document access controls with role-based permissions enforced at retrieval time, output filtering to detect and redact sensitive entities, and monitoring for anomalous query patterns
D) Apply constitutional AI techniques to train the model to follow instructions that prohibit sharing confidential information regardless of user prompts

---

**Q22.** A financial services company uses an LLM to generate investment advice summaries. Regulators require that all AI-generated advice be explainable and that the company must be able to demonstrate which information sources influenced specific recommendations. The current system uses a black-box LLM API. Which solution BEST satisfies the regulatory explainability requirements?

A) Implement a citation-based RAG architecture where the system retrieves specific source documents, includes document IDs in the generation context, and prompts the model to cite sources for each claim in its output, with post-processing validation to ensure citations are accurate
B) Switch to a smaller open-source model that allows inspection of attention weights and use attention visualization to show which input tokens influenced each output token
C) Use SHAP (SHapley Additive exPlanations) values to compute feature importance scores for each input and present these scores to regulators as the explanation
D) Log all prompts and outputs with timestamps, and use chain-of-thought prompting to have the model explain its reasoning process in the output

---

**Q23.** An AI research team is developing a large multimodal model and discovers during red-teaming that the model can generate detailed instructions for synthesizing hazardous materials when provided with images of chemistry lab equipment combined with specific text prompts. The model will be released as open-source. Which risk mitigation strategy is MOST aligned with responsible AI release practices for dual-use capabilities?

A) Release the model with usage guidelines prohibiting harmful applications and implement API-level content filtering for hosted versions, while providing the unfiltered weights for research purposes
B) Remove the problematic capability through targeted unlearning or ablation techniques to prevent the model from generating hazardous content, then release the modified model openly
C) Release the full model openly with comprehensive documentation of risks, arguing that open access enables the research community to develop better defenses and that restricting access only delays inevitable proliferation
D) Conduct structured access release: provide the model only through a gated API with mandatory use-case review, monitoring of queries for misuse patterns, and rate limiting, while publishing detailed capability evaluations and risk assessments

---

**Q24.** A company deploys an AI-powered content moderation system for a social media platform. The system flags and auto-removes posts based on policy violations. After six months, civil rights groups report that the system disproportionately removes content from minority communities discussing discrimination they face, while allowing similar policy-violating content from majority groups. Audit logs show the model has 94% overall accuracy. Which analysis approach would BEST identify the root cause of this fairness issue?

A) Compute demographic parity metrics by comparing removal rates across different demographic groups and adjust decision thresholds to equalize removal rates
B) Implement an appeals process where users can contest removals, and use the appeals data to retrain the model on cases where it made mistakes
C) Analyze the training data composition and labeling: check representation of different communities in training data, examine whether discussing discrimination was incorrectly labeled as violating policies, evaluate if cultural context and dialect differences affected labeling consistency, and measure performance metrics separately for each demographic subgroup
D) Switch to a larger foundation model with better general language understanding capabilities, as the issue likely stems from the base model's limited capacity

---

**Q25.** A data scientist is analyzing a dataset with 50,000 rows where the target variable 'customer_churn' is imbalanced (5% positive class). After applying SMOTE to balance the classes, the validation accuracy increased from 85% to 92%, but the production model shows significantly worse precision on the minority class. What is the MOST likely cause and appropriate solution?

A) SMOTE creates synthetic examples that don't represent real-world distributions. Use class_weight parameter instead or combine SMOTE with undersampling (SMOTETomek).
B) The validation set was too small. Increase validation set size to 30% of total data and reapply SMOTE to both train and validation sets.
C) SMOTE was applied before splitting train/test data, causing data leakage. Apply SMOTE only to training data after the split.
D) SMOTE requires feature scaling before application. Apply StandardScaler before SMOTE and retrain the model.

---

**Q26.** A team is building features for a time-series forecasting model predicting hourly energy consumption. They create a feature 'rolling_mean_24h' using a 24-hour rolling average of consumption. During backtesting on historical data from 2024-2025, the model achieves 98% accuracy, but in production (real-time inference), accuracy drops to 72%. Which implementation error is MOST likely responsible?

A) The 24-hour window is too large for hourly predictions. Reduce window size to 6 hours and add day-of-week as a categorical feature.
B) Energy consumption has seasonal patterns not captured by rolling means. Add Fourier terms for daily and weekly seasonality to the feature set.
C) The model was trained on data with missing hours interpolated, but production data has actual gaps. Implement forward-fill imputation before calculating rolling features.
D) The rolling window calculation included future data points during training (forward-looking bias). Use .shift(1) to ensure only past values are used for each prediction.

---

**Q27.** A dataset contains a 'transaction_amount' feature with extreme outliers (99th percentile = $500, max = $2.5M). A data scientist needs to prepare this feature for a gradient boosting model. After comparing four preprocessing approaches on validation data, which approach should be PREFERRED based on typical gradient boosting behavior and robustness requirements?

A) Apply log transformation log(transaction_amount + 1) to compress the range, then use the transformed values directly without further scaling.
B) Clip outliers using IQR method (values beyond Q3 + 1.5×IQR set to that threshold), then apply StandardScaler normalization.
C) Create three features: original amount, log(amount + 1), and a binary flag for amounts > 99th percentile. Let the model learn which to use.
D) Use quantile transformation with output_distribution='normal' and n_quantiles=1000 to map values to a normal distribution.

---

**Q28.** A visualization dashboard displays model prediction confidence scores for 10,000 daily transactions. The data science team receives complaints that users cannot identify which predictions need manual review. The current chart shows all 10,000 points as a scatter plot with confidence on the y-axis. Which visualization modification would BEST support the business requirement of identifying review-worthy predictions?

A) Replace scatter plot with a histogram showing the distribution of confidence scores binned into 20 equal-width intervals, with annotations marking the 10th and 90th percentiles.
B) Create a combined view: histogram of confidence scores with color-coded regions (red for <0.7 confidence, yellow for 0.7-0.9, green for >0.9), plus a sortable data table filtered to show only red region predictions with transaction IDs.
C) Convert to a hexbin plot to reduce overplotting, using color intensity to show density, and add a horizontal reference line at the decision threshold (e.g., 0.5 confidence).
D) Use a box plot grouped by prediction class (positive/negative) to show confidence distribution quartiles, with individual outlier points displayed beyond the whiskers.

---

**Q29.** A feature engineering pipeline creates interaction terms between numerical features 'age' (range: 18-85) and 'income' (range: $15K-$500K) resulting in 'age_x_income' (range: 270K-42.5M). During model training with L2 regularization (alpha=0.01), the coefficient for 'age_x_income' is near zero while individual 'age' and 'income' coefficients are significant. What is the MOST appropriate action?

A) The large magnitude of 'age_x_income' values causes regularization to disproportionately penalize its coefficient. Scale all features using StandardScaler before model training.
B) The interaction term is not predictive. Remove 'age_x_income' from the feature set and retrain the model.
C) Increase regularization strength to alpha=0.1 to better handle the multicollinearity introduced by the interaction term.
D) Replace 'age_x_income' with 'log(age) × log(income)' to reduce the magnitude and make the interaction term more interpretable.

---

**Q30.** A data analysis reveals that a categorical feature 'product_category' with 150 unique values has a correlation coefficient of 0.89 with the target variable after applying target encoding (mean target value per category). The model validation AUC is 0.94, but when deployed to production with new data from different regions, AUC drops to 0.68. What is the PRIMARY issue and correct remediation?

A) Target encoding was applied before train/test split, causing data leakage. Apply target encoding separately to train and test sets using only training set statistics for test encoding.
B) Target encoding caused severe overfitting by memorizing train-set target statistics. Use leave-one-out encoding during training and add smoothing (m=10) to regularize rare categories.
C) The high correlation with target indicates the feature is leaking label information. Replace target encoding with one-hot encoding, handling high cardinality with feature hashing (n_features=50).
D) 150 categories is too high for target encoding to generalize. Reduce cardinality by grouping rare categories (frequency <1%) into an 'OTHER' category, then reapply target encoding.

---

**Q31.** A research team is evaluating two prompt strategies for a medical diagnosis assistant. Strategy A achieves 92% accuracy on the validation set but shows significant performance degradation when tested on data from different hospitals. Strategy B achieves 87% accuracy on the same validation set but maintains consistent performance across all hospital datasets. The team must choose one strategy for production deployment across a nationwide health system. Which approach should they take?

A) Deploy Strategy A because higher validation accuracy indicates better model performance and will deliver superior results in production
B) Combine both strategies using an ensemble approach that switches between them based on input characteristics to maximize accuracy
C) Deploy Strategy A initially and monitor for performance degradation, then switch to Strategy B only if issues arise in production
D) Deploy Strategy B because consistent cross-domain performance indicates better generalization and reduced risk of deployment failures

---

**Q32.** An e-commerce company is testing a new LLM-powered product recommendation system. During A/B testing with 5% of traffic over 3 days, the LLM variant shows a 15% increase in click-through rate but a 3% decrease in actual purchases compared to the rule-based baseline. The LLM variant also has 200ms higher p95 latency. Management wants to roll out the LLM system to all users immediately based on the CTR improvement. What is the BEST evaluation concern to raise?

A) The 200ms latency increase will compound at scale and likely cause timeout errors affecting user experience and requiring infrastructure upgrades
B) Three days of testing is insufficient to capture weekly purchasing patterns, seasonal variations, and the true business metric (purchases) shows negative results
C) The 5% sample size is too small to draw statistically significant conclusions and the test should be expanded to at least 20% of traffic before making decisions
D) The LLM system should be evaluated using offline metrics like BLEU score and perplexity before A/B testing to ensure recommendation quality

---

**Q33.** A legal document analysis system uses an LLM to extract key clauses. The team evaluates it using exact match accuracy against human annotations and achieves 76% accuracy. A senior engineer reviews the failures and finds that 18% of 'incorrect' predictions are actually valid alternative extractions that capture the same legal meaning using different text spans. The team needs to report evaluation metrics to stakeholders. Which metric strategy provides the most accurate assessment of system performance?

A) Report the 76% exact match accuracy as the primary metric since it represents objective ground truth alignment and is reproducible across evaluators
B) Create a relaxed matching metric that credits semantically equivalent extractions, resulting in 94% accuracy, and report this as the primary metric with exact match as secondary
C) Discard the current annotations as unreliable, re-annotate with multiple lawyers to establish inter-annotator agreement, then restart evaluation with consensus labels
D) Report both 76% exact match and 94% semantic match with clear definitions, explaining that the true performance for the legal use case is better represented by semantic match

---

**Q34.** A content moderation system uses an LLM to classify user comments as safe, sensitive, or harmful. During red-teaming evaluation, adversarial researchers discover that adding phrases like 'hypothetically speaking' or 'in a fictional context' causes the model to misclassify 34% of harmful content as safe. The system currently has 97% accuracy on the standard test set. Which response BEST addresses this evaluation finding?

A) Add examples with these adversarial phrases to the training data and retrain the model to specifically recognize and handle these evasion patterns
B) Create an adversarial evaluation benchmark incorporating these patterns, augment training data with adversarial examples, establish continuous red-teaming, and add this metric to production monitoring
C) Implement a preprocessing filter that strips out common jailbreak phrases like 'hypothetically' and 'fictional context' before LLM classification
D) Document this as a known limitation since the 97% standard accuracy is high and these adversarial cases represent edge cases unlikely to appear frequently in production

---

**Q35.** A financial services company is evaluating an LLM for generating investment summaries. They compare GPT-4, Claude 3, and a fine-tuned Llama model using human preference ratings from 5 financial analysts. After 100 evaluations, GPT-4 has a 68% preference rate, Claude 3 has 52%, and the fine-tuned Llama has 48%. The team calculates that they need 2,400 more evaluations to achieve statistical significance (p<0.05) between the top two models given the current variance. What is the BEST evaluation strategy adjustment?

A) Proceed with 2,400 additional human evaluations to achieve statistical significance since rigorous evaluation is critical for financial applications and the cost is justified
B) Deploy GPT-4 immediately since it has the highest preference rate and the 16-point lead over Claude 3 indicates clear superiority regardless of statistical significance
C) Redesign the evaluation to use pairwise comparisons with more evaluators per pair, or implement LLM-as-judge evaluation with GPT-4 to increase evaluation throughput while maintaining quality
D) Switch to automated metrics like ROUGE and BERTScore which can evaluate thousands of examples quickly and provide objective measurements without human evaluation costs

---

**Q36.** A financial services company is deploying a fine-tuned Llama 3.1 70B model for document analysis. The model must process 500 concurrent requests with P95 latency under 2 seconds while maintaining cost efficiency. The infrastructure team proposes using NVIDIA Triton Inference Server with TensorRT-LLM optimization. Which deployment configuration BEST meets these requirements?

A) Deploy on 8x L40S GPUs with tensor parallelism degree 4, inflight batching enabled, max_tokens=2048, and continuous batching with KV cache management
B) Deploy on 4x A100 80GB GPUs with dynamic batching enabled, max_batch_size=32, and KV cache reuse disabled to ensure consistency
C) Deploy on 2x H100 80GB GPUs with pipeline parallelism across both GPUs, static batching with batch_size=64, and FP32 precision for maximum accuracy
D) Deploy on 16x A10G GPUs with model parallelism disabled, dynamic batching with preferred_batch_size=[8,16,32], and INT8 quantization via TensorRT

---

**Q37.** An AI research team needs to deploy multiple LLM variants (Mixtral 8x7B, GPT-J 6B, and a custom 13B model) on a shared inference cluster. Each model requires different preprocessing pipelines and tokenization schemes. Which NVIDIA NIM (NVIDIA Inference Microservice) deployment strategy provides the MOST efficient resource utilization and model management?

A) Deploy each model as a separate NIM container with dedicated GPU allocation, using Kubernetes HPA based on request queue depth and average token throughput metrics
B) Deploy models on separate virtual GPU (vGPU) instances using NVIDIA vComputeServer, with each model getting fractional GPU allocation and isolated memory spaces
C) Use a single multi-model NIM instance with dynamic model loading, shared GPU memory pool, and NVIDIA Triton's model ensemble feature for preprocessing pipeline orchestration
D) Implement NIM with model repository polling, load all models simultaneously into GPU memory, and use round-robin scheduling with client-side model selection headers

---

**Q38.** A healthcare SaaS provider is implementing RAG (Retrieval-Augmented Generation) using NVIDIA NeMo Retriever with a vector database containing 50 million medical document embeddings. The system experiences inconsistent retrieval quality where semantically similar queries return different result sets. Performance profiling shows the embedding model (NV-Embed-v2) outputs are normalized, but retrieval precision varies significantly. What is the MOST likely root cause and solution?

A) The vector database index type is using HNSW with insufficient ef_construction parameter—increase ef_construction from 200 to 512 and rebuild the index for better graph connectivity
B) The reranking stage is missing—add NVIDIA NeMo Reranker after initial retrieval to reorder top-K results using cross-encoder scoring before passing to the LLM
C) NV-Embed-v2 requires domain-specific fine-tuning on medical terminology—fine-tune the embedding model using NeMo Framework with curated medical Q&A pairs and domain corpus
D) The chunking strategy is producing variable-length segments that embed differently—implement fixed 512-token chunks with 50-token overlap and re-embed the entire corpus

---

**Q39.** Which combination of NVIDIA software stack components is required to deploy a multi-node training job for a custom 30B parameter decoder-only transformer model using 8-way tensor parallelism and 4-way pipeline parallelism across 32 A100 GPUs?

A) NeMo Framework + Megatron-LM + NCCL + cuDNN + nvFuser for kernel fusion during training
B) TensorRT-LLM + Triton Inference Server + CUDA Graphs + cuBLAS for distributed inference optimization
C) PyTorch + CUDA Toolkit + NCCL + Nsight Systems + apex.amp for mixed precision distributed training
D) NeMo Framework + Transformer Engine + NCCL + CUDA Toolkit + FP8 recipe with Megatron-Core parallelism utilities

---

**Q40.** A media company is deploying a video content moderation system using NVIDIA Maxine for real-time video understanding combined with an LLM for policy decision-making. The system must process 1000 concurrent video streams at 30 FPS, extract visual features, and query a Llama 3 8B model for each frame's policy compliance. The architecture team debates between two approaches: (A) Use Maxine SDK to extract features on edge devices, send features to centralized LLM inference, or (B) Stream raw video to datacenter, process with Maxine on inference servers co-located with LLM. Which approach BEST optimizes for latency and cost?

A) Approach A: Edge processing minimizes network bandwidth (features are 10KB vs 500KB raw frames), reduces datacenter compute costs, and enables sub-100ms round-trip latency through pre-computed feature caching
B) Approach B: Datacenter processing enables dynamic Maxine model updates without edge deployment, shares GPU resources between Maxine and LLM inference using MIG (Multi-Instance GPU), and eliminates edge hardware costs
C) Approach B with optimization: Use NVIDIA Aerial SDK for video stream ingestion, co-locate Maxine and LLM on same GPU using CUDA streams for concurrent execution, and batch LLM queries across multiple frames using KV cache sharing
D) Hybrid approach: Use Maxine VFX SDK on edge for initial filtering (flag only suspicious frames), stream only flagged frames to datacenter for full Maxine analysis + LLM inference, reducing processed volume by 80-90%

---

**Q41.** An enterprise is comparing deployment options for serving a Mistral 7B Instruct model. Option 1: NVIDIA NIM with default configuration. Option 2: Custom deployment using TensorRT-LLM with quantization, served via Triton. Security requires that model weights never leave the corporate network, and the deployment must support A/B testing between the base model and a fine-tuned variant. Which option provides the BEST balance of performance, operational simplicity, and security?

A) Option 2: TensorRT-LLM with FP8 quantization provides 2-3x throughput improvement over NIM defaults, Triton's model versioning natively supports A/B testing, and on-premises deployment ensures model weight security
B) Option 1: NIM containers can be deployed on-premises via NVIDIA AI Enterprise, include optimized TensorRT-LLM engines by default, support multiple model versions simultaneously, and provide faster time-to-production with pre-built containers
C) Hybrid: Use NIM for the base model (faster deployment) and custom TensorRT-LLM for the fine-tuned variant (better performance), route traffic via Kubernetes Istio service mesh for A/B splitting, deploy both on-premises
D) Option 2: Build custom TensorRT-LLM engines with INT4 AWQ quantization for maximum throughput, use Triton ensemble for A/B logic, deploy in air-gapped datacenter, and leverage lower licensing costs without NVIDIA AI Enterprise

---

**Q42.** A financial services company is deploying a multi-tenant LLM inference service using NVIDIA Triton Inference Server. Each tenant requires isolated model execution with guaranteed throughput of 50 tokens/second per request, and the system must support dynamic batching while preventing cross-tenant data leakage. The infrastructure uses NVIDIA A100 GPUs with MIG (Multi-Instance GPU) enabled. Which configuration approach BEST meets these requirements?

A) Deploy separate Triton server instances per tenant using Docker containers, configure dynamic batching with max_batch_size=32, and use CUDA_VISIBLE_DEVICES to assign different MIG instances to each container
B) Use a single Triton server with model versioning to separate tenants, enable sequence batching with direct scheduling, and rely on CUDA stream isolation for tenant separation
C) Deploy Triton with ensemble models where each tenant has a dedicated preprocessing pipeline, use the rate_limiter backend parameter to guarantee throughput, and enable concurrent model execution without MIG
D) Configure Triton model instances equal to tenant count, use the instance_group parameter with KIND_GPU and count=1 per tenant, enable priority scheduling, and implement tenant routing at the API gateway level

---

**Q43.** An AI research team is optimizing a Llama 3.1 70B model deployment for production inference. After profiling with NVIDIA Nsight Systems, they identify that the attention mechanism consumes 62% of inference latency, with significant memory bandwidth bottlenecks during the KV cache reads. The deployment target is NVIDIA H100 GPUs running TensorRT-LLM. Which optimization strategy will provide the GREATEST latency reduction for this specific bottleneck?

A) Apply 4-bit weight quantization using GPTQ for the entire model, enable CUDA graphs for kernel launch overhead reduction, and increase the max_batch_size parameter to improve GPU utilization
B) Implement Flash Attention 2 with paged attention in TensorRT-LLM, enable INT8 KV cache quantization, and configure --use_paged_kv_cache with optimal block size based on sequence length distribution
C) Use TensorRT-LLM's FP8 inference with Transformer Engine, enable multi-query attention (MQA) by modifying the model architecture, and deploy with pipeline parallelism across 2 GPUs
D) Implement speculative decoding with a Llama 3.1 8B draft model, enable KV cache offloading to CPU memory using unified memory, and configure prefill-decode disaggregation

---

**Q44.** A company is implementing RAG (Retrieval-Augmented Generation) using NVIDIA NeMo Retriever with a vector database containing 50 million embeddings. The system must support 1000 concurrent queries with P95 latency under 200ms for the complete RAG pipeline (retrieval + generation). Current profiling shows: embedding generation 15ms, vector search 180ms, LLM generation 120ms. The vector database uses FAISS with IVF index. Which architectural change will MOST effectively meet the latency SLA?

A) Upgrade to FAISS GPU implementation with the existing IVF index, increase nprobe parameter from 16 to 128 for better recall, and add 4 additional A100 GPUs to distribute the search load
B) Implement hierarchical vector search with two-stage retrieval (coarse HNSW + fine IVF), cache the top 10K most frequent query embeddings in Redis, and use async I/O for parallel retrieval and generation
C) Replace FAISS IVF with NVIDIA RAFT CAGRA (Cuda Anns GRAph-based) index for GPU-accelerated ANN search, deploy embeddings on GPU memory, and implement query batching with batch size 8-16 for concurrent requests
D) Migrate to a distributed vector database (Milvus with 8 nodes), use Product Quantization (PQ) compression to reduce memory footprint, and increase the number of retrieved documents from 10 to 50 for better context

---

