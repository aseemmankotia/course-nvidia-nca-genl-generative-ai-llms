# Practice Test 1: NVIDIA NCA-GENL: Generative AI & LLM Certification Prep

- **Questions:** 44
- **Time limit:** 90 minutes
- **Passing score:** 70%%

> Answers are in `answer-key-1.md`. Track results in `score-tracker.md`.

---

**Q1.** A financial services company needs to predict customer loan default risk using historical data with 47 features including income, credit score, and payment history. The model must provide probability scores and explain which features most influenced each prediction for regulatory compliance. Which approach BEST meets these requirements?

A) Use a deep neural network with 5 hidden layers and ReLU activation to capture complex feature interactions, then apply gradient-based attribution methods
B) Implement a simple logistic regression model since it provides natural interpretability through coefficient weights
C) Train a gradient boosting model (XGBoost or LightGBM) and use SHAP values to provide feature importance explanations for individual predictions
D) Deploy a transformer-based model fine-tuned on the tabular data with attention weights serving as explanations

---

**Q2.** During neural network training, validation loss decreases for 15 epochs, then increases for the next 10 epochs while training loss continues to decrease. Training accuracy is 94% and validation accuracy is 78%. What is the PRIMARY issue and the BEST solution?

A) The model has high bias and is underfitting; increase model complexity by adding more layers
B) The model is overfitting the training data; implement dropout regularization and early stopping at epoch 15
C) The model is experiencing vanishing gradients; switch from sigmoid to ReLU activation functions
D) The learning rate is too high causing unstable convergence; reduce learning rate by factor of 10

---

**Q3.** A convolutional neural network for image classification uses the following architecture: Conv(32, 3×3) → ReLU → MaxPool(2×2) → Conv(64, 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(128) → Dense(10). Input images are 64×64 RGB. How many parameters are in the first Dense(128) layer?

A) 524,416 parameters
B) 131,200 parameters
C) 1,048,704 parameters
D) 2,097,280 parameters

---

**Q4.** Which activation function should be used in the output layer of a neural network that predicts multiple non-mutually-exclusive labels for medical diagnosis, where a patient can have diabetes AND hypertension AND heart disease simultaneously?

A) Softmax activation, which ensures all output probabilities sum to 1.0
B) Sigmoid activation applied independently to each output neuron
C) ReLU activation to prevent vanishing gradients in deep networks
D) Tanh activation to center outputs between -1 and 1 for better convergence

---

**Q5.** A company is training a recurrent neural network (RNN) for sentiment analysis on customer reviews. After 50 epochs, the model achieves only 52% accuracy on the validation set (barely better than random). Gradients during backpropagation through time are measured at 0.0001 in early time steps. Which modification would MOST effectively address this issue?

A) Increase the learning rate from 0.001 to 0.01 to allow larger weight updates
B) Add batch normalization layers between each RNN layer to stabilize gradients
C) Increase the sequence length from 50 tokens to 200 tokens to provide more context
D) Replace the standard RNN cells with Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) cells

---

**Q6.** A machine learning engineer is comparing two trained models on a binary classification task (fraud detection). Model A has Precision=0.85 and Recall=0.60. Model B has Precision=0.70 and Recall=0.88. The business cost of missing a fraud case is 10× higher than the cost of falsely flagging a legitimate transaction. Which model should be deployed?

A) Model A, because higher precision reduces false positives and overall operational costs
B) Model A, because it has a higher F1 score providing better balanced performance
C) Model B, because higher recall is critical when the cost of false negatives (missing fraud) is much higher than false positives
D) Neither model; retrain with balanced class weights to achieve Precision and Recall both above 0.80

---

**Q7.** A financial services company is building a document classification system that must process 10,000-word legal contracts. The system needs to understand references between clauses that may be 8,000 words apart. Which architectural component of transformers makes this feasible compared to RNNs?

A) Bidirectional processing that allows the model to read text forwards and backwards simultaneously
B) Positional encoding that maintains word order information throughout the network
C) Self-attention mechanism that computes relationships between all token pairs regardless of distance
D) Layer normalization that stabilizes training on long sequences

---

**Q8.** During inference, a GPT-based model generates text by sampling the next token from a probability distribution. The temperature parameter is set to 0.2. What behavior should the development team expect?

A) More creative and diverse outputs with occasional factual inconsistencies
B) Balanced outputs with moderate creativity and consistency
C) Faster inference speed due to reduced computation in the sampling layer
D) More deterministic and focused outputs that closely follow training patterns

---

**Q9.** An ML team fine-tunes a pre-trained BERT model for medical entity recognition. After training, the model performs well on the training hospital's data but poorly on data from other hospitals using different terminology. Which characteristic of transformer models explains this behavior?

A) The multi-head attention mechanism overfits to specific token patterns in the training data
B) The feedforward layers learn hospital-specific representations that don't generalize
C) The tokenizer's vocabulary doesn't include medical terms from other hospitals
D) The model's context window is too small to capture domain-specific terminology

---

**Q10.** A company needs to deploy a language model for real-time customer service chat. The model must respond within 200ms while handling 1,000 concurrent users. Which transformer optimization technique directly reduces per-request latency?

A) Implementing KV-cache to store previously computed attention keys and values during autoregressive generation
B) Using quantization to reduce model weights from FP32 to INT8 precision
C) Applying LoRA (Low-Rank Adaptation) to reduce the number of trainable parameters
D) Deploying multiple model replicas behind a load balancer to distribute requests

---

**Q11.** An organization trains a custom LLM from scratch and notices that deeper layers (layers 20-24 of a 24-layer model) show lower activation magnitudes and less diverse attention patterns compared to middle layers (layers 10-14). What does this indicate about the model's architecture?

A) The model has insufficient training data for the number of parameters
B) The learning rate was too high, causing gradient instability in deeper layers
C) The model may benefit from fewer layers as deeper layers are learning redundant representations
D) The model requires additional regularization techniques like dropout in later layers

---

**Q12.** A research team implements scaled dot-product attention with input sequences of 4,096 tokens. To prevent memory overflow during training with batch size 8, which modification to the attention mechanism should they implement?

A) Sparse attention that only computes attention for a fixed window of nearby tokens
B) Flash Attention that recomputes attention scores during backward pass instead of storing them
C) Multi-query attention that shares key and value projections across attention heads
D) Gradient checkpointing that stores only a subset of activations during forward pass

---

**Q13.** A research team is fine-tuning a decoder-only transformer model for a code generation task. During training, they notice that the model performs well on short code snippets but struggles with generating longer, coherent functions that maintain context across 200+ tokens. The model uses standard causal attention masks. Which architectural modification would BEST address this issue while maintaining computational efficiency?

A) Add Rotary Position Embeddings (RoPE) to better capture relative positional information at longer distances
B) Implement sliding window attention with a window size of 512 tokens and local attention patterns
C) Replace the causal attention mechanism with full bidirectional attention across all tokens
D) Increase the number of attention heads from 8 to 32 while reducing the model dimension per head

---

**Q14.** A developer is building an AI assistant that must generate SQL queries from natural language. Initial tests show the model frequently generates syntactically correct but logically incorrect queries (e.g., using wrong JOIN types or missing WHERE clauses). Which prompt engineering technique would BEST improve query accuracy?

A) Add examples of correct natural language-to-SQL conversions in the system prompt with explanations of the logic
B) Increase the temperature parameter to 0.9 to allow more creative query generation
C) Use a longer context window by including the entire database schema documentation
D) Implement prompt chaining where one model generates the query and another model validates syntax

---

**Q15.** A company needs its LLM application to consistently return product recommendations in a specific JSON format with fields: product_id (integer), name (string), price (float), and reasoning (string). Early tests show the model sometimes returns valid JSON but with incorrect field types or missing fields. What is the MOST reliable approach to ensure consistent output format?

A) Use structured output enforcement with JSON schema validation in the API call
B) Include multiple examples of the correct JSON format in the prompt and request the model to follow them exactly
C) Set temperature to 0.0 and add a statement 'You MUST return valid JSON with all required fields'
D) Implement retry logic that re-prompts the model up to 3 times if the output doesn't match the format

---

**Q16.** An AI application generates technical documentation from code. Users report that outputs often include hallucinated function names and parameters that don't exist in the actual codebase. The application currently uses a single prompt with the code and instruction to 'document this code thoroughly.' Which combination of techniques would BEST reduce hallucinations?

A) Use chain-of-thought prompting to have the model first list all functions and parameters it sees, then generate documentation based only on that list
B) Switch to a larger model with more parameters, as larger models hallucinate less frequently
C) Add the instruction 'Do not make up any information. Only describe what exists in the code provided.' to the prompt
D) Reduce the max_tokens parameter to force shorter, more focused responses

---

**Q17.** A customer service AI must handle inquiries across 12 different product categories, each with unique policies, return windows, and warranty terms. The application currently uses one massive prompt containing all policies (8,000 tokens), causing slow response times and high costs. Which architecture would BEST optimize performance while maintaining accuracy?

A) Use prompt chaining with an initial classifier model to identify the product category, then route to category-specific prompts containing only relevant policies
B) Split the 8,000-token prompt into 12 separate prompts and randomly select one for each query to reduce average context size
C) Implement RAG (Retrieval-Augmented Generation) to dynamically retrieve only relevant policy sections based on the customer query
D) Use prompt compression techniques to reduce the 8,000 tokens to under 2,000 tokens by abbreviating policy text

---

**Q18.** A legal document analysis application uses an LLM to extract key dates, parties, and obligations from contracts. During testing, the model correctly identifies most information but occasionally misses critical details buried in dense legal language or confuses similar but distinct terms (e.g., 'Licensor' vs 'Licensee'). The current prompt is: 'Extract key information from this contract.' What single change would MOST improve extraction accuracy?

A) Increase the context window size to ensure the entire contract fits in one prompt
B) Use a higher temperature setting (0.8-1.0) to encourage more thorough exploration of the document
C) Break the contract into smaller chunks and process each chunk separately, then combine results
D) Add specific definitions of each term to extract and explicit examples of each type of information from sample contracts

---

**Q19.** An e-commerce company's AI chatbot must provide product recommendations while adhering to strict content guidelines: never mention competitors, avoid making medical claims, and never promise specific outcomes. Initial deployment shows the model occasionally violates these rules despite guidelines in the system prompt. Which technique provides the MOST robust safeguard against violations?

A) Implement a two-stage process: generate response with original prompt, then use a second specialized prompt to review and revise the response for guideline violations
B) Add more emphatic language to the system prompt such as 'NEVER UNDER ANY CIRCUMSTANCES mention competitors or make medical claims'
C) Use constitutional AI principles by having the model critique its own responses against guidelines before finalizing
D) Create a deterministic post-processing filter that scans outputs for banned terms and competitor names, blocking violating responses

---

**Q20.** A financial services company is deploying a generative AI system to assist loan officers in making credit decisions. The system generates explanations for recommended approval or denial of loan applications. Which approach BEST ensures the system remains aligned with the company's responsible AI principles while maintaining regulatory compliance?

A) Use prompt engineering to instruct the model to avoid considering protected characteristics, then deploy the system with automated logging of all decisions
B) Fine-tune the model on historical loan data to maximize prediction accuracy, then conduct quarterly bias audits on approved applications
C) Deploy the system with confidence thresholds that automatically approve high-confidence recommendations and flag low-confidence cases for human review
D) Implement a human-in-the-loop workflow where loan officers must review and approve all AI-generated recommendations before final decisions, with mandatory documentation of any overrides

---

**Q21.** During red-teaming exercises, a security researcher discovers that a customer-facing chatbot can be manipulated through carefully crafted prompts to reveal training data that includes customer email addresses. What is the MOST comprehensive mitigation strategy?

A) Add a system prompt instructing the model to never reveal personal information under any circumstances
B) Rate-limit API calls from individual users to prevent automated extraction attempts
C) Implement output filtering to detect and redact email addresses and other PII patterns before responses reach users, combined with retraining on sanitized data and regular adversarial testing
D) Fine-tune the model with examples of prompt injection attempts and desired refusal responses

---

**Q22.** What is the primary purpose of Constitutional AI (CAI) in model alignment?

A) To encrypt model weights and prevent unauthorized access to the training data
B) To train models to critique and revise their own responses based on a set of principles, reducing reliance on human feedback for harmfulness criteria
C) To implement legal compliance checking that ensures all model outputs meet regulatory requirements
D) To create a governance framework that defines organizational roles and responsibilities for AI deployment

---

**Q23.** A healthcare AI startup is developing a diagnostic assistant that analyzes medical images and patient history to suggest potential diagnoses. During testing, the team discovers the model performs significantly worse (15% lower accuracy) on images from patients with darker skin tones. The model was trained on a dataset of 50,000 images, with 8% representing darker skin tones. Which intervention addresses the root cause MOST effectively?

A) Collect and add 10,000 additional images of darker skin tones to the training set, then retrain the model with balanced sampling and evaluate performance across demographic groups
B) Apply post-processing calibration that adjusts confidence scores upward for predictions on darker skin tone images
C) Add a disclaimer in the user interface warning that accuracy may be lower for certain skin tones
D) Use data augmentation techniques to synthetically modify existing images and increase apparent diversity

---

**Q24.** An enterprise is implementing RLHF (Reinforcement Learning from Human Feedback) to align their internal coding assistant with company coding standards and security practices. During reward model training, which practice introduces the GREATEST risk of reward hacking?

A) Using a small team of 5 senior engineers to provide all feedback, rather than a diverse group of engineers at different skill levels
B) Training the reward model on a narrow set of examples that emphasize code brevity and execution speed, without equally weighting security, maintainability, and documentation quality
C) Collecting feedback over a 3-month period rather than continuously updating the reward model with new examples
D) Having different engineers rate the same outputs to measure inter-rater reliability before finalizing the reward model

---

**Q25.** A data scientist is preparing a dataset containing customer purchase history for machine learning. The dataset has a 'purchase_date' column in ISO format (YYYY-MM-DD). To improve model performance for predicting seasonal buying patterns, which feature engineering approach would be MOST effective?

A) Convert purchase_date to Unix timestamp and normalize to 0-1 range
B) One-hot encode the purchase_date column into separate binary features
C) Create a single ordinal feature ranking dates from earliest to latest
D) Extract cyclical features using sine and cosine transformations of month and day_of_week

---

**Q26.** A machine learning pipeline processes streaming data with multiple numerical features that have different scales (age: 0-100, income: 20000-500000, credit_score: 300-850). The team notices that gradient-based models are converging slowly and performing poorly. Which preprocessing technique should be applied FIRST to address this issue?

A) Use Principal Component Analysis (PCA) to reduce dimensionality
B) Apply log transformation to compress the range of all features
C) Remove outliers using IQR method before any scaling
D) Apply StandardScaler to normalize all features to mean=0 and std=1

---

**Q27.** During exploratory data analysis, a data scientist discovers that a critical numerical feature has 15% missing values. The feature shows strong correlation with the target variable (r=0.72). The missing values appear to be Missing Completely At Random (MCAR). Business stakeholders require using all available data points. Which imputation strategy would be MOST appropriate?

A) Drop all rows with missing values in this feature to maintain data quality
B) Replace missing values with the global mean of the feature
C) Forward-fill missing values using the previous valid observation
D) Use K-Nearest Neighbors (KNN) imputation based on similar records

---

**Q28.** A company needs to visualize the relationship between customer satisfaction scores (1-10) and multiple categorical variables (product_category, region, support_tier) for executive presentation. The dataset contains 50,000 records across 8 product categories, 12 regions, and 3 support tiers. Which visualization approach would BEST communicate these multi-dimensional relationships?

A) Create a 3D scatter plot with color coding for the three categorical dimensions
B) Use a faceted violin plot showing satisfaction distribution across categories with hue encoding for support_tier
C) Generate separate box plots for satisfaction scores grouped by each categorical variable
D) Build a correlation heatmap showing pairwise relationships between all variables

---

**Q29.** A retail analytics team is building a customer segmentation model using transaction data. The 'customer_id' column is a unique alphanumeric identifier (e.g., 'CUST_A1B2C3'), and 'transaction_amount' ranges from $5 to $50,000 with extreme right skew (90th percentile: $200, 99th percentile: $5,000). Which combination of preprocessing steps is MOST appropriate for this scenario?

A) One-hot encode customer_id, apply MinMaxScaler to transaction_amount, then remove outliers above 99th percentile
B) Drop customer_id column, apply log transformation to transaction_amount, then apply StandardScaler
C) Hash encode customer_id to fixed dimensions, bin transaction_amount into quantile-based categories, then label encode
D) Keep customer_id as index, apply Box-Cox transformation to transaction_amount, then apply RobustScaler

---

**Q30.** A data science team analyzes model feature importance and discovers that a derived feature 'days_since_last_purchase' has the highest importance score (0.45), while the original features 'last_purchase_date' and 'current_date' used to create it have importance scores of 0.02 each. What does this pattern indicate about the feature engineering process?

A) The derived feature is causing data leakage and should be removed from the model
B) The original features should be removed to prevent multicollinearity with the derived feature
C) The feature engineering successfully created a more predictive representation by capturing the relevant temporal relationship
D) The model is overfitting to the derived feature due to high correlation with the target

---

**Q31.** A company is evaluating their generative AI chatbot that provides customer support responses. They want to measure how well the AI's answers match the quality and accuracy of responses written by their expert support team. The evaluation dataset contains 500 customer questions with corresponding expert-written reference answers. Which evaluation approach would BEST measure the similarity between AI-generated responses and expert responses?

A) Use semantic similarity metrics like cosine similarity on sentence embeddings from both responses
B) Calculate BLEU score by comparing n-gram overlap between generated responses and reference answers
C) Count the exact character-level edit distance (Levenshtein distance) between responses
D) Measure response latency and token generation speed as quality indicators

---

**Q32.** An e-commerce company has deployed a product description generator using a large language model. During A/B testing, Version A (temperature=0.3) generates descriptions that convert at 4.2% while Version B (temperature=0.9) converts at 3.8%. However, customer feedback surveys show Version B has 15% higher satisfaction scores. The business goal is to maximize revenue while maintaining customer trust. Which action should the team take?

A) Deploy Version A immediately since conversion rate directly impacts revenue and is 10.5% higher
B) Deploy Version B because customer satisfaction predicts long-term customer lifetime value and retention, which outweighs short-term conversion differences
C) Run an extended A/B test for 4 more weeks to reach statistical significance on both metrics
D) Implement a hybrid approach using Version B for new customers and Version A for returning customers to balance acquisition and retention

---

**Q33.** A healthcare AI team is building a medical report summarization system. They need to evaluate hallucination rates where the model generates clinical information not present in the source reports. The team has 200 test reports with human-verified summaries. Which evaluation methodology would MOST effectively detect hallucinations?

A) Implement entailment checking where an NLI model verifies each summary sentence is entailed by the source report
B) Use perplexity scores to measure how surprised the model is by its own generations
C) Calculate ROUGE-L scores between generated summaries and reference summaries
D) Measure the model's confidence scores on generated tokens to identify uncertain outputs

---

**Q34.** A financial services company is comparing three prompt engineering strategies for their investment analysis assistant: Chain-of-Thought (CoT), ReAct (Reasoning + Acting), and direct prompting. After testing on 300 analysis scenarios, CoT achieves 78% accuracy, ReAct achieves 85% accuracy but costs 3.2x more in tokens, and direct prompting achieves 71% accuracy. The application serves 50,000 queries daily with a $15,000 monthly inference budget. Which solution BEST balances accuracy and cost constraints?

A) Deploy CoT as the default method since it provides good accuracy improvement over direct prompting at reasonable cost
B) Implement ReAct for all queries to maximize accuracy since investment decisions justify the higher cost
C) Use a routing system: direct prompting for simple queries (identified by keyword matching), CoT for complex queries, eliminating ReAct entirely
D) Use direct prompting for 80% of queries and ReAct for the top 20% highest-value queries identified by transaction amount

---

**Q35.** A content moderation team is evaluating their AI system that classifies user-generated content as safe, questionable, or unsafe. After deployment, they discover the system has 94% precision and 71% recall for detecting unsafe content. The business priority is protecting users from harmful content while minimizing false positives that frustrate legitimate users. Which adjustment would BEST align the system with business priorities?

A) Increase the classification threshold to improve precision further, ensuring flagged content is truly unsafe
B) Maintain current thresholds since 94% precision indicates the system rarely makes mistakes
C) Lower the classification threshold to increase recall, accepting more false positives to catch more unsafe content
D) Implement a two-stage system: use current model for initial screening, then human review for all questionable classifications

---

**Q36.** A financial services company needs to deploy a Llama 2 70B model for real-time fraud detection with sub-100ms latency requirements. The model will handle 10,000 concurrent requests during peak hours. Which NVIDIA deployment solution BEST meets these requirements?

A) Deploy using NVIDIA Triton Inference Server with TensorRT-LLM optimization and dynamic batching on A100 GPUs
B) Use NVIDIA NeMo Framework with PyTorch backend on H100 GPUs without quantization
C) Deploy using standard PyTorch inference on V100 GPUs with model parallelism
D) Use NVIDIA AI Enterprise with vLLM on T4 GPUs with FP32 precision

---

**Q37.** An AI team is implementing RAG (Retrieval-Augmented Generation) for a customer support chatbot using NVIDIA NeMo Retriever. The system must handle 50 million document embeddings with semantic search latency under 50ms. What is the MINIMUM GPU memory requirement per GPU for the embedding model deployment?

A) 8GB for the embedding model and vector operations
B) 24GB to accommodate embedding model, batch processing, and FAISS indices
C) 80GB to store all embeddings in GPU memory simultaneously
D) 40GB for multi-GPU sharding of the embedding workload

---

**Q38.** A developer is fine-tuning a GPT model using NVIDIA NeMo Framework with the following command: `python train.py model.tensor_model_parallel_size=4 model.pipeline_model_parallel_size=2`. The training job fails with a NCCL timeout error. What is the MOST likely root cause?

A) InfiniBand network connectivity issue between GPU nodes preventing collective communication
B) Insufficient GPU memory for the model weights and optimizer states
C) Incorrect gradient accumulation steps configuration in the training script
D) Model checkpoint incompatibility with the specified parallelism configuration

---

**Q39.** Which NVIDIA TensorRT-LLM optimization technique provides the GREATEST inference speedup for large language models while maintaining quality above 95% of the original FP16 model?

A) SmoothQuant INT8 weight-only quantization with per-channel scaling
B) FP8 quantization with Transformer Engine on Hopper architecture GPUs
C) AWQ (Activation-aware Weight Quantization) with 4-bit weights
D) Dynamic per-token quantization with FP16 fallback for outlier activations

---

**Q40.** A company is deploying a multi-tenant LLM service using NVIDIA Triton Inference Server. Each tenant requires isolated model instances with dedicated GPU memory allocation. The service must support LoRA adapters for per-tenant customization. Which Triton configuration setting is REQUIRED to enable this architecture?

A) Set instance_group with CUDA shared memory allocation and enable model warmup
B) Configure model versioning with separate model repositories per tenant
C) Enable dynamic batching with per-request routing based on tenant ID headers
D) Use the Python backend with separate model instances and configure instance_group count equal to the number of tenants

---

**Q41.** An ML engineer needs to quantize a Mixtral 8x7B model using NVIDIA's quantization toolkit. The deployment target is A100 40GB GPUs. What is the MAXIMUM number of bits per weight that allows the full model to fit in a single GPU while maintaining kernel efficiency with TensorRT-LLM?

A) 8-bit INT8 quantization with symmetric per-tensor scaling
B) 6-bit mixed precision quantization with activation caching disabled
C) 4-bit GPTQ quantization with group size 128
D) 3-bit NF3 quantization with dequantization on-the-fly

---

**Q42.** A financial services company needs to deploy a fine-tuned Llama 3.1 70B model for real-time fraud detection with sub-200ms latency requirements. The model must handle 500 concurrent requests during peak hours and support dynamic batching. Which NVIDIA deployment solution BEST meets these requirements?

A) Use NVIDIA NeMo Framework's built-in inference server with PyTorch backend
B) Deploy using NVIDIA Triton Inference Server with TensorRT-LLM optimization and dynamic batching enabled
C) Deploy directly using vLLM with continuous batching on bare metal servers
D) Use NVIDIA AI Enterprise with Base Command Manager for model orchestration

---

**Q43.** An enterprise is building a RAG (Retrieval-Augmented Generation) pipeline using NVIDIA NeMo Retriever. The system must index 50 million documents and support semantic search with re-ranking. Which component combination provides the embedding generation and re-ranking capabilities?

A) NVIDIA cuVS library for embeddings with Triton Inference Server hosting a separate re-ranker model
B) RAPIDS cuDF for document processing with PyTorch-based embedding and re-ranking
C) NVIDIA Merlin for embeddings with TensorRT for re-ranking inference
D) NVIDIA NeMo Retriever uses its own embedding microservice and separate re-ranking microservice, both deployable via NIM

---

**Q44.** A development team needs to optimize their custom LLM's KV-cache memory usage during inference to increase batch size from 8 to 32 on existing A100 GPUs. The model uses standard attention mechanisms. Which optimization technique directly addresses KV-cache memory consumption?

A) Implement FlashAttention-2 to reduce memory overhead during attention computation
B) Apply INT8 quantization using TensorRT-LLM's quantization toolkit
C) Enable PagedAttention (used in vLLM) to manage KV-cache as paged blocks with dynamic allocation
D) Use tensor parallelism to distribute the model across multiple GPUs

---

