# ML & Neural Network Fundamentals — Quick Reference

## Learning Paradigms

| Paradigm | Data Type | Goal | Example Use Cases |
|----------|-----------|------|-------------------|
| **Supervised** | Labeled (input + output) | Predict outputs from inputs | Spam detection, image classification, fraud detection |
| **Unsupervised** | Unlabeled (input only) | Find hidden patterns | Customer segmentation, anomaly detection, topic modeling |
| **Reinforcement** | Agent + environment + rewards | Learn optimal actions | Robotics, game AI, recommendation systems with feedback |

**Exam Trap:** If problem mentions "no labels" → unsupervised. If "agent receives rewards" → reinforcement.

---

## Dataset Splits

| Set | Purpose | Typical Size | When Used |
|-----|---------|--------------|----------|
| **Training** | Learn model parameters (weights) | 60-80% | Every training iteration |
| **Validation** | Tune hyperparameters, detect overfitting | 10-20% | After each epoch to check performance |
| **Test** | Final evaluation only | 10-20% | Once at the very end, never during training |

**Rule:** Never touch test data until final evaluation. Using it earlier invalidates your results.

---

## Overfitting vs Underfitting

| Condition | Training Loss | Validation Loss | Cause | Solution |
|-----------|--------------|-----------------|-------|----------|
| **Overfitting** | Very low (e.g., 0.02) | High (e.g., 0.18) | Model memorizes training data | L1/L2 regularization, dropout, more training data, reduce model complexity |
| **Underfitting** | High (e.g., 0.65) | High (e.g., 0.68) | Model too simple | Increase model capacity, add features, train longer |
| **Good fit** | Low (e.g., 0.12) | Similar low (e.g., 0.15) | Balanced | No action needed |

**Exam Scenario:** If training accuracy = 95% but validation = 72% → overfitting (23-point gap).

---

## Regularization Techniques

| Technique | How It Works | Effect | When to Use |
|-----------|--------------|--------|-------------|
| **L1 (Lasso)** | Adds Σ∣weights∣ to loss | Sparse weights (many → 0) | Feature selection, interpretability |
| **L2 (Ridge)** | Adds Σ(weights²) to loss | Small weights (rarely 0) | General overfitting prevention (most common) |
| **Dropout** | Randomly deactivates neurons during training | Forces redundancy, prevents co-adaptation | Deep neural networks |

**Dropout at Inference:** Turned OFF (rate = 0). All neurons active. Exam tests this.

---

## Activation Functions

| Function | Formula | Range | Use Case | Note |
|----------|---------|-------|----------|------|
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers | Fast, reduces vanishing gradients |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary classification output | Outputs interpreted as probabilities |
| **Tanh** | (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | (-1, 1) | Hidden layers (less common than ReLU) | Zero-centered output |
| **Softmax** | eˣⁱ / Σeˣʲ | (0, 1), sums to 1 | Multi-class classification output | Outputs probability distribution |

**Exam Rule:** ReLU in hidden layers. Sigmoid for binary output. Softmax for multi-class output.

---

## Loss Functions

| Loss | Use Case | Output Type | Formula |
|------|----------|-------------|----------|
| **Cross-Entropy** | Classification | Probabilities | -Σ y·log(ŷ) |
| **Mean Squared Error (MSE)** | Regression | Continuous values | Σ(y - ŷ)² / n |
| **Mean Absolute Error (MAE)** | Regression (robust to outliers) | Continuous values | Σ∣y - ŷ∣ / n |

**Exam Trap:** Using MSE for classification or cross-entropy for regression is wrong.

---

## Embeddings

**Definition:** Dense vector representations (e.g., 768 dimensions) that encode semantic meaning.

| Model | Embedding Dimension | Use Case |
|-------|---------------------|----------|
| BERT-base | 768 | General NLP, sentence encoding |
| BERT-large | 1024 | Higher accuracy NLP tasks |
| Sentence Transformers | 384-768 | Semantic search, similarity |
| GPT-3 | 12288 | Large-scale language generation |

**Why Embeddings Matter for LLMs:**
- Capture semantic relationships ("king" - "man" + "woman" ≈ "queen")
- Enable transfer learning (pre-trained embeddings → downstream tasks)
- Cosine similarity measures semantic closeness (0.85 = very similar)

**Exam Note:** High cosine similarity between embeddings = words used in similar contexts.

---

## Gradient Descent

**Purpose:** Minimize loss by updating weights in direction opposite to gradient.

**Process:**
1. Forward pass: Compute predictions and loss
2. Backward pass (backpropagation): Compute gradients
3. Update weights: w = w - learning_rate × gradient

**Learning Rate:**
- Too high → Training unstable, loss oscillates
- Too low → Training very slow
- Typical range: 0.001 - 0.01

**Exam Tip:** Adam optimizer is most common (adaptive learning rates).
