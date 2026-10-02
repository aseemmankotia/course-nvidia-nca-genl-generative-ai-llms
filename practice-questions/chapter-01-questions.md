## Chapter 1: Machine Learning and Neural Network Fundamentals — Practice Questions

**Q1.** A data science team is building a model to predict customer churn. They have a dataset of 100,000 customer records with historical labels indicating whether each customer churned. They split the data 70/15/15 for training, validation, and test sets. After training, the model achieves 95% accuracy on training data but only 72% on validation data. What is the BEST next step?

A) Increase the training dataset size by moving samples from the validation set
B) Apply L2 regularization to the model and reduce model complexity
C) Switch from supervised learning to unsupervised learning approach
D) Remove the validation set and use only training and test sets

<details>
<summary>Answer</summary>

**Correct: B**

The 23-point gap between training (95%) and validation (72%) accuracy is a classic sign of overfitting. L2 regularization penalizes large weights and reduces model complexity, directly addressing overfitting. This is the standard remedy.

**Why the others are wrong:**
- Moving samples from validation to training would make overfitting worse by giving the model more data to memorize and reducing validation set reliability
- This is a labeled prediction problem (churn yes/no), which is definitionally supervised learning. Unsupervised learning cannot predict labeled outcomes
- The validation set is essential for detecting overfitting. Removing it would eliminate the ability to tune hyperparameters and detect this exact problem

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q2.** An AI startup is building a recommendation system for a streaming platform. They have user viewing history but no explicit ratings or labels. They want to group users with similar viewing patterns to make recommendations. Which machine learning paradigm should they use?

A) Unsupervised learning with clustering
B) Supervised learning with regression
C) Reinforcement learning with Q-learning
D) Supervised learning with classification

<details>
<summary>Answer</summary>

**Correct: A**

No labels exist (no explicit ratings), and the goal is to discover hidden patterns (grouping similar users). This is the defining use case for unsupervised learning, specifically clustering algorithms like K-means.

**Why the others are wrong:**
- Supervised learning requires labeled data. Regression predicts continuous values from labels, but there are no labels in this scenario
- Reinforcement learning requires an environment where an agent takes actions and receives rewards. Grouping historical data has no agent-environment interaction
- Supervised classification requires labeled categories. The problem explicitly states there are no labels, only viewing history patterns

</details>

---

**Q3.** A robotics company is training an autonomous robot to navigate a warehouse. The robot receives +10 points for successful package delivery, -5 points for collisions, and 0 points for safe movement. The robot learns through trial and error over thousands of episodes. Which learning paradigm is this?

A) Supervised learning because the robot learns from labeled examples
B) Unsupervised learning because there are no predefined paths
C) Semi-supervised learning combining labeled and unlabeled data
D) Reinforcement learning because the robot learns from rewards and penalties

<details>
<summary>Answer</summary>

**Correct: D**

The robot learns through trial-and-error interaction with an environment, receiving numerical rewards (+10, -5, 0) for actions. This agent-environment-reward structure is the definition of reinforcement learning.

**Why the others are wrong:**
- There are no labeled input-output pairs. The robot isn't shown 'correct' navigation paths to mimic; it discovers them through rewards
- While there are no predefined paths, unsupervised learning finds patterns in unlabeled data. This scenario has clear reward signals guiding learning
- Semi-supervised learning uses a small labeled set plus large unlabeled set. This scenario has no labeled examples, only reward signals

</details>

---

**Q4.** A financial services company splits their fraud detection dataset into 60% training, 20% validation, and 20% test sets. After deployment, the model performs significantly worse on real production data than on the test set. What is the MOST likely cause?

A) The test set was too small at only 20% of the data
B) The model was evaluated on validation data instead of test data
C) L2 regularization was applied during training
D) The training, validation, and test sets came from the same time period and production data has different patterns

<details>
<summary>Answer</summary>

**Correct: D**

This describes data distribution shift. If train/val/test all came from the same time period (e.g., 2024 data), but production is 2026 data with new fraud patterns, the model won't generalize. The test set must represent production distribution.

**Why the others are wrong:**
- 20% test set is standard and adequate. The issue is not size but representativeness of production data patterns
- The problem states the model performed well on the test set but poorly in production. This indicates the test set itself doesn't match production, not evaluation methodology
- L2 regularization improves generalization and reduces overfitting. It would help performance, not hurt it. The issue is data distribution mismatch

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q5.** A deep learning model for image classification uses dropout with a rate of 0.5 during training. What happens to the dropout layers during inference when making predictions on new images?

A) Dropout is disabled and all neurons are active, with outputs scaled appropriately
B) Dropout continues with rate 0.5 to maintain consistency
C) Dropout rate is reduced to 0.25 for inference
D) Dropout randomly activates different neurons for each prediction

<details>
<summary>Answer</summary>

**Correct: A**

During inference, dropout is turned off (rate = 0) and all neurons are used. Outputs are scaled to account for the fact that during training only 50% of neurons were active on average. This ensures consistent predictions.

**Why the others are wrong:**
- Keeping dropout active during inference would make predictions random and inconsistent. The same input would produce different outputs each time
- The dropout rate doesn't gradually decrease. It's either active (training) or completely off (inference). No intermediate rate is used
- Randomly activating neurons during inference defeats the purpose of making reliable predictions. Dropout is a training-time regularization technique only

</details>

---

**Q6.** An NLP model represents the word 'king' as a 768-dimensional dense vector embedding. The embedding for 'queen' has high cosine similarity (0.85) to 'king'. What does this embedding relationship demonstrate?

A) The embeddings capture semantic meaning and the words appear in similar contexts
B) The words have similar character sequences
C) The model has overfitted on these two words
D) The embedding dimension of 768 is too large

<details>
<summary>Answer</summary>

**Correct: A**

High cosine similarity between embeddings means the vectors point in similar directions in the embedding space. This happens when words appear in similar contexts (both are royalty, leadership roles, etc.), capturing semantic relationships.

**Why the others are wrong:**
- Character similarity would be measured by edit distance, not embedding cosine similarity. 'King' and 'queen' share only 'e' but have semantically similar embeddings
- High similarity between related concept embeddings is the desired outcome, not overfitting. Overfitting would show as poor performance on validation data
- 768 dimensions is standard for models like BERT. The similarity score (0.85) indicates the embeddings are working correctly, not that dimensionality is wrong

</details>

---

**Q7.** A company fine-tunes a large language model for customer support. During training, they observe that training loss decreases steadily to 0.02, but validation loss decreases until epoch 15, then increases to 0.18 by epoch 30. The model uses cross-entropy loss. What should they do?

A) Continue training until training loss reaches 0.01 for better performance
B) Use the model weights from epoch 15 when validation loss was lowest
C) Switch from cross-entropy loss to mean squared error loss
D) Increase the learning rate to speed up convergence

<details>
<summary>Answer</summary>

**Correct: B**

Validation loss increasing while training loss decreases is textbook overfitting. The model peaked at epoch 15. This is when to stop training (early stopping) and use those weights. Training beyond this point memorizes training data.

**Why the others are wrong:**
- Continuing training would worsen overfitting. The validation loss is already rising, indicating the model is memorizing training data rather than learning generalizable patterns
- Cross-entropy is the correct loss for classification tasks (customer support intent classification). MSE is for regression. Changing the loss function doesn't address overfitting
- Increasing learning rate would make training unstable and likely worse. The issue is overfitting (model complexity), not slow convergence

> ⚠️ Frequently missed on the exam — read the options carefully.

</details>

---

**Q8.** A neural network for sentiment analysis uses ReLU activation in hidden layers and sigmoid activation in the output layer. Why is sigmoid used specifically in the output layer for this binary classification task?

A) Sigmoid outputs values between 0 and 1, which can be interpreted as probabilities for the positive class
B) Sigmoid is computationally faster than ReLU for final predictions
C) Sigmoid prevents vanishing gradients in the output layer
D) ReLU would produce negative sentiment scores which are invalid

<details>
<summary>Answer</summary>

**Correct: A**

For binary classification, sigmoid maps any input to (0,1), which represents P(positive class). Output > 0.5 predicts positive, < 0.5 predicts negative. This probabilistic interpretation is why sigmoid is standard for binary classification outputs.

**Why the others are wrong:**
- Sigmoid is actually slower than ReLU due to exponential computation. Speed is not the reason; the (0,1) output range is the reason
- Sigmoid actually suffers from vanishing gradients for extreme inputs. It's used despite this because the output range (0,1) is required for probability interpretation
- ReLU outputs max(0,x), so negatives become 0, not preserved. But the real reason is ReLU outputs unbounded positive values, not probabilities. Sigmoid bounds output to (0,1)

</details>

---

