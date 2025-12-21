# Evolution Strategies for Logistic Regression on the Heart Disease Dataset

## 1.Introduction

In this project, we study **Evolution Strategies (ES)** as a derivative-free optimization method and apply them to train a **binary logistic regression classifier** on the Heart Disease dataset. Unlike gradient-based optimization methods (e.g., gradient descent), ES relies on stochastic perturbations and selection, making it suitable for problems where gradients are unavailable, unreliable, or noisy.

The goal is to learn a model that predicts the **presence or absence of heart disease** using clinical features, and to analyze its performance using standard classification metrics.

---

## 2. Dataset Description and Preprocessing

### 2.1 Dataset

We use the UCI Heart Disease dataset. Each row corresponds to a patient record with clinical measurements such as age, cholesterol level, chest pain type, and exercise-induced angina.

The original target variable is:

- `num ∈ {0,1,2,3,4}`

where:

- `0` indicates absence of heart disease,
- values greater than `0` indicate presence of heart disease.

### 2.2 Target Binarization

Since logistic regression is a **binary classifier**, we convert the target into a binary variable:

- `y = 0` → no disease
- `y = 1` → disease present

This is done by mapping:

```
y = 1 if num > 0 else 0

```

### 2.3 Train/Test Split

The dataset is split into:

- 70% training data
- 30% test data

The split is **stratified** with respect to the target label to preserve class proportions in both sets.

### 2.4 Feature Scaling

All numerical features are standardized using **z-score normalization**:

$x_j^{scaled} = \frac{x_j - \mu_j}{\sigma_j}$

where (\mu_j) and (\sigma_j) are computed **only on the training set** and then applied to both training and test sets.

This improves numerical stability and makes optimization easier.

---

## 3. Logistic Regression Model

### 3.1 Model Definition

Logistic regression models the probability of heart disease as:

$\hat{y} = \sigma(W^T x + b)$

where:

- (W) is the weight vector,
- (b) is the bias,
- $(\sigma(z) = \frac{1}{1 + e^{-z}})$ is the sigmoid function.

### 3.2 Loss Function

We optimize the **cross-entropy loss**:

$\mathcal{L}{CE} = -\frac{1}{N} \sum{i=1}^N \big[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\big]$

An optional **L2 regularization** term is added to penalize large weights:

[$\mathcal{L}(\theta) = \mathcal{L}{CE} + \lambda{reg} ||W||_2^2$]

---

## 4. Evolution Strategies (ES)

### 4.1 Parameter Representation

Each individual in the ES population represents a candidate solution:

[$\theta = [W, b] \in \mathbb{R}^{d+1}$]

Additionally, each individual maintains **strategy parameters** (mutation step sizes):

[$\sigma \in \mathbb{R}^{d+1}$]

Thus, an individual is:

```
Individual = (theta, sigma)

```

---

### 4.2 Initialization

- Model parameters (W, b) are initialized uniformly in ([-0.1, 0.1]).
- Step sizes (\sigma_i) are initialized uniformly in ([0.01, 0.1]).

---

### 4.3 Self-Adaptive Mutation

Mutation is performed using **self-adaptive Evolution Strategies**, where step sizes evolve along with parameters.

We define learning rates:

$\tau = \frac{1}{\sqrt{2n}}, \quad \tau' = \frac{1}{\sqrt{2\sqrt{n}}}$

where (n = d+1).

Step-size update:

$\sigma_i' = \sigma_i \cdot \exp(\tau N(0,1) + \tau' N_i(0,1))$

Parameter update:

$\theta_i' = \theta_i + \sigma_i' N_i(0,1)$

To ensure numerical stability:

- Step sizes are lower-bounded by (10^{-6})
- Parameters are clipped to ([-5, 5])

---

### 4.4 Fitness Evaluation

Fitness is defined as the **negative loss**:

$fitness(\theta) = -\mathcal{L}(\theta)$

This converts the minimization problem into a maximization problem, which aligns with ES selection.

---

### 4.5 Selection Strategy

We use a **(μ + λ)-ES** scheme:

- μ parents generate λ offspring
- Parents and offspring are merged
- The top μ individuals (by fitness) survive

This guarantees **elitism**, ensuring the best solutions are preserved.

---

## 5. Experimental Results

### 5.1 Fitness Convergence

We track:

- Best training loss per generation
- Mean training loss per generation

Results show a decreasing trend in both curves, indicating successful convergence of the ES algorithm.

![Figure_2.png](attachment:be41a20c-47ac-49df-9af6-6135a5cbeb0b:Figure_2.png)

### 5.2 Training Accuracy

Training accuracy steadily improves across generations, demonstrating that minimizing cross-entropy loss leads to better classification performance.

### 5.3 Test Performance

After training, the best individual is evaluated on the test set. We report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

![Figure_1.png](attachment:9d44b369-875c-4676-bc12-bb3ba7f7b2cd:Figure_1.png)

## 6. Error Analysis

From the confusion matrix, we analyze:

- False positives (predicting disease when absent)
- False negatives (missing actual disease cases)

In a medical context, **false negatives are more dangerous**, as they correspond to missed diagnoses. Therefore, recall is especially important.

---

## 7. Sensitivity Analysis

Qualitative experiments show that:

- Larger populations (μ, λ) improve stability but increase computation time
- Too small mutation rates cause stagnation
- L2 regularization helps prevent overfitting

---

## 8. Conclusion

This project demonstrates that Evolution Strategies can successfully train a logistic regression classifier without using gradient information. Despite being computationally more expensive than gradient-based methods, ES provides a flexible and robust optimization framework.

The final model achieves competitive classification performance on the Heart Disease dataset while adhering strictly to the constraint of avoiding built-in machine learning models.

## Implementation Notes

---

- Language: Python
- Libraries: NumPy, Pandas, Matplotlib, scikit-learn (metrics only)
- Random seed: fixed where applicable

All experiments are fully reproducible using the provided codebase.

### Repository Structure:

```jsx
ctx3 print // This is my open source tool! 
┌── 📂 Project structure:
├── .gitignore (11 bytes)
├── Heart Disease dataset.csv (157859 bytes)
├── data.py (720 bytes) // Data loader!
├── es_component.py (3057 bytes) // Main ES, initialize_population, mutate, evaluate_population & etc.
├── main.py (1074 bytes)
├── plot.py (414 bytes) // Plotter Functions!
└── utilites.py (421 bytes) // sigmoid, predict_proba, cross_entropy_los
```

```jsx
ctx3 context -j
{
  "root": ".",
  "files": [
    {
      "name": ".gitignore",
      "type": "gitignore",
      "path": ".gitignore",
      "size": 11,
      "lines": 1,
      "lastEdited": "2025-12-21 23:11:00.351105939 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "Heart Disease dataset.csv",
      "type": "csv",
      "path": "Heart Disease dataset.csv",
      "size": 157859,
      "lines": 610,
      "lastEdited": "2025-12-21 20:31:25.024905045 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "data.cpython-313.pyc",
      "type": "pyc",
      "path": "__pycache__/data.cpython-313.pyc",
      "size": 1314,
      "lines": 11,
      "lastEdited": "2025-12-21 22:21:14.83554162 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "es_component.cpython-313.pyc",
      "type": "pyc",
      "path": "__pycache__/es_component.cpython-313.pyc",
      "size": 5375,
      "lines": 36,
      "lastEdited": "2025-12-21 22:18:02.872029977 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "plot.cpython-313.pyc",
      "type": "pyc",
      "path": "__pycache__/plot.cpython-313.pyc",
      "size": 1158,
      "lines": 5,
      "lastEdited": "2025-12-21 20:35:09.417801079 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "utilites.cpython-313.pyc",
      "type": "pyc",
      "path": "__pycache__/utilites.cpython-313.pyc",
      "size": 1237,
      "lines": 10,
      "lastEdited": "2025-12-21 20:35:09.417603746 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "data.py",
      "type": "py",
      "path": "data.py",
      "size": 720,
      "lines": 28,
      "lastEdited": "2025-12-21 22:18:53.813311111 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "es_component.py",
      "type": "py",
      "path": "es_component.py",
      "size": 3057,
      "lines": 106,
      "lastEdited": "2025-12-21 22:17:17.168105939 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "main.py",
      "type": "py",
      "path": "main.py",
      "size": 1074,
      "lines": 32,
      "lastEdited": "2025-12-21 23:21:05.444720002 +0330 +0330",
      "isEntryPoint": true
    },
    {
      "name": "plot.py",
      "type": "py",
      "path": "plot.py",
      "size": 414,
      "lines": 17,
      "lastEdited": "2025-12-21 23:20:46.104783318 +0330 +0330",
      "isEntryPoint": false
    },
    {
      "name": "utilites.py",
      "type": "py",
      "path": "utilites.py",
      "size": 421,
      "lines": 19,
      "lastEdited": "2025-12-21 23:20:37.244794734 +0330 +0330",
      "isEntryPoint": false
    }
  ],
  "total_files": 11,
  "total_dirs": 2,
  "dependencies": null,
  "readme": ""
}
```