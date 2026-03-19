
# Concrete Strength Prediction using Neural Networks (Keras)

## Overview
This project builds and evaluates deep learning regression models using Keras to predict the compressive strength of concrete based on its composition.

The dataset includes features such as cement, water, aggregates, and additives, and the goal is to learn a mapping from these inputs to the final strength.

---

## Dataset
- Source: IBM Cognitive Class concrete dataset
- Target variable: `Strength`
- Features: Cement, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, etc.

Example:
A 28-day-old concrete mix with:
- Cement: 540
- Water: 162
- Superplasticizer: 2.5
- Coarse Aggregate: 1040
- Fine Aggregate: 676  

→ Strength: 79.99 MPa

---

## Workflow

### 1. Data Preparation
- Split into:
  - Predictors `X`
  - Target `y`
- Normalize features:

Plain text formula:
```
X_norm = (X - mean(X)) / std(X)
```

---

### 2. Model Architecture

Two models were tested:

#### Model 1
- Input layer
- 2 hidden layers (50 neurons each, ReLU)
- Output layer (1 neuron)

#### Model 2 (Deeper)
- Input layer
- 5 hidden layers (50 neurons each, ReLU)
- Output layer

---

## Mathematical Formulation

### Neural Network Forward Pass

Each layer computes:

```
z = W * x + b
a = activation(z)
```

For ReLU:
```
ReLU(z) = max(0, z)
```

Final output (regression):
```
y_hat = output of last layer (linear)
```

---

### Loss Function (Mean Squared Error)

```
MSE = (1/n) * sum((y - y_hat)^2)
```

This measures prediction error.

---

### Optimization (Adam)

Weights are updated iteratively:

```
w = w - learning_rate * gradient(loss)
```

Adam improves this by using adaptive learning rates and momentum.

---

## Training

- Epochs: 100
- Validation split:
  - Model 1: 30%
  - Model 2: 10%

Key idea:
- More layers → higher capacity → better pattern learning
- Less validation split → more training data → better generalization (if not overfitting)

---

## Key Observations

- Increasing hidden layers improves the model’s ability to capture complex relationships.
- Normalization is critical for stable training.
- Using more training data (smaller validation split) improves performance.
- Deep models perform better but risk overfitting if not controlled.

---

## Requirements

Install dependencies:

```
pip install numpy==2.0.2 pandas==2.2.3 tensorflow==2.18.0
```

---

## How to Run

1. Open the notebook
2. Run all cells sequentially
3. Ensure all other blocks are commented when testing specific configurations

---

## Project Structure

```
Keras-regression-models.ipynb
```

---

## Summary

This project demonstrates:
- Regression with neural networks
- Effect of model depth
- Importance of normalization
- Trade-off between training and validation data

The approach can be generalized to any tabular regression problem.
````
