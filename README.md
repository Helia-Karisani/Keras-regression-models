
# Concrete Strength Prediction with Keras Regression Models

## Overview
This project uses Keras to build neural network regression models for predicting the compressive strength of concrete from its material composition. The notebook explores how model depth and training configuration affect predictive performance on a tabular dataset.

The target is a continuous numeric value, so this is a regression task rather than a classification task.

---

## Dataset
The dataset contains concrete mixture attributes and the resulting compressive strength.

### Input features
- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Fine Aggregate
- Age

### Target
- Concrete compressive strength

This setup makes the project a supervised learning problem where the model learns a function:

```text
f(x) = y
```

where:
- `x` = vector of concrete mixture features
- `y` = predicted compressive strength

---

## Method Used
The notebook uses a feedforward artificial neural network implemented with Keras.

Each hidden layer applies a linear transformation followed by a nonlinear activation:

```text
z = W x + b
a = ReLU(z)
```

where:
- `W` = weight matrix
- `x` = input to the layer
- `b` = bias
- `a` = output activation

The ReLU activation function is:

```text
ReLU(z) = max(0, z)
```

Because this is a regression problem, the output layer uses a single neuron with linear output:

```text
y_hat = final layer output
```

---

## Data Normalization
Before training, the predictors are normalized so that each feature is centered and scaled.

Plain text formula:

```text
x_normalized = (x - mean(x)) / std(x)
```

This helps training because features on very different scales can make optimization unstable or slower.

---

## Loss Function
The models are trained using mean squared error (MSE), which is standard for regression.

```text
MSE = (1 / n) * sum((y - y_hat)^2)
```

where:
- `y` = true value
- `y_hat` = predicted value
- `n` = number of samples

A lower MSE means the model's predictions are closer to the true concrete strength values.

---

## Optimization
The notebook uses the Adam optimizer.

In general, neural network training updates weights to reduce the loss:

```text
w_new = w_old - learning_rate * d(loss)/dw
```

Adam improves standard gradient descent by adaptively adjusting update sizes for different parameters and typically converges faster in practice.

---

## Model Configurations

### First model
The first regression model uses:
- Input layer
- Hidden layer with 50 neurons and ReLU
- Hidden layer with 50 neurons and ReLU
- Output layer with 1 neuron

### Modified model
The second model increases depth and uses:
- Input layer
- 5 hidden layers
- 50 neurons in each hidden layer
- ReLU activation in each hidden layer
- Output layer with 1 neuron

This allows the deeper model to represent more complex nonlinear relationships in the dataset.

---

## Training Setup
The notebook trains both models for 100 epochs.

### Initial configuration
- Normalized predictors
- Validation split: 0.3
- Epochs: 100

### Modified configuration
- Normalized predictors
- Validation split: 0.1
- Epochs: 100
- More hidden layers

---

## Why This Works
A neural network learns patterns by composing multiple transformations across layers. For tabular data like this concrete dataset, the model can learn how combinations of ingredients and age influence compressive strength.

A deeper network can represent more complicated functions, which may improve fit when the relationship between inputs and output is not purely linear.

---

## File
```text
Keras-regression-models.ipynb
```

---

## Requirements
Install the required packages before running the notebook.

```bash
pip install numpy pandas tensorflow keras scikit-learn matplotlib
```

---

## How to Run
1. Open the notebook in Jupyter Notebook or JupyterLab.
2. Install the required dependencies.
3. Run the cells in order.
4. Review the training output for both model configurations.

---

## Project Highlights
- Regression modeling with Keras
- Feature normalization
- Mean squared error loss
- Adam optimizer
- Comparison of shallow and deeper neural network architectures
- Effect of changing validation split

---

## Conclusion
This project demonstrates how neural networks can be applied to a real-world regression problem using structured data. It also shows that changes in architecture and data split can affect model learning and final performance.

Increasing the number of hidden layers gives the model greater ability to capture complex patterns in the data, which can improve how well it fits the training set and, in turn, strengthen its predictions.

Using a smaller validation split leaves more data available for training, giving the model more examples from which to learn. With access to a larger training set, it can better recognize underlying trends and potentially achieve stronger overall performance.
````
