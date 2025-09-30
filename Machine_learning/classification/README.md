# Softmax Regression (Multinomial Logistic Regression) from Scratch

This Python code implements a **Softmax Regression** (also known as **Multinomial Logistic Regression**) classifier using **Gradient Descent**.  
It is designed for **multi-class classification**, where the goal is to assign an input to one of several possible classes.  

The implementation relies on **NumPy** for matrix operations.

---

## 📘 Code Explanation

### 1. The `LogisticRegression` Class

- **`__init__(self, learning_rate=0.01)`**  
  - Initializes the model parameters.  
  - `self.weights`: Model weights (coefficients), set during training.  
  - `self.lr`: Learning rate (step size for gradient descent).  
  - `self.losses`: Stores the loss at each iteration for monitoring convergence.

- **`softmax(self, z)`**  
  - Applies the softmax function to raw scores `z`.  
  - Uses numerical stability trick: `np.exp(z - np.max(z, axis=1, keepdims=True))` to prevent overflow.  
  - Returns probabilities that sum to 1 across each row (sample).

- **`fit(self, train_x, train_y, iterations=1000)`**  
  - Trains the model using gradient descent.  
  - Steps:  
    1. **Setup:**  
       - `m`: number of samples, `n`: number of features, `k`: number of classes.  
       - Initialize weights: shape `(n+1, k)` → `+1` for bias.  
       - Add bias column to features: `X = np.c_[np.ones(m), train_x]`.  
       - One-hot encode labels: `Y = np.eye(k)[train_y]`.
    2. **Training loop:**  
       - Compute linear scores: `z = X ⋅ weights`.  
       - Apply softmax to get probabilities: `y_pred`.  
       - Compute cross-entropy loss:  
         ```python
         loss = (-1/m) * np.sum(Y * np.log(y_pred + 1e-15))
         ```  
       - Compute gradient: `(1/m) * X.T ⋅ (y_pred - Y)`.  
       - Update weights: `weights -= lr * gradient`.

- **`predict(self, test_x)`**  
  - Adds bias term.  
  - Computes softmax probabilities.  
  - Returns predicted class labels using `np.argmax`.

---

## 🧮 Example with Matrices

Suppose we have:

- `n = 2` features  
- `k = 3` classes  
- `m = 4` samples  
- Learning rate `lr = 0.1`

### 1. Initial Setup

| Input | Shape | Example | Description |
|-------|-------|---------|-------------|
| `train_x` | (4,2) | `[[1,2],[3,4],[5,6],[7,8]]` | Features |
| `train_y` | (4,) | `[0,1,2,0]` | Class labels |

### 2. Prepare Data

- **Add Bias:**  
  ```text
  X =
  [[1,1,2],
   [1,3,4],
   [1,5,6],
   [1,7,8]]
  Shape: (4,3)

- **One-Hot Encode Labels:**
    ```text
    Y =
    [[1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0]]
    Shape: (4,3)

- **Initial Weights:**
    ```text
    weights = 
        [[0,0,0],
         [0,0,0],
         [0,0,0]]
    Shape: (3,3)

## 3. Iteration 1
Linear Scores (z):
        z = X ⋅ weights = 0 (all zeros initially)

Predicted Probabilities (softmax) Since all scores are zero → uniform distribution

    ```text
    y_pred =
    [[1/3, 1/3, 1/3],
    [1/3, 1/3, 1/3],
    [1/3, 1/3, 1/3],
    [1/3, 1/3, 1/3]]
    Shape: (4,3)

- **Error (y_pred - Y):**
    ```text
    [[-2/3,  1/3,  1/3],
    [ 1/3, -2/3,  1/3],
    [ 1/3,  1/3, -2/3],
    [-2/3,  1/3,  1/3]]

- **gradient:**
    ```text
    gradient = (1/m) * X.T ⋅ (y_pred - Y)
    Shape: (3,3)

- **Weight Update:**
    ```text
    weights = weights - lr ⋅ gradient

This process repeats for each iteration until convergence.