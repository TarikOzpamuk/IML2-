# Subtask 1

We evaluate Support Vector Machines (SVMs) using different kernels (`linear`, `poly`, `rbf`, `sigmoid`) on the **digits** dataset.
Both default-parameter performance and hyperparameter‑tuned performance.
Tuning was performed using **k-fold cross‑validation** over predefined hyperparameter grids.

---

## 1. Default Parameter Results

| Kernel     | Test Accuracy |
|------------|---------------|
| Linear     | **0.9796**    |
| Polynomial | **0.9889**    |
| RBF        | **0.9870**    |
| Sigmoid    | **0.9074**    |

The polynomial and RBF kernels perform best under default settings, while the sigmoid kernel performs significantly worse.

---

## 2. Hyperparameter Tuning (k‑fold CV)

A grid search was carried out for each kernel:

### **Linear Kernel**

- **Best parameters:** `C = 0.1`
- **Best CV score:** 0.9785
- **Test accuracy:** 0.9796
- **Improvement:** +0.0000

Hyperparameter tuning does not improve performance.

---

### **Polynomial Kernel**

- **Best parameters:**
  - `C = 0.1`
  - `degree = 3`
  - `gamma = auto`
- **Best CV score:** 0.9865
- **Test accuracy:** 0.9889
- **Improvement:** +0.0000

Hyperparameter tuning does not improve performance.

---

### **RBF Kernel**

- **Best parameters:**
  - `C = 10`
  - `gamma = 0.001`
- **Best CV score:** 0.9920
- **Test accuracy:** 0.9907
- **Improvement:** **+0.0037**

This is the only kernel where tuning yields a clear improvement.
Increasing `C` makes the decision boundary harder, while decreasing `gamma` reduces overfitting by smoothing the kernel.

---

### **Sigmoid Kernel**

- **Best parameters:**
  - `C = 1`
  - `gamma = scale`
- **Best CV score:** 0.9069
- **Test accuracy:** 0.9074
- **Improvement:** +0.0000

Hyperparameter tuning does not improve performance.

---

## 3. Summary Table

| Kernel  | Best Parameters                    | CV Score | Test Accuracy | Default Accuracy | Improvement |
|---------|------------------------------------|----------|---------------|------------------|-------------|
| Linear  | `C=0.1`                            | 0.9785   | 0.9796        | 0.9796           | +0.0000     |
| Poly    | `C=0.1`, `degree=3`, `gamma=auto` | 0.9865   | 0.9889        | 0.9889           | +0.0000     |
| RBF     | `C=10`, `gamma=0.001`              | 0.9920   | **0.9907**    | 0.9870           | **+0.0037** |
| Sigmoid | `C=1`, `gamma=scale`                | 0.9069   | 0.9074        | 0.9074           | +0.0000     |

---

## 4. Conclusions

- The **polynomial** and **RBF** kernels achieve the best test performance overall.
- **Only the RBF kernel benefits from tuning**, improving accuracy by +0.0037.
- The **sigmoid kernel consistently underperforms**, regardless of hyperparameters.
- For this dataset, **RBF with tuned parameters (`C=10`, `gamma=0.001`)** is the best-performing model.


# Subtask 2

## 1. Motivation and Idea Behind the Kernel

The digits dataset consists of **8×8 grayscale images**. Each image has a spatial structure: central pixels are usually more informative while outer pixels contain mostly background (black).

A standard RBF kernel treats all features equally. To exploit the spatial structure, we designed an **anisotropic RBF kernel**, allowing different feature dimensions to contribute differently to the distance measure.

The kernel modifies the RBF formula by applying a feature-specific weight vector (A):

$$K(x,y) = \exp\left(-\sum_i A_i (x_i - y_i)^2\right)$$

- Mid-range PCA components (which capture important digit structure) receive *higher weight*.
- Others receive a smaller baseline weight.

This creates a similarity measure adapted to the structure of digit images.

## 2. Implementation

The kernel function:

```python
def anisotropic_rbf(A):
    def kernel(X, Y):
        diffs = X[:, None, :] - Y[None, :, :]
        quad = np.einsum('...i,i,...i->...', diffs, A, diffs)
        return np.exp(-quad)
    return kernel
```

A valid kernel must be positive semi-definite (PSD).
Because this is a Gaussian kernel with a diagonal precision matrix, it is PSD and therefore a valid Mercer kernel.

## 4. Results

### Best Configuration

- PCA components: **42**
- C: **10**
- gamma: **0.001**
- Cross-validation score: **0.9928**
- Test accuracy: **0.9926**

### Confusion Matrix

```
[[53  0  0  0  0  0  0  0  0  0]
 [ 0 50  0  0  0  0  0  0  0  0]
 [ 0  0 47  0  0  0  0  0  0  0]
 [ 0  0  0 53  0  1  0  0  0  0]
 [ 0  0  0  0 60  0  0  0  0  0]
 [ 0  0  0  0  0 66  0  0  0  0]
 [ 0  0  0  0  0  0 53  0  0  0]
 [ 0  0  0  0  0  0  0 54  0  1]
 [ 0  0  0  0  0  0  0  0 43  0]
 [ 0  0  0  1  0  0  0  1  0 57]]
```

### Comparison to Previous Kernels

| Kernel                     | Best Accuracy |
|----------------------------|---------------|
| Linear                     | 0.9796        |
| Poly                       | 0.9889        |
| RBF with tuned parameters  | 0.9907        |
| **Custom Anisotropic RBF** | **0.9926**    |

The custom kernel achieves the best accuracy overall.

## 5. Validity Argument

The kernel is valid because:

- It is equivalent to an RBF kernel with diagonal covariance.
- RBF kernels with arbitrary positive diagonal scalings satisfy Mercer's theorem.
- Sums and products of PSD kernels remain PSD.

Thus, the anisotropic RBF is **a valid PSD kernel**.