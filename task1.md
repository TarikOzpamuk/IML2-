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

We did a grid search for each kernel:

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

# Subtask 2

## 1. Motivation and Idea Behind the Kernel

The digits dataset consists of **8×8 grayscale images**. Each image has a spatial structure: central pixels are usually more informative while outer pixels contain mostly background (black).

A standard RBF kernel treats all features equally. To exploit the spatial structure, we designed an **anisotropic RBF kernel**, allowing different features to contribute differently to the distance measure.

The kernel modifies the RBF by applying a weight vector (A):

$$K(x,y) = \exp\left(-\sum_i A_i (x_i - y_i)^2\right)$$

- Mid-range PCA components receive *higher weight*.
- Others receive a smaller weight.

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

# Subtask 3
## 1. How the Confusion Matrix Is Built

We manually compute the matrix as follows:

1. Create a zero matrix of size `K x K`, with `K = 10` classes.
2. For each test sample, increase the entry at `(true_label, predicted_label)` by one.

Rows correspond to the true class; columns correspond to the predicted class.

---

## 2. Results

### 2.1 Best SVM with standard RBF kernel

**Confusion matrix**

```
[[53  0  0  0  0  0  0  0  0  0]
 [ 0 50  0  0  0  0  0  0  0  0]
 [ 0  0 47  0  0  0  0  0  0  0]
 [ 0  0  0 52  0  1  0  0  1  0]
 [ 0  0  0  0 60  0  0  0  0  0]
 [ 0  0  0  0  0 66  0  0  0  0]
 [ 0  0  0  0  0  0 53  0  0  0]
 [ 0  0  0  0  0  0  0 54  0  1]
 [ 0  0  0  0  0  0  0  0 43  0]
 [ 0  0  0  1  0  0  0  1  0 57]]
```

- Test samples: `540`
- Correct predictions (diagonal sum): `535`
- Accuracy: `0.9907` (99.07%)

### 2.2 Best SVM with custom anisotropic RBF kernel

**Confusion matrix**

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

- Test samples: `540`
- Correct predictions: `536`
- Accuracy: `0.9926` (99.26%)

The anisotropic kernel correctly classifies one additional digit compared to the tuned RBF model.

---

## 3. Takeaways

- Both matrices are almost perfectly diagonal, meaning nearly all digits are classified correctly.
- The standard RBF kernel misclassifies digit `3` twice (as `5` and `8`); the anisotropic version fixes the `3 → 8` mistake.
- Each model mislabels one digit `7` as `9`, and digit `9` is occasionally predicted as `3` or `7`.
- Aside from the single recovered error, their error patterns are nearly identical.