import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import product

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
default_scores = {}

for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    default_scores[kernel] = acc
    print(f"Kernel: {kernel} | Test Accuracy (default params): {acc:.4f}")


param_grids = {
    'linear': {
        'C': [0.1, 1, 10, 100]
    },
    'poly': {
        'C': [0.1, 1, 10],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    },
    'rbf': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    },
    'sigmoid': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
}


def create_kfold_splits(X, y, n_splits=5, shuffle=True, random_state=42):
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    fold_size = n_samples // n_splits
    splits = []
    
    for i in range(n_splits):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        splits.append((train_indices, val_indices))
    
    return splits


def grid_search_with_cv(X_train, y_train, kernel, param_grid, n_splits=5):
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_score = -1
    best_params = None
    best_model = None
    
    splits = create_kfold_splits(X_train, y_train, n_splits=n_splits)
    
    print(f"  Testing {len(param_combinations)} parameter combinations...")
    
    for param_combo in param_combinations:
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))
        
        # Perform k-fold cross-validation
        cv_scores = []
        
        for train_idx, val_idx in splits:
            # Split data for this fold
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Create and train model
            svm = SVC(kernel=kernel, **params)
            svm.fit(X_fold_train, y_fold_train)
            
            # Evaluate on validation set
            y_pred = svm.predict(X_fold_val)
            score = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(score)
        
        # Calculate average CV score
        avg_score = np.mean(cv_scores)
        
        # Update best if this is better
        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()
            # Retrain on full training set with best params
            best_model = SVC(kernel=kernel, **best_params)
            best_model.fit(X_train, y_train)
    
    return best_params, best_score, best_model

best_models = {}
results = {}

print("Performing k-fold cross-validation for hyperparameter tuning...\n")

for kernel in kernels:
    print(f"Tuning {kernel} kernel...")
    
    # Perform grid search with k-fold cross-validation manually
    best_params, cv_score, best_model = grid_search_with_cv(
        X_train, y_train, kernel, param_grids[kernel], n_splits=5
    )
    
    best_models[kernel] = best_model
    
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    results[kernel] = {
        'best_params': best_params,
        'cv_score': cv_score,
        'test_accuracy': test_accuracy
    }
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {cv_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}\n")

# TASK 1.1 Report
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
for kernel, result in results.items():
    print(f"\n{kernel.upper()} Kernel:")
    print(f"  Best parameters: {result['best_params']}")
    print(f"  Cross-validation score: {result['cv_score']:.4f}")
    print(f"  Test set accuracy: {result['test_accuracy']:.4f}")
    print(f"  Default params test accuracy: {default_scores[kernel]:.4f}")
    improvement = result['test_accuracy'] - default_scores[kernel]
    print(f"  Improvement over default: {improvement:+.4f}")


