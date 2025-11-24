import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from itertools import product

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

def anisotropic_rbf(A):
    def kernel(X, Y):
        diffs = X[:, None, :] - Y[None, :, :]
        quad = np.einsum('...i,i,...i->...', diffs, A, diffs)
        return np.exp(-quad)
    return kernel

n_features = X_train.shape[1]
gamma = 0.001

# Function to generate A for a given number of dimensions
def generate_A(n_dims, gamma=gamma):
    """Generate A vector for anisotropic RBF kernel with center emphasis"""
    A = np.ones(n_dims) * gamma
    center_start = 2 * n_dims // 5
    center_end = 3 * n_dims // 5
    A[center_start:center_end] = 2 * gamma
    return A

param_grid = {
    'pca_components': [41, 42, 43, 44, 45, 46, 47, 48, 49],
    'C': [1, 10, 100],
    'gamma': [0.001 , 0.01, 0.1]
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


def grid_search_with_cv(X_train, y_train, A_generator, param_grid, n_splits=5):
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

            # PCA
            pca = PCA(n_components=params['pca_components'])
            X_fold_train = pca.fit_transform(X_fold_train)
            X_fold_val = pca.transform(X_fold_val)
            
            # Create kernel with A matching PCA dimensions
            n_pca_components = params['pca_components']
            A_pca = A_generator(n_pca_components)
            kernel = anisotropic_rbf(A_pca)

            # Separate SVC parameters from preprocessing parameters
            svc_params = {k: v for k, v in params.items() if k != 'pca_components'}
            
            # Create and train model
            svm = SVC(kernel=kernel, **svc_params)
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
    pca = PCA(n_components=best_params['pca_components'])
    X_train_pca = pca.fit_transform(X_train)
    
    # Create kernel with A matching PCA dimensions
    n_pca_components = best_params['pca_components']
    A_pca = A_generator(n_pca_components)
    kernel = anisotropic_rbf(A_pca)
    
    # Separate SVC parameters
    svc_params = {k: v for k, v in best_params.items() if k != 'pca_components'}
    best_model = SVC(kernel=kernel, **svc_params)
    best_model.fit(X_train_pca, y_train)
    
    # Return model, preprocessing objects, and results
    return best_params, best_score, best_model, pca


def custom_confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix without sklearn helper."""
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[label_to_index[truth], label_to_index[pred]] += 1
    return matrix, labels

best_models = {}
results = {}

print("Performing k-fold cross-validation for hyperparameter tuning...\n")

print(f"Tuning anisotropic RBF kernel...")
    
best_params, cv_score, best_model, pca = grid_search_with_cv(
    X_train, y_train, generate_A, param_grid, n_splits=5
)

# Apply preprocessing to test set
X_test_pca = pca.transform(X_test)
y_pred_test = best_model.predict(X_test_pca)
    
results['anisotropic RBF'] = {
    'best_params': best_params,
    'cv_score': cv_score,
    'test_accuracy': accuracy_score(y_test, y_pred_test)
}
    
# Confusion matrix
cm, labels = custom_confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(cm)

# TASK 1.2 Report
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
for kernel, result in results.items():
    print(f"\n{kernel.upper()} Kernel:")
    print(f"  Best parameters: {result['best_params']}")
    print(f"  Cross-validation score: {result['cv_score']:.4f}")
    print(f"  Test set accuracy: {result['test_accuracy']:.4f}")


