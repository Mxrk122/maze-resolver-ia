import numpy as np

def create_polynomial_features(X, degree):
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))
    
    for d in range(1, degree+1):
        for feat in range(n_features):
            X_poly = np.hstack((X_poly, np.power(X[:, feat:feat+1], d)))
            
    return X_poly
