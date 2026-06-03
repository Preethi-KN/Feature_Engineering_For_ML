# Principal Component Analysis (PCA) - Summary

## Overview
Principal Component Analysis (PCA) is an unsupervised machine learning technique used for dimensionality reduction. It transforms correlated features into a smaller set of uncorrelated variables called principal components while preserving most of the dataset's variance.

## Why Use PCA?
- Reduce feature dimensions
- Remove redundancy
- Improve computational efficiency
- Simplify visualization
- Reduce noise

## Core Concepts

### Principal Components
- PC1 captures the maximum variance.
- PC2 captures the next highest variance and is orthogonal to PC1.
- Subsequent components capture decreasing variance.

### Eigenvectors and Eigenvalues
- Eigenvectors define component directions.
- Eigenvalues measure component importance.
- Higher eigenvalues indicate more explained variance.

## PCA Steps
1. Standardize the data.
2. Compute the covariance matrix.
3. Calculate eigenvalues and eigenvectors.
4. Select top principal components.
5. Project data onto the selected components.

## Advantages
- Handles multicollinearity
- Reduces dimensionality
- Improves model training speed
- Enables visualization of high-dimensional data
- Removes noise

## Limitations
- Components are harder to interpret
- Some information may be lost
- Sensitive to scaling
- Assumes linear relationships

## Common Applications
- Feature engineering
- Data visualization
- Image compression
- Finance and risk analysis
- Bioinformatics

## Example (Scikit-Learn)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
```

## Best Practices
- Standardize features before PCA.
- Use explained variance ratio to choose components.
- Validate model performance before and after PCA.
- Retain enough components to preserve useful information.

## Conclusion
PCA is a powerful dimensionality reduction technique that preserves the most important patterns in data while reducing complexity, making datasets easier to analyze, visualize, and model.

Reference:
https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/
