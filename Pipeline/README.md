# Machine Learning Pipeline with Scikit-Learn

This repository demonstrates the implementation of an end-to-end Machine Learning workflow using Scikit-Learn's `Pipeline` class. By encapsulating data preprocessing, feature transformation, and model estimation into a single object, this project ensures reproducibility, eliminates data leakage, and provides a clean interface for production deployment.

---

## Overview

In traditional Machine Learning workflows, applying data preprocessing step-by-step (e.g., handling scaling, text encoding, and PCA separately) on both the training and testing sets is error-prone. This repository showcases how to use `sklearn.pipeline.Pipeline` to treat an entire ML sequence as a single unified component.

The sequence implemented follows a strict layout:
1. **Transformers (`fit_transform` steps)**: Arbitrary number of data clearing/modifying steps (e.g., `StandardScaler`, `OneHotEncoder`, `PCA`).
2. **Estimator (`fit`/`predict` step)**: The final algorithm used to make predictions (e.g., `LogisticRegression`, `SVM`).

---

## Why Use `sklearn.pipeline.Pipeline`?

Based on industry best practices highlighted by standard implementations, utilizing pipelines provides several key operational benefits:

* **Prevents Data Leakage:** Ensures that statistical transformations (such as scaling parameters like mean and variance) are calculated *only* on the training split, and then correctly applied (rather than recalculated) onto the testing split.
* **Code Readability and Maintenance:** Replaces scattered preprocessing steps with a clean, sequential array of named tuples.
* **Modularity:** Easily switch models or adjust preprocessing parameters without breaking the downstream application architecture.
* **Seamless Hyperparameter Tuning:** Allows utilizing meta-estimators like `GridSearchCV` or `RandomizedSearchCV` across both preprocessing options and algorithm parameters simultaneously.

