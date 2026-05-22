# ⚙️ Scaling, Normalization & Standardization

---

## Scaling Techniques
---

## Requirements

```bash
pip install pandas numpy scikit-learn
```

---

## Dataset

Download the sample dataset used in the examples: [Housing.csv](https://media.geeksforgeeks.org/wp-content/uploads/20250114174407596134/SampleFile.csv)

```python
import pandas as pd
import numpy as np

df = pd.read_csv('Housing.csv')
df = df.select_dtypes(include=np.number)
df.head()
```

---

### 1. 📐 Absolute Maximum Scaling

Rescales each feature by dividing all values by the **maximum absolute value** of that feature, mapping values to the range **[-1, 1]**.

**Formula:**

```
X_scaled = X_i / max(|X|)
```

**Characteristics:**
- Output range: -1 to 1
- Sensitive to outliers (can skew the max absolute value)
- Suitable for sparse data

**Code:**

```python
max_abs = np.max(np.abs(df), axis=0)
scaled_df = df / max_abs
scaled_df.head()
```

---

### 2. 📏 Min-Max Scaling

Transforms features by mapping values to a fixed range, typically **[0, 1]**, using the minimum and maximum values of each feature.

**Formula:**

```
X_scaled = (X_i - X_min) / (X_max - X_min)
```

**Characteristics:**
- Output range: 0 to 1
- Sensitive to outliers (min/max can be skewed)
- Suitable for neural networks and bounded input features

**Code:**

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
scaled_df.head()
```

---

### 3. 🔄 Normalization (Vector Normalization)

Scales each **data sample (row)** so that its vector length (Euclidean norm) equals 1. Focuses on the **direction** of data points rather than their magnitude.

**Formula:**

```
X_scaled = X_i / ||X||
```

Where `||X||` is the Euclidean norm (L2 norm) of the row vector.

**Characteristics:**
- Normalizes each sample to unit length
- Not applicable to outlier sensitivity (operates per row, not per column)
- Suitable for direction-based similarity metrics (e.g., cosine similarity, text classification, clustering)

**Code:**

```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
scaled_df.head()
```

---

### 4. 📊 Standardization (Z-Score Scaling)

Centers features by subtracting the **mean** and scales by dividing by the **standard deviation**, producing features with zero mean and unit variance.

**Formula:**

```
X_scaled = (X_i - μ) / σ
```

Where `μ` = mean, `σ` = standard deviation.

**Characteristics:**
- Output: mean = 0, variance = 1
- Moderate sensitivity to outliers
- Assumes approximately normal distribution
- Suitable for linear regression, logistic regression, SVMs, and neural networks

**Code:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print(scaled_df.head())
```

---

### 5. 🛡️ Robust Scaling

Uses the **median** and **Interquartile Range (IQR)** instead of mean and standard deviation, making it resistant to outliers and skewed distributions.

**Formula:**

```
X_scaled = (X_i - X_median) / IQR
```

**Characteristics:**
- Low sensitivity to outliers
- Centers data around the median
- Scales based on the spread of the middle 50% of data (IQR)
- Suitable for datasets with extreme values or noise

**Code:**

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print(scaled_df.head())
```

---

## 📊 Comparison Table

| Technique | Formula Basis | Output Range | Outlier Sensitivity | Best Use Case |
|---|---|---|---|---|
| Absolute Maximum Scaling | Max absolute value | [-1, 1] | High | Sparse data, simple scaling |
| Min-Max Scaling | Min & Max | [0, 1] | High | Neural networks, bounded features |
| Normalization (Vector) | Euclidean norm (per row) | Unit norm | Not applicable | Cosine similarity, text classification |
| Standardization (Z-Score) | Mean & Std deviation | Mean=0, Var=1 | Moderate | Most ML algorithms, normal data |
| Robust Scaling | Median & IQR | Unbounded | Low | Data with outliers, skewed distributions |

---

## ✅ Advantages of Feature Scaling

- **Improves Model Performance** — Presents features at comparable scales, enhancing accuracy
- **Speeds Up Convergence** — Helps gradient-based algorithms (e.g., SGD) train faster
- **Prevents Feature Bias** — Avoids dominance by large-scale features
- **Increases Numerical Stability** — Reduces risks of overflow or underflow in computations
- **Algorithm Compatibility** — Makes data suitable for distance- and gradient-based models like SVM, KNN, and neural networks

---

## 🧭 Choosing the Right Technique

| Situation | Recommended Technique |
|---|---|
| Data has no outliers, bounded range needed | Min-Max Scaling |
| Data is approximately normally distributed | Standardization |
| Data has significant outliers | Robust Scaling |
| Measuring similarity between samples | Normalization (Vector) |
| Sparse data or simple use case | Absolute Maximum Scaling |

---

## 📚 Related Topics

- [Exploratory Data Analysis (EDA)](https://www.geeksforgeeks.org/data-analysis/what-is-exploratory-data-analysis/)
- [Feature Selection Techniques](https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/)
- [Dimensionality Reduction](https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/)
- [Handling Outliers in Data](https://www.geeksforgeeks.org/machine-learning/what-are-outliers-in-data/)

---
