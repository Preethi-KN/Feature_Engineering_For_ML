# 🔄 Feature Transformation Techniques in Machine Learning

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why Feature Transformation?](#-why-feature-transformation)
- [Types of Feature Transformation Techniques](#-types-of-feature-transformation-techniques)
- [1. Function Transformers](#1-function-transformers)
  - [Log Transform](#-log-transform)
  - [Square Transform](#-square-transform)
  - [Square Root Transform](#-square-root-transform)
  - [Reciprocal Transform](#-reciprocal-transform)
  - [Custom Transforms](#-custom-transforms)
- [2. Power Transformers](#2-power-transformers)
  - [Box-Cox Transform](#-box-cox-transform)
  - [Yeo-Johnson Transform](#-yeo-johnson-transform)
- [3. Quantile Transformers](#3-quantile-transformers)
- [Quick Comparison Table](#-quick-comparison-table)
- [Key Takeaways](#-key-takeaways)
- [When to Use Which?](#-when-to-use-which)
- [Prerequisites](#-prerequisites)

---

## Overview

Most machine learning algorithms are **statistics-dependent** — they assume the data follows a certain distribution, ideally a **normal (Gaussian) distribution**. However, real-world datasets rarely come pre-distributed this way.

**Feature Transformation** refers to the process of converting data from one form to another — preserving the essential information — to make it more suitable for ML algorithms.

```
Raw Data (skewed) ──► Feature Transformer ──► Normally Distributed Data ──► ML Model
```

---

## Why Feature Transformation?

- Many ML algorithms perform significantly better on **normally distributed** data
- Skewed data can cause models to underperform or produce biased results
- Transformations help reduce the effect of **outliers**
- They improve **convergence speed** in gradient-based algorithms
- Ensures features are on a **comparable scale**

---

## 📂 Types of Feature Transformation Techniques

There are **3 main categories** of feature transformation techniques:

```
Feature Transformation Techniques
│
├── 1. Function Transformers
│   ├── Log Transform
│   ├── Square Transform
│   ├── Square Root Transform
│   ├── Reciprocal Transform
│   └── Custom Transforms
│
├── 2. Power Transformers
│   ├── Box-Cox Transform
│   └── Yeo-Johnson Transform
│
└── 3. Quantile Transformers
```

---

## 1. Function Transformers

Function transformers apply a **specific mathematical function** to every observation in the data to bring it closer to a normal distribution.

> There is no single best function transformer — selection depends on **domain knowledge** and the **shape of your data distribution**.

---

### 📌 Log Transform

Applies the **logarithm** to each observation. Works exceptionally well on **right-skewed (positively skewed)** data.

**Formula:**
```
x_transformed = log(x + 1)    # log1p used to handle x = 0
```

**When to use:**
- Data is right-skewed
- Values span multiple orders of magnitude (e.g., income, population)
- All values are **positive**

**Python Implementation:**
```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

transform = FunctionTransformer(func=np.log1p)
transformed_data = transform.fit_transform(data)
```

> ⚠️ **Note:** Log of zero is undefined. Use `np.log1p` (which computes `log(1 + x)`) to handle zero values safely.

---

### 📌 Square Transform

Applies the **square function** to each observation.

**Formula:**
```
x_transformed = x²
```

**When to use:**
- Data is left-skewed
- You want to amplify larger values

**Python Implementation:**
```python
import numpy as np

transformed_data = np.square(data)
```

> ⚠️ **Note:** Squaring amplifies outliers — use with caution if your data has extreme values.

---

### 📌 Square Root Transform

Applies the **square root** to each observation. Works well on **left-skewed (negatively skewed)** data.

**Formula:**
```
x_transformed = √x
```

**When to use:**
- Data is left-skewed
- Values are non-negative (counts, frequencies)

**Python Implementation:**
```python
import numpy as np

transformed_data = np.sqrt(data)
```

> ⚠️ **Note:** Cannot be applied to negative values directly.

---

### 📌 Reciprocal Transform

Applies the **reciprocal (inverse)** of each observation.

**Formula:**
```
x_transformed = 1 / x
```

**When to use:**
- When the relationship between features and target is inversely proportional
- Useful in specific domain scenarios

**Python Implementation:**
```python
import numpy as np

transformed_data = np.reciprocal(data)
```

> ⚠️ **Note:** Cannot be applied when any value is **zero** (division by zero).

---

### 📌 Custom Transforms

When standard transforms don't work, you can create a **domain-specific custom transformation** using any mathematical function — sin, cos, tan, cube root, etc.

**Python Implementation:**
```python
import numpy as np

# Trigonometric transforms
sin_transformed_data = np.sin(data)
cos_transformed_data = np.cos(data)
tan_transformed_data = np.tan(data)

# Cube root (handles negatives)
cbrt_transformed_data = np.cbrt(data)
```

**Using FunctionTransformer with a custom function:**
```python
from sklearn.preprocessing import FunctionTransformer

def custom_transform(x):
    return np.cbrt(x)  # cube root example

transformer = FunctionTransformer(func=custom_transform)
transformed_data = transformer.fit_transform(data)
```

---

## 2. Power Transformers

Power transformers apply a **power function** to the data observations to stabilise variance and achieve a more Gaussian-like distribution. Scikit-learn provides the `PowerTransformer` class for both methods below.

---

### 📌 Box-Cox Transform

A parametric transform where power **lambda (λ)** is applied to each observation. The optimal λ is found through iteration.

**Mathematical Formula:**

```
         ┌  ln(Xᵢ)           if λ = 0
X'ᵢ  =   │
         └  (Xᵢ^λ - 1) / λ  if λ ≠ 0
```

**Key Properties:**
- Transformed values lie between **-5 and +5**
- Lambda (λ) is determined automatically via **maximum likelihood estimation**
- **Only works on strictly positive values** (all x > 0)

**Python Implementation:**
```python
from sklearn.preprocessing import PowerTransformer

boxcox = PowerTransformer(method='box-cox')
data_transformed = boxcox.fit_transform(data)
```

**Limitation:**
> ❌ Cannot be applied to **zero or negative** data values.

---

### 📌 Yeo-Johnson Transform

An **advanced extension of Box-Cox** that handles zero and negative values. This is the **default method** in scikit-learn's `PowerTransformer`.

**Mathematical Formula:**

```
          ┌  ((y + 1)^λ - 1) / λ          for y ≥ 0 and λ ≠ 0
          │  log(y + 1)                    for y ≥ 0 and λ = 0
X'ᵢ  =   │
          │  -((1 - y)^(2-λ) - 1)/(2-λ)  for y < 0 and λ ≠ 2
          └  -log(1 - y)                  for y < 0 and λ = 2
```

**Key Properties:**
- Works on **positive, negative, and zero** values
- More flexible than Box-Cox
- Default method in scikit-learn's `PowerTransformer`

**Python Implementation:**
```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson is the default method
yeo_johnson = PowerTransformer()  # method='yeo-johnson' by default
data_transformed = yeo_johnson.fit_transform(data)

# Or explicitly:
yeo_johnson = PowerTransformer(method='yeo-johnson')
data_transformed = yeo_johnson.fit_transform(data)
```

**Advantage over Box-Cox:**
> ✅ Can handle **zero and negative values** — making it applicable to a wider range of datasets.

---

## 3. Quantile Transformers

Quantile transformation is a **non-parametric** transformation technique that maps the data to a specified output distribution (uniform or normal) based on quantile information.

**How it works:**
1. Computes the quantile distribution of the input data
2. Maps each value to the corresponding quantile of the target distribution
3. Output is either **uniform** or **normal** distribution

**Key Properties:**
- Works on **any numerical data** (positive, negative, zero)
- Robust to outliers (maps extreme values to the edge of the target distribution)
- Non-linear transformation

**Python Implementation:**
```python
from sklearn.preprocessing import QuantileTransformer

# Output as Normal Distribution
quantile_trans = QuantileTransformer(output_distribution='normal')
data_transformed = quantile_trans.fit_transform(data)

# Output as Uniform Distribution
quantile_trans_uniform = QuantileTransformer(output_distribution='uniform')
data_transformed_uniform = quantile_trans_uniform.fit_transform(data)
```

**Parameters:**

| Parameter | Values | Description |
|---|---|---|
| `output_distribution` | `'normal'`, `'uniform'` | Target distribution for transformed data |
| `n_quantiles` | int (default=1000) | Number of quantiles computed |
| `random_state` | int | For reproducibility |

> ⚠️ **Note:** For small datasets, the quantile estimate may be unstable. Use a sufficiently large dataset for reliable results.

---

## 📊 Quick Comparison Table

| Technique | Handles Negatives | Handles Zeros | Best For | Scikit-learn Class |
|---|---|---|---|---|
| **Log Transform** | ❌ | ⚠️ (use log1p) | Right-skewed data | `FunctionTransformer(np.log1p)` |
| **Square Transform** | ✅ | ✅ | Left-skewed data | `FunctionTransformer(np.square)` |
| **Square Root Transform** | ❌ | ✅ | Left-skewed, count data | `FunctionTransformer(np.sqrt)` |
| **Reciprocal Transform** | ✅ | ❌ | Inverse relationships | `FunctionTransformer(np.reciprocal)` |
| **Custom Transform** | Depends | Depends | Domain-specific patterns | `FunctionTransformer(custom_fn)` |
| **Box-Cox** | ❌ | ❌ | Strictly positive data | `PowerTransformer(method='box-cox')` |
| **Yeo-Johnson** | ✅ | ✅ | Any numeric data | `PowerTransformer()` |
| **Quantile Transform** | ✅ | ✅ | Any numeric data | `QuantileTransformer()` |

---

## 🔑 Key Takeaways

- ✅ Feature transformation converts data to a **normal distribution** for better algorithm performance
- 📈 **Log transform** works best on **right-skewed** data
- 📉 **Square root transform** works best on **left-skewed** data
- 🔧 **Custom transforms** can be built using domain knowledge when standard ones fail
- ❗ **Box-Cox** only applies to **strictly positive** values (output range: -5 to 5)
- ✅ **Yeo-Johnson** extends Box-Cox to also handle **zero and negative** values
- 🎯 **Quantile Transformer** is the most flexible — works on any numerical data and is robust to outliers

---

## 🗺️ When to Use Which?

```
Is your data right-skewed?
    └─ Yes ──► Try Log Transform first
    └─ No  ──► Is your data left-skewed?
                    └─ Yes ──► Try Square Root Transform
                    └─ No  ──► Does data have only positive values?
                                    └─ Yes ──► Box-Cox Transform
                                    └─ No  ──► Yeo-Johnson Transform
                                               OR Quantile Transformer
```

**General rule of thumb:**
- Start simple: try **Log** or **Square Root** first
- If those fail: try **Box-Cox** (positive data) or **Yeo-Johnson** (any data)
- For maximum flexibility with robust outlier handling: use **Quantile Transformer**
- When you have domain knowledge: build a **Custom Transform**

---

## 🛠️ Prerequisites

To run the code examples in this guide, you'll need:

```bash
pip install numpy scikit-learn pandas matplotlib seaborn
```

**Imports used throughout:**
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer
)
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 📚 Further Reading

- [Scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [NumPy Mathematical Functions](https://numpy.org/doc/stable/reference/routines.math.html)
- [Normal Distribution — GeeksforGeeks](https://www.geeksforgeeks.org/data-science/mathematics-probability-distributions-set-3-normal-distribution/)
- [Box-Cox Transformation using Python](https://www.geeksforgeeks.org/python/box-cox-transformation-using-python/)
- [Feature Engineering Overview](https://www.geeksforgeeks.org/machine-learning/what-is-feature-engineering/)

---
