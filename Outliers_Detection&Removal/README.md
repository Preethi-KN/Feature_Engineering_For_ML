# 🎯 Outlier Detection and Removal in Machine Learning

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What is an Outlier?](#-what-is-an-outlier)
- [Why Outliers Matter in ML](#-why-outliers-matter-in-ml)
- [Sources of Outliers](#-sources-of-outliers)
- [Types of Outliers](#-types-of-outliers)
  - [Global Outliers (Point Anomalies)](#1-global-outliers-point-anomalies)
  - [Contextual Outliers](#2-contextual-outliers)
  - [Collective Outliers](#3-collective-outliers)
- [Dataset Setup](#-dataset-setup)
- [Step 1 — Visualising Outliers](#-step-1--visualising-outliers)
- [Detection Methods](#-detection-methods)
  - [Method 1 — Z-Score](#-method-1--z-score)
  - [Method 2 — IQR (Interquartile Range)](#-method-2--iqr-interquartile-range)
  - [Method 3 — Isolation Forest](#-method-3--isolation-forest)
  - [Method 4 — Local Outlier Factor (LOF)](#-method-4--local-outlier-factor-lof)
- [Outlier Removal Strategies](#-outlier-removal-strategies)
  - [Strategy 1 — Deletion](#1-deletion)
  - [Strategy 2 — Capping (Winsorisation)](#2-capping-winsorisation)
  - [Strategy 3 — Transformation](#3-transformation)
  - [Strategy 4 — Imputation](#4-imputation)
  - [Strategy 5 — Use Robust Algorithms](#5-use-robust-algorithms)
- [Full Pipeline — Detect & Remove](#-full-pipeline--detect--remove)
- [Comparison of Detection Techniques](#-comparison-of-detection-techniques)
- [When to Remove vs. Keep Outliers](#-when-to-remove-vs-keep-outliers)
- [Decision Guide](#-decision-guide)
- [Key Takeaways](#-key-takeaways)
- [Prerequisites](#-prerequisites)
- [Further Reading](#-further-reading)

---

## 🧭 Overview

In machine learning, **outliers** are data points that deviate significantly from the general distribution of the dataset. They may occur due to errors in data collection, natural variation, or rare events.

While sometimes they carry valuable signal — such as in **fraud detection** or **anomaly monitoring** — in many cases they negatively affect model accuracy, skew statistical summaries, and bias parameter estimation.

```
Dataset with Outliers
        │
        ▼
┌───────────────────┐
│  Visualise Data   │  ← Boxplot, scatter plot, histogram
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Detect Outliers  │  ← Z-Score / IQR / Isolation Forest / LOF
└────────┬──────────┘
         │
         ▼
┌─────────────────────────┐
│  Decide: Remove or Keep? │  ← Domain knowledge + data type
└────────┬────────────────┘
         │
         ▼
┌───────────────────────────────────┐
│  Handle: Delete / Cap / Transform │
└────────┬──────────────────────────┘
         │
         ▼
   Clean Dataset ──► ML Model
```

---

## 🔍 What is an Outlier?

An outlier is a data point that lies **abnormally far from the rest** of the dataset — it doesn't conform to the expected pattern or distribution.

```
Normal data:  [10, 12, 11, 13, 10, 12, 14, 11, 13, 12]
               ──── all cluster between 10–14 ────

With outlier: [10, 12, 11, 13, 10, 12, 14, 11, 13, 12, 95]
                                                          ▲
                                                     outlier!
```

---

## ⚠️ Why Outliers Matter in ML

| Impact | Description |
|---|---|
| **Bias in parameter estimation** | Linear regression coefficients are pulled toward outliers |
| **Increased variance** | Models become less stable and generalise poorly |
| **Reduced accuracy** | Predictions distorted by extreme training examples |
| **Misleading statistics** | Mean, standard deviation, and correlation all shift |
| **Algorithm sensitivity** | KNN, K-Means, SVM, and Linear Models are highly sensitive to outliers |
| **Broken normality assumptions** | Many algorithms assume Gaussian distributions — outliers violate this |

**Visual example — effect on linear regression:**
```
Without outlier:   y = 2x + 1   ← clean fit
With outlier:      y = 2.8x - 5 ← line pulled toward the extreme point
```

---

## 🔎 Sources of Outliers

```
Outliers arise from:
    │
    ├── Data Entry Errors
    │       └── Typos, unit mismatches (kg entered as grams), copy-paste mistakes
    │
    ├── Measurement Noise
    │       └── Sensor failures, faulty instruments, signal interference
    │
    ├── Genuine Rare Events
    │       └── Stock market crashes, extreme weather, record-breaking performances
    │
    ├── Data Processing Bugs
    │       └── ETL pipeline errors, join mismatches, encoding issues
    │
    └── Sampling Bias
            └── Non-representative sample captures edge cases disproportionately
```

> 💡 **Key insight:** Always determine *why* an outlier exists before deciding to remove it.
> A genuine rare event (fraud, equipment failure) is valuable signal — a data entry error is noise.

---

## 📂 Types of Outliers

---

### 1. Global Outliers (Point Anomalies)

Individual data points that lie **far from the entire dataset's distribution**.

```
Example — Wine alcohol content:
  Normal range:  10% – 15%
  Global outlier: 20%  ← far from all other values, in any context
```

```
Dataset:  [11.2, 12.1, 10.8, 13.4, 12.9, 11.5, 20.1]
                                                  ▲
                                          global outlier
```

**Detection:** Z-Score, IQR — both work well for global outliers.

---

### 2. Contextual Outliers

Data points that are **outliers only in a specific context or condition** — but would be normal in another context.

```
Example — Daily temperature:
  30°C in July (summer)    → completely normal ✅
  30°C in January (winter) → contextual outlier ⚠️

Example — Transaction amount:
  $10,000 transfer from a business account → normal ✅
  $10,000 transfer from a student account  → contextual outlier ⚠️
```

**Detection:** Requires domain knowledge + conditional analysis; simple statistical methods are insufficient.

---

### 3. Collective Outliers

A **group of data points** that are individually unremarkable but anomalous when considered together.

```
Example — Wine dataset:
  Individual values of residual sugar and acidity may seem fine.
  But a sudden simultaneous spike in BOTH across multiple samples
  suggests a batch contamination event → collective outlier.

Example — Network traffic:
  Individual packets look normal.
  But 10,000 packets per second from one IP = DDoS attack → collective outlier.
```

**Detection:** Requires multivariate methods like Isolation Forest or LOF.

---

### Types Summary Table

| Type | Scope | Example | Best Detection Method |
|---|---|---|---|
| **Global** | Single point, any context | Blood pressure of 250 mmHg | Z-Score, IQR |
| **Contextual** | Single point, specific context | 35°C in winter | Conditional analysis, domain rules |
| **Collective** | Group of points together | Batch of anomalous sensor readings | Isolation Forest, LOF, clustering |

---

## 🗂️ Dataset Setup

All code examples use the **Wine Quality Dataset** (Red Wine). Download it from the reference link or use the setup below.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# Load dataset
df = pd.read_csv("winequality-red.csv")
print("Shape:", df.shape)
print(df.head())

# Features only (drop target)
data = df.drop("quality", axis=1)
print("\nColumns:", list(data.columns))
```

**Dataset overview:**

| Feature | Description | Type |
|---|---|---|
| `fixed acidity` | Tartaric acid content | Numeric |
| `volatile acidity` | Acetic acid content | Numeric |
| `citric acid` | Citric acid content | Numeric |
| `residual sugar` | Remaining sugar after fermentation | Numeric |
| `chlorides` | Salt content | Numeric |
| `free sulfur dioxide` | Free SO₂ | Numeric |
| `total sulfur dioxide` | Total SO₂ | Numeric |
| `density` | Density of wine | Numeric |
| `pH` | pH level | Numeric |
| `sulphates` | Sulphate content | Numeric |
| `alcohol` | Alcohol percentage | Numeric |
| `quality` | Target: quality score (3–8) | Integer |

---

## 📊 Step 1 — Visualising Outliers

Always **visualise your data first** before applying detection algorithms. Boxplots are the most common starting point.

### Boxplot — All Features

```python
plt.figure(figsize=(14, 7))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Boxplots of Wine Features — Black Dots = Outliers")
plt.tight_layout()
plt.show()
```

> 📌 **Reading a boxplot:**
> ```
>          ┌──────────┐
>  ───●────┤  IQR box ├────────●●    ← black dots = outliers
>          └──────────┘
>    ▲        ▲      ▲              ▲
>  min/      Q1     Q3           max/
> whisker                       whisker
>
>  Whiskers extend to Q1 - 1.5×IQR and Q3 + 1.5×IQR
>  Points beyond whiskers = potential outliers
> ```

### Histogram with KDE — Single Feature

```python
plt.figure(figsize=(10, 4))
sns.histplot(data['alcohol'], kde=True, color='steelblue')
plt.title("Distribution of Alcohol Content")
plt.xlabel("Alcohol %")
plt.show()
```

### Scatter Plot — Two Features

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(x='alcohol', y='residual sugar', data=df)
plt.title("Alcohol vs Residual Sugar")
plt.show()
```

---

## 🔬 Detection Methods

---

## 📌 Method 1 — Z-Score

A **statistical technique** that detects outliers based on how far a data point is from the mean, measured in standard deviations. Assumes the data follows a **normal distribution**.

### Formula

```
        x - μ
Z  =  ─────────
          σ

Where:
  Z  = Z-score (standard score)
  x  = individual data value
  μ  = mean of the dataset
  σ  = standard deviation of the dataset

Rule: |Z| > 3  →  outlier
```

### How It Works

```
Bell curve:
         │
    ░░░░░│░░░░░
  ░░     │     ░░
░░       │       ░░
──────────────────────
   -3σ  -2σ  μ  +2σ  +3σ

Points beyond ±3σ (the extreme tails) are flagged as outliers.
99.7% of data falls within ±3σ for a normal distribution.
```

### Implementation

```python
from scipy import stats
import numpy as np

# Compute Z-scores for all columns
z_scores = np.abs(stats.zscore(data))

# Flag outliers: |Z| > 3
outliers_z = np.where(z_scores > 3)

print("Outlier positions (row, col):")
print(list(zip(outliers_z[0][:10], outliers_z[1][:10])))

# Count outliers per column
outlier_counts_z = (z_scores > 3).sum(axis=0)
print("\nOutlier count per column:")
for col, count in zip(data.columns, outlier_counts_z):
    if count > 0:
        print(f"  {col}: {count} outliers")

# Get only clean rows (no outlier in any column)
mask_z = (z_scores < 3).all(axis=1)
data_clean_z = data[mask_z]
print(f"\nRows before: {len(data)}")
print(f"Rows after Z-score filter: {len(data_clean_z)}")
```

### Visualise Z-Score Outliers

```python
# Single column example
col = 'total sulfur dioxide'
z_col = np.abs(stats.zscore(data[col]))
outlier_mask = z_col > 3

plt.figure(figsize=(10, 4))
plt.plot(data[col].values, color='steelblue', label='Normal')
plt.scatter(np.where(outlier_mask)[0], data[col][outlier_mask],
            color='red', zorder=5, label='Outlier (Z > 3)')
plt.title(f"Z-Score Outliers — {col}")
plt.legend()
plt.show()
```

| ✅ Pros | ❌ Cons |
|---|---|
| Simple, fast, easy to implement | Unreliable for non-normal / skewed data |
| Works well on Gaussian distributions | Sensitive to the outliers it's trying to detect (mean/std shift) |
| Interpretable threshold (|Z| > 3) | Not suitable for small datasets (< 30 samples) |
| Detects global outliers effectively | Assumes univariate normality per column |

> 🔧 **Threshold tuning:** Use `|Z| > 2` for stricter detection (more flagged) or `|Z| > 3.5` for looser detection (fewer flagged). Default is **3**.

---

## 📌 Method 2 — IQR (Interquartile Range)

A **robust statistical method** that identifies outliers by examining the spread of the **middle 50% of the data**. Doesn't assume any distribution — works well on skewed data.

### Formula

```
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Any value < Lower Bound  OR  > Upper Bound  →  outlier
```

### How It Works

```
Sorted data:
  [10, 11, 12, 13, 14, 15, 16, 17, 18, 95]
        ▲                   ▲
       Q1                  Q3

IQR = Q3 - Q1 = 16.75 - 11.75 = 5.0

Lower Bound = 11.75 - 1.5 × 5.0 = 4.25
Upper Bound = 16.75 + 1.5 × 5.0 = 24.25

Value 95 → above upper bound = OUTLIER ⚠️
```

### Implementation

```python
# Compute Q1, Q3, IQR for all columns
Q1  = data.quantile(0.25)
Q3  = data.quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Boolean mask of outliers
outliers_iqr = ((data < lower_bound) | (data > upper_bound))

print("Outlier count per column (IQR method):")
print(outliers_iqr.sum())

# Remove rows with any outlier
mask_iqr = ~outliers_iqr.any(axis=1)
data_clean_iqr = data[mask_iqr]
print(f"\nRows before: {len(data)}")
print(f"Rows after IQR filter: {len(data_clean_iqr)}")
```

### Visualise IQR Bounds

```python
col = 'residual sugar'

Q1_col  = data[col].quantile(0.25)
Q3_col  = data[col].quantile(0.75)
IQR_col = Q3_col - Q1_col
lb = Q1_col - 1.5 * IQR_col
ub = Q3_col + 1.5 * IQR_col

plt.figure(figsize=(10, 4))
plt.plot(data[col].values, color='steelblue', label='Data')
plt.axhline(ub, color='red',    linestyle='--', label=f'Upper bound ({ub:.2f})')
plt.axhline(lb, color='orange', linestyle='--', label=f'Lower bound ({lb:.2f})')
outliers = data[col][(data[col] < lb) | (data[col] > ub)]
plt.scatter(outliers.index, outliers.values, color='red', zorder=5, label='Outliers')
plt.title(f"IQR Outlier Detection — {col}")
plt.legend()
plt.show()
```

### Adjusting the IQR Multiplier

```python
# Stricter: flag more outliers
outliers_strict = ((data < Q1 - 1.0 * IQR) | (data > Q3 + 1.0 * IQR))

# More lenient: flag fewer outliers
outliers_lenient = ((data < Q1 - 2.0 * IQR) | (data > Q3 + 2.0 * IQR))
```

| ✅ Pros | ❌ Cons |
|---|---|
| Robust — not distorted by extreme values | Less sensitive for extremely skewed data |
| Non-parametric — no distribution assumption | May flag too many outliers in heavy-tailed data |
| Simple and interpretable | Univariate — doesn't capture multivariate outliers |
| Directly maps to boxplot visualisation | Fixed 1.5 multiplier may not suit all domains |

---

## 📌 Method 3 — Isolation Forest

A **model-based anomaly detection algorithm** that isolates outliers instead of profiling normal data. Builds multiple random decision trees — outliers are isolated in fewer splits because they are rare and different.

### How It Works

```
Step 1: Randomly select a feature
Step 2: Randomly select a split value between min and max of that feature
Step 3: Recursively repeat until each point is isolated

Normal point:    requires MANY splits to isolate (deep in the tree)
Outlier point:   isolated in FEW splits (shallow in the tree)

Path length ∝ normality:
  Short average path length → OUTLIER  (-1)
  Long  average path length → NORMAL   (+1)
```

### Implementation

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    contamination=0.05,   # expected proportion of outliers (5%)
    random_state=42,
    n_estimators=100      # number of isolation trees
)

# fit_predict: -1 = outlier, 1 = normal
y_pred_iso = iso_forest.fit_predict(data)

# Add predictions to DataFrame
df["IsoForest_Label"] = y_pred_iso
print(df["IsoForest_Label"].value_counts())
#  1    1519   ← normal
# -1      80   ← outliers

# Get anomaly scores (more negative = more anomalous)
scores = iso_forest.score_samples(data)
df["IsoForest_Score"] = scores
```

### Visualise Results

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="alcohol", y="residual sugar",
    data=df, hue="IsoForest_Label",
    palette={1: "steelblue", -1: "red"},
    style="IsoForest_Label"
)
plt.title("Isolation Forest — Outlier Detection")
plt.legend(title="Label", labels=["Normal (1)", "Outlier (-1)"])
plt.show()
```

### Remove Isolation Forest Outliers

```python
data_clean_iso = df[df["IsoForest_Label"] == 1].drop(
    columns=["IsoForest_Label", "IsoForest_Score"]
)
print(f"Rows before: {len(df)}")
print(f"Rows after Isolation Forest removal: {len(data_clean_iso)}")
```

### Tuning the `contamination` Parameter

```python
# If you know ~2% are outliers:
IsolationForest(contamination=0.02)

# If you don't know: use 'auto' (scikit-learn default)
IsolationForest(contamination='auto')
```

| ✅ Pros | ❌ Cons |
|---|---|
| Handles high-dimensional data efficiently | Requires setting contamination parameter |
| Fast — O(n log n) complexity | Results vary with contamination estimate |
| No distance or density calculations needed | Less interpretable than statistical methods |
| Detects both global and collective outliers | Random splits can vary — set `random_state` |

---

## 📌 Method 4 — Local Outlier Factor (LOF)

A **density-based anomaly detection** method that compares the local density of each point to its neighbours. Points with significantly lower density than their neighbours are flagged as outliers.

### How It Works

```
Step 1: For each point, find k nearest neighbours
Step 2: Estimate reachability distance (smoothed distance to neighbours)
Step 3: Compute local reachability density (LRD):
          LRD = 1 / (average reachability distance to k neighbours)
Step 4: Compute LOF score:
          LOF = average(LRD of neighbours) / LRD of the point

LOF ≈ 1  →  similar density to neighbours → NORMAL
LOF >> 1 →  much lower density than neighbours → OUTLIER
```

```
Visualising density:
  Dense cluster:  ●●●●●●   ← high LRD, LOF ≈ 1
  Sparse outlier:     ●    ← low LRD, LOF >> 1
                         ↑ far from cluster = outlier
```

### Implementation

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,        # neighbourhood size (k)
    contamination=0.05     # expected outlier proportion
)

# fit_predict: -1 = outlier, 1 = normal
y_pred_lof = lof.fit_predict(data)

df["LOF_Label"] = y_pred_lof
print(df["LOF_Label"].value_counts())
#  1    1519   ← normal
# -1      80   ← outliers

# LOF scores (more negative = more anomalous)
lof_scores = lof.negative_outlier_factor_
df["LOF_Score"] = lof_scores
```

### Visualise Results

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="alcohol", y="volatile acidity",
    data=df, hue="LOF_Label",
    palette={1: "steelblue", -1: "red"},
    style="LOF_Label"
)
plt.title("Local Outlier Factor (LOF) — Outlier Detection")
plt.legend(title="Label", labels=["Normal (1)", "Outlier (-1)"])
plt.show()
```

### Visualise LOF Scores

```python
# Plot LOF scores to understand the distribution of outlierness
plt.figure(figsize=(10, 4))
plt.scatter(range(len(lof_scores)), lof_scores,
            c=['red' if s < -1.5 else 'steelblue' for s in lof_scores],
            alpha=0.6, s=10)
plt.axhline(-1.5, color='red', linestyle='--', label='Threshold (-1.5)')
plt.title("LOF Anomaly Scores")
plt.xlabel("Sample Index")
plt.ylabel("Negative Outlier Factor")
plt.legend()
plt.show()
```

### Remove LOF Outliers

```python
data_clean_lof = df[df["LOF_Label"] == 1].drop(
    columns=["LOF_Label", "LOF_Score"]
)
print(f"Rows before: {len(df)}")
print(f"Rows after LOF removal: {len(data_clean_lof)}")
```

| ✅ Pros | ❌ Cons |
|---|---|
| Detects local outliers in clusters of varying density | Sensitive to the choice of k (n_neighbors) |
| Works well with complex, non-uniform distributions | Computationally more expensive than Z-Score/IQR |
| Identifies outliers that global methods miss | Requires contamination parameter estimate |
| No assumption on data distribution | Not suitable for very high-dimensional sparse data |

---

## 🗑️ Outlier Removal Strategies

Once detected, outliers can be handled in several ways depending on context.

---

### 1. Deletion

Simply remove rows identified as outliers. Safest when outliers are data entry errors.

```python
# After Z-Score detection
mask_z   = (np.abs(stats.zscore(data)) < 3).all(axis=1)
data_no_outliers = data[mask_z]

# After IQR detection
Q1, Q3   = data.quantile(0.25), data.quantile(0.75)
IQR      = Q3 - Q1
mask_iqr = ~((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).any(axis=1)
data_no_outliers_iqr = data[mask_iqr]

# After Isolation Forest
data_no_outliers_iso = df[df["IsoForest_Label"] == 1]
```

| ✅ Pros | ❌ Cons |
|---|---|
| Simple and clean | Reduces dataset size |
| Prevents contamination of model | Risk of removing genuine rare events |

---

### 2. Capping (Winsorisation)

Instead of removing outliers, **clip** them to the boundary values — replacing extremes with the nearest acceptable value.

```python
# Cap using IQR bounds
Q1  = data.quantile(0.25)
Q3  = data.quantile(0.75)
IQR = Q3 - Q1
lb  = Q1 - 1.5 * IQR
ub  = Q3 + 1.5 * IQR

data_capped = data.clip(lower=lb, upper=ub, axis=1)

# Cap using percentiles (e.g., 1st and 99th percentile)
p01 = data.quantile(0.01)
p99 = data.quantile(0.99)
data_winsorized = data.clip(lower=p01, upper=p99, axis=1)

# Using scipy Winsorize on a single column
from scipy.stats.mstats import winsorize
data['alcohol_w'] = winsorize(data['alcohol'], limits=[0.05, 0.05])
```

| ✅ Pros | ❌ Cons |
|---|---|
| Preserves all rows — no data loss | Modifies the true value of data points |
| Retains information that outlier exists | Boundary values become over-represented |
| Reduces extreme value influence | May not be appropriate for all features |

---

### 3. Transformation

Apply a mathematical transformation to **reduce the influence of outliers** without removing them.

```python
import numpy as np

# Log transformation — compresses large values
data['residual_sugar_log'] = np.log1p(data['residual sugar'])

# Square root transformation
data['chlorides_sqrt'] = np.sqrt(data['chlorides'])

# Box-Cox (requires positive values)
from scipy.stats import boxcox
data['sulphates_bc'], lambda_val = boxcox(data['sulphates'])

# Yeo-Johnson (handles negatives and zeros)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
data['alcohol_yj'] = pt.fit_transform(data[['alcohol']])
```

| ✅ Pros | ❌ Cons |
|---|---|
| Preserves all data points | Changes interpretability of the feature |
| Reduces skewness and outlier influence | Not all transformations work for all features |
| Often improves model performance | Inverse transformation needed for predictions |

---

### 4. Imputation

Replace outlier values with **estimated values** (mean, median, or model-based). Treats outliers similarly to missing values.

```python
import numpy as np

# Identify outlier positions using Z-score
z_scores = np.abs(stats.zscore(data))
data_imputed = data.copy()

# Replace outliers with column median
for col_idx, col in enumerate(data.columns):
    col_median = data[col].median()
    outlier_mask = z_scores[:, col_idx] > 3
    data_imputed.loc[outlier_mask, col] = col_median

print("Outliers replaced with column medians.")
print(data_imputed.describe())
```

| ✅ Pros | ❌ Cons |
|---|---|
| No data loss | Introduces artificial values |
| Preserves row count | May mask genuine anomalies |
| Good when outlier is likely a measurement error | Adds noise if median is a poor substitute |

---

### 5. Use Robust Algorithms

Some ML algorithms are inherently **less sensitive to outliers** — a valid strategy is to simply choose a robust model.

| Algorithm | Sensitivity to Outliers | Notes |
|---|---|---|
| **Linear Regression** | High ❌ | Minimises squared error — heavily influenced by outliers |
| **Lasso / Ridge Regression** | Medium ⚠️ | Regularisation reduces but doesn't eliminate outlier effect |
| **Huber Regression** | Low ✅ | Uses L1 loss for outliers, L2 for normal points |
| **Decision Trees** | Low ✅ | Split-based — outliers rarely affect splits significantly |
| **Random Forest** | Low ✅ | Ensemble of trees — outlier influence is averaged out |
| **SVR with RBF Kernel** | Low ✅ | ε-insensitive loss ignores small deviations |
| **Median-based methods** | Very Low ✅ | Median is inherently robust to extremes |

```python
# Example: Huber Regressor — robust linear regression
from sklearn.linear_model import HuberRegressor

model = HuberRegressor(epsilon=1.35)   # epsilon controls robustness
model.fit(X_train, y_train)
```

---

## 🔁 Full Pipeline — Detect & Remove

A complete end-to-end pipeline combining detection and removal:

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def outlier_pipeline(df, target_col, method='iqr', contamination=0.05):
    """
    Full outlier detection and removal pipeline.

    Parameters:
        df            : Input DataFrame
        target_col    : Target column to exclude from outlier detection
        method        : 'zscore' | 'iqr' | 'isoforest' | 'lof'
        contamination : Expected outlier fraction (for isoforest/lof)

    Returns:
        df_clean : DataFrame with outliers removed
        report   : Dictionary with detection statistics
    """
    features = df.drop(columns=[target_col])
    report   = {'method': method, 'rows_before': len(df)}

    if method == 'zscore':
        z = np.abs(stats.zscore(features.select_dtypes(include=np.number)))
        mask = (z < 3).all(axis=1)

    elif method == 'iqr':
        num = features.select_dtypes(include=np.number)
        Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((num < Q1 - 1.5 * IQR) | (num > Q3 + 1.5 * IQR)).any(axis=1)

    elif method == 'isoforest':
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(features.select_dtypes(include=np.number))
        mask = preds == 1

    elif method == 'lof':
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(contamination=contamination)
        preds = lof.fit_predict(features.select_dtypes(include=np.number))
        mask = preds == 1

    df_clean = df[mask].reset_index(drop=True)
    report['rows_after']   = len(df_clean)
    report['removed']      = report['rows_before'] - report['rows_after']
    report['removed_pct']  = round(report['removed'] / report['rows_before'] * 100, 2)

    print(f"\n── Outlier Removal Report ({method.upper()}) ──")
    print(f"  Rows before : {report['rows_before']}")
    print(f"  Rows after  : {report['rows_after']}")
    print(f"  Removed     : {report['removed']} ({report['removed_pct']}%)")

    return df_clean, report


# Usage
df_clean, report = outlier_pipeline(df, target_col='quality', method='iqr')
```

---

## 📊 Comparison of Detection Techniques

| Technique | Type | Key Idea | Works Best For | Pros | Cons |
|---|---|---|---|---|---|
| **Z-Score** | Statistical | Flags points far from mean (in SD units) | Normally distributed data | Simple, fast, interpretable | Unreliable for skewed / non-normal data |
| **IQR** | Statistical | Flags points outside 1.5×IQR from Q1/Q3 | Any univariate data, boxplots | Robust, non-parametric | Doesn't capture multivariate outliers |
| **Isolation Forest** | Model-based | Isolates anomalies via random tree splits | High-dimensional datasets | Efficient, handles many features | Requires contamination estimate |
| **LOF** | Density-based | Compares local density to neighbours | Clustered / varying-density data | Detects local outliers well | Sensitive to k, computationally costlier |

---

## 🤔 When to Remove vs. Keep Outliers

| Situation | Recommendation |
|---|---|
| Outlier is a **data entry error** (typo, unit mismatch) | ✅ Remove or correct |
| Outlier is a **measurement/sensor error** | ✅ Remove or impute |
| Outlier represents a **genuine rare event** (fraud, failure) | ⚠️ Keep — it's valuable signal |
| Outlier appears in the **target variable** | ⚠️ Investigate carefully before removing |
| Using **fraud or anomaly detection** as the goal | ❌ Never remove — outliers ARE the target |
| Outlier is from a **different population** (wrong segment) | ✅ Remove — it's a data quality issue |
| Outlier affects **model convergence** severely | ✅ Cap or transform rather than delete |

---

## 🗺️ Decision Guide

```
Detected an outlier?
    │
    ├─► Is it a data error (typo, sensor fault)?
    │       └─► YES ──► Correct or DELETE it
    │
    ├─► Is missingness domain-meaningful (fraud, anomaly)?
    │       └─► YES ──► KEEP IT — it's your signal
    │
    ├─► Is the dataset large enough to lose rows?
    │       └─► NO  ──► CAP (Winsorise) or IMPUTE instead of delete
    │
    ├─► Is the feature heavily skewed?
    │       └─► YES ──► TRANSFORM (log, sqrt, Box-Cox)
    │
    ├─► Is the outlier in high-dimensional space?
    │       └─► YES ──► Use ISOLATION FOREST or LOF
    │
    └─► Are you unsure about the distribution?
            └─► YES ──► Use IQR (non-parametric, safe default)
```

---

## 🔑 Key Takeaways

- ✅ **Always visualise** your data with boxplots and scatter plots before applying any detection algorithm
- 🔵 Understand the **type of outlier** (global / contextual / collective) — it determines the right detection method
- 📊 **Z-Score** is fast and simple but assumes normality — use only on Gaussian-distributed features
- 📦 **IQR** is the robust default for univariate outlier detection — works on any distribution
- 🌲 **Isolation Forest** is the best choice for high-dimensional, multivariate outlier detection
- 🌐 **LOF** excels when data has clusters of varying density — it finds local anomalies that global methods miss
- ⚠️ **Never automatically delete** outliers — always investigate whether they represent real-world signal
- 🔢 **Capping (Winsorisation)** is a safer alternative to deletion — it preserves sample size while reducing outlier influence
- 🤖 When in doubt, **use a robust algorithm** (Random Forest, Huber Regression) instead of removing outliers
- 🎯 In **fraud detection and anomaly monitoring**, outliers are the target — removing them defeats the purpose

---

## 🛠️ Prerequisites

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

**All imports used in this guide:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler
```

---

## 📚 Further Reading

- [Scikit-learn: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Scikit-learn: IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Scikit-learn: LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [Z-Score for Outlier Detection — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/z-score-for-outlier-detection-python/)
- [IQR Method — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/interquartile-range-to-detect-outliers-in-data/)
- [Isolation Forest — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/what-is-isolation-forest/)
- [Local Outlier Factor — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/local-outlier-factor/)
---
