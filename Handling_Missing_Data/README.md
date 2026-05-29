# 🔍 Handling Missing Data Techniques in Machine Learning
---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why Missing Data Matters](#-why-missing-data-matters)
- [Reasons Behind Missing Values](#-reasons-behind-missing-values)
- [Types of Missing Data](#-types-of-missing-data)
  - [MCAR — Missing Completely at Random](#1-mcar--missing-completely-at-random)
  - [MAR — Missing at Random](#2-mar--missing-at-random)
  - [MNAR — Missing Not at Random](#3-mnar--missing-not-at-random)
- [Representing Missing Values](#-representing-missing-values)
- [Detecting Missing Values](#-detecting-missing-values)
- [Sample Dataset Setup](#-sample-dataset-setup)
- [Strategies for Handling Missing Values](#-strategies-for-handling-missing-values)
  - [Strategy 1 — Deletion](#-strategy-1--deletion)
    - [Row Deletion (dropna)](#1-row-deletion-dropna)
    - [Column Deletion](#2-column-deletion)
  - [Strategy 2 — Imputation](#-strategy-2--imputation)
    - [Mean Imputation](#1-mean-imputation)
    - [Median Imputation](#2-median-imputation)
    - [Mode Imputation](#3-mode-imputation)
    - [Forward Fill](#4-forward-fill-ffill)
    - [Backward Fill](#5-backward-fill-bfill)
    - [Constant / Custom Value Fill](#6-constant--custom-value-fill)
  - [Strategy 3 — Interpolation](#-strategy-3--interpolation)
    - [Linear Interpolation](#1-linear-interpolation)
    - [Quadratic Interpolation](#2-quadratic-interpolation)
  - [Strategy 4 — Advanced / ML-Based Imputation](#-strategy-4--advanced--ml-based-imputation)
    - [KNN Imputer](#1-knn-imputer)
    - [Iterative Imputer (MICE)](#2-iterative-imputer-mice)
- [Pandas Quick-Reference Functions](#-pandas-quick-reference-functions)
- [Impact of Handling Missing Values](#-impact-of-handling-missing-values)
- [Quick Comparison Table](#-quick-comparison-table)
- [Decision Guide — When to Use Which?](#-decision-guide--when-to-use-which)
- [Key Takeaways](#-key-takeaways)
- [Prerequisites](#-prerequisites)
- [Further Reading](#-further-reading)

---

## 🧭 Overview

Missing values are one of the most common challenges in real-world datasets. They appear when some entries are left blank, marked as `NaN`, `None`, `NULL`, or represented by special placeholders like `-999` or `"Unknown"`.

If not handled properly, missing data can:

- Reduce model accuracy
- Introduce bias into predictions
- Break algorithms that require complete input data
- Produce misleading statistical summaries

```
Raw Data (with NaNs)
        │
        ▼
┌───────────────────┐
│  Detect & Profile │  ← .isnull(), .info(), .describe()
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Understand Type  │  ← MCAR / MAR / MNAR
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Choose Strategy  │  ← Delete / Impute / Interpolate / ML-based
└────────┬──────────┘
         │
         ▼
  Clean Dataset ──► ML Model
```

---

## ❓ Why Missing Data Matters

| Impact | Description |
|---|---|
| **Model Accuracy** | Training on incomplete data leads to poor generalisation |
| **Bias** | Non-random missing data skews predictions toward certain groups |
| **Sample Size** | Losing rows reduces statistical power |
| **Algorithm Compatibility** | Many algorithms (SVM, Linear Regression) cannot handle NaN values at all |
| **Data Integrity** | Unchecked NaNs silently corrupt downstream computations |

---

## 🔎 Reasons Behind Missing Values

Understanding *why* data is missing is the first step to choosing the right strategy.

```
Missing Values
    │
    ├── Technical Issues
    │       └── Failed data collection, transmission errors, sensor failures
    │
    ├── Human Errors
    │       └── Incorrect data entry, skipped fields, copy-paste mistakes
    │
    ├── Privacy Concerns
    │       └── Respondents omitting sensitive info (income, age, health)
    │
    └── Data Processing Issues
            └── Merging/joining datasets with mismatched keys, ETL bugs
```

> 💡 **Pro tip:** Always ask *why* data is missing before choosing *how* to handle it.
> A random technical glitch (MCAR) calls for a different strategy than a deliberate privacy omission (MNAR).

---

## 📂 Types of Missing Data

Understanding the **mechanism** behind missingness is fundamental — it determines which handling strategy is statistically valid.

---

### 1. MCAR — Missing Completely at Random

The probability of a value being missing is **completely independent** of any variable in the dataset — it is pure random chance.

```
Example:
  A sensor randomly fails once a week regardless of temperature, load, or time.
  Survey responses randomly lost due to a server crash.

  Missing: completely random — no pattern
```

**Implication:** Any imputation or deletion method is statistically valid. Deletion does not introduce bias.

---

### 2. MAR — Missing at Random

Missingness **depends on other observed variables** but NOT on the missing value itself.

```
Example:
  Men are less likely to answer questions about mental health (gender → missingness),
  but among men who do respond, the missing answers are random.

  Missing: explained by other variables in the dataset
```

**Implication:** Imputation is recommended — use other observed variables to estimate missing values. Deletion can introduce bias.

---

### 3. MNAR — Missing Not at Random

Missingness is **directly related to the value that is missing** itself — the most problematic type.

```
Example:
  High-income individuals deliberately leave the "Annual Income" field blank.
  Patients with severe symptoms are less likely to complete health surveys.

  Missing: related to the unobserved value — structural missingness
```

**Implication:** Standard imputation is unreliable. Requires domain expertise, sensitivity analysis, or specialised models. Simply imputing with mean/median will introduce systematic bias.

---

### Summary Table — Types of Missing Data

| Type | Depends On | Bias Risk | Recommended Strategy |
|---|---|---|---|
| **MCAR** | Nothing — pure random | Low | Any method; deletion is safe |
| **MAR** | Other observed variables | Medium | Imputation using related features |
| **MNAR** | The missing value itself | High | Domain analysis; specialised models |

---

## 🏷️ Representing Missing Values in Datasets

Missing values appear in many forms depending on the data source:

| Representation | Example | Common In |
|---|---|---|
| **Blank cells** | ` ` (empty) | CSV files, spreadsheets |
| **NaN** | `np.nan` | Pandas DataFrames, Python |
| **None** | `None` | Python objects, JSON |
| **NULL** | `NULL` | SQL databases |
| **Special numbers** | `-999`, `0`, `-1` | Legacy systems, sensors |
| **Text flags** | `"NA"`, `"Unknown"`, `"MISSING"` | Survey data, ERP exports |
| **Infinity** | `np.inf` | Computed features with division |

```python
import pandas as pd
import numpy as np

# All of these represent missing data — handle all consistently
df.replace([-999, 'Unknown', 'NA', 'MISSING', '', ' '], np.nan, inplace=True)
```

---

## 🔬 Detecting Missing Values

Before handling missing data, always **profile it** thoroughly.

```python
import pandas as pd
import numpy as np

# ── Basic detection ───────────────────────────────────────────
df.isnull()              # Boolean DataFrame: True where NaN
df.isnull().sum()        # Count of NaNs per column
df.isnull().sum() / len(df) * 100  # Percentage missing per column

df.notnull()             # Inverse: True where NOT NaN
df.info()                # Summary: dtype + non-null count per column
df.isna()                # Same as isnull()

# ── Full missing value profile ────────────────────────────────
def missing_profile(df):
    total     = df.isnull().sum()
    percent   = (df.isnull().sum() / len(df) * 100).round(2)
    dtype     = df.dtypes
    profile   = pd.DataFrame({
        'Missing Count'  : total,
        'Missing %'      : percent,
        'Data Type'      : dtype
    })
    return profile[profile['Missing Count'] > 0].sort_values('Missing %', ascending=False)

print(missing_profile(df))

# ── Visualise missingness ─────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# Bar chart of missing %
missing_pct = df.isnull().sum() / len(df) * 100
missing_pct[missing_pct > 0].plot(kind='bar', color='tomato')
plt.title('Missing Value % per Column')
plt.ylabel('Missing %')
plt.show()
```

### Pandas Quick-Reference Functions

| Function | Description |
|---|---|
| `.isnull()` | Returns boolean DataFrame — `True` where value is missing |
| `.notnull()` | Returns boolean DataFrame — `True` where value is present |
| `.isna()` | Alias for `.isnull()` |
| `.info()` | Summary with non-null counts and dtypes per column |
| `.describe()` | Statistical summary — missing values reduce the count |
| `.dropna()` | Remove rows or columns with missing values |
| `.fillna()` | Fill missing values with a specified value or method |
| `.interpolate()` | Fill missing values using interpolation |
| `.replace()` | Replace specific values (e.g. `-999` → `NaN`) |

---

## 🗂️ Sample Dataset Setup

All code examples in this guide use the following sample DataFrame:

```python
import pandas as pd
import numpy as np

data = {
    'School ID': [101, 102, 103, np.nan, 105, 106, 107, 108],
    'Name'     : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Address'  : ['123 Main St', '456 Oak Ave', '789 Pine Ln', '101 Elm St',
                  np.nan, '222 Maple Rd', '444 Cedar Blvd', '555 Birch Dr'],
    'City'     : ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai',
                  'Kolkata', np.nan, 'Pune', 'Jaipur'],
    'Subject'  : ['Math', 'English', 'Science', 'Math',
                  'History', 'Math', 'Science', 'English'],
    'Marks'    : [85, 92, 78, 89, np.nan, 95, 80, 88],
    'Rank'     : [2, 1, 4, 3, 8, 1, 5, 3],
    'Grade'    : ['B', 'A', 'C', 'B', 'D', 'A', 'C', 'B']
}

df = pd.DataFrame(data)
print(df)
print("\nMissing values per column:")
print(df.isnull().sum())
```

**Output — Missing values:**
```
School ID    1
Address      1
City         1
Marks        1
dtype: int64   (4 missing values across 4 columns)
```

---

## 🛠️ Strategies for Handling Missing Values

---

## 🗑️ Strategy 1 — Deletion

The simplest approach — remove rows or columns containing missing values. Best used when missingness is MCAR and the missing proportion is small.

---

### 1. Row Deletion (dropna)

Removes any row that contains at least one missing value.

```python
# Remove all rows with any NaN
df_cleaned = df.dropna()
print("Shape before:", df.shape)
print("Shape after :", df_cleaned.shape)

# Remove rows only where specific columns are NaN
df_cleaned_subset = df.dropna(subset=['Marks', 'City'])

# Remove rows where ALL values are NaN (entirely empty rows)
df_cleaned_all = df.dropna(how='all')

# Keep rows with at least 5 non-null values
df_thresh = df.dropna(thresh=5)
```

**dropna() Parameters:**

| Parameter | Values | Description |
|---|---|---|
| `axis` | `0` (rows), `1` (cols) | What to drop |
| `how` | `'any'`, `'all'` | Drop if any NaN or only if all NaN |
| `thresh` | int | Minimum number of non-NaN values required |
| `subset` | list of columns | Only consider these columns for NaN check |

**Pros and Cons:**

| ✅ Pros | ❌ Cons |
|---|---|
| Simple and fast | Reduces dataset size |
| Results in a fully complete dataset | Can introduce bias if missingness is not MCAR |
| No assumptions about missing values | Loses potentially valuable data |

> ⚠️ **Rule of thumb:** Only use deletion if missing values are < 5% of the dataset.

---

### 2. Column Deletion

Remove entire columns with too many missing values.

```python
# Drop columns with more than 40% missing values
threshold = 0.4
df_col_dropped = df.loc[:, df.isnull().mean() < threshold]

# Drop a specific column
df_no_address = df.drop(columns=['Address'])
```

> ⚠️ Only drop a column if it has very high missingness (> 60–70%) AND low predictive value.
> Never drop a column that is a key feature without careful consideration.

---

## 🩹 Strategy 2 — Imputation

Imputation replaces missing values with **estimated values** derived from the existing data. Preserves sample size but may introduce distributional assumptions.

---

### 1. Mean Imputation

Replaces missing values with the **arithmetic mean** of the column. Best for **normally distributed** numeric data without many outliers.

```python
# Manual
mean_imputation = df['Marks'].fillna(df['Marks'].mean())
print(mean_imputation)

# In-place on DataFrame
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

# Using Scikit-learn SimpleImputer
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(strategy='mean')
df[['Marks']] = imputer.fit_transform(df[['Marks']])
```

**When to use:** Continuous numeric features with roughly normal distribution and no significant outliers.

| ✅ Pros | ❌ Cons |
|---|---|
| Simple and fast | Reduces variance — compresses the distribution |
| Preserves sample size | Distorts correlation between variables |
| Good for normally distributed data | Sensitive to outliers — pulls mean toward extremes |

---

### 2. Median Imputation

Replaces missing values with the **median** of the column. More robust than mean — preferred for **skewed data or data with outliers**.

```python
# Manual
median_imputation = df['Marks'].fillna(df['Marks'].median())

# Scikit-learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df[['Marks']] = imputer.fit_transform(df[['Marks']])
```

**When to use:** Numeric features with skewed distributions or outliers (income, house prices, trade volumes).

| ✅ Pros | ❌ Cons |
|---|---|
| Robust to outliers | Still reduces variance |
| Better than mean for skewed data | Ignores relationships between variables |

---

### 3. Mode Imputation

Replaces missing values with the **most frequently occurring value**. The only valid simple imputation method for **categorical features**.

```python
# Manual — for categorical
mode_imputation = df['City'].fillna(df['City'].mode().iloc[0])

# For numeric
mode_num = df['Marks'].fillna(df['Marks'].mode().iloc[0])

# Scikit-learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
df[['City']] = imputer.fit_transform(df[['City']])
```

**When to use:** Categorical features (city, grade, subject, gender).

| ✅ Pros | ❌ Cons |
|---|---|
| Only valid simple method for categoricals | Can over-represent the most common category |
| Easy to implement | May be misleading when no clear mode exists |

---

### 4. Forward Fill (ffill)

Propagates the **last valid value forward** to fill subsequent missing values. Ideal for **time-series or ordered sequential data**.

```python
# Single column
forward_fill = df['Marks'].ffill()

# Entire DataFrame
df_ffill = df.ffill()

# Limit how many consecutive NaNs to fill
df_ffill_limited = df['Marks'].ffill(limit=2)
```

**When to use:** Ordered data where the previous value is a good estimate — stock prices, sensor readings, sequential records.

| ✅ Pros | ❌ Cons |
|---|---|
| Preserves temporal/sequential patterns | Inaccurate over long gaps |
| No distributional assumption | Not suitable for non-ordered data |

```
Before ffill:  [85, 92, NaN, NaN, 89, NaN, 95]
After  ffill:  [85, 92,  92,  92, 89,  89, 95]
```

---

### 5. Backward Fill (bfill)

Propagates the **next valid value backward** to fill preceding missing values.

```python
# Single column
backward_fill = df['Marks'].bfill()

# Entire DataFrame
df_bfill = df.bfill()
```

**When to use:** Same context as forward fill — ordered data where the next value is a better estimate than the previous.

```
Before bfill:  [85, NaN, NaN, 89, NaN, 95]
After  bfill:  [85,  89,  89, 89,  95, 95]
```

| ✅ Pros | ❌ Cons |
|---|---|
| Good for series where future context is known | Uses future values — data leakage risk in some cases |
| Preserves sequence structure | Inaccurate over large gaps |

---

### 6. Constant / Custom Value Fill

Fills missing values with a **user-specified constant** — useful when a specific placeholder is meaningful.

```python
# Fill numeric column with 0
df['Marks'] = df['Marks'].fillna(0)

# Fill categorical column with 'Unknown'
df['City'] = df['City'].fillna('Unknown')

# Fill with a domain-specific value
df['School ID'] = df['School ID'].fillna(-1)   # -1 = not assigned

# Scikit-learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
df[['City']] = imputer.fit_transform(df[['City']])
```

**When to use:** When missingness is itself meaningful (e.g., no city = remote/unregistered, no score = absent).

---

## 📈 Strategy 3 — Interpolation

Interpolation estimates missing values by **fitting a function through surrounding data points**. More sophisticated than flat imputation — captures underlying trends.

---

### 1. Linear Interpolation

Assumes a **straight line** between two adjacent non-missing values and estimates the missing point on that line.

```python
# Linear interpolation
linear_interp = df['Marks'].interpolate(method='linear')
print(linear_interp)
```

**How it works:**
```
Values: [85, NaN, NaN, 92]
         │               │
         └───── gap ─────┘
Linear:  [85,  87.3, 89.7, 92]   ← evenly spaced between 85 and 92
```

| ✅ Pros | ❌ Cons |
|---|---|
| Captures linear trends | Assumes straight-line relationship |
| Better than flat imputation for series | Inaccurate for non-linear patterns |
| Simple to apply | Not suitable for large gaps |

---

### 2. Quadratic Interpolation

Fits a **quadratic curve (parabola)** through three adjacent non-missing values for a smoother, more flexible fit.

```python
# Quadratic interpolation
quadratic_interp = df['Marks'].interpolate(method='quadratic')
print(quadratic_interp)
```

**Other interpolation methods available:**

```python
# Polynomial (specify degree)
df['Marks'].interpolate(method='polynomial', order=3)

# Time-based (for time-indexed DataFrames)
df['Marks'].interpolate(method='time')

# Spline
df['Marks'].interpolate(method='spline', order=3)

# Nearest neighbour
df['Marks'].interpolate(method='nearest')
```

**Interpolation methods summary:**

| Method | Best For | Notes |
|---|---|---|
| `'linear'` | Evenly spaced data | Default; assumes straight-line trend |
| `'quadratic'` | Curved trends | Needs at least 3 surrounding points |
| `'polynomial'` | Complex patterns | Set `order` parameter |
| `'time'` | Datetime-indexed series | Accounts for uneven time gaps |
| `'spline'` | Smooth curves | Flexible; set `order` |
| `'nearest'` | Categorical-like numeric | Copies nearest value |

| ✅ Pros | ❌ Cons |
|---|---|
| Captures data trends accurately | Assumes a specific functional form |
| Better than mean/median for series | Can overshoot at boundaries (Runge's phenomenon) |
| Multiple methods available | Not suitable for categorical data |

---

## 🤖 Strategy 4 — Advanced / ML-Based Imputation

When simple imputation is insufficient, use machine learning to predict missing values using relationships between features.

---

### 1. KNN Imputer

Finds the **k nearest neighbours** of each row with missing data (based on other features) and fills in the missing value using the average of those neighbours.

```python
from sklearn.impute import KNNImputer
import numpy as np

data_numeric = df[['School ID', 'Marks', 'Rank']].copy()

imputer = KNNImputer(n_neighbors=3)
data_imputed = imputer.fit_transform(data_numeric)

df_knn = pd.DataFrame(data_imputed, columns=['School ID', 'Marks', 'Rank'])
print(df_knn)
```

**How it works:**
```
Row with missing 'Marks' → find 3 most similar rows by 'School ID' and 'Rank'
                         → average their 'Marks' values
                         → fill in the missing value
```

**KNNImputer parameters:**

| Parameter | Default | Description |
|---|---|---|
| `n_neighbors` | 5 | Number of neighbours to consider |
| `weights` | `'uniform'` | `'uniform'` or `'distance'` weighting |
| `metric` | `'nan_euclidean'` | Distance metric (handles NaN) |

| ✅ Pros | ❌ Cons |
|---|---|
| Uses feature relationships | Computationally expensive on large datasets |
| More accurate than mean/median | Sensitive to scale — requires normalisation first |
| Non-parametric — no distribution assumption | Choice of k affects quality |

> ⚠️ **Always scale your features before using KNNImputer** using `StandardScaler` or `MinMaxScaler`.

---

### 2. Iterative Imputer (MICE)

**MICE (Multiple Imputation by Chained Equations)** — models each feature with missing values as a function of all other features, iteratively refining the estimates. The gold standard for advanced imputation.

```python
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # must enable first
import numpy as np

data_numeric = df[['School ID', 'Marks', 'Rank']].copy()

imputer = IterativeImputer(
    max_iter=10,       # number of imputation rounds
    random_state=42,
    initial_strategy='mean'  # starting values before iteration
)

data_imputed = imputer.fit_transform(data_numeric)
df_mice = pd.DataFrame(data_imputed, columns=['School ID', 'Marks', 'Rank'])
print(df_mice)
```

**How MICE works:**
```
Round 1:
  Fill all NaNs with mean (initial guess)
  For each column with missing values:
    → Train a regression model using all OTHER columns as features
    → Predict missing values in that column

Round 2, 3 ... n:
  Re-impute each column using updated values from previous round
  → Repeat until convergence
```

| ✅ Pros | ❌ Cons |
|---|---|
| Most statistically rigorous method | Computationally intensive |
| Accounts for relationships between all features | Requires more configuration |
| Well-suited for MAR data | Can be slow on wide datasets |
| Produces multiple plausible imputations | Experimental in scikit-learn |

---

## 📊 Quick Comparison Table

| Strategy | Method | Data Type | Missing % | When to Use |
|---|---|---|---|---|
| **Deletion** | `dropna()` | Any | < 5% | MCAR, small missing proportion |
| **Mean Imputation** | `fillna(mean)` | Numeric | Any | Normal distribution, no outliers |
| **Median Imputation** | `fillna(median)` | Numeric | Any | Skewed data, outliers present |
| **Mode Imputation** | `fillna(mode)` | Categorical / Numeric | Any | Categorical features |
| **Forward Fill** | `ffill()` | Numeric / Categorical | Moderate | Time-series, ordered sequential data |
| **Backward Fill** | `bfill()` | Numeric / Categorical | Moderate | Time-series, ordered sequential data |
| **Constant Fill** | `fillna(value)` | Any | Any | Missing = meaningful category |
| **Linear Interpolation** | `interpolate('linear')` | Numeric | Moderate | Data with linear trends |
| **Quadratic Interpolation** | `interpolate('quadratic')` | Numeric | Moderate | Data with curved trends |
| **KNN Imputer** | `KNNImputer()` | Numeric | Moderate | Feature-correlated missingness |
| **MICE / Iterative** | `IterativeImputer()` | Numeric | High | MAR data, best accuracy needed |

---

## 🗺️ Decision Guide — When to Use Which?

```
How much data is missing?
    │
    ├─► < 5% ──► Consider DELETION (dropna) if MCAR
    │
    └─► > 5% ──► Use IMPUTATION or INTERPOLATION
                     │
                     ├─► Is data ordered / time-series?
                     │       └─► YES ──► Forward Fill / Backward Fill / Interpolation
                     │
                     ├─► Is feature CATEGORICAL?
                     │       └─► YES ──► Mode Imputation or Constant Fill ('Unknown')
                     │
                     ├─► Is feature NUMERIC with outliers or skew?
                     │       └─► YES ──► Median Imputation
                     │
                     ├─► Is feature NUMERIC and normally distributed?
                     │       └─► YES ──► Mean Imputation
                     │
                     ├─► Is missing data related to other features (MAR)?
                     │       └─► YES ──► KNN Imputer or Iterative Imputer (MICE)
                     │
                     └─► Is highest accuracy essential?
                             └─► YES ──► Iterative Imputer (MICE)
```

---

## 💡 Impact of Handling Missing Values Correctly

| Outcome | Description |
|---|---|
| **Improved Data Quality** | Cleaner datasets are more reliable for training and analysis |
| **Enhanced Model Performance** | Complete data leads to more accurate predictions |
| **Preserved Data Integrity** | Consistency maintained across the dataset |
| **Reduced Bias** | Prevents systematic distortion in model outputs |
| **Better Statistical Summaries** | Means, variances and correlations reflect true patterns |

---

## 🔑 Key Takeaways

- ✅ Always **detect and profile** missing values before choosing a strategy — use `.isnull().sum()` and visualise with heatmaps
- 🔵 Understand the **type of missingness** (MCAR / MAR / MNAR) — it determines which method is statistically valid
- 🗑️ **Deletion** is only safe when missingness is MCAR and < 5% of the data
- 📏 **Mean imputation** is fast but distorts variance — prefer **median** when outliers are present
- 🏷️ **Mode imputation** is the correct choice for categorical features
- 📈 **Forward/backward fill** is best for time-series and ordered data
- 🤖 **KNN Imputer** and **Iterative Imputer (MICE)** are the most accurate for MAR data with feature relationships
- ⚠️ No method is perfect — always validate imputation quality by comparing model performance before and after
- 🔁 For **MNAR data**, no automated method is reliable — domain expertise is essential

---

## 🛠️ Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

**All imports used in this guide:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

---

## 📚 Further Reading

- [Scikit-learn: Imputation of Missing Values](https://scikit-learn.org/stable/modules/impute.html)
- [Scikit-learn: SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Scikit-learn: KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
- [Scikit-learn: IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)
- [Pandas: Working with Missing Data](https://www.geeksforgeeks.org/data-analysis/working-with-missing-data-in-pandas/)
- [Pandas fillna() documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)
- [Pandas interpolate() documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html)

---

