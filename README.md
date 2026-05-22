# 🛠️  Feature Engineering

---

## What is Feature Engineering?

Feature Engineering is the process of **selecting, creating, or modifying features** (input variables or data) to help machine learning models learn patterns more effectively. It transforms raw, messy real-world data into meaningful inputs that improve model accuracy and performance.

This step may include:
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Creating new features
- Combining existing features

---

## Why is Feature Engineering Important?

Well-engineered features directly influence how well a model performs:

| Benefit | Description |
|---|---|
| ✅ Improve Accuracy | Right features help the model learn better and make more accurate predictions |
| ✅ Reduce Overfitting | Fewer, more important features prevent the model from memorizing training data |
| ✅ Boost Interpretability | Well-chosen features make it easier to understand how the model makes predictions |
| ✅ Enhance Efficiency | Focusing on key features speeds up training and prediction, saving time and resources |

---

## Processes Involved in Feature Engineering

### 1. 🏗️ Feature Creation
Generating new features from domain knowledge or by observing patterns in the data:

- **Domain-specific** — Created based on industry or business rules
- **Data-driven** — Derived by recognizing patterns in data
- **Synthetic** — Formed by combining existing features

### 2. 🔄 Feature Transformation
Adjusting features to improve model learning:

- **Normalization & Scaling** — Adjust the range of features for consistency
- **Encoding** — Convert categorical data to numerical form (e.g., one-hot encoding)
- **Mathematical Transformations** — Logarithmic transformations for skewed data

### 3. 🔍 Feature Extraction
Extracting meaningful features to reduce dimensionality and improve accuracy:

- **Dimensionality Reduction** — Techniques like PCA reduce features while preserving important information
- **Aggregation & Combination** — Summing or averaging features to simplify the model

### 4. 🎯 Feature Selection
Choosing a subset of the most relevant features:

- **Filter Methods** — Based on statistical measures like correlation
- **Wrapper Methods** — Select features based on model performance
- **Embedded Methods** — Feature selection integrated within model training

### 5. 📏 Feature Scaling
Ensuring all features contribute equally to the model:

- **Min-Max Scaling** — Rescales values to a fixed range (e.g., 0 to 1)
- **Standard Scaling** — Normalizes to mean = 0 and variance = 1

---

## Steps in Feature Engineering

```
1. Data Cleaning       →  Fix errors and inconsistencies in the dataset
2. Data Transformation →  Scale, normalize, and encode raw data
3. Feature Extraction  →  Derive new features from existing ones
4. Feature Selection   →  Choose the most relevant features
5. Feature Iteration   →  Refine features based on model performance
```

---

## Common Techniques with Code Examples

### 1. 🔢 One-Hot Encoding

Converts categorical variables into binary indicator columns so ML models can process them.

```python
import pandas as pd

data = {'Color': ['Red', 'Blue', 'Green', 'Blue']}
df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df, columns=['Color'], prefix='Color')
print(df_encoded)
```

**Output:**
```
   Color_Blue  Color_Green  Color_Red
0       False        False       True
1        True        False      False
2       False         True      False
3        True        False      False
```

---

### 2. 📦 Binning

Transforms continuous variables into discrete bins (categories) for easier analysis.

```python
import pandas as pd

data = {'Age': [23, 45, 18, 34, 67, 50, 21]}
df = pd.DataFrame(data)

bins = [0, 20, 40, 60, 100]
labels = ['0-20', '21-40', '41-60', '61+']

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
print(df)
```

**Output:**
```
   Age Age_Group
0   23     21-40
1   45     41-60
2   18      0-20
3   34     21-40
4   67       61+
5   50     41-60
6   21     21-40
```

---

### 3. 📝 Text Data Preprocessing

Prepares raw text for ML by removing stop words, stemming, and vectorizing.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

texts = ["This is a sample sentence.", "Text data preprocessing is important."]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
vectorizer = CountVectorizer()

def preprocess_text(text):
    words = text.split()
    words = [stemmer.stem(word)
             for word in words if word.lower() not in stop_words]
    return " ".join(words)

cleaned_texts = [preprocess_text(text) for text in texts]
X = vectorizer.fit_transform(cleaned_texts)

print("Cleaned Texts:", cleaned_texts)
print("Vectorized Text:", X.toarray())
```

**Steps performed:**
- Tokenizes input text into words
- Removes common stop words (e.g., "is", "a", "the")
- Applies stemming to reduce words to their root form
- Converts cleaned text to a numerical matrix using CountVectorizer

---

### 4. ✂️ Feature Splitting

Divides a single composite feature into multiple sub-features to uncover richer insights.

```python
import pandas as pd

data = {'Full_Address': [
    '123 Elm St, Springfield, 12345',
    '456 Oak Rd, Shelbyville, 67890'
]}
df = pd.DataFrame(data)

df[['Street', 'City', 'Zipcode']] = df['Full_Address'].str.extract(
    r'([0-9]+\s[\w\s]+),\s([\w\s]+),\s(\d+)')

print(df)
```

**Output:**
```
                     Full_Address      Street         City Zipcode
0  123 Elm St, Springfield, 12345  123 Elm St  Springfield   12345
1  456 Oak Rd, Shelbyville, 67890  456 Oak Rd  Shelbyville   67890
```

---

## Popular Feature Engineering Tools

| Tool | Description |
|---|---|
| **Featuretools** | Automates feature engineering from structured data; integrates with pandas and scikit-learn |
| **TPOT** | Uses genetic algorithms to optimize ML pipelines including feature selection |
| **DataRobot** | End-to-end automated ML with feature engineering, model selection, and deployment |
| **Alteryx** | Visual drag-and-drop workflow builder for feature extraction, transformation, and cleaning |
| **H2O.ai** | Supports automated and manual feature engineering with scaling, imputation, and encoding |

---

## Requirements

```bash
pip install pandas numpy scikit-learn nltk featuretools
```

For NLTK stop words:
```python
import nltk
nltk.download('stopwords')
```

---

## Feature Engineering Workflow Summary

```
Raw Data
   ↓
Data Cleaning      (handle nulls, fix errors)
   ↓
Data Transformation (encode, scale, normalize)
   ↓
Feature Extraction  (create new features)
   ↓
Feature Selection   (keep only relevant features)
   ↓
Feature Iteration   (evaluate and refine)
   ↓
Model-Ready Dataset
```

---

## 📚 Related Topics

- [Feature Engineering: Scaling, Normalization & Standardization](https://www.geeksforgeeks.org/machine-learning/feature-engineering-scaling-normalization-and-standardization/)
- [Feature Selection Techniques in ML](https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/)
- [Dimensionality Reduction](https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/)
- [One-Hot Encoding](https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/)
- [Exploratory Data Analysis in Python](https://www.geeksforgeeks.org/data-analysis/exploratory-data-analysis-in-python/)

---
