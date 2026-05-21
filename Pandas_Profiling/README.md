# 📊 Pandas Profiling in Python

A guide to using **Pandas Profiling** — a powerful Python library that automatically generates comprehensive exploratory data analysis (EDA) reports from a Pandas DataFrame in a single line of code.

---

## What is Pandas Profiling?

Pandas Profiling extends the Pandas library to generate detailed, interactive HTML/JSON reports for any dataset. Instead of writing dozens of lines of EDA code, it produces a complete analysis covering statistics, correlations, missing values, and more — automatically.

---

## Installation

```bash
pip install ydata_profiling
```

---

## Requirements

- Python 3.12.x
- pandas
- pandas-profiling

---

## ProfileReport() Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | DataFrame | — | The DataFrame to be analyzed |
| `bins` | int | `10` | Number of bins in histogram |
| `check_correlation` | bool | `True` | Whether to check for correlations |
| `correlation_threshold` | float | `0.9` | Threshold to flag a variable pair as correlated |
| `correlation_overrides` | list | `None` | Variables to exclude from correlation rejection |
| `check_recoded` | bool | `False` | Whether to check recoded correlation (memory heavy) |
| `pool_size` | int | CPU count | Number of workers in the thread pool |

---

## Report Sections

### 🗂 Overview
Three tabs providing a high-level summary of the dataset:

- **Overview** — Total variables, number of observations, duplicates, and missing values
- **Alerts** — Highlights correlated variables, unique values, skewness, and missing data warnings
- **Reproduction** — Report generation timestamps, duration, and software version

### 📌 Variables
Per-column breakdown including:
- Data type (numeric, categorical, etc.)
- Distinct and missing value counts
- Memory usage
- Distribution histogram (for numeric types)
- Value counts (for categorical types)

### 🔗 Correlations
Visual heatmaps for statistical relationships between variables:
- **Pearson Correlation** — Linear relationships between continuous variables
- **Spearman Correlation** — Rank-based (monotonic) relationships

### ❓ Missing Values
Bar chart visualization showing which columns have missing data and how many values are absent.

### 🧾 Sample
Displays the **first 10** and **last 10** rows of the dataset for a quick data sanity check.

---

## Output Files

| Format | Method | Description |
|---|---|---|
| HTML | `profile.to_file("output.html")` | Interactive report viewable in any browser |
| JSON | `profile.to_file("output.json")` | Machine-readable structured report |

---

## Use Cases

- Quick EDA before model building
- Understanding data quality (nulls, duplicates, skewness)
- Sharing dataset summaries with teammates
- Detecting highly correlated features before feature selection

---

## Notes

- For large datasets, profiling can be memory-intensive. Use `minimal=True` for a faster, lighter report:
  ```python
  profile = ProfileReport(data, minimal=True)
  ```
- `check_recoded=True` is computationally expensive — recommended only for small datasets.
- Works best in **Jupyter Notebook** for inline rendering.

---

## License

This guide is based on open content from [GeeksforGeeks](https://www.geeksforgeeks.org/python/pandas-profiling-in-python/) for educational purposes.
