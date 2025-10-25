# datly
A comprehensive JavaScript library for data analysis, statistics, machine learning, and visualization.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Dataframe Operations](#dataframe-operations)
5. [Descriptive Statistics](#descriptive-statistics)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Probability Distributions](#probability-distributions)
8. [Hypothesis Testing](#hypothesis-testing)
9. [Correlation Analysis](#correlation-analysis)
10. [Regression Models](#regression-models)
11. [Classification Models](#classification-models)
12. [Clustering](#clustering)
13. [Ensemble Methods](#ensemble-methods)
14. [Visualization](#visualization)

---

## Introduction

datly is a comprehensive JavaScript library that brings powerful data analysis, statistical testing, machine learning, and visualization capabilities to the browser and Node.js environments.

### Key Features

- **Descriptive Statistics**: Mean, median, variance, standard deviation, skewness, kurtosis
- **Statistical Tests**: t-tests, ANOVA, chi-square, normality tests
- **Machine Learning**: Linear/logistic regression, KNN, decision trees, random forests, Naive Bayes
- **Clustering**: K-means clustering
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
- **Data Visualization**: Histograms, scatter plots, box plots, heatmaps, and more
- **Time Series**: Moving averages, exponential smoothing, autocorrelation

---

## Installation

### Browser (CDN)

```html
<script src="https://unpkg.com/datly"></script>
<script>
  const result = datly.mean([1, 2, 3, 4, 5]);
  console.log(result.value); // Access the mean value directly
</script>
```

### Module Import

```javascript
import * as datly from 'datly';

// All functions return JavaScript objects
const stats = datly.describe([1, 2, 3, 4, 5]);
console.log(stats.mean); // Direct property access
console.log(stats.std);  // No parsing needed
```

> **Note**: All datly functions return JavaScript objects (not strings or YAML). This means you can directly access properties like `result.value`, `result.mean`, `dataframe.columns`, etc.

---

## Core Concepts

### Output Format

All analysis functions return results as JavaScript objects with a consistent structure:

```javascript
{
  type: "statistic",
  name: "mean",
  value: 3,
  n: 5
}
```

This format makes it easy to:
- Access results programmatically with dot notation (e.g., `result.value`)
- Integrate with JavaScript applications
- Serialize to JSON for storage or transmission
- Display results in web interfaces

---

## Dataframe Operations

### `df_from_csv(content, options = {})`

Creates a dataframe from CSV content.

**Parameters:**
- `content`: CSV string content
- `options`:
  - `delimiter`: Column delimiter (default: ',')
  - `header`: First row contains headers (default: true)
  - `skipEmptyLines`: Skip empty lines (default: true)

**Returns:**
```javascript
{
  type: "dataframe",
  columns: ["name", "age", "salary"],
  data: [
    { name: "alice", age: 30, salary: 50000 },
    { name: "bob", age: 25, salary: 45000 }
  ],
  shape: [2, 3]
}
```

**Example:**
```javascript
const csvContent = `name,age,salary
Alice,30,50000
Bob,25,45000
Charlie,35,60000`;

const df = datly.df_from_csv(csvContent);
console.log(df);
```

---

### `df_from_json(input)`

Creates a dataframe from JSON data. Accepts multiple formats:
- Array of objects
- Single object (converted to single-row dataframe)
- Structured JSON with headers and data arrays
- String (parsed as JSON)

**Returns:**
```javascript
{
  type: "dataframe",
  columns: ["name", "age", "department"],
  data: [
    { name: "alice", age: 30, department: "engineering" },
    { name: "bob", age: 25, department: "sales" }
  ],
  shape: [2, 3]
}
```

**Example:**
```javascript
// From array of objects
const data = [
  { name: 'Alice', age: 30, department: 'Engineering' },
  { name: 'Bob', age: 25, department: 'Sales' }
];
const df = datly.df_from_json(data);

// From JSON string
const jsonString = '[{"name":"Alice","age":30},{"name":"Bob","age":25}]';
const df2 = datly.df_from_json(jsonString);

// From structured format
const structured = {
  headers: ['name', 'age'],
  data: [['Alice', 30], ['Bob', 25]]
};
const df3 = datly.df_from_json(structured);
```

---

### `df_from_array(array)`

Creates a dataframe from an array of objects.

**Parameters:**
- `array`: Array of objects with consistent keys

**Returns:**
```javascript
{
  type: "dataframe",
  columns: ["product", "price", "stock"],
  data: [
    { product: "laptop", price: 999, stock: 15 },
    { product: "mouse", price: 25, stock: 50 }
  ],
  shape: [2, 3]
}
```

**Example:**
```javascript
const products = [
  { product: 'Laptop', price: 999, stock: 15 },
  { product: 'Mouse', price: 25, stock: 50 },
  { product: 'Keyboard', price: 75, stock: 30 }
];

const df = datly.df_from_array(products);
```

---

### `df_from_object(object, options = {})`

Creates a dataframe from a single object. Can flatten nested structures.

**Parameters:**
- `object`: JavaScript object
- `options`:
  - `flatten`: Flatten nested objects (default: true)
  - `maxDepth`: Maximum depth for flattening (default: 10)

**Returns (flattened):**
```javascript
{
  type: "dataframe",
  columns: [
    "user.name", "user.age", "user.address.city",
    "user.address.country", "orders"
  ],
  data: [
    {
      "user.name": "alice",
      "user.age": 30,
      "user.address.city": "new york",
      "user.address.country": "usa",
      "orders": [
        { id: 1, total: 150 },
        { id: 2, total: 200 }
      ]
    }
  ],
  shape: [1, 5]
}
```

**Example:**
```javascript
// Flattened (default)
const user = {
  name: 'Alice',
  age: 30,
  address: {
    city: 'New York',
    country: 'USA'
  },
  orders: [
    { id: 1, total: 150 },
    { id: 2, total: 200 }
  ]
};

const df = datly.df_from_object(user);
// Flattened columns: name, age, address.city, address.country, etc.

// Non-flattened (key-value pairs)
const df2 = datly.df_from_object(user, { flatten: false });
```

---

## Basic Operations

### `df_get_column(dataframe, column)`

Extracts a single column as an array.

**Returns:**
```javascript
[30, 25, 35] // Array of values
```

**Example:**
```javascript
const df = datly.df_from_json([
  { name: 'Alice', age: 30 },
  { name: 'Bob', age: 25 },
  { name: 'Charlie', age: 35 }
]);

const ages = datly.df_get_column(df, 'age');
console.log(ages); // [30, 25, 35]
```

---

### `df_get_value(dataframe, column)`

Gets the first value from a column. Useful for single-row dataframes.

**Returns:**
```javascript
30 // Single value
```

**Example:**
```javascript
const userObj = { name: 'Alice', age: 30, city: 'NYC' };
const df = datly.df_from_object(userObj);

const age = datly.df_get_value(df, 'age');
console.log(age); // 30
```

---

### `df_get_columns(dataframe, columns)`

Extracts multiple columns as an object of arrays.

**Returns:**
```javascript
{
  name: ['Alice', 'Bob', 'Charlie'],
  age: [30, 25, 35]
}
```

**Example:**
```javascript
const df = datly.df_from_json([
  { name: 'Alice', age: 30, salary: 50000 },
  { name: 'Bob', age: 25, salary: 45000 }
]);

const subset = datly.df_get_columns(df, ['name', 'age']);
console.log(subset);
```

---

### `df_head(dataframe, n = 5)`

Returns the first n rows.

**Returns:**
```javascript
{
  type: "dataframe",
  columns: ["name", "age"],
  data: [
    { name: "alice", age: 30 },
    { name: "bob", age: 25 }
  ],
  shape: [2, 2]
}
```

**Example:**
```javascript
const df = datly.df_from_json([...largeDataset]);
const first3 = datly.df_head(df, 3);
```

---

### `df_tail(dataframe, n = 5)`

Returns the last n rows.

**Example:**
```javascript
const df = datly.df_from_json([...largeDataset]);
const last3 = datly.df_tail(df, 3);
```

---

## Descriptive Statistics

### Basic Statistical Functions

All statistical functions return JavaScript objects with consistent structure.

#### `mean(array)`

Calculates the arithmetic mean.

**Returns:**
```javascript
{
  type: "statistic",
  name: "mean",
  value: 3,
  n: 5
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.mean(data);
console.log(result.value); // 3
```

#### `median(array)`

Calculates the median value.

**Returns:**
```javascript
{
  type: "statistic",
  name: "median",
  value: 3,
  n: 5
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.median(data);
console.log(result.value); // 3
```

#### `variance(array)`

Calculates the sample variance.

**Returns:**
```javascript
{
  type: "statistic",
  name: "variance",
  value: 2.5,
  n: 5
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.variance(data);
console.log(result.value); // 2.5
```

#### `std(array)`

Calculates the sample standard deviation.

**Returns:**
```javascript
{
  type: "statistic",
  name: "standard_deviation",
  value: 1.58,
  n: 5
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.std(data);
console.log(result.value); // 1.58
```

#### `skewness(array)`

Calculates the skewness (asymmetry measure).

**Returns:**
```javascript
{
  type: "statistic",
  name: "skewness",
  value: 0,
  n: 5,
  interpretation: "symmetric"
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.skewness(data);
console.log(result.interpretation); // "symmetric"
```

#### `kurtosis(array)`

Calculates the kurtosis (tail heaviness measure).

**Returns:**
```javascript
{
  type: "statistic",
  name: "kurtosis",
  value: -1.2,
  n: 5,
  interpretation: "platykurtic"
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.kurtosis(data);
console.log(result.interpretation); // "platykurtic"
```

#### `percentile(array, p)`

Calculates the p-th percentile.

**Parameters:**
- `array`: Array of numbers
- `p`: Percentile (0-100)

**Returns:**
```javascript
{
  type: "statistic",
  name: "percentile",
  percentile: 75,
  value: 4,
  n: 5
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.percentile(data, 75);
console.log(result.value); // 4
```

#### `quantile(array, q)`

Calculates the q-th quantile.

**Parameters:**
- `array`: Array of numbers
- `q`: Quantile (0-1)

**Example:**
```javascript
const data = [1, 2, 3, 4, 5];
const result = datly.quantile(data, 0.75);
console.log(result.value); // 4
```

#### `describe(array)`

Provides comprehensive descriptive statistics.

**Returns:**
```javascript
{
  type: "descriptive_statistics",
  n: 5,
  mean: 3,
  median: 3,
  std: 1.58,
  variance: 2.5,
  min: 1,
  max: 5,
  q1: 2,
  q3: 4,
  iqr: 2,
  skewness: 0,
  kurtosis: -1.2
}
```

**Example:**
```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const result = datly.describe(data);
console.log(result.mean); // Access mean directly
console.log(result.std);  // Access standard deviation
```

---

## Exploratory Data Analysis

### `eda_overview(data)`

Provides a comprehensive overview of a dataset.

**Parameters:**
- `data`: Array of objects or 2D array

**Returns:**
```javascript
{
  type: "eda_overview",
  n_observations: 100,
  n_variables: 5,
  variables: [
    {
      name: "age",
      type: "numeric",
      missing: 0,
      unique: 25,
      mean: 35.5,
      std: 12.3
    },
    {
      name: "department",
      type: "categorical",
      missing: 2,
      unique: 4,
      mode: "engineering",
      frequency: 45
    }
  ],
  memory_usage: "2.1kb"
}
```

**Example:**
```javascript
const employees = [
  { name: 'Alice', age: 30, salary: 50000, department: 'Engineering' },
  { name: 'Bob', age: 25, salary: 45000, department: 'Sales' },
  { name: 'Charlie', age: 35, salary: 60000, department: 'Engineering' }
];

const overview = datly.eda_overview(employees);
console.log(overview);
```

### `missing_values(data)`

Analyzes missing values in the dataset.

**Returns:**
```javascript
{
  type: "missing_values_analysis",
  total_missing: 15,
  missing_percentage: 7.5,
  variables: [
    { name: "age", missing: 0, percentage: 0 },
    { name: "salary", missing: 5, percentage: 25 },
    { name: "department", missing: 10, percentage: 50 }
  ]
}
```

**Example:**
```javascript
const data = [
  { age: 30, salary: 50000, department: 'Engineering' },
  { age: null, salary: 45000, department: null },
  { age: 35, salary: null, department: 'Engineering' }
];

const missing = datly.missing_values(data);
console.log(missing);
```

### `outliers_zscore(array, threshold = 3)`

Detects outliers using Z-score method.

**Parameters:**
- `array`: Array of numbers
- `threshold`: Z-score threshold (default: 3)

**Returns:**
```javascript
{
  type: "outlier_detection",
  method: "zscore",
  threshold: 3,
  n_outliers: 2,
  outlier_indices: [5, 12],
  outlier_values: [200, 30]
}
```

**Example:**
```javascript
const data = [10, 12, 14, 15, 16, 200, 18, 19, 20, 21, 22, 23, 30];
const outliers = datly.outliers_zscore(data, 3);
console.log(outliers);
```

---

## Probability Distributions

### Normal Distribution

#### `normal_pdf(x, mean = 0, std = 1)`

Calculates the probability density function of the normal distribution.

**Returns:**
```javascript
{
  type: "probability_density",
  distribution: "normal",
  x: 0,
  mean: 0,
  std: 1,
  pdf: 0.399
}
```

**Example:**
```javascript
const pdf = datly.normal_pdf(0, 0, 1);
console.log(pdf.pdf); // 0.399
```

#### `normal_cdf(x, mean = 0, std = 1)`

Calculates the cumulative distribution function.

**Returns:**
```javascript
{
  type: "cumulative_probability",
  distribution: "normal",
  x: 0,
  mean: 0,
  std: 1,
  cdf: 0.5
}
```

**Example:**
```javascript
const cdf = datly.normal_cdf(1.96, 0, 1);
console.log(cdf.cdf); // ~0.975
```

### Random Sampling

#### `random_normal(n, mean = 0, std = 1, seed = null)`

Generates random samples from a normal distribution.

**Parameters:**
- `n`: Number of samples
- `mean`: Mean of the distribution
- `std`: Standard deviation
- `seed`: Random seed for reproducibility

**Returns:**
```javascript
{
  type: "random_sample",
  distribution: "normal",
  n: 100,
  mean: 0,
  std: 1,
  seed: 42,
  sample: [0.674, -0.423, 1.764, ...],
  sample_mean: 0.054,
  sample_std: 0.986
}
```

**Example:**
```javascript
const samples = datly.random_normal(100, 0, 1, 42);
console.log(samples.sample.length); // 100
console.log(samples.sample_mean);   // ~0.054
```

---

## Hypothesis Testing

### T-Tests

#### `ttest_1samp(array, popmean)`

One-sample t-test.

**Parameters:**
- `array`: Sample data
- `popmean`: Population mean to test against

**Returns:**
```javascript
{
  type: "hypothesis_test",
  test: "one_sample_ttest",
  n: 20,
  sample_mean: 5.2,
  population_mean: 5.0,
  t_statistic: 1.89,
  p_value: 0.074,
  degrees_of_freedom: 19,
  confidence_interval: [4.87, 5.53],
  conclusion: "fail_to_reject_h0",
  alpha: 0.05
}
```

**Example:**
```javascript
const sample = [4.8, 5.1, 5.3, 4.9, 5.2, 5.0, 5.4, 4.7, 5.1, 5.0];
const result = datly.ttest_1samp(sample, 5.0);
console.log(result.p_value);    // 0.074
console.log(result.conclusion); // "fail_to_reject_h0"
```

#### `ttest_ind(array1, array2)`

Independent two-sample t-test.

**Returns:**
```javascript
{
  type: "hypothesis_test",
  test: "independent_ttest",
  n1: 15,
  n2: 18,
  mean1: 5.2,
  mean2: 4.8,
  t_statistic: 2.45,
  p_value: 0.019,
  degrees_of_freedom: 31,
  confidence_interval: [0.067, 0.733],
  conclusion: "reject_h0",
  alpha: 0.05
}
```

**Example:**
```javascript
const group1 = [5.1, 5.3, 4.9, 5.2, 5.0];
const group2 = [4.8, 4.6, 4.9, 4.7, 4.5];
const result = datly.ttest_ind(group1, group2);
console.log(result.p_value < 0.05); // true (significant difference)
```

### ANOVA

#### `anova_oneway(groups)`

One-way ANOVA test.

**Parameters:**
- `groups`: Array of arrays, each representing a group

**Returns:**
```javascript
{
  type: "hypothesis_test",
  test: "one_way_anova",
  n_groups: 3,
  total_n: 45,
  f_statistic: 8.76,
  p_value: 0.001,
  between_groups_df: 2,
  within_groups_df: 42,
  total_df: 44,
  between_groups_ss: 125.4,
  within_groups_ss: 301.2,
  total_ss: 426.6,
  conclusion: "reject_h0",
  alpha: 0.05
}
```

**Example:**
```javascript
const group1 = [23, 25, 28, 30, 32];
const group2 = [18, 20, 22, 24, 26];
const group3 = [15, 17, 19, 21, 23];

const result = datly.anova_oneway([group1, group2, group3]);
console.log(result);
```

### Normality Tests

#### `shapiro_wilk(array)`

Shapiro-Wilk test for normality.

**Returns:**
```yaml
type: hypothesis_test
test: shapiro_wilk
n: 50
w_statistic: 0.973
p_value: 0.284
conclusion: fail_to_reject_h0
interpretation: data_appears_normal
alpha: 0.05
```

**Example:**
```javascript
const data = datly.random_normal(50, 0, 1, 42);
const parsedData = JSON.parse(data).sample;
const result = datly.shapiro_wilk(parsedData);
console.log(result);
```

---

## Correlation Analysis

### `correlation(x, y, method = 'pearson')`

Calculates correlation between two variables.

**Parameters:**
- `x`: First variable array
- `y`: Second variable array
- `method`: 'pearson', 'spearman', or 'kendall'

**Returns:**
```yaml
type: correlation
method: pearson
correlation: 0.87
n: 20
p_value: 0.001
confidence_interval:
  - 0.68
  - 0.95
interpretation: strong_positive
```

**Example:**
```javascript
const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

const result = datly.correlation(x, y, 'pearson');
console.log(result);
```

### `df_corr(dataframe, method = 'pearson')`

Calculates correlation matrix for a dataframe.

**Returns:**
```yaml
type: correlation_matrix
method: pearson
variables:
  - age
  - salary
  - experience
matrix:
  - - 1.000
    - 0.856
    - 0.923
  - - 0.856
    - 1.000
    - 0.789
  - - 0.923
    - 0.789
    - 1.000
```

**Example:**
```javascript
const employees = [
  { age: 25, salary: 50000, experience: 2 },
  { age: 30, salary: 60000, experience: 5 },
  { age: 35, salary: 70000, experience: 8 },
  { age: 40, salary: 80000, experience: 12 }
];

const corrMatrix = datly.df_corr(employees, 'pearson');
console.log(corrMatrix);
```

---

## Regression Models

### Linear Regression

#### `train_linear_regression(X, y)`

Trains a linear regression model.

**Parameters:**
- `X`: Feature matrix (2D array)
- `y`: Target vector (1D array)

**Returns:**
```yaml
type: model
algorithm: linear_regression
n_features: 2
n_samples: 100
coefficients:
  - 2.45
  - -1.23
intercept: 0.67
r_squared: 0.78
mse: 15.4
training_score: 0.78
```

**Example:**
```javascript
const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
const y = [3, 5, 7, 9, 11];

const model = datly.train_linear_regression(X, y);
console.log(model);
```

#### `predict_linear(model, X)`

Makes predictions using a trained linear regression model.

**Returns:**
```yaml
type: predictions
algorithm: linear_regression
n_predictions: 5
predictions:
  - 3.12
  - 5.57
  - 7.02
  - 9.47
  - 11.92
```

**Example:**
```javascript
const X_test = [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];
const predictions = datly.predict_linear(model, X_test);
console.log(predictions);
```

### Logistic Regression

#### `train_logistic_regression(X, y, options = {})`

Trains a logistic regression model for binary classification.

**Parameters:**
- `X`: Feature matrix
- `y`: Binary target vector (0s and 1s)
- `options`: Training options (learning_rate, max_iterations, tolerance)

**Returns:**
```yaml
type: model
algorithm: logistic_regression
n_features: 2
n_samples: 100
coefficients:
  - 1.45
  - -0.89
intercept: 0.23
accuracy: 0.85
log_likelihood: -45.6
iterations: 150
converged: true
```

**Example:**
```javascript
const X = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5]];
const y = [0, 0, 1, 1, 1, 1];

const options = {
  learning_rate: 0.01,
  max_iterations: 1000,
  tolerance: 1e-6
};

const model = datly.train_logistic_regression(X, y, options);
console.log(model);
```

#### `predict_logistic(model, X)`

Makes predictions using a trained logistic regression model.

**Returns:**
```yaml
type: predictions
algorithm: logistic_regression
n_predictions: 3
predictions:
  - 0
  - 1
  - 1
probabilities:
  - 0.23
  - 0.78
  - 0.85
```

**Example:**
```javascript
const X_test = [[2, 3], [4, 5], [6, 7]];
const predictions = datly.predict_logistic(model, X_test);
console.log(predictions);
```

---

## Classification Models

### K-Nearest Neighbors (KNN)

#### `train_knn(X, y, k = 3)`

Trains a KNN classifier.

**Parameters:**
- `X`: Feature matrix
- `y`: Target vector
- `k`: Number of neighbors (default: 3)

**Returns:**
```yaml
type: model
algorithm: knn
k: 3
n_features: 2
n_samples: 100
classes:
  - 0
  - 1
  - 2
training_accuracy: 0.92
```

**Example:**
```javascript
const X = [[1, 2], [2, 3], [3, 1], [1, 3], [2, 1], [3, 2]];
const y = [0, 0, 1, 1, 2, 2];

const model = datly.train_knn(X, y, 3);
console.log(model);
```

#### `predict_knn(model, X)`

Makes predictions using a trained KNN model.

**Returns:**
```yaml
type: predictions
algorithm: knn
k: 3
n_predictions: 2
predictions:
  - 1
  - 0
distances:
  - - 1.41
    - 2.24
    - 1.00
  - - 1.00
    - 1.41
    - 2.83
```

**Example:**
```javascript
const X_test = [[2.5, 2], [1.5, 2.5]];
const predictions = datly.predict_knn(model, X_test);
console.log(predictions);
```

### Decision Tree

#### `train_decision_tree(X, y, options = {})`

Trains a decision tree classifier.

**Parameters:**
- `X`: Feature matrix
- `y`: Target vector
- `options`: Tree options (max_depth, min_samples_split, min_samples_leaf)

**Returns:**
```yaml
type: model
algorithm: decision_tree
max_depth: 5
n_features: 4
n_samples: 150
classes:
  - 0
  - 1
  - 2
tree_depth: 3
n_nodes: 7
feature_importance:
  - 0.45
  - 0.32
  - 0.15
  - 0.08
training_accuracy: 0.96
```

**Example:**
```javascript
const X = [
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3.0, 1.4, 0.2],
  [7.0, 3.2, 4.7, 1.4],
  [6.4, 3.2, 4.5, 1.5]
];
const y = [0, 0, 1, 1];

const options = {
  max_depth: 5,
  min_samples_split: 2,
  min_samples_leaf: 1
};

const model = datly.train_decision_tree(X, y, options);
console.log(model);
```

### Naive Bayes

#### `train_naive_bayes(X, y)`

Trains a Gaussian Naive Bayes classifier.

**Returns:**
```yaml
type: model
algorithm: naive_bayes
variant: gaussian
n_features: 4
n_samples: 150
classes:
  - 0
  - 1
  - 2
class_priors:
  - 0.33
  - 0.33
  - 0.34
training_accuracy: 0.94
```

**Example:**
```javascript
const X = [
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3.0, 1.4, 0.2],
  [7.0, 3.2, 4.7, 1.4],
  [6.4, 3.2, 4.5, 1.5]
];
const y = [0, 0, 1, 1];

const model = datly.train_naive_bayes(X, y);
console.log(model);
```

---

## Clustering

### K-Means Clustering

#### `kmeans(X, k, options = {})`

Performs K-means clustering.

**Parameters:**
- `X`: Data matrix
- `k`: Number of clusters
- `options`: Algorithm options (max_iterations, tolerance, seed)

**Returns:**
```yaml
type: clustering_result
algorithm: kmeans
k: 3
n_samples: 100
n_features: 2
iterations: 15
converged: true
inertia: 45.7
centroids:
  - - 2.1
    - 3.2
  - - 5.8
    - 1.4
  - - 8.3
    - 6.7
labels:
  - 0
  - 0
  - 1
  - 2
  - 1
```

**Example:**
```javascript
const X = [
  [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
];

const options = {
  max_iterations: 100,
  tolerance: 1e-4,
  seed: 42
};

const result = datly.kmeans(X, 3, options);
console.log(result);
```

---

## Ensemble Methods

### Random Forest

#### `train_random_forest(X, y, options = {})`

Trains a random forest classifier.

**Parameters:**
- `X`: Feature matrix
- `y`: Target vector
- `options`: Forest options (n_trees, max_depth, max_features, sample_ratio)

**Returns:**
```yaml
type: model
algorithm: random_forest
n_trees: 100
max_depth: 10
n_features: 4
n_samples: 150
classes:
  - 0
  - 1
  - 2
oob_score: 0.91
feature_importance:
  - 0.35
  - 0.28
  - 0.22
  - 0.15
training_accuracy: 0.98
```

**Example:**
```javascript
const X = [
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3.0, 1.4, 0.2],
  [7.0, 3.2, 4.7, 1.4],
  [6.4, 3.2, 4.5, 1.5]
];
const y = [0, 0, 1, 1];

const options = {
  n_trees: 100,
  max_depth: 10,
  max_features: 'sqrt',
  sample_ratio: 0.8
};

const model = datly.train_random_forest(X, y, options);
console.log(model);
```

---

## Model Evaluation and Utilities

### Data Splitting

#### `train_test_split(X, y, test_size = 0.2, seed = null)`

Splits data into training and testing sets.

**Returns:**
```yaml
type: data_split
train_size: 0.8
test_size: 0.2
n_samples: 100
n_train: 80
n_test: 20
seed: 42
indices:
  train:
    - 0
    - 3
    - 5
    # ... more indices
  test:
    - 1
    - 2
    - 4
    # ... more indices
```

**Example:**
```javascript
const X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]];
const y = [0, 1, 0, 1, 0];

const split = datly.train_test_split(X, y, 0.2, 42);
console.log(split);

// Use indices to create splits
const trainIndices = JSON.parse(split).indices.train;
const testIndices = JSON.parse(split).indices.test;

const X_train = trainIndices.map(i => X[i]);
const y_train = trainIndices.map(i => y[i]);
const X_test = testIndices.map(i => X[i]);
const y_test = testIndices.map(i => y[i]);
```

### Feature Scaling

#### `standard_scaler_fit(X)`

Fits a standard scaler to the data.

**Returns:**
```yaml
type: scaler
method: standard
n_features: 3
n_samples: 100
means:
  - 2.5
  - 15.3
  - 0.8
stds:
  - 1.2
  - 5.6
  - 0.3
```

**Example:**
```javascript
const X = [[1, 10, 0.5], [2, 15, 0.7], [3, 20, 0.9], [4, 25, 1.1]];
const scaler = datly.standard_scaler_fit(X);
console.log(scaler);
```

#### `standard_scaler_transform(scaler, X)`

Transforms data using a fitted scaler.

**Returns:**
```yaml
type: scaled_data
method: standard
n_samples: 4
n_features: 3
preview:
  - - -1.34
    - -0.89
    - -1.00
  - - -0.45
    - -0.07
    - -0.33
  - - 0.45
    - 0.75
    - 0.33
  - - 1.34
    - 1.21
    - 1.00
```

**Example:**
```javascript
const X_scaled = datly.standard_scaler_transform(scaler, X);
console.log(X_scaled);
```

### Model Metrics

#### `metrics_classification(y_true, y_pred)`

Calculates classification metrics.

**Returns:**
```yaml
type: classification_metrics
accuracy: 0.85
precision: 0.83
recall: 0.87
f1_score: 0.85
confusion_matrix:
  - - 25
    - 3
  - - 5
    - 27
support:
  - 28
  - 32
```

**Example:**
```javascript
const y_true = [0, 0, 1, 1, 0, 1, 1, 0];
const y_pred = [0, 1, 1, 1, 0, 1, 0, 0];

const metrics = datly.metrics_classification(y_true, y_pred);
console.log(metrics);
```

#### `metrics_regression(y_true, y_pred)`

Calculates regression metrics.

**Returns:**
```yaml
type: regression_metrics
mae: 2.15
mse: 6.78
rmse: 2.60
r2: 0.78
explained_variance: 0.79
```

**Example:**
```javascript
const y_true = [3, -0.5, 2, 7];
const y_pred = [2.5, 0.0, 2, 8];

const metrics = datly.metrics_regression(y_true, y_pred);
console.log(metrics);
```

---

## Visualization

All visualization functions create SVG-based charts that can be rendered in the browser. They accept optional configuration and a selector for where to render the chart.

### Configuration Options

Common options for all plots:
- `width`: Chart width in pixels (default: 400)
- `height`: Chart height in pixels (default: 400)
- `color`: Primary color (default: '#000')
- `background`: Background color (default: '#fff')
- `title`: Chart title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label

### `plotHistogram(array, options = {}, selector)`

Creates a histogram showing the distribution of values.

**Additional Options:**
- `bins`: Number of bins (default: 10)

**Example:**
```javascript
const data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5];
datly.plotHistogram(data, {
  width: 600,
  height: 400,
  bins: 8,
  title: 'Value Distribution',
  xlabel: 'Values',
  ylabel: 'Frequency',
  color: '#4CAF50'
}, '#chart-container');
```

### `plotScatter(x, y, options = {}, selector)`

Creates a scatter plot showing the relationship between two variables.

**Additional Options:**
- `size`: Point size (default: 4)

**Example:**
```javascript
const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const y = [2, 4, 3, 5, 6, 8, 7, 9, 8, 10];
datly.plotScatter(x, y, {
  width: 600,
  height: 400,
  title: 'Correlation Analysis',
  xlabel: 'X Variable',
  ylabel: 'Y Variable',
  size: 6,
  color: '#2196F3'
}, '#scatter-plot');
```

### `plotLine(x, y, options = {}, selector)`

Creates a line chart for time series or continuous data.

**Additional Options:**
- `lineWidth`: Line width (default: 2)
- `showPoints`: Show data points (default: false)

**Example:**
```javascript
const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const sales = [100, 120, 140, 110, 160, 180, 200, 190, 220, 240, 260, 280];
datly.plotLine(months, sales, {
  width: 800,
  height: 400,
  lineWidth: 3,
  showPoints: true,
  title: 'Monthly Sales Trend',
  xlabel: 'Month',
  ylabel: 'Sales ($000)',
  color: '#FF5722'
}, '#line-chart');
```

### `plotBar(categories, values, options = {}, selector)`

Creates a bar chart for categorical data.

**Example:**
```javascript
const categories = ['Q1', 'Q2', 'Q3', 'Q4'];
const revenues = [120, 150, 180, 200];
datly.plotBar(categories, revenues, {
  width: 600,
  height: 400,
  title: 'Quarterly Revenue',
  xlabel: 'Quarter',
  ylabel: 'Revenue ($M)',
  color: '#9C27B0'
}, '#bar-chart');
```

### `plotBoxplot(data, options = {}, selector)`

Creates box plots showing distribution statistics for one or more groups.

**Parameters:**
- `data`: Array of arrays (each array is a group) or single array
- `options`:
  - `labels`: Array of group labels

**Example:**
```javascript
const group1 = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const group2 = [2, 3, 4, 5, 6, 7, 8, 9, 10];
const group3 = [3, 4, 5, 6, 7, 8, 9, 10, 11];

datly.plotBoxplot([group1, group2, group3], {
  labels: ['Control', 'Treatment A', 'Treatment B'],
  title: 'Treatment Comparison',
  ylabel: 'Response Value',
  width: 600,
  height: 400
}, '#boxplot');
```

### `plotPie(labels, values, options = {}, selector)`

Creates a pie chart for proportional data.

**Additional Options:**
- `showLabels`: Display labels (default: true)

**Example:**
```javascript
const categories = ['Desktop', 'Mobile', 'Tablet'];
const usage = [45, 40, 15];
datly.plotPie(categories, usage, {
  width: 500,
  height: 500,
  title: 'Device Usage Distribution',
  showLabels: true
}, '#pie-chart');
```

### `plotHeatmap(matrix, options = {}, selector)`

Creates a heatmap visualization for correlation matrices or 2D data.

**Additional Options:**
- `labels`: Array of variable names
- `showValues`: Display correlation values (default: true)

**Example:**
```javascript
const corrMatrix = [
  [1.0, 0.8, 0.3, 0.1],
  [0.8, 1.0, 0.5, 0.2],
  [0.3, 0.5, 1.0, 0.7],
  [0.1, 0.2, 0.7, 1.0]
];

datly.plotHeatmap(corrMatrix, {
  labels: ['Age', 'Income', 'Education', 'Experience'],
  showValues: true,
  title: 'Correlation Matrix',
  width: 500,
  height: 500
}, '#heatmap');
```

### `plotViolin(data, options = {}, selector)`

Creates violin plots showing distribution density for multiple groups.

**Parameters:**
- `data`: Array of arrays or single array
- `options`:
  - `labels`: Group labels

**Example:**
```javascript
const before = [5.1, 5.3, 4.9, 5.2, 5.0, 4.8, 5.1, 5.4];
const after = [5.8, 6.1, 5.9, 6.2, 6.0, 5.7, 6.0, 6.3];

datly.plotViolin([before, after], {
  labels: ['Before Treatment', 'After Treatment'],
  title: 'Treatment Effect Distribution',
  ylabel: 'Measurement',
  width: 600,
  height: 400
}, '#violin-plot');
```

### `plotDensity(array, options = {}, selector)`

Creates a kernel density plot showing the probability density function.

**Additional Options:**
- `bandwidth`: Smoothing bandwidth (default: 5)

**Example:**
```javascript
const data = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7];
datly.plotDensity(data, {
  bandwidth: 0.5,
  title: 'Data Distribution (Kernel Density)',
  xlabel: 'Values',
  ylabel: 'Density',
  width: 600,
  height: 400
}, '#density-plot');
```

### `plotQQ(array, options = {}, selector)`

Creates a Q-Q plot for assessing normality of data.

**Example:**
```javascript
const data = [1.2, 2.3, 1.8, 2.1, 1.9, 2.0, 2.4, 1.7, 2.2, 1.6];
datly.plotQQ(data, {
  title: 'Q-Q Plot for Normality Check',
  xlabel: 'Theoretical Quantiles',
  ylabel: 'Sample Quantiles',
  width: 500,
  height: 500
}, '#qq-plot');
```

### `plotParallel(data, columns, options = {}, selector)`

Creates a parallel coordinates plot for multivariate data visualization.

**Parameters:**
- `data`: Array of objects
- `columns`: Array of column names to include
- `options`:
  - `colors`: Array of colors for each observation

**Example:**
```javascript
const employees = [
  { age: 25, salary: 50000, experience: 2, satisfaction: 7 },
  { age: 30, salary: 60000, experience: 5, satisfaction: 8 },
  { age: 35, salary: 70000, experience: 8, satisfaction: 6 },
  { age: 40, salary: 80000, experience: 12, satisfaction: 9 }
];

datly.plotParallel(employees, ['age', 'salary', 'experience', 'satisfaction'], {
  title: 'Employee Profile Analysis',
  width: 800,
  height: 400
}, '#parallel-plot');
```

### `plotPairplot(data, columns, options = {}, selector)`

Creates a pairplot matrix showing all pairwise relationships between variables.

**Parameters:**
- `data`: Array of objects
- `columns`: Array of column names
- `options`:
  - `size`: Size of each subplot (default: 120)
  - `color`: Point color

**Example:**
```javascript
const iris = [
  { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2 },
  { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2 },
  { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4 },
  { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5 }
];

datly.plotPairplot(iris, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], {
  size: 150,
  color: '#E91E63'
}, '#pairplot');
```

### `plotMultiline(series, options = {}, selector)`

Creates a multi-line chart for comparing multiple time series.

**Parameters:**
- `series`: Array of objects with `name` and `data` properties
  - `data`: Array of `{x, y}` objects
- `options`:
  - `legend`: Show legend (default: false)

**Example:**
```javascript
const timeSeries = [
  {
    name: 'Product A',
    data: [{x: 1, y: 10}, {x: 2, y: 15}, {x: 3, y: 12}, {x: 4, y: 18}]
  },
  {
    name: 'Product B',
    data: [{x: 1, y: 8}, {x: 2, y: 12}, {x: 3, y: 16}, {x: 4, y: 14}]
  },
  {
    name: 'Product C',
    data: [{x: 1, y: 12}, {x: 2, y: 9}, {x: 3, y: 14}, {x: 4, y: 16}]
  }
];

datly.plotMultiline(timeSeries, {
  legend: true,
  title: 'Product Sales Comparison',
  xlabel: 'Quarter',
  ylabel: 'Sales (Units)',
  width: 700,
  height: 400
}, '#multiline-chart');
```

---

## Complete Example Workflow

Here's a comprehensive example demonstrating a typical data analysis workflow using datly:

```javascript
// 1. Load and explore data
const employeeData = [
  { age: 25, salary: 50000, experience: 2, department: 'IT', performance: 85 },
  { age: 30, salary: 60000, experience: 5, department: 'HR', performance: 90 },
  { age: 35, salary: 70000, experience: 8, department: 'IT', performance: 88 },
  { age: 28, salary: 55000, experience: 3, department: 'Sales', performance: 82 },
  { age: 42, salary: 85000, experience: 15, department: 'IT', performance: 95 },
  { age: 31, salary: 62000, experience: 6, department: 'HR', performance: 87 },
  { age: 26, salary: 48000, experience: 1, department: 'Sales', performance: 78 },
  { age: 38, salary: 75000, experience: 12, department: 'IT', performance: 92 }
];

// 2. Perform exploratory data analysis
const overview = datly.eda_overview(employeeData);
console.log('Dataset Overview:', overview);

// 3. Calculate descriptive statistics for salary
const salaries = employeeData.map(emp => emp.salary);
const salaryStats = datly.describe(salaries);
console.log('Salary Statistics:', salaryStats);

// 4. Check correlations between numeric variables
const correlations = datly.df_corr(employeeData, 'pearson');
console.log('Correlation Matrix:', correlations);

// 5. Visualize salary distribution
datly.plotHistogram(salaries, {
  title: 'Salary Distribution',
  xlabel: 'Salary ($)',
  ylabel: 'Frequency',
  bins: 6,
  color: '#2196F3'
}, '#salary-histogram');

// 6. Analyze relationship between experience and salary
const experience = employeeData.map(emp => emp.experience);
datly.plotScatter(experience, salaries, {
  title: 'Experience vs Salary',
  xlabel: 'Years of Experience',
  ylabel: 'Salary ($)',
  color: '#4CAF50'
}, '#experience-salary-scatter');

// 7. Prepare data for machine learning
const X = employeeData.map(emp => [emp.age, emp.experience]);
const y = salaries;

// 8. Split data into training and testing sets
const split = datly.train_test_split(X, y, 0.3, 42);
const trainIndices = split.indices.train;
const testIndices = split.indices.test;

const X_train = trainIndices.map(i => X[i]);
const y_train = trainIndices.map(i => y[i]);
const X_test = testIndices.map(i => X[i]);
const y_test = testIndices.map(i => y[i]);

// 9. Scale features for better model performance
const scaler = datly.standard_scaler_fit(X_train);
const X_train_scaled = datly.standard_scaler_transform(scaler, X_train);
const X_test_scaled = datly.standard_scaler_transform(scaler, X_test);

// 10. Train linear regression model
const model = datly.train_linear_regression(X_train_scaled.data, y_train);
console.log('Linear Regression Model:', model);

// 11. Make predictions
const predictions = datly.predict_linear(model, X_test_scaled.data);
console.log('Predictions:', predictions);

// 12. Evaluate model performance
const metrics = datly.metrics_regression(y_test, predictions.predictions);
console.log('Model Performance:', metrics);

// 13. Visualize actual vs predicted values
datly.plotScatter(y_test, predictions.predictions, {
  title: 'Actual vs Predicted Salaries',
  xlabel: 'Actual Salary ($)',
  ylabel: 'Predicted Salary ($)',
  color: '#FF5722'
}, '#prediction-scatter');

// 14. Compare salary distributions by department
const departments = ['IT', 'HR', 'Sales'];
const deptSalaries = departments.map(dept =>
  employeeData.filter(emp => emp.department === dept).map(emp => emp.salary)
);

datly.plotBoxplot(deptSalaries, {
  labels: departments,
  title: 'Salary Distribution by Department',
  ylabel: 'Salary ($)',
  width: 600,
  height: 400
}, '#department-boxplot');

// 15. Perform clustering analysis
const clusterData = employeeData.map(emp => [emp.age, emp.salary / 1000]); // Normalize salary
const clusterResult = datly.kmeans(clusterData, 3, { seed: 42 });
console.log('Clustering Results:', clusterResult);

// 16. Test for salary differences between departments
const itSalaries = employeeData.filter(emp => emp.department === 'IT').map(emp => emp.salary);
const hrSalaries = employeeData.filter(emp => emp.department === 'HR').map(emp => emp.salary);
const salesSalaries = employeeData.filter(emp => emp.department === 'Sales').map(emp => emp.salary);

const anovaResult = datly.anova_oneway([itSalaries, hrSalaries, salesSalaries]);
console.log('ANOVA Test (Salary by Department):', anovaResult);

// 17. Create comprehensive visualization dashboard
// Correlation heatmap
const numericData = employeeData.map(emp => [emp.age, emp.salary / 1000, emp.experience, emp.performance]);
const corrMatrix = [
  [1.0, 0.75, 0.95, 0.62],
  [0.75, 1.0, 0.68, 0.43],
  [0.95, 0.68, 1.0, 0.71],
  [0.62, 0.43, 0.71, 1.0]
];

datly.plotHeatmap(corrMatrix, {
  labels: ['Age', 'Salary (k)', 'Experience', 'Performance'],
  title: 'Employee Metrics Correlation',
  showValues: true
}, '#correlation-heatmap');
```

---

## Tips and Best Practices

1. **Data Preparation**: Always check for missing values and outliers before analysis using `missing_values()` and `outliers_zscore()`
2. **Feature Scaling**: Scale features before training distance-based models (KNN) or neural networks using `standard_scaler_fit()` and `standard_scaler_transform()`
3. **Cross-Validation**: Use `train_test_split()` to assess model performance on unseen data
4. **Model Selection**: Start with simple models (linear regression) before trying complex ones
5. **Hyperparameter Tuning**: Experiment with different parameters (k in KNN, max_depth in trees)
6. **Visualization**: Always visualize your data and results using the plotting functions to gain insights
7. **Statistical Tests**: Check assumptions (normality using `shapiro_wilk()`) before parametric tests
8. **Object Access**: Results are returned as JavaScript objects - access properties directly (e.g., `result.value`, `result.p_value`)

---

## API Reference Summary

### Statistics Functions
- `mean(array)`, `median(array)`, `variance(array)`, `std(array)`
- `skewness(array)`, `kurtosis(array)`, `percentile(array, p)`
- `describe(array)` - comprehensive statistics

### Dataframe Operations
- `df_from_csv()`, `df_from_json()`, `df_from_array()`, `df_from_object()`
- `df_get_column()`, `df_get_value()`, `df_get_columns()`
- `df_head()`, `df_tail()`, `df_corr()`

### Machine Learning
- `train_linear_regression()`, `predict_linear()`
- `train_logistic_regression()`, `predict_logistic()`
- `train_knn()`, `predict_knn()`
- `train_decision_tree()`, `train_random_forest()`
- `train_naive_bayes()`, `kmeans()`

### Statistical Tests
- `ttest_1samp()`, `ttest_ind()`, `anova_oneway()`
- `shapiro_wilk()`, `correlation()`

### Utilities
- `train_test_split()`, `standard_scaler_fit()`, `standard_scaler_transform()`
- `metrics_classification()`, `metrics_regression()`
- `eda_overview()`, `missing_values()`, `outliers_zscore()`

### Visualization
- `plotHistogram()`, `plotScatter()`, `plotLine()`, `plotBar()`
- `plotBoxplot()`, `plotPie()`, `plotHeatmap()`, `plotViolin()`
- `plotDensity()`, `plotQQ()`, `plotParallel()`, `plotPairplot()`, `plotMultiline()`

---

## License

This documentation is provided as-is. Please refer to the library's official repository for licensing information.

---

## Support

For issues, questions, or contributions, please visit the official datly repository.