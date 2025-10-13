# datly
A comprehensive JavaScript library for data analysis, statistics, machine learning, and visualization.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Data Preparation](#data-preparation)
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
  console.log(result);
</script>
```

### Module Import

```javascript
import * as datly from 'datly';
```

---

## Core Concepts

### Output Format

All analysis functions return results in a structured YAML-like text format that can be parsed or displayed:

```yaml
type: statistic
name: mean
value: 3
n: 5
```

This format makes it easy to:
- Display results in a readable format
- Parse results programmatically
- Store analysis outputs as text
- Share results across different systems

---

## Data Preparation

### `dataframe_from_json(data)`

Creates a dataframe summary from JSON data.

**Parameters:**
- `data`: Array of objects or single object

**Returns:**
```yaml
type: dataframe
columns:
  - name
  - age
  - salary
n_rows: 100
n_cols: 3
dtypes:
  - string
  - number
  - number
preview:
  - name: alice
    age: 30
    salary: 50000
  - name: bob
    age: 25
    salary: 45000
```

**Example:**
```javascript
const data = [
  { name: 'Alice', age: 30, salary: 50000 },
  { name: 'Bob', age: 25, salary: 45000 },
  { name: 'Charlie', age: 35, salary: 60000 }
];

const df = datly.dataframe_from_json(data);
console.log(df);
```

---

## Descriptive Statistics

### `mean(array)`

Calculates the arithmetic mean of an array of numbers.

**Returns:**
```yaml
type: statistic
name: mean
n: 5
value: 3
```

**Example:**
```javascript
datly.mean([1, 2, 3, 4, 5]); // 3
```

### `median(array)`

Calculates the median value.

**Returns:**
```yaml
type: statistic
name: median
n: 5
value: 3
```

**Example:**
```javascript
datly.median([1, 2, 3, 4, 5]); // 3
datly.median([1, 2, 3, 4]); // 2.5
```

### `variance(array, sample = true)`

Calculates the variance.

**Parameters:**
- `array`: Array of numbers
- `sample`: If true, uses sample variance (n-1); if false, uses population variance (n)

**Returns:**
```yaml
type: statistic
name: variance
sample: true
n: 5
value: 2.5
```

**Example:**
```javascript
datly.variance([1, 2, 3, 4, 5]); // Sample variance
datly.variance([1, 2, 3, 4, 5], false); // Population variance
```

### `stddeviation(array, sample = true)`

Calculates the standard deviation.

**Returns:**
```yaml
type: statistic
name: std_deviation
sample: true
n: 5
value: 1.5811388300841898
```

**Example:**
```javascript
datly.stddeviation([1, 2, 3, 4, 5]);
```

### `minv(array)`

Returns the minimum value.

**Returns:**
```yaml
type: statistic
name: min
value: 1
```

### `maxv(array)`

Returns the maximum value.

**Returns:**
```yaml
type: statistic
name: max
value: 5
```

### `quantile(array, q)`

Calculates the q-th quantile (0 ≤ q ≤ 1).

**Returns:**
```yaml
type: statistic
name: quantile
q: 0.25
n: 100
value: 25.5
```

**Example:**
```javascript
datly.quantile([1, 2, 3, 4, 5], 0.25); // First quartile
datly.quantile([1, 2, 3, 4, 5], 0.5);  // Median
datly.quantile([1, 2, 3, 4, 5], 0.75); // Third quartile
```

### `skewness(array)`

Calculates the skewness (measure of asymmetry).

**Returns:**
```yaml
type: statistic
name: skewness
value: 0
```

**Example:**
```javascript
datly.skewness([1, 2, 3, 4, 5]); // ~0 for symmetric data
```

### `kurtosis(array)`

Calculates the kurtosis (measure of tailedness).

**Returns:**
```yaml
type: statistic
name: kurtosis
value: -1.2
```

**Example:**
```javascript
datly.kurtosis([1, 2, 3, 4, 5]);
```

---

## Exploratory Data Analysis

### `df_describe(data)`

Generates comprehensive descriptive statistics for a dataset.

**Returns:**
```yaml
type: describe
columns:
  age:
    dtype: number
    count: 100
    missing: 0
    mean: 35.5
    std: 10.2
    min: 18
    q1: 28
    median: 35
    q3: 43
    max: 65
    skewness: 0.15
    kurtosis: -0.5
  name:
    dtype: string
    count: 100
    missing: 2
    unique: 95
    top:
      - value: john
        freq: 3
      - value: alice
        freq: 2
```

**Example:**
```javascript
const data = [
  { age: 25, salary: 50000, dept: 'IT' },
  { age: 30, salary: 60000, dept: 'HR' },
  { age: 35, salary: 70000, dept: 'IT' }
];

const description = datly.df_describe(data);
console.log(description);
```

### `df_missing_report(data)`

Analyzes missing values in the dataset.

**Returns:**
```yaml
type: missing_report
rows:
  - column: age
    missing: 5
    missing_rate: 0.05
  - column: salary
    missing: 0
    missing_rate: 0
  - column: name
    missing: 10
    missing_rate: 0.1
```

**Example:**
```javascript
const report = datly.df_missing_report(data);
```

### `df_corr(data, method = 'pearson')`

Calculates correlation matrix between numeric columns.

**Parameters:**
- `data`: Array of objects
- `method`: 'pearson' or 'spearman'

**Returns:**
```yaml
type: correlation_matrix
method: pearson
matrix:
  age:
    age: 1
    salary: 0.85
    experience: 0.92
  salary:
    age: 0.85
    salary: 1
    experience: 0.78
  experience:
    age: 0.92
    salary: 0.78
    experience: 1
```

**Example:**
```javascript
const corr = datly.df_corr(data, 'pearson');
const spearman = datly.df_corr(data, 'spearman');
```

### `eda_overview(data)`

Generates a comprehensive EDA report combining describe, missing values, and correlation.

**Returns:**
```yaml
type: eda
summary:
  age:
    dtype: number
    count: 100
    mean: 35.5
    std: 10.2
    ...
missing:
  - column: age
    missing: 5
    missing_rate: 0.05
correlation:
  age:
    age: 1
    salary: 0.85
```

**Example:**
```javascript
const overview = datly.eda_overview(data);
```

---

## Probability Distributions

### Normal Distribution

#### `normal_pdf(x, mu = 0, sigma = 1)`

Probability density function of normal distribution.

**Returns:**
```yaml
type: distribution
name: normal_pdf
params:
  mu: 0
  sigma: 1
value: 0.3989422804014327
```

**Example:**
```javascript
datly.normal_pdf(0); // PDF at x=0
datly.normal_pdf([0, 1, 2], 0, 1); // PDF for multiple values
```

#### `normal_cdf(x, mu = 0, sigma = 1)`

Cumulative distribution function of normal distribution.

**Returns:**
```yaml
type: distribution
name: normal_cdf
params:
  mu: 0
  sigma: 1
value: 0.5
```

**Example:**
```javascript
datly.normal_cdf(0); // P(X ≤ 0)
datly.normal_cdf(1.96); // P(X ≤ 1.96) ≈ 0.975
```

#### `normal_ppf(p, mu = 0, sigma = 1)`

Percent point function (inverse CDF) of normal distribution.

**Returns:**
```yaml
type: distribution
name: normal_ppf
params:
  mu: 0
  sigma: 1
value: 1.959963984540054
```

**Example:**
```javascript
datly.normal_ppf(0.975); // Returns ~1.96
```

### Binomial Distribution

#### `binomial_pmf(k, n, p)`

Probability mass function of binomial distribution.

**Parameters:**
- `k`: Number of successes (can be array)
- `n`: Number of trials
- `p`: Probability of success

**Returns:**
```yaml
type: distribution
name: binomial_pmf
params:
  n: 10
  p: 0.5
value: 0.24609375
```

**Example:**
```javascript
datly.binomial_pmf(5, 10, 0.5); // P(X = 5)
datly.binomial_pmf([0, 1, 2, 3], 10, 0.3); // Multiple values
```

#### `binomial_cdf(k, n, p)`

Cumulative distribution function of binomial distribution.

**Returns:**
```yaml
type: distribution
name: binomial_cdf
params:
  n: 10
  p: 0.5
value: 0.623046875
```

### Poisson Distribution

#### `poisson_pmf(k, lambda)`

Probability mass function of Poisson distribution.

**Returns:**
```yaml
type: distribution
name: poisson_pmf
params:
  lambda: 3
value: 0.22404180765538775
```

**Example:**
```javascript
datly.poisson_pmf(3, 3); // P(X = 3) when λ = 3
```

#### `poisson_cdf(k, lambda)`

Cumulative distribution function of Poisson distribution.

**Returns:**
```yaml
type: distribution
name: poisson_cdf
params:
  lambda: 3
value: 0.6472319374260858
```

---

## Hypothesis Testing

### `t_test_one_sample(array, hypothesized_mean)`

One-sample t-test.

**Returns:**
```yaml
type: hypothesis_test
name: one_sample_t_test
statistic: 2.345
df: 99
p_value: 0.021
mean: 105
hypothesized_mean: 100
```

**Example:**
```javascript
const data = [102, 98, 105, 110, 95, 100, 108];
datly.t_test_one_sample(data, 100);
```

### `t_test_paired(array1, array2)`

Paired samples t-test.

**Returns:**
```yaml
type: hypothesis_test
name: paired_t_test
statistic: 3.456
df: 29
p_value: 0.0018
mean_difference: 2.5
```

**Example:**
```javascript
const before = [120, 115, 130, 125, 140];
const after = [115, 110, 125, 120, 135];
datly.t_test_paired(before, after);
```

### `t_test_independent(array1, array2, equal_var = true)`

Independent samples t-test.

**Parameters:**
- `equal_var`: If true, assumes equal variances (pooled t-test); if false, uses Welch's t-test

**Returns:**
```yaml
type: hypothesis_test
name: independent_t_test
statistic: 2.105
df: 48
p_value: 0.041
means:
  group_a: 105.5
  group_b: 98.3
```

**Example:**
```javascript
const group1 = [100, 105, 110, 115, 120];
const group2 = [95, 98, 100, 102, 105];
datly.t_test_independent(group1, group2);
```

### `z_test_one_sample(array, mu = 0, sigma = null, alpha = 0.05)`

One-sample z-test with confidence interval.

**Returns:**
```yaml
type: hypothesis_test
name: one_sample_z_test
statistic: 2.345
p_value: 0.019
ci_lower: 102.5
ci_upper: 107.5
confidence: 0.95
extra:
  sample_mean: 105
  hypothesized_mean: 100
  se: 2.13
  sigma_used: 10
  n: 22
  effect_size: 0.5
```

**Example:**
```javascript
datly.z_test_one_sample([102, 98, 105, 110], 100, 5, 0.05);
```

### `anova_oneway(groups, alpha = 0.05)`

One-way ANOVA test.

**Parameters:**
- `groups`: Array of arrays, each representing a group

**Returns:**
```yaml
type: hypothesis_test
name: anova_oneway
statistic: 5.678
df:
  between: 2
  within: 27
p_value: 0.009
confidence: 0.95
extra:
  group_means:
    - 102.5
    - 108.3
    - 115.7
  grand_mean: 108.8
  ssb: 450.5
  ssw: 890.2
```

**Example:**
```javascript
const group1 = [100, 105, 110];
const group2 = [108, 112, 115];
const group3 = [115, 120, 125];
datly.anova_oneway([group1, group2, group3]);
```

### `chi_square_independence(observed, alpha = 0.05)`

Chi-square test for independence (contingency table).

**Parameters:**
- `observed`: 2D array (contingency table)

**Returns:**
```yaml
type: hypothesis_test
name: chi_square_independence
statistic: 8.456
df: 2
p_value: 0.015
confidence: 0.95
extra:
  observed:
    - - 10
      - 20
      - 30
    - - 15
      - 25
      - 35
  expected:
    - - 12.5
      - 22.5
      - 32.5
    - - 12.5
      - 22.5
      - 32.5
  dof: 2
```

**Example:**
```javascript
const table = [
  [10, 20, 30],
  [15, 25, 35]
];
datly.chi_square_independence(table);
```

### `chi_square_goodness(observed, expected, alpha = 0.05)`

Chi-square goodness of fit test.

**Returns:**
```yaml
type: hypothesis_test
name: chi_square_goodness_of_fit
statistic: 3.456
df: 3
p_value: 0.327
confidence: 0.95
extra:
  observed:
    - 45
    - 55
    - 48
    - 52
  expected:
    - 50
    - 50
    - 50
    - 50
  dof: 3
```

**Example:**
```javascript
const observed = [45, 55, 48, 52];
const expected = [50, 50, 50, 50];
datly.chi_square_goodness(observed, expected);
```

### `shapiro_wilk(array)`

Shapiro-Wilk test for normality.

**Returns:**
```yaml
type: hypothesis_test
name: shapiro_wilk
statistic: 0.987
n: 50
note: approximation; w > 0.9 suggests normality
```

**Example:**
```javascript
datly.shapiro_wilk([1.2, 2.3, 1.8, 2.1, 1.9, 2.0]);
```

### `jarque_bera(array)`

Jarque-Bera test for normality.

**Returns:**
```yaml
type: hypothesis_test
name: jarque_bera
statistic: 2.345
n: 100
df: 2
note: tests normality; low p-value rejects normality
```

### `levene_test(groups)`

Levene's test for homogeneity of variance.

**Returns:**
```yaml
type: hypothesis_test
name: levene_test
statistic: 1.234
df_between: 2
df_within: 27
note: tests homogeneity of variance
```

**Example:**
```javascript
const g1 = [1, 2, 3, 4, 5];
const g2 = [2, 3, 4, 5, 6];
const g3 = [3, 4, 5, 6, 7];
datly.levene_test([g1, g2, g3]);
```

### `kruskal_wallis(groups)`

Kruskal-Wallis H-test (non-parametric alternative to ANOVA).

**Returns:**
```yaml
type: hypothesis_test
name: kruskal_wallis
statistic: 8.765
df: 2
note: non-parametric alternative to anova
```

### `mann_whitney(array1, array2)`

Mann-Whitney U test (non-parametric alternative to t-test).

**Returns:**
```yaml
type: hypothesis_test
name: mann_whitney_u
statistic: 45
z_score: -1.234
p_value: 0.217
note: non-parametric alternative to t-test
```

### `wilcoxon_signed_rank(array1, array2)`

Wilcoxon signed-rank test (non-parametric paired test).

**Returns:**
```yaml
type: hypothesis_test
name: wilcoxon_signed_rank
statistic: 28
z_score: 1.567
p_value: 0.117
n: 20
```

### Confidence Intervals

#### `confidence_interval_mean(array, confidence = 0.95)`

Confidence interval for the mean.

**Returns:**
```yaml
type: confidence_interval
parameter: mean
confidence: 0.95
n: 50
mean: 102.5
lower: 98.3
upper: 106.7
margin: 4.2
```

#### `confidence_interval_proportion(successes, n, confidence = 0.95)`

Confidence interval for a proportion.

**Returns:**
```yaml
type: confidence_interval
parameter: proportion
confidence: 0.95
n: 100
proportion: 0.65
lower: 0.551
upper: 0.749
margin: 0.099
```

#### `confidence_interval_variance(array, confidence = 0.95)`

Confidence interval for variance.

**Returns:**
```yaml
type: confidence_interval
parameter: variance
confidence: 0.95
n: 30
variance: 25.5
lower: 18.2
upper: 38.7
```

#### `confidence_interval_difference(array1, array2, confidence = 0.95)`

Confidence interval for difference of means.

**Returns:**
```yaml
type: confidence_interval
parameter: difference_of_means
confidence: 0.95
difference: 5.5
lower: 2.3
upper: 8.7
margin: 3.2
means:
  group_a: 105.5
  group_b: 100
```

---

## Correlation Analysis

### `corr_pearson(array1, array2)`

Pearson correlation coefficient.

**Returns:**
```yaml
type: statistic
name: pearson_correlation
value: 0.856
```

**Example:**
```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 5, 4, 5];
datly.corr_pearson(x, y);
```

### `corr_spearman(array1, array2)`

Spearman rank correlation coefficient.

**Returns:**
```yaml
type: statistic
name: spearman_correlation
value: 0.9
```

### `corr_kendall(array1, array2)`

Kendall's tau correlation coefficient.

**Returns:**
```yaml
type: statistic
name: kendall_tau
value: 0.8
concordant: 8
discordant: 2
n: 5
```

### `corr_partial(array1, array2, array3)`

Partial correlation controlling for a third variable.

**Returns:**
```yaml
type: statistic
name: partial_correlation
value: 0.456
controlling_for: third_variable
```

### `corr_matrix_all(data)`

Comprehensive correlation matrix with Pearson, Spearman, and Kendall.

**Returns:**
```yaml
type: correlation_analysis
pearson:
  age:
    age: 1
    salary: 0.85
  salary:
    age: 0.85
    salary: 1
spearman:
  age:
    age: 1
    salary: 0.82
  salary:
    age: 0.82
    salary: 1
kendall:
  age:
    age: 1
    salary: 0.75
  salary:
    age: 0.75
    salary: 1
```

---

## Regression Models

### Linear Regression

#### `train_linear_regression(X, y)`

Trains a multiple linear regression model.

**Parameters:**
- `X`: 2D array of features [[x1, x2, ...], ...]
- `y`: Array of target values

**Returns:**
```yaml
type: linear_regression
weights:
  - 2.5
  - 1.8
  - -0.3
mse: 12.34
r2: 0.856
n: 100
p: 2
```

**Example:**
```javascript
const X = [[1, 2], [2, 3], [3, 4], [4, 5]];
const y = [3, 5, 7, 9];
const model = datly.train_linear_regression(X, y);
```

#### `predict_linear(model, X)`

Makes predictions using a trained linear regression model.

**Parameters:**
- `model`: Model text/object from `train_linear_regression`
- `X`: 2D array of features

**Returns:**
```yaml
type: prediction
name: linear_regression
predictions:
  - 105.3
  - 110.7
  - 98.2
```

**Example:**
```javascript
const predictions = datly.predict_linear(model, [[5, 6], [6, 7]]);
```

### Logistic Regression

#### `train_logistic_regression(X, y, options = {})`

Trains a logistic regression model for binary classification.

**Parameters:**
- `X`: 2D array of features
- `y`: Array of binary labels (0 or 1)
- `options`:
  - `learning_rate`: Learning rate (default: 0.1)
  - `iterations`: Number of iterations (default: 1000)
  - `l2`: L2 regularization parameter (default: 0)

**Returns:**
```yaml
type: logistic_regression
weights:
  - 0.5
  - 1.2
  - -0.8
accuracy: 0.92
n: 100
p: 2
```

**Example:**
```javascript
const X = [[1, 2], [2, 3], [3, 1], [4, 2]];
const y = [0, 0, 1, 1];
const model = datly.train_logistic_regression(X, y, {
  learning_rate: 0.1,
  iterations: 1000,
  l2: 0.01
});
```

#### `predict_logistic(model, X, threshold = 0.5)`

Makes predictions using a trained logistic regression model.

**Returns:**
```yaml
type: prediction
name: logistic_regression
threshold: 0.5
probabilities:
  - 0.234
  - 0.789
  - 0.456
classes:
  - 0
  - 1
  - 0
```

**Example:**
```javascript
const predictions = datly.predict_logistic(model, [[5, 6], [6, 7]], 0.5);
```

---

## Classification Models

### K-Nearest Neighbors (KNN)

#### `train_knn_classifier(X, y, k = 5)`

Trains a KNN classifier.

**Parameters:**
- `X`: 2D array of features
- `y`: Array of class labels
- `k`: Number of neighbors (default: 5)

**Returns:**
```yaml
type: knn_classifier
k: 5
x:
  - - 1
    - 2
  - - 2
    - 3
y:
  - 0
  - 1
n: 100
p: 2
```

**Example:**
```javascript
const X = [[1, 2], [2, 3], [3, 1], [4, 2]];
const y = [0, 0, 1, 1];
const model = datly.train_knn_classifier(X, y, 3);
```

#### `predict_knn_classifier(model, X)`

Makes predictions using KNN classifier.

**Returns:**
```yaml
type: prediction
name: knn_classifier
k: 5
predictions:
  - 0
  - 1
  - 1
```

#### `train_knn_regressor(X, y, k = 5)`

Trains a KNN regressor.

**Returns:**
```yaml
type: knn_regressor
k: 5
x:
  - - 1
    - 2
  - - 2
    - 3
y:
  - 10.5
  - 12.3
n: 100
p: 2
```

#### `predict_knn_regressor(model, X)`

Makes predictions using KNN regressor.

**Returns:**
```yaml
type: prediction
name: knn_regressor
k: 5
predictions:
  - 10.7
  - 11.8
  - 12.5
```

### Decision Trees

#### `train_decision_tree_classifier(X, y, options = {})`

Trains a decision tree classifier.

**Parameters:**
- `options`:
  - `max_depth`: Maximum depth of tree (default: 5)
  - `min_samples_split`: Minimum samples required to split (default: 2)

**Returns:**
```yaml
type: decision_tree_classifier
tree:
  leaf: false
  feature: 0
  threshold: 2.5
  left:
    leaf: true
    prediction: 0
    n: 50
  right:
    leaf: true
    prediction: 1
    n: 50
max_depth: 5
min_samples: 2
n: 100
p: 2
```

**Example:**
```javascript
const model = datly.train_decision_tree_classifier(X, y, {
  max_depth: 5,
  min_samples_split: 2
});
```

#### `train_decision_tree_regressor(X, y, options = {})`

Trains a decision tree regressor.

**Returns:**
```yaml
type: decision_tree_regressor
tree:
  leaf: false
  feature: 0
  threshold: 2.5
  left: ...
  right: ...
max_depth: 5
min_samples: 2
n: 100
p: 2
```

#### `predict_decision_tree(model, X)`

Makes predictions using a decision tree.

**Returns:**
```yaml
type: prediction
name: decision_tree_classifier
predictions:
  - 0
  - 1
  - 1
```

### Random Forest

#### `train_random_forest_classifier(X, y, options = {})`

Trains a random forest classifier.

**Parameters:**
- `options`:
  - `n_estimators`: Number of trees (default: 10)
  - `max_depth`: Maximum depth (default: 5)
  - `min_samples_split`: Minimum samples to split (default: 2)
  - `seed`: Random seed (default: 42)

**Returns:**
```yaml
type: random_forest_classifier
trees:
  - leaf: false
    feature: 0
    threshold: 2.5
    ...
  - leaf: false
    feature: 1
    threshold: 3.2
    ...
n_trees: 10
max_depth: 5
min_samples: 2
n: 100
p: 2
```

**Example:**
```javascript
const model = datly.train_random_forest_classifier(X, y, {
  n_estimators: 10,
  max_depth: 5,
  seed: 42
});
```

#### `train_random_forest_regressor(X, y, options = {})`

Trains a random forest regressor.

**Returns:**
```yaml
type: random_forest_regressor
trees: [...]
n_trees: 10
max_depth: 5
min_samples: 2
n: 100
p: 2
```

#### `predict_random_forest_classifier(model, X)`

Makes predictions using random forest classifier.

**Returns:**
```yaml
type: prediction
name: random_forest_classifier
n_trees: 10
predictions:
  - 0
  - 1
  - 1
```

#### `predict_random_forest_regressor(model, X)`

Makes predictions using random forest regressor.

**Returns:**
```yaml
type: prediction
name: random_forest_regressor
n_trees: 10
predictions:
  - 10.7
  - 11.8
  - 12.5
```

### Naive Bayes

#### `train_naive_bayes(X, y)`

Trains a Gaussian Naive Bayes classifier.

**Parameters:**
- `X`: 2D array of features
- `y`: Array of class labels

**Returns:**
```yaml
type: naive_bayes
classes:
  - 0
  - 1
priors:
  0: 0.5
  1: 0.5
stats:
  0:
    - mean: 2.5
      std: 1.2
    - mean: 3.1
      std: 0.8
  1:
    - mean: 5.2
      std: 1.5
    - mean: 6.3
      std: 1.1
n: 100
p: 2
```

**Example:**
```javascript
const X = [[1, 2], [2, 3], [5, 6], [6, 7]];
const y = [0, 0, 1, 1];
const model = datly.train_naive_bayes(X, y);
```

#### `predict_naive_bayes(model, X)`

Makes predictions using Naive Bayes classifier.

**Returns:**
```yaml
type: prediction
name: naive_bayes
predictions:
  - 0
  - 1
  - 1
```

---

## Clustering

### K-Means Clustering

#### `train_kmeans(X, k = 3, options = {})`

Trains a K-means clustering model.

**Parameters:**
- `X`: 2D array of features
- `k`: Number of clusters (default: 3)
- `options`:
  - `max_iterations`: Maximum iterations (default: 100)
  - `seed`: Random seed (default: 42)

**Returns:**
```yaml
type: kmeans
k: 3
centroids:
  - - 2.1
    - 3.5
  - - 5.8
    - 6.2
  - - 9.1
    - 8.7
inertia: 45.67
n: 150
p: 2
```

**Example:**
```javascript
const X = [[1, 2], [2, 3], [5, 6], [6, 7], [9, 8], [10, 9]];
const model = datly.train_kmeans(X, 3, {
  max_iterations: 100,
  seed: 42
});
```

#### `predict_kmeans(model, X)`

Assigns cluster labels to new data points.

**Returns:**
```yaml
type: prediction
name: kmeans
k: 3
cluster_labels:
  - 0
  - 0
  - 1
  - 1
  - 2
  - 2
```

**Example:**
```javascript
const newData = [[1.5, 2.5], [5.5, 6.5], [9.5, 8.5]];
const clusters = datly.predict_kmeans(model, newData);
```

---

## Ensemble Methods

### `ensemble_voting_classifier(models, X, method = 'hard')`

Combines multiple classifier predictions through voting.

**Parameters:**
- `models`: Array of trained model texts/objects
- `X`: 2D array of features
- `method`: 'hard' for majority voting, 'soft' for probability averaging

**Returns:**
```yaml
type: ensemble_prediction
method: voting_hard
n_models: 3
predictions:
  - 0
  - 1
  - 1
  - 0
```

**Example:**
```javascript
const model1 = datly.train_logistic_regression(X, y);
const model2 = datly.train_knn_classifier(X, y, 5);
const model3 = datly.train_decision_tree_classifier(X, y);

const ensemble = datly.ensemble_voting_classifier(
  [model1, model2, model3],
  X_test,
  'hard'
);
```

### `ensemble_voting_regressor(models, X)`

Combines multiple regressor predictions through averaging.

**Returns:**
```yaml
type: ensemble_prediction
method: voting_average
n_models: 3
predictions:
  - 105.3
  - 110.7
  - 98.2
```

**Example:**
```javascript
const model1 = datly.train_linear_regression(X, y);
const model2 = datly.train_knn_regressor(X, y, 5);
const model3 = datly.train_decision_tree_regressor(X, y);

const ensemble = datly.ensemble_voting_regressor(
  [model1, model2, model3],
  X_test
);
```

---

## Model Evaluation

### `train_test_split(X, y, test_size = 0.2, seed = 42)`

Splits data into training and testing sets.

**Parameters:**
- `X`: 2D array of features
- `y`: Array of labels
- `test_size`: Proportion for test set (default: 0.2)
- `seed`: Random seed (default: 42)

**Returns:**
```yaml
type: split
sizes:
  train: 80
  test: 20
indices:
  train:
    - 0
    - 2
    - 3
    ...
  test:
    - 1
    - 4
    ...
preview:
  x_train:
    - - 1
      - 2
    - - 3
      - 4
  y_train:
    - 0
    - 1
    - 0
```

**Example:**
```javascript
const split = datly.train_test_split(X, y, 0.2, 42);
// Use split.indices to extract train/test data
```

### Classification Metrics

#### `metrics_classification(y_true, y_pred)`

Calculates classification metrics including accuracy, precision, recall, and F1-score.

**Returns:**
```yaml
type: metric
name: classification_report
confusion_matrix:
  tp: 45
  fp: 5
  tn: 42
  fn: 8
accuracy: 0.87
precision: 0.9
recall: 0.849
f1: 0.874
```

**Example:**
```javascript
const y_true = [0, 1, 1, 0, 1, 1, 0, 0];
const y_pred = [0, 1, 0, 0, 1, 1, 0, 1];
const metrics = datly.metrics_classification(y_true, y_pred);
```

### Regression Metrics

#### `metrics_regression(y_true, y_pred)`

Calculates regression metrics including MSE, MAE, and R².

**Returns:**
```yaml
type: metric
name: regression_report
mse: 12.34
mae: 2.87
r2: 0.856
```

**Example:**
```javascript
const y_true = [3.0, 5.0, 7.0, 9.0];
const y_pred = [2.8, 5.2, 6.9, 9.1];
const metrics = datly.metrics_regression(y_true, y_pred);
```

### Cross Validation

#### `cross_validate(X, y, model_type, options = {})`

Performs k-fold cross-validation.

**Parameters:**
- `X`: 2D array of features
- `y`: Array of labels
- `model_type`: String - 'linear_regression', 'logistic_regression', 'knn_classifier', 'decision_tree_classifier', 'random_forest_classifier'
- `options`:
  - `k_folds`: Number of folds (default: 5)
  - Model-specific options (e.g., `k` for KNN, `max_depth` for trees)

**Returns:**
```yaml
type: cross_validation
model_type: logistic_regression
k_folds: 5
scores:
  - 0.85
  - 0.88
  - 0.82
  - 0.87
  - 0.86
mean_score: 0.856
std_score: 0.022
```

**Example:**
```javascript
const cv = datly.cross_validate(X, y, 'logistic_regression', {
  k_folds: 5,
  learning_rate: 0.1,
  iterations: 1000
});
```

### Feature Importance

#### `feature_importance_tree(model)`

Extracts feature importance from tree-based models.

**Parameters:**
- `model`: Trained decision tree or random forest model

**Returns:**
```yaml
type: feature_importance
model: random_forest_classifier
n_trees: 10
importance:
  - 0.45
  - 0.32
  - 0.15
  - 0.08
```

**Example:**
```javascript
const model = datly.train_random_forest_classifier(X, y);
const importance = datly.feature_importance_tree(model);
```

---

## Data Preprocessing

### Scaling

#### `standard_scaler_fit(X)`

Fits a standard scaler (z-score normalization).

**Returns:**
```yaml
type: standard_scaler
params:
  - mean: 50.5
    std: 15.2
  - mean: 100.3
    std: 25.7
n: 100
p: 2
```

**Example:**
```javascript
const X = [[50, 100], [60, 120], [40, 90]];
const scaler = datly.standard_scaler_fit(X);
```

#### `standard_scaler_transform(scaler, X)`

Transforms data using fitted standard scaler.

**Returns:**
```yaml
type: scaled_data
method: standard
preview:
  - - 0.0
    - 0.0
  - - 0.625
    - 0.767
  - - -0.625
    - -0.767
```

**Example:**
```javascript
const scaled = datly.standard_scaler_transform(scaler, X);
```

#### `minmax_scaler_fit(X)`

Fits a min-max scaler (scales to [0, 1] range).

**Returns:**
```yaml
type: minmax_scaler
params:
  - min: 40
    max: 60
  - min: 90
    max: 120
n: 100
p: 2
```

#### `minmax_scaler_transform(scaler, X)`

Transforms data using fitted min-max scaler.

**Returns:**
```yaml
type: scaled_data
method: minmax
preview:
  - - 0.5
    - 0.333
  - - 1.0
    - 1.0
  - - 0.0
    - 0.0
```

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

#### `train_pca(X, n_components = 2)`

Trains a PCA model.

**Parameters:**
- `X`: 2D array of features
- `n_components`: Number of principal components (default: 2)

**Returns:**
```yaml
type: pca
n_components: 2
means:
  - 50.5
  - 100.3
  - 75.8
components:
  - - 0.707
    - 0.707
    - 0.0
  - - -0.707
    - 0.707
    - 0.0
n: 100
p: 3
```

**Example:**
```javascript
const X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
const pca = datly.train_pca(X, 2);
```

#### `transform_pca(model, X)`

Transforms data to principal component space.

**Returns:**
```yaml
type: pca_transform
n_components: 2
preview:
  - - 2.121
    - 0.0
  - - 0.707
    - 0.0
  - - -1.414
    - 0.0
```

**Example:**
```javascript
const transformed = datly.transform_pca(pca, X);
```

---

## Time Series Analysis

### `moving_average(array, window = 3)`

Calculates moving average.

**Parameters:**
- `array`: Time series data
- `window`: Window size (default: 3)

**Returns:**
```yaml
type: time_series
method: moving_average
window: 3
values:
  - 10
  - 15
  - 20
  - 22
  - 25
```

**Example:**
```javascript
const data = [10, 20, 30, 20, 30, 25];
const ma = datly.moving_average(data, 3);
```

### `exponential_smoothing(array, alpha = 0.3)`

Applies exponential smoothing.

**Parameters:**
- `array`: Time series data
- `alpha`: Smoothing parameter (0 < α < 1)

**Returns:**
```yaml
type: time_series
method: exponential_smoothing
alpha: 0.3
values:
  - 10
  - 13
  - 18.1
  - 18.47
  - 21.73
```

**Example:**
```javascript
const smoothed = datly.exponential_smoothing(data, 0.3);
```

### `autocorrelation(array, lag = 1)`

Calculates autocorrelation at a given lag.

**Parameters:**
- `array`: Time series data
- `lag`: Lag value (default: 1)

**Returns:**
```yaml
type: statistic
name: autocorrelation
lag: 1
value: 0.456
```

**Example:**
```javascript
const acf = datly.autocorrelation(data, 1);
```

---

## Outlier Detection

### `outliers_iqr(array)`

Detects outliers using the IQR (Interquartile Range) method.

**Returns:**
```yaml
type: outlier_detection
method: iqr
lower_bound: 45.5
upper_bound: 154.5
n_outliers: 3
outlier_indices:
  - 5
  - 12
  - 23
outlier_values:
  - 200
  - 30
  - 180
```

**Example:**
```javascript
const data = [50, 55, 60, 65, 70, 200, 75, 80];
const outliers = datly.outliers_iqr(data);
```

### `outliers_zscore(array, threshold = 3)`

Detects outliers using z-score method.

**Parameters:**
- `array`: Array of numbers
- `threshold`: Z-score threshold (default: 3)

**Returns:**
```yaml
type: outlier_detection
method: zscore
threshold: 3
n_outliers: 2
outlier_indices:
  - 5
  - 12
outlier_values:
  - 200
  - 30
```

**Example:**
```javascript
const outliers = datly.outliers_zscore(data, 3);
```

---

## Visualization

All visualization functions create SVG-based charts. They accept optional configuration and a selector for where to render the chart.

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

Creates a histogram.

**Additional Options:**
- `bins`: Number of bins (default: 10)

**Example:**
```javascript
const data = [1, 2, 2, 3, 3, 3, 4, 4, 5];
datly.plotHistogram(data, {
  width: 600,
  height: 400,
  bins: 10,
  title: 'Distribution',
  color: '#4CAF50'
}, '#chart');
```

### `plotScatter(x, y, options = {}, selector)`

Creates a scatter plot.

**Additional Options:**
- `size`: Point size (default: 4)

**Example:**
```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 3, 5, 6];
datly.plotScatter(x, y, {
  width: 600,
  height: 400,
  title: 'Scatter Plot',
  xlabel: 'X Variable',
  ylabel: 'Y Variable',
  size: 5
}, '#chart');
```

### `plotLine(x, y, options = {}, selector)`

Creates a line chart.

**Additional Options:**
- `lineWidth`: Line width (default: 2)
- `showPoints`: Show data points (default: false)

**Example:**
```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 3, 5, 6];
datly.plotLine(x, y, {
  lineWidth: 3,
  showPoints: true,
  title: 'Time Series'
}, '#chart');
```

### `plotBar(categories, values, options = {}, selector)`

Creates a bar chart.

**Example:**
```javascript
const categories = ['A', 'B', 'C', 'D'];
const values = [10, 25, 15, 30];
datly.plotBar(categories, values, {
  title: 'Sales by Category',
  ylabel: 'Sales ($)'
}, '#chart');
```

### `plotBoxplot(data, options = {}, selector)`

Creates box plots for one or more groups.

**Parameters:**
- `data`: Array of arrays (each array is a group) or single array
- `options`:
  - `labels`: Array of group labels

**Example:**
```javascript
const group1 = [1, 2, 3, 4, 5, 6];
const group2 = [2, 3, 4, 5, 6, 7];
const group3 = [3, 4, 5, 6, 7, 8];

datly.plotBoxplot([group1, group2, group3], {
  labels: ['Group A', 'Group B', 'Group C'],
  title: 'Comparison'
}, '#chart');
```

### `plotPie(labels, values, options = {}, selector)`

Creates a pie chart.

**Additional Options:**
- `showLabels`: Display labels (default: true)

**Example:**
```javascript
const labels = ['Category A', 'Category B', 'Category C'];
const values = [30, 45, 25];
datly.plotPie(labels, values, {
  title: 'Market Share',
  showLabels: true
}, '#chart');
```

### `plotHeatmap(matrix, options = {}, selector)`

Creates a heatmap for a correlation matrix.

**Additional Options:**
- `labels`: Array of variable names
- `showValues`: Display correlation values (default: true)

**Example:**
```javascript
const corrMatrix = [
  [1.0, 0.8, 0.3],
  [0.8, 1.0, 0.5],
  [0.3, 0.5, 1.0]
];

datly.plotHeatmap(corrMatrix, {
  labels: ['Var1', 'Var2', 'Var3'],
  showValues: true,
  title: 'Correlation Matrix'
}, '#chart');
```

### `plotViolin(data, options = {}, selector)`

Creates violin plots showing distribution density.

**Parameters:**
- `data`: Array of arrays or single array
- `options`:
  - `labels`: Group labels

**Example:**
```javascript
const group1 = [1, 2, 2, 3, 3, 3, 4, 4, 5];
const group2 = [2, 3, 3, 4, 4, 4, 5, 5, 6];

datly.plotViolin([group1, group2], {
  labels: ['Before', 'After'],
  title: 'Distribution Comparison'
}, '#chart');
```

### `plotDensity(array, options = {}, selector)`

Creates a kernel density plot.

**Additional Options:**
- `bandwidth`: Smoothing bandwidth (default: 5)

**Example:**
```javascript
const data = [1, 2, 2, 3, 3, 3, 4, 4, 5];
datly.plotDensity(data, {
  bandwidth: 0.5,
  title: 'Density Plot'
}, '#chart');
```

### `plotQQ(array, options = {}, selector)`

Creates a Q-Q plot for normality assessment.

**Example:**
```javascript
const data = [1.2, 2.3, 1.8, 2.1, 1.9, 2.0, 2.4];
datly.plotQQ(data, {
  title: 'Q-Q Plot'
}, '#chart');
```

### `plotParallel(data, columns, options = {}, selector)`

Creates a parallel coordinates plot.

**Parameters:**
- `data`: Array of objects
- `columns`: Array of column names to include
- `options`:
  - `colors`: Array of colors for each observation

**Example:**
```javascript
const data = [
  { age: 25, salary: 50000, experience: 2 },
  { age: 30, salary: 60000, experience: 5 },
  { age: 35, salary: 70000, experience: 8 }
];

datly.plotParallel(data, ['age', 'salary', 'experience'], {
  title: 'Parallel Coordinates'
}, '#chart');
```

### `plotPairplot(data, columns, options = {}, selector)`

Creates a pairplot matrix showing all pairwise relationships.

**Parameters:**
- `data`: Array of objects
- `columns`: Array of column names
- `options`:
  - `size`: Size of each subplot (default: 120)
  - `color`: Point color

**Example:**
```javascript
const data = [
  { age: 25, salary: 50000, experience: 2 },
  { age: 30, salary: 60000, experience: 5 },
  { age: 35, salary: 70000, experience: 8 }
];

datly.plotPairplot(data, ['age', 'salary', 'experience'], {
  size: 150
}, '#chart');
```

### `plotMultiline(series, options = {}, selector)`

Creates a multi-line chart for comparing time series.

**Parameters:**
- `series`: Array of objects with `name` and `data` properties
  - `data`: Array of `{x, y}` objects
- `options`:
  - `legend`: Show legend (default: false)

**Example:**
```javascript
const series = [
  {
    name: 'Series A',
    data: [{x: 1, y: 10}, {x: 2, y: 20}, {x: 3, y: 15}]
  },
  {
    name: 'Series B',
    data: [{x: 1, y: 15}, {x: 2, y: 25}, {x: 3, y: 20}]
  }
];

datly.plotMultiline(series, {
  legend: true,
  title: 'Comparison'
}, '#chart');
```

---

## Complete Example Workflow

Here's a complete example demonstrating a typical data analysis workflow:

```javascript
// 1. Load and explore data
const data = [
  { age: 25, salary: 50000, experience: 2, department: 'IT' },
  { age: 30, salary: 60000, experience: 5, department: 'HR' },
  { age: 35, salary: 70000, experience: 8, department: 'IT' },
  // ... more data
];

// 2. Perform EDA
const overview = datly.eda_overview(data);
console.log(overview);

// 3. Check correlations
const correlations = datly.df_corr(data, 'pearson');
console.log(correlations);

// 4. Prepare features and target
const X = data.map(d => [d.age, d.experience]);
const y = data.map(d => d.salary);

// 5. Split data
const split = datly.train_test_split(X, y, 0.2, 42);
const trainIndices = split.indices.train;
const testIndices = split.indices.test;

const X_train = trainIndices.map(i => X[i]);
const y_train = trainIndices.map(i => y[i]);
const X_test = testIndices.map(i => X[i]);
const y_test = testIndices.map(i => y[i]);

// 6. Scale features
const scaler = datly.standard_scaler_fit(X_train);
const X_train_scaled = datly.standard_scaler_transform(scaler, X_train);
const X_test_scaled = datly.standard_scaler_transform(scaler, X_test);

// 7. Train model
const model = datly.train_linear_regression(
  JSON.parse(X_train_scaled).preview,
  y_train
);

// 8. Make predictions
const predictions = datly.predict_linear(
  model,
  JSON.parse(X_test_scaled).preview
);

// 9. Evaluate model
const metrics = datly.metrics_regression(
  y_test,
  JSON.parse(predictions).predictions
);
console.log(metrics);

// 10. Visualize results
datly.plotScatter(y_test, JSON.parse(predictions).predictions, {
  title: 'Actual vs Predicted',
  xlabel: 'Actual',
  ylabel: 'Predicted'
}, '#results');
```

---

## Tips and Best Practices

1. **Data Preparation**: Always check for missing values and outliers before analysis
2. **Feature Scaling**: Scale features before training distance-based models (KNN, SVM)
3. **Cross-Validation**: Use cross-validation to assess model performance reliably
4. **Model Selection**: Start with simple models (linear regression) before trying complex ones
5. **Hyperparameter Tuning**: Experiment with different hyperparameters (k in KNN, max_depth in trees)
6. **Visualization**: Always visualize your data and results to gain insights
7. **Statistical Tests**: Check assumptions (normality, homogeneity) before parametric tests

---

## License

This documentation is provided as-is. Please refer to the library's official repository for licensing information.

---

## Support

For issues, questions, or contributions, please visit the official Datly.js repository.