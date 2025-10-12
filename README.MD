# Datly

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![D3](https://img.shields.io/badge/d3-v7-orange.svg)
![NPM](https://img.shields.io/badge/npm-datly-red.svg)

 **Javascript toolkit for data science, statistical analysis and machine learning.**
 ---

### ⚡ Key Features:
- 📈 **Complete Statistics Suite** - From descriptive stats to advanced hypothesis testing
- 🤖 **7 ML Algorithms** - Classification, regression, and ensemble methods
- 📊 **13 Visualization Types** - Interactive D3.js charts with one-line commands (requires D3.js)
- 🔄 **Auto-Analysis** - Intelligent data exploration with automated insights
- 🎨 **Zero Config** - Works out of the box, customizable when needed
- 🌐 **Universal** - Same API for browser and Node.js
---

**Datly** is a comprehensive JavaScript library for statistical analysis, machine learning, and data visualization. Built for both browser and Node.js environments, it provides a complete toolkit for data scientists, analysts, and developers.

---

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
  - [1. Data Loading](#1-data-loading)
  - [2. Data Validation](#2-data-validation)
  - [3. Utility Functions](#3-utility-functions)
  - [4. Central Tendency](#4-central-tendency)
  - [5. Dispersion Measures](#5-dispersion-measures)
  - [6. Position Measures](#6-position-measures)
  - [7. Shape Analysis](#7-shape-analysis)
  - [8. Hypothesis Testing](#8-hypothesis-testing)
  - [9. Confidence Intervals](#9-confidence-intervals)
  - [10. Normality Tests](#10-normality-tests)
  - [11. Correlation Analysis](#11-correlation-analysis)
  - [12. Regression Analysis](#12-regression-analysis)
  - [13. Report Generation](#13-report-generation)
  - [14. Pattern Detection](#14-pattern-detection)
  - [15. Result Interpretation](#15-result-interpretation)
  - [16. Auto-Analysis](#16-auto-analysis)
  - [17. Machine Learning](#17-machine-learning)
  - [18. Data Visualization](#18-data-visualization)
- [Complete Examples](#complete-examples)
- [API Reference](#api-reference)

---

## 🚀 Installation

### Browser (CDN)

```html
<!-- Include Datly -->
<script src="https://unpkg.com/datly"></script>

<!-- Include D3.js (only if using visualization module) -->
<script src="https://unpkg.com/d3"></script>

<script>
  const datly = new Datly();
</script>
```

### Node.js (NPM)

```bash
# Core library (statistics and machine learning)
npm install datly

# With visualization support (includes D3.js)
npm install datly d3
```

```javascript
const Datly = require('datly');
const datly = new Datly();
```

---

## ⚡ Quick Start

```javascript
// Initialize the library
const datly = new Datly();

// Load data from CSV
const data = await datly.loadCSV('data.csv');

// Calculate mean
const ages = [25, 30, 35, 40, 45];
const meanAge = datly.mean(ages);
console.log('Mean Age:', meanAge); // 35

// Perform t-test
const group1 = [23, 25, 27, 29, 31];
const group2 = [30, 32, 34, 36, 38];
const tTest = datly.tTest(group1, group2);
console.log('T-Test Result:', tTest);

// Create visualization
datly.plotHistogram(ages, {
  title: 'Age Distribution',
  xlabel: 'Age',
  ylabel: 'Frequency'
});
```

---

## 📦 Core Modules

### 1. Data Loading

Load data from various sources including CSV, JSON, and file systems.

#### Methods

##### `loadCSV(filePath, options)`
Load data from a CSV file.

```javascript
const data = await datly.loadCSV('sales.csv', {
  delimiter: ',',
  header: true,
  skipEmptyLines: true,
  encoding: 'utf8'
});

console.log(data);
// {
//   headers: ['product', 'sales', 'revenue'],
//   data: [
//     { product: 'A', sales: 100, revenue: 5000 },
//     { product: 'B', sales: 150, revenue: 7500 }
//   ],
//   length: 2,
//   columns: 3
// }
```

**Parameters:**
- `filePath` (string): Path to CSV file
- `options` (object): Configuration options
  - `delimiter` (string): Column delimiter (default: ',')
  - `header` (boolean): First row contains headers (default: true)
  - `skipEmptyLines` (boolean): Skip empty rows (default: true)
  - `encoding` (string): File encoding (default: 'utf8')

**Returns:** Dataset object with headers, data array, length, and column count

---

##### `loadJSON(source, options)`
Load data from JSON file, string, or object.

```javascript
// From file
const data1 = await datly.loadJSON('data.json');

// From JSON string
const jsonString = '{"users": [{"name": "John", "age": 30}]}';
const data2 = await datly.loadJSON(jsonString);

// From object
const obj = {
  headers: ['name', 'age'],
  data: [
    { name: 'Alice', age: 25 },
    { name: 'Bob', age: 30 }
  ]
};
const data3 = await datly.loadJSON(obj);
```

**Parameters:**
- `source` (string|object): JSON file path, string, or object
- `options` (object): Configuration options
  - `validateTypes` (boolean): Auto-infer data types (default: true)
  - `autoInferHeaders` (boolean): Extract headers from data (default: true)

**Returns:** Structured dataset object

---

##### `parseCSV(text, options)`
Parse CSV text into structured data.

```javascript
const csvText = `name,age,city
John,30,NYC
Jane,25,LA`;

const parsed = datly.parseCSV(csvText, {
  delimiter: ',',
  header: true
});

console.log(parsed.data);
// [
//   { name: 'John', age: 30, city: 'NYC' },
//   { name: 'Jane', age: 25, city: 'LA' }
// ]
```

---

##### `cleanData(dataset)`
Remove rows with all null values.

```javascript
const dataset = {
  headers: ['a', 'b', 'c'],
  data: [
    { a: 1, b: 2, c: 3 },
    { a: null, b: null, c: null }, // Will be removed
    { a: 4, b: 5, c: 6 }
  ],
  length: 3,
  columns: 3
};

const cleaned = datly.cleanData(dataset);
console.log(cleaned.length); // 2
```

---

##### `getColumn(dataset, columnName)`
Extract a single column as an array.

```javascript
const data = {
  headers: ['name', 'age', 'salary'],
  data: [
    { name: 'Alice', age: 25, salary: 50000 },
    { name: 'Bob', age: 30, salary: 60000 },
    { name: 'Charlie', age: null, salary: 55000 }
  ]
};

const ages = datly.getColumn(data, 'age');
console.log(ages); // [25, 30] (null values filtered)
```

---

##### `getColumns(dataset, columnNames)`
Extract multiple columns as an object.

```javascript
const columns = datly.getColumns(data, ['age', 'salary']);
console.log(columns);
// {
//   age: [25, 30],
//   salary: [50000, 60000, 55000]
// }
```

---

##### `filterRows(dataset, predicate)`
Filter dataset rows based on condition.

```javascript
const filtered = datly.filterRows(data, row => row.age > 25);
console.log(filtered.data);
// [{ name: 'Bob', age: 30, salary: 60000 }]
```

---

##### `sortBy(dataset, column, order)`
Sort dataset by column.

```javascript
const sorted = datly.sortBy(data, 'salary', 'desc');
console.log(sorted.data);
// [
//   { name: 'Bob', age: 30, salary: 60000 },
//   { name: 'Charlie', age: null, salary: 55000 },
//   { name: 'Alice', age: 25, salary: 50000 }
// ]
```

---

### 2. Data Validation

Validate data integrity and structure.

#### Methods

##### `validateData(dataset)`
Validate dataset structure and integrity.

```javascript
const dataset = {
  headers: ['a', 'b'],
  data: [
    { a: 1, b: 2 },
    { a: 3, b: 4, c: 5 } // Extra column 'c'
  ]
};

const validation = datly.validateData(dataset);
console.log(validation);
// {
//   valid: true,
//   errors: [],
//   warnings: ['Row 1: Extra columns: c']
// }
```

---

##### `validateNumericColumn(column)`
Validate that column contains numeric values.

```javascript
const column = [1, 2, 'three', 4, NaN, 5];
const result = datly.validateNumericColumn(column);
console.log(result);
// {
//   valid: true,
//   validCount: 4,
//   invalidCount: 2,
//   cleanData: [1, 2, 4, 5]
// }
```

---

##### `validateSampleSize(sample, minSize)`
Ensure sample has minimum required size.

```javascript
const sample = [1, 2, 3];
try {
  datly.validateSampleSize(sample, 5);
} catch (error) {
  console.log(error.message); // "Sample size (3) must be at least 5"
}
```

---

##### `validateConfidenceLevel(level)`
Validate confidence level is between 0 and 1.

```javascript
datly.validateConfidenceLevel(0.95); // true
datly.validateConfidenceLevel(1.5);  // throws error
```

---

### 3. Utility Functions

General-purpose statistical utilities.

#### Methods

##### `detectOutliers(data, method)`
Detect outliers using various methods.

```javascript
const data = [1, 2, 3, 4, 5, 100]; // 100 is an outlier

// IQR method (default)
const outliers1 = datly.detectOutliers(data, 'iqr');
console.log(outliers1);
// {
//   outliers: [100],
//   indices: [5],
//   count: 1,
//   percentage: 16.67
// }

// Z-score method
const outliers2 = datly.detectOutliers(data, 'zscore');

// Modified Z-score method
const outliers3 = datly.detectOutliers(data, 'modified_zscore');
```

**Methods available:**
- `'iqr'`: Interquartile Range method (default)
- `'zscore'`: Z-score method (|z| > 3)
- `'modified_zscore'`: Modified Z-score method

---

##### `frequencyTable(data)`
Create frequency distribution table.

```javascript
const colors = ['red', 'blue', 'red', 'green', 'blue', 'red'];
const freq = datly.frequencyTable(colors);
console.log(freq);
// [
//   { value: 'red', frequency: 3, relativeFrequency: 0.5, percentage: 50 },
//   { value: 'blue', frequency: 2, relativeFrequency: 0.333, percentage: 33.33 },
//   { value: 'green', frequency: 1, relativeFrequency: 0.167, percentage: 16.67 }
// ]
```

---

##### `groupBy(dataset, column, aggregations)`
Group data and calculate aggregations.

```javascript
const data = {
  headers: ['category', 'sales', 'profit'],
  data: [
    { category: 'A', sales: 100, profit: 20 },
    { category: 'B', sales: 150, profit: 30 },
    { category: 'A', sales: 200, profit: 40 },
    { category: 'B', sales: 180, profit: 35 }
  ]
};

const grouped = datly.groupBy(data, 'category', {
  sales: 'mean',
  profit: 'sum'
});

console.log(grouped);
// {
//   A: {
//     count: 2,
//     mean_sales: 150,
//     sum_profit: 60,
//     data: [...]
//   },
//   B: {
//     count: 2,
//     mean_sales: 165,
//     sum_profit: 65,
//     data: [...]
//   }
// }
```

**Aggregation functions:**
- `'mean'`: Average value
- `'median'`: Median value
- `'sum'`: Sum of values
- `'min'`: Minimum value
- `'max'`: Maximum value
- `'std'`: Standard deviation
- `'var'`: Variance
- `'count'`: Count of values

---

##### `sample(dataset, size, method)`
Extract a sample from dataset.

```javascript
const data = {
  headers: ['id', 'value'],
  data: Array.from({ length: 100 }, (_, i) => ({ id: i, value: i * 10 })),
  length: 100
};

// Random sampling
const randomSample = datly.sample(data, 10, 'random');

// Systematic sampling
const systematicSample = datly.sample(data, 10, 'systematic');

// First n records
const firstSample = datly.sample(data, 10, 'first');

// Last n records
const lastSample = datly.sample(data, 10, 'last');
```

---

##### `bootstrap(data, statistic, iterations)`
Bootstrap resampling for estimating statistic distribution.

```javascript
const data = [23, 25, 27, 29, 31, 33, 35];

const bootstrap = datly.bootstrap(data, 'mean', 1000);
console.log(bootstrap);
// {
//   bootstrapStats: [...], // 1000 bootstrap means
//   mean: 29.5,
//   standardError: 1.23,
//   confidenceInterval: { lower: 27.1, upper: 31.9 }
// }
```

**Statistics available:**
- `'mean'`: Bootstrap mean
- `'median'`: Bootstrap median
- `'std'`: Bootstrap standard deviation
- `'var'`: Bootstrap variance
- Custom function: Pass your own function

---

##### `contingencyTable(column1, column2)`
Create contingency table for categorical data.

```javascript
const gender = ['M', 'F', 'M', 'F', 'M', 'F'];
const preference = ['A', 'B', 'A', 'A', 'B', 'B'];

const table = datly.contingencyTable(gender, preference);
console.log(table);
// {
//   table: {
//     M: { A: 2, B: 1 },
//     F: { A: 1, B: 2 }
//   },
//   totals: {
//     row: { M: 3, F: 3 },
//     col: { A: 3, B: 3 },
//     grand: 6
//   },
//   rows: ['M', 'F'],
//   columns: ['A', 'B']
// }
```

---

### 4. Central Tendency

Measures of central tendency (mean, median, mode).

#### Methods

##### `mean(data)`
Calculate arithmetic mean.

```javascript
const data = [10, 20, 30, 40, 50];
const avg = datly.mean(data);
console.log(avg); // 30
```

---

##### `median(data)`
Calculate median (middle value).

```javascript
const data1 = [1, 3, 5, 7, 9];
console.log(datly.median(data1)); // 5

const data2 = [1, 3, 5, 7];
console.log(datly.median(data2)); // 4 (average of 3 and 5)
```

---

##### `mode(data)`
Find mode (most frequent value).

```javascript
const data = [1, 2, 2, 3, 3, 3, 4, 4];
const modeResult = datly.mode(data);
console.log(modeResult);
// {
//   values: [3],
//   frequency: 3,
//   isMultimodal: false,
//   isUniform: false
// }

// Multimodal example
const data2 = [1, 1, 2, 2, 3];
const modeResult2 = datly.mode(data2);
console.log(modeResult2);
// {
//   values: [1, 2],
//   frequency: 2,
//   isMultimodal: true,
//   isUniform: false
// }
```

---

##### `geometricMean(data)`
Calculate geometric mean (for positive values).

```javascript
const data = [2, 8, 32]; // Growth rates
const geoMean = datly.geometricMean(data);
console.log(geoMean); // 8 (∛(2×8×32))
```

**Use cases:** Growth rates, ratios, percentages

---

##### `harmonicMean(data)`
Calculate harmonic mean (for rates).

```javascript
const speeds = [60, 40, 30]; // km/h on different segments
const harmMean = datly.harmonicMean(speeds);
console.log(harmMean); // 40 (average speed)
```

**Use cases:** Average rates, speeds, ratios

---

##### `trimmedMean(data, percentage)`
Calculate trimmed mean (remove extreme values).

```javascript
const data = [1, 2, 3, 100, 4, 5, 6]; // 100 is outlier
const trimmed = datly.trimmedMean(data, 10); // Trim 10% from each end
console.log(trimmed); // ~3.75 (without extreme values)
```

---

##### `weightedMean(values, weights)`
Calculate weighted average.

```javascript
const grades = [85, 90, 78, 92];
const weights = [0.3, 0.3, 0.2, 0.2]; // Exam weights
const weightedGrade = datly.weightedMean(grades, weights);
console.log(weightedGrade); // 86.5
```eoMean); // 8 (∛(2×8×32))
```

**Use cases:** Growth rates, ratios, percentages

---

##### `harmonicMean(data)`
Calculate harmonic mean (for rates).

```javascript
const speeds = [60, 40, 30]; // km/h on different segments
const harmMean = datly.centralTendency.harmonicMean(speeds);
console.log(harmMean); // 40 (average speed)
```

**Use cases:** Average rates, speeds, ratios

---

##### `trimmedMean(data, percentage)`
Calculate trimmed mean (remove extreme values).

```javascript
const data = [1, 2, 3, 100, 4, 5, 6]; // 100 is outlier
const trimmed = datly.centralTendency.trimmedMean(data, 10); // Trim 10% from each end
console.log(trimmed); // ~3.75 (without extreme values)
```

---

##### `weightedMean(values, weights)`
Calculate weighted average.

```javascript
const grades = [85, 90, 78, 92];
const weights = [0.3, 0.3, 0.2, 0.2]; // Exam weights
const weightedGrade = datly.centralTendency.weightedMean(grades, weights);
console.log(weightedGrade); // 86.5
```

---

### 5. Dispersion Measures

Measures of variability and spread.

#### Methods

##### `variance(data, sample)`
Calculate variance.

```javascript
const data = [2, 4, 6, 8, 10];

// Sample variance (default)
const sampleVar = datly.variance(data, true);
console.log(sampleVar); // 10

// Population variance
const popVar = datly.variance(data, false);
console.log(popVar); // 8
```

---

##### `standardDeviation(data, sample)`
Calculate standard deviation.

```javascript
const data = [2, 4, 6, 8, 10];
const std = datly.standardDeviation(data);
console.log(std); // 3.162 (√10)
```

---

##### `range(data)`
Calculate range (max - min).

```javascript
const data = [5, 10, 15, 20, 25];
const rangeResult = datly.range(data);
console.log(rangeResult);
// {
//   range: 20,
//   min: 5,
//   max: 25
// }
```

---

##### `interquartileRange(data)`
Calculate IQR (Q3 - Q1).

```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const iqr = datly.interquartileRange(data);
console.log(iqr);
// {
//   iqr: 4,
//   q1: 3,
//   q3: 7
// }
```

---

##### `coefficientOfVariation(data)`
Calculate coefficient of variation (CV%).

```javascript
const data = [10, 20, 30, 40, 50];
const cv = datly.coefficientOfVariation(data);
console.log(cv);
// {
//   cv: 0.471,
//   cvPercent: 47.1
// }
```

**Interpretation:**
- CV < 15%: Low variability
- CV 15-30%: Moderate variability
- CV > 30%: High variability

---

##### `meanAbsoluteDeviation(data)`
Calculate MAD (mean absolute deviation).

```javascript
const data = [2, 4, 6, 8, 10];
const mad = datly.meanAbsoluteDeviation(data);
console.log(mad);
// {
//   mad: 2.4,
//   mean: 6
// }
```

---

##### `standardError(data)`
Calculate standard error of the mean.

```javascript
const data = [10, 12, 14, 16, 18, 20];
const se = datly.standardError(data);
console.log(se); // 1.29 (σ/√n)
```

---

### 6. Position Measures

Quantiles, percentiles, and ranking.

#### Methods

##### `quantile(data, q)`
Calculate quantile.

```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Median (0.5 quantile)
console.log(datly.quantile(data, 0.5)); // 5.5

// First quartile
console.log(datly.quantile(data, 0.25)); // 3.25

// Third quartile
console.log(datly.quantile(data, 0.75)); // 7.75
```

---

##### `percentile(data, p)`
Calculate percentile (0-100 scale).

```javascript
const scores = [65, 70, 75, 80, 85, 90, 95];
const p90 = datly.percentile(scores, 90);
console.log(p90); // 93.5
```

---

##### `quartiles(data)`
Calculate all quartiles.

```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const quartiles = datly.quartiles(data);
console.log(quartiles);
// {
//   q1: 2.5,
//   q2: 5,    // Median
//   q3: 7.5,
//   iqr: 5
// }
```

---

##### `percentileRank(data, value)`
Calculate percentile rank of a value.

```javascript
const scores = [60, 70, 80, 90, 100];
const rank = datly.percentileRank(scores, 80);
console.log(rank); // 50 (80 is at the 50th percentile)
```

---

##### `zScore(data, value)`
Calculate z-score (standardized value).

```javascript
const data = [10, 20, 30, 40, 50];
const z = datly.zScore(data, 40);
console.log(z); // 0.632 standard deviations above mean
```

**Interpretation:**
- |z| < 1: Within 1 standard deviation
- |z| < 2: Within 2 standard deviations
- |z| > 3: Potential outlier

---

##### `boxplotStats(data)`
Calculate box plot statistics.

```javascript
const data = [1, 2, 3, 4, 5, 100]; // 100 is outlier
const stats = datly.boxplotStats(data);
console.log(stats);
// {
//   min: 1,
//   q1: 2,
//   median: 3.5,
//   q3: 5,
//   max: 5,
//   iqr: 3,
//   lowerFence: -2.5,
//   upperFence: 9.5,
//   outliers: [100],
//   outlierCount: 1
// }
```

---

##### `rank(data, method)`
Rank data values.

```javascript
const data = [10, 20, 20, 30, 40];

// Average ranking (ties get average rank)
const ranks1 = datly.rank(data, 'average');
console.log(ranks1); // [1, 2.5, 2.5, 4, 5]

// Min ranking (ties get minimum rank)
const ranks2 = datly.rank(data, 'min');
console.log(ranks2); // [1, 2, 2, 4, 5]

// Max ranking (ties get maximum rank)
const ranks3 = datly.rank(data, 'max');
console.log(ranks3); // [1, 3, 3, 4, 5]
```

---

### 7. Shape Analysis

Distribution shape: skewness and kurtosis.

#### Methods

##### `skewness(data, bias)`
Calculate skewness (measure of asymmetry).

```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]; // Right-skewed

// Biased estimate (default)
const skew1 = datly.skewness(data, true);
console.log(skew1); // Positive value

// Unbiased estimate
const skew2 = datly.skewness(data, false);
console.log(skew2);
```

**Interpretation:**
- Skewness < -1: Highly left-skewed
- -1 < Skewness < -0.5: Moderately left-skewed
- -0.5 < Skewness < 0.5: Approximately symmetric
- 0.5 < Skewness < 1: Moderately right-skewed
- Skewness > 1: Highly right-skewed

---

##### `kurtosis(data, bias, excess)`
Calculate kurtosis (measure of tailedness).

```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9];

// Excess kurtosis (default)
const kurt1 = datly.kurtosis(data, false, true);
console.log(kurt1); // -3 adjustment applied

// Regular kurtosis
const kurt2 = datly.kurtosis(data, false, false);
console.log(kurt2);
```

**Interpretation (excess kurtosis):**
- Excess < -1: Platykurtic (thin tails)
- -1 < Excess < 1: Mesokurtic (normal)
- Excess > 1: Leptokurtic (fat tails)

---

##### `isNormalDistribution(data, alpha)`
Test if data follows normal distribution.

```javascript
const normalData = [2, 3, 4, 4, 5, 5, 5, 6, 6, 7];
const test = datly.isNormalDistribution(normalData, 0.05);
console.log(test);
// {
//   shapiroWilk: { statistic: 0.95, pValue: 0.12, isNormal: true },
//   jarqueBera: { statistic: 1.23, pValue: 0.54, isNormal: true },
//   skewness: 0.12,
//   kurtosis: -0.34,
//   isNormalByTests: true
// }
```

---

##### `jarqueBeraTest(data, alpha)`
Jarque-Bera normality test.

```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const jb = datly.jarqueBeraTest(data);
console.log(jb);
// {
//   statistic: 0.84,
//   pValue: 0.66,
//   skewness: 0,
//   excessKurtosis: -1.2,
//   isNormal: true
// }
```

---

### 8. Hypothesis Testing

Statistical hypothesis tests.

#### Methods

##### `tTest(sample1, sample2, type, alpha)`
Perform t-test.

```javascript
// One-sample t-test
const sample = [23, 25, 27, 29, 31];
const populationMean = 25;
const oneSample = datly.tTest(sample, populationMean, 'one-sample');
console.log(oneSample);
// {
//   type: 'one-sample',
//   statistic: 1.89,
//   pValue: 0.13,
//   degreesOfFreedom: 4,
//   significant: false
// }

// Two-sample t-test
const group1 = [23, 25, 27, 29, 31];
const group2 = [28, 30, 32, 34, 36];
const twoSample = datly.tTest(group1, group2, 'two-sample');
console.log(twoSample);
// {
//   type: 'two-sample',
//   statistic: -3.46,
//   pValue: 0.008,
//   significant: true,
//   meanDifference: -5
// }

// Paired t-test
const before = [120, 125, 130, 128, 122];
const after = [115, 118, 125, 120, 115];
const paired = datly.tTest(before, after, 'paired');
console.log(paired);
```

---

##### `zTest(sample, populationMean, populationStd, alpha)`
Perform z-test (known population variance).

```javascript
const sample = [105, 110, 108, 112, 115];
const popMean = 100;
const popStd = 15;

const zTest = datly.zTest(sample, popMean, popStd);
console.log(zTest);
// {
//   type: 'z-test',
//   statistic: 1.49,
//   pValue: 0.136,
//   significant: false,
//   sampleMean: 110
// }
```

---

##### `anovaTest(groups, alpha)`
One-way ANOVA test.

```javascript
const groupA = [23, 25, 27, 29];
const groupB = [30, 32, 34, 36];
const groupC = [28, 30, 32, 34];

const anova = datly.anovaTest([groupA, groupB, groupC]);
console.log(anova);
// {
//   type: 'one-way-anova',
//   statistic: 12.45,
//   pValue: 0.001,
//   dfBetween: 2,
//   dfWithin: 9,
//   significant: true,
//   groupMeans: [26, 33, 31]
// }
```

---

##### `chiSquareTest(column1, column2, alpha)`
Chi-square test of independence.

```javascript
const gender = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'];
const preference = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'B'];

const chiTest = datly.chiSquareTest(gender, preference);
console.log(chiTest);
// {
//   type: 'chi-square-independence',
//   statistic: 0.5,
//   pValue: 0.48,
//   degreesOfFreedom: 1,
//   significant: false,
//   cramersV: 0.25
// }
```

---

##### `mannWhitneyTest(sample1, sample2, alpha)`
Mann-Whitney U test (non-parametric alternative to t-test).

```javascript
const group1 = [1, 2, 3, 4, 5];
const group2 = [6, 7, 8, 9, 10];

const mwTest = datly.mannWhitneyTest(group1, group2);
console.log(mwTest);
// {
//   type: 'mann-whitney-u',
//   statistic: 0,
//   u1: 0,
//   u2: 25,
//   zStatistic: -2.61,
//   pValue: 0.009,
//   significant: true
// }
```

---

### 9. Confidence Intervals

Estimate population parameters with confidence.

#### Methods

##### `mean(data, confidence)`
Confidence interval for mean.

```javascript
const data = [23, 25, 27, 29, 31, 33, 35];
const ci = datly.confidenceInterval(data, 0.95);
console.log(ci);
// {
//   mean: 29,
//   standardError: 1.63,
//   marginOfError: 4.03,
//   lowerBound: 24.97,
//   upperBound: 33.03,
//   confidence: 0.95,
//   degreesOfFreedom: 6
// }
```

---

##### `proportion(successes, total, confidence)`
Confidence interval for proportion.

```javascript
const successes = 45; // Number of successes
const total = 100;    // Total trials

const ci = datly.confidenceIntervals.proportion(successes, total, 0.95);
console.log(ci);
// {
//   normal: {
//     proportion: 0.45,
//     lowerBound: 0.353,
//     upperBound: 0.547,
//     confidence: 0.95
//   },
//   wilson: {
//     proportion: 0.45,
//     center: 0.453,
//     lowerBound: 0.355,
//     upperBound: 0.551,
//     confidence: 0.95
//   },
//   recommended: {...} // Wilson interval for small samples
// }
```

---

##### `variance(data, confidence)`
Confidence interval for variance.

```javascript
const data = [10, 12, 14, 16, 18, 20];
const ci = datly.confidenceIntervals.variance(data, 0.95);
console.log(ci);
// {
//   sampleVariance: 14.67,
//   lowerBound: 6.23,
//   upperBound: 73.33,
//   confidence: 0.95
// }
```

---

##### `meanDifference(sample1, sample2, confidence, equalVariances)`
Confidence interval for difference between two means.

```javascript
const before = [120, 125, 130, 128, 122];
const after = [115, 118, 125, 120, 115];

const ci = datly.confidenceIntervals.meanDifference(before, after, 0.95);
console.log(ci);
// {
//   meanDifference: 9.2,
//   sample1Mean: 125,
//   sample2Mean: 115.8,
//   lowerBound: 3.5,
//   upperBound: 14.9,
//   confidence: 0.95
// }
```

---

##### `correlation(x, y, confidence, method)`
Confidence interval for correlation coefficient.

```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 5, 4, 5];

const ci = datly.confidenceIntervals.correlation(x, y, 0.95);
console.log(ci);
// {
//   correlation: 0.775,
//   fisherZ: 1.033,
//   lowerBound: 0.034,
//   upperBound: 0.965,
//   confidence: 0.95
// }
```

---

##### `bootstrapCI(data, statistic, confidence, iterations)`
Bootstrap confidence interval.

```javascript
const data = [23, 25, 27, 29, 31];

// Bootstrap CI for median
const ci = datly.confidenceIntervals.bootstrapCI(data, 'median', 0.95, 1000);
console.log(ci);
// {
//   originalStatistic: 27,
//   bootstrapMean: 27.1,
//   bias: 0.1,
//   standardError: 1.4,
//   lowerBound: 24.5,
//   upperBound: 29.8,
//   confidence: 0.95,
//   iterations: 1000
// }
```

---

### 10. Normality Tests

Test if data follows normal distribution.

#### Methods

##### `shapiroWilk(data, alpha)`
Shapiro-Wilk test (most powerful for small samples).

```javascript
const data = [2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5];
const sw = datly.normalityTests.shapiroWilk(data);
console.log(sw);
// {
//   statistic: 0.96,
//   pValue: 0.82,
//   isNormal: true,
//   alpha: 0.05,
//   interpretation: "Fail to reject null hypothesis..."
// }
```

**Best for:** Sample sizes 3-5000

---

##### `kolmogorovSmirnov(data, alpha)`
Kolmogorov-Smirnov test.

```javascript
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const ks = datly.normalityTests.kolmogorovSmirnov(data);
console.log(ks);
// {
//   statistic: 0.15,
//   pValue: 0.95,
//   isNormal: true,
//   lambda: 0.47
// }
```

**Best for:** Large samples, continuous data

---

##### `andersonDarling(data, alpha)`
Anderson-Darling test (sensitive to tails).

```javascript
const data = [10, 12, 14, 15, 16, 18, 20];
const ad = datly.normalityTests.andersonDarling(data);
console.log(ad);
// {
//   statistic: 0.234,
//   adjustedStatistic: 0.247,
//   pValue: 0.75,
//   isNormal: true
// }
```

**Best for:** Detecting tail deviations

---

##### `jarqueBera(data, alpha)`
Jarque-Bera test (based on skewness and kurtosis).

```javascript
const data = [5, 6, 7, 8, 9, 10, 11, 12];
const jb = datly.normalityTests.jarqueBera(data);
console.log(jb);
// {
//   statistic: 0.45,
//   pValue: 0.80,
//   skewness: 0,
//   excessKurtosis: -1.2,
//   isNormal: true
// }
```

**Best for:** Large samples (n > 30)

---

##### `dagoTest(data, alpha)`
D'Agostino K² test.

```javascript
const data = Array.from({ length: 50 }, () => Math.random() * 10);
const dago = datly.normalityTests.dagoTest(data);
console.log(dago);
// {
//   statistic: 2.34,
//   pValue: 0.31,
//   isNormal: true,
//   skewness: 0.23,
//   excessKurtosis: -0.45
// }
```

**Best for:** Medium to large samples (n ≥ 20)

---

##### `batchNormalityTest(data, alpha)`
Run all normality tests at once.

```javascript
const data = [12, 14, 15, 16, 17, 18, 20, 22];
const batch = datly.normalityTests.batchNormalityTest(data);
console.log(batch);
// {
//   individualTests: {
//     shapiroWilk: {...},
//     jarqueBera: {...},
//     andersonDarling: {...},
//     kolmogorovSmirnov: {...}
//   },
//   summary: {
//     testsRun: 4,
//     testsPassingNormality: 4,
//     consensusNormal: true,
//     strongNormalEvidence: true
//   },
//   recommendation: "Strong evidence for normality..."
// }
```

---

### 11. Correlation Analysis

Measure relationships between variables.

#### Methods

##### `pearson(x, y)`
Pearson correlation (linear relationships).

```javascript
const height = [160, 165, 170, 175, 180];
const weight = [55, 60, 65, 70, 75];

const corr = datly.correlation.pearson(height, weight);
console.log(corr);
// {
//   correlation: 1.0,
//   pValue: 0.000,
//   tStatistic: Infinity,
//   significant: true,
//   confidenceInterval: { lower: 1.0, upper: 1.0 }
// }
```

**Range:** -1 (perfect negative) to +1 (perfect positive)

**Interpretation:**
- |r| < 0.3: Weak
- 0.3 ≤ |r| < 0.7: Moderate
- |r| ≥ 0.7: Strong

---

##### `spearman(x, y)`
Spearman correlation (monotonic relationships, rank-based).

```javascript
const x = [1, 2, 3, 4, 5];
const y = [1, 4, 9, 16, 25]; // Non-linear but monotonic

const corr = datly.correlation.spearman(x, y);
console.log(corr);
// {
//   correlation: 1.0,
//   pValue: 0.000,
//   significant: true,
//   xRanks: [1, 2, 3, 4, 5],
//   yRanks: [1, 2, 3, 4, 5]
// }
```

**Use when:** Data is ordinal or non-linear

---

##### `kendall(x, y)`
Kendall's Tau correlation.

```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 1, 4, 3, 5];

const corr = datly.correlation.kendall(x, y);
console.log(corr);
// {
//   correlation: 0.6,
//   pValue: 0.142,
//   zStatistic: 1.47,
//   concordantPairs: 8,
//   discordantPairs: 2,
//   significant: false
// }
```

**Use when:** Small sample sizes, ordinal data

---

##### `matrix(dataset, method)`
Correlation matrix for multiple variables.

```javascript
const data = {
  headers: ['age', 'income', 'spending'],
  data: [
    { age: 25, income: 30000, spending: 20000 },
    { age: 30, income: 40000, spending: 25000 },
    { age: 35, income: 50000, spending: 30000 },
    { age: 40, income: 60000, spending: 35000 }
  ]
};

const matrix = datly.correlation.matrix(data, 'pearson');
console.log(matrix);
// {
//   correlations: {
//     age: { age: 1, income: 1, spending: 1 },
//     income: { age: 1, income: 1, spending: 1 },
//     spending: { age: 1, income: 1, spending: 1 }
//   },
//   pValues: {...},
//   strongCorrelations: [
//     { variable1: 'age', variable2: 'income', correlation: 1.0 }
//   ]
// }
```

---

##### `partialCorrelation(x, y, z)`
Partial correlation (controlling for third variable).

```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 3, 4, 5, 6];
const z = [1, 1, 2, 2, 3]; // Control variable

const partial = datly.correlation.partialCorrelation(x, y, z);
console.log(partial);
// {
//   correlation: 0.95,
//   pValue: 0.013,
//   significant: true,
//   controllingFor: 'third variable'
// }
```

---

##### `covariance(x, y, sample)`
Calculate covariance.

```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 6, 8, 10];

// Sample covariance
const cov = datly.correlation.covariance(x, y, true);
console.log(cov);
// {
//   covariance: 5,
//   meanX: 3,
//   meanY: 6,
//   sampleSize: 5
// }
```

---

### 12. Regression Analysis

Model relationships and make predictions.

#### Methods

##### `linear(x, y)`
Simple linear regression.

```javascript
const x = [1, 2, 3, 4, 5];
const y = [2, 4, 5, 4, 5];

const model = datly.regression.linear(x, y);
console.log(model);
// {
//   slope: 0.6,
//   intercept: 2.2,
//   rSquared: 0.46,
//   adjustedRSquared: 0.28,
//   equation: 'y = 2.2000 + 0.6000x',
//   pValueSlope: 0.158,
//   pValueModel: 0.158,
//   residuals: [-0.2, 0.2, 0.4, -1.0, 0.6],
//   predicted: [2.8, 3.4, 4.0, 4.6, 5.2]
// }
```

---

##### `multiple(dataset, dependent, independents)`
Multiple linear regression.

```javascript
const data = {
  headers: ['sales', 'advertising', 'price', 'competition'],
  data: [
    { sales: 100, advertising: 10, price: 50, competition: 3 },
    { sales: 150, advertising: 15, price: 45, competition: 2 },
    { sales: 120, advertising: 12, price: 48, competition: 4 },
    { sales: 180, advertising: 20, price: 40, competition: 1 }
  ]
};

const model = datly.regression.multiple(
  data,
  'sales',
  ['advertising', 'price', 'competition']
);

console.log(model);
// {
//   coefficients: [
//     { variable: 'Intercept', coefficient: 50, pValue: 0.05 },
//     { variable: 'advertising', coefficient: 5.5, pValue: 0.01 },
//     { variable: 'price', coefficient: -2.1, pValue: 0.03 },
//     { variable: 'competition', coefficient: -10, pValue: 0.02 }
//   ],
//   rSquared: 0.95,
//   adjustedRSquared: 0.90,
//   fStatistic: 19.0,
//   pValueModel: 0.001,
//   equation: 'y = 50.0000 + 5.5000*advertising + -2.1000*price + -10.0000*competition'
// }
```

---

##### `polynomial(x, y, degree)`
Polynomial regression.

```javascript
const x = [1, 2, 3, 4, 5];
const y = [1, 4, 9, 16, 25]; // y = x²

const model = datly.regression.polynomial(x, y, 2);
console.log(model);
// {
//   coefficients: [0, 0, 1], // y = 0 + 0x + 1x²
//   degree: 2,
//   rSquared: 1.0,
//   equation: 'y = 0.0000 + 0.0000*x + 1.0000*x^2',
//   predicted: [1, 4, 9, 16, 25]
// }
```

---

##### `logistic(x, y, maxIterations, tolerance)`
Logistic regression (binary classification).

```javascript
const x = [1, 2, 3, 4, 5, 6];
const y = [0, 0, 0, 1, 1, 1]; // Binary outcome

const model = datly.regression.logistic(x, y);
console.log(model);
// {
//   intercept: -3.5,
//   slope: 1.2,
//   probabilities: [0.12, 0.23, 0.38, 0.55, 0.70, 0.81],
//   predicted: [0, 0, 0, 1, 1, 1],
//   accuracy: 1.0,
//   logLikelihood: -2.1,
//   mcFaddenR2: 0.68
// }
```

---

##### `predict(model, x)`
Make predictions using fitted model.

```javascript
// After fitting a model
const newX = [6, 7, 8];
const predictions = datly.regression.predict(model, newX);
console.log(predictions); // [5.8, 6.4, 7.0]
```

---

### 13. Report Generation

Generate comprehensive statistical reports.

#### Methods

##### `summary(dataset)`
Generate complete statistical summary.

```javascript
const data = {
  headers: ['age', 'income', 'department'],
  data: [
    { age: 25, income: 30000, department: 'Sales' },
    { age: 30, income: 45000, department: 'IT' },
    { age: 35, income: 50000, department: 'Sales' },
    { age: 40, income: 60000, department: 'IT' }
  ],
  length: 4,
  columns: 3
};

const report = datly.reportGenerator.summary(data);
console.log(report);
// {
//   title: 'Statistical Summary Report',
//   generatedAt: '2025-01-15T10:30:00.000Z',
//   basicInfo: {
//     totalRows: 4,
//     totalColumns: 3,
//     headers: ['age', 'income', 'department']
//   },
//   columnAnalysis: {
//     age: {
//       type: 'numeric',
//       mean: 32.5,
//       median: 32.5,
//       min: 25,
//       max: 40,
//       standardDeviation: 6.45
//     },
//     income: {
//       type: 'numeric',
//       mean: 46250,
//       median: 47500,
//       ...
//     },
//     department: {
//       type: 'categorical',
//       categories: [...],
//       mostFrequent: { value: 'Sales', frequency: 2 }
//     }
//   },
//   dataQuality: {
//     overallScore: 95,
//     completenessScore: 100,
//     consistencyScore: 90
//   },
//   keyInsights: [
//     {
//       type: 'correlation',
//       title: 'Strong correlation between age and income',
//       importance: 8
//     }
//   ],
//   recommendations: [...]
// }
```

---

##### `exportSummary(summary, format)`
Export report in different formats.

```javascript
const report = datly.reportGenerator.summary(data);

// Export as JSON
const json = datly.reportGenerator.exportSummary(report, 'json');

// Export as text
const text = datly.reportGenerator.exportSummary(report, 'text');
console.log(text);
// STATISTICAL SUMMARY REPORT
// Generated: 1/15/2025, 10:30:00 AM
// ==================================================
//
// BASIC INFORMATION
// --------------------
// Rows: 4
// Columns: 3
// ...

// Export as CSV
const csv = datly.reportGenerator.exportSummary(report, 'csv');
```

---

### 14. Pattern Detection

Automatically detect patterns in data.

#### Methods

##### `analyze(dataset)`
Comprehensive pattern analysis.

```javascript
const data = {
  headers: ['date', 'sales', 'temperature'],
  data: [
    { date: '2024-01-01', sales: 100, temperature: 20 },
    { date: '2024-01-02', sales: 110, temperature: 22 },
    { date: '2024-01-03', sales: 105, temperature: 21 },
    { date: '2024-01-04', sales: 115, temperature: 23 }
  ],
  length: 4,
  columns: 3
};

const patterns = datly.patternDetector.analyze(data);
console.log(patterns);
// {
//   patterns: {
//     trends: [
//       {
//         column: 'sales',
//         direction: 'increasing',
//         slope: 5,
//         rSquared: 0.75,
//         strength: 'strong'
//       }
//     ],
//     seasonality: [...],
//     outliers: [...],
//     correlations: {
//       strongCorrelations: [
//         {
//           variable1: 'sales',
//           variable2: 'temperature',
//           correlation: 0.95,
//           strength: 'very_strong'
//         }
//       ]
//     },
//     distributions: [...],
//     clustering: [...],
//     temporal: [...]
//   },
//   insights: [
//     {
//       type: 'trend',
//       importance: 'high',
//       message: 'Found 1 strong trend(s) in your data',
//       details: ['sales: increasing trend']
//     }
//   ]
// }
```

---

### 15. Result Interpretation

Interpret statistical test results in plain language.

#### Methods

##### `interpret(testResult)`
Interpret any statistical test result.

```javascript
// After performing a t-test
const tTestResult = datly.hypothesisTesting.tTest(group1, group2);

const interpretation = datly.interpreter.interpret(tTestResult);
console.log(interpretation);
// {
//   testType: 't-test',
//   summary: 'significant difference between groups (t = -3.46, p = 0.008)',
//   conclusion: {
//     decision: 'reject_null',
//     statement: 'At the 95% confidence level, we reject the null hypothesis',
//     pValue: 0.008,
//     confidenceLevel: 95
//   },
//   significance: {
//     level: 'strong',
//     pValue: 0.008,
//     interpretation: 'Strong evidence against null hypothesis',
//     isSignificant: true
//   },
//   effectSize: {
//     value: 0.85,
//     magnitude: 'Large',
//     interpretation: 'large effect size'
//   },
//   plainLanguage: '✓ SIGNIFICANT RESULT: Found a meaningful difference between the groups. (p-value: 0.0080)',
//   recommendations: [
//     'Very strong result - investigate practical significance',
//     'Replicate findings with independent data when possible'
//   ]
// }
```

---

### 16. Auto-Analysis

Automated end-to-end analysis.

#### Methods

##### `autoAnalyze(dataset, options)`
Perform comprehensive automatic analysis.

```javascript
const data = {
  headers: ['age', 'income', 'gender', 'purchase'],
  data: [
    { age: 25, income: 30000, gender: 'M', purchase: 100 },
    { age: 30, income: 45000, gender: 'F', purchase: 150 },
    { age: 35, income: 50000, gender: 'M', purchase: 120 },
    { age: 40, income: 60000, gender: 'F', purchase: 180 },
    // ... more data
  ],
  length: 100,
  columns: 4
};

const analysis = datly.autoAnalyzer.autoAnalyze(data, {
  minCorrelationThreshold: 0.3,
  significanceLevel: 0.05,
  generateVisualizations: true,
  includeAdvancedAnalysis: true
});

console.log(analysis);
// {
//   metadata: {
//     analysisDate: '2025-01-15T10:30:00.000Z',
//     datasetSize: 100,
//     columnsAnalyzed: 4
//   },
//   variableClassification: {
//     quantitative: [
//       { name: 'age', type: 'quantitative', subtype: 'discrete' },
//       { name: 'income', type: 'quantitative', subtype: 'continuous' },
//       { name: 'purchase', type: 'quantitative', subtype: 'continuous' }
//     ],
//     qualitative: [
//       { name: 'gender', type: 'binary', categories: ['M', 'F'] }
//     ]
//   },
//   descriptiveStatistics: {
//     age: { mean: 32.5, median: 32, std: 6.45, ... },
//     income: { mean: 46250, median: 47500, ... },
//     purchase: { mean: 137.5, median: 135, ... }
//   },
//   correlationAnalysis: {
//     strongCorrelations: [
//       {
//         variable1: 'income',
//         variable2: 'purchase',
//         correlation: 0.92,
//         significance: true
//       }
//     ]
//   },
//   regressionAnalysis: {
//     models: [
//       {
//         independent: 'income',
//         dependent: 'purchase',
//         rSquared: 0.85,
//         significant: true,
//         equation: 'y = 10.5 + 0.0025*income'
//       }
//     ]
//   },
//   distributionAnalysis: {
//     age: {
//       isNormal: true,
//       normalityPValue: 0.15,
//       skewness: 0.12,
//       distributionType: 'normal'
//     }
//   },
//   outlierAnalysis: {
//     income: {
//       count: 2,
//       percentage: 2,
//       severity: 'low'
//     }
//   },
//   insights: [
//     {
//       category: 'overview',
//       priority: 'high',
//       title: 'Dataset Composition',
//       description: 'Dataset with 100 records, 3 numeric and 1 categorical variables',
//       icon: '📊'
//     },
//     {
//       category: 'correlation',
//       priority: 'high',
//       title: 'Very strong correlation between income and purchase',
//       description: 'Positive correlation of 0.920',
//       icon: '🔗'
//     }
//   ],
//   visualizationSuggestions: [
//     {
//       type: 'scatter',
//       variables: ['income', 'purchase'],
//       priority: 'high',
//       title: 'income vs purchase'
//     },
//     {
//       type: 'histogram',
//       variable: 'age',
//       priority: 'medium',
//       title: 'Distribution of age'
//     }
//   ],
//   summary: {
//     totalInsights: 8,
//     highPriorityInsights: 3,
//     keyFindings: [...],
//     recommendations: [
//       'Explore correlations identified for possible predictive modeling',
//       'Consider transformations for non-normal distributions'
//     ]
//   }
// }
```

---

### 17. Machine Learning

Build and train ML models.

#### Creating Models

##### Linear Regression

```javascript
// Create model
const model = datly.ml.createLinearRegression({
  learningRate: 0.01,
  iterations: 1000,
  regularization: 'l2', // 'l1', 'l2', or null
  lambda: 0.01
});

// Prepare data
const X = [
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5]
];
const y = [3, 5, 7, 9];

// Train model
model.fit(X, y, true); // true = normalize features

// Make predictions
const predictions = model.predict([[5, 6]]);
console.log(predictions); // [11]

// Evaluate model
const score = model.score(X, y);
console.log(score);
// {
//   r2Score: 1.0,
//   mse: 0.0,
//   rmse: 0.0,
//   mae: 0.0
// }
```

---

##### Logistic Regression

```javascript
// Create model for classification
const model = datly.ml.createLogisticRegression({
  learningRate: 0.01,
  iterations: 1000
});

// Binary classification data
const X = [
  [1, 2], [2, 3], [3, 4], [4, 5],
  [5, 6], [6, 7], [7, 8], [8, 9]
];
const y = [0, 0, 0, 0, 1, 1, 1, 1];

// Train
model.fit(X, y);

// Predict classes
const predictions = model.predict([[3.5, 4.5], [7, 8]]);
console.log(predictions); // [0, 1]

// Predict probabilities
const probabilities = model.predictProba([[3.5, 4.5]]);
console.log(probabilities); // [{ 0: 0.62, 1: 0.38 }]

// Evaluate
const score = model.score(X, y);
console.log(score);
// {
//   accuracy: 1.0,
//   confusionMatrix: {...},
//   classMetrics: {...}
// }
```

---

##### K-Nearest Neighbors (KNN)

```javascript
// Create KNN model
const model = datly.ml.createKNN({
  k: 5,
  metric: 'euclidean', // 'euclidean', 'manhattan', 'minkowski'
  weights: 'uniform'   // 'uniform' or 'distance'
});

// Prepare data
const X = [
  [1, 2], [2, 3], [3, 4],
  [6, 7], [7, 8], [8, 9]
];
const y = [0, 0, 0, 1, 1, 1]; // Classes

// Train (KNN just stores the data)
model.fit(X, y, true, 'classification');

// Predict
const predictions = model.predict([[2, 2], [7, 7]]);
console.log(predictions); // [0, 1]

// Predict with probabilities
const proba = model.predictProba([[4, 5]]);
console.log(proba); // [{ 0: 0.4, 1: 0.6 }]
```

---

##### Decision Tree

```javascript
// Create decision tree
const model = datly.ml.createDecisionTree({
  maxDepth: 10,
  minSamplesSplit: 2,
  minSamplesLeaf: 1,
  criterion: 'gini' // 'gini' or 'entropy'
});

// Train
const X = [
  [2.5], [3.5], [4.5], [5.5], [6.5], [7.5]
];
const y = [0, 0, 1, 1, 2, 2]; // Multi-class

model.fit(X, y, 'classification');

// Predict
const predictions = model.predict([[3.0], [6.0]]);
console.log(predictions); // [0, 1]

// Get feature importance
const importance = model.getFeatureImportance();
console.log(importance); // { feature_0: 1.0 }

// Model summary
const summary = model.summary();
console.log(summary);
// {
//   modelType: 'Decision Tree',
//   taskType: 'classification',
//   trainingMetrics: {
//     treeDepth: 3,
//     leafCount: 6,
//     nodeCount: 11
//   }
// }
```

---

##### Random Forest

```javascript
// Create random forest
const model = datly.ml.createRandomForest({
  nEstimators: 100,    // Number of trees
  maxDepth: 10,
  minSamplesSplit: 2,
  maxFeatures: 'sqrt', // 'sqrt', 'log2', or number
  bootstrap: true
});

// Train
const X = [
  [1, 2], [2, 3], [3, 4], [4, 5],
  [5, 6], [6, 7], [7, 8], [8, 9]
];
const y = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'];

model.fit(X, y, 'classification');

// Predict
const predictions = model.predict([[2.5, 3.5], [7, 8]]);
console.log(predictions); // ['A', 'C']

// Get feature importance
const importance = model.getFeatureImportance();
console.log(importance); // [0.6, 0.4]
```

---

##### Naive Bayes

```javascript
// Create Naive Bayes classifier
const model = datly.ml.createNaiveBayes({
  type: 'gaussian' // 'gaussian', 'multinomial', or 'bernoulli'
});

// Train
const X = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8]
];
const y = ['spam', 'spam', 'spam', 'ham', 'ham', 'ham'];

model.fit(X, y);

// Predict
const predictions = model.predict([[2, 2], [6, 6]]);
console.log(predictions); // ['spam', 'ham']

// Predict probabilities
const proba = model.predictProba([[4, 5]]);
console.log(proba); // [{ spam: 0.3, ham: 0.7 }]
```

---

##### Support Vector Machine (SVM)

```javascript
// Create SVM
const model = datly.ml.createSVM({
  C: 1.0,              // Regularization parameter
  kernel: 'linear',    // 'linear', 'rbf', 'poly'
  gamma: 'scale',      // 'scale', 'auto', or number
  degree: 3,           // For polynomial kernel
  learningRate: 0.001,
  iterations: 1000
});

// Train
const X = [
  [1, 2], [2, 3], [3, 4],
  [6, 7], [7, 8], [8, 9]
];
const y = [0, 0, 0, 1, 1, 1];

model.fit(X, y);

// Predict
const predictions = model.predict([[2, 2], [7, 7]]);
console.log(predictions); // [0, 1]

// Summary
const summary = model.summary();
console.log(summary.trainingMetrics.nSupportVectors); // Number of support vectors
```

---

#### Model Utilities

##### Train-Test Split

```javascript
const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
const y = [0, 0, 1, 1, 1];

const split = datly.ml.trainTestSplit(X, y, 0.2, true);
// {
//   X_train: [[...], [...], [...]],  // 80% of data
//   X_test: [[...], [...]],           // 20% of data
//   y_train: [0, 1, 1],
//   y_test: [0, 1]
// }
```

---

##### Cross-Validation

```javascript
const model = datly.ml.createKNN({ k: 3 });
const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]];
const y = [0, 0, 0, 0, 1, 1, 1, 1];

const cv = datly.ml.crossValidate(model, X, y, 5, 'classification');
console.log(cv);
// {
//   scores: [1.0, 0.8, 1.0, 0.8, 0.9],
//   meanScore: 0.9,
//   stdScore: 0.089,
//   folds: 5
// }
```

---

##### Compare Models

```javascript
const X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]];
const y = [0, 0, 0, 1, 1, 1];

const models = [
  { name: 'KNN', model: datly.ml.createKNN({ k: 3 }) },
  { name: 'Decision Tree', model: datly.ml.createDecisionTree() },
  { name: 'Logistic Regression', model: datly.ml.createLogisticRegression() }
];

const comparison = datly.ml.compareModels(models, X, y, 'classification');
console.log(comparison);
// {
//   results: [
//     { name: 'KNN', score: 0.95, trainTime: 5, evalTime: 2 },
//     { name: 'Decision Tree', score: 0.90, trainTime: 15, evalTime: 1 },
//     { name: 'Logistic Regression', score: 0.85, trainTime: 100, evalTime: 1 }
//   ],
//   bestModel: { name: 'KNN', score: 0.95, ... },
//   comparison: "📊 MODEL COMPARISON REPORT\n..."
// }
```

---

##### Quick Train (One-liner)

```javascript
// Train and evaluate a model in one line
const result = datly.ml.quickTrain(
  'randomforest',  // Model type
  X,               // Features
  y,               // Target
  {
    taskType: 'classification',
    testSize: 0.2,
    normalize: true,
    nEstimators: 50
  }
);

console.log(result);
// {
//   model: RandomForest {...},
//   score: {
//     accuracy: 0.95,
//     confusionMatrix: {...}
//   },
//   trainTime: 150,
//   summary: {...}
// }
```

---

##### Feature Engineering

```javascript
// Polynomial features
const X = [[1], [2], [3]];
const polyFeatures = datly.ml.polynomialFeatures(X, 2);
console.log(polyFeatures);
// [[1, 1], [2, 4], [3, 9]] // [x, x²]

// Standard scaling
const data = [[1, 2], [3, 4], [5, 6]];
const scaler = datly.ml.standardScaler(data);
console.log(scaler.scaled);
// [[-1.22, -1.22], [0, 0], [1.22, 1.22]]

// Transform new data
const newData = [[2, 3]];
const scaled = scaler.transform(newData);
console.log(scaled);

// Min-Max scaling
const minMaxScaler = datly.ml.minMaxScaler(data, [0, 1]);
console.log(minMaxScaler.scaled);
// [[0, 0], [0.5, 0.5], [1, 1]]
```

---

##### ROC Curve

```javascript
const yTrue = [0, 0, 1, 1, 1, 0, 1, 0];
const yProba = [0.1, 0.3, 0.6, 0.8, 0.9, 0.2, 0.7, 0.4];

const roc = datly.ml.rocCurve(yTrue, yProba);
console.log(roc);
// {
//   fpr: [0, 0, 0.25, 0.25, 0.5, ...],
//   tpr: [0, 0.2, 0.2, 0.4, 0.4, ...],
//   auc: 0.85,  // Area Under Curve
//   thresholds: [...]
// }
```

---

##### Precision-Recall Curve

```javascript
const yTrue = [0, 0, 1, 1, 1, 0, 1, 0];
const yProba = [0.1, 0.3, 0.6, 0.8, 0.9, 0.2, 0.7, 0.4];

const pr = datly.ml.precisionRecallCurve(yTrue, yProba);
console.log(pr);
// {
//   precision: [0.5, 0.6, 0.67, ...],
//   recall: [1.0, 0.8, 0.67, ...],
//   thresholds: [...]
// }
```

---

### 18. Data Visualization

Create interactive D3.js visualizations (requires D3.js library).

#### Setup

```javascript
// Initialize visualizer
const viz = datly.viz;

// Or create custom container
const customViz = new Datly.Visualizer('my-container-id');
```

---

#### Methods

##### `histogram(data, options)`

```javascript
const ages = [23, 25, 27, 29, 31, 33, 35, 37, 39, 41];

datly.plotHistogram(ages, {
  title: 'Age Distribution',
  xlabel: 'Age',
  ylabel: 'Frequency',
  bins: 10,
  color: '#4299e1',
  width: 800,
  height: 600
});
```

---

##### `boxplot(data, options)`

```javascript
// Single box plot
const data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
datly.plotBoxplot(data1, {
  title: 'Sales Distribution',
  ylabel: 'Sales ($)'
});

// Multiple box plots
const groupA = [20, 22, 23, 25, 27];
const groupB = [30, 32, 35, 37, 40];
const groupC = [25, 27, 29, 31, 33];

datly.plotBoxplot([groupA, groupB, groupC], {
  title: 'Sales by Region',
  labels: ['North', 'South', 'East'],
  ylabel: 'Sales ($)'
});
```

---

##### `scatter(x, y, options)`

```javascript
const height = [160, 165, 170, 175, 180, 185];
const weight = [55, 60, 65, 70, 75, 80];

datly.plotScatter(height, weight, {
  title: 'Height vs Weight',
  xlabel: 'Height (cm)',
  ylabel: 'Weight (kg)',
  color: '#e74c3c',
  size: 6,
  labels: ['Person 1', 'Person 2', ...] // Optional
});
```

---

##### `line(x, y, options)`

```javascript
const months = [1, 2, 3, 4, 5, 6];
const revenue = [100, 120, 140, 130, 160, 180];

datly.plotLine(months, revenue, {
  title: 'Monthly Revenue',
  xlabel: 'Month',
  ylabel: 'Revenue ($1000)',
  color: '#2ecc71',
  lineWidth: 3,
  showPoints: true
});
```

---

##### `bar(categories, values, options)`

```javascript
const products = ['Product A', 'Product B', 'Product C'];
const sales = [150, 230, 180];

datly.plotBar(products, sales, {
  title: 'Sales by Product',
  xlabel: 'Product',
  ylabel: 'Sales',
  color: '#f39c12',
  horizontal: false // Set to true for horizontal bars
});
```

---

##### `pie(labels, values, options)`

```javascript
const categories = ['Electronics', 'Clothing', 'Food', 'Other'];
const amounts = [35, 25, 20, 20];

datly.plotPie(categories, amounts, {
  title: 'Sales Distribution',
  showLabels: true,
  showPercentage: true
});
```

---

##### `heatmap(matrix, options)`

```javascript
const correlationMatrix = [
  [1.0, 0.8, 0.3],
  [0.8, 1.0, 0.5],
  [0.3, 0.5, 1.0]
];

datly.plotHeatmap(correlationMatrix, {
  title: 'Correlation Heatmap',
  labels: ['Var1', 'Var2', 'Var3'],
  colorScheme: 'RdBu', // Color scheme
  showValues: true
});
```

---

##### `violin(data, options)`

```javascript
const groupA = [1, 2, 3, 4, 5, 6, 7];
const groupB = [3, 4, 5, 6, 7, 8, 9];

datly.plotViolin([groupA, groupB], {
  title: 'Distribution Comparison',
  labels: ['Control', 'Treatment'],
  ylabel: 'Value',
  color: '#9b59b6'
});
```

---

##### `density(data, options)`

```javascript
const data = [23, 25, 27, 29, 31, 33, 35];

datly.plotDensity(data, {
  title: 'Density Plot',
  xlabel: 'Value',
  ylabel: 'Density',
  color: '#1abc9c',
  bandwidth: null // Auto-calculate or specify
});
```

---

##### `qqplot(data, options)`

```javascript
const data = [2.3, 2.5, 2.7, 2.9, 3.1, 3.3];

datly.plotQQ(data, {
  title: 'Q-Q Plot',
  xlabel: 'Theoretical Quantiles',
  ylabel: 'Sample Quantiles',
  color: '#34495e'
});
```

---

##### `parallel(data, dimensions, options)`

```javascript
const data = [
  { age: 25, income: 30000, spending: 15000, satisfaction: 7 },
  { age: 30, income: 45000, spending: 20000, satisfaction: 8 },
  { age: 35, income: 60000, spending: 25000, satisfaction: 9 }
];

datly.plotParallel(data, ['age', 'income', 'spending', 'satisfaction'], {
  title: 'Parallel Coordinates Plot',
  colors: ['#e74c3c', '#3498db', '#2ecc71']
});
```

---

##### `pairplot(data, columns, options)`

```javascript
const data = [
  { height: 160, weight: 55, age: 25 },
  { height: 165, weight: 60, age: 30 },
  { height: 170, weight: 65, age: 35 }
];

datly.plotPairplot(data, ['height', 'weight', 'age'], {
  title: 'Pair Plot',
  color: '#3498db',
  size: 3
});
```

---

##### `multiline(series, options)`

```javascript
const series = [
  {
    name: 'Series A',
    data: [
      { x: 1, y: 10 },
      { x: 2, y: 15 },
      { x: 3, y: 12 }
    ]
  },
  {
    name: 'Series B',
    data: [
      { x: 1, y: 5 },
      { x: 2, y: 8 },
      { x: 3, y: 10 }
    ]
  }
];

datly.plotMultiline(series, {
  title: 'Multiple Time Series',
  xlabel: 'Time',
  ylabel: 'Value',
  legend: true
});
```

---

##### Special Visualization Methods

```javascript
// Correlation matrix heatmap from dataset
const data = {
  headers: ['var1', 'var2', 'var3'],
  data: [
    { var1: 1, var2: 2, var3: 3 },
    { var1: 2, var2: 4, var3: 5 },
    { var1: 3, var2: 5, var3: 7 }
  ]
};

datly.plotCorrelationMatrix(data, {
  title: 'Correlation Matrix'
});

// Distribution plot from dataset column
datly.plotDistribution(data, 'var1', {
  title: 'Distribution of var1'
});

// Compare multiple distributions
datly.plotMultipleDistributions(data, ['var1', 'var2', 'var3'], {
  title: 'Distribution Comparison'
});
```

---

## 🎯 Complete Examples

### Example 1: Comprehensive Data Analysis

```javascript
const datly = new Datly();

// Load data
const data = datly.dataLoader.loadCSV('sales_data.csv');

// Validate data
const validation = datly.validator.validateData(data);
if (!validation.valid) {
  console.error('Data validation failed:', validation.errors);
}

// Get descriptive statistics
const sales = datly.dataLoader.getColumn(data, 'sales');
console.log('Mean Sales:', datly.centralTendency.mean(sales));
console.log('Median Sales:', datly.centralTendency.median(sales));
console.log('Std Dev:', datly.dispersion.standardDeviation(sales));

// Check for outliers
const outliers = datly.utils.detectOutliers(sales, 'iqr');
console.log('Outliers:', outliers);

// Test normality
const normalityTest = datly.normalityTests.shapiroWilk(sales);
console.log('Is Normal:', normalityTest.isNormal);

// Generate report
const report = datly.reportGenerator.summary(data);
console.log(report);

// Visualize (requires D3.js)
datly.plotHistogram(sales, { title: 'Sales Distribution' });
datly.plotBoxplot(sales, { title: 'Sales Box Plot' });
```

---

### Example 2: Hypothesis Testing Workflow

```javascript
const datly = new Datly();

// Two groups to compare
const controlGroup = [23, 25, 27, 29, 31, 33];
const treatmentGroup = [28, 30, 32, 34, 36, 38];

// Perform t-test
const tTest = datly.hypothesisTesting.tTest(
  controlGroup,
  treatmentGroup,
  'two-sample'
);

// Interpret results
const interpretation = datly.interpreter.interpret(tTest);
console.log(interpretation.plainLanguage);
console.log('Decision:', interpretation.conclusion.decision);
console.log('Effect Size:', interpretation.effectSize.magnitude);

// Calculate confidence interval for difference
const ci = datly.confidenceIntervals.meanDifference(
  controlGroup,
  treatmentGroup,
  0.95
);
console.log('95% CI for difference:', ci.lowerBound, 'to', ci.upperBound);

// Visualize (requires D3.js)
datly.plotBoxplot([controlGroup, treatmentGroup], {
  title: 'Control vs Treatment',
  labels: ['Control', 'Treatment']
});
```

---

### Example 3: Correlation and Regression

```javascript
const datly = new Datly();

const data = {
  headers: ['advertising', 'sales'],
  data: [
    { advertising: 10, sales: 100 },
    { advertising: 15, sales: 150 },
    { advertising: 20, sales: 180 },
    { advertising: 25, sales: 230 },
    { advertising: 30, sales: 270 }
  ]
};

// Extract columns
const advertising = datly.dataLoader.getColumn(data, 'advertising');
const sales = datly.dataLoader.getColumn(data, 'sales');

// Calculate correlation
const correlation = datly.correlation.pearson(advertising, sales);
console.log('Correlation:', correlation.correlation);
console.log('P-value:', correlation.pValue);

// Fit regression model
const model = datly.regression.linear(advertising, sales);
console.log('Equation:', model.equation);
console.log('R²:', model.rSquared);
console.log('Model significant:', model.pValueModel < 0.05);

// Make prediction
const newAdvertising = [35];
const prediction = datly.regression.predict(model, newAdvertising);
console.log('Predicted sales for $35k advertising:', prediction[0]);

// Visualize (requires D3.js)
datly.plotScatter(advertising, sales, {
  title: 'Advertising vs Sales',
  xlabel: 'Advertising Budget ($1000)',
  ylabel: 'Sales ($1000)'
});
```

---

### Example 4: Machine Learning Pipeline

```javascript
const datly = new Datly();

// Load data
const data = datly.dataLoader.loadJSON('iris.json');

// Prepare features and target
const X = data.data.map(row => [
  row.sepal_length,
  row.sepal_width,
  row.petal_length,
  row.petal_width
]);
const y = data.data.map(row => row.species);

// Split data
const split = datly.ml.trainTestSplit(X, y, 0.2, true);

// Create and train model
const model = datly.ml.createRandomForest({
  nEstimators: 100,
  maxDepth: 10
});

model.fit(split.X_train, split.y_train, 'classification');

// Evaluate
const score = model.score(split.X_test, split.y_test);
console.log('Accuracy:', score.accuracy);
console.log('Confusion Matrix:', score.confusionMatrix.display);

// Cross-validation
const cv = datly.ml.crossValidate(model, X, y, 5, 'classification');
console.log('CV Mean Score:', cv.meanScore);
console.log('CV Std:', cv.stdScore);

// Feature importance
const importance = model.getFeatureImportance();
console.log('Feature Importance:', importance);
```

---

### Example 5: Automatic Analysis

```javascript
const datly = new Datly();

// Load your data
const data = datly.dataLoader.loadCSV('customer_data.csv');

// Run automatic analysis
const analysis = datly.autoAnalyzer.autoAnalyze(data, {
  minCorrelationThreshold: 0.5,
  significanceLevel: 0.05,
  generateVisualizations: true
});

// View insights
analysis.insights.forEach(insight => {
  console.log(`${insight.icon} [${insight.priority}] ${insight.title}`);
  console.log(`   ${insight.description}`);
  if (insight.recommendation) {
    console.log(`   → ${insight.recommendation}`);
  }
});

// View recommended visualizations
analysis.visualizationSuggestions.forEach(viz => {
  console.log(`📊 ${viz.title} (${viz.type}) - Priority: ${viz.priority}`);
});

// Generate report
const textReport = datly.reportGenerator.exportSummary(
  analysis,
  'text'
);
console.log(textReport);
```

---

## 📖 API Reference

### Core Classes

- **`DataLoader`**: Load and manipulate datasets
- **`Validator`**: Validate data integrity
- **`Utils`**: Utility functions for data analysis
- **`CentralTendency`**: Mean, median, mode calculations
- **`Dispersion`**: Variance, standard deviation measures
- **`Position`**: Quantiles, percentiles, rankings
- **`Shape`**: Skewness and kurtosis analysis
- **`HypothesisTesting`**: Statistical tests
- **`ConfidenceIntervals`**: Interval estimation
- **`NormalityTests`**: Test for normal distribution
- **`Correlation`**: Correlation analysis
- **`Regression`**: Regression modeling
- **`ReportGenerator`**: Generate statistical reports
- **`PatternDetector`**: Detect patterns in data
- **`Interpreter`**: Interpret statistical results
- **`AutoAnalyzer`**: Automated analysis
- **`ML`**: Machine learning models
- **`Visualizer`**: Data visualization (requires D3.js)

---

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

**Requirements:**
- Modern JavaScript (ES6+)
- D3.js v7+ (only required for visualization module)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

MIT License

Copyright (c) 2025 Datly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.

---

## 📧 Contact

For questions, issues, or feature requests:
- GitHub Issues: [github.com/yourrepo/datly/issues](https://github.com/yourrepo/datly/issues)
- NPM Package: [npmjs.com/package/datly](https://npmjs.com/package/datly)

**Made with ❤️ for the data science community**