# DS Toolkit 📊

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![D3](https://img.shields.io/badge/d3-v7-orange.svg)
![NPM](https://img.shields.io/badge/npm-ds--toolkit-red.svg)

> **Javascript toolkit for data science, statistical analysis and machine learning.**

**DS Toolkit** is a powerful, all-in-one JavaScript library that brings professional data science capabilities to both browser and Node.js environments. With 150+ statistical methods, 7 machine learning algorithms, and 13 interactive visualizations powered by D3.js, it's everything you need for serious data analysis in JavaScript.

### 🎯 Perfect for:
- **Data Scientists** - Professional statistical analysis and ML modeling
- **Analysts** - Quick insights and automated reporting
- **Developers** - Easy-to-use API with zero configuration
- **Researchers** - Comprehensive hypothesis testing and validation

### ⚡ Key Features:
- 📈 **Complete Statistics Suite** - From descriptive stats to advanced hypothesis testing
- 🤖 **7 ML Algorithms** - Classification, regression, and ensemble methods
- 📊 **13 Visualization Types** - Interactive D3.js charts with one-line commands
- 🔄 **Auto-Analysis** - Intelligent data exploration with automated insights
- 🎨 **Zero Config** - Works out of the box, customizable when needed
- 🌐 **Universal** - Same API for browser and Node.js

---

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Complete Examples](#complete-examples)
- [API Reference](#api-reference)

---

## 🚀 Installation

### Browser (CDN)

```html
<!-- Include D3.js -->
<script src="https://d3js.org/d3.v7.min.js"></script>

<!-- Include DS Toolkit -->
<script src="path/to/ds-toolkit.min.js"></script>

<script>
  const ds = new DSToolkit();
</script>
```

### Node.js (NPM)

```bash
npm install ds-toolkit d3
```

```javascript
import DSToolkit from 'ds-toolkit';
const ds = new DSToolkit();
```

---

## ⚡ Quick Start

```javascript
// Initialize the library
const ds = new DSToolkit();

// Load data from CSV
const data = await ds.loadCSV('data.csv');

// Calculate mean
const ages = [25, 30, 35, 40, 45];
const meanAge = ds.mean(ages);
console.log('Mean Age:', meanAge); // 35

// Perform t-test
const group1 = [23, 25, 27, 29, 31];
const group2 = [30, 32, 34, 36, 38];
const tTest = ds.tTest(group1, group2);
console.log('T-Test Result:', tTest);

// Create visualization
ds.plotHistogram(ages, {
  title: 'Age Distribution',
  xlabel: 'Age',
  ylabel: 'Frequency'
});
```

---

## 📦 Core Modules

### 1. Data Loading & Validation

#### Load CSV
```javascript
const data = await ds.loadCSV('sales.csv', {
  delimiter: ',',
  header: true,
  skipEmptyLines: true,
  encoding: 'utf8'
});
```

#### Load JSON
```javascript
const data = await ds.loadJSON('data.json');
```

#### Clean Data
```javascript
const cleaned = ds.cleanData(dataset);
```

#### Validate Data
```javascript
const validation = ds.validateData(dataset);
console.log(validation.valid); // true/false
```

---

### 2. Descriptive Statistics

#### Central Tendency
```javascript
const data = [10, 20, 30, 40, 50];

ds.mean(data);              // 30
ds.median(data);            // 30
ds.mode(data);              // { values: [...], frequency: ... }
ds.geometricMean(data);     // 25.2
ds.harmonicMean(data);      // 21.9
ds.trimmedMean(data, 10);   // Remove 10% from each end
```

#### Dispersion
```javascript
ds.variance(data);                  // Sample variance
ds.standardDeviation(data);         // Standard deviation
ds.range(data);                     // { range, min, max }
ds.interquartileRange(data);        // IQR
ds.coefficientOfVariation(data);    // CV%
ds.meanAbsoluteDeviation(data);     // MAD
```

#### Position & Quantiles
```javascript
ds.quantile(data, 0.25);            // First quartile
ds.percentile(data, 75);            // 75th percentile
ds.quartiles(data);                 // { q1, q2, q3, iqr }
ds.quintiles(data);                 // 5 quintiles
ds.deciles(data);                   // 9 deciles
ds.percentileRank(data, value);     // Percentile of value
ds.zScore(data, value);             // Z-score
ds.boxplotStats(data);              // Complete boxplot stats
ds.fiveNumberSummary(data);         // Min, Q1, Median, Q3, Max
ds.rank(data, 'average');           // Rank values
ds.normalizedRank(data);            // Ranks 0-1
ds.standardizedValues(data);        // Z-scores for all
```

#### Shape Analysis
```javascript
ds.skewness(data);                  // Asymmetry measure
ds.kurtosis(data);                  // Tailedness measure
ds.isNormalDistribution(data);      // Normality test
```

---

### 3. Inferential Statistics

#### Hypothesis Testing
```javascript
// T-tests
ds.tTest(sample1, sample2, 'two-sample');  // Independent samples
ds.tTest(sample, popMean, 'one-sample');   // One sample
ds.tTest(before, after, 'paired');         // Paired samples

// Other tests
ds.zTest(sample, popMean, popStd);         // Z-test
ds.anovaTest([group1, group2, group3]);    // One-way ANOVA
ds.chiSquareTest(col1, col2);              // Chi-square test
```

#### Confidence Intervals
```javascript
ds.confidenceInterval(sample, 0.95);       // CI for mean
```

#### Normality Tests
```javascript
ds.shapiroWilkTest(sample);                // Shapiro-Wilk test
```

---

### 4. Correlation & Regression

#### Correlation
```javascript
ds.correlationPearson(x, y);               // Pearson correlation
ds.correlationSpearman(x, y);              // Spearman correlation
ds.correlationMatrix(dataset);             // Correlation matrix
ds.covariance(x, y);                       // Covariance
```

#### Regression
```javascript
ds.linearRegression(x, y);                 // Simple linear regression
```

---

### 5. Utility Functions

```javascript
// Outlier detection
ds.detectOutliers(data, 'iqr');            // IQR method
ds.detectOutliers(data, 'zscore');         // Z-score method

// Frequency analysis
ds.frequencyTable(data);                   // Frequency distribution

// Grouping
ds.groupBy(dataset, 'column', { sales: 'mean', profit: 'sum' });

// Sampling
ds.sample(dataset, 100, 'random');         // Random sampling
```

---

### 6. Reports & Insights

```javascript
// Generate summary report
const report = ds.generateSummaryReport(dataset);

// Identify patterns
const patterns = ds.identifyPatterns(dataset);

// Interpret results
const interpretation = ds.interpretResults(testResult);

// Auto-analysis
const analysis = ds.autoAnalyze(dataset);

// Quick analysis from file
const result = await ds.quickAnalysis('data.csv');
```

---

### 7. Machine Learning

#### Create Models

```javascript
// Regression
const linReg = ds.createLinearRegression({
  learningRate: 0.01,
  iterations: 1000,
  regularization: 'l2',
  lambda: 0.01
});

// Classification
const logReg = ds.createLogisticRegression();
const knn = ds.createKNN({ k: 5 });
const tree = ds.createDecisionTree({ maxDepth: 10 });
const forest = ds.createRandomForest({ nEstimators: 100 });
const nb = ds.createNaiveBayes({ type: 'gaussian' });
const svm = ds.createSVM({ kernel: 'linear' });
```

#### Train & Evaluate

```javascript
// Train
model.fit(X_train, y_train);

// Predict
const predictions = model.predict(X_test);
const probabilities = model.predictProba(X_test);

// Evaluate
const score = model.score(X_test, y_test);
console.log('Accuracy:', score.accuracy);
console.log(score.confusionMatrix.display);

// Summary
const summary = model.summary();
```

#### ML Utilities

```javascript
// Split data
const { X_train, X_test, y_train, y_test } =
  ds.trainTestSplit(X, y, 0.2, true);

// Cross-validation
const cv = ds.crossValidate(model, X, y, 5, 'classification');

// Compare models
const comparison = ds.compareModels(models, X, y);

// Quick train
const result = ds.quickTrain('randomforest', X, y, {
  taskType: 'classification',
  testSize: 0.2
});
```

#### Feature Engineering

```javascript
// Polynomial features
const polyFeatures = ds.polynomialFeatures(X, 2);

// Scaling
const scaler = ds.standardScaler(X);
const scaled = scaler.scaled;
const newScaled = scaler.transform(X_new);

// Min-Max scaling
const minMaxScaler = ds.minMaxScaler(X, [0, 1]);
```

#### Metrics

```javascript
// ROC Curve
const roc = ds.rocCurve(y_true, y_proba);
console.log('AUC:', roc.auc);

// Precision-Recall Curve
const pr = ds.precisionRecallCurve(y_true, y_proba);
```

---

### 8. Data Visualization

#### Basic Plots

```javascript
// Histogram
ds.plotHistogram(data, {
  title: 'Distribution',
  xlabel: 'Value',
  ylabel: 'Frequency',
  bins: 30,
  color: '#4299e1'
});

// Box Plot
ds.plotBoxplot([group1, group2], {
  title: 'Group Comparison',
  labels: ['Group A', 'Group B']
});

// Scatter Plot
ds.plotScatter(x, y, {
  title: 'X vs Y',
  xlabel: 'X',
  ylabel: 'Y',
  color: '#ed64a6',
  size: 5
});

// Line Chart
ds.plotLine(time, values, {
  title: 'Time Series',
  lineWidth: 2,
  showPoints: true
});

// Bar Chart
ds.plotBar(categories, values, {
  title: 'Categories',
  horizontal: false
});

// Pie Chart
ds.plotPie(labels, values, {
  title: 'Proportions',
  showPercentage: true
});
```

#### Advanced Plots

```javascript
// Heatmap / Correlation Matrix
ds.plotHeatmap(matrix, {
  title: 'Correlation Matrix',
  labels: ['Var1', 'Var2', 'Var3'],
  colorScheme: 'RdYlBu',
  showValues: true
});

// Violin Plot
ds.plotViolin(data, {
  title: 'Distribution Shape'
});

// Density Plot (KDE)
ds.plotDensity(data, {
  title: 'Density Estimation'
});

// Q-Q Plot
ds.plotQQ(data, {
  title: 'Normality Check'
});

// Parallel Coordinates
ds.plotParallel(data, ['var1', 'var2', 'var3'], {
  title: 'Multi-dimensional Data'
});

// Pair Plot (Scatter Matrix)
ds.plotPairplot(data, ['x1', 'x2', 'x3'], {
  title: 'Relationships'
});

// Multi-line Chart
ds.plotMultiline(series, {
  title: 'Multiple Series',
  legend: true
});
```

#### Helper Methods

```javascript
// Plot correlation matrix from dataset
ds.plotCorrelationMatrix(dataset);

// Plot distribution of a column
ds.plotDistribution(dataset, 'columnName');

// Compare multiple distributions
ds.plotMultipleDistributions(dataset, ['col1', 'col2', 'col3']);
```

---

## 🎯 Complete Examples

### Example 1: Full Data Analysis Pipeline

```javascript
import DSToolkit from 'ds-toolkit';
const ds = new DSToolkit();

// 1. Load data
const data = await ds.loadCSV('sales_data.csv');

// 2. Validate
const validation = ds.validateData(data);
if (!validation.valid) {
  console.error('Validation errors:', validation.errors);
}

// 3. Extract column
const sales = data.data.map(row => row.sales);

// 4. Descriptive statistics
console.log('Mean:', ds.mean(sales));
console.log('Median:', ds.median(sales));
console.log('Std Dev:', ds.standardDeviation(sales));

// 5. Check outliers
const outliers = ds.detectOutliers(sales, 'iqr');
console.log('Outliers found:', outliers.count);

// 6. Test normality
const normalityTest = ds.shapiroWilkTest(sales);
console.log('Is normal:', normalityTest.isNormal);

// 7. Generate report
const report = ds.generateSummaryReport(data);
console.log(report);

// 8. Visualize
ds.plotHistogram(sales, { title: 'Sales Distribution' });
ds.plotBoxplot(sales, { title: 'Sales Box Plot' });
ds.plotQQ(sales, { title: 'Normality Check' });
```

---

### Example 2: Machine Learning Pipeline

```javascript
import DSToolkit from 'ds-toolkit';
const ds = new DSToolkit();

// 1. Prepare data
const X = [
  [1, 2], [2, 3], [3, 4], [4, 5],
  [5, 6], [6, 7], [7, 8], [8, 9]
];
const y = [0, 0, 0, 0, 1, 1, 1, 1];

// 2. Split data
const { X_train, X_test, y_train, y_test } =
  ds.trainTestSplit(X, y, 0.2);

// 3. Normalize features
const scaler = ds.standardScaler(X_train);
const X_train_scaled = scaler.scaled;
const X_test_scaled = scaler.transform(X_test);

// 4. Create and train model
const model = ds.createRandomForest({
  nEstimators: 100,
  maxDepth: 10
});

model.fit(X_train_scaled, y_train, 'classification');

// 5. Evaluate
const score = model.score(X_test_scaled, y_test);
console.log('Accuracy:', (score.accuracy * 100).toFixed(2) + '%');
console.log(score.confusionMatrix.display);

// 6. Cross-validation
const cv = ds.crossValidate(model, X, y, 5, 'classification');
console.log('CV Mean Score:', cv.meanScore.toFixed(4));

// 7. Feature importance
const importance = model.getFeatureImportance();
console.log('Feature Importance:', importance);

// 8. Visualize confusion matrix
ds.plotHeatmap(score.confusionMatrix.matrix, {
  title: 'Confusion Matrix',
  labels: score.confusionMatrix.classes,
  showValues: true
});
```

---

### Example 3: Hypothesis Testing

```javascript
import DSToolkit from 'ds-toolkit';
const ds = new DSToolkit();

// Data: before and after treatment
const before = [120, 125, 130, 128, 122, 135, 118];
const after = [115, 118, 125, 120, 115, 128, 112];

// 1. Paired t-test
const tTest = ds.tTest(before, after, 'paired');
console.log('T-statistic:', tTest.statistic);
console.log('P-value:', tTest.pValue);
console.log('Significant:', tTest.significant);

// 2. Interpret results
const interpretation = ds.interpretResults(tTest);
console.log(interpretation.plainLanguage);
console.log('Effect size:', interpretation.effectSize.magnitude);

// 3. Calculate confidence interval
const ci = ds.confidenceInterval(after, 0.95);
console.log('95% CI:', ci.lowerBound, 'to', ci.upperBound);

// 4. Visualize
ds.plotBoxplot([before, after], {
  title: 'Before vs After Treatment',
  labels: ['Before', 'After']
});
```

---

### Example 4: Correlation & Regression

```javascript
import DSToolkit from 'ds-toolkit';
const ds = new DSToolkit();

const advertising = [10, 15, 20, 25, 30, 35, 40];
const sales = [100, 150, 180, 230, 270, 310, 360];

// 1. Calculate correlation
const corr = ds.correlationPearson(advertising, sales);
console.log('Correlation:', corr.toFixed(3));

// 2. Fit regression model
const model = ds.linearRegression(advertising, sales);
console.log('Equation:', model.equation);
console.log('R²:', model.rSquared.toFixed(4));

// 3. Visualize
ds.plotScatter(advertising, sales, {
  title: 'Advertising vs Sales',
  xlabel: 'Advertising Budget ($1000)',
  ylabel: 'Sales ($1000)'
});
```

---

### Example 5: Auto-Analysis

```javascript
import DSToolkit from 'ds-toolkit';
const ds = new DSToolkit();

// Load and analyze in one step
const result = await ds.quickAnalysis('customer_data.csv', {
  minCorrelationThreshold: 0.5,
  significanceLevel: 0.05
});

// Results are automatically printed to console
// Insights, patterns, and recommendations included
```

---

## 📖 API Reference

### Complete Method List

**Data Loading:**
- `loadCSV(filePath, options)` - Load CSV file
- `loadJSON(jsonInput, options)` - Load JSON data
- `cleanData(dataset)` - Remove invalid rows
- `validateData(dataset)` - Validate structure
- `getDataInfo(dataset)` - Get dataset info

**Descriptive - Central Tendency:**
- `mean(column)` - Arithmetic mean
- `median(column)` - Median value
- `mode(column)` - Most frequent value(s)
- `geometricMean(column)` - Geometric mean
- `harmonicMean(column)` - Harmonic mean
- `trimmedMean(column, percentage)` - Trimmed mean

**Descriptive - Dispersion:**
- `variance(column)` - Variance
- `standardDeviation(column)` - Standard deviation
- `range(column)` - Range (max - min)
- `interquartileRange(column)` - IQR (Q3 - Q1)
- `coefficientOfVariation(column)` - CV percentage
- `meanAbsoluteDeviation(column)` - MAD

**Descriptive - Position:**
- `quantile(column, q)` - Quantile value
- `percentile(column, p)` - Percentile value
- `quartiles(column)` - Q1, Q2, Q3, IQR
- `quintiles(column)` - 5 quintiles
- `deciles(column)` - 9 deciles
- `percentileRank(column, value)` - Percentile of value
- `zScore(column, value)` - Z-score of value
- `boxplotStats(column)` - Complete boxplot stats
- `fiveNumberSummary(column)` - Min, Q1, Med, Q3, Max
- `rank(column, method)` - Rank values
- `normalizedRank(column)` - Normalized ranks
- `standardizedValues(column)` - Standardize all values

**Descriptive - Shape:**
- `skewness(column)` - Measure of asymmetry
- `kurtosis(column)` - Measure of tailedness
- `isNormalDistribution(column)` - Test normality

**Inferential - Tests:**
- `tTest(sample1, sample2, type)` - T-test
- `zTest(sample, popMean, popStd)` - Z-test
- `anovaTest(groups)` - One-way ANOVA
- `chiSquareTest(col1, col2)` - Chi-square test
- `confidenceInterval(sample, confidence)` - CI for mean
- `shapiroWilkTest(sample)` - Shapiro-Wilk test

**Multivariate:**
- `correlationPearson(col1, col2)` - Pearson correlation
- `correlationSpearman(col1, col2)` - Spearman correlation
- `correlationMatrix(dataset)` - Full correlation matrix
- `covariance(col1, col2)` - Covariance
- `linearRegression(x, y)` - Simple linear regression

**Utilities:**
- `detectOutliers(column, method)` - Detect outliers
- `frequencyTable(column)` - Frequency distribution
- `groupBy(dataset, column, aggregation)` - Group and aggregate
- `sample(dataset, size, method)` - Extract sample

**Reports & Analysis:**
- `generateSummaryReport(dataset)` - Complete report
- `identifyPatterns(dataset)` - Pattern detection
- `interpretResults(testResult)` - Interpret results
- `autoAnalyze(dataset, options)` - Auto-analysis
- `quickAnalysis(filePath, options)` - Quick file analysis

**Machine Learning - Models:**
- `createLinearRegression(options)` - Linear regression model
- `createLogisticRegression(options)` - Logistic regression model
- `createKNN(options)` - K-Nearest Neighbors model
- `createDecisionTree(options)` - Decision tree model
- `createRandomForest(options)` - Random forest model
- `createNaiveBayes(options)` - Naive Bayes model
- `createSVM(options)` - Support Vector Machine model

**Machine Learning - Utilities:**
- `trainTestSplit(X, y, testSize, shuffle)` - Split dataset
- `crossValidate(model, X, y, folds, taskType)` - Cross-validation
- `compareModels(models, X, y, taskType)` - Compare models
- `quickTrain(modelType, X, y, options)` - Quick training

**Machine Learning - Feature Engineering:**
- `polynomialFeatures(X, degree)` - Create polynomial features
- `standardScaler(X)` - Standard scaling (z-score)
- `minMaxScaler(X, featureRange)` - Min-max scaling

**Machine Learning - Metrics:**
- `rocCurve(yTrue, yProba)` - ROC curve and AUC
- `precisionRecallCurve(yTrue, yProba)` - PR curve

**Visualization - Basic:**
- `plotHistogram(data, options)` - Histogram
- `plotBoxplot(data, options)` - Box plot
- `plotScatter(xData, yData, options)` - Scatter plot
- `plotLine(xData, yData, options)` - Line chart
- `plotBar(categories, values, options)` - Bar chart
- `plotPie(labels, values, options)` - Pie chart

**Visualization - Advanced:**
- `plotHeatmap(matrix, options)` - Heatmap
- `plotViolin(data, options)` - Violin plot
- `plotDensity(data, options)` - Density plot (KDE)
- `plotQQ(data, options)` - Q-Q plot
- `plotParallel(data, dimensions, options)` - Parallel coordinates
- `plotPairplot(data, variables, options)` - Pair plot
- `plotMultiline(series, options)` - Multi-line chart

**Visualization - Helpers:**
- `plotCorrelationMatrix(dataset, options)` - Auto correlation matrix
- `plotDistribution(dataset, columnName, options)` - Column distribution
- `plotMultipleDistributions(dataset, columnNames, options)` - Compare distributions

---

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

**Requirements:**
- D3.js v7+
- Modern JavaScript (ES6+)

---

## 📝 License

MIT License - Copyright (c) 2025 DS Toolkit

---

## 📧 Contact

- NPM Package: [npmjs.com/package/ds-toolkit](https://npmjs.com/package/ds-toolkit)
- GitHub: [github.com/yourrepo/ds-toolkit](https://github.com/yourrepo/ds-toolkit)
- Issues: [github.com/yourrepo/ds-toolkit/issues](https://github.com/yourrepo/ds-toolkit/issues)

---

**Made with ❤️ for the data science community**