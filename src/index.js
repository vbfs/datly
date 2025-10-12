import DataLoader from "./core/dataLoader.js";
import Validator from "./core/validator.js";
import Utils from "./core/utils.js";
import CentralTendency from "./descriptive/centralTendency.js";
import Dispersion from "./descriptive/dispersion.js";
import Position from "./descriptive/position.js";
import Shape from "./descriptive/shape.js";
import HypothesisTesting from "./inferential/hypothesisTesting.js";
import ConfidenceIntervals from "./inferential/confidenceIntervals.js";
import NormalityTests from "./inferential/normalityTests.js";
import Correlation from "./multivariate/correlation.js";
import Regression from "./multivariate/regression.js";
import ReportGenerator from "./insights/reportGenerator.js";
import PatternDetector from "./insights/patternDetector.js";
import Interpreter from "./insights/interpreter.js";
import AutoAnalyzer from "./insights/autoAnalyser.js";
import MachineLearning from "./ml/machineLearning.js";
import DataViz from "./dataviz/index.js";

class DSToolkit {
  constructor() {
    this.dataLoader = new DataLoader();
    this.validator = new Validator();
    this.utils = new Utils();
    this.centralTendency = new CentralTendency();
    this.dispersion = new Dispersion();
    this.position = new Position();
    this.shape = new Shape();
    this.hypothesisTesting = new HypothesisTesting();
    this.confidenceIntervals = new ConfidenceIntervals();
    this.normalityTests = new NormalityTests();
    this.correlation = new Correlation();
    this.regression = new Regression();
    this.reportGenerator = new ReportGenerator();
    this.patternDetector = new PatternDetector();
    this.interpreter = new Interpreter();
    this.autoAnalyzer = new AutoAnalyzer(this);
    this.ml = new MachineLearning();
    this.viz = new DataViz();
  }

  // ====== Loaders / Utils ======
  async loadCSV(filePath, options = {}) {
    return this.dataLoader.loadCSV(filePath, options);
  }

  async loadJSON(jsonInput, options = {}) {
    return this.dataLoader.loadJSON(jsonInput, options);
  }

  cleanData(dataset) {
    return this.dataLoader.cleanData(dataset);
  }

  validateData(dataset) {
    return this.validator.validateData(dataset);
  }

  getDataInfo(dataset) {
    return this.dataLoader.getDataInfo(dataset);
  }

  // ====== Descriptive: Central Tendency ======
  mean(column) {
    return this.centralTendency.mean(column);
  }

  median(column) {
    return this.centralTendency.median(column);
  }

  mode(column) {
    return this.centralTendency.mode(column);
  }

  geometricMean(column) {
    return this.centralTendency.geometricMean(column);
  }

  harmonicMean(column) {
    return this.centralTendency.harmonicMean(column);
  }

  trimmedMean(column, percentage) {
    return this.centralTendency.trimmedMean(column, percentage);
  }

  // ====== Descriptive: Dispersion ======
  variance(column) {
    return this.dispersion.variance(column);
  }

  standardDeviation(column) {
    return this.dispersion.standardDeviation(column);
  }

  range(column) {
    return this.dispersion.range(column);
  }

  interquartileRange(column) {
    return this.dispersion.interquartileRange(column);
  }

  coefficientOfVariation(column) {
    return this.dispersion.coefficientOfVariation(column);
  }

  meanAbsoluteDeviation(column) {
    return this.dispersion.meanAbsoluteDeviation(column);
  }

  // ====== Descriptive: Position & Shape ======
  quantile(column, q) {
    return this.position.quantile(column, q);
  }

  percentile(column, p) {
    return this.position.percentile(column, p);
  }

  quartiles(column) {
    return this.position.quartiles(column);
  }

  quintiles(column) {
    return this.position.quintiles(column);
  }

  deciles(column) {
    return this.position.deciles(column);
  }

  percentileRank(column, value) {
    return this.position.percentileRank(column, value);
  }

  zScore(column, value) {
    return this.position.zScore(column, value);
  }

  boxplotStats(column) {
    return this.position.boxplotStats(column);
  }

  fiveNumberSummary(column) {
    return this.position.fiveNumberSummary(column);
  }

  rank(column, method = 'average') {
    return this.position.rank(column, method);
  }

  normalizedRank(column) {
    return this.position.normalizedRank(column);
  }

  standardizedValues(column) {
    return this.position.standardizedValues(column);
  }

  skewness(column) {
    return this.shape.skewness(column);
  }

  kurtosis(column) {
    return this.shape.kurtosis(column);
  }

  isNormalDistribution(column) {
    return this.shape.isNormalDistribution(column);
  }

  // ====== Multivariate ======
  correlationPearson(col1, col2) {
    return this.correlation.pearson(col1, col2);
  }

  correlationSpearman(col1, col2) {
    return this.correlation.spearman(col1, col2);
  }

  correlationMatrix(dataset) {
    return this.correlation.matrix(dataset);
  }

  covariance(col1, col2) {
    return this.correlation.covariance(col1, col2);
  }

  // ====== Inferential ======
  tTest(sample1, sample2, type = "two-sample") {
    return this.hypothesisTesting.tTest(sample1, sample2, type);
  }

  zTest(sample, populationMean, populationStd) {
    return this.hypothesisTesting.zTest(sample, populationMean, populationStd);
  }

  anovaTest(groups) {
    return this.hypothesisTesting.anovaTest(groups);
  }

  chiSquareTest(col1, col2) {
    return this.hypothesisTesting.chiSquareTest(col1, col2);
  }

  confidenceInterval(sample, confidence = 0.95) {
    return this.confidenceIntervals.mean(sample, confidence);
  }

  shapiroWilkTest(sample) {
    return this.normalityTests.shapiroWilk(sample);
  }

  // ====== Regression ======
  linearRegression(x, y) {
    return this.regression.linear(x, y);
  }

  // ====== Utils ======
  detectOutliers(column, method = "iqr") {
    return this.utils.detectOutliers(column, method);
  }

  frequencyTable(column) {
    return this.utils.frequencyTable(column);
  }

  groupBy(dataset, column, aggregation) {
    return this.utils.groupBy(dataset, column, aggregation);
  }

  sample(dataset, size, method = "random") {
    return this.utils.sample(dataset, size, method);
  }

  // ====== Insights/Reports ======
  generateSummaryReport(dataset) {
    return this.reportGenerator.summary(dataset);
  }

  identifyPatterns(dataset) {
    return this.patternDetector.analyze(dataset);
  }

  interpretResults(testResult) {
    return this.interpreter.interpret(testResult);
  }

  // ====== AutoAnalyzer ======
  autoAnalyze(dataset, options = {}) {
    return this.autoAnalyzer.autoAnalyze(dataset, options);
  }

  async autoAnalyzeFromFile(filePath, loadOptions = {}, analysisOptions = {}) {
    let dataset;
    const lower = filePath.toLowerCase();
    if (lower.endsWith(".csv")) {
      dataset = await this.loadCSV(filePath, loadOptions);
    } else if (lower.endsWith(".json")) {
      dataset = await this.loadJSON(filePath, loadOptions);
    } else {
      throw new Error("Formato de arquivo não suportado. Use CSV ou JSON.");
    }
    return this.autoAnalyze(dataset, analysisOptions);
  }

  async quickAnalysis(filePath, options = {}) {
    const result = await this.autoAnalyzeFromFile(filePath, {}, options);

    console.log("\n" + "=".repeat(60));
    console.log("📊 AUTO REPORT");
    console.log("=".repeat(60));

    console.log(`\n📈 EXECUTIVE RESUME:`);
    console.log(`• Total insights: ${result.summary.totalInsights}`);
    console.log(`• Priority Insights: ${result.summary.highPriorityInsights}`);

    console.log(`\n🔍 MAIN INSIGHTS:`);
    result.summary.keyFindings.forEach((f, i) => {
      console.log(`${i + 1}. ${f.title}`);
      console.log(`   ${f.description}`);
    });

    console.log(`\n💡 RECOMMENDATIONS:`);
    result.summary.recommendations.forEach((rec, i) => {
      console.log(`${i + 1}. ${rec}`);
    });

    console.log("\n" + "=".repeat(60));
    return result;
  }

  // ====== Machine Learning: Model Creation ======
  createLinearRegression(options) {
    return this.ml.createLinearRegression(options);
  }

  createLogisticRegression(options) {
    return this.ml.createLogisticRegression(options);
  }

  createKNN(options) {
    return this.ml.createKNN(options);
  }

  createDecisionTree(options) {
    return this.ml.createDecisionTree(options);
  }

  createRandomForest(options) {
    return this.ml.createRandomForest(options);
  }

  createNaiveBayes(options) {
    return this.ml.createNaiveBayes(options);
  }

  createSVM(options) {
    return this.ml.createSVM(options);
  }

  // ====== Machine Learning: Utilities ======
  trainTestSplit(X, y, testSize = 0.2, shuffle = true) {
    return this.ml.trainTestSplit(X, y, testSize, shuffle);
  }

  crossValidate(model, X, y, folds = 5, taskType = 'classification') {
    return this.ml.crossValidate(model, X, y, folds, taskType);
  }

  compareModels(models, X, y, taskType = 'classification') {
    return this.ml.compareModels(models, X, y, taskType);
  }

  quickTrain(modelType, X, y, options = {}) {
    return this.ml.quickTrain(modelType, X, y, options);
  }

  // ====== Machine Learning: Feature Engineering ======
  polynomialFeatures(X, degree = 2) {
    return this.ml.polynomialFeatures(X, degree);
  }

  standardScaler(X) {
    return this.ml.standardScaler(X);
  }

  minMaxScaler(X, featureRange = [0, 1]) {
    return this.ml.minMaxScaler(X, featureRange);
  }

  // ====== Machine Learning: Metrics ======
  rocCurve(yTrue, yProba) {
    return this.ml.rocCurve(yTrue, yProba);
  }

  precisionRecallCurve(yTrue, yProba) {
    return this.ml.precisionRecallCurve(yTrue, yProba);
  }

  // ====== Data Visualization: Basic Plots ======
  plotHistogram(data, options) {
    return this.viz.histogram(data, options);
  }

  plotBoxplot(data, options) {
    return this.viz.boxplot(data, options);
  }

  plotScatter(xData, yData, options) {
    return this.viz.scatter(xData, yData, options);
  }

  plotLine(xData, yData, options) {
    return this.viz.line(xData, yData, options);
  }

  plotBar(categories, values, options) {
    return this.viz.bar(categories, values, options);
  }

  plotPie(labels, values, options) {
    return this.viz.pie(labels, values, options);
  }

  // ====== Data Visualization: Advanced Plots ======
  plotHeatmap(matrix, options) {
    return this.viz.heatmap(matrix, options);
  }

  plotViolin(data, options) {
    return this.viz.violin(data, options);
  }

  plotDensity(data, options) {
    return this.viz.density(data, options);
  }

  plotQQ(data, options) {
    return this.viz.qqplot(data, options);
  }

  plotParallel(data, dimensions, options) {
    return this.viz.parallel(data, dimensions, options);
  }

  plotPairplot(data, variables, options) {
    return this.viz.pairplot(data, variables, options);
  }

  plotMultiline(series, options) {
    return this.viz.multiline(series, options);
  }

  // ====== Data Visualization: Helpers ======
  plotCorrelationMatrix(dataset, options = {}) {
    const columns = Object.keys(dataset[0]).filter(
      (key) => typeof dataset[0][key] === "number"
    );

    const n = columns.length;
    const matrix = Array(n)
      .fill(0)
      .map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const col1 = dataset.map((row) => row[columns[i]]);
        const col2 = dataset.map((row) => row[columns[j]]);
        matrix[i][j] = this.correlationPearson(col1, col2);
      }
    }

    return this.viz.heatmap(matrix, {
      title: "Correlation Matrix",
      labels: columns,
      ...options,
    });
  }

  plotDistribution(dataset, columnName, options = {}) {
    const data = dataset.map((row) => row[columnName]).filter((v) => !isNaN(v));

    return this.viz.histogram(data, {
      title: `Distribution of ${columnName}`,
      xlabel: columnName,
      ylabel: "Frequency",
      ...options,
    });
  }

  plotMultipleDistributions(dataset, columnNames, options = {}) {
    const data = columnNames.map((col) =>
      dataset.map((row) => row[col]).filter((v) => !isNaN(v))
    );

    return this.viz.boxplot(data, {
      title: "Distribution Comparison",
      labels: columnNames,
      ...options,
    });
  }

}

export default DSToolkit;