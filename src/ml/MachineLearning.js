import LinearRegression from './LinearRegression.js';
import LogisticRegression from './LogisticRegression.js';
import KNearestNeighbors from './KNearestNeighbors.js';
import DecisionTree from './DecisionTree.js';
import RandomForest from './RandomForest.js';
import NaiveBayes from './NaiveBayes.js';
import SupportVectorMachine from './SupportVectorMachine.js';

class MachineLearning {
  constructor() {
    // Models are instantiated on demand
  }

  // ====== Regression Models ======
  createLinearRegression(options = {}) {
    const {
      learningRate = 0.01,
      iterations = 1000,
      regularization = null,
      lambda = 0.01
    } = options;

    return new LinearRegression(learningRate, iterations, regularization, lambda);
  }

  // ====== Classification Models ======
  createLogisticRegression(options = {}) {
    const {
      learningRate = 0.01,
      iterations = 1000,
      regularization = null,
      lambda = 0.01
    } = options;

    return new LogisticRegression(learningRate, iterations, regularization, lambda);
  }

  createKNN(options = {}) {
    const {
      k = 5,
      metric = 'euclidean',
      weights = 'uniform'
    } = options;

    return new KNearestNeighbors(k, metric, weights);
  }

  createDecisionTree(options = {}) {
    const {
      maxDepth = 10,
      minSamplesSplit = 2,
      minSamplesLeaf = 1,
      criterion = 'gini'
    } = options;

    return new DecisionTree(maxDepth, minSamplesSplit, minSamplesLeaf, criterion);
  }

  createRandomForest(options = {}) {
    const {
      nEstimators = 100,
      maxDepth = 10,
      minSamplesSplit = 2,
      minSamplesLeaf = 1,
      maxFeatures = 'sqrt',
      criterion = 'gini',
      bootstrap = true
    } = options;

    return new RandomForest(
      nEstimators,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      maxFeatures,
      criterion,
      bootstrap
    );
  }

  createNaiveBayes(options = {}) {
    const { type = 'gaussian' } = options;
    return new NaiveBayes(type);
  }

  createSVM(options = {}) {
    const {
      C = 1.0,
      kernel = 'linear',
      gamma = 'scale',
      degree = 3,
      learningRate = 0.001,
      iterations = 1000
    } = options;

    return new SupportVectorMachine(C, kernel, gamma, degree, learningRate, iterations);
  }

  // ====== Model Evaluation Utilities ======
  crossValidate(model, X, y, folds = 5, taskType = 'classification') {
    const n = X.length;
    const foldSize = Math.floor(n / folds);
    const indices = Array.from({ length: n }, (_, i) => i);

    // Shuffle indices
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const scores = [];

    for (let fold = 0; fold < folds; fold++) {
      const testStart = fold * foldSize;
      const testEnd = fold === folds - 1 ? n : testStart + foldSize;

      const testIndices = indices.slice(testStart, testEnd);
      const trainIndices = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

      const X_train = trainIndices.map(i => X[i]);
      const y_train = trainIndices.map(i => y[i]);
      const X_test = testIndices.map(i => X[i]);
      const y_test = testIndices.map(i => y[i]);

      // Create a new instance of the model
      const foldModel = Object.create(Object.getPrototypeOf(model));
      Object.assign(foldModel, model);

      // Train and evaluate
      foldModel.fit(X_train, y_train, taskType);
      const result = foldModel.score(X_test, y_test);

      if (taskType === 'classification') {
        scores.push(result.accuracy);
      } else {
        scores.push(result.r2Score);
      }
    }

    const meanScore = scores.reduce((sum, s) => sum + s, 0) / scores.length;
    const stdScore = Math.sqrt(
      scores.reduce((sum, s) => sum + Math.pow(s - meanScore, 2), 0) / scores.length
    );

    return {
      scores: scores,
      meanScore: meanScore,
      stdScore: stdScore,
      folds: folds
    };
  }

  trainTestSplit(X, y, testSize = 0.2, shuffle = true) {
    const n = X.length;
    const indices = Array.from({ length: n }, (_, i) => i);

    if (shuffle) {
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }

    const testCount = Math.floor(n * testSize);
    const trainCount = n - testCount;

    const trainIndices = indices.slice(0, trainCount);
    const testIndices = indices.slice(trainCount);

    return {
      X_train: trainIndices.map(i => X[i]),
      X_test: testIndices.map(i => X[i]),
      y_train: trainIndices.map(i => y[i]),
      y_test: testIndices.map(i => y[i])
    };
  }

  // ====== Model Comparison ======
  compareModels(models, X, y, taskType = 'classification') {
    const { X_train, X_test, y_train, y_test } = this.trainTestSplit(X, y, 0.2);
    const results = [];

    models.forEach(({ name, model }) => {
      const startTime = Date.now();

      model.fit(X_train, y_train, taskType);
      const trainTime = Date.now() - startTime;

      const evalStart = Date.now();
      const score = model.score(X_test, y_test);
      const evalTime = Date.now() - evalStart;

      results.push({
        name: name,
        score: taskType === 'classification' ? score.accuracy : score.r2Score,
        trainTime: trainTime,
        evalTime: evalTime,
        fullScore: score
      });
    });

    // Sort by score
    results.sort((a, b) => b.score - a.score);

    return {
      results: results,
      bestModel: results[0],
      comparison: this.generateComparisonReport(results, taskType)
    };
  }

  generateComparisonReport(results, taskType) {
    const metric = taskType === 'classification' ? 'Accuracy' : 'R² Score';

    let report = '\n' + '='.repeat(70) + '\n';
    report += '📊 MODEL COMPARISON REPORT\n';
    report += '='.repeat(70) + '\n\n';

    report += `Metric: ${metric}\n\n`;
    report += 'Rank | Model                    | Score    | Train Time | Eval Time\n';
    report += '-----+-------------------------+----------+------------+-----------\n';

    results.forEach((result, idx) => {
      const rank = (idx + 1).toString().padStart(4);
      const name = result.name.padEnd(24);
      const score = result.score.toFixed(4).padStart(8);
      const trainTime = (result.trainTime + 'ms').padStart(10);
      const evalTime = (result.evalTime + 'ms').padStart(9);

      report += `${rank} | ${name} | ${score} | ${trainTime} | ${evalTime}\n`;
    });

    report += '\n' + '='.repeat(70) + '\n';
    report += `🏆 Best Model: ${results[0].name} (${metric}: ${results[0].score.toFixed(4)})\n`;
    report += '='.repeat(70) + '\n';

    return report;
  }

  // ====== Feature Engineering ======
  polynomialFeatures(X, degree = 2) {
    return X.map(row => {
      const features = [...row];

      // Add polynomial features
      for (let d = 2; d <= degree; d++) {
        for (let i = 0; i < row.length; i++) {
          features.push(Math.pow(row[i], d));
        }
      }

      // Add interaction features
      if (degree >= 2) {
        for (let i = 0; i < row.length; i++) {
          for (let j = i + 1; j < row.length; j++) {
            features.push(row[i] * row[j]);
          }
        }
      }

      return features;
    });
  }

  standardScaler(X) {
    const n = X.length;
    const m = X[0].length;
    const means = [];
    const stds = [];

    for (let j = 0; j < m; j++) {
      const column = X.map(row => row[j]);
      const mean = column.reduce((sum, val) => sum + val, 0) / n;
      const variance = column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
      const std = Math.sqrt(variance);

      means.push(mean);
      stds.push(std === 0 ? 1 : std);
    }

    const scaled = X.map(row =>
      row.map((val, j) => (val - means[j]) / stds[j])
    );

    return {
      scaled: scaled,
      means: means,
      stds: stds,
      transform: (newX) => newX.map(row =>
        row.map((val, j) => (val - means[j]) / stds[j])
      )
    };
  }

  minMaxScaler(X, featureRange = [0, 1]) {
    const n = X.length;
    const m = X[0].length;
    const mins = [];
    const maxs = [];
    const [min_range, max_range] = featureRange;

    for (let j = 0; j < m; j++) {
      const column = X.map(row => row[j]);
      mins.push(Math.min(...column));
      maxs.push(Math.max(...column));
    }

    const scaled = X.map(row =>
      row.map((val, j) => {
        const range = maxs[j] - mins[j];
        if (range === 0) return min_range;
        return min_range + ((val - mins[j]) / range) * (max_range - min_range);
      })
    );

    return {
      scaled: scaled,
      mins: mins,
      maxs: maxs,
      transform: (newX) => newX.map(row =>
        row.map((val, j) => {
          const range = maxs[j] - mins[j];
          if (range === 0) return min_range;
          return min_range + ((val - mins[j]) / range) * (max_range - min_range);
        })
      )
    };
  }

  // ====== Metrics ======
  rocCurve(yTrue, yProba) {
    const scores = yProba.map((proba, i) => ({
      probability: typeof proba === 'object' ? Object.values(proba)[1] : proba,
      label: yTrue[i]
    }));

    scores.sort((a, b) => b.probability - a.probability);

    const positives = yTrue.filter(y => y === 1 || y === true).length;
    const negatives = yTrue.length - positives;

    const tpr = [0];
    const fpr = [0];
    let tp = 0;
    let fp = 0;

    scores.forEach(score => {
      if (score.label === 1 || score.label === true) {
        tp++;
      } else {
        fp++;
      }
      tpr.push(tp / positives);
      fpr.push(fp / negatives);
    });

    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < fpr.length; i++) {
      auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
    }

    return {
      fpr: fpr,
      tpr: tpr,
      auc: auc,
      thresholds: scores.map(s => s.probability)
    };
  }

  precisionRecallCurve(yTrue, yProba) {
    const scores = yProba.map((proba, i) => ({
      probability: typeof proba === 'object' ? Object.values(proba)[1] : proba,
      label: yTrue[i]
    }));

    scores.sort((a, b) => b.probability - a.probability);

    const precision = [];
    const recall = [];
    let tp = 0;
    let fp = 0;
    const totalPositives = yTrue.filter(y => y === 1 || y === true).length;

    scores.forEach(score => {
      if (score.label === 1 || score.label === true) {
        tp++;
      } else {
        fp++;
      }

      const currentPrecision = tp / (tp + fp);
      const currentRecall = tp / totalPositives;

      precision.push(currentPrecision);
      recall.push(currentRecall);
    });

    return {
      precision: precision,
      recall: recall,
      thresholds: scores.map(s => s.probability)
    };
  }

  // ====== Quick Training Helper ======
  quickTrain(modelType, X, y, options = {}) {
    const { taskType = 'classification', testSize = 0.2, normalize = true } = options;

    let model;

    switch (modelType.toLowerCase()) {
      case 'linear':
      case 'linearregression':
        model = this.createLinearRegression(options);
        break;
      case 'logistic':
      case 'logisticregression':
        model = this.createLogisticRegression(options);
        break;
      case 'knn':
        model = this.createKNN(options);
        break;
      case 'tree':
      case 'decisiontree':
        model = this.createDecisionTree(options);
        break;
      case 'forest':
      case 'randomforest':
        model = this.createRandomForest(options);
        break;
      case 'naivebayes':
      case 'nb':
        model = this.createNaiveBayes(options);
        break;
      case 'svm':
        model = this.createSVM(options);
        break;
      default:
        throw new Error(`Unknown model type: ${modelType}`);
    }

    const { X_train, X_test, y_train, y_test } = this.trainTestSplit(X, y, testSize);

    console.log(`\n🚀 Training ${modelType}...`);
    const startTime = Date.now();

    model.fit(X_train, y_train, normalize, taskType);

    const trainTime = Date.now() - startTime;
    console.log(`✅ Training completed in ${trainTime}ms`);

    console.log(`\n📊 Evaluating model...`);
    const score = model.score(X_test, y_test);

    console.log(`\n${'='.repeat(60)}`);
    console.log(`📈 RESULTS`);
    console.log(`${'='.repeat(60)}`);

    if (taskType === 'classification') {
      console.log(`Accuracy: ${(score.accuracy * 100).toFixed(2)}%`);
      console.log(`\nConfusion Matrix:${score.confusionMatrix.display}`);

      console.log(`\nPer-Class Metrics:`);
      Object.keys(score.classMetrics).forEach(cls => {
        const m = score.classMetrics[cls];
        console.log(`  ${cls}:`);
        console.log(`    Precision: ${(m.precision * 100).toFixed(2)}%`);
        console.log(`    Recall: ${(m.recall * 100).toFixed(2)}%`);
        console.log(`    F1-Score: ${(m.f1Score * 100).toFixed(2)}%`);
      });
    } else {
      console.log(`R² Score: ${score.r2Score.toFixed(4)}`);
      console.log(`MSE: ${score.mse.toFixed(4)}`);
      console.log(`RMSE: ${score.rmse.toFixed(4)}`);
      console.log(`MAE: ${score.mae.toFixed(4)}`);
    }

    console.log(`\n${'='.repeat(60)}\n`);

    return {
      model: model,
      score: score,
      trainTime: trainTime,
      summary: model.summary()
    };
  }
}

export default MachineLearning;