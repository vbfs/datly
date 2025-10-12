import BaseModel from './baseModel.js';

class DecisionTree extends BaseModel {
  constructor(maxDepth = 10, minSamplesSplit = 2, minSamplesLeaf = 1, criterion = 'gini') {
    super();
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.criterion = criterion; // 'gini', 'entropy' for classification; 'mse' for regression
    this.tree = null;
    this.taskType = null;
    this.classes = null;
  }

  fit(X, y, taskType = 'classification') {
    this.validateTrainingData(X, y);

    this.taskType = taskType;
    const X_train = X.map(row => Array.isArray(row) ? row : [row]);

    if (taskType === 'classification') {
      this.classes = [...new Set(y)].sort();
    }

    this.tree = this.buildTree(X_train, y, 0);
    this.trained = true;

    this.trainingMetrics = {
      treeDepth: this.getTreeDepth(this.tree),
      leafCount: this.getLeafCount(this.tree),
      nodeCount: this.getNodeCount(this.tree),
      taskType: this.taskType
    };

    return this;
  }

  buildTree(X, y, depth) {
    const nSamples = X.length;
    const nFeatures = X[0].length;

    // Stopping criteria
    if (depth >= this.maxDepth ||
        nSamples < this.minSamplesSplit ||
        this.isPure(y)) {
      return this.createLeaf(y);
    }

    // Find best split
    let bestSplit = null;
    let bestScore = -Infinity;

    for (let featureIdx = 0; featureIdx < nFeatures; featureIdx++) {
      const thresholds = this.getThresholds(X, featureIdx);

      for (const threshold of thresholds) {
        const { left, right } = this.splitData(X, y, featureIdx, threshold);

        if (left.y.length < this.minSamplesLeaf ||
            right.y.length < this.minSamplesLeaf) {
          continue;
        }

        const score = this.calculateSplitScore(y, left.y, right.y);

        if (score > bestScore) {
          bestScore = score;
          bestSplit = {
            featureIdx,
            threshold,
            left,
            right
          };
        }
      }
    }

    if (!bestSplit) {
      return this.createLeaf(y);
    }

    // Recursively build subtrees
    return {
      featureIdx: bestSplit.featureIdx,
      threshold: bestSplit.threshold,
      left: this.buildTree(bestSplit.left.X, bestSplit.left.y, depth + 1),
      right: this.buildTree(bestSplit.right.X, bestSplit.right.y, depth + 1),
      isLeaf: false
    };
  }

  getThresholds(X, featureIdx) {
    const values = [...new Set(X.map(row => row[featureIdx]))].sort((a, b) => a - b);
    const thresholds = [];

    for (let i = 0; i < values.length - 1; i++) {
      thresholds.push((values[i] + values[i + 1]) / 2);
    }

    return thresholds;
  }

  splitData(X, y, featureIdx, threshold) {
    const leftX = [], leftY = [];
    const rightX = [], rightY = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][featureIdx] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return {
      left: { X: leftX, y: leftY },
      right: { X: rightX, y: rightY }
    };
  }

  isPure(y) {
    return new Set(y).size === 1;
  }

  createLeaf(y) {
    if (this.taskType === 'classification') {
      const counts = {};
      y.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      const prediction = Object.keys(counts).reduce((a, b) =>
        counts[a] > counts[b] ? a : b
      );
      return {
        isLeaf: true,
        prediction: prediction,
        samples: y.length,
        distribution: counts
      };
    } else {
      const mean = y.reduce((sum, val) => sum + val, 0) / y.length;
      return {
        isLeaf: true,
        prediction: mean,
        samples: y.length
      };
    }
  }

  calculateSplitScore(parentY, leftY, rightY) {
    if (this.taskType === 'classification') {
      const parentImpurity = this.calculateImpurity(parentY);
      const n = parentY.length;
      const nLeft = leftY.length;
      const nRight = rightY.length;

      const leftImpurity = this.calculateImpurity(leftY);
      const rightImpurity = this.calculateImpurity(rightY);

      const weightedImpurity = (nLeft / n) * leftImpurity + (nRight / n) * rightImpurity;
      return parentImpurity - weightedImpurity; // Information gain
    } else {
      const parentVariance = this.calculateVariance(parentY);
      const n = parentY.length;
      const nLeft = leftY.length;
      const nRight = rightY.length;

      const leftVariance = this.calculateVariance(leftY);
      const rightVariance = this.calculateVariance(rightY);

      const weightedVariance = (nLeft / n) * leftVariance + (nRight / n) * rightVariance;
      return parentVariance - weightedVariance; // Variance reduction
    }
  }

  calculateImpurity(y) {
    const counts = {};
    y.forEach(label => {
      counts[label] = (counts[label] || 0) + 1;
    });

    const n = y.length;
    const probabilities = Object.values(counts).map(count => count / n);

    if (this.criterion === 'gini') {
      return 1 - probabilities.reduce((sum, p) => sum + p * p, 0);
    } else if (this.criterion === 'entropy') {
      return -probabilities.reduce((sum, p) => sum + p * Math.log2(p), 0);
    }
  }

  calculateVariance(y) {
    if (y.length === 0) return 0;
    const mean = y.reduce((sum, val) => sum + val, 0) / y.length;
    return y.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / y.length;
  }

  predictSingle(x, node = this.tree) {
    if (node.isLeaf) {
      return node.prediction;
    }

    if (x[node.featureIdx] <= node.threshold) {
      return this.predictSingle(x, node.left);
    } else {
      return this.predictSingle(x, node.right);
    }
  }

  predict(X) {
    this.validatePredictionData(X);

    const X_test = X.map(row => Array.isArray(row) ? row : [row]);
    return X_test.map(x => this.predictSingle(x));
  }

  predictProba(X) {
    if (this.taskType !== 'classification') {
      throw new Error('predictProba is only available for classification tasks');
    }

    this.validatePredictionData(X);

    const X_test = X.map(row => Array.isArray(row) ? row : [row]);

    return X_test.map(x => {
      const leaf = this.findLeaf(x);
      const total = leaf.samples;
      const probas = {};

      this.classes.forEach(cls => {
        probas[cls] = (leaf.distribution[cls] || 0) / total;
      });

      return probas;
    });
  }

  findLeaf(x, node = this.tree) {
    if (node.isLeaf) {
      return node;
    }

    if (x[node.featureIdx] <= node.threshold) {
      return this.findLeaf(x, node.left);
    } else {
      return this.findLeaf(x, node.right);
    }
  }

  score(X, y) {
    const predictions = this.predict(X);

    if (this.taskType === 'classification') {
      let correct = 0;
      for (let i = 0; i < y.length; i++) {
        if (predictions[i] === y[i]) correct++;
      }
      const accuracy = correct / y.length;

      const cm = this.confusionMatrix(y, predictions);
      const metrics = this.calculateClassMetrics(cm);

      return {
        accuracy: accuracy,
        confusionMatrix: cm,
        classMetrics: metrics,
        predictions: predictions
      };
    } else {
      const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;

      const ssRes = predictions.reduce((sum, pred, i) =>
        sum + Math.pow(y[i] - pred, 2), 0);
      const ssTot = y.reduce((sum, val) =>
        sum + Math.pow(val - yMean, 2), 0);

      const r2 = 1 - (ssRes / ssTot);
      const mse = ssRes / y.length;
      const rmse = Math.sqrt(mse);
      const mae = predictions.reduce((sum, pred, i) =>
        sum + Math.abs(y[i] - pred), 0) / y.length;

      return {
        r2Score: r2,
        mse: mse,
        rmse: rmse,
        mae: mae,
        predictions: predictions,
        residuals: predictions.map((pred, i) => y[i] - pred)
      };
    }
  }

  confusionMatrix(yTrue, yPred) {
    const n = this.classes.length;
    const matrix = Array(n).fill(0).map(() => Array(n).fill(0));

    for (let i = 0; i < yTrue.length; i++) {
      const trueIdx = this.classes.indexOf(yTrue[i]);
      const predIdx = this.classes.indexOf(yPred[i]);
      matrix[trueIdx][predIdx]++;
    }

    return {
      matrix: matrix,
      classes: this.classes,
      display: this.formatConfusionMatrix(matrix)
    };
  }

  formatConfusionMatrix(matrix) {
    const maxLen = Math.max(...matrix.flat().map(v => v.toString().length), 8);
    const pad = (str) => str.toString().padStart(maxLen);

    let output = '\n' + ' '.repeat(maxLen + 2) + 'Predicted\n';
    output += ' '.repeat(maxLen + 2) + this.classes.map(c => pad(c)).join(' ') + '\n';

    for (let i = 0; i < matrix.length; i++) {
      if (i === 0) output += 'Actual ';
      else output += '       ';
      output += pad(this.classes[i]) + ' ';
      output += matrix[i].map(v => pad(v)).join(' ') + '\n';
    }

    return output;
  }

  calculateClassMetrics(cm) {
    const matrix = cm.matrix;
    const metrics = {};

    this.classes.forEach((cls, i) => {
      const tp = matrix[i][i];
      const fn = matrix[i].reduce((sum, val) => sum + val, 0) - tp;
      const fp = matrix.map(row => row[i]).reduce((sum, val) => sum + val, 0) - tp;

      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

      metrics[cls] = {
        precision: precision,
        recall: recall,
        f1Score: f1,
        support: tp + fn
      };
    });

    return metrics;
  }

  getTreeDepth(node) {
    if (node.isLeaf) return 0;
    return 1 + Math.max(this.getTreeDepth(node.left), this.getTreeDepth(node.right));
  }

  getLeafCount(node) {
    if (node.isLeaf) return 1;
    return this.getLeafCount(node.left) + this.getLeafCount(node.right);
  }

  getNodeCount(node) {
    if (node.isLeaf) return 1;
    return 1 + this.getNodeCount(node.left) + this.getNodeCount(node.right);
  }

  getFeatureImportance() {
    const importance = {};
    this.calculateImportance(this.tree, importance);

    const total = Object.values(importance).reduce((sum, val) => sum + val, 0);
    Object.keys(importance).forEach(key => {
      importance[key] /= total;
    });

    return importance;
  }

  calculateImportance(node, importance) {
    if (node.isLeaf) return;

    const featureName = `feature_${node.featureIdx}`;
    importance[featureName] = (importance[featureName] || 0) + 1;

    this.calculateImportance(node.left, importance);
    this.calculateImportance(node.right, importance);
  }

  summary() {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }

    return {
      modelType: 'Decision Tree',
      taskType: this.taskType,
      trainingMetrics: this.trainingMetrics,
      featureImportance: this.getFeatureImportance(),
      hyperparameters: {
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        criterion: this.criterion
      }
    };
  }
}

export default DecisionTree;