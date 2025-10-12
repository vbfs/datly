import BaseModel from './baseModel.js';

class KNearestNeighbors extends BaseModel {
  constructor(k = 5, metric = 'euclidean', weights = 'uniform') {
    super();
    this.k = k;
    this.metric = metric; // 'euclidean', 'manhattan', 'minkowski'
    this.weights = weights; // 'uniform' or 'distance'
    this.X_train = null;
    this.y_train = null;
    this.normParams = null;
    this.taskType = null; // 'classification' or 'regression'
  }

  euclideanDistance(x1, x2) {
    return Math.sqrt(
      x1.reduce((sum, val, i) => sum + Math.pow(val - x2[i], 2), 0)
    );
  }

  manhattanDistance(x1, x2) {
    return x1.reduce((sum, val, i) => sum + Math.abs(val - x2[i]), 0);
  }

  minkowskiDistance(x1, x2, p = 3) {
    return Math.pow(
      x1.reduce((sum, val, i) => sum + Math.pow(Math.abs(val - x2[i]), p), 0),
      1 / p
    );
  }

  calculateDistance(x1, x2) {
    switch (this.metric) {
      case 'manhattan':
        return this.manhattanDistance(x1, x2);
      case 'minkowski':
        return this.minkowskiDistance(x1, x2);
      case 'euclidean':
      default:
        return this.euclideanDistance(x1, x2);
    }
  }

  fit(X, y, normalize = true, taskType = 'classification') {
    this.validateTrainingData(X, y);

    this.taskType = taskType;
    let X_train = X.map(row => Array.isArray(row) ? row : [row]);

    if (normalize) {
      const { normalized, means, stds } = this.normalizeFeatures(X_train);
      this.X_train = normalized;
      this.normParams = { means, stds };
    } else {
      this.X_train = X_train;
    }

    this.y_train = [...y];
    this.trained = true;

    this.trainingMetrics = {
      samples: this.X_train.length,
      features: this.X_train[0].length,
      taskType: this.taskType
    };

    return this;
  }

  findKNearest(x) {
    const distances = this.X_train.map((trainPoint, idx) => ({
      distance: this.calculateDistance(x, trainPoint),
      index: idx,
      label: this.y_train[idx]
    }));

    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, this.k);
  }

  predictSingleClassification(x) {
    const neighbors = this.findKNearest(x);

    if (this.weights === 'uniform') {
      const votes = {};
      neighbors.forEach(neighbor => {
        votes[neighbor.label] = (votes[neighbor.label] || 0) + 1;
      });

      return Object.keys(votes).reduce((a, b) =>
        votes[a] > votes[b] ? a : b
      );
    } else {
      // Distance-weighted voting
      const votes = {};
      neighbors.forEach(neighbor => {
        const weight = neighbor.distance === 0 ? 1e10 : 1 / neighbor.distance;
        votes[neighbor.label] = (votes[neighbor.label] || 0) + weight;
      });

      return Object.keys(votes).reduce((a, b) =>
        votes[a] > votes[b] ? a : b
      );
    }
  }

  predictSingleRegression(x) {
    const neighbors = this.findKNearest(x);

    if (this.weights === 'uniform') {
      return neighbors.reduce((sum, n) => sum + n.label, 0) / neighbors.length;
    } else {
      // Distance-weighted average
      let weightedSum = 0;
      let totalWeight = 0;

      neighbors.forEach(neighbor => {
        const weight = neighbor.distance === 0 ? 1e10 : 1 / neighbor.distance;
        weightedSum += neighbor.label * weight;
        totalWeight += weight;
      });

      return weightedSum / totalWeight;
    }
  }

  predict(X) {
    this.validatePredictionData(X);

    let X_test = X.map(row => Array.isArray(row) ? row : [row]);

    if (this.normParams) {
      const { means, stds } = this.normParams;
      X_test = X_test.map(row =>
        row.map((val, j) => (val - means[j]) / stds[j])
      );
    }

    if (this.taskType === 'classification') {
      return X_test.map(x => this.predictSingleClassification(x));
    } else {
      return X_test.map(x => this.predictSingleRegression(x));
    }
  }

  predictProba(X) {
    if (this.taskType !== 'classification') {
      throw new Error('predictProba is only available for classification tasks');
    }

    this.validatePredictionData(X);

    let X_test = X.map(row => Array.isArray(row) ? row : [row]);

    if (this.normParams) {
      const { means, stds } = this.normParams;
      X_test = X_test.map(row =>
        row.map((val, j) => (val - means[j]) / stds[j])
      );
    }

    const classes = [...new Set(this.y_train)].sort();

    return X_test.map(x => {
      const neighbors = this.findKNearest(x);
      const probas = {};

      classes.forEach(cls => {
        probas[cls] = 0;
      });

      if (this.weights === 'uniform') {
        neighbors.forEach(neighbor => {
          probas[neighbor.label] += 1 / this.k;
        });
      } else {
        let totalWeight = 0;
        const weights = {};

        neighbors.forEach(neighbor => {
          const weight = neighbor.distance === 0 ? 1e10 : 1 / neighbor.distance;
          weights[neighbor.label] = (weights[neighbor.label] || 0) + weight;
          totalWeight += weight;
        });

        Object.keys(weights).forEach(label => {
          probas[label] = weights[label] / totalWeight;
        });
      }

      return probas;
    });
  }

  score(X, y) {
    const predictions = this.predict(X);

    if (this.taskType === 'classification') {
      let correct = 0;
      for (let i = 0; i < y.length; i++) {
        if (predictions[i] === y[i]) correct++;
      }
      const accuracy = correct / y.length;

      const classes = [...new Set([...y, ...predictions])].sort();
      const cm = this.confusionMatrix(y, predictions, classes);
      const metrics = this.calculateClassMetrics(cm, classes);

      return {
        accuracy: accuracy,
        confusionMatrix: cm,
        classMetrics: metrics,
        predictions: predictions
      };
    } else {
      // Regression metrics
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

  confusionMatrix(yTrue, yPred, classes) {
    const n = classes.length;
    const matrix = Array(n).fill(0).map(() => Array(n).fill(0));

    for (let i = 0; i < yTrue.length; i++) {
      const trueIdx = classes.indexOf(yTrue[i]);
      const predIdx = classes.indexOf(yPred[i]);
      matrix[trueIdx][predIdx]++;
    }

    return {
      matrix: matrix,
      classes: classes,
      display: this.formatConfusionMatrix(matrix, classes)
    };
  }

  formatConfusionMatrix(matrix, classes) {
    const maxLen = Math.max(...matrix.flat().map(v => v.toString().length), 8);
    const pad = (str) => str.toString().padStart(maxLen);

    let output = '\n' + ' '.repeat(maxLen + 2) + 'Predicted\n';
    output += ' '.repeat(maxLen + 2) + classes.map(c => pad(c)).join(' ') + '\n';

    for (let i = 0; i < matrix.length; i++) {
      if (i === 0) output += 'Actual ';
      else output += '       ';
      output += pad(classes[i]) + ' ';
      output += matrix[i].map(v => pad(v)).join(' ') + '\n';
    }

    return output;
  }

  calculateClassMetrics(cm, classes) {
    const matrix = cm.matrix;
    const metrics = {};

    classes.forEach((cls, i) => {
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

  summary() {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }

    return {
      modelType: 'K-Nearest Neighbors',
      taskType: this.taskType,
      trainingMetrics: this.trainingMetrics,
      hyperparameters: {
        k: this.k,
        metric: this.metric,
        weights: this.weights
      }
    };
  }
}

export default KNearestNeighbors;