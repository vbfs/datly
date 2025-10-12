import BaseModel from './baseModel.js';

class SupportVectorMachine extends BaseModel {
  constructor(C = 1.0, kernel = 'linear', gamma = 'scale', degree = 3, learningRate = 0.001, iterations = 1000) {
    super();
    this.C = C; // Regularization parameter
    this.kernel = kernel; // 'linear', 'rbf', 'poly'
    this.gamma = gamma; // Kernel coefficient ('scale', 'auto', or number)
    this.degree = degree; // Degree for polynomial kernel
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.weights = null;
    this.bias = null;
    this.supportVectors = null;
    this.classes = null;
    this.normParams = null;
  }

  fit(X, y, normalize = true) {
    this.validateTrainingData(X, y);

    let X_train = X.map(row => Array.isArray(row) ? row : [row]);
    this.classes = [...new Set(y)].sort();

    if (this.classes.length !== 2) {
      throw new Error('SVM currently only supports binary classification');
    }

    // Convert labels to -1 and 1
    const yBinary = y.map(label => label === this.classes[1] ? 1 : -1);

    if (normalize) {
      const { normalized, means, stds } = this.normalizeFeatures(X_train);
      X_train = normalized;
      this.normParams = { means, stds };
    }

    const nSamples = X_train.length;
    const nFeatures = X_train[0].length;

    // Calculate gamma if set to 'scale' or 'auto'
    if (this.gamma === 'scale') {
      const variance = this.calculateVariance(X_train);
      this.gamma = 1 / (nFeatures * variance);
    } else if (this.gamma === 'auto') {
      this.gamma = 1 / nFeatures;
    }

    // Initialize weights and bias
    this.weights = Array(nFeatures).fill(0);
    this.bias = 0;

    const losses = [];

    // Simplified SMO-like algorithm (gradient descent)
    for (let iter = 0; iter < this.iterations; iter++) {
      let loss = 0;

      for (let i = 0; i < nSamples; i++) {
        const xi = X_train[i];
        const yi = yBinary[i];

        const prediction = this.decisionFunction([xi])[0];
        const margin = yi * prediction;

        if (margin < 1) {
          // Update weights for misclassified or margin violations
          for (let j = 0; j < nFeatures; j++) {
            this.weights[j] += this.learningRate * (yi * xi[j] - 2 * (1 / this.C) * this.weights[j]);
          }
          this.bias += this.learningRate * yi;
          loss += 1 - margin;
        } else {
          // Update weights for correct classifications
          for (let j = 0; j < nFeatures; j++) {
            this.weights[j] += this.learningRate * (-2 * (1 / this.C) * this.weights[j]);
          }
        }
      }

      // Add regularization term to loss
      const regTerm = (1 / this.C) * this.weights.reduce((sum, w) => sum + w * w, 0);
      losses.push(loss / nSamples + regTerm);
    }

    // Identify support vectors (samples near the margin)
    this.identifySupportVectors(X_train, yBinary);

    this.trained = true;

    this.trainingMetrics = {
      finalLoss: losses[losses.length - 1],
      losses: losses,
      nSupportVectors: this.supportVectors.length,
      supportVectorRatio: this.supportVectors.length / nSamples
    };

    return this;
  }

  calculateVariance(X) {
    const n = X.length;
    const m = X[0].length;
    let totalVariance = 0;

    for (let j = 0; j < m; j++) {
      const column = X.map(row => row[j]);
      const mean = column.reduce((sum, val) => sum + val, 0) / n;
      const variance = column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
      totalVariance += variance;
    }

    return totalVariance / m;
  }

  identifySupportVectors(X, y) {
    this.supportVectors = [];

    for (let i = 0; i < X.length; i++) {
      const prediction = this.decisionFunction([X[i]])[0];
      const margin = Math.abs(prediction);

      // Support vectors are points close to the decision boundary
      if (margin < 1.5) {
        this.supportVectors.push({
          index: i,
          vector: X[i],
          label: y[i],
          margin: margin
        });
      }
    }
  }

  kernelFunction(x1, x2) {
    if (this.kernel === 'linear') {
      return x1.reduce((sum, val, i) => sum + val * x2[i], 0);
    } else if (this.kernel === 'rbf') {
      const squaredDistance = x1.reduce((sum, val, i) =>
        sum + Math.pow(val - x2[i], 2), 0);
      return Math.exp(-this.gamma * squaredDistance);
    } else if (this.kernel === 'poly') {
      const dotProduct = x1.reduce((sum, val, i) => sum + val * x2[i], 0);
      return Math.pow(dotProduct + 1, this.degree);
    }

    return 0;
  }

  decisionFunction(X) {
    return X.map(x => {
      let score = this.bias;
      for (let j = 0; j < this.weights.length; j++) {
        score += this.weights[j] * x[j];
      }
      return score;
    });
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

    const decisions = this.decisionFunction(X_test);
    return decisions.map(score => score >= 0 ? this.classes[1] : this.classes[0]);
  }

  predictProba(X) {
    this.validatePredictionData(X);

    let X_test = X.map(row => Array.isArray(row) ? row : [row]);

    if (this.normParams) {
      const { means, stds } = this.normParams;
      X_test = X_test.map(row =>
        row.map((val, j) => (val - means[j]) / stds[j])
      );
    }

    const decisions = this.decisionFunction(X_test);

    // Use sigmoid function to convert decision scores to probabilities
    return decisions.map(score => {
      const prob1 = 1 / (1 + Math.exp(-score));
      return {
        [this.classes[0]]: 1 - prob1,
        [this.classes[1]]: prob1
      };
    });
  }

  score(X, y) {
    const predictions = this.predict(X);

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

  summary() {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }

    return {
      modelType: 'Support Vector Machine',
      classes: this.classes,
      trainingMetrics: this.trainingMetrics,
      hyperparameters: {
        C: this.C,
        kernel: this.kernel,
        gamma: this.gamma,
        degree: this.degree,
        learningRate: this.learningRate,
        iterations: this.iterations
      }
    };
  }
}

export default SupportVectorMachine;