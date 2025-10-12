import BaseModel from './baseModel.js';

class LogisticRegression extends BaseModel {
  constructor(learningRate = 0.01, iterations = 1000, regularization = null, lambda = 0.01) {
    super();
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.regularization = regularization;
    this.lambda = lambda;
    this.weights = null;
    this.bias = null;
    this.normParams = null;
    this.classes = null;
    this.multiclass = false;
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  softmax(z) {
    const maxZ = Math.max(...z);
    const expZ = z.map(val => Math.exp(val - maxZ));
    const sumExpZ = expZ.reduce((a, b) => a + b, 0);
    return expZ.map(val => val / sumExpZ);
  }

  fit(X, y, normalize = true) {
    this.validateTrainingData(X, y);

    this.classes = [...new Set(y)].sort();
    this.multiclass = this.classes.length > 2;

    let X_train = X.map(row => Array.isArray(row) ? row : [row]);

    if (normalize) {
      const { normalized, means, stds } = this.normalizeFeatures(X_train);
      X_train = normalized;
      this.normParams = { means, stds };
    }

    if (this.multiclass) {
      this.fitMulticlass(X_train, y);
    } else {
      this.fitBinary(X_train, y);
    }

    this.trained = true;
    return this;
  }

  fitBinary(X_train, y) {
    const n = X_train.length;
    const m = X_train[0].length;

    // Convert labels to 0 and 1
    const yBinary = y.map(label => label === this.classes[1] ? 1 : 0);

    this.weights = Array(m).fill(0);
    this.bias = 0;

    const losses = [];

    for (let iter = 0; iter < this.iterations; iter++) {
      const predictions = X_train.map(x => {
        const z = this.bias + x.reduce((sum, val, j) => sum + val * this.weights[j], 0);
        return this.sigmoid(z);
      });

      const weightGradients = Array(m).fill(0);
      let biasGradient = 0;

      for (let i = 0; i < n; i++) {
        const error = predictions[i] - yBinary[i];
        biasGradient += error;
        for (let j = 0; j < m; j++) {
          weightGradients[j] += error * X_train[i][j];
        }
      }

      for (let j = 0; j < m; j++) {
        if (this.regularization === 'l2') {
          weightGradients[j] += this.lambda * this.weights[j];
        } else if (this.regularization === 'l1') {
          weightGradients[j] += this.lambda * Math.sign(this.weights[j]);
        }
        this.weights[j] -= (this.learningRate / n) * weightGradients[j];
      }

      this.bias -= (this.learningRate / n) * biasGradient;

      const loss = this.calculateBinaryLoss(predictions, yBinary);
      losses.push(loss);
    }

    this.trainingMetrics = {
      finalLoss: losses[losses.length - 1],
      losses: losses,
      weights: [...this.weights],
      bias: this.bias
    };
  }

  fitMulticlass(X_train, y) {
    const n = X_train.length;
    const m = X_train[0].length;
    const k = this.classes.length;

    // One-hot encode labels
    const yOneHot = y.map(label => {
      const encoded = Array(k).fill(0);
      encoded[this.classes.indexOf(label)] = 1;
      return encoded;
    });

    this.weights = Array(k).fill(0).map(() => Array(m).fill(0));
    this.bias = Array(k).fill(0);

    const losses = [];

    for (let iter = 0; iter < this.iterations; iter++) {
      const predictions = X_train.map(x => {
        const z = this.bias.map((b, c) =>
          b + x.reduce((sum, val, j) => sum + val * this.weights[c][j], 0)
        );
        return this.softmax(z);
      });

      for (let c = 0; c < k; c++) {
        let biasGradient = 0;
        const weightGradients = Array(m).fill(0);

        for (let i = 0; i < n; i++) {
          const error = predictions[i][c] - yOneHot[i][c];
          biasGradient += error;
          for (let j = 0; j < m; j++) {
            weightGradients[j] += error * X_train[i][j];
          }
        }

        for (let j = 0; j < m; j++) {
          if (this.regularization === 'l2') {
            weightGradients[j] += this.lambda * this.weights[c][j];
          }
          this.weights[c][j] -= (this.learningRate / n) * weightGradients[j];
        }

        this.bias[c] -= (this.learningRate / n) * biasGradient;
      }

      const loss = this.calculateMulticlassLoss(predictions, yOneHot);
      losses.push(loss);
    }

    this.trainingMetrics = {
      finalLoss: losses[losses.length - 1],
      losses: losses
    };
  }

  predict(X, returnProba = false) {
    this.validatePredictionData(X);

    let X_test = X.map(row => Array.isArray(row) ? row : [row]);

    if (this.normParams) {
      const { means, stds } = this.normParams;
      X_test = X_test.map(row =>
        row.map((val, j) => (val - means[j]) / stds[j])
      );
    }

    if (this.multiclass) {
      const probas = X_test.map(x => {
        const z = this.bias.map((b, c) =>
          b + x.reduce((sum, val, j) => sum + val * this.weights[c][j], 0)
        );
        return this.softmax(z);
      });

      if (returnProba) {
        return probas.map(proba => {
          const obj = {};
          this.classes.forEach((cls, i) => {
            obj[cls] = proba[i];
          });
          return obj;
        });
      }

      return probas.map(proba => {
        const maxIdx = proba.indexOf(Math.max(...proba));
        return this.classes[maxIdx];
      });
    } else {
      const probas = X_test.map(x => {
        const z = this.bias + x.reduce((sum, val, j) => sum + val * this.weights[j], 0);
        return this.sigmoid(z);
      });

      if (returnProba) {
        return probas.map(p => ({
          [this.classes[0]]: 1 - p,
          [this.classes[1]]: p
        }));
      }

      return probas.map(p => p >= 0.5 ? this.classes[1] : this.classes[0]);
    }
  }

  predictProba(X) {
    return this.predict(X, true);
  }

  calculateBinaryLoss(predictions, y) {
    const eps = 1e-15;
    const loss = predictions.reduce((sum, pred, i) => {
      const p = Math.max(eps, Math.min(1 - eps, pred));
      return sum - (y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
    }, 0) / predictions.length;

    if (this.regularization === 'l2') {
      const l2 = this.weights.reduce((sum, w) => sum + w * w, 0);
      return loss + this.lambda * l2 / 2;
    }
    return loss;
  }

  calculateMulticlassLoss(predictions, yOneHot) {
    const eps = 1e-15;
    let loss = 0;
    for (let i = 0; i < predictions.length; i++) {
      for (let c = 0; c < yOneHot[i].length; c++) {
        const p = Math.max(eps, Math.min(1 - eps, predictions[i][c]));
        loss -= yOneHot[i][c] * Math.log(p);
      }
    }
    return loss / predictions.length;
  }

  score(X, y) {
    const predictions = this.predict(X);
    const probas = this.predictProba(X);

    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      if (predictions[i] === y[i]) correct++;
    }
    const accuracy = correct / y.length;

    // Confusion Matrix
    const cm = this.confusionMatrix(y, predictions);

    // Per-class metrics
    const metrics = this.calculateClassMetrics(cm);

    return {
      accuracy: accuracy,
      confusionMatrix: cm,
      classMetrics: metrics,
      predictions: predictions,
      probabilities: probas
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
      const tn = matrix.reduce((sum, row, r) =>
        sum + row.reduce((s, val, c) => s + (r !== i && c !== i ? val : 0), 0), 0);

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
      modelType: 'Logistic Regression',
      classes: this.classes,
      multiclass: this.multiclass,
      trainingMetrics: this.trainingMetrics,
      hyperparameters: {
        learningRate: this.learningRate,
        iterations: this.iterations,
        regularization: this.regularization,
        lambda: this.lambda
      }
    };
  }
}

export default LogisticRegression;