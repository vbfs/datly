import BaseModel from './baseModel.js';

class LinearRegression extends BaseModel {
  constructor(learningRate = 0.01, iterations = 1000, regularization = null, lambda = 0.01) {
    super();
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.regularization = regularization; // 'l1', 'l2', or null
    this.lambda = lambda;
    this.weights = null;
    this.bias = null;
    this.normParams = null;
  }

  fit(X, y, normalize = true) {
    this.validateTrainingData(X, y);

    let X_train = X.map(row => Array.isArray(row) ? row : [row]);

    if (normalize) {
      const { normalized, means, stds } = this.normalizeFeatures(X_train);
      X_train = normalized;
      this.normParams = { means, stds };
    }

    const n = X_train.length;
    const m = X_train[0].length;

    this.weights = Array(m).fill(0);
    this.bias = 0;

    const losses = [];

    for (let iter = 0; iter < this.iterations; iter++) {
      const predictions = X_train.map(x => this.predictSingle(x));
      const errors = predictions.map((pred, i) => pred - y[i]);

      // Gradient descent
      const weightGradients = Array(m).fill(0);
      let biasGradient = 0;

      for (let i = 0; i < n; i++) {
        biasGradient += errors[i];
        for (let j = 0; j < m; j++) {
          weightGradients[j] += errors[i] * X_train[i][j];
        }
      }

      // Apply regularization
      for (let j = 0; j < m; j++) {
        if (this.regularization === 'l2') {
          weightGradients[j] += this.lambda * this.weights[j];
        } else if (this.regularization === 'l1') {
          weightGradients[j] += this.lambda * Math.sign(this.weights[j]);
        }
        this.weights[j] -= (this.learningRate / n) * weightGradients[j];
      }

      this.bias -= (this.learningRate / n) * biasGradient;

      // Calculate loss
      const loss = this.calculateLoss(predictions, y);
      losses.push(loss);
    }

    this.trained = true;
    this.trainingMetrics = {
      finalLoss: losses[losses.length - 1],
      losses: losses,
      weights: [...this.weights],
      bias: this.bias
    };

    return this;
  }

  predictSingle(x) {
    let sum = this.bias;
    for (let j = 0; j < this.weights.length; j++) {
      sum += this.weights[j] * x[j];
    }
    return sum;
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

    return X_test.map(x => this.predictSingle(x));
  }

  calculateLoss(predictions, y) {
    const mse = predictions.reduce((sum, pred, i) =>
      sum + Math.pow(pred - y[i], 2), 0) / predictions.length;

    if (this.regularization === 'l2') {
      const l2 = this.weights.reduce((sum, w) => sum + w * w, 0);
      return mse + this.lambda * l2;
    } else if (this.regularization === 'l1') {
      const l1 = this.weights.reduce((sum, w) => sum + Math.abs(w), 0);
      return mse + this.lambda * l1;
    }

    return mse;
  }

  score(X, y) {
    const predictions = this.predict(X);
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

  getCoefficients() {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }
    return {
      weights: [...this.weights],
      bias: this.bias,
      equation: this.getEquation()
    };
  }

  getEquation() {
    let eq = `y = ${this.bias.toFixed(4)}`;
    this.weights.forEach((w, i) => {
      const sign = w >= 0 ? '+' : '';
      eq += ` ${sign} ${w.toFixed(4)}*x${i + 1}`;
    });
    return eq;
  }

  summary() {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }

    return {
      modelType: 'Linear Regression',
      coefficients: this.getCoefficients(),
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

export default LinearRegression;