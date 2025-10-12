import BaseModel from './baseModel.js';
import { plot } from 'nodeplotlib'; // 📊 para plotar loss (precisa instalar nodeplotlib)

class LogisticRegression extends BaseModel {
  constructor({
    learningRate = 0.01,
    iterations = 1000,
    batchSize = null,
    regularization = null,
    lambda = 0.01,
    earlyStopping = false,
    tol = 1e-6,
    randomInit = true
  } = {}) {
    super();
    this.learningRate = learningRate;
    this.iterations = iterations;
    this.batchSize = batchSize;
    this.regularization = regularization;
    this.lambda = lambda;
    this.earlyStopping = earlyStopping;
    this.tol = tol;
    this.randomInit = randomInit;
    this.weights = null;
    this.bias = null;
    this.normParams = null;
    this.classes = null;
    this.multiclass = false;
    this.losses = [];
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  softmax(z) {
    const maxZ = Math.max(...z);
    const expZ = z.map(v => Math.exp(v - maxZ));
    const sum = expZ.reduce((a, b) => a + b, 0);
    return expZ.map(v => v / sum);
  }

  fit(X, y, normalize = true) {
    this.validateTrainingData(X, y);

    this.classes = [...new Set(y)].sort((a, b) =>
      typeof a === 'number' && typeof b === 'number'
        ? a - b
        : String(a).localeCompare(String(b))
    );
    this.multiclass = this.classes.length > 2;

    let X_train = X.map(r => (Array.isArray(r) ? r : [r]));
    if (normalize) {
      const { normalized, means, stds } = this.normalizeFeaturesSafe(X_train);
      X_train = normalized;
      this.normParams = { means, stds };
    }

    this.multiclass
      ? this.fitMulticlass(X_train, y)
      : this.fitBinary(X_train, y);

    this.trained = true;
    return this;
  }

  fitBinary(X, y) {
    const n = X.length;
    const m = X[0].length;
    const yBin = y.map(label => (label === this.classes[1] ? 1 : 0));

    this.weights = this.randomInit
      ? Array(m).fill(0).map(() => Math.random() * 0.01)
      : Array(m).fill(0);
    this.bias = 0;

    let prevLoss = Infinity;
    this.losses = [];

    for (let iter = 0; iter < this.iterations; iter++) {
      const { Xb, yb } = this.getBatch(X, yBin);
      const predictions = Xb.map(row => this.sigmoid(this.linear(row)));

      const { weightGradients, biasGradient } = this.gradientBinary(Xb, yb, predictions);

      for (let j = 0; j < m; j++) {
        this.weights[j] -= (this.learningRate / Xb.length) * weightGradients[j];
      }
      this.bias -= (this.learningRate / Xb.length) * biasGradient;

      const loss = this.calculateBinaryLoss(predictions, yb);
      this.losses.push(loss);

      if (this.earlyStopping && Math.abs(prevLoss - loss) < this.tol) break;
      prevLoss = loss;
    }

    this.trainingMetrics = {
      finalLoss: this.losses[this.losses.length - 1],
      losses: this.losses,
      weights: [...this.weights],
      bias: this.bias
    };
  }

  fitMulticlass(X, y) {
    const n = X.length;
    const m = X[0].length;
    const k = this.classes.length;
    const yOneHot = this.oneHotEncode(y, k);

    this.weights = Array(k)
      .fill(0)
      .map(() =>
        this.randomInit
          ? Array(m).fill(0).map(() => Math.random() * 0.01)
          : Array(m).fill(0)
      );
    this.bias = Array(k).fill(0);

    let prevLoss = Infinity;
    this.losses = [];

    for (let iter = 0; iter < this.iterations; iter++) {
      const { Xb, yb } = this.getBatch(X, yOneHot);
      const predictions = Xb.map(row => this.forwardMulticlass(row));

      for (let c = 0; c < k; c++) {
        const grad = this.gradientMulticlass(Xb, yb, predictions, c);
        for (let j = 0; j < m; j++) {
          this.weights[c][j] -= (this.learningRate / Xb.length) * grad.weight[j];
        }
        this.bias[c] -= (this.learningRate / Xb.length) * grad.bias;
      }

      const loss = this.calculateMulticlassLoss(predictions, yb);
      this.losses.push(loss);

      if (this.earlyStopping && Math.abs(prevLoss - loss) < this.tol) break;
      prevLoss = loss;
    }

    this.trainingMetrics = {
      finalLoss: this.losses[this.losses.length - 1],
      losses: this.losses
    };
  }

  predict(X, returnProba = false) {
    this.validatePredictionData(X);
    let X_test = X.map(r => (Array.isArray(r) ? r : [r]));

    if (this.normParams) {
      const { means, stds } = this.normParams;
      X_test = X_test.map(r =>
        r.map((v, j) => (v - means[j]) / (stds[j] || 1))
      );
    }

    return this.multiclass
      ? this.predictMulticlass(X_test, returnProba)
      : this.predictBinary(X_test, returnProba);
  }

  predictBinary(X, returnProba) {
    return X.map(row => {
      const p = this.sigmoid(this.linear(row));
      if (returnProba) {
        return { [this.classes[0]]: 1 - p, [this.classes[1]]: p };
      }
      return p >= 0.5 ? this.classes[1] : this.classes[0];
    });
  }

  predictMulticlass(X, returnProba) {
    return X.map(row => {
      const probs = this.forwardMulticlass(row);
      if (returnProba) {
        const out = {};
        this.classes.forEach((cls, i) => (out[cls] = probs[i]));
        return out;
      }
      const maxIdx = probs.indexOf(Math.max(...probs));
      return this.classes[maxIdx];
    });
  }

  // ---------- Auxiliares ----------
  linear(x) {
    return this.bias + x.reduce((s, v, j) => s + v * this.weights[j], 0);
  }

  forwardMulticlass(x) {
    const z = this.bias.map((b, c) =>
      b + x.reduce((s, v, j) => s + v * this.weights[c][j], 0)
    );
    return this.softmax(z);
  }

  getBatch(X, y) {
    if (!this.batchSize || this.batchSize >= X.length) {
      return { Xb: X, yb: y };
    }
    const idx = Math.floor(Math.random() * (X.length - this.batchSize));
    return {
      Xb: X.slice(idx, idx + this.batchSize),
      yb: y.slice(idx, idx + this.batchSize)
    };
  }

  gradientBinary(X, y, predictions) {
    const m = X[0].length;
    const weightGradients = Array(m).fill(0);
    let biasGradient = 0;

    for (let i = 0; i < X.length; i++) {
      const error = predictions[i] - y[i];
      biasGradient += error;
      for (let j = 0; j < m; j++) {
        weightGradients[j] += error * X[i][j];
      }
    }

    if (this.regularization) {
      for (let j = 0; j < m; j++) {
        if (this.regularization === 'l2') weightGradients[j] += this.lambda * this.weights[j];
        if (this.regularization === 'l1') weightGradients[j] += this.lambda * Math.sign(this.weights[j]);
      }
    }

    return { weightGradients, biasGradient };
  }

  gradientMulticlass(X, y, predictions, c) {
    const m = X[0].length;
    const weightGradients = Array(m).fill(0);
    let biasGradient = 0;

    for (let i = 0; i < X.length; i++) {
      const error = predictions[i][c] - y[i][c];
      biasGradient += error;
      for (let j = 0; j < m; j++) {
        weightGradients[j] += error * X[i][j];
      }
    }

    if (this.regularization === 'l2') {
      for (let j = 0; j < m; j++) {
        weightGradients[j] += this.lambda * this.weights[c][j];
      }
    }

    return { weight: weightGradients, bias: biasGradient };
  }

  normalizeFeaturesSafe(X) {
    const m = X[0].length;
    const means = Array(m).fill(0);
    const stds = Array(m).fill(0);

    for (let j = 0; j < m; j++) {
      const col = X.map(r => r[j]);
      const mean = col.reduce((a, b) => a + b, 0) / col.length;
      const std = Math.sqrt(col.reduce((a, b) => a + (b - mean) ** 2, 0) / col.length);
      means[j] = mean;
      stds[j] = std || 1;
    }

    const normalized = X.map(r => r.map((v, j) => (v - means[j]) / stds[j]));
    return { normalized, means, stds };
  }

  oneHotEncode(y, k) {
    return y.map(label => {
      const arr = Array(k).fill(0);
      arr[this.classes.indexOf(label)] = 1;
      return arr;
    });
  }

  calculateBinaryLoss(predictions, y) {
    const eps = 1e-15;
    let loss = 0;
    for (let i = 0; i < predictions.length; i++) {
      const p = Math.min(Math.max(predictions[i], eps), 1 - eps);
      loss -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
    }
    loss /= predictions.length;

    if (this.regularization === 'l2') {
      const reg = this.weights.reduce((s, w) => s + w * w, 0);
      loss += (this.lambda / 2) * reg;
    }

    return loss;
  }

  calculateMulticlassLoss(predictions, yOneHot) {
    const eps = 1e-15;
    let loss = 0;
    for (let i = 0; i < predictions.length; i++) {
      for (let c = 0; c < yOneHot[i].length; c++) {
        const p = Math.min(Math.max(predictions[i][c], eps), 1 - eps);
        loss -= yOneHot[i][c] * Math.log(p);
      }
    }
    return loss / predictions.length;
  }

  // ---------- 🆕 ROC & AUC ----------
  rocCurve(X, y) {
    if (this.multiclass) {
      console.warn('ROC Curve disponível apenas para problemas binários');
      return null;
    }

    const proba = this.predict(X, true).map(p => p[this.classes[1]]);
    const thresholds = [...new Set(proba)].sort((a, b) => b - a);
    const points = [];

    for (const t of thresholds) {
      let tp = 0, fp = 0, tn = 0, fn = 0;
      for (let i = 0; i < y.length; i++) {
        const actual = y[i] === this.classes[1] ? 1 : 0;
        const pred = proba[i] >= t ? 1 : 0;
        if (actual === 1 && pred === 1) tp++;
        else if (actual === 0 && pred === 1) fp++;
        else if (actual === 0 && pred === 0) tn++;
        else if (actual === 1 && pred === 0) fn++;
      }
      const tpr = tp / (tp + fn);
      const fpr = fp / (fp + tn);
      points.push({ fpr, tpr });
    }

    // Ordena por FPR crescente
    points.sort((a, b) => a.fpr - b.fpr);
    return points;
  }

  aucScore(X, y) {
    const curve = this.rocCurve(X, y);
    if (!curve) return null;

    let auc = 0;
    for (let i = 1; i < curve.length; i++) {
      const x1 = curve[i - 1].fpr;
      const x2 = curve[i].fpr;
      const y1 = curve[i - 1].tpr;
      const y2 = curve[i].tpr;
      auc += (x2 - x1) * (y1 + y2) / 2; // trapezoidal rule
    }
    return Math.abs(auc);
  }

  // ---------- 🆕 Plot da curva de perda ----------
  plotLoss() {
    const x = Array.from({ length: this.losses.length }, (_, i) => i + 1);
    const y = this.losses;
    plot([{ x, y, type: 'line', name: 'Loss' }], {
      title: 'Curva de Perda',
      xaxis: { title: 'Iterações' },
      yaxis: { title: 'Loss' }
    });
  }

  score(X, y) {
    const yPred = this.predict(X);
    const yProba = this.predict(X, true);
    const accuracy = yPred.filter((p, i) => p === y[i]).length / y.length;
    const cm = this.confusionMatrix(y, yPred);
    const metrics = this.calculateClassMetrics(cm);
    const auc = !this.multiclass ? this.aucScore(X, y) : null;
    const roc = !this.multiclass ? this.rocCurve(X, y) : null;

    return {
      accuracy,
      auc,
      roc,
      confusionMatrix: cm,
      classMetrics: metrics,
      predictions: yPred,
      probabilities: yProba
    };
  }

  summary() {
    if (!this.trained) throw new Error('Model must be trained first');

    return {
      modelType: 'Logistic Regression',
      classes: this.classes,
      multiclass: this.multiclass,
      trainingMetrics: this.trainingMetrics,
      hyperparameters: {
        learningRate: this.learningRate,
        iterations: this.iterations,
        regularization: this.regularization,
        lambda: this.lambda,
        batchSize: this.batchSize,
        earlyStopping: this.earlyStopping
      }
    };
  }
}

export default LogisticRegression;
