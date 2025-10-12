class BaseModel {
  constructor() {
    this.trained = false;
    this.model = null;
    this.features = null;
    this.target = null;
    this.trainingMetrics = {};
  }

  validateTrainingData(X, y) {
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error('X must be a non-empty array');
    }
    if (!Array.isArray(y) || y.length === 0) {
      throw new Error('y must be a non-empty array');
    }
    if (X.length !== y.length) {
      throw new Error('X and y must have the same length');
    }
  }

  validatePredictionData(X) {
    if (!this.trained) {
      throw new Error('Model must be trained before making predictions');
    }
    if (!Array.isArray(X) || X.length === 0) {
      throw new Error('X must be a non-empty array');
    }
  }

  normalizeFeatures(X) {
    const n = X.length;
    const m = X[0].length;
    const normalized = [];
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

    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < m; j++) {
        row.push((X[i][j] - means[j]) / stds[j]);
      }
      normalized.push(row);
    }

    return { normalized, means, stds };
  }

  splitTrainTest(X, y, testSize = 0.2, shuffle = true) {
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

  save() {
    if (!this.trained) {
      throw new Error('Cannot save untrained model');
    }
    return {
      model: this.model,
      features: this.features,
      target: this.target,
      trainingMetrics: this.trainingMetrics,
      timestamp: new Date().toISOString()
    };
  }

  load(modelData) {
    this.model = modelData.model;
    this.features = modelData.features;
    this.target = modelData.target;
    this.trainingMetrics = modelData.trainingMetrics;
    this.trained = true;
  }
}

export default BaseModel;