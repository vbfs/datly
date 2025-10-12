import BaseModel from './baseModel.js';
import DecisionTree from './DecisionTree.js';

class RandomForest extends BaseModel {
  constructor(nEstimators = 100, maxDepth = 10, minSamplesSplit = 2, minSamplesLeaf = 1,
              maxFeatures = 'sqrt', criterion = 'gini', bootstrap = true) {
    super();
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.maxFeatures = maxFeatures; // 'sqrt', 'log2', number, or null (all features)
    this.criterion = criterion;
    this.bootstrap = bootstrap;
    this.trees = [];
    this.taskType = null;
    this.classes = null;
    this.featureIndices = [];
  }

  fit(X, y, taskType = 'classification') {
    this.validateTrainingData(X, y);

    this.taskType = taskType;
    const X_train = X.map(row => Array.isArray(row) ? row : [row]);
    const nFeatures = X_train[0].length;

    if (taskType === 'classification') {
      this.classes = [...new Set(y)].sort();
    }

    const maxFeaturesCount = this.getMaxFeaturesCount(nFeatures);

    // Train multiple trees
    for (let i = 0; i < this.nEstimators; i++) {
      // Bootstrap sampling
      const { X_sample, y_sample } = this.bootstrap ?
        this.bootstrapSample(X_train, y) :
        { X_sample: X_train, y_sample: y };

      // Random feature selection
      const featureIndices = this.selectRandomFeatures(nFeatures, maxFeaturesCount);
      this.featureIndices.push(featureIndices);

      // Extract selected features
      const X_subset = X_sample.map(row =>
        featureIndices.map(idx => row[idx])
      );

      // Train tree
      const tree = new DecisionTree(
        this.maxDepth,
        this.minSamplesSplit,
        this.minSamplesLeaf,
        this.criterion
      );
      tree.fit(X_subset, y_sample, taskType);
      this.trees.push(tree);
    }

    this.trained = true;

    this.trainingMetrics = {
      nEstimators: this.nEstimators,
      avgTreeDepth: this.trees.reduce((sum, tree) =>
        sum + tree.trainingMetrics.treeDepth, 0) / this.nEstimators,
      avgLeafCount: this.trees.reduce((sum, tree) =>
        sum + tree.trainingMetrics.leafCount, 0) / this.nEstimators,
      taskType: this.taskType
    };

    return this;
  }

  getMaxFeaturesCount(nFeatures) {
    if (typeof this.maxFeatures === 'number') {
      return Math.min(this.maxFeatures, nFeatures);
    } else if (this.maxFeatures === 'sqrt') {
      return Math.floor(Math.sqrt(nFeatures));
    } else if (this.maxFeatures === 'log2') {
      return Math.floor(Math.log2(nFeatures));
    } else {
      return nFeatures; // null = all features
    }
  }

  selectRandomFeatures(nFeatures, count) {
    const indices = Array.from({ length: nFeatures }, (_, i) => i);
    const selected = [];

    for (let i = 0; i < count; i++) {
      const randomIdx = Math.floor(Math.random() * indices.length);
      selected.push(indices[randomIdx]);
      indices.splice(randomIdx, 1);
    }

    return selected.sort((a, b) => a - b);
  }

  bootstrapSample(X, y) {
    const n = X.length;
    const X_sample = [];
    const y_sample = [];

    for (let i = 0; i < n; i++) {
      const randomIdx = Math.floor(Math.random() * n);
      X_sample.push(X[randomIdx]);
      y_sample.push(y[randomIdx]);
    }

    return { X_sample, y_sample };
  }

  predict(X) {
    this.validatePredictionData(X);

    const X_test = X.map(row => Array.isArray(row) ? row : [row]);

    if (this.taskType === 'classification') {
      return X_test.map(x => {
        const votes = {};

        this.trees.forEach((tree, idx) => {
          const X_subset = this.featureIndices[idx].map(i => x[i]);
          const prediction = tree.predict([X_subset])[0];
          votes[prediction] = (votes[prediction] || 0) + 1;
        });

        return Object.keys(votes).reduce((a, b) =>
          votes[a] > votes[b] ? a : b
        );
      });
    } else {
      return X_test.map(x => {
        const predictions = this.trees.map((tree, idx) => {
          const X_subset = this.featureIndices[idx].map(i => x[i]);
          return tree.predict([X_subset])[0];
        });

        return predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
      });
    }
  }

  predictProba(X) {
    if (this.taskType !== 'classification') {
      throw new Error('predictProba is only available for classification tasks');
    }

    this.validatePredictionData(X);

    const X_test = X.map(row => Array.isArray(row) ? row : [row]);

    return X_test.map(x => {
      const classCounts = {};

      this.classes.forEach(cls => {
        classCounts[cls] = 0;
      });

      this.trees.forEach((tree, idx) => {
        const X_subset = this.featureIndices[idx].map(i => x[i]);
        const proba = tree.predictProba([X_subset])[0];

        Object.keys(proba).forEach(cls => {
          classCounts[cls] += proba[cls];
        });
      });

      const probas = {};
      Object.keys(classCounts).forEach(cls => {
        probas[cls] = classCounts[cls] / this.nEstimators;
      });

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

  getFeatureImportance() {
    const nFeatures = this.featureIndices[0].length;
    const importance = Array(nFeatures).fill(0);

    this.trees.forEach((tree, idx) => {
      const treeImportance = tree.getFeatureImportance();
      const featureMap = this.featureIndices[idx];

      Object.keys(treeImportance).forEach(key => {
        const localIdx = parseInt(key.split('_')[1]);
        const globalIdx = featureMap[localIdx];
        importance[globalIdx] += treeImportance[key];
      });
    });

    const total = importance.reduce((sum, val) => sum + val, 0);
    return importance.map(val => val / total);
  }

  summary() {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }

    return {
      modelType: 'Random Forest',
      taskType: this.taskType,
      trainingMetrics: this.trainingMetrics,
      featureImportance: this.getFeatureImportance(),
      hyperparameters: {
        nEstimators: this.nEstimators,
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        maxFeatures: this.maxFeatures,
        criterion: this.criterion,
        bootstrap: this.bootstrap
      }
    };
  }
}

export default RandomForest;