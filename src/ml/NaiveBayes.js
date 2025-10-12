import BaseModel from './baseModel.js';

class NaiveBayes extends BaseModel {
  constructor(type = 'gaussian') {
    super();
    this.type = type; // 'gaussian', 'multinomial', 'bernoulli'
    this.classes = null;
    this.classPriors = {};
    this.parameters = {};
  }

  fit(X, y) {
    this.validateTrainingData(X, y);

    const X_train = X.map(row => Array.isArray(row) ? row : [row]);
    this.classes = [...new Set(y)].sort();
    const nSamples = X_train.length;
    const nFeatures = X_train[0].length;

    // Calculate class priors
    this.classes.forEach(cls => {
      const count = y.filter(label => label === cls).length;
      this.classPriors[cls] = count / nSamples;
    });

    // Calculate parameters for each class
    if (this.type === 'gaussian') {
      this.fitGaussian(X_train, y, nFeatures);
    } else if (this.type === 'multinomial') {
      this.fitMultinomial(X_train, y, nFeatures);
    } else if (this.type === 'bernoulli') {
      this.fitBernoulli(X_train, y, nFeatures);
    }

    this.trained = true;

    this.trainingMetrics = {
      nClasses: this.classes.length,
      nFeatures: nFeatures,
      nSamples: nSamples,
      type: this.type
    };

    return this;
  }

  fitGaussian(X, y, nFeatures) {
    this.classes.forEach(cls => {
      const classData = X.filter((_, idx) => y[idx] === cls);
      this.parameters[cls] = {
        means: [],
        variances: []
      };

      for (let j = 0; j < nFeatures; j++) {
        const feature = classData.map(row => row[j]);
        const mean = feature.reduce((sum, val) => sum + val, 0) / feature.length;
        const variance = feature.reduce((sum, val) =>
          sum + Math.pow(val - mean, 2), 0) / feature.length;

        this.parameters[cls].means.push(mean);
        this.parameters[cls].variances.push(variance + 1e-9); // Add small value to avoid division by zero
      }
    });
  }

  fitMultinomial(X, y, nFeatures) {
    const alpha = 1.0; // Laplace smoothing

    this.classes.forEach(cls => {
      const classData = X.filter((_, idx) => y[idx] === cls);
      this.parameters[cls] = {
        featureProbs: []
      };

      for (let j = 0; j < nFeatures; j++) {
        const featureSum = classData.reduce((sum, row) => sum + row[j], 0);
        const totalCount = classData.reduce((sum, row) =>
          sum + row.reduce((s, val) => s + val, 0), 0);

        const prob = (featureSum + alpha) / (totalCount + alpha * nFeatures);
        this.parameters[cls].featureProbs.push(prob);
      }
    });
  }

  fitBernoulli(X, y, nFeatures) {
    const alpha = 1.0; // Laplace smoothing

    this.classes.forEach(cls => {
      const classData = X.filter((_, idx) => y[idx] === cls);
      const nClassSamples = classData.length;

      this.parameters[cls] = {
        featureProbs: []
      };

      for (let j = 0; j < nFeatures; j++) {
        const featureCount = classData.filter(row => row[j] === 1).length;
        const prob = (featureCount + alpha) / (nClassSamples + 2 * alpha);
        this.parameters[cls].featureProbs.push(prob);
      }
    });
  }

  gaussianProbability(x, mean, variance) {
    const exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
    return exponent / Math.sqrt(2 * Math.PI * variance);
  }

  predictSingle(x) {
    const posteriors = {};

    this.classes.forEach(cls => {
      let logProb = Math.log(this.classPriors[cls]);

      if (this.type === 'gaussian') {
        const params = this.parameters[cls];
        for (let j = 0; j < x.length; j++) {
          const prob = this.gaussianProbability(x[j], params.means[j], params.variances[j]);
          logProb += Math.log(prob + 1e-9);
        }
      } else if (this.type === 'multinomial') {
        const params = this.parameters[cls];
        for (let j = 0; j < x.length; j++) {
          logProb += x[j] * Math.log(params.featureProbs[j] + 1e-9);
        }
      } else if (this.type === 'bernoulli') {
        const params = this.parameters[cls];
        for (let j = 0; j < x.length; j++) {
          const prob = x[j] === 1 ? params.featureProbs[j] : 1 - params.featureProbs[j];
          logProb += Math.log(prob + 1e-9);
        }
      }

      posteriors[cls] = logProb;
    });

    return Object.keys(posteriors).reduce((a, b) =>
      posteriors[a] > posteriors[b] ? a : b
    );
  }

  predict(X) {
    this.validatePredictionData(X);

    const X_test = X.map(row => Array.isArray(row) ? row : [row]);
    return X_test.map(x => this.predictSingle(x));
  }

  predictProba(X) {
    this.validatePredictionData(X);

    const X_test = X.map(row => Array.isArray(row) ? row : [row]);

    return X_test.map(x => {
      const logPosteriors = {};

      this.classes.forEach(cls => {
        let logProb = Math.log(this.classPriors[cls]);

        if (this.type === 'gaussian') {
          const params = this.parameters[cls];
          for (let j = 0; j < x.length; j++) {
            const prob = this.gaussianProbability(x[j], params.means[j], params.variances[j]);
            logProb += Math.log(prob + 1e-9);
          }
        } else if (this.type === 'multinomial') {
          const params = this.parameters[cls];
          for (let j = 0; j < x.length; j++) {
            logProb += x[j] * Math.log(params.featureProbs[j] + 1e-9);
          }
        } else if (this.type === 'bernoulli') {
          const params = this.parameters[cls];
          for (let j = 0; j < x.length; j++) {
            const prob = x[j] === 1 ? params.featureProbs[j] : 1 - params.featureProbs[j];
            logProb += Math.log(prob + 1e-9);
          }
        }

        logPosteriors[cls] = logProb;
      });

      // Convert log probabilities to probabilities
      const maxLogProb = Math.max(...Object.values(logPosteriors));
      const expProbs = {};
      let sumExpProbs = 0;

      this.classes.forEach(cls => {
        expProbs[cls] = Math.exp(logPosteriors[cls] - maxLogProb);
        sumExpProbs += expProbs[cls];
      });

      const probas = {};
      this.classes.forEach(cls => {
        probas[cls] = expProbs[cls] / sumExpProbs;
      });

      return probas;
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
      modelType: 'Naive Bayes',
      naiveBayesType: this.type,
      classes: this.classes,
      classPriors: this.classPriors,
      trainingMetrics: this.trainingMetrics
    };
  }
}

export default NaiveBayes;