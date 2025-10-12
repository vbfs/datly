class ClassificationMetrics {
  confusionMatrix(yTrue, yPred) {
    const classes = [...new Set([...yTrue, ...yPred])].sort();
    const n = classes.length;
    const matrix = Array(n).fill(0).map(() => Array(n).fill(0));
    const classIndex = new Map(classes.map((c, i) => [c, i]));

    for (let i = 0; i < yTrue.length; i++) {
      const trueIdx = classIndex.get(yTrue[i]);
      const predIdx = classIndex.get(yPred[i]);
      matrix[trueIdx][predIdx]++;
    }

    return {
      matrix,
      classes,
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

  accuracy(yTrue, yPred) {
    let correct = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === yPred[i]) correct++;
    }
    return correct / yTrue.length;
  }

  precision(yTrue, yPred, average = 'weighted') {
    const cm = this.confusionMatrix(yTrue, yPred);
    const classes = cm.classes;
    const matrix = cm.matrix;
    const precisions = [];

    for (let i = 0; i < classes.length; i++) {
      const tp = matrix[i][i];
      const fp = matrix.map((row, idx) => idx !== i ? row[i] : 0).reduce((a, b) => a + b, 0);
      precisions.push(tp + fp > 0 ? tp / (tp + fp) : 0);
    }

    return this.averageMetric(precisions, yTrue, classes, average);
  }

  recall(yTrue, yPred, average = 'weighted') {
    const cm = this.confusionMatrix(yTrue, yPred);
    const classes = cm.classes;
    const matrix = cm.matrix;
    const recalls = [];

    for (let i = 0; i < classes.length; i++) {
      const tp = matrix[i][i];
      const fn = matrix[i].reduce((a, b) => a + b, 0) - tp;
      recalls.push(tp + fn > 0 ? tp / (tp + fn) : 0);
    }

    return this.averageMetric(recalls, yTrue, classes, average);
  }

  f1Score(yTrue, yPred, average = 'weighted') {
    const cm = this.confusionMatrix(yTrue, yPred);
    const classes = cm.classes;
    const matrix = cm.matrix;
    const f1Scores = [];

    for (let i = 0; i < classes.length; i++) {
      const tp = matrix[i][i];
      const fp = matrix.map((row, idx) => idx !== i ? row[i] : 0).reduce((a, b) => a + b, 0);
      const fn = matrix[i].reduce((a, b) => a + b, 0) - tp;

      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

      f1Scores.push(f1);
    }

    return this.averageMetric(f1Scores, yTrue, classes, average);
  }

  averageMetric(metrics, yTrue, classes, average) {
    if (average === 'macro') {
      return metrics.reduce((sum, m) => sum + m, 0) / metrics.length;
    } else if (average === 'weighted') {
      const classCounts = classes.map(cls =>
        yTrue.filter(y => y === cls).length
      );
      const total = yTrue.length;

      let weightedSum = 0;
      for (let i = 0; i < metrics.length; i++) {
        weightedSum += metrics[i] * (classCounts[i] / total);
      }
      return weightedSum;
    } else if (average === 'micro') {
      const cm = this.confusionMatrix(yTrue, yTrue);
      const matrix = cm.matrix;

      let totalTp = 0;
      let totalFp = 0;
      let totalFn = 0;

      for (let i = 0; i < classes.length; i++) {
        const tp = matrix[i][i];
        const fp = matrix.map((row, idx) => idx !== i ? row[i] : 0).reduce((a, b) => a + b, 0);
        const fn = matrix[i].reduce((a, b) => a + b, 0) - tp;

        totalTp += tp;
        totalFp += fp;
        totalFn += fn;
      }

      return totalTp / (totalTp + totalFp);
    } else if (average === null || average === 'none') {
      return metrics;
    }

    throw new Error('Unknown average method. Use: macro, weighted, micro, or null');
  }

  classificationReport(yTrue, yPred) {
    const cm = this.confusionMatrix(yTrue, yPred);
    const classes = cm.classes;
    const matrix = cm.matrix;

    const report = {
      classes: {},
      accuracy: this.accuracy(yTrue, yPred),
      macroAvg: {},
      weightedAvg: {}
    };

    // Per-class metrics
    for (let i = 0; i < classes.length; i++) {
      const tp = matrix[i][i];
      const fp = matrix.map((row, idx) => idx !== i ? row[i] : 0).reduce((a, b) => a + b, 0);
      const fn = matrix[i].reduce((a, b) => a + b, 0) - tp;
      const support = tp + fn;

      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

      report.classes[classes[i]] = {
        precision: precision,
        recall: recall,
        f1Score: f1,
        support: support
      };
    }

    // Macro average
    report.macroAvg = {
      precision: this.precision(yTrue, yPred, 'macro'),
      recall: this.recall(yTrue, yPred, 'macro'),
      f1Score: this.f1Score(yTrue, yPred, 'macro'),
      support: yTrue.length
    };

    // Weighted average
    report.weightedAvg = {
      precision: this.precision(yTrue, yPred, 'weighted'),
      recall: this.recall(yTrue, yPred, 'weighted'),
      f1Score: this.f1Score(yTrue, yPred, 'weighted'),
      support: yTrue.length
    };

    return report;
  }

  formatClassificationReport(yTrue, yPred) {
    const report = this.classificationReport(yTrue, yPred);
    const classes = Object.keys(report.classes);

    let output = '\n' + '='.repeat(70) + '\n';
    output += 'CLASSIFICATION REPORT\n';
    output += '='.repeat(70) + '\n\n';

    output += '           Precision    Recall    F1-Score    Support\n';
    output += '-'.repeat(70) + '\n';

    classes.forEach(cls => {
      const metrics = report.classes[cls];
      output += `${cls.toString().padEnd(10)} `;
      output += `${metrics.precision.toFixed(4).padStart(9)} `;
      output += `${metrics.recall.toFixed(4).padStart(9)} `;
      output += `${metrics.f1Score.toFixed(4).padStart(9)} `;
      output += `${metrics.support.toString().padStart(10)}\n`;
    });

    output += '-'.repeat(70) + '\n';
    output += `accuracy   ${' '.repeat(28)} ${report.accuracy.toFixed(4).padStart(9)} `;
    output += `${yTrue.length.toString().padStart(10)}\n`;

    output += `macro avg  `;
    output += `${report.macroAvg.precision.toFixed(4).padStart(9)} `;
    output += `${report.macroAvg.recall.toFixed(4).padStart(9)} `;
    output += `${report.macroAvg.f1Score.toFixed(4).padStart(9)} `;
    output += `${report.macroAvg.support.toString().padStart(10)}\n`;

    output += `weighted avg `;
    output += `${report.weightedAvg.precision.toFixed(4).padStart(7)} `;
    output += `${report.weightedAvg.recall.toFixed(4).padStart(9)} `;
    output += `${report.weightedAvg.f1Score.toFixed(4).padStart(9)} `;
    output += `${report.weightedAvg.support.toString().padStart(10)}\n`;

    output += '='.repeat(70) + '\n';

    return output;
  }

  matthewsCorrCoef(yTrue, yPred) {
    const cm = this.confusionMatrix(yTrue, yPred);

    if (cm.classes.length !== 2) {
      throw new Error('Matthews Correlation Coefficient only works for binary classification');
    }

    const matrix = cm.matrix;
    const tp = matrix[0][0];
    const tn = matrix[1][1];
    const fp = matrix[0][1];
    const fn = matrix[1][0];

    const numerator = (tp * tn) - (fp * fn);
    const denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  cohenKappa(yTrue, yPred) {
    const cm = this.confusionMatrix(yTrue, yPred);
    const matrix = cm.matrix;
    const n = yTrue.length;

    // Observed agreement
    let po = 0;
    for (let i = 0; i < matrix.length; i++) {
      po += matrix[i][i];
    }
    po /= n;

    // Expected agreement
    let pe = 0;
    for (let i = 0; i < matrix.length; i++) {
      const rowSum = matrix[i].reduce((a, b) => a + b, 0);
      const colSum = matrix.reduce((sum, row) => sum + row[i], 0);
      pe += (rowSum * colSum) / (n * n);
    }

    return (po - pe) / (1 - pe);
  }

  specificity(yTrue, yPred, positiveClass = null) {
    const cm = this.confusionMatrix(yTrue, yPred);
    const classes = cm.classes;
    const matrix = cm.matrix;

    if (positiveClass === null) {
      positiveClass = classes[0];
    }

    const posIdx = classes.indexOf(positiveClass);
    if (posIdx === -1) {
      throw new Error(`Positive class ${positiveClass} not found in data`);
    }

    const tn = matrix.reduce((sum, row, i) => {
      return sum + row.reduce((s, val, j) => {
        return s + (i !== posIdx && j !== posIdx ? val : 0);
      }, 0);
    }, 0);

    const fp = matrix.reduce((sum, row, i) => {
      return sum + (i !== posIdx ? row[posIdx] : 0);
    }, 0);

    return tn + fp > 0 ? tn / (tn + fp) : 0;
  }

  sensitivity(yTrue, yPred, positiveClass = null) {
    // Sensitivity is the same as recall
    if (positiveClass === null) {
      return this.recall(yTrue, yPred, 'macro');
    }

    const cm = this.confusionMatrix(yTrue, yPred);
    const classes = cm.classes;
    const matrix = cm.matrix;
    const posIdx = classes.indexOf(positiveClass);

    if (posIdx === -1) {
      throw new Error(`Positive class ${positiveClass} not found in data`);
    }

    const tp = matrix[posIdx][posIdx];
    const fn = matrix[posIdx].reduce((a, b) => a + b, 0) - tp;

    return tp + fn > 0 ? tp / (tp + fn) : 0;
  }

  balancedAccuracy(yTrue, yPred) {
    const cm = this.confusionMatrix(yTrue, yPred);
    const classes = cm.classes;
    const matrix = cm.matrix;

    let sensitivities = [];

    for (let i = 0; i < classes.length; i++) {
      const tp = matrix[i][i];
      const fn = matrix[i].reduce((a, b) => a + b, 0) - tp;
      const sensitivity = tp + fn > 0 ? tp / (tp + fn) : 0;
      sensitivities.push(sensitivity);
    }

    return sensitivities.reduce((sum, s) => sum + s, 0) / sensitivities.length;
  }
}

export default ClassificationMetrics;