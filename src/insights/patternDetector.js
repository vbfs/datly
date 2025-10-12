class PatternDetector {
  analyze(dataset) {
    if (!dataset || !dataset.data || !dataset.headers) {
      throw new Error("Invalid dataset format");
    }

    const patterns = {
      trends: this.detectTrends(dataset),
      seasonality: this.detectSeasonality(dataset),
      outliers: this.detectOutliers(dataset),
      correlations: this.detectCorrelationPatterns(dataset),
      distributions: this.detectDistributionPatterns(dataset),
      clustering: this.detectSimpleClusters(dataset),
      temporal: this.detectTemporalPatterns(dataset),
    };

    return {
      timestamp: new Date().toISOString(),
      datasetSize: dataset.length,
      patterns: patterns,
      summary: this.generateSummary(patterns),
      insights: this.generateInsights(patterns),
    };
  }

  detectTrends(dataset) {
    const numericColumns = this.getNumericColumns(dataset);
    const trends = [];

    numericColumns.forEach((column) => {
      const values = this.getColumnValues(dataset, column);
      if (values.length < 5) return;

      const trendData = values.map((value, index) => ({ x: index, y: value }));
      const trendResult = this.calculateTrend(trendData);

      if (Math.abs(trendResult.slope) > 0.01 && trendResult.rSquared > 0.3) {
        trends.push({
          column: column,
          direction: trendResult.slope > 0 ? "increasing" : "decreasing",
          slope: trendResult.slope,
          rSquared: trendResult.rSquared,
          strength: this.classifyTrendStrength(trendResult.rSquared),
          significance: trendResult.pValue < 0.05,
        });
      }
    });

    return trends;
  }

  detectSeasonality(dataset) {
    const dateColumns = this.getDateColumns(dataset);
    const numericColumns = this.getNumericColumns(dataset);
    const seasonality = [];

    if (dateColumns.length === 0) return seasonality;

    dateColumns.forEach((dateCol) => {
      numericColumns.forEach((numCol) => {
        const timeSeries = this.createTimeSeries(dataset, dateCol, numCol);
        if (timeSeries.length < 12) return;

        const seasonalPattern = this.analyzeSeasonalPattern(timeSeries);
        if (seasonalPattern.strength > 0.3) {
          seasonality.push({
            dateColumn: dateCol,
            valueColumn: numCol,
            strength: seasonalPattern.strength,
            period: seasonalPattern.period,
            peaks: seasonalPattern.peaks,
            pattern: seasonalPattern.type,
          });
        }
      });
    });

    return seasonality;
  }

  detectOutliers(dataset) {
    const numericColumns = this.getNumericColumns(dataset);
    const outliers = [];

    numericColumns.forEach((column) => {
      const values = this.getColumnValues(dataset, column);
      if (values.length < 10) return;

      const outlierResults = this.findOutliers(values);
      if (outlierResults.count > 0) {
        outliers.push({
          column: column,
          count: outlierResults.count,
          percentage: outlierResults.percentage,
          method: "IQR",
          outlierValues: outlierResults.values.slice(0, 5),
          severity: this.classifyOutlierSeverity(outlierResults.percentage),
        });
      }
    });

    return outliers;
  }

  detectCorrelationPatterns(dataset) {
    const numericColumns = this.getNumericColumns(dataset);
    if (numericColumns.length < 2)
      return { strongCorrelations: [], clusters: [] };

    const correlationMatrix = this.buildCorrelationMatrix(
      dataset,
      numericColumns
    );
    const strongCorrelations = this.findStrongCorrelations(correlationMatrix);
    const clusters = this.findCorrelationClusters(correlationMatrix);

    return {
      strongCorrelations: strongCorrelations,
      clusters: clusters,
      avgCorrelation: this.calculateAverageCorrelation(correlationMatrix),
    };
  }

  detectDistributionPatterns(dataset) {
    const numericColumns = this.getNumericColumns(dataset);
    const distributions = [];

    numericColumns.forEach((column) => {
      const values = this.getColumnValues(dataset, column);
      if (values.length < 20) return;

      const distInfo = this.analyzeDistribution(values);
      distributions.push({
        column: column,
        type: distInfo.type,
        skewness: distInfo.skewness,
        kurtosis: distInfo.kurtosis,
        isNormal:
          Math.abs(distInfo.skewness) < 1 && Math.abs(distInfo.kurtosis) < 1,
        transformation: this.suggestTransformation(distInfo),
      });
    });

    return distributions;
  }

  detectSimpleClusters(dataset) {
    const numericColumns = this.getNumericColumns(dataset);
    if (numericColumns.length < 2) return [];

    const clusters = [];

    for (let i = 0; i < numericColumns.length; i++) {
      for (let j = i + 1; j < numericColumns.length; j++) {
        const col1 = numericColumns[i];
        const col2 = numericColumns[j];
        const data = this.getTwoColumnData(dataset, col1, col2);

        if (data.length < 10) continue;

        const clusterResult = this.performSimpleKMeans(data, 3);
        if (clusterResult.quality > 0.5) {
          clusters.push({
            variables: [col1, col2],
            clusters: clusterResult.centers,
            quality: clusterResult.quality,
            interpretation: this.interpretClusterQuality(clusterResult.quality),
          });
        }
      }
    }

    return clusters;
  }

  detectTemporalPatterns(dataset) {
    const dateColumns = this.getDateColumns(dataset);
    const patterns = [];

    dateColumns.forEach((dateCol) => {
      const dates = this.getColumnValues(dataset, dateCol)
        .map((val) => new Date(val))
        .filter((date) => !isNaN(date.getTime()))
        .sort((a, b) => a - b);

      if (dates.length < 5) return;

      const intervals = this.calculateIntervals(dates);
      const frequency = this.determineFrequency(intervals);
      const gaps = this.findGaps(dates, frequency);

      patterns.push({
        column: dateCol,
        frequency: frequency,
        totalSpan: dates[dates.length - 1] - dates[0],
        avgInterval:
          intervals.reduce((sum, int) => sum + int, 0) / intervals.length,
        gaps: gaps.length,
        pattern: gaps.length > dates.length * 0.1 ? "irregular" : "regular",
      });
    });

    return patterns;
  }

  getNumericColumns(dataset) {
    return dataset.headers.filter((header) => {
      const values = dataset.data.map((row) => row[header]);
      const numericCount = values.filter(
        (val) => typeof val === "number" && !isNaN(val) && isFinite(val)
      ).length;
      return numericCount > values.length * 0.7;
    });
  }

  getDateColumns(dataset) {
    return dataset.headers.filter((header) => {
      const values = dataset.data.map((row) => row[header]);
      const dateCount = values.filter((val) => {
        if (typeof val === "string") {
          const date = new Date(val);
          return !isNaN(date.getTime());
        }
        return false;
      }).length;
      return dateCount > values.length * 0.7;
    });
  }

  getColumnValues(dataset, column) {
    return dataset.data
      .map((row) => row[column])
      .filter((val) => typeof val === "number" && !isNaN(val) && isFinite(val));
  }

  getTwoColumnData(dataset, col1, col2) {
    return dataset.data
      .map((row) => ({ x: row[col1], y: row[col2] }))
      .filter(
        (point) =>
          typeof point.x === "number" &&
          !isNaN(point.x) &&
          isFinite(point.x) &&
          typeof point.y === "number" &&
          !isNaN(point.y) &&
          isFinite(point.y)
      );
  }

  createTimeSeries(dataset, dateCol, valueCol) {
    return dataset.data
      .map((row) => ({
        date: new Date(row[dateCol]),
        value: row[valueCol],
      }))
      .filter(
        (point) =>
          !isNaN(point.date.getTime()) &&
          typeof point.value === "number" &&
          !isNaN(point.value) &&
          isFinite(point.value)
      )
      .sort((a, b) => a.date - b.date);
  }

  calculateTrend(data) {
    const n = data.length;
    const sumX = data.reduce((sum, point) => sum + point.x, 0);
    const sumY = data.reduce((sum, point) => sum + point.y, 0);
    const sumXY = data.reduce((sum, point) => sum + point.x * point.y, 0);
    const sumXX = data.reduce((sum, point) => sum + point.x * point.x, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const predicted = data.map((point) => intercept + slope * point.x);
    const ssRes = data.reduce(
      (sum, point, i) => sum + Math.pow(point.y - predicted[i], 2),
      0
    );
    const ssTot = data.reduce(
      (sum, point) => sum + Math.pow(point.y - sumY / n, 2),
      0
    );
    const rSquared = 1 - ssRes / ssTot;

    const stderr = Math.sqrt(ssRes / (n - 2));
    const tStat = slope / stderr;
    const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), n - 2));

    return { slope, intercept, rSquared, pValue };
  }

  analyzeSeasonalPattern(timeSeries) {
    const values = timeSeries.map((point) => point.value);
    const n = values.length;

    if (n < 12) return { strength: 0 };

    let bestPeriod = 12;
    let maxCorrelation = 0;

    for (let period = 4; period <= Math.min(n / 3, 24); period++) {
      const correlation = this.calculateAutoCorrelation(values, period);
      if (correlation > maxCorrelation) {
        maxCorrelation = correlation;
        bestPeriod = period;
      }
    }

    const peaks = this.findPeaks(values);
    const valleys = this.findValleys(values);

    return {
      strength: maxCorrelation,
      period: bestPeriod,
      peaks: peaks.length,
      valleys: valleys.length,
      type: this.classifySeasonalType(
        maxCorrelation,
        peaks.length,
        valleys.length
      ),
    };
  }

  findOutliers(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const q1 = this.calculateQuantile(sorted, 0.25);
    const q3 = this.calculateQuantile(sorted, 0.75);
    const iqr = q3 - q1;

    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    const outlierValues = values.filter(
      (val) => val < lowerBound || val > upperBound
    );

    return {
      count: outlierValues.length,
      percentage: (outlierValues.length / values.length) * 100,
      values: outlierValues,
      bounds: { lower: lowerBound, upper: upperBound },
    };
  }

  buildCorrelationMatrix(dataset, columns) {
    const matrix = {};

    columns.forEach((col1) => {
      matrix[col1] = {};
      columns.forEach((col2) => {
        if (col1 === col2) {
          matrix[col1][col2] = 1;
        } else {
          const values1 = this.getColumnValues(dataset, col1);
          const values2 = this.getColumnValues(dataset, col2);
          matrix[col1][col2] = this.calculateCorrelation(values1, values2);
        }
      });
    });

    return matrix;
  }

  findStrongCorrelations(matrix) {
    const correlations = [];
    const columns = Object.keys(matrix);

    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        const col1 = columns[i];
        const col2 = columns[j];
        const correlation = matrix[col1][col2];

        if (Math.abs(correlation) > 0.7) {
          correlations.push({
            variable1: col1,
            variable2: col2,
            correlation: correlation,
            strength: this.getCorrelationStrength(Math.abs(correlation)),
            direction: correlation > 0 ? "positive" : "negative",
          });
        }
      }
    }

    return correlations.sort(
      (a, b) => Math.abs(b.correlation) - Math.abs(a.correlation)
    );
  }

  findCorrelationClusters(matrix) {
    const columns = Object.keys(matrix);
    const clusters = [];
    const visited = new Set();

    columns.forEach((col) => {
      if (visited.has(col)) return;

      const cluster = [col];
      visited.add(col);

      columns.forEach((other) => {
        if (!visited.has(other) && Math.abs(matrix[col][other]) > 0.7) {
          cluster.push(other);
          visited.add(other);
        }
      });

      if (cluster.length > 1) {
        clusters.push(cluster);
      }
    });

    return clusters;
  }

  analyzeDistribution(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance =
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      (values.length - 1);
    const stdDev = Math.sqrt(variance);

    const skewness = this.calculateSkewness(values, mean, stdDev);
    const kurtosis = this.calculateKurtosis(values, mean, stdDev);

    return {
      mean,
      stdDev,
      skewness,
      kurtosis,
      type: this.classifyDistribution(skewness, kurtosis),
    };
  }

  performSimpleKMeans(data, k) {
    let centers = this.initializeCenters(data, k);
    let assignments = new Array(data.length);
    let changed = true;
    let iterations = 0;

    while (changed && iterations < 50) {
      changed = false;

      for (let i = 0; i < data.length; i++) {
        const distances = centers.map((center) =>
          Math.sqrt(
            Math.pow(data[i].x - center.x, 2) +
              Math.pow(data[i].y - center.y, 2)
          )
        );
        const newAssignment = distances.indexOf(Math.min(...distances));

        if (assignments[i] !== newAssignment) {
          changed = true;
          assignments[i] = newAssignment;
        }
      }

      for (let j = 0; j < k; j++) {
        const clusterPoints = data.filter((_, i) => assignments[i] === j);
        if (clusterPoints.length > 0) {
          centers[j] = {
            x:
              clusterPoints.reduce((sum, p) => sum + p.x, 0) /
              clusterPoints.length,
            y:
              clusterPoints.reduce((sum, p) => sum + p.y, 0) /
              clusterPoints.length,
          };
        }
      }

      iterations++;
    }

    const quality = this.calculateClusterQuality(data, assignments, centers);

    return { centers, assignments, quality };
  }

  calculateIntervals(dates) {
    const intervals = [];
    for (let i = 1; i < dates.length; i++) {
      intervals.push(dates[i] - dates[i - 1]);
    }
    return intervals;
  }

  determineFrequency(intervals) {
    if (intervals.length === 0) return "unknown";

    const avgInterval =
      intervals.reduce((sum, int) => sum + int, 0) / intervals.length;
    const dayInMs = 24 * 60 * 60 * 1000;

    if (avgInterval < dayInMs) return "sub_daily";
    if (avgInterval < dayInMs * 2) return "daily";
    if (avgInterval < dayInMs * 8) return "weekly";
    if (avgInterval < dayInMs * 35) return "monthly";
    return "yearly";
  }

  findGaps(dates, expectedFrequency) {
    const expectedInterval = this.getExpectedInterval(expectedFrequency);
    const gaps = [];

    for (let i = 1; i < dates.length; i++) {
      const actualInterval = dates[i] - dates[i - 1];
      if (actualInterval > expectedInterval * 2) {
        gaps.push({
          start: dates[i - 1],
          end: dates[i],
          duration: actualInterval,
        });
      }
    }

    return gaps;
  }

  classifyTrendStrength(rSquared) {
    if (rSquared > 0.8) return "very_strong";
    if (rSquared > 0.6) return "strong";
    if (rSquared > 0.4) return "moderate";
    if (rSquared > 0.2) return "weak";
    return "very_weak";
  }

  classifyOutlierSeverity(percentage) {
    if (percentage > 10) return "severe";
    if (percentage > 5) return "moderate";
    if (percentage > 1) return "mild";
    return "minimal";
  }

  classifySeasonalType(strength, peaks, valleys) {
    if (strength > 0.7) return "strong_seasonal";
    if (strength > 0.5) return "moderate_seasonal";
    if (strength > 0.3) return "weak_seasonal";
    return "no_seasonality";
  }

  classifyDistribution(skewness, kurtosis) {
    if (Math.abs(skewness) < 0.5 && Math.abs(kurtosis) < 0.5) return "normal";
    if (skewness > 1) return "right_skewed";
    if (skewness < -1) return "left_skewed";
    if (kurtosis > 1) return "heavy_tailed";
    return "irregular";
  }

  suggestTransformation(distInfo) {
    if (distInfo.type === "normal") return "none";
    if (distInfo.skewness > 1) return "log_transform";
    if (distInfo.skewness < -1) return "square_transform";
    return "standardization";
  }

  getCorrelationStrength(r) {
    if (r >= 0.9) return "very_strong";
    if (r >= 0.7) return "strong";
    if (r >= 0.5) return "moderate";
    if (r >= 0.3) return "weak";
    return "very_weak";
  }

  interpretClusterQuality(quality) {
    if (quality > 0.7) return "excellent";
    if (quality > 0.5) return "good";
    if (quality > 0.3) return "fair";
    return "poor";
  }

  generateSummary(patterns) {
    const summary = {};

    Object.keys(patterns).forEach((key) => {
      if (Array.isArray(patterns[key])) {
        summary[key] = patterns[key].length;
      } else if (typeof patterns[key] === "object" && patterns[key] !== null) {
        summary[key] = Object.keys(patterns[key]).length;
      } else {
        summary[key] = 0;
      }
    });

    summary.totalPatterns = Object.values(summary).reduce(
      (sum, count) => sum + count,
      0
    );

    return summary;
  }

  generateInsights(patterns) {
    const insights = [];

    if (patterns.trends.length > 0) {
      const strongTrends = patterns.trends.filter(
        (t) => t.strength === "strong" || t.strength === "very_strong"
      );
      if (strongTrends.length > 0) {
        insights.push({
          type: "trend",
          importance: "high",
          message: `Found ${strongTrends.length} strong trend(s) in your data`,
          details: strongTrends.map((t) => `${t.column}: ${t.direction} trend`),
        });
      }
    }

    if (patterns.correlations.strongCorrelations.length > 0) {
      insights.push({
        type: "correlation",
        importance: "medium",
        message: `Discovered ${patterns.correlations.strongCorrelations.length} strong correlation(s)`,
        details: patterns.correlations.strongCorrelations
          .slice(0, 3)
          .map(
            (c) =>
              `${c.variable1} ↔ ${c.variable2}: ${c.strength} ${c.direction}`
          ),
      });
    }

    if (patterns.outliers.length > 0) {
      const severeOutliers = patterns.outliers.filter(
        (o) => o.severity === "severe"
      );
      if (severeOutliers.length > 0) {
        insights.push({
          type: "outliers",
          importance: "high",
          message: `Detected severe outliers in ${severeOutliers.length} column(s)`,
          details: severeOutliers.map(
            (o) =>
              `${o.column}: ${o.count} outliers (${o.percentage.toFixed(1)}%)`
          ),
        });
      }
    }

    if (patterns.seasonality.length > 0) {
      insights.push({
        type: "seasonality",
        importance: "medium",
        message: `Found seasonal patterns in ${patterns.seasonality.length} time series`,
        details: patterns.seasonality.map(
          (s) => `${s.valueColumn}: ${s.pattern} (period: ${s.period})`
        ),
      });
    }

    if (patterns.clustering.length > 0) {
      const goodClusters = patterns.clustering.filter((c) => c.quality > 0.5);
      if (goodClusters.length > 0) {
        insights.push({
          type: "clustering",
          importance: "medium",
          message: `Identified ${goodClusters.length} natural cluster(s) in the data`,
          details: goodClusters.map(
            (c) => `${c.variables.join(" vs ")}: ${c.interpretation} clusters`
          ),
        });
      }
    }

    return insights.sort((a, b) => {
      const importance = { high: 3, medium: 2, low: 1 };
      return importance[b.importance] - importance[a.importance];
    });
  }

  calculateAutoCorrelation(values, lag) {
    if (lag >= values.length) return 0;

    const n = values.length - lag;
    const mean1 = values.slice(0, n).reduce((sum, val) => sum + val, 0) / n;
    const mean2 = values.slice(lag).reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let sum1 = 0;
    let sum2 = 0;

    for (let i = 0; i < n; i++) {
      const diff1 = values[i] - mean1;
      const diff2 = values[i + lag] - mean2;
      numerator += diff1 * diff2;
      sum1 += diff1 * diff1;
      sum2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(sum1 * sum2);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  findPeaks(values) {
    const peaks = [];
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] > values[i - 1] && values[i] > values[i + 1]) {
        peaks.push(i);
      }
    }
    return peaks;
  }

  findValleys(values) {
    const valleys = [];
    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] < values[i - 1] && values[i] < values[i + 1]) {
        valleys.push(i);
      }
    }
    return valleys;
  }

  calculateQuantile(sortedArray, q) {
    const index = (sortedArray.length - 1) * q;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;

    if (lower === upper) {
      return sortedArray[lower];
    }

    return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
  }

  calculateCorrelation(x, y) {
    if (x.length !== y.length || x.length < 3) return 0;

    const n = x.length;
    const meanX = x.reduce((sum, val) => sum + val, 0) / n;
    const meanY = y.reduce((sum, val) => sum + val, 0) / n;

    let numerator = 0;
    let sumXSquared = 0;
    let sumYSquared = 0;

    for (let i = 0; i < n; i++) {
      const xDiff = x[i] - meanX;
      const yDiff = y[i] - meanY;
      numerator += xDiff * yDiff;
      sumXSquared += xDiff * xDiff;
      sumYSquared += yDiff * yDiff;
    }

    const denominator = Math.sqrt(sumXSquared * sumYSquared);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  calculateSkewness(values, mean, stdDev) {
    if (stdDev === 0) return 0;

    const n = values.length;
    const skewSum = values.reduce((sum, val) => {
      return sum + Math.pow((val - mean) / stdDev, 3);
    }, 0);

    return (n / ((n - 1) * (n - 2))) * skewSum;
  }

  calculateKurtosis(values, mean, stdDev) {
    if (stdDev === 0) return 0;

    const n = values.length;
    const kurtSum = values.reduce((sum, val) => {
      return sum + Math.pow((val - mean) / stdDev, 4);
    }, 0);

    return (
      ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * kurtSum -
      (3 * Math.pow(n - 1, 2)) / ((n - 2) * (n - 3))
    );
  }

  initializeCenters(data, k) {
    const centers = [];
    const minX = Math.min(...data.map((p) => p.x));
    const maxX = Math.max(...data.map((p) => p.x));
    const minY = Math.min(...data.map((p) => p.y));
    const maxY = Math.max(...data.map((p) => p.y));

    for (let i = 0; i < k; i++) {
      centers.push({
        x: minX + Math.random() * (maxX - minX),
        y: minY + Math.random() * (maxY - minY),
      });
    }

    return centers;
  }

  calculateClusterQuality(data, assignments, centers) {
    let totalWithinSS = 0;
    let totalBetweenSS = 0;

    const overallCentroid = {
      x: data.reduce((sum, p) => sum + p.x, 0) / data.length,
      y: data.reduce((sum, p) => sum + p.y, 0) / data.length,
    };

    centers.forEach((center, clusterIndex) => {
      const clusterPoints = data.filter(
        (_, i) => assignments[i] === clusterIndex
      );

      clusterPoints.forEach((point) => {
        totalWithinSS +=
          Math.pow(point.x - center.x, 2) + Math.pow(point.y - center.y, 2);
      });

      totalBetweenSS +=
        clusterPoints.length *
        (Math.pow(center.x - overallCentroid.x, 2) +
          Math.pow(center.y - overallCentroid.y, 2));
    });

    const totalSS = totalWithinSS + totalBetweenSS;
    return totalSS > 0 ? totalBetweenSS / totalSS : 0;
  }

  calculateAverageCorrelation(matrix) {
    const columns = Object.keys(matrix);
    let sum = 0;
    let count = 0;

    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        sum += Math.abs(matrix[columns[i]][columns[j]]);
        count++;
      }
    }
  }
  calculateAverageCorrelation(matrix) {
    const columns = Object.keys(matrix);
    let sum = 0;
    let count = 0;

    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        sum += Math.abs(matrix[columns[i]][columns[j]]);
        count++;
      }
    }

    return count > 0 ? sum / count : 0;
  }

  getExpectedInterval(frequency) {
    const dayInMs = 24 * 60 * 60 * 1000;

    switch (frequency) {
      case "daily":
        return dayInMs;
      case "weekly":
        return dayInMs * 7;
      case "monthly":
        return dayInMs * 30;
      case "yearly":
        return dayInMs * 365;
      default:
        return dayInMs;
    }
  }

  tCDF(t, df) {
    if (df <= 0) return 0.5;

    const x = df / (t * t + df);
    return 1 - 0.5 * this.incompleteBeta(df / 2, 0.5, x);
  }

  incompleteBeta(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;

    let result = 0;
    let term = 1;

    for (let n = 0; n < 100; n++) {
      if (n > 0) {
        term *= (x * (a + n - 1)) / n;
      }
      result += term / (a + n);
      if (Math.abs(term) < 1e-10) break;
    }

    return result * Math.pow(x, a) * Math.pow(1 - x, b);
  }
}

export default PatternDetector;
