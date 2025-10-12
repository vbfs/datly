class Regression {
    linear(x, y) {
        if (!Array.isArray(x) || !Array.isArray(y)) {
            throw new Error('Both inputs must be arrays');
        }

        if (x.length !== y.length) {
            throw new Error('Arrays must have the same length');
        }

        const validPairs = [];
        for (let i = 0; i < x.length; i++) {
            if (typeof x[i] === 'number' && typeof y[i] === 'number' &&
                !isNaN(x[i]) && !isNaN(y[i]) && isFinite(x[i]) && isFinite(y[i])) {
                validPairs.push({ x: x[i], y: y[i] });
            }
        }

        if (validPairs.length < 3) {
            throw new Error('Need at least 3 valid paired observations');
        }

        const n = validPairs.length;
        const xValues = validPairs.map(pair => pair.x);
        const yValues = validPairs.map(pair => pair.y);

        const meanX = xValues.reduce((sum, val) => sum + val, 0) / n;
        const meanY = yValues.reduce((sum, val) => sum + val, 0) / n;

        let numerator = 0;
        let denominator = 0;

        for (let i = 0; i < n; i++) {
            const xDiff = xValues[i] - meanX;
            const yDiff = yValues[i] - meanY;
            numerator += xDiff * yDiff;
            denominator += xDiff * xDiff;
        }

        if (denominator === 0) {
            throw new Error('Cannot perform regression: X values have zero variance');
        }

        const slope = numerator / denominator;
        const intercept = meanY - slope * meanX;

        const predicted = xValues.map(x => intercept + slope * x);
        const residuals = yValues.map((y, i) => y - predicted[i]);

        const ssResidual = residuals.reduce((sum, r) => sum + r * r, 0);
        const ssTotal = yValues.reduce((sum, y) => sum + Math.pow(y - meanY, 2), 0);
        const ssRegression = ssTotal - ssResidual;

        const rSquared = ssTotal === 0 ? 1 : ssRegression / ssTotal;
        const adjustedRSquared = 1 - ((ssResidual / (n - 2)) / (ssTotal / (n - 1)));

        const mse = ssResidual / (n - 2);
        const rmse = Math.sqrt(mse);
        const standardErrorSlope = Math.sqrt(mse / denominator);
        const standardErrorIntercept = Math.sqrt(mse * (1/n + (meanX * meanX) / denominator));

        const tStatSlope = slope / standardErrorSlope;
        const tStatIntercept = intercept / standardErrorIntercept;
        const df = n - 2;

        const pValueSlope = 2 * (1 - this.tCDF(Math.abs(tStatSlope), df));
        const pValueIntercept = 2 * (1 - this.tCDF(Math.abs(tStatIntercept), df));

        const fStatistic = (ssRegression / 1) / (ssResidual / df);
        const pValueModel = 1 - this.fCDF(fStatistic, 1, df);

        return {
            slope: slope,
            intercept: intercept,
            rSquared: rSquared,
            adjustedRSquared: adjustedRSquared,
            correlation: Math.sqrt(rSquared) * Math.sign(slope),
            standardErrorSlope: standardErrorSlope,
            standardErrorIntercept: standardErrorIntercept,
            tStatSlope: tStatSlope,
            tStatIntercept: tStatIntercept,
            pValueSlope: pValueSlope,
            pValueIntercept: pValueIntercept,
            fStatistic: fStatistic,
            pValueModel: pValueModel,
            degreesOfFreedom: df,
            mse: mse,
            rmse: rmse,
            residuals: residuals,
            predicted: predicted,
            sampleSize: n,
            equation: `y = ${intercept.toFixed(4)} + ${slope.toFixed(4)}x`,
            residualAnalysis: this.analyzeResiduals(residuals, predicted)
        };
    }

    multiple(dataset, dependentVariable, independentVariables) {
        if (!dataset || !dataset.data || !Array.isArray(dataset.data)) {
            throw new Error('Invalid dataset format');
        }

        if (!dataset.headers.includes(dependentVariable)) {
            throw new Error(`Dependent variable '${dependentVariable}' not found in dataset`);
        }

        const missingVars = independentVariables.filter(var_ => !dataset.headers.includes(var_));
        if (missingVars.length > 0) {
            throw new Error(`Independent variables not found: ${missingVars.join(', ')}`);
        }

        const validRows = dataset.data.filter(row => {
            return [dependentVariable, ...independentVariables].every(variable => {
                const value = row[variable];
                return typeof value === 'number' && !isNaN(value) && isFinite(value);
            });
        });

        if (validRows.length < independentVariables.length + 2) {
            throw new Error(`Need at least ${independentVariables.length + 2} valid observations`);
        }

        const n = validRows.length;
        const k = independentVariables.length;

        const y = validRows.map(row => row[dependentVariable]);
        const X = validRows.map(row => [1, ...independentVariables.map(var_ => row[var_])]);

        const XTranspose = this.transpose(X);
        const XTX = this.matrixMultiply(XTranspose, X);
        const XTXInverse = this.matrixInverse(XTX);
        const XTY = this.matrixVectorMultiply(XTranspose, y);
        const coefficients = this.matrixVectorMultiply(XTXInverse, XTY);

        const predicted = X.map(row =>
            coefficients.reduce((sum, coef, i) => sum + coef * row[i], 0)
        );

        const residuals = y.map((actual, i) => actual - predicted[i]);
        const meanY = y.reduce((sum, val) => sum + val, 0) / n;

        const ssResidual = residuals.reduce((sum, r) => sum + r * r, 0);
        const ssTotal = y.reduce((sum, val) => sum + Math.pow(val - meanY, 2), 0);
        const ssRegression = ssTotal - ssResidual;

        const rSquared = ssTotal === 0 ? 1 : ssRegression / ssTotal;
        const adjustedRSquared = 1 - ((ssResidual / (n - k - 1)) / (ssTotal / (n - 1)));

        const mse = ssResidual / (n - k - 1);
        const rmse = Math.sqrt(mse);

        const standardErrors = coefficients.map((_, i) => Math.sqrt(mse * XTXInverse[i][i]));
        const tStats = coefficients.map((coef, i) => coef / standardErrors[i]);
        const pValues = tStats.map(t => 2 * (1 - this.tCDF(Math.abs(t), n - k - 1)));

        const fStatistic = (ssRegression / k) / (ssResidual / (n - k - 1));
        const pValueModel = 1 - this.fCDF(fStatistic, k, n - k - 1);

        const coefficientData = coefficients.map((coef, i) => ({
            variable: i === 0 ? 'Intercept' : independentVariables[i - 1],
            coefficient: coef,
            standardError: standardErrors[i],
            tStatistic: tStats[i],
            pValue: pValues[i],
            significant: pValues[i] < 0.05
        }));

        return {
            coefficients: coefficientData,
            intercept: coefficients[0],
            rSquared: rSquared,
            adjustedRSquared: adjustedRSquared,
            fStatistic: fStatistic,
            pValueModel: pValueModel,
            mse: mse,
            rmse: rmse,
            residuals: residuals,
            predicted: predicted,
            sampleSize: n,
            degreesOfFreedom: n - k - 1,
            dependentVariable: dependentVariable,
            independentVariables: independentVariables,
            equation: this.buildEquation(coefficientData),
            residualAnalysis: this.analyzeResiduals(residuals, predicted)
        };
    }

    polynomial(x, y, degree = 2) {
        if (!Array.isArray(x) || !Array.isArray(y)) {
            throw new Error('Both inputs must be arrays');
        }

        if (x.length !== y.length) {
            throw new Error('Arrays must have the same length');
        }

        if (degree < 1 || degree > 10) {
            throw new Error('Degree must be between 1 and 10');
        }

        const validPairs = [];
        for (let i = 0; i < x.length; i++) {
            if (typeof x[i] === 'number' && typeof y[i] === 'number' &&
                !isNaN(x[i]) && !isNaN(y[i]) && isFinite(x[i]) && isFinite(y[i])) {
                validPairs.push({ x: x[i], y: y[i] });
            }
        }

        const n = validPairs.length;
        if (n < degree + 2) {
            throw new Error(`Need at least ${degree + 2} valid observations for degree ${degree} polynomial`);
        }

        const xValues = validPairs.map(pair => pair.x);
        const yValues = validPairs.map(pair => pair.y);

        const X = xValues.map(x => {
            const row = [1];
            for (let i = 1; i <= degree; i++) {
                row.push(Math.pow(x, i));
            }
            return row;
        });

        const XTranspose = this.transpose(X);
        const XTX = this.matrixMultiply(XTranspose, X);
        const XTXInverse = this.matrixInverse(XTX);
        const XTY = this.matrixVectorMultiply(XTranspose, yValues);
        const coefficients = this.matrixVectorMultiply(XTXInverse, XTY);

        const predicted = X.map(row =>
            coefficients.reduce((sum, coef, i) => sum + coef * row[i], 0)
        );

        const residuals = yValues.map((actual, i) => actual - predicted[i]);
        const meanY = yValues.reduce((sum, val) => sum + val, 0) / n;

        const ssResidual = residuals.reduce((sum, r) => sum + r * r, 0);
        const ssTotal = yValues.reduce((sum, val) => sum + Math.pow(val - meanY, 2), 0);
        const ssRegression = ssTotal - ssResidual;

        const rSquared = ssTotal === 0 ? 1 : ssRegression / ssTotal;
        const adjustedRSquared = 1 - ((ssResidual / (n - degree - 1)) / (ssTotal / (n - 1)));

        const mse = ssResidual / (n - degree - 1);
        const rmse = Math.sqrt(mse);

        const standardErrors = coefficients.map((_, i) => Math.sqrt(mse * XTXInverse[i][i]));
        const tStats = coefficients.map((coef, i) => coef / standardErrors[i]);
        const pValues = tStats.map(t => 2 * (1 - this.tCDF(Math.abs(t), n - degree - 1)));

        return {
            coefficients: coefficients,
            degree: degree,
            rSquared: rSquared,
            adjustedRSquared: adjustedRSquared,
            mse: mse,
            rmse: rmse,
            residuals: residuals,
            predicted: predicted,
            sampleSize: n,
            equation: this.buildPolynomialEquation(coefficients),
            residualAnalysis: this.analyzeResiduals(residuals, predicted),
            standardErrors: standardErrors,
            tStatistics: tStats,
            pValues: pValues
        };
    }

    logistic(x, y, maxIterations = 100, tolerance = 1e-6) {
        if (!Array.isArray(x) || !Array.isArray(y)) {
            throw new Error('Both inputs must be arrays');
        }

        const validPairs = [];
        for (let i = 0; i < x.length; i++) {
            if (typeof x[i] === 'number' && typeof y[i] === 'number' &&
                !isNaN(x[i]) && !isNaN(y[i]) && isFinite(x[i]) && isFinite(y[i]) &&
                (y[i] === 0 || y[i] === 1)) {
                validPairs.push({ x: x[i], y: y[i] });
            }
        }

        if (validPairs.length < 10) {
            throw new Error('Need at least 10 valid observations for logistic regression');
        }

        const xValues = validPairs.map(pair => pair.x);
        const yValues = validPairs.map(pair => pair.y);
        const n = validPairs.length;

        let beta0 = 0;
        let beta1 = 0;

        for (let iter = 0; iter < maxIterations; iter++) {
            const probabilities = xValues.map(x => this.sigmoid(beta0 + beta1 * x));
            const weights = probabilities.map(p => p * (1 - p));

            let score0 = 0, score1 = 0;
            let info00 = 0, info01 = 0, info11 = 0;

            for (let i = 0; i < n; i++) {
                const residual = yValues[i] - probabilities[i];
                score0 += residual;
                score1 += residual * xValues[i];

                info00 += weights[i];
                info01 += weights[i] * xValues[i];
                info11 += weights[i] * xValues[i] * xValues[i];
            }

            const determinant = info00 * info11 - info01 * info01;
            if (Math.abs(determinant) < 1e-10) {
                throw new Error('Information matrix is singular');
            }

            const delta0 = (info11 * score0 - info01 * score1) / determinant;
            const delta1 = (info00 * score1 - info01 * score0) / determinant;

            beta0 += delta0;
            beta1 += delta1;

            if (Math.abs(delta0) < tolerance && Math.abs(delta1) < tolerance) {
                break;
            }
        }

        const finalProbabilities = xValues.map(x => this.sigmoid(beta0 + beta1 * x));
        const predicted = finalProbabilities.map(p => p >= 0.5 ? 1 : 0);

        const logLikelihood = yValues.reduce((sum, y, i) => {
            const p = finalProbabilities[i];
            return sum + y * Math.log(p + 1e-15) + (1 - y) * Math.log(1 - p + 1e-15);
        }, 0);

        const nullLogLikelihood = this.calculateNullLogLikelihood(yValues);
        const mcFaddenR2 = 1 - (logLikelihood / nullLogLikelihood);

        const accuracy = predicted.reduce((sum, pred, i) => sum + (pred === yValues[i] ? 1 : 0), 0) / n;

        return {
            intercept: beta0,
            slope: beta1,
            probabilities: finalProbabilities,
            predicted: predicted,
            logLikelihood: logLikelihood,
            mcFaddenR2: mcFaddenR2,
            accuracy: accuracy,
            sampleSize: n,
            equation: `p = 1 / (1 + exp(-(${beta0.toFixed(4)} + ${beta1.toFixed(4)}x)))`,
            confusionMatrix: this.calculateConfusionMatrix(yValues, predicted)
        };
    }

    predict(model, newX) {
        if (!model || typeof model !== 'object') {
            throw new Error('Invalid model object');
        }

        if (model.coefficients && Array.isArray(model.coefficients)) {
            if (Array.isArray(newX[0])) {
                return newX.map(row => {
                    const extendedRow = [1, ...row];
                    return model.coefficients.reduce((sum, coef, i) => sum + coef.coefficient * extendedRow[i], 0);
                });
            } else {
                const extendedRow = [1, ...newX];
                return model.coefficients.reduce((sum, coef, i) => sum + coef.coefficient * extendedRow[i], 0);
            }
        } else if (model.slope !== undefined && model.intercept !== undefined) {
            if (Array.isArray(newX)) {
                return newX.map(x => model.intercept + model.slope * x);
            } else {
                return model.intercept + model.slope * newX;
            }
        } else if (model.coefficients && model.degree !== undefined) {
            if (Array.isArray(newX)) {
                return newX.map(x => {
                    let result = model.coefficients[0];
                    for (let i = 1; i <= model.degree; i++) {
                        result += model.coefficients[i] * Math.pow(x, i);
                    }
                    return result;
                });
            } else {
                let result = model.coefficients[0];
                for (let i = 1; i <= model.degree; i++) {
                    result += model.coefficients[i] * Math.pow(newX, i);
                }
                return result;
            }
        } else {
            throw new Error('Unknown model type');
        }
    }

    analyzeResiduals(residuals, predicted) {
        const n = residuals.length;
        const meanResidual = residuals.reduce((sum, r) => sum + r, 0) / n;
        const stdResidual = Math.sqrt(residuals.reduce((sum, r) => sum + Math.pow(r - meanResidual, 2), 0) / (n - 1));

        const standardizedResiduals = residuals.map(r => r / stdResidual);
        const outliers = standardizedResiduals.map((sr, i) => ({ index: i, value: sr }))
                                            .filter(item => Math.abs(item.value) > 2);

        const durbinWatson = this.calculateDurbinWatson(residuals);

        return {
            mean: meanResidual,
            standardDeviation: stdResidual,
            standardizedResiduals: standardizedResiduals,
            outliers: outliers,
            durbinWatson: durbinWatson,
            normalityTest: this.testResidualNormality(residuals)
        };
    }

    calculateDurbinWatson(residuals) {
        let numerator = 0;
        let denominator = 0;

        for (let i = 1; i < residuals.length; i++) {
            numerator += Math.pow(residuals[i] - residuals[i - 1], 2);
        }

        for (let i = 0; i < residuals.length; i++) {
            denominator += Math.pow(residuals[i], 2);
        }

        return numerator / denominator;
    }

    testResidualNormality(residuals) {
        const n = residuals.length;
        const mean = residuals.reduce((sum, r) => sum + r, 0) / n;
        const variance = residuals.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
            return { isNormal: true, pValue: 1 };
        }

        const skewness = residuals.reduce((sum, r) => sum + Math.pow((r - mean) / stdDev, 3), 0) / n;
        const kurtosis = residuals.reduce((sum, r) => sum + Math.pow((r - mean) / stdDev, 4), 0) / n - 3;

        const jarqueBera = (n / 6) * (Math.pow(skewness, 2) + Math.pow(kurtosis, 2) / 4);
        const pValue = 1 - this.chiSquareCDF(jarqueBera, 2);

        return {
            jarqueBeraStatistic: jarqueBera,
            pValue: pValue,
            isNormal: pValue > 0.05,
            skewness: skewness,
            kurtosis: kurtosis
        };
    }

    buildEquation(coefficientData) {
        const terms = coefficientData.map(coef => {
            if (coef.variable === 'Intercept') {
                return coef.coefficient.toFixed(4);
            } else {
                const sign = coef.coefficient >= 0 ? '+' : '';
                return `${sign}${coef.coefficient.toFixed(4)}*${coef.variable}`;
            }
        });

        return `y = ${terms.join(' ')}`;
    }

    buildPolynomialEquation(coefficients) {
        const terms = coefficients.map((coef, i) => {
            if (i === 0) {
                return coef.toFixed(4);
            } else if (i === 1) {
                const sign = coef >= 0 ? '+' : '';
                return `${sign}${coef.toFixed(4)}*x`;
            } else {
                const sign = coef >= 0 ? '+' : '';
                return `${sign}${coef.toFixed(4)}*x^${i}`;
            }
        });

        return `y = ${terms.join(' ')}`;
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
    }

    calculateNullLogLikelihood(y) {
        const p = y.reduce((sum, val) => sum + val, 0) / y.length;
        return y.reduce((sum, val) => {
            return sum + val * Math.log(p + 1e-15) + (1 - val) * Math.log(1 - p + 1e-15);
        }, 0);
    }

    calculateConfusionMatrix(actual, predicted) {
        let tp = 0, fp = 0, tn = 0, fn = 0;

        for (let i = 0; i < actual.length; i++) {
            if (actual[i] === 1 && predicted[i] === 1) tp++;
            else if (actual[i] === 0 && predicted[i] === 1) fp++;
            else if (actual[i] === 0 && predicted[i] === 0) tn++;
            else if (actual[i] === 1 && predicted[i] === 0) fn++;
        }

        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
        const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
        const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

        return {
            truePositive: tp,
            falsePositive: fp,
            trueNegative: tn,
            falseNegative: fn,
            precision: precision,
            recall: recall,
            specificity: specificity,
            f1Score: f1Score
        };
    }

    matrixMultiply(A, B) {
        const rowsA = A.length;
        const colsA = A[0].length;
        const colsB = B[0].length;

        const result = Array(rowsA).fill().map(() => Array(colsB).fill(0));

        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                for (let k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    matrixVectorMultiply(A, b) {
        return A.map(row => row.reduce((sum, val, i) => sum + val * b[i], 0));
    }

    transpose(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    matrixInverse(matrix) {
        const n = matrix.length;
        const identity = Array(n).fill().map((_, i) => Array(n).fill().map((_, j) => i === j ? 1 : 0));
        const augmented = matrix.map((row, i) => [...row, ...identity[i]]);

        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];

            const pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) {
                throw new Error('Matrix is singular and cannot be inverted');
            }

            for (let j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }

            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = augmented[k][i];
                    for (let j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        return augmented.map(row => row.slice(n));
    }

    tCDF(t, df) {
        if (df <= 0) return 0.5;

        const x = df / (t * t + df);
        return 1 - 0.5 * this.incompleteBeta(df / 2, 0.5, x);
    }

    fCDF(f, df1, df2) {
        if (f <= 0) return 0;

        const x = df2 / (df2 + df1 * f);
        return 1 - this.incompleteBeta(df2 / 2, df1 / 2, x);
    }

    chiSquareCDF(x, df) {
        if (x <= 0) return 0;
        return this.incompleteGamma(df / 2, x / 2) / this.gamma(df / 2);
    }

    incompleteBeta(a, b, x) {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

        let result = 0;
        let term = 1;

        for (let n = 0; n < 100; n++) {
            if (n > 0) {
                term *= x * (a + n - 1) / n;
            }
            result += term / (a + n);
            if (Math.abs(term) < 1e-10) break;
        }

        return result * Math.pow(x, a) * Math.pow(1 - x, b);
    }

    incompleteGamma(a, x) {
        if (x <= 0) return 0;

        let sum = 1;
        let term = 1;

        for (let n = 1; n < 100; n++) {
            term *= x / (a + n - 1);
            sum += term;
            if (Math.abs(term) < 1e-12) break;
        }

        return Math.pow(x, a) * Math.exp(-x) * sum;
    }

    gamma(x) {
        if (x < 0.5) {
            return Math.PI / (Math.sin(Math.PI * x) * this.gamma(1 - x));
        }

        x -= 1;
        let result = 0.99999999999980993;
        const coefficients = [
            676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012,
            9.9843695780195716e-6, 1.5056327351493116e-7
        ];

        for (let i = 0; i < coefficients.length; i++) {
            result += coefficients[i] / (x + i + 1);
        }

        const t = x + coefficients.length - 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * result;
    }
}

export default Regression;