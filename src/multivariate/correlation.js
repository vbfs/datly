class Correlation {
    pearson(x, y) {
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
        let sumXSquared = 0;
        let sumYSquared = 0;

        for (let i = 0; i < n; i++) {
            const xDiff = xValues[i] - meanX;
            const yDiff = yValues[i] - meanY;
            numerator += xDiff * yDiff;
            sumXSquared += xDiff * xDiff;
            sumYSquared += yDiff * yDiff;
        }

        const denominator = Math.sqrt(sumXSquared * sumYSquared);

        if (denominator === 0) {
            return {
                correlation: 0,
                pValue: 1,
                tStatistic: 0,
                degreesOfFreedom: n - 2,
                significant: false,
                confidenceInterval: { lower: 0, upper: 0 },
                sampleSize: n
            };
        }

        const r = numerator / denominator;
        const tStat = r * Math.sqrt((n - 2) / (1 - r * r));
        const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), n - 2));

        const confidenceInterval = this.pearsonConfidenceInterval(r, n);

        return {
            correlation: r,
            pValue: pValue,
            tStatistic: tStat,
            degreesOfFreedom: n - 2,
            significant: pValue < 0.05,
            confidenceInterval: confidenceInterval,
            sampleSize: n,
            interpretation: this.interpretCorrelation(r, pValue)
        };
    }

    spearman(x, y) {
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
                validPairs.push({ x: x[i], y: y[i], originalIndex: i });
            }
        }

        if (validPairs.length < 3) {
            throw new Error('Need at least 3 valid paired observations');
        }

        const n = validPairs.length;
        const xRanks = this.calculateRanks(validPairs.map(pair => pair.x));
        const yRanks = this.calculateRanks(validPairs.map(pair => pair.y));

        const rho = this.pearsonFromArrays(xRanks, yRanks);
        const tStat = rho * Math.sqrt((n - 2) / (1 - rho * rho));
        const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), n - 2));

        return {
            correlation: rho,
            pValue: pValue,
            tStatistic: tStat,
            degreesOfFreedom: n - 2,
            significant: pValue < 0.05,
            sampleSize: n,
            xRanks: xRanks,
            yRanks: yRanks,
            interpretation: this.interpretCorrelation(rho, pValue, 'Spearman')
        };
    }

    kendall(x, y) {
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
        let concordant = 0;
        let discordant = 0;
        let tiesX = 0;
        let tiesY = 0;
        let tiesXY = 0;

        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const xDiff = validPairs[i].x - validPairs[j].x;
                const yDiff = validPairs[i].y - validPairs[j].y;

                if (xDiff === 0 && yDiff === 0) {
                    tiesXY++;
                } else if (xDiff === 0) {
                    tiesX++;
                } else if (yDiff === 0) {
                    tiesY++;
                } else if (xDiff * yDiff > 0) {
                    concordant++;
                } else {
                    discordant++;
                }
            }
        }

        const totalPairs = n * (n - 1) / 2;
        const tau = (concordant - discordant) / Math.sqrt((totalPairs - tiesX) * (totalPairs - tiesY));

        const variance = (2 * (2 * n + 5)) / (9 * n * (n - 1));
        const zStat = tau / Math.sqrt(variance);
        const pValue = 2 * (1 - this.normalCDF(Math.abs(zStat)));

        return {
            correlation: tau,
            pValue: pValue,
            zStatistic: zStat,
            concordantPairs: concordant,
            discordantPairs: discordant,
            tiesX: tiesX,
            tiesY: tiesY,
            tiesXY: tiesXY,
            significant: pValue < 0.05,
            sampleSize: n,
            interpretation: this.interpretCorrelation(tau, pValue, 'Kendall')
        };
    }

    matrix(dataset, method = 'pearson') {
        if (!dataset || !dataset.headers || !dataset.data) {
            throw new Error('Invalid dataset format');
        }

        const numericColumns = dataset.headers.filter(header => {
            const column = dataset.data.map(row => row[header]);
            const numericValues = column.filter(val =>
                typeof val === 'number' && !isNaN(val) && isFinite(val)
            );
            return numericValues.length > 0;
        });

        if (numericColumns.length < 2) {
            throw new Error('Need at least 2 numeric columns for correlation matrix');
        }

        const correlationMatrix = {};
        const pValueMatrix = {};
        const sampleSizeMatrix = {};

        numericColumns.forEach(col1 => {
            correlationMatrix[col1] = {};
            pValueMatrix[col1] = {};
            sampleSizeMatrix[col1] = {};

            numericColumns.forEach(col2 => {
                if (col1 === col2) {
                    correlationMatrix[col1][col2] = 1;
                    pValueMatrix[col1][col2] = 0;
                    sampleSizeMatrix[col1][col2] = dataset.data.length;
                } else {
                    const x = dataset.data.map(row => row[col1]);
                    const y = dataset.data.map(row => row[col2]);

                    try {
                        let result;
                        switch (method) {
                            case 'pearson':
                                result = this.pearson(x, y);
                                break;
                            case 'spearman':
                                result = this.spearman(x, y);
                                break;
                            case 'kendall':
                                result = this.kendall(x, y);
                                break;
                            default:
                                throw new Error(`Unknown correlation method: ${method}`);
                        }

                        correlationMatrix[col1][col2] = result.correlation;
                        pValueMatrix[col1][col2] = result.pValue;
                        sampleSizeMatrix[col1][col2] = result.sampleSize;
                    } catch (error) {
                        correlationMatrix[col1][col2] = NaN;
                        pValueMatrix[col1][col2] = NaN;
                        sampleSizeMatrix[col1][col2] = 0;
                    }
                }
            });
        });

        return {
            correlations: correlationMatrix,
            pValues: pValueMatrix,
            sampleSizes: sampleSizeMatrix,
            columns: numericColumns,
            method: method,
            strongCorrelations: this.findStrongCorrelations(correlationMatrix, pValueMatrix),
            summary: this.summarizeCorrelationMatrix(correlationMatrix, pValueMatrix, numericColumns)
        };
    }

    covariance(x, y, sample = true) {
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

        if (validPairs.length < 2) {
            throw new Error('Need at least 2 valid paired observations');
        }

        const n = validPairs.length;
        const xValues = validPairs.map(pair => pair.x);
        const yValues = validPairs.map(pair => pair.y);

        const meanX = xValues.reduce((sum, val) => sum + val, 0) / n;
        const meanY = yValues.reduce((sum, val) => sum + val, 0) / n;

        const covariance = xValues.reduce((sum, xVal, i) => {
            return sum + (xVal - meanX) * (yValues[i] - meanY);
        }, 0) / (sample ? n - 1 : n);

        return {
            covariance: covariance,
            meanX: meanX,
            meanY: meanY,
            sampleSize: n,
            sample: sample
        };
    }

    covarianceMatrix(dataset, sample = true) {
        if (!dataset || !dataset.headers || !dataset.data) {
            throw new Error('Invalid dataset format');
        }

        const numericColumns = dataset.headers.filter(header => {
            const column = dataset.data.map(row => row[header]);
            const numericValues = column.filter(val =>
                typeof val === 'number' && !isNaN(val) && isFinite(val)
            );
            return numericValues.length > 0;
        });

        if (numericColumns.length < 2) {
            throw new Error('Need at least 2 numeric columns for covariance matrix');
        }

        const covMatrix = {};

        numericColumns.forEach(col1 => {
            covMatrix[col1] = {};

            numericColumns.forEach(col2 => {
                const x = dataset.data.map(row => row[col1]);
                const y = dataset.data.map(row => row[col2]);

                const result = this.covariance(x, y, sample);
                covMatrix[col1][col2] = result.covariance;
            });
        });

        return {
            covariance: covMatrix,
            columns: numericColumns,
            sample: sample
        };
    }

    partialCorrelation(x, y, z) {
        const rxy = this.pearson(x, y);
        const rxz = this.pearson(x, z);
        const ryz = this.pearson(y, z);

        const numerator = rxy.correlation - (rxz.correlation * ryz.correlation);
        const denominator = Math.sqrt((1 - rxz.correlation ** 2) * (1 - ryz.correlation ** 2));

        if (denominator === 0) {
            return {
                correlation: 0,
                pValue: 1,
                significant: false
            };
        }

        const partialR = numerator / denominator;
        const n = Math.min(rxy.sampleSize, rxz.sampleSize, ryz.sampleSize);
        const df = n - 3;
        const tStat = partialR * Math.sqrt(df / (1 - partialR ** 2));
        const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), df));

        return {
            correlation: partialR,
            pValue: pValue,
            tStatistic: tStat,
            degreesOfFreedom: df,
            significant: pValue < 0.05,
            sampleSize: n,
            controllingFor: 'third variable'
        };
    }

    calculateRanks(values) {
        const indexed = values.map((value, index) => ({ value, index }));
        indexed.sort((a, b) => a.value - b.value);

        const ranks = new Array(values.length);
        let currentRank = 1;

        for (let i = 0; i < indexed.length; i++) {
            const tiedValues = [indexed[i]];

            while (i + 1 < indexed.length && indexed[i + 1].value === indexed[i].value) {
                i++;
                tiedValues.push(indexed[i]);
            }

            const averageRank = (currentRank + currentRank + tiedValues.length - 1) / 2;
            tiedValues.forEach(item => {
                ranks[item.index] = averageRank;
            });

            currentRank += tiedValues.length;
        }

        return ranks;
    }

    pearsonFromArrays(x, y) {
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

    pearsonConfidenceInterval(r, n, confidence = 0.95) {
        if (Math.abs(r) >= 1) {
            return { lower: r, upper: r };
        }

        const fisherZ = 0.5 * Math.log((1 + r) / (1 - r));
        const standardError = 1 / Math.sqrt(n - 3);
        const alpha = 1 - confidence;
        const zCritical = this.normalInverse(1 - alpha / 2);
        const marginOfError = zCritical * standardError;

        const lowerZ = fisherZ - marginOfError;
        const upperZ = fisherZ + marginOfError;

        const lowerR = (Math.exp(2 * lowerZ) - 1) / (Math.exp(2 * lowerZ) + 1);
        const upperR = (Math.exp(2 * upperZ) - 1) / (Math.exp(2 * upperZ) + 1);

        return { lower: lowerR, upper: upperR };
    }

    findStrongCorrelations(correlationMatrix, pValueMatrix, threshold = 0.7) {
        const strongCorrelations = [];
        const columns = Object.keys(correlationMatrix);

        for (let i = 0; i < columns.length; i++) {
            for (let j = i + 1; j < columns.length; j++) {
                const col1 = columns[i];
                const col2 = columns[j];
                const correlation = correlationMatrix[col1][col2];
                const pValue = pValueMatrix[col1][col2];

                if (Math.abs(correlation) >= threshold && pValue < 0.05) {
                    strongCorrelations.push({
                        variable1: col1,
                        variable2: col2,
                        correlation: correlation,
                        pValue: pValue,
                        strength: this.getCorrelationStrength(Math.abs(correlation))
                    });
                }
            }
        }

        return strongCorrelations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
    }

    summarizeCorrelationMatrix(correlationMatrix, pValueMatrix, columns) {
        let totalCorrelations = 0;
        let significantCorrelations = 0;
        let strongPositive = 0;
        let strongNegative = 0;
        let maxCorrelation = 0;
        let minCorrelation = 0;

        for (let i = 0; i < columns.length; i++) {
            for (let j = i + 1; j < columns.length; j++) {
                const col1 = columns[i];
                const col2 = columns[j];
                const correlation = correlationMatrix[col1][col2];
                const pValue = pValueMatrix[col1][col2];

                if (!isNaN(correlation)) {
                    totalCorrelations++;

                    if (pValue < 0.05) {
                        significantCorrelations++;
                    }

                    if (correlation > 0.7) strongPositive++;
                    if (correlation < -0.7) strongNegative++;

                    maxCorrelation = Math.max(maxCorrelation, correlation);
                    minCorrelation = Math.min(minCorrelation, correlation);
                }
            }
        }

        return {
            totalPairs: totalCorrelations,
            significantPairs: significantCorrelations,
            strongPositiveCorrelations: strongPositive,
            strongNegativeCorrelations: strongNegative,
            maxCorrelation: maxCorrelation,
            minCorrelation: minCorrelation,
            averageAbsoluteCorrelation: this.calculateAverageAbsoluteCorrelation(correlationMatrix, columns)
        };
    }

    calculateAverageAbsoluteCorrelation(correlationMatrix, columns) {
        let sum = 0;
        let count = 0;

        for (let i = 0; i < columns.length; i++) {
            for (let j = i + 1; j < columns.length; j++) {
                const correlation = correlationMatrix[columns[i]][columns[j]];
                if (!isNaN(correlation)) {
                    sum += Math.abs(correlation);
                    count++;
                }
            }
        }

        return count > 0 ? sum / count : 0;
    }

    getCorrelationStrength(absCorrelation) {
        if (absCorrelation >= 0.9) return 'Very Strong';
        if (absCorrelation >= 0.7) return 'Strong';
        if (absCorrelation >= 0.5) return 'Moderate';
        if (absCorrelation >= 0.3) return 'Weak';
        return 'Very Weak';
    }

    interpretCorrelation(correlation, pValue, method = 'Pearson') {
        const strength = this.getCorrelationStrength(Math.abs(correlation));
        const direction = correlation > 0 ? 'positive' : 'negative';
        const significance = pValue < 0.05 ? 'significant' : 'not significant';

        return `${method} correlation: ${strength} ${direction} relationship (r = ${correlation.toFixed(4)}, p = ${pValue.toFixed(4)}, ${significance})`;
    }

    tCDF(t, df) {
        const x = df / (t * t + df);
        return 1 - 0.5 * this.incompleteBeta(df / 2, 0.5, x);
    }

    normalCDF(z) {
        return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
    }

    normalInverse(p) {
        if (p <= 0 || p >= 1) throw new Error('p must be between 0 and 1');

        const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                   1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
        const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                   6.680131188771972e+01, -1.328068155288572e+01, 1];

        if (p > 0.5) return -this.normalInverse(1 - p);

        const q = Math.sqrt(-2 * Math.log(p));
        let num = a[5];
        let den = b[5];

        for (let i = 4; i >= 0; i--) {
            num = num * q + a[i];
            den = den * q + b[i];
        }

        return num / den;
    }

    erf(x) {
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const p = 0.3275911;

        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x);

        const t = 1 / (1 + p * x);
        const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return sign * y;
    }

    incompleteBeta(a, b, x) {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

        const bt = Math.exp(this.logGamma(a + b) - this.logGamma(a) - this.logGamma(b) +
                          a * Math.log(x) + b * Math.log(1 - x));

        if (x < (a + 1) / (a + b + 2)) {
            return bt * this.continuedFractionBeta(a, b, x) / a;
        } else {
            return 1 - bt * this.continuedFractionBeta(b, a, 1 - x) / b;
        }
    }

    continuedFractionBeta(a, b, x) {
        const qab = a + b;
        const qap = a + 1;
        const qam = a - 1;
        let c = 1;
        let d = 1 - qab * x / qap;

        if (Math.abs(d) < 1e-30) d = 1e-30;
        d = 1 / d;
        let h = d;

        for (let m = 1; m <= 100; m++) {
            const m2 = 2 * m;
            let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            const del = d * c;
            h *= del;

            if (Math.abs(del - 1) < 1e-12) break;
        }

        return h;
    }

    logGamma(x) {
        const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
                     -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
        let ser = 1.000000000190015;

        const xx = x;
        let y = x;
        let tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);

        for (let j = 0; j < 6; j++) ser += cof[j] / ++y;

        return -tmp + Math.log(2.5066282746310005 * ser / xx);
    }
}

export default Correlation;