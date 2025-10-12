class ConfidenceIntervals {
    mean(sample, confidence = 0.95) {
        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 2) {
            throw new Error('Need at least 2 valid observations');
        }

        const n = validSample.length;
        const mean = validSample.reduce((sum, val) => sum + val, 0) / n;
        const variance = validSample.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdError = Math.sqrt(variance / n);
        const alpha = 1 - confidence;
        const tCritical = this.tInverse(1 - alpha / 2, n - 1);
        const marginOfError = tCritical * stdError;

        return {
            mean: mean,
            standardError: stdError,
            marginOfError: marginOfError,
            lowerBound: mean - marginOfError,
            upperBound: mean + marginOfError,
            confidence: confidence,
            degreesOfFreedom: n - 1,
            sampleSize: n
        };
    }

    meanKnownVariance(sample, populationStd, confidence = 0.95) {
        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        if (populationStd <= 0) {
            throw new Error('Population standard deviation must be positive');
        }

        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const n = validSample.length;
        const mean = validSample.reduce((sum, val) => sum + val, 0) / n;
        const stdError = populationStd / Math.sqrt(n);
        const alpha = 1 - confidence;
        const zCritical = this.normalInverse(1 - alpha / 2);
        const marginOfError = zCritical * stdError;

        return {
            mean: mean,
            standardError: stdError,
            marginOfError: marginOfError,
            lowerBound: mean - marginOfError,
            upperBound: mean + marginOfError,
            confidence: confidence,
            sampleSize: n,
            populationStd: populationStd
        };
    }

    proportion(successes, total, confidence = 0.95) {
        if (!Number.isInteger(successes) || !Number.isInteger(total)) {
            throw new Error('Successes and total must be integers');
        }

        if (successes < 0 || total <= 0 || successes > total) {
            throw new Error('Invalid values: 0 ≤ successes ≤ total, total > 0');
        }

        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        const p = successes / total;
        const alpha = 1 - confidence;
        const zCritical = this.normalInverse(1 - alpha / 2);

        if (total * p < 5 || total * (1 - p) < 5) {
            console.warn('Warning: Normal approximation may not be accurate. Consider using Wilson score interval.');
        }

        const stdError = Math.sqrt(p * (1 - p) / total);
        const marginOfError = zCritical * stdError;

        const normalInterval = {
            proportion: p,
            standardError: stdError,
            marginOfError: marginOfError,
            lowerBound: Math.max(0, p - marginOfError),
            upperBound: Math.min(1, p + marginOfError),
            confidence: confidence,
            sampleSize: total,
            successes: successes
        };

        const wilsonInterval = this.wilsonScoreInterval(successes, total, confidence);

        return {
            normal: normalInterval,
            wilson: wilsonInterval,
            recommended: total * p >= 5 && total * (1 - p) >= 5 ? normalInterval : wilsonInterval
        };
    }

    wilsonScoreInterval(successes, total, confidence = 0.95) {
        const p = successes / total;
        const alpha = 1 - confidence;
        const z = this.normalInverse(1 - alpha / 2);
        const z2 = z * z;

        const denominator = 1 + z2 / total;
        const center = (p + z2 / (2 * total)) / denominator;
        const halfWidth = z * Math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator;

        return {
            proportion: p,
            center: center,
            halfWidth: halfWidth,
            lowerBound: center - halfWidth,
            upperBound: center + halfWidth,
            confidence: confidence,
            sampleSize: total,
            successes: successes
        };
    }

    variance(sample, confidence = 0.95) {
        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 2) {
            throw new Error('Need at least 2 valid observations');
        }

        const n = validSample.length;
        const mean = validSample.reduce((sum, val) => sum + val, 0) / n;
        const sampleVariance = validSample.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);

        const alpha = 1 - confidence;
        const df = n - 1;
        const chiSquareLower = this.chiSquareInverse(alpha / 2, df);
        const chiSquareUpper = this.chiSquareInverse(1 - alpha / 2, df);

        const lowerBound = (df * sampleVariance) / chiSquareUpper;
        const upperBound = (df * sampleVariance) / chiSquareLower;

        return {
            sampleVariance: sampleVariance,
            lowerBound: lowerBound,
            upperBound: upperBound,
            confidence: confidence,
            degreesOfFreedom: df,
            sampleSize: n
        };
    }

    standardDeviation(sample, confidence = 0.95) {
        const varianceCI = this.variance(sample, confidence);

        return {
            sampleStandardDeviation: Math.sqrt(varianceCI.sampleVariance),
            lowerBound: Math.sqrt(varianceCI.lowerBound),
            upperBound: Math.sqrt(varianceCI.upperBound),
            confidence: confidence,
            degreesOfFreedom: varianceCI.degreesOfFreedom,
            sampleSize: varianceCI.sampleSize
        };
    }

    meanDifference(sample1, sample2, confidence = 0.95, equalVariances = false) {
        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        const validSample1 = sample1.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );
        const validSample2 = sample2.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample1.length < 2 || validSample2.length < 2) {
            throw new Error('Both samples must have at least 2 valid observations');
        }

        const n1 = validSample1.length;
        const n2 = validSample2.length;
        const mean1 = validSample1.reduce((sum, val) => sum + val, 0) / n1;
        const mean2 = validSample2.reduce((sum, val) => sum + val, 0) / n2;
        const meanDiff = mean1 - mean2;

        const var1 = validSample1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / (n1 - 1);
        const var2 = validSample2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / (n2 - 1);

        let stdError, df;

        if (equalVariances) {
            const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
            stdError = Math.sqrt(pooledVar * (1/n1 + 1/n2));
            df = n1 + n2 - 2;
        } else {
            stdError = Math.sqrt(var1/n1 + var2/n2);
            df = Math.pow(var1/n1 + var2/n2, 2) /
                (Math.pow(var1/n1, 2)/(n1-1) + Math.pow(var2/n2, 2)/(n2-1));
        }

        const alpha = 1 - confidence;
        const tCritical = this.tInverse(1 - alpha / 2, df);
        const marginOfError = tCritical * stdError;

        return {
            meanDifference: meanDiff,
            sample1Mean: mean1,
            sample2Mean: mean2,
            standardError: stdError,
            marginOfError: marginOfError,
            lowerBound: meanDiff - marginOfError,
            upperBound: meanDiff + marginOfError,
            confidence: confidence,
            degreesOfFreedom: df,
            equalVariances: equalVariances
        };
    }

    pairedMeanDifference(sample1, sample2, confidence = 0.95) {
        if (sample1.length !== sample2.length) {
            throw new Error('Paired samples must have equal length');
        }

        const differences = [];
        for (let i = 0; i < sample1.length; i++) {
            if (typeof sample1[i] === 'number' && typeof sample2[i] === 'number' &&
                !isNaN(sample1[i]) && !isNaN(sample2[i]) &&
                isFinite(sample1[i]) && isFinite(sample2[i])) {
                differences.push(sample1[i] - sample2[i]);
            }
        }

        if (differences.length < 2) {
            throw new Error('Need at least 2 valid paired observations');
        }

        return this.mean(differences, confidence);
    }

    correlation(x, y, confidence = 0.95, method = 'pearson') {
        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        if (x.length !== y.length) {
            throw new Error('x and y must have the same length');
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
        let r;

        if (method === 'pearson') {
            const xVals = validPairs.map(pair => pair.x);
            const yVals = validPairs.map(pair => pair.y);
            r = this.pearsonCorrelation(xVals, yVals);
        } else {
            throw new Error('Only Pearson correlation is supported for confidence intervals');
        }

        if (Math.abs(r) >= 1) {
            throw new Error('Cannot calculate confidence interval for perfect correlation');
        }

        const fisherZ = 0.5 * Math.log((1 + r) / (1 - r));
        const stdError = 1 / Math.sqrt(n - 3);
        const alpha = 1 - confidence;
        const zCritical = this.normalInverse(1 - alpha / 2);
        const marginOfError = zCritical * stdError;

        const lowerZ = fisherZ - marginOfError;
        const upperZ = fisherZ + marginOfError;

        const lowerR = (Math.exp(2 * lowerZ) - 1) / (Math.exp(2 * lowerZ) + 1);
        const upperR = (Math.exp(2 * upperZ) - 1) / (Math.exp(2 * upperZ) + 1);

        return {
            correlation: r,
            fisherZ: fisherZ,
            standardError: stdError,
            lowerBound: lowerR,
            upperBound: upperR,
            confidence: confidence,
            sampleSize: n
        };
    }

    pearsonCorrelation(x, y) {
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

    median(sample, confidence = 0.95) {
        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 3) {
            throw new Error('Need at least 3 valid observations for median CI');
        }

        const sorted = [...validSample].sort((a, b) => a - b);
        const n = sorted.length;
        const alpha = 1 - confidence;

        const z = this.normalInverse(1 - alpha / 2);
        const j = Math.floor(n / 2 - z * Math.sqrt(n) / 2);
        const k = Math.ceil(n / 2 + z * Math.sqrt(n) / 2);

        const lowerIndex = Math.max(0, j - 1);
        const upperIndex = Math.min(n - 1, k - 1);

        const median = n % 2 === 0 ?
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2 :
            sorted[Math.floor(n / 2)];

        return {
            median: median,
            lowerBound: sorted[lowerIndex],
            upperBound: sorted[upperIndex],
            confidence: confidence,
            sampleSize: n,
            lowerIndex: lowerIndex + 1,
            upperIndex: upperIndex + 1
        };
    }

    bootstrapCI(sample, statistic, confidence = 0.95, iterations = 1000) {
        if (confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be between 0 and 1');
        }

        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const bootstrapStats = [];

        for (let i = 0; i < iterations; i++) {
            const bootstrapSample = [];
            for (let j = 0; j < validSample.length; j++) {
                const randomIndex = Math.floor(Math.random() * validSample.length);
                bootstrapSample.push(validSample[randomIndex]);
            }

            const stat = this.calculateStatistic(bootstrapSample, statistic);
            bootstrapStats.push(stat);
        }

        bootstrapStats.sort((a, b) => a - b);

        const alpha = 1 - confidence;
        const lowerIndex = Math.floor((alpha / 2) * iterations);
        const upperIndex = Math.floor((1 - alpha / 2) * iterations) - 1;

        const originalStat = this.calculateStatistic(validSample, statistic);
        const bootstrapMean = bootstrapStats.reduce((sum, val) => sum + val, 0) / bootstrapStats.length;

        return {
            originalStatistic: originalStat,
            bootstrapMean: bootstrapMean,
            bias: bootstrapMean - originalStat,
            standardError: this.calculateStandardDeviation(bootstrapStats),
            lowerBound: bootstrapStats[lowerIndex],
            upperBound: bootstrapStats[upperIndex],
            confidence: confidence,
            iterations: iterations
        };
    }

    calculateStatistic(sample, statistic) {
        switch (statistic) {
            case 'mean':
                return sample.reduce((sum, val) => sum + val, 0) / sample.length;
            case 'median':
                const sorted = [...sample].sort((a, b) => a - b);
                const mid = Math.floor(sorted.length / 2);
                return sorted.length % 2 === 0 ?
                    (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
            case 'std':
                return this.calculateStandardDeviation(sample);
            case 'var':
                return this.calculateVariance(sample);
            default:
                if (typeof statistic === 'function') {
                    return statistic(sample);
                }
                throw new Error(`Unknown statistic: ${statistic}`);
        }
    }

    calculateStandardDeviation(sample) {
        const mean = sample.reduce((sum, val) => sum + val, 0) / sample.length;
        const variance = sample.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (sample.length - 1);
        return Math.sqrt(variance);
    }

    calculateVariance(sample) {
        const mean = sample.reduce((sum, val) => sum + val, 0) / sample.length;
        return sample.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (sample.length - 1);
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

    tInverse(p, df) {
        if (p <= 0 || p >= 1) return NaN;

        let t = this.normalInverse(p);
        const c1 = t / 4;
        const c2 = (5 * t * t * t + 16 * t) / 96;
        const c3 = (3 * t * t * t * t * t + 19 * t * t * t + 17 * t) / 384;

        return t + c1 / df + c2 / (df * df) + c3 / (df * df * df);
    }

    chiSquareInverse(p, df) {
        if (p <= 0 || p >= 1) throw new Error('p must be between 0 and 1');
        if (df <= 0) throw new Error('Degrees of freedom must be positive');

        let x = df;
        const tolerance = 1e-12;
        const maxIterations = 100;

        for (let i = 0; i < maxIterations; i++) {
            const fx = this.chiSquareCDF(x, df) - p;
            const fpx = this.chiSquarePDF(x, df);

            if (Math.abs(fx) < tolerance) break;
            if (fpx === 0) break;

            x = x - fx / fpx;
            if (x <= 0) x = df / 2;
        }

        return Math.max(0, x);
    }

    chiSquareCDF(x, df) {
        if (x <= 0) return 0;
        return this.incompleteGamma(df / 2, x / 2) / this.gamma(df / 2);
    }

    chiSquarePDF(x, df) {
        if (x <= 0) return 0;
        return Math.pow(x, df / 2 - 1) * Math.exp(-x / 2) / (Math.pow(2, df / 2) * this.gamma(df / 2));
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
        const coefficients = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
        ];

        if (x < 0.5) {
            return Math.PI / (Math.sin(Math.PI * x) * this.gamma(1 - x));
        }

        x -= 1;
        let result = coefficients[0];
        for (let i = 1; i < coefficients.length; i++) {
            result += coefficients[i] / (x + i);
        }

        const t = x + coefficients.length - 1.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * result;
    }
}

export default ConfidenceIntervals;