class NormalityTests {
    shapiroWilk(sample, alpha = 0.05) {
        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 3 || validSample.length > 5000) {
            throw new Error('Shapiro-Wilk test requires between 3 and 5000 observations');
        }

        const n = validSample.length;
        const sorted = [...validSample].sort((a, b) => a - b);

        const mean = sorted.reduce((sum, val) => sum + val, 0) / n;
        const ss = sorted.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);

        if (ss === 0) {
            return {
                statistic: NaN,
                pValue: NaN,
                isNormal: false,
                error: 'All values are identical'
            };
        }

        let b = 0;
        for (let i = 0; i < Math.floor(n / 2); i++) {
            const a = this.shapiroWilkCoefficient(i + 1, n);
            b += a * (sorted[n - 1 - i] - sorted[i]);
        }

        const w = (b * b) / ss;
        const pValue = this.shapiroWilkPValue(w, n);

        return {
            statistic: w,
            pValue: pValue,
            isNormal: pValue > alpha,
            alpha: alpha,
            sampleSize: n,
            interpretation: this.interpretNormalityResult(pValue, alpha, 'Shapiro-Wilk')
        };
    }

    shapiroWilkCoefficient(i, n) {
        const c = [
            0, 0.7071, 0.7071, 0.6872, 0.6646, 0.6431, 0.6233, 0.6052, 0.5888, 0.5739, 0.5601
        ];

        if (n <= 10 && i <= n) {
            return c[i] || 0.5;
        }

        const m = 0.5;
        const s = 1;
        return m + s * this.normalInverse((i - 0.375) / (n + 0.25));
    }

    shapiroWilkPValue(w, n) {
        if (n < 3) return 1;
        if (w >= 1) return 1;
        if (w <= 0) return 0;

        const ln_w = Math.log(w);
        let z;

        if (n <= 11) {
            const gamma = 0.459 * n - 2.273;
            z = -gamma * ln_w;
        } else {
            const mu = -1.5861 - 0.31082 * Math.log(n) - 0.083751 * Math.log(n) ** 2 + 0.0038915 * Math.log(n) ** 3;
            const sigma = Math.exp(-0.4803 - 0.082676 * Math.log(n) + 0.0030302 * Math.log(n) ** 2);
            z = (ln_w - mu) / sigma;
        }

        return 1 - this.standardNormalCDF(z);
    }

    kolmogorovSmirnov(sample, alpha = 0.05) {
        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 5) {
            throw new Error('Kolmogorov-Smirnov test requires at least 5 observations');
        }

        const n = validSample.length;
        const sorted = [...validSample].sort((a, b) => a - b);

        const mean = sorted.reduce((sum, val) => sum + val, 0) / n;
        const variance = sorted.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
            return {
                statistic: NaN,
                pValue: NaN,
                isNormal: false,
                error: 'All values are identical'
            };
        }

        let maxD = 0;

        for (let i = 0; i < n; i++) {
            const standardized = (sorted[i] - mean) / stdDev;
            const empiricalCDF = (i + 1) / n;
            const theoreticalCDF = this.standardNormalCDF(standardized);

            const d1 = Math.abs(empiricalCDF - theoreticalCDF);
            const d2 = Math.abs((i / n) - theoreticalCDF);

            maxD = Math.max(maxD, d1, d2);
        }

        const sqrtN = Math.sqrt(n);
        const lambda = maxD * sqrtN;
        const pValue = this.kolmogorovSmirnovPValue(lambda);

        return {
            statistic: maxD,
            pValue: pValue,
            isNormal: pValue > alpha,
            alpha: alpha,
            sampleSize: n,
            lambda: lambda,
            interpretation: this.interpretNormalityResult(pValue, alpha, 'Kolmogorov-Smirnov')
        };
    }

    kolmogorovSmirnovPValue(lambda) {
        if (lambda <= 0) return 1;
        if (lambda > 7) return 0;

        let sum = 0;
        for (let k = 1; k <= 100; k++) {
            const term = 2 * Math.pow(-1, k - 1) * Math.exp(-2 * k * k * lambda * lambda);
            sum += term;
            if (Math.abs(term) < 1e-12) break;
        }

        return Math.min(1, Math.max(0, sum));
    }

    andersonDarling(sample, alpha = 0.05) {
        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 8) {
            throw new Error('Anderson-Darling test requires at least 8 observations');
        }

        const n = validSample.length;
        const sorted = [...validSample].sort((a, b) => a - b);

        const mean = sorted.reduce((sum, val) => sum + val, 0) / n;
        const variance = sorted.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
            return {
                statistic: NaN,
                pValue: NaN,
                isNormal: false,
                error: 'All values are identical'
            };
        }

        let sum = 0;
        for (let i = 0; i < n; i++) {
            const standardized = (sorted[i] - mean) / stdDev;
            const phi = this.standardNormalCDF(standardized);
            const phiComplement = this.standardNormalCDF(-standardized);

            if (phi > 0 && phiComplement > 0) {
                sum += (2 * i + 1) * (Math.log(phi) + Math.log(phiComplement));
            }
        }

        const a2 = -n - (1 / n) * sum;
        const a2Star = a2 * (1 + 0.75 / n + 2.25 / (n * n));

        const pValue = this.andersonDarlingPValue(a2Star);

        return {
            statistic: a2,
            adjustedStatistic: a2Star,
            pValue: pValue,
            isNormal: pValue > alpha,
            alpha: alpha,
            sampleSize: n,
            interpretation: this.interpretNormalityResult(pValue, alpha, 'Anderson-Darling')
        };
    }

    andersonDarlingPValue(a2Star) {
        if (a2Star <= 0.2) {
            return 1 - Math.exp(-1.2337141 / a2Star) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * a2Star) * a2Star) * a2Star) * a2Star) * a2Star);
        } else if (a2Star <= 0.34) {
            return 1 - Math.exp(-0.9177603 - 1.25156 * a2Star) * (1.38033 + (0.421981 - 0.668119 * a2Star) * a2Star);
        } else if (a2Star < 0.6) {
            return Math.exp(0.731 - 3.009 * a2Star + 4.86 * a2Star * a2Star);
        } else if (a2Star < 10) {
            return Math.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * a2Star) * a2Star) * a2Star) * a2Star) * a2Star);
        } else {
            return 0;
        }
    }

    jarqueBera(sample, alpha = 0.05) {
        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 4) {
            throw new Error('Jarque-Bera test requires at least 4 observations');
        }

        const n = validSample.length;
        const skewness = this.calculateSkewness(validSample);
        const kurtosis = this.calculateKurtosis(validSample, true);

        const jb = (n / 6) * (Math.pow(skewness, 2) + Math.pow(kurtosis, 2) / 4);
        const pValue = 1 - this.chiSquareCDF(jb, 2);

        return {
            statistic: jb,
            pValue: pValue,
            isNormal: pValue > alpha,
            alpha: alpha,
            sampleSize: n,
            skewness: skewness,
            excessKurtosis: kurtosis,
            interpretation: this.interpretNormalityResult(pValue, alpha, 'Jarque-Bera')
        };
    }

    dagoTest(sample, alpha = 0.05) {
        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 20) {
            throw new Error("D'Agostino test requires at least 20 observations");
        }

        const n = validSample.length;
        const skewness = this.calculateSkewness(validSample);
        const kurtosis = this.calculateKurtosis(validSample, true);

        const skewnessZ = this.skewnessZScore(skewness, n);
        const kurtosisZ = this.kurtosisZScore(kurtosis, n);

        const k2 = skewnessZ * skewnessZ + kurtosisZ * kurtosisZ;
        const pValue = 1 - this.chiSquareCDF(k2, 2);

        return {
            statistic: k2,
            pValue: pValue,
            isNormal: pValue > alpha,
            alpha: alpha,
            sampleSize: n,
            skewness: skewness,
            excessKurtosis: kurtosis,
            skewnessZ: skewnessZ,
            kurtosisZ: kurtosisZ,
            interpretation: this.interpretNormalityResult(pValue, alpha, "D'Agostino K-squared")
        };
    }

    skewnessZScore(skewness, n) {
        const y = skewness * Math.sqrt((n + 1) * (n + 3) / (6 * (n - 2)));
        const beta2 = 3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9));
        const w2 = -1 + Math.sqrt(2 * (beta2 - 1));
        const delta = 1 / Math.sqrt(0.5 * Math.log(w2));
        const alpha = Math.sqrt(2 / (w2 - 1));

        return delta * Math.log(y / alpha + Math.sqrt((y / alpha) ** 2 + 1));
    }

    kurtosisZScore(kurtosis, n) {
        const e = 3 * (n - 1) / (n + 1);
        const varb2 = 24 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
        const x = (kurtosis - e) / Math.sqrt(varb2);
        const sqrtb1 = 6 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * Math.sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)));

        const a = 6 + 8 / sqrtb1 * (2 / sqrtb1 + Math.sqrt(1 + 4 / (sqrtb1 ** 2)));

        return Math.sqrt(9 * a / 2) * ((1 - 2 / a) / (1 + x * Math.sqrt(2 / (a - 4))) - 1);
    }

    lillieforsTest(sample, alpha = 0.05) {
        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length < 4 || validSample.length > 1000) {
            throw new Error('Lilliefors test requires between 4 and 1000 observations');
        }

        const n = validSample.length;
        const sorted = [...validSample].sort((a, b) => a - b);

        const mean = sorted.reduce((sum, val) => sum + val, 0) / n;
        const variance = sorted.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
            return {
                statistic: NaN,
                pValue: NaN,
                isNormal: false,
                error: 'All values are identical'
            };
        }

        let maxD = 0;

        for (let i = 0; i < n; i++) {
            const standardized = (sorted[i] - mean) / stdDev;
            const empiricalCDF = (i + 1) / n;
            const theoreticalCDF = this.standardNormalCDF(standardized);

            const d1 = Math.abs(empiricalCDF - theoreticalCDF);
            const d2 = Math.abs((i / n) - theoreticalCDF);

            maxD = Math.max(maxD, d1, d2);
        }

        const pValue = this.lillieforsPValue(maxD, n);

        return {
            statistic: maxD,
            pValue: pValue,
            isNormal: pValue > alpha,
            alpha: alpha,
            sampleSize: n,
            interpretation: this.interpretNormalityResult(pValue, alpha, 'Lilliefors')
        };
    }

    lillieforsPValue(d, n) {
        const criticalValues = {
            4: 0.381, 5: 0.337, 6: 0.319, 7: 0.300, 8: 0.285,
            9: 0.271, 10: 0.258, 11: 0.249, 12: 0.242, 13: 0.234,
            14: 0.227, 15: 0.220, 16: 0.213, 17: 0.206, 18: 0.200,
            19: 0.195, 20: 0.190, 25: 0.173, 30: 0.161, 40: 0.144,
            50: 0.131, 100: 0.096
        };

        let criticalValue;
        if (criticalValues[n]) {
            criticalValue = criticalValues[n];
        } else if (n > 100) {
            criticalValue = 0.886 / Math.sqrt(n);
        } else {
            const keys = Object.keys(criticalValues).map(Number).sort((a, b) => a - b);
            const lower = keys.filter(k => k <= n).pop();
            const upper = keys.filter(k => k >= n)[0];

            if (lower === upper) {
                criticalValue = criticalValues[lower];
            } else {
                const ratio = (n - lower) / (upper - lower);
                criticalValue = criticalValues[lower] + ratio * (criticalValues[upper] - criticalValues[lower]);
            }
        }

        if (d > criticalValue) {
            return 0.01;
        } else if (d < criticalValue * 0.8) {
            return 0.2;
        } else {
            return 0.05;
        }
    }

    batchNormalityTest(sample, alpha = 0.05) {
        const results = {};

        try {
            results.shapiroWilk = this.shapiroWilk(sample, alpha);
        } catch (error) {
            results.shapiroWilk = { error: error.message };
        }

        try {
            results.jarqueBera = this.jarqueBera(sample, alpha);
        } catch (error) {
            results.jarqueBera = { error: error.message };
        }

        try {
            results.andersonDarling = this.andersonDarling(sample, alpha);
        } catch (error) {
            results.andersonDarling = { error: error.message };
        }

        try {
            results.kolmogorovSmirnov = this.kolmogorovSmirnov(sample, alpha);
        } catch (error) {
            results.kolmogorovSmirnov = { error: error.message };
        }

        try {
            if (sample.length >= 20) {
                results.dagostino = this.dagoTest(sample, alpha);
            }
        } catch (error) {
            results.dagostino = { error: error.message };
        }

        const validTests = Object.entries(results).filter(([_, result]) => !result.error && result.pValue !== undefined);
        const normalCount = validTests.filter(([_, result]) => result.isNormal).length;
        const totalTests = validTests.length;

        return {
            individualTests: results,
            summary: {
                testsRun: totalTests,
                testsPassingNormality: normalCount,
                consensusNormal: normalCount >= Math.ceil(totalTests / 2),
                strongNormalEvidence: normalCount === totalTests,
                strongNonNormalEvidence: normalCount === 0
            },
            recommendation: this.getNormalityRecommendation(results, totalTests, normalCount)
        };
    }

    getNormalityRecommendation(results, totalTests, normalCount) {
        if (totalTests === 0) {
            return "Unable to assess normality - insufficient data or all tests failed";
        }

        const ratio = normalCount / totalTests;

        if (ratio === 1) {
            return "Strong evidence for normality - all tests indicate normal distribution";
        } else if (ratio >= 0.75) {
            return "Good evidence for normality - most tests indicate normal distribution";
        } else if (ratio >= 0.5) {
            return "Mixed evidence - consider visual inspection and domain knowledge";
        } else if (ratio > 0) {
            return "Evidence against normality - most tests indicate non-normal distribution";
        } else {
            return "Strong evidence against normality - all tests indicate non-normal distribution";
        }
    }

    calculateSkewness(sample, bias = false) {
        const n = sample.length;
        const mean = sample.reduce((sum, val) => sum + val, 0) / n;
        const variance = sample.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) return 0;

        const skewSum = sample.reduce((sum, val) => {
            return sum + Math.pow((val - mean) / stdDev, 3);
        }, 0);

        if (bias) {
            return skewSum / n;
        } else {
            return (n / ((n - 1) * (n - 2))) * skewSum;
        }
    }

    calculateKurtosis(sample, fisher = true, bias = false) {
        const n = sample.length;
        const mean = sample.reduce((sum, val) => sum + val, 0) / n;
        const variance = sample.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) return fisher ? -3 : 0;

        const kurtSum = sample.reduce((sum, val) => {
            return sum + Math.pow((val - mean) / stdDev, 4);
        }, 0);

        let kurtosis;
        if (bias) {
            kurtosis = kurtSum / n;
        } else {
            kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * kurtSum -
                      (3 * Math.pow(n - 1, 2)) / ((n - 2) * (n - 3));
        }

        return fisher ? kurtosis - 3 : kurtosis;
    }

    interpretNormalityResult(pValue, alpha, testName) {
        if (pValue > alpha) {
            return `${testName} test: Fail to reject null hypothesis (p-value = ${pValue.toFixed(4)} > α = ${alpha}). Data appears to be normally distributed.`;
        } else {
            return `${testName} test: Reject null hypothesis (p-value = ${pValue.toFixed(4)} ≤ α = ${alpha}). Data appears to be non-normally distributed.`;
        }
    }

    standardNormalCDF(z) {
        return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
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

    chiSquareCDF(x, df) {
        if (x <= 0) return 0;
        return this.incompleteGamma(df / 2, x / 2) / this.gamma(df / 2);
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

export default NormalityTests;