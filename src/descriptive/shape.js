class Shape {
    skewness(column, bias = true) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length < 3) {
            throw new Error('Skewness calculation requires at least 3 values');
        }

        const n = validValues.length;
        const mean = validValues.reduce((sum, val) => sum + val, 0) / n;

        const variance = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
            return 0;
        }

        const skewSum = validValues.reduce((sum, val) => {
            return sum + Math.pow((val - mean) / stdDev, 3);
        }, 0);

        if (bias) {
            return skewSum / n;
        } else {
            return (n / ((n - 1) * (n - 2))) * skewSum;
        }
    }

    kurtosis(column, bias = true, fisher = true) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length < 4) {
            throw new Error('Kurtosis calculation requires at least 4 values');
        }

        const n = validValues.length;
        const mean = validValues.reduce((sum, val) => sum + val, 0) / n;

        const variance = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
            return fisher ? -3 : 0;
        }

        const kurtSum = validValues.reduce((sum, val) => {
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

    isNormalDistribution(column, alpha = 0.05) {
        const shapiroResult = this.shapiroWilkTest(column);
        const jarqueBeraResult = this.jarqueBeraTest(column);

        return {
            shapiroWilk: {
                statistic: shapiroResult.statistic,
                pValue: shapiroResult.pValue,
                isNormal: shapiroResult.pValue > alpha
            },
            jarqueBera: {
                statistic: jarqueBeraResult.statistic,
                pValue: jarqueBeraResult.pValue,
                isNormal: jarqueBeraResult.pValue > alpha
            },
            skewness: this.skewness(column, false),
            kurtosis: this.kurtosis(column, false, true),
            isNormalByTests: shapiroResult.pValue > alpha && jarqueBeraResult.pValue > alpha
        };
    }

    shapiroWilkTest(column) {
        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length < 3 || validValues.length > 5000) {
            throw new Error('Shapiro-Wilk test requires between 3 and 5000 observations');
        }

        const n = validValues.length;
        const sorted = [...validValues].sort((a, b) => a - b);

        const mean = sorted.reduce((sum, val) => sum + val, 0) / n;
        const ss = sorted.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);

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
            isNormal: pValue > 0.05
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

    normalInverse(p) {
        if (p <= 0 || p >= 1) {
            throw new Error('p must be between 0 and 1');
        }

        const a0 = -3.969683028665376e+01;
        const a1 = 2.209460984245205e+02;
        const a2 = -2.759285104469687e+02;
        const a3 = 1.383577518672690e+02;
        const a4 = -3.066479806614716e+01;
        const a5 = 2.506628277459239e+00;

        const b1 = -5.447609879822406e+01;
        const b2 = 1.615858368580409e+02;
        const b3 = -1.556989798598866e+02;
        const b4 = 6.680131188771972e+01;
        const b5 = -1.328068155288572e+01;

        if (p > 0.5) {
            return -this.normalInverse(1 - p);
        }

        const q = Math.sqrt(-2 * Math.log(p));
        return (((((a5 * q + a4) * q + a3) * q + a2) * q + a1) * q + a0) /
               ((((b5 * q + b4) * q + b3) * q + b2) * q + b1) * q + 1;
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

    jarqueBeraTest(column) {
        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length < 4) {
            throw new Error('Jarque-Bera test requires at least 4 observations');
        }

        const n = validValues.length;
        const skew = this.skewness(column, false);
        const kurt = this.kurtosis(column, false, true);

        const jb = (n / 6) * (Math.pow(skew, 2) + Math.pow(kurt, 2) / 4);
        const pValue = 1 - this.chiSquareCDF(jb, 2);

        return {
            statistic: jb,
            pValue: pValue,
            skewness: skew,
            kurtosis: kurt,
            isNormal: pValue > 0.05
        };
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

        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return sign * y;
    }

    chiSquareCDF(x, df) {
        if (x <= 0) return 0;

        return this.incompleteGamma(df / 2, x / 2) / this.gamma(df / 2);
    }

    incompleteGamma(a, x) {
        if (x <= 0) return 0;
        if (a <= 0) return 1;

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
        let result = 1;
        const coefficients = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
        ];

        result = coefficients[0];
        for (let i = 1; i < coefficients.length; i++) {
            result += coefficients[i] / (x + i);
        }

        const t = x + coefficients.length - 1.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * result;
    }

    momentCoefficient(column, moment) {
        if (typeof moment !== 'number' || moment < 1) {
            throw new Error('Moment must be a positive integer');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const mean = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
        const momentSum = validValues.reduce((sum, val) => sum + Math.pow(val - mean, moment), 0);

        return momentSum / validValues.length;
    }

    pearsonSkewness(column, mode = 1) {
        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const mean = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
        const variance = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (validValues.length - 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
            return 0;
        }

        if (mode === 1) {
            const sorted = [...validValues].sort((a, b) => a - b);
            const median = sorted.length % 2 === 0 ?
                (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2 :
                sorted[Math.floor(sorted.length / 2)];

            return (mean - median) / stdDev;
        } else if (mode === 2) {
            return 3 * (mean - this.median(validValues)) / stdDev;
        } else {
            throw new Error('Mode must be 1 or 2');
        }
    }

    median(arr) {
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ?
            (sorted[mid - 1] + sorted[mid]) / 2 :
            sorted[mid];
    }
}

export default Shape;