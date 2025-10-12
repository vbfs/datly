class HypothesisTesting {
    tTest(sample1, sample2, type = 'two-sample', alpha = 0.05) {
        const validSample1 = sample1.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample1.length < 2) {
            throw new Error('Sample 1 must have at least 2 valid values');
        }

        switch (type) {
            case 'one-sample':
                return this.oneSampleTTest(validSample1, sample2, alpha);
            case 'two-sample':
                const validSample2 = sample2.filter(val =>
                    typeof val === 'number' && !isNaN(val) && isFinite(val)
                );
                if (validSample2.length < 2) {
                    throw new Error('Sample 2 must have at least 2 valid values');
                }
                return this.twoSampleTTest(validSample1, validSample2, alpha);
            case 'paired':
                if (sample1.length !== sample2.length) {
                    throw new Error('Paired samples must have the same length');
                }
                return this.pairedTTest(validSample1, sample2, alpha);
            default:
                throw new Error('Unknown t-test type. Use: one-sample, two-sample, or paired');
        }
    }

    oneSampleTTest(sample, mu0, alpha = 0.05) {
        const n = sample.length;
        const mean = sample.reduce((sum, val) => sum + val, 0) / n;
        const variance = sample.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
        const stdError = Math.sqrt(variance / n);

        if (stdError === 0) {
            throw new Error('Cannot perform t-test when standard error is zero');
        }

        const tStat = (mean - mu0) / stdError;
        const df = n - 1;
        const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), df));

        return {
            type: 'one-sample',
            statistic: tStat,
            pValue: pValue,
            degreesOfFreedom: df,
            sampleMean: mean,
            hypothesizedMean: mu0,
            standardError: stdError,
            criticalValue: this.tInverse(1 - alpha / 2, df),
            significant: pValue < alpha,
            alpha: alpha
        };
    }

    twoSampleTTest(sample1, sample2, alpha = 0.05, equalVariances = false) {
        const n1 = sample1.length;
        const n2 = sample2.length;

        const mean1 = sample1.reduce((sum, val) => sum + val, 0) / n1;
        const mean2 = sample2.reduce((sum, val) => sum + val, 0) / n2;

        const var1 = sample1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / (n1 - 1);
        const var2 = sample2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / (n2 - 1);

        let tStat, df, stdError;

        if (equalVariances) {
            const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
            stdError = Math.sqrt(pooledVar * (1/n1 + 1/n2));
            df = n1 + n2 - 2;
        } else {
            stdError = Math.sqrt(var1/n1 + var2/n2);
            df = Math.pow(var1/n1 + var2/n2, 2) /
                (Math.pow(var1/n1, 2)/(n1-1) + Math.pow(var2/n2, 2)/(n2-1));
        }

        if (stdError === 0) {
            throw new Error('Cannot perform t-test when standard error is zero');
        }

        tStat = (mean1 - mean2) / stdError;
        const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), df));

        return {
            type: 'two-sample',
            statistic: tStat,
            pValue: pValue,
            degreesOfFreedom: df,
            sample1Mean: mean1,
            sample2Mean: mean2,
            meanDifference: mean1 - mean2,
            standardError: stdError,
            equalVariances: equalVariances,
            criticalValue: this.tInverse(1 - alpha / 2, df),
            significant: pValue < alpha,
            alpha: alpha
        };
    }

    pairedTTest(sample1, sample2, alpha = 0.05) {
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

        return this.oneSampleTTest(differences, 0, alpha);
    }

    zTest(sample, populationMean, populationStd, alpha = 0.05) {
        const validSample = sample.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample.length === 0) {
            throw new Error('No valid numeric values found');
        }

        if (populationStd <= 0) {
            throw new Error('Population standard deviation must be positive');
        }

        const n = validSample.length;
        const sampleMean = validSample.reduce((sum, val) => sum + val, 0) / n;
        const standardError = populationStd / Math.sqrt(n);
        const zStat = (sampleMean - populationMean) / standardError;
        const pValue = 2 * (1 - this.normalCDF(Math.abs(zStat)));

        return {
            type: 'z-test',
            statistic: zStat,
            pValue: pValue,
            sampleMean: sampleMean,
            populationMean: populationMean,
            populationStd: populationStd,
            sampleSize: n,
            standardError: standardError,
            criticalValue: this.normalInverse(1 - alpha / 2),
            significant: pValue < alpha,
            alpha: alpha
        };
    }

    anovaTest(groups, alpha = 0.05) {
        if (!Array.isArray(groups) || groups.length < 2) {
            throw new Error('ANOVA requires at least 2 groups');
        }

        const validGroups = groups.map(group =>
            group.filter(val => typeof val === 'number' && !isNaN(val) && isFinite(val))
        );

        validGroups.forEach((group, index) => {
            if (group.length < 2) {
                throw new Error(`Group ${index + 1} must have at least 2 valid values`);
            }
        });

        const k = validGroups.length;
        const groupMeans = validGroups.map(group =>
            group.reduce((sum, val) => sum + val, 0) / group.length
        );
        const groupSizes = validGroups.map(group => group.length);
        const totalN = groupSizes.reduce((sum, n) => sum + n, 0);

        const allValues = validGroups.flat();
        const grandMean = allValues.reduce((sum, val) => sum + val, 0) / totalN;

        const ssb = validGroups.reduce((sum, group, index) => {
            return sum + groupSizes[index] * Math.pow(groupMeans[index] - grandMean, 2);
        }, 0);

        const ssw = validGroups.reduce((sum, group, index) => {
            const groupMean = groupMeans[index];
            return sum + group.reduce((groupSum, val) =>
                groupSum + Math.pow(val - groupMean, 2), 0
            );
        }, 0);

        const dfBetween = k - 1;
        const dfWithin = totalN - k;
        const msb = ssb / dfBetween;
        const msw = ssw / dfWithin;

        if (msw === 0) {
            throw new Error('Cannot perform ANOVA when within-group variance is zero');
        }

        const fStat = msb / msw;
        const pValue = 1 - this.fCDF(fStat, dfBetween, dfWithin);

        return {
            type: 'one-way-anova',
            statistic: fStat,
            pValue: pValue,
            dfBetween: dfBetween,
            dfWithin: dfWithin,
            sumOfSquaresBetween: ssb,
            sumOfSquaresWithin: ssw,
            meanSquareBetween: msb,
            meanSquareWithin: msw,
            groupMeans: groupMeans,
            grandMean: grandMean,
            significant: pValue < alpha,
            alpha: alpha
        };
    }

    chiSquareTest(col1, col2, alpha = 0.05) {
        if (col1.length !== col2.length) {
            throw new Error('Columns must have the same length');
        }

        const contingencyResult = this.createContingencyTable(col1, col2);
        const { table, totals, rows, columns } = contingencyResult;

        let chiSquareStat = 0;
        const expected = {};

        rows.forEach(row => {
            expected[row] = {};
            columns.forEach(col => {
                const expectedFreq = (totals.row[row] * totals.col[col]) / totals.grand;
                expected[row][col] = expectedFreq;

                if (expectedFreq < 5) {
                    console.warn(`Warning: Expected frequency (${expectedFreq.toFixed(2)}) is less than 5`);
                }

                const observed = table[row][col];
                chiSquareStat += Math.pow(observed - expectedFreq, 2) / expectedFreq;
            });
        });

        const df = (rows.length - 1) * (columns.length - 1);
        const pValue = 1 - this.chiSquareCDF(chiSquareStat, df);

        return {
            type: 'chi-square-independence',
            statistic: chiSquareStat,
            pValue: pValue,
            degreesOfFreedom: df,
            observed: table,
            expected: expected,
            significant: pValue < alpha,
            alpha: alpha,
            cramersV: this.cramersV(chiSquareStat, totals.grand, Math.min(rows.length, columns.length))
        };
    }

    createContingencyTable(col1, col2) {
        const uniqueCol1 = [...new Set(col1)];
        const uniqueCol2 = [...new Set(col2)];

        const table = {};
        const totals = { row: {}, col: {}, grand: 0 };

        uniqueCol1.forEach(val1 => {
            table[val1] = {};
            totals.row[val1] = 0;
        });

        uniqueCol2.forEach(val2 => {
            totals.col[val2] = 0;
        });

        for (let i = 0; i < col1.length; i++) {
            const val1 = col1[i];
            const val2 = col2[i];

            if (!table[val1][val2]) {
                table[val1][val2] = 0;
            }

            table[val1][val2]++;
            totals.row[val1]++;
            totals.col[val2]++;
            totals.grand++;
        }

        uniqueCol1.forEach(val1 => {
            uniqueCol2.forEach(val2 => {
                if (!table[val1][val2]) {
                    table[val1][val2] = 0;
                }
            });
        });

        return { table, totals, rows: uniqueCol1, columns: uniqueCol2 };
    }

    cramersV(chiSquare, n, minDimension) {
        return Math.sqrt(chiSquare / (n * (minDimension - 1)));
    }

    mannWhitneyTest(sample1, sample2, alpha = 0.05) {
        const validSample1 = sample1.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );
        const validSample2 = sample2.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validSample1.length === 0 || validSample2.length === 0) {
            throw new Error('Both samples must contain valid numeric values');
        }

        const n1 = validSample1.length;
        const n2 = validSample2.length;

        const combined = [
            ...validSample1.map(val => ({ value: val, group: 1 })),
            ...validSample2.map(val => ({ value: val, group: 2 }))
        ];

        combined.sort((a, b) => a.value - b.value);

        let currentRank = 1;
        for (let i = 0; i < combined.length; i++) {
            const tiedValues = [combined[i]];
            while (i + 1 < combined.length && combined[i + 1].value === combined[i].value) {
                i++;
                tiedValues.push(combined[i]);
            }

            const avgRank = (currentRank + currentRank + tiedValues.length - 1) / 2;
            tiedValues.forEach(item => item.rank = avgRank);
            currentRank += tiedValues.length;
        }

        const r1 = combined.filter(item => item.group === 1)
                           .reduce((sum, item) => sum + item.rank, 0);

        const u1 = r1 - (n1 * (n1 + 1)) / 2;
        const u2 = n1 * n2 - u1;
        const uStat = Math.min(u1, u2);

        const meanU = (n1 * n2) / 2;
        const stdU = Math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12);
        const zStat = (uStat - meanU) / stdU;
        const pValue = 2 * (1 - this.normalCDF(Math.abs(zStat)));

        return {
            type: 'mann-whitney-u',
            statistic: uStat,
            u1: u1,
            u2: u2,
            zStatistic: zStat,
            pValue: pValue,
            sample1Size: n1,
            sample2Size: n2,
            significant: pValue < alpha,
            alpha: alpha
        };
    }

    tCDF(t, df) {
        const x = df / (t * t + df);
        return 1 - 0.5 * this.incompleteBeta(df / 2, 0.5, x);
    }

    tInverse(p, df) {
        if (p <= 0 || p >= 1) return NaN;

        let t = this.normalInverse(p);
        const c1 = t / 4;
        const c2 = (5 * t * t * t + 16 * t) / 96;
        const c3 = (3 * t * t * t * t * t + 19 * t * t * t + 17 * t) / 384;

        return t + c1 / df + c2 / (df * df) + c3 / (df * df * df);
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

    fCDF(f, df1, df2) {
        const x = df2 / (df2 + df1 * f);
        return 1 - this.incompleteBeta(df2 / 2, df1 / 2, x);
    }

    chiSquareCDF(x, df) {
        if (x <= 0) return 0;
        return this.incompleteGamma(df / 2, x / 2) / this.gamma(df / 2);
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
        return Math.exp(this.logGamma(x));
    }
}

export default HypothesisTesting;