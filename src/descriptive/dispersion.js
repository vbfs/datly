class Dispersion {
    variance(column, sample = true) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        if (sample && validValues.length < 2) {
            throw new Error('Sample variance requires at least 2 values');
        }

        const mean = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
        const sumSquaredDiff = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);

        const denominator = sample ? validValues.length - 1 : validValues.length;
        return sumSquaredDiff / denominator;
    }

    standardDeviation(column, sample = true) {
        return Math.sqrt(this.variance(column, sample));
    }

    range(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const min = Math.min(...validValues);
        const max = Math.max(...validValues);

        return {
            range: max - min,
            min: min,
            max: max
        };
    }

    interquartileRange(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const sorted = validValues.sort((a, b) => a - b);
        const q1 = this.quantile(sorted, 0.25);
        const q3 = this.quantile(sorted, 0.75);

        return {
            iqr: q3 - q1,
            q1: q1,
            q3: q3
        };
    }

    quantile(sortedArray, q) {
        const index = (sortedArray.length - 1) * q;
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index % 1;

        if (lower === upper) {
            return sortedArray[lower];
        }

        return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
    }

    coefficientOfVariation(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const mean = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;

        if (mean === 0) {
            throw new Error('Cannot calculate coefficient of variation when mean is zero');
        }

        const std = this.standardDeviation(column);
        return {
            cv: std / Math.abs(mean),
            cvPercent: (std / Math.abs(mean)) * 100
        };
    }

    meanAbsoluteDeviation(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const mean = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
        const sumAbsDiff = validValues.reduce((sum, val) => sum + Math.abs(val - mean), 0);

        return {
            mad: sumAbsDiff / validValues.length,
            mean: mean
        };
    }

    medianAbsoluteDeviation(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const sorted = validValues.sort((a, b) => a - b);
        const median = this.quantile(sorted, 0.5);
        const deviations = validValues.map(val => Math.abs(val - median));
        const sortedDeviations = deviations.sort((a, b) => a - b);

        return {
            mad: this.quantile(sortedDeviations, 0.5),
            median: median
        };
    }

    standardError(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const std = this.standardDeviation(column);
        return std / Math.sqrt(validValues.length);
    }

    quartileCoefficient(column) {
        const iqrResult = this.interquartileRange(column);
        const q1 = iqrResult.q1;
        const q3 = iqrResult.q3;

        if (q1 + q3 === 0) {
            throw new Error('Cannot calculate quartile coefficient when Q1 + Q3 = 0');
        }

        return (q3 - q1) / (q3 + q1);
    }

    percentileRange(column, lowerPercentile, upperPercentile) {
        if (lowerPercentile >= upperPercentile) {
            throw new Error('Lower percentile must be less than upper percentile');
        }

        if (lowerPercentile < 0 || upperPercentile > 100) {
            throw new Error('Percentiles must be between 0 and 100');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const sorted = validValues.sort((a, b) => a - b);
        const lowerValue = this.quantile(sorted, lowerPercentile / 100);
        const upperValue = this.quantile(sorted, upperPercentile / 100);

        return {
            range: upperValue - lowerValue,
            lowerValue: lowerValue,
            upperValue: upperValue,
            lowerPercentile: lowerPercentile,
            upperPercentile: upperPercentile
        };
    }

    giniCoefficient(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val) && val >= 0
        );

        if (validValues.length === 0) {
            throw new Error('Gini coefficient requires non-negative numeric values');
        }

        const sorted = validValues.sort((a, b) => a - b);
        const n = sorted.length;
        const mean = sorted.reduce((sum, val) => sum + val, 0) / n;

        if (mean === 0) {
            return 0;
        }

        let numerator = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                numerator += Math.abs(sorted[i] - sorted[j]);
            }
        }

        return numerator / (2 * n * n * mean);
    }

    robustScale(column) {
        const iqrResult = this.interquartileRange(column);
        const median = this.quantile(column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        ).sort((a, b) => a - b), 0.5);

        const scaled = column.map(val => {
            if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                return iqrResult.iqr !== 0 ? (val - median) / iqrResult.iqr : 0;
            }
            return null;
        });

        return {
            scaledValues: scaled,
            median: median,
            iqr: iqrResult.iqr
        };
    }
}

export default Dispersion;