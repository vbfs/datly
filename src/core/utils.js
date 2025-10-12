class Utils {
    detectOutliers(column, method = 'iqr') {
        const sortedData = [...column].sort((a, b) => a - b);
        const outliers = [];
        const indices = [];

        switch (method) {
            case 'iqr':
                const q1 = this.quantile(sortedData, 0.25);
                const q3 = this.quantile(sortedData, 0.75);
                const iqr = q3 - q1;
                const lowerBound = q1 - 1.5 * iqr;
                const upperBound = q3 + 1.5 * iqr;

                column.forEach((value, index) => {
                    if (value < lowerBound || value > upperBound) {
                        outliers.push(value);
                        indices.push(index);
                    }
                });
                break;

            case 'zscore':
                const mean = this.mean(column);
                const std = this.standardDeviation(column);

                column.forEach((value, index) => {
                    const zscore = Math.abs((value - mean) / std);
                    if (zscore > 3) {
                        outliers.push(value);
                        indices.push(index);
                    }
                });
                break;

            case 'modified_zscore':
                const median = this.median(column);
                const deviations = column.map(x => Math.abs(x - median));
                const mad = this.median(deviations);

                column.forEach((value, index) => {
                    const modifiedZScore = 0.6745 * (value - median) / mad;
                    if (Math.abs(modifiedZScore) > 3.5) {
                        outliers.push(value);
                        indices.push(index);
                    }
                });
                break;

            default:
                throw new Error(`Unknown outlier detection method: ${method}`);
        }

        return {
            outliers,
            indices,
            count: outliers.length,
            percentage: (outliers.length / column.length) * 100
        };
    }

    frequencyTable(column) {
        const frequencies = {};
        const total = column.length;

        column.forEach(value => {
            const key = value === null || value === undefined ? 'null' : String(value);
            frequencies[key] = (frequencies[key] || 0) + 1;
        });

        const result = Object.entries(frequencies).map(([value, count]) => ({
            value: value === 'null' ? null : value,
            frequency: count,
            relativeFrequency: count / total,
            percentage: (count / total) * 100
        }));

        return result.sort((a, b) => b.frequency - a.frequency);
    }

    groupBy(dataset, column, aggregation) {
        const groups = {};

        dataset.data.forEach(row => {
            const key = row[column];
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(row);
        });

        const result = {};
        Object.entries(groups).forEach(([key, rows]) => {
            result[key] = {
                count: rows.length,
                data: rows
            };

            if (aggregation && typeof aggregation === 'object') {
                Object.entries(aggregation).forEach(([targetCol, func]) => {
                    const values = rows.map(row => row[targetCol]).filter(v =>
                        typeof v === 'number' && !isNaN(v)
                    );

                    if (values.length > 0) {
                        result[key][`${func}_${targetCol}`] = this.applyAggregation(values, func);
                    }
                });
            }
        });

        return result;
    }

    applyAggregation(values, func) {
        switch (func) {
            case 'mean': return this.mean(values);
            case 'median': return this.median(values);
            case 'sum': return values.reduce((a, b) => a + b, 0);
            case 'min': return Math.min(...values);
            case 'max': return Math.max(...values);
            case 'std': return this.standardDeviation(values);
            case 'var': return this.variance(values);
            case 'count': return values.length;
            default: throw new Error(`Unknown aggregation function: ${func}`);
        }
    }

    sample(dataset, size, method = 'random') {
        if (size >= dataset.length) {
            return { ...dataset };
        }

        let sampledData;

        switch (method) {
            case 'random':
                const indices = this.randomSample(dataset.length, size);
                sampledData = indices.map(i => dataset.data[i]);
                break;

            case 'systematic':
                const interval = Math.floor(dataset.length / size);
                sampledData = [];
                for (let i = 0; i < size; i++) {
                    sampledData.push(dataset.data[i * interval]);
                }
                break;

            case 'first':
                sampledData = dataset.data.slice(0, size);
                break;

            case 'last':
                sampledData = dataset.data.slice(-size);
                break;

            default:
                throw new Error(`Unknown sampling method: ${method}`);
        }

        return {
            ...dataset,
            data: sampledData,
            length: sampledData.length
        };
    }

    randomSample(populationSize, sampleSize) {
        const indices = Array.from({ length: populationSize }, (_, i) => i);
        const sample = [];

        for (let i = 0; i < sampleSize; i++) {
            const randomIndex = Math.floor(Math.random() * indices.length);
            sample.push(indices.splice(randomIndex, 1)[0]);
        }

        return sample;
    }

    bootstrap(sample, statistic, iterations = 1000) {
        const bootstrapStats = [];

        for (let i = 0; i < iterations; i++) {
            const bootstrapSample = [];
            for (let j = 0; j < sample.length; j++) {
                const randomIndex = Math.floor(Math.random() * sample.length);
                bootstrapSample.push(sample[randomIndex]);
            }

            const stat = this.applyStatistic(bootstrapSample, statistic);
            bootstrapStats.push(stat);
        }

        return {
            bootstrapStats: bootstrapStats.sort((a, b) => a - b),
            mean: this.mean(bootstrapStats),
            standardError: this.standardDeviation(bootstrapStats),
            confidenceInterval: {
                lower: this.quantile(bootstrapStats, 0.025),
                upper: this.quantile(bootstrapStats, 0.975)
            }
        };
    }

    applyStatistic(sample, statistic) {
        switch (statistic) {
            case 'mean': return this.mean(sample);
            case 'median': return this.median(sample);
            case 'std': return this.standardDeviation(sample);
            case 'var': return this.variance(sample);
            default:
                if (typeof statistic === 'function') {
                    return statistic(sample);
                }
                throw new Error(`Unknown statistic: ${statistic}`);
        }
    }

    contingencyTable(col1, col2) {
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

    mean(arr) {
        return arr.reduce((sum, val) => sum + val, 0) / arr.length;
    }

    median(arr) {
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ?
            (sorted[mid - 1] + sorted[mid]) / 2 :
            sorted[mid];
    }

    quantile(arr, q) {
        const sorted = [...arr].sort((a, b) => a - b);
        const index = (sorted.length - 1) * q;
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index % 1;

        if (lower === upper) {
            return sorted[lower];
        }

        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    standardDeviation(arr) {
        return Math.sqrt(this.variance(arr));
    }

    variance(arr) {
        const mean = this.mean(arr);
        return arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (arr.length - 1);
    }

    round(value, decimals = 4) {
        return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
    }

    formatNumber(value, decimals = 4) {
        if (typeof value !== 'number') return value;
        return this.round(value, decimals);
    }
}

export default Utils;