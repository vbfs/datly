class Position {
    quantile(column, q) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        if (typeof q !== 'number' || q < 0 || q > 1) {
            throw new Error('Quantile must be between 0 and 1');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const sorted = validValues.sort((a, b) => a - b);
        const index = (sorted.length - 1) * q;
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index % 1;

        if (lower === upper) {
            return sorted[lower];
        }

        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    percentile(column, p) {
        if (typeof p !== 'number' || p < 0 || p > 100) {
            throw new Error('Percentile must be between 0 and 100');
        }

        return this.quantile(column, p / 100);
    }

    quartiles(column) {
        return {
            q1: this.quantile(column, 0.25),
            q2: this.quantile(column, 0.5),
            q3: this.quantile(column, 0.75),
            iqr: this.quantile(column, 0.75) - this.quantile(column, 0.25)
        };
    }

    quintiles(column) {
        return {
            q1: this.quantile(column, 0.2),
            q2: this.quantile(column, 0.4),
            q3: this.quantile(column, 0.6),
            q4: this.quantile(column, 0.8)
        };
    }

    deciles(column) {
        const deciles = {};
        for (let i = 1; i <= 9; i++) {
            deciles[`d${i}`] = this.quantile(column, i / 10);
        }
        return deciles;
    }

    percentileRank(column, value) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        if (typeof value !== 'number' || !isFinite(value)) {
            throw new Error('Value must be a finite number');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const countBelow = validValues.filter(val => val < value).length;
        const countEqual = validValues.filter(val => val === value).length;

        return ((countBelow + 0.5 * countEqual) / validValues.length) * 100;
    }

    zScore(column, value) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        if (typeof value !== 'number' || !isFinite(value)) {
            throw new Error('Value must be a finite number');
        }

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
            throw new Error('Cannot calculate z-score when standard deviation is zero');
        }

        return (value - mean) / stdDev;
    }

    boxplotStats(column) {
        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const sorted = validValues.sort((a, b) => a - b);
        const q1 = this.quantile(sorted, 0.25);
        const q2 = this.quantile(sorted, 0.5);
        const q3 = this.quantile(sorted, 0.75);
        const iqr = q3 - q1;

        const lowerFence = q1 - 1.5 * iqr;
        const upperFence = q3 + 1.5 * iqr;

        const outliers = sorted.filter(val => val < lowerFence || val > upperFence);
        const inliers = sorted.filter(val => val >= lowerFence && val <= upperFence);

        return {
            min: Math.min(...inliers),
            q1: q1,
            median: q2,
            q3: q3,
            max: Math.max(...inliers),
            iqr: iqr,
            lowerFence: lowerFence,
            upperFence: upperFence,
            outliers: outliers,
            outlierCount: outliers.length
        };
    }

    fiveNumberSummary(column) {
        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        return {
            minimum: Math.min(...validValues),
            q1: this.quantile(validValues, 0.25),
            median: this.quantile(validValues, 0.5),
            q3: this.quantile(validValues, 0.75),
            maximum: Math.max(...validValues)
        };
    }

    rank(column, method = 'average') {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validIndices = [];
        const validValues = [];

        column.forEach((val, index) => {
            if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                validIndices.push(index);
                validValues.push({ value: val, originalIndex: index });
            }
        });

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        validValues.sort((a, b) => a.value - b.value);

        const ranks = new Array(column.length).fill(null);

        let currentRank = 1;
        for (let i = 0; i < validValues.length; i++) {
            const currentValue = validValues[i].value;
            const tiedIndices = [i];

            while (i + 1 < validValues.length && validValues[i + 1].value === currentValue) {
                i++;
                tiedIndices.push(i);
            }

            let assignedRank;
            switch (method) {
                case 'average':
                    assignedRank = (currentRank + currentRank + tiedIndices.length - 1) / 2;
                    break;
                case 'min':
                    assignedRank = currentRank;
                    break;
                case 'max':
                    assignedRank = currentRank + tiedIndices.length - 1;
                    break;
                case 'first':
                    tiedIndices.forEach((idx, pos) => {
                        ranks[validValues[idx].originalIndex] = currentRank + pos;
                    });
                    currentRank += tiedIndices.length;
                    continue;
                default:
                    throw new Error('Unknown ranking method. Use: average, min, max, or first');
            }

            tiedIndices.forEach(idx => {
                ranks[validValues[idx].originalIndex] = assignedRank;
            });

            currentRank += tiedIndices.length;
        }

        return ranks;
    }

    normalizedRank(column) {
        const ranks = this.rank(column);
        const validRanks = ranks.filter(rank => rank !== null);
        const maxRank = Math.max(...validRanks);

        return ranks.map(rank => rank !== null ? (rank - 1) / (maxRank - 1) : null);
    }

    standardizedValues(column) {
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
            return column.map(val =>
                typeof val === 'number' && !isNaN(val) && isFinite(val) ? 0 : null
            );
        }

        return column.map(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val) ?
            (val - mean) / stdDev : null
        );
    }
}

export default Position;