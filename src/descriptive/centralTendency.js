class CentralTendency {
    mean(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        return validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
    }

    median(column) {
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
        const middle = Math.floor(sorted.length / 2);

        return sorted.length % 2 === 0 ?
            (sorted[middle - 1] + sorted[middle]) / 2 :
            sorted[middle];
    }

    mode(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const frequency = {};
        let maxFreq = 0;

        column.forEach(value => {
            const key = value === null || value === undefined ? 'null' : String(value);
            frequency[key] = (frequency[key] || 0) + 1;
            maxFreq = Math.max(maxFreq, frequency[key]);
        });

        const modes = Object.entries(frequency)
            .filter(([_, freq]) => freq === maxFreq)
            .map(([value, _]) => value === 'null' ? null : this.parseValue(value));

        return {
            values: modes,
            frequency: maxFreq,
            isMultimodal: modes.length > 1,
            isUniform: maxFreq === 1 && Object.keys(frequency).length === column.length
        };
    }

    parseValue(str) {
        if (/^-?\d+$/.test(str)) return parseInt(str, 10);
        if (/^-?\d*\.\d+$/.test(str)) return parseFloat(str);
        if (str === 'true') return true;
        if (str === 'false') return false;
        return str;
    }

    geometricMean(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val) && val > 0
        );

        if (validValues.length === 0) {
            throw new Error('Geometric mean requires positive numeric values');
        }

        const logSum = validValues.reduce((sum, val) => sum + Math.log(val), 0);
        return Math.exp(logSum / validValues.length);
    }

    harmonicMean(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val) && val > 0
        );

        if (validValues.length === 0) {
            throw new Error('Harmonic mean requires positive numeric values');
        }

        const reciprocalSum = validValues.reduce((sum, val) => sum + (1 / val), 0);
        return validValues.length / reciprocalSum;
    }

    trimmedMean(column, percentage) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        if (typeof percentage !== 'number' || percentage < 0 || percentage >= 50) {
            throw new Error('Percentage must be between 0 and 50');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const sorted = validValues.sort((a, b) => a - b);
        const trimCount = Math.floor((percentage / 100) * sorted.length);
        const trimmed = sorted.slice(trimCount, sorted.length - trimCount);

        if (trimmed.length === 0) {
            throw new Error('Too much trimming - no values remain');
        }

        return trimmed.reduce((sum, val) => sum + val, 0) / trimmed.length;
    }

    quadraticMean(column) {
        if (!Array.isArray(column) || column.length === 0) {
            throw new Error('Column must be a non-empty array');
        }

        const validValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (validValues.length === 0) {
            throw new Error('No valid numeric values found');
        }

        const sumOfSquares = validValues.reduce((sum, val) => sum + val * val, 0);
        return Math.sqrt(sumOfSquares / validValues.length);
    }

    weightedMean(values, weights) {
        if (!Array.isArray(values) || !Array.isArray(weights)) {
            throw new Error('Values and weights must be arrays');
        }

        if (values.length !== weights.length) {
            throw new Error('Values and weights must have the same length');
        }

        if (values.length === 0) {
            throw new Error('Arrays must not be empty');
        }

        const validPairs = [];
        for (let i = 0; i < values.length; i++) {
            if (typeof values[i] === 'number' && typeof weights[i] === 'number' &&
                !isNaN(values[i]) && !isNaN(weights[i]) &&
                isFinite(values[i]) && isFinite(weights[i]) && weights[i] >= 0) {
                validPairs.push({ value: values[i], weight: weights[i] });
            }
        }

        if (validPairs.length === 0) {
            throw new Error('No valid value-weight pairs found');
        }

        const totalWeight = validPairs.reduce((sum, pair) => sum + pair.weight, 0);

        if (totalWeight === 0) {
            throw new Error('Total weight cannot be zero');
        }

        const weightedSum = validPairs.reduce((sum, pair) => sum + pair.value * pair.weight, 0);
        return weightedSum / totalWeight;
    }

    midrange(column) {
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

        return (min + max) / 2;
    }
}

export default CentralTendency;