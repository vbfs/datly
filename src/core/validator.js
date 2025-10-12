class Validator {
    validateData(dataset) {
        const errors = [];
        const warnings = [];

        if (!dataset || typeof dataset !== 'object') {
            errors.push('Dataset must be an object');
            return { valid: false, errors, warnings };
        }

        if (!dataset.data || !Array.isArray(dataset.data)) {
            errors.push('Dataset must contain a data array');
        }

        if (!dataset.headers || !Array.isArray(dataset.headers)) {
            errors.push('Dataset must contain a headers array');
        }

        if (dataset.data && dataset.data.length === 0) {
            warnings.push('Dataset is empty');
        }

        if (dataset.data && dataset.headers) {
            const headerSet = new Set(dataset.headers);
            if (headerSet.size !== dataset.headers.length) {
                errors.push('Duplicate column headers found');
            }

            dataset.data.forEach((row, index) => {
                const rowKeys = Object.keys(row);
                const missingHeaders = dataset.headers.filter(h => !rowKeys.includes(h));
                const extraKeys = rowKeys.filter(k => !dataset.headers.includes(k));

                if (missingHeaders.length > 0) {
                    warnings.push(`Row ${index}: Missing columns: ${missingHeaders.join(', ')}`);
                }

                if (extraKeys.length > 0) {
                    warnings.push(`Row ${index}: Extra columns: ${extraKeys.join(', ')}`);
                }
            });
        }

        return {
            valid: errors.length === 0,
            errors,
            warnings
        };
    }

    validateNumericColumn(column) {
        if (!Array.isArray(column)) {
            throw new Error('Column must be an array');
        }

        const numericValues = column.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        );

        if (numericValues.length === 0) {
            throw new Error('Column contains no valid numeric values');
        }

        return {
            valid: true,
            validCount: numericValues.length,
            invalidCount: column.length - numericValues.length,
            cleanData: numericValues
        };
    }

    validateSampleSize(sample, minSize = 2) {
        if (!Array.isArray(sample)) {
            throw new Error('Sample must be an array');
        }

        if (sample.length < minSize) {
            throw new Error(`Sample size (${sample.length}) must be at least ${minSize}`);
        }

        return true;
    }

    validateConfidenceLevel(confidence) {
        if (typeof confidence !== 'number' || confidence <= 0 || confidence >= 1) {
            throw new Error('Confidence level must be a number between 0 and 1');
        }
        return true;
    }

    validateCorrelationInputs(col1, col2) {
        this.validateNumericColumn(col1);
        this.validateNumericColumn(col2);

        if (col1.length !== col2.length) {
            throw new Error('Columns must have the same length');
        }

        if (col1.length < 3) {
            throw new Error('Need at least 3 paired observations for correlation');
        }

        return true;
    }

    validateRegressionInputs(x, y) {
        this.validateNumericColumn(x);
        this.validateNumericColumn(y);

        if (x.length !== y.length) {
            throw new Error('X and Y arrays must have the same length');
        }

        if (x.length < 3) {
            throw new Error('Need at least 3 data points for regression');
        }

        const xVariance = this.calculateVariance(x);
        if (xVariance === 0) {
            throw new Error('X values must have non-zero variance');
        }

        return true;
    }

    calculateVariance(arr) {
        const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
        return arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (arr.length - 1);
    }

    validateGroupsForANOVA(groups) {
        if (!Array.isArray(groups) || groups.length < 2) {
            throw new Error('ANOVA requires at least 2 groups');
        }

        groups.forEach((group, index) => {
            if (!Array.isArray(group)) {
                throw new Error(`Group ${index} must be an array`);
            }

            this.validateSampleSize(group, 2);
            this.validateNumericColumn(group);
        });

        return true;
    }

    validateContingencyTable(col1, col2) {
        if (!Array.isArray(col1) || !Array.isArray(col2)) {
            throw new Error('Both columns must be arrays');
        }

        if (col1.length !== col2.length) {
            throw new Error('Columns must have the same length');
        }

        if (col1.length < 5) {
            throw new Error('Need at least 5 observations for chi-square test');
        }

        return true;
    }

    isInteger(value) {
        return typeof value === 'number' && Number.isInteger(value);
    }

    isPositive(value) {
        return typeof value === 'number' && value > 0;
    }

    isInRange(value, min, max) {
        return typeof value === 'number' && value >= min && value <= max;
    }

    hasMinimumObservations(data, minimum) {
        return Array.isArray(data) && data.length >= minimum;
    }

    checkForConstantValues(column) {
        const uniqueValues = new Set(column);
        return uniqueValues.size === 1;
    }

    validateHypothesisTestInputs(sample1, sample2, testType) {
        this.validateSampleSize(sample1, 2);

        if (testType === 'two-sample' || testType === 'paired') {
            this.validateSampleSize(sample2, 2);

            if (testType === 'paired' && sample1.length !== sample2.length) {
                throw new Error('Paired samples must have the same length');
            }
        }

        this.validateNumericColumn(sample1);
        if (sample2) {
            this.validateNumericColumn(sample2);
        }

        return true;
    }
}

export default Validator;