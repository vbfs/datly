class ReportGenerator {
    summary(dataset) {
        if (!dataset || !dataset.data || !dataset.headers) {
            throw new Error('Invalid dataset format');
        }

        const basicInfo = this.getBasicInfo(dataset);
        const columnAnalysis = this.analyzeColumns(dataset);
        const dataQuality = this.assessDataQuality(dataset);
        const distributions = this.analyzeDistributions(dataset);
        const relationships = this.analyzeRelationships(dataset);
        const insights = this.generateKeyInsights(dataset, columnAnalysis, relationships);

        return {
            title: 'Statistical Summary Report',
            generatedAt: new Date().toISOString(),
            basicInfo: basicInfo,
            columnAnalysis: columnAnalysis,
            dataQuality: dataQuality,
            distributions: distributions,
            relationships: relationships,
            keyInsights: insights,
            recommendations: this.generateRecommendations(dataQuality, columnAnalysis, relationships)
        };
    }

    getBasicInfo(dataset) {
        return {
            totalRows: dataset.length,
            totalColumns: dataset.columns,
            headers: dataset.headers,
            memoryFootprint: this.estimateMemoryFootprint(dataset),
            dataTypes: this.getDataTypes(dataset)
        };
    }

    analyzeColumns(dataset) {
        const analysis = {};

        dataset.headers.forEach(header => {
            const column = dataset.data.map(row => row[header]);
            const columnType = this.inferColumnType(column);

            analysis[header] = {
                type: columnType,
                totalCount: column.length,
                validCount: this.getValidCount(column),
                nullCount: this.getNullCount(column),
                uniqueCount: this.getUniqueCount(column),
                nullPercentage: this.getNullPercentage(column),
                ...this.getTypeSpecificAnalysis(column, columnType)
            };
        });

        return analysis;
    }

    getTypeSpecificAnalysis(column, type) {
        const validValues = column.filter(val => val !== null && val !== undefined);

        if (type === 'numeric') {
            const numericValues = validValues.filter(val =>
                typeof val === 'number' && !isNaN(val) && isFinite(val)
            );

            if (numericValues.length === 0) return {};

            return {
                min: Math.min(...numericValues),
                max: Math.max(...numericValues),
                mean: this.calculateMean(numericValues),
                median: this.calculateMedian(numericValues),
                standardDeviation: this.calculateStandardDeviation(numericValues),
                variance: this.calculateVariance(numericValues),
                skewness: this.calculateSkewness(numericValues),
                kurtosis: this.calculateKurtosis(numericValues),
                quartiles: this.calculateQuartiles(numericValues),
                outliers: this.detectOutliers(numericValues),
                distribution: this.classifyDistribution(numericValues)
            };
        } else if (type === 'categorical') {
            const frequencyTable = this.calculateFrequencyTable(validValues);
            return {
                categories: frequencyTable,
                mostFrequent: this.getMostFrequent(frequencyTable),
                leastFrequent: this.getLeastFrequent(frequencyTable),
                entropy: this.calculateEntropy(frequencyTable),
                concentration: this.calculateConcentration(frequencyTable)
            };
        } else if (type === 'datetime') {
            const dates = validValues.filter(val => !isNaN(new Date(val).getTime()));
            if (dates.length === 0) return {};

            const timestamps = dates.map(date => new Date(date).getTime());
            return {
                earliest: new Date(Math.min(...timestamps)).toISOString(),
                latest: new Date(Math.max(...timestamps)).toISOString(),
                span: Math.max(...timestamps) - Math.min(...timestamps),
                frequency: this.analyzeDateFrequency(dates)
            };
        }

        return {};
    }

    assessDataQuality(dataset) {
        const issues = [];
        let overallScore = 100;

        const completenessScore = this.assessCompleteness(dataset);
        const consistencyScore = this.assessConsistency(dataset);
        const uniquenessScore = this.assessUniqueness(dataset);
        const validityScore = this.assessValidity(dataset);

        overallScore = (completenessScore + consistencyScore + uniquenessScore + validityScore) / 4;

        if (completenessScore < 80) {
            issues.push({
                type: 'completeness',
                severity: completenessScore < 50 ? 'high' : 'medium',
                description: `${(100 - completenessScore).toFixed(1)}% of data is missing`
            });
        }

        if (consistencyScore < 80) {
            issues.push({
                type: 'consistency',
                severity: consistencyScore < 50 ? 'high' : 'medium',
                description: 'Data consistency issues detected'
            });
        }

        return {
            overallScore: overallScore,
            completenessScore: completenessScore,
            consistencyScore: consistencyScore,
            uniquenessScore: uniquenessScore,
            validityScore: validityScore,
            issues: issues,
            recommendation: this.getQualityRecommendation(overallScore)
        };
    }

    analyzeDistributions(dataset) {
        const distributions = {};

        dataset.headers.forEach(header => {
            const column = dataset.data.map(row => row[header]);
            const validValues = column.filter(val =>
                typeof val === 'number' && !isNaN(val) && isFinite(val)
            );

            if (validValues.length > 5) {
                distributions[header] = {
                    type: this.classifyDistribution(validValues),
                    normalityTest: this.testNormality(validValues),
                    histogram: this.createHistogram(validValues),
                    descriptiveStats: this.getDescriptiveStats(validValues)
                };
            }
        });

        return distributions;
    }

    analyzeRelationships(dataset) {
        const numericColumns = dataset.headers.filter(header => {
            const column = dataset.data.map(row => row[header]);
            const numericCount = column.filter(val =>
                typeof val === 'number' && !isNaN(val) && isFinite(val)
            ).length;
            return numericCount > column.length * 0.5;
        });

        if (numericColumns.length < 2) {
            return { correlations: {}, strongRelationships: [] };
        }

        const correlations = {};
        const strongRelationships = [];

        for (let i = 0; i < numericColumns.length; i++) {
            correlations[numericColumns[i]] = {};
            for (let j = 0; j < numericColumns.length; j++) {
                if (i === j) {
                    correlations[numericColumns[i]][numericColumns[j]] = 1;
                } else {
                    const col1 = dataset.data.map(row => row[numericColumns[i]]);
                    const col2 = dataset.data.map(row => row[numericColumns[j]]);
                    const correlation = this.calculatePearsonCorrelation(col1, col2);
                    correlations[numericColumns[i]][numericColumns[j]] = correlation;

                    if (Math.abs(correlation) > 0.7 && i < j) {
                        strongRelationships.push({
                            variable1: numericColumns[i],
                            variable2: numericColumns[j],
                            correlation: correlation,
                            strength: this.getCorrelationStrength(Math.abs(correlation)),
                            direction: correlation > 0 ? 'positive' : 'negative'
                        });
                    }
                }
            }
        }

        return {
            correlations: correlations,
            strongRelationships: strongRelationships,
            averageCorrelation: this.calculateAverageCorrelation(correlations, numericColumns)
        };
    }

    generateKeyInsights(dataset, columnAnalysis, relationships) {
        const insights = [];

        insights.push(...this.generateDataVolumeInsights(dataset));
        insights.push(...this.generateColumnInsights(columnAnalysis));
        insights.push(...this.generateRelationshipInsights(relationships));
        insights.push(...this.generateDistributionInsights(columnAnalysis));
        insights.push(...this.generateQualityInsights(dataset, columnAnalysis));

        return insights.sort((a, b) => b.importance - a.importance).slice(0, 10);
    }

    generateDataVolumeInsights(dataset) {
        const insights = [];

        if (dataset.length > 10000) {
            insights.push({
                type: 'volume',
                title: 'Large Dataset Detected',
                description: `Dataset contains ${dataset.length.toLocaleString()} rows, which is suitable for robust statistical analysis.`,
                importance: 7,
                actionable: false
            });
        } else if (dataset.length < 30) {
            insights.push({
                type: 'volume',
                title: 'Small Sample Size Warning',
                description: `Dataset has only ${dataset.length} rows. Statistical tests may lack power.`,
                importance: 8,
                actionable: true,
                recommendation: 'Consider collecting more data for reliable statistical inference.'
            });
        }

        if (dataset.columns > 50) {
            insights.push({
                type: 'dimensionality',
                title: 'High-Dimensional Dataset',
                description: `Dataset has ${dataset.columns} columns, which may benefit from dimensionality reduction.`,
                importance: 6,
                actionable: true,
                recommendation: 'Consider feature selection or PCA to reduce dimensionality.'
            });
        }

        return insights;
    }

    generateColumnInsights(columnAnalysis) {
        const insights = [];
        const columns = Object.keys(columnAnalysis);

        const highNullColumns = columns.filter(col =>
            columnAnalysis[col].nullPercentage > 25
        );

        if (highNullColumns.length > 0) {
            insights.push({
                type: 'data_quality',
                title: 'High Missing Data Detected',
                description: `Columns ${highNullColumns.join(', ')} have >25% missing values.`,
                importance: 9,
                actionable: true,
                recommendation: 'Consider imputation strategies or removing these columns.'
            });
        }

        const skewedColumns = columns.filter(col => {
            const analysis = columnAnalysis[col];
            return analysis.skewness && Math.abs(analysis.skewness) > 2;
        });

        if (skewedColumns.length > 0) {
            insights.push({
                type: 'distribution',
                title: 'Highly Skewed Variables Found',
                description: `Columns ${skewedColumns.join(', ')} show extreme skewness.`,
                importance: 7,
                actionable: true,
                recommendation: 'Consider log transformation or other normalization techniques.'
            });
        }

        const constantColumns = columns.filter(col =>
            columnAnalysis[col].uniqueCount === 1
        );

        if (constantColumns.length > 0) {
            insights.push({
                type: 'data_quality',
                title: 'Constant Variables Detected',
                description: `Columns ${constantColumns.join(', ')} have no variation.`,
                importance: 8,
                actionable: true,
                recommendation: 'Remove these columns as they provide no information.'
            });
        }

        return insights;
    }

    generateRelationshipInsights(relationships) {
        const insights = [];

        if (relationships.strongRelationships.length > 0) {
            const strongest = relationships.strongRelationships[0];
            insights.push({
                type: 'correlation',
                title: 'Strong Correlation Found',
                description: `${strongest.variable1} and ${strongest.variable2} have a ${strongest.strength.toLowerCase()} ${strongest.direction} correlation (r = ${strongest.correlation.toFixed(3)}).`,
                importance: 8,
                actionable: true,
                recommendation: 'Investigate this relationship further with regression analysis.'
            });
        }

        const multicollinearPairs = relationships.strongRelationships.filter(rel =>
            Math.abs(rel.correlation) > 0.9
        );

        if (multicollinearPairs.length > 0) {
            insights.push({
                type: 'multicollinearity',
                title: 'Potential Multicollinearity Detected',
                description: `Very high correlations found between some variables.`,
                importance: 7,
                actionable: true,
                recommendation: 'Consider removing redundant variables before modeling.'
            });
        }

        if (relationships.averageCorrelation > 0.5) {
            insights.push({
                type: 'correlation',
                title: 'Generally High Inter-Variable Correlations',
                description: `Average correlation is ${relationships.averageCorrelation.toFixed(3)}, indicating related variables.`,
                importance: 6,
                actionable: false
            });
        }

        return insights;
    }

    generateDistributionInsights(columnAnalysis) {
        const insights = [];
        const numericColumns = Object.keys(columnAnalysis).filter(col =>
            columnAnalysis[col].type === 'numeric'
        );

        const normalColumns = numericColumns.filter(col => {
            const analysis = columnAnalysis[col];
            return analysis.distribution === 'normal' ||
                   (Math.abs(analysis.skewness || 0) < 0.5 && Math.abs(analysis.kurtosis || 0) < 0.5);
        });

        if (normalColumns.length > numericColumns.length * 0.7) {
            insights.push({
                type: 'distribution',
                title: 'Most Variables Normally Distributed',
                description: `${normalColumns.length} out of ${numericColumns.length} numeric variables appear normally distributed.`,
                importance: 6,
                actionable: false
            });
        }

        const outliersColumns = numericColumns.filter(col => {
            const analysis = columnAnalysis[col];
            return analysis.outliers && analysis.outliers.count > 0;
        });

        if (outliersColumns.length > 0) {
            const totalOutliers = outliersColumns.reduce((sum, col) =>
                sum + columnAnalysis[col].outliers.count, 0
            );

            insights.push({
                type: 'outliers',
                title: 'Outliers Detected',
                description: `Found ${totalOutliers} outliers across ${outliersColumns.length} variables.`,
                importance: 7,
                actionable: true,
                recommendation: 'Investigate outliers to determine if they represent errors or genuine extreme values.'
            });
        }

        return insights;
    }

    generateQualityInsights(dataset, columnAnalysis) {
        const insights = [];
        const columns = Object.keys(columnAnalysis);

        const duplicateRows = this.countDuplicateRows(dataset);
        if (duplicateRows > 0) {
            insights.push({
                type: 'data_quality',
                title: 'Duplicate Rows Found',
                description: `Dataset contains ${duplicateRows} duplicate rows.`,
                importance: 8,
                actionable: true,
                recommendation: 'Remove duplicate rows to avoid bias in analysis.'
            });
        }

        const totalMissingCells = columns.reduce((sum, col) =>
            sum + columnAnalysis[col].nullCount, 0
        );
        const totalCells = dataset.length * dataset.columns;
        const missingPercentage = (totalMissingCells / totalCells) * 100;

        if (missingPercentage > 10) {
            insights.push({
                type: 'data_quality',
                title: 'Significant Missing Data',
                description: `${missingPercentage.toFixed(1)}% of all data points are missing.`,
                importance: 9,
                actionable: true,
                recommendation: 'Develop a comprehensive missing data strategy.'
            });
        }

        return insights;
    }

    generateRecommendations(dataQuality, columnAnalysis, relationships) {
        const recommendations = [];

        if (dataQuality.overallScore < 70) {
            recommendations.push({
                priority: 'high',
                category: 'data_cleaning',
                title: 'Improve Data Quality',
                description: 'Address missing values, outliers, and inconsistencies before analysis.',
                steps: [
                    'Handle missing values through imputation or removal',
                    'Investigate and address outliers',
                    'Standardize data formats and categories',
                    'Validate data integrity'
                ]
            });
        }

        if (relationships.strongRelationships.length > 0) {
            recommendations.push({
                priority: 'medium',
                category: 'analysis',
                title: 'Explore Strong Relationships',
                description: 'Investigate detected correlations with deeper analysis.',
                steps: [
                    'Perform regression analysis on highly correlated variables',
                    'Create visualizations to understand relationships',
                    'Test for causality where appropriate'
                ]
            });
        }

        const numericColumns = Object.keys(columnAnalysis).filter(col =>
            columnAnalysis[col].type === 'numeric'
        ).length;

        if (numericColumns > 2) {
            recommendations.push({
                priority: 'low',
                category: 'modeling',
                title: 'Consider Advanced Analytics',
                description: 'Dataset is suitable for machine learning approaches.',
                steps: [
                    'Perform feature selection',
                    'Try different modeling approaches',
                    'Validate models with cross-validation',
                    'Interpret model results'
                ]
            });
        }

        return recommendations;
    }

    inferColumnType(column) {
        const validValues = column.filter(val => val !== null && val !== undefined);
        if (validValues.length === 0) return 'unknown';

        const numericCount = validValues.filter(val =>
            typeof val === 'number' && !isNaN(val) && isFinite(val)
        ).length;

        const dateCount = validValues.filter(val => {
            if (typeof val === 'string') {
                const date = new Date(val);
                return !isNaN(date.getTime());
            }
            return false;
        }).length;

        if (numericCount / validValues.length > 0.8) return 'numeric';
        if (dateCount / validValues.length > 0.8) return 'datetime';

        return 'categorical';
    }

    getValidCount(column) {
        return column.filter(val => val !== null && val !== undefined).length;
    }

    getNullCount(column) {
        return column.filter(val => val === null || val === undefined).length;
    }

    getUniqueCount(column) {
        const validValues = column.filter(val => val !== null && val !== undefined);
        return new Set(validValues).size;
    }

    getNullPercentage(column) {
        return (this.getNullCount(column) / column.length) * 100;
    }

    calculateMean(values) {
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }

    calculateMedian(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ?
            (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
    }

    calculateStandardDeviation(values) {
        const mean = this.calculateMean(values);
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (values.length - 1);
        return Math.sqrt(variance);
    }

    calculateVariance(values) {
        const mean = this.calculateMean(values);
        return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (values.length - 1);
    }

    calculateSkewness(values) {
        const n = values.length;
        const mean = this.calculateMean(values);
        const stdDev = this.calculateStandardDeviation(values);

        if (stdDev === 0) return 0;

        const skewSum = values.reduce((sum, val) => {
            return sum + Math.pow((val - mean) / stdDev, 3);
        }, 0);

        return (n / ((n - 1) * (n - 2))) * skewSum;
    }

    calculateKurtosis(values) {
        const n = values.length;
        const mean = this.calculateMean(values);
        const stdDev = this.calculateStandardDeviation(values);

        if (stdDev === 0) return -3;

        const kurtSum = values.reduce((sum, val) => {
            return sum + Math.pow((val - mean) / stdDev, 4);
        }, 0);

        return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * kurtSum -
               (3 * Math.pow(n - 1, 2)) / ((n - 2) * (n - 3));
    }

    calculateQuartiles(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const n = sorted.length;

        const q1Index = Math.floor(n * 0.25);
        const q2Index = Math.floor(n * 0.5);
        const q3Index = Math.floor(n * 0.75);

        return {
            q1: sorted[q1Index],
            q2: sorted[q2Index],
            q3: sorted[q3Index],
            iqr: sorted[q3Index] - sorted[q1Index]
        };
    }

    detectOutliers(values) {
        const quartiles = this.calculateQuartiles(values);
        const lowerBound = quartiles.q1 - 1.5 * quartiles.iqr;
        const upperBound = quartiles.q3 + 1.5 * quartiles.iqr;

        const outliers = values.filter(val => val < lowerBound || val > upperBound);

        return {
            count: outliers.length,
            percentage: (outliers.length / values.length) * 100,
            values: outliers,
            lowerBound: lowerBound,
            upperBound: upperBound
        };
    }

    classifyDistribution(values) {
        const skewness = this.calculateSkewness(values);
        const kurtosis = this.calculateKurtosis(values);

        if (Math.abs(skewness) < 0.5 && Math.abs(kurtosis) < 0.5) return 'normal';
        if (skewness > 1) return 'right_skewed';
        if (skewness < -1) return 'left_skewed';
        if (kurtosis > 1) return 'heavy_tailed';
        if (kurtosis < -1) return 'light_tailed';

        return 'unknown';
    }

    calculateFrequencyTable(values) {
        const frequencies = {};
        values.forEach(value => {
            const key = String(value);
            frequencies[key] = (frequencies[key] || 0) + 1;
        });

        const total = values.length;
        return Object.entries(frequencies).map(([value, count]) => ({
            value: value,
            count: count,
            percentage: (count / total) * 100
        })).sort((a, b) => b.count - a.count);
    }

    getMostFrequent(frequencyTable) {
        return frequencyTable[0] || null;
    }

    getLeastFrequent(frequencyTable) {
        return frequencyTable[frequencyTable.length - 1] || null;
    }

    calculateEntropy(frequencyTable) {
        const total = frequencyTable.reduce((sum, item) => sum + item.count, 0);
        return frequencyTable.reduce((entropy, item) => {
            const probability = item.count / total;
            return entropy - probability * Math.log2(probability);
        }, 0);
    }

    calculateConcentration(frequencyTable) {
        const total = frequencyTable.reduce((sum, item) => sum + item.count, 0);
        const topCategory = frequencyTable[0];
        return topCategory ? (topCategory.count / total) * 100 : 0;
    }

    analyzeDateFrequency(dates) {
        const timestamps = dates.map(date => new Date(date).getTime());
        const sorted = timestamps.sort((a, b) => a - b);

        if (sorted.length < 2) return 'insufficient_data';

        const intervals = [];
        for (let i = 1; i < sorted.length; i++) {
            intervals.push(sorted[i] - sorted[i - 1]);
        }

        const avgInterval = intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length;
        const dayInMs = 24 * 60 * 60 * 1000;

        if (avgInterval < dayInMs) return 'sub_daily';
        if (avgInterval < 7 * dayInMs) return 'daily';
        if (avgInterval < 30 * dayInMs) return 'weekly';
        if (avgInterval < 365 * dayInMs) return 'monthly';

        return 'yearly';
    }

    assessCompleteness(dataset) {
        const totalCells = dataset.length * dataset.columns;
        let completeCells = 0;

        dataset.data.forEach(row => {
            dataset.headers.forEach(header => {
                if (row[header] !== null && row[header] !== undefined) {
                    completeCells++;
                }
            });
        });

        return (completeCells / totalCells) * 100;
    }

    assessConsistency(dataset) {
        let score = 100;

        dataset.headers.forEach(header => {
            const column = dataset.data.map(row => row[header]);
            const types = new Set(column.filter(val => val !== null && val !== undefined)
                                       .map(val => typeof val));

            if (types.size > 1) {
                score -= 10;
            }
        });

        return Math.max(0, score);
    }

    assessUniqueness(dataset) {
        const duplicates = this.countDuplicateRows(dataset);
        return Math.max(0, 100 - (duplicates / dataset.length) * 100);
    }

    assessValidity(dataset) {
        let score = 100;
        let totalValues = 0;
        let invalidValues = 0;

        dataset.data.forEach(row => {
            dataset.headers.forEach(header => {
                const value = row[header];
                if (value !== null && value !== undefined) {
                    totalValues++;
                    if (typeof value === 'number' && !isFinite(value)) {
                        invalidValues++;
                    }
                }
            });
        });

        if (totalValues > 0) {
            score = Math.max(0, 100 - (invalidValues / totalValues) * 100);
        }

        return score;
    }

    getQualityRecommendation(score) {
        if (score >= 90) return 'Excellent data quality - ready for analysis';
        if (score >= 80) return 'Good data quality - minor cleaning recommended';
        if (score >= 70) return 'Fair data quality - significant cleaning needed';
        if (score >= 60) return 'Poor data quality - extensive preprocessing required';
        return 'Very poor data quality - major data work needed before analysis';
    }

    testNormality(values) {
        if (values.length < 8) return { test: 'insufficient_data' };

        const mean = this.calculateMean(values);
        const stdDev = this.calculateStandardDeviation(values);
        const skewness = this.calculateSkewness(values);
        const kurtosis = this.calculateKurtosis(values);

        const jarqueBera = (values.length / 6) * (Math.pow(skewness, 2) + Math.pow(kurtosis, 2) / 4);
        const pValue = 1 - this.chiSquareCDF(jarqueBera, 2);

        return {
            test: 'jarque_bera',
            statistic: jarqueBera,
            pValue: pValue,
            isNormal: pValue > 0.05,
            skewness: skewness,
            kurtosis: kurtosis
        };
    }

    createHistogram(values, bins = 10) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / bins;

        const histogram = Array(bins).fill(0);

        values.forEach(value => {
            let binIndex = Math.floor((value - min) / binWidth);
            if (binIndex === bins) binIndex = bins - 1;
            histogram[binIndex]++;
        });

        return histogram.map((count, index) => ({
            binStart: min + index * binWidth,
            binEnd: min + (index + 1) * binWidth,
            count: count,
            percentage: (count / values.length) * 100
        }));
    }

    getDescriptiveStats(values) {
        return {
            count: values.length,
            mean: this.calculateMean(values),
            median: this.calculateMedian(values),
            min: Math.min(...values),
            max: Math.max(...values),
            std: this.calculateStandardDeviation(values),
            var: this.calculateVariance(values),
            skewness: this.calculateSkewness(values),
            kurtosis: this.calculateKurtosis(values)
        };
    }

    calculatePearsonCorrelation(x, y) {
        const validPairs = [];
        for (let i = 0; i < x.length; i++) {
            if (typeof x[i] === 'number' && typeof y[i] === 'number' &&
                !isNaN(x[i]) && !isNaN(y[i]) && isFinite(x[i]) && isFinite(y[i])) {
                validPairs.push({ x: x[i], y: y[i] });
            }
        }

        if (validPairs.length < 3) return 0;

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
        return denominator === 0 ? 0 : numerator / denominator;
    }

    getCorrelationStrength(absCorrelation) {
        if (absCorrelation >= 0.9) return 'Very Strong';
        if (absCorrelation >= 0.7) return 'Strong';
        if (absCorrelation >= 0.5) return 'Moderate';
        if (absCorrelation >= 0.3) return 'Weak';
        return 'Very Weak';
    }

    calculateAverageCorrelation(correlations, columns) {
        let sum = 0;
        let count = 0;

        for (let i = 0; i < columns.length; i++) {
            for (let j = i + 1; j < columns.length; j++) {
                const correlation = correlations[columns[i]][columns[j]];
                if (!isNaN(correlation)) {
                    sum += Math.abs(correlation);
                    count++;
                }
            }
        }

        return count > 0 ? sum / count : 0;
    }

    countDuplicateRows(dataset) {
        const seen = new Set();
        let duplicates = 0;

        dataset.data.forEach(row => {
            const rowString = JSON.stringify(row);
            if (seen.has(rowString)) {
                duplicates++;
            } else {
                seen.add(rowString);
            }
        });

        return duplicates;
    }

    getDataTypes(dataset) {
        const types = {};

        dataset.headers.forEach(header => {
            const column = dataset.data.map(row => row[header]);
            const validValues = column.filter(val => val !== null && val !== undefined);

            if (validValues.length === 0) {
                types[header] = 'empty';
                return;
            }

            const typeSet = new Set(validValues.map(val => typeof val));
            if (typeSet.size === 1) {
                types[header] = Array.from(typeSet)[0];
            } else {
                types[header] = 'mixed';
            }
        });

        return types;
    }

    estimateMemoryFootprint(dataset) {
        let totalBytes = 0;

        dataset.data.forEach(row => {
            dataset.headers.forEach(header => {
                const value = row[header];
                if (typeof value === 'string') {
                    totalBytes += value.length * 2;
                } else if (typeof value === 'number') {
                    totalBytes += 8;
                } else if (typeof value === 'boolean') {
                    totalBytes += 1;
                } else {
                    totalBytes += 8;
                }
            });
        });

        const sizeInKB = totalBytes / 1024;
        const sizeInMB = sizeInKB / 1024;

        if (sizeInMB >= 1) {
            return `${sizeInMB.toFixed(2)} MB`;
        } else if (sizeInKB >= 1) {
            return `${sizeInKB.toFixed(2)} KB`;
        } else {
            return `${totalBytes} bytes`;
        }
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

    generateTextReport(summaryData) {
        let report = '';

        report += `STATISTICAL SUMMARY REPORT\n`;
        report += `Generated: ${new Date(summaryData.generatedAt).toLocaleString()}\n`;
        report += `${'='.repeat(50)}\n\n`;

        report += `BASIC INFORMATION\n`;
        report += `-`.repeat(20) + '\n';
        report += `Rows: ${summaryData.basicInfo.totalRows.toLocaleString()}\n`;
        report += `Columns: ${summaryData.basicInfo.totalColumns}\n`;
        report += `Memory: ${summaryData.basicInfo.memoryFootprint}\n\n`;

        report += `DATA QUALITY\n`;
        report += `-`.repeat(20) + '\n';
        report += `Overall Score: ${summaryData.dataQuality.overallScore.toFixed(1)}/100\n`;
        report += `Completeness: ${summaryData.dataQuality.completenessScore.toFixed(1)}%\n`;
        report += `Consistency: ${summaryData.dataQuality.consistencyScore.toFixed(1)}%\n`;
        report += `${summaryData.dataQuality.recommendation}\n\n`;

        if (summaryData.keyInsights.length > 0) {
            report += `KEY INSIGHTS\n`;
            report += `-`.repeat(20) + '\n';
            summaryData.keyInsights.slice(0, 5).forEach((insight, index) => {
                report += `${index + 1}. ${insight.title}\n`;
                report += `   ${insight.description}\n`;
                if (insight.recommendation) {
                    report += `   → ${insight.recommendation}\n`;
                }
                report += '\n';
            });
        }

        if (summaryData.relationships.strongRelationships.length > 0) {
            report += `STRONG RELATIONSHIPS\n`;
            report += `-`.repeat(20) + '\n';
            summaryData.relationships.strongRelationships.slice(0, 3).forEach(rel => {
                report += `${rel.variable1} ↔ ${rel.variable2}: ${rel.correlation.toFixed(3)} (${rel.strength})\n`;
            });
            report += '\n';
        }

        if (summaryData.recommendations.length > 0) {
            report += `RECOMMENDATIONS\n`;
            report += `-`.repeat(20) + '\n';
            summaryData.recommendations.forEach((rec, index) => {
                report += `${index + 1}. [${rec.priority.toUpperCase()}] ${rec.title}\n`;
                report += `   ${rec.description}\n`;
                rec.steps.forEach(step => {
                    report += `   • ${step}\n`;
                });
                report += '\n';
            });
        }

        return report;
    }

    exportSummary(summaryData, format = 'json') {
        switch (format) {
            case 'json':
                return JSON.stringify(summaryData, null, 2);
            case 'text':
                return this.generateTextReport(summaryData);
            case 'csv':
                return this.generateCSVReport(summaryData);
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }

    generateCSVReport(summaryData) {
        let csv = 'Metric,Value\n';

        csv += `Total Rows,${summaryData.basicInfo.totalRows}\n`;
        csv += `Total Columns,${summaryData.basicInfo.totalColumns}\n`;
        csv += `Overall Quality Score,${summaryData.dataQuality.overallScore.toFixed(1)}\n`;
        csv += `Completeness Score,${summaryData.dataQuality.completenessScore.toFixed(1)}\n`;
        csv += `Consistency Score,${summaryData.dataQuality.consistencyScore.toFixed(1)}\n`;
        csv += `Strong Relationships,${summaryData.relationships.strongRelationships.length}\n`;
        csv += `Key Insights,${summaryData.keyInsights.length}\n`;

        if (summaryData.relationships.strongRelationships.length > 0) {
            csv += '\nVariable 1,Variable 2,Correlation,Strength\n';
            summaryData.relationships.strongRelationships.forEach(rel => {
                csv += `${rel.variable1},${rel.variable2},${rel.correlation.toFixed(4)},${rel.strength}\n`;
            });
        }

        return csv;
    }
}

export default ReportGenerator;