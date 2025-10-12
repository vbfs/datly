class Interpreter {
    interpret(testResult) {
        if (!testResult || typeof testResult !== 'object') {
            throw new Error('Invalid test result object');
        }

        const testType = this.identifyTestType(testResult);

        return {
            testType: testType,
            summary: this.generateSummary(testResult, testType),
            conclusion: this.generateConclusion(testResult, testType),
            significance: this.interpretSignificance(testResult),
            effectSize: this.interpretEffectSize(testResult, testType),
            confidence: this.assessConfidence(testResult),
            assumptions: this.checkAssumptions(testType),
            recommendations: this.generateRecommendations(testResult, testType),
            plainLanguage: this.generatePlainLanguageSummary(testResult, testType)
        };
    }

    identifyTestType(testResult) {
        if (testResult.type) return testResult.type;

        if (testResult.correlation !== undefined) return 'correlation';
        if (testResult.rSquared !== undefined) return 'regression';
        if (testResult.fStatistic !== undefined) return 'anova';
        if (testResult.tStatistic !== undefined || testResult.statistic !== undefined) {
            if (testResult.degreesOfFreedom !== undefined) return 't-test';
            return 'z-test';
        }
        if (testResult.isNormal !== undefined) return 'normality-test';
        if (testResult.clusters !== undefined) return 'clustering';

        return 'general-test';
    }

    generateSummary(testResult, testType) {
        switch (testType) {
            case 'correlation':
                return this.summarizeCorrelation(testResult);
            case 'regression':
                return this.summarizeRegression(testResult);
            case 't-test':
                return this.summarizeTTest(testResult);
            case 'z-test':
                return this.summarizeZTest(testResult);
            case 'anova':
                return this.summarizeANOVA(testResult);
            case 'normality-test':
                return this.summarizeNormalityTest(testResult);
            default:
                return this.summarizeGeneral(testResult);
        }
    }

    summarizeCorrelation(testResult) {
        const r = testResult.correlation;
        const strength = this.getCorrelationStrength(Math.abs(r));
        const direction = r > 0 ? 'positive' : 'negative';
        const significance = testResult.pValue < 0.05 ? 'significant' : 'not significant';

        return `${strength} ${direction} correlation (r = ${r.toFixed(3)}) that is ${significance}`;
    }

    summarizeRegression(testResult) {
        const r2 = testResult.rSquared;
        const variance = (r2 * 100).toFixed(1);
        const significance = testResult.pValueModel < 0.05 ? 'significant' : 'not significant';

        return `${significance} regression model explaining ${variance}% of variance (R² = ${r2.toFixed(3)})`;
    }

    summarizeTTest(testResult) {
        const significance = testResult.pValue < 0.05 ? 'significant' : 'not significant';
        const t = testResult.statistic || testResult.tStatistic;

        return `${significance} difference between groups (t = ${t.toFixed(3)}, p = ${testResult.pValue.toFixed(4)})`;
    }

    summarizeZTest(testResult) {
        const significance = testResult.pValue < 0.05 ? 'significant' : 'not significant';
        const z = testResult.statistic || testResult.zStatistic;

        return `${significance} result compared to population (z = ${z.toFixed(3)}, p = ${testResult.pValue.toFixed(4)})`;
    }

    summarizeANOVA(testResult) {
        const significance = testResult.pValueModel < 0.05 ? 'significant' : 'not significant';
        const f = testResult.fStatistic;

        return `${significance} differences between groups (F = ${f.toFixed(3)}, p = ${testResult.pValueModel.toFixed(4)})`;
    }

    summarizeNormalityTest(testResult) {
        const conclusion = testResult.isNormal ? 'normally distributed' : 'not normally distributed';
        const testName = testResult.test || 'normality test';

        return `Data appears ${conclusion} (${testName}, p = ${testResult.pValue?.toFixed(4) || 'N/A'})`;
    }

    summarizeGeneral(testResult) {
        if (testResult.pValue !== undefined) {
            const significance = testResult.pValue < 0.05 ? 'significant' : 'not significant';
            return `${significance} statistical result (p = ${testResult.pValue.toFixed(4)})`;
        }

        return 'Statistical analysis completed';
    }

    generateConclusion(testResult, testType) {
        const alpha = testResult.alpha || 0.05;
        const pValue = testResult.pValue || testResult.pValueModel;

        if (pValue === undefined) {
            return {
                decision: 'inconclusive',
                statement: 'Cannot determine statistical significance - p-value unavailable'
            };
        }

        const rejectNull = pValue < alpha;
        const confidenceLevel = ((1 - alpha) * 100).toFixed(0);

        let statement = '';
        if (rejectNull) {
            statement = `At the ${confidenceLevel}% confidence level, we reject the null hypothesis (p = ${pValue.toFixed(4)} < ${alpha}).`;
        } else {
            statement = `At the ${confidenceLevel}% confidence level, we fail to reject the null hypothesis (p = ${pValue.toFixed(4)} ≥ ${alpha}).`;
        }

        return {
            decision: rejectNull ? 'reject_null' : 'fail_to_reject_null',
            statement: statement,
            alpha: alpha,
            pValue: pValue,
            confidenceLevel: parseInt(confidenceLevel)
        };
    }

    interpretSignificance(testResult) {
        const pValue = testResult.pValue || testResult.pValueModel;

        if (pValue === undefined) {
            return {
                level: 'unknown',
                interpretation: 'P-value not available'
            };
        }

        let level, interpretation;

        if (pValue < 0.001) {
            level = 'very_strong';
            interpretation = 'Very strong evidence against null hypothesis';
        } else if (pValue < 0.01) {
            level = 'strong';
            interpretation = 'Strong evidence against null hypothesis';
        } else if (pValue < 0.05) {
            level = 'moderate';
            interpretation = 'Moderate evidence against null hypothesis';
        } else if (pValue < 0.1) {
            level = 'weak';
            interpretation = 'Weak evidence against null hypothesis';
        } else {
            level = 'none';
            interpretation = 'No evidence against null hypothesis';
        }

        return {
            level: level,
            pValue: pValue,
            interpretation: interpretation,
            isSignificant: pValue < 0.05
        };
    }

    interpretEffectSize(testResult, testType) {
        switch (testType) {
            case 'correlation':
                return this.interpretCorrelationEffect(testResult);
            case 'regression':
                return this.interpretRegressionEffect(testResult);
            case 't-test':
                return this.interpretTTestEffect(testResult);
            case 'anova':
                return this.interpretANOVAEffect(testResult);
            default:
                return { interpretation: 'Effect size not available for this test type' };
        }
    }

    interpretCorrelationEffect(testResult) {
        const r = Math.abs(testResult.correlation);
        const rSquared = r * r;

        return {
            value: r,
            magnitude: this.getCorrelationStrength(r),
            varianceExplained: (rSquared * 100).toFixed(1) + '%',
            interpretation: `${this.getCorrelationStrength(r).toLowerCase()} relationship`,
            cohen: this.getCohenCorrelation(r)
        };
    }

    interpretRegressionEffect(testResult) {
        const r2 = testResult.rSquared;

        return {
            value: r2,
            magnitude: this.getRSquaredMagnitude(r2),
            varianceExplained: (r2 * 100).toFixed(1) + '%',
            interpretation: `${this.getRSquaredMagnitude(r2).toLowerCase()} explanatory power`
        };
    }

    interpretTTestEffect(testResult) {
        if (testResult.sample1Mean !== undefined && testResult.sample2Mean !== undefined) {
            const diff = Math.abs(testResult.sample1Mean - testResult.sample2Mean);
            const pooledStd = testResult.standardError * Math.sqrt(2);
            const cohensD = diff / pooledStd;

            return {
                value: cohensD,
                magnitude: this.getCohenD(cohensD),
                interpretation: `${this.getCohenD(cohensD).toLowerCase()} effect size`,
                meanDifference: diff
            };
        }

        return { interpretation: 'Effect size cannot be calculated - insufficient data' };
    }

    interpretANOVAEffect(testResult) {
        if (testResult.sumOfSquaresBetween && testResult.sumOfSquaresWithin) {
            const etaSquared = testResult.sumOfSquaresBetween /
                (testResult.sumOfSquaresBetween + testResult.sumOfSquaresWithin);

            return {
                value: etaSquared,
                magnitude: this.getEtaSquared(etaSquared),
                varianceExplained: (etaSquared * 100).toFixed(1) + '%',
                interpretation: `${this.getEtaSquared(etaSquared).toLowerCase()} effect size`
            };
        }

        return { interpretation: 'Effect size cannot be calculated - insufficient data' };
    }

    assessConfidence(testResult) {
        const factors = [];
        let confidence = 'medium';

        if (testResult.sampleSize) {
            if (testResult.sampleSize > 100) {
                factors.push('Large sample size increases reliability');
                confidence = 'high';
            } else if (testResult.sampleSize < 30) {
                factors.push('Small sample size may limit reliability');
                confidence = 'low';
            }
        }

        const pValue = testResult.pValue || testResult.pValueModel;
        if (pValue !== undefined) {
            if (pValue < 0.001) {
                factors.push('Very low p-value strengthens confidence');
                confidence = confidence === 'low' ? 'medium' : 'high';
            } else if (pValue > 0.1) {
                factors.push('High p-value suggests weak evidence');
                confidence = 'low';
            }
        }

        if (testResult.confidenceInterval) {
            const ci = testResult.confidenceInterval;
            const width = Math.abs(ci.upper - ci.lower);
            const estimate = Math.abs((ci.upper + ci.lower) / 2);

            if (estimate > 0 && width / estimate < 0.2) {
                factors.push('Narrow confidence interval indicates precision');
            } else {
                factors.push('Wide confidence interval indicates uncertainty');
                confidence = confidence === 'high' ? 'medium' : 'low';
            }
        }

        return {
            level: confidence,
            factors: factors,
            recommendation: this.getConfidenceRecommendation(confidence)
        };
    }

    checkAssumptions(testType) {
        const assumptions = [];

        switch (testType) {
            case 't-test':
                assumptions.push(
                    'Normality: Data should be approximately normally distributed',
                    'Independence: Observations should be independent',
                    'Equal variances: Groups should have similar variances (for independent samples)'
                );
                break;
            case 'anova':
                assumptions.push(
                    'Normality: Residuals should be normally distributed',
                    'Homogeneity: Groups should have equal variances',
                    'Independence: Observations should be independent'
                );
                break;
            case 'correlation':
                assumptions.push(
                    'Linearity: Relationship should be linear',
                    'Normality: Variables should be approximately normal',
                    'Homoscedasticity: Constant variance across range'
                );
                break;
            case 'regression':
                assumptions.push(
                    'Linearity: Linear relationship between variables',
                    'Independence: Residuals should be independent',
                    'Homoscedasticity: Constant variance of residuals',
                    'Normality: Residuals should be normally distributed'
                );
                break;
            default:
                assumptions.push('Check test-specific assumptions in documentation');
        }

        return {
            testType: testType,
            assumptions: assumptions,
            importance: 'Violating assumptions may invalidate results'
        };
    }

    generateRecommendations(testResult, testType) {
        const recommendations = [];
        const pValue = testResult.pValue || testResult.pValueModel;

        if (pValue !== undefined) {
            if (pValue < 0.001) {
                recommendations.push('Very strong result - investigate practical significance and effect size');
            } else if (pValue >= 0.05 && pValue < 0.1) {
                recommendations.push('Marginally significant - consider collecting more data or using different approach');
            } else if (pValue >= 0.1) {
                recommendations.push('No significant effect found - examine data quality and consider alternative hypotheses');
            }
        }

        if (testType === 'correlation' && testResult.sampleSize && testResult.sampleSize < 30) {
            recommendations.push('Small sample size - correlation may not be reliable');
        }

        if (testType === 'regression' && testResult.rSquared < 0.3) {
            recommendations.push('Low R² - consider additional predictors or different model');
        }

        if (testResult.assumptions && testResult.assumptions.violated) {
            recommendations.push('Assumptions may be violated - consider alternative tests or data transformations');
        }

        recommendations.push('Replicate findings with independent data when possible');

        return recommendations;
    }

    generatePlainLanguageSummary(testResult, testType) {
        const pValue = testResult.pValue || testResult.pValueModel;
        const isSignificant = pValue && pValue < 0.05;

        let summary = '';

        if (isSignificant) {
            summary += '✓ SIGNIFICANT RESULT: ';
        } else {
            summary += '✗ NOT SIGNIFICANT: ';
        }

        switch (testType) {
            case 'correlation':
                const r = testResult.correlation;
                const strength = this.getCorrelationStrength(Math.abs(r));
                if (isSignificant) {
                    summary += `Found a ${strength.toLowerCase()} ${r > 0 ? 'positive' : 'negative'} relationship between the variables.`;
                } else {
                    summary += 'No meaningful relationship found between the variables.';
                }
                break;

            case 'regression':
                const variance = (testResult.rSquared * 100).toFixed(0);
                if (isSignificant) {
                    summary += `The model successfully predicts the outcome, explaining ${variance}% of the variation.`;
                } else {
                    summary += 'The model does not provide meaningful predictions.';
                }
                break;

            case 't-test':
                if (isSignificant) {
                    summary += 'Found a meaningful difference between the groups.';
                } else {
                    summary += 'No meaningful difference found between the groups.';
                }
                break;

            case 'anova':
                if (isSignificant) {
                    summary += 'Found meaningful differences between at least some groups.';
                } else {
                    summary += 'No meaningful differences found between groups.';
                }
                break;

            case 'normality-test':
                if (testResult.isNormal) {
                    summary += 'Data follows a normal distribution - suitable for standard statistical tests.';
                } else {
                    summary += 'Data does not follow a normal distribution - consider alternative tests.';
                }
                break;

            default:
                if (isSignificant) {
                    summary += 'The statistical test shows a significant result.';
                } else {
                    summary += 'The statistical test shows no significant result.';
                }
        }

        if (pValue !== undefined) {
            summary += ` (p-value: ${pValue.toFixed(4)})`;
        }

        return summary;
    }

    getCorrelationStrength(r) {
        if (r >= 0.9) return 'Very Strong';
        if (r >= 0.7) return 'Strong';
        if (r >= 0.5) return 'Moderate';
        if (r >= 0.3) return 'Weak';
        return 'Very Weak';
    }

    getCohenCorrelation(r) {
        if (r >= 0.5) return 'Large effect';
        if (r >= 0.3) return 'Medium effect';
        if (r >= 0.1) return 'Small effect';
        return 'Negligible effect';
    }

    getRSquaredMagnitude(r2) {
        if (r2 >= 0.7) return 'Strong';
        if (r2 >= 0.5) return 'Moderate';
        if (r2 >= 0.3) return 'Weak';
        return 'Very Weak';
    }

    getCohenD(d) {
        if (d >= 0.8) return 'Large';
        if (d >= 0.5) return 'Medium';
        if (d >= 0.2) return 'Small';
        return 'Negligible';
    }

    getEtaSquared(eta2) {
        if (eta2 >= 0.14) return 'Large';
        if (eta2 >= 0.06) return 'Medium';
        if (eta2 >= 0.01) return 'Small';
        return 'Negligible';
    }

    getConfidenceRecommendation(confidence) {
        switch (confidence) {
            case 'high':
                return 'Results appear robust and reliable';
            case 'medium':
                return 'Results are reasonably reliable but verify when possible';
            case 'low':
                return 'Interpret results with caution - consider additional validation';
            default:
                return 'Assess result reliability based on context';
        }
    }

    formatForReport(interpretation) {
        return {
            title: `${interpretation.testType.toUpperCase()} Results`,
            summary: interpretation.summary,
            conclusion: interpretation.conclusion.statement,
            significance: interpretation.significance.interpretation,
            effect: interpretation.effectSize.interpretation,
            confidence: interpretation.confidence.level,
            recommendations: interpretation.recommendations,
            plainLanguage: interpretation.plainLanguage
        };
    }

    explainStatistic(testResult, testType) {
        const explanations = {
            'correlation': 'Correlation measures the linear relationship between two variables, ranging from -1 to +1.',
            'regression': 'R² shows how much variance in the outcome is explained by the predictors.',
            't-test': 'T-test compares means between groups or against a known value.',
            'anova': 'ANOVA tests whether there are differences between multiple group means.',
            'z-test': 'Z-test compares a sample mean to a population mean when population variance is known.',
            'normality-test': 'Tests whether data follows a normal (bell-curve) distribution.'
        };

        return explanations[testType] || 'Statistical test to evaluate hypotheses about data.';
    }

    generateActionItems(interpretation) {
        const actions = [];
        const testType = interpretation.testType;
        const isSignificant = interpretation.significance.isSignificant;

        if (isSignificant) {
            actions.push('Examine practical significance of the finding');
            actions.push('Consider replicating with independent data');

            if (testType === 'correlation') {
                actions.push('Explore potential causal relationships');
            }
            if (testType === 'regression') {
                actions.push('Validate model with new data');
            }
        } else {
            actions.push('Review data collection methods');
            actions.push('Consider if sample size was adequate');
            actions.push('Explore alternative analytical approaches');
        }

        actions.push('Document methodology and assumptions');

        return actions;
    }
}

export default Interpreter;