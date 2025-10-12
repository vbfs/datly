/**
 * AutoAnalyzer - Módulo de análise automática para StatLibrary
 * Identifica tipos de variáveis e gera insights automaticamente
 */

class AutoAnalyzer {
  constructor(statsInstance) {
    this.stats = statsInstance;
    this.insights = [];
    this.visualizations = [];
  }

  /**
   * Análise completa automática de um dataset
   * @param {Object} dataset - { headers: string[], data: Array<Record<string, any>> }
   * @param {Object} options
   * @returns {Object} Relatório completo
   */
  autoAnalyze(dataset, options = {}) {
    const config = {
      minCorrelationThreshold: 0.3,
      significanceLevel: 0.05,
      generateVisualizations: true,
      includeAdvancedAnalysis: true,
      ...options
    };

    console.log('🔍 Iniciando análise automática...');

    // 1) Validar
    const validation = this.stats.validateData(dataset);
    if (!validation.valid) {
      throw new Error(`Dados inválidos: ${validation.errors.join(', ')}`);
    }

    // 2) Classificar variáveis
    const variableTypes = this.classifyVariables(dataset);
    console.log(`📊 Identificadas ${variableTypes.quantitative.length} variáveis quantitativas e ${variableTypes.qualitative.length} qualitativas`);

    // 3) Análises
    const descriptiveAnalysis  = this.performDescriptiveAnalysis(dataset, variableTypes);
    const correlationAnalysis  = this.performCorrelationAnalysis(dataset, variableTypes.quantitative, config);
    const regressionAnalysis   = this.performRegressionAnalysis(dataset, variableTypes.quantitative, correlationAnalysis, config);
    const distributionAnalysis = this.performDistributionAnalysis(dataset, variableTypes);
    const outlierAnalysis      = this.performOutlierAnalysis(dataset, variableTypes.quantitative);
    const temporalAnalysis     = this.performTemporalAnalysis(dataset, variableTypes);

    // 4) Insights + Visualizações
    const insights = this.generateAutoInsights(
      dataset,
      {
        variableTypes,
        descriptiveAnalysis,
        correlationAnalysis,
        regressionAnalysis,
        distributionAnalysis,
        outlierAnalysis,
        temporalAnalysis
      },
      config
    );

    const visualizationSuggestions = this.suggestVisualizations(
      variableTypes,
      correlationAnalysis,
      distributionAnalysis
    );

    console.log('✅ Análise concluída!');

    return {
      metadata: {
        analysisDate: new Date().toISOString(),
        datasetSize: dataset.length,
        columnsAnalyzed: dataset.headers.length,
        configuration: config
      },
      variableClassification: variableTypes,
      descriptiveStatistics: descriptiveAnalysis,
      correlationAnalysis,
      regressionAnalysis,
      distributionAnalysis,
      outlierAnalysis,
      temporalAnalysis,
      insights,
      visualizationSuggestions,
      summary: this.generateExecutiveSummary(insights)
    };
  }

  // =======================
  // Classificação de variáveis
  // =======================
  classifyVariables(dataset) {
    const quantitative = [];
    const qualitative = [];
    const datetime = [];
    const binary = [];
    const ordinal = [];

    dataset.headers.forEach(header => {
      const column = dataset.data.map(row => row[header]);
      const nonNullValues = column.filter(val => val != null);

      if (nonNullValues.length === 0) {
        qualitative.push({
          name: header,
          type: 'empty',
          description: 'Coluna vazia'
        });
        return;
      }

      const classification = this.classifyVariable(nonNullValues, header);

      switch (classification.type) {
        case 'quantitative':
          quantitative.push(classification);
          break;
        case 'datetime':
          datetime.push(classification);
          break;
        case 'binary':
          binary.push(classification);
          break;
        case 'ordinal':
          ordinal.push(classification);
          break;
        default:
          qualitative.push(classification);
      }
    });

    return { quantitative, qualitative, datetime, binary, ordinal };
  }

  classifyVariable(values, name) {
    const uniqueValues = [...new Set(values)];
    const numericValues = values.filter(v => typeof v === 'number' && !isNaN(v));
    const numericRatio = numericValues.length / values.length;

    // datetime?
    if (this.isDateTimeColumn(values)) {
      return {
        name,
        type: 'datetime',
        uniqueCount: uniqueValues.length,
        description: 'Variável temporal'
      };
    }

    // quantitativa
    if (numericRatio > 0.8) {
      const subtype = this.determineQuantitativeSubtype(numericValues);
      return {
        name,
        type: 'quantitative',
        subtype,
        uniqueCount: uniqueValues.length,
        description: `Variável quantitativa ${subtype}`,
        range: {
          min: Math.min(...numericValues),
          max: Math.max(...numericValues)
        }
      };
    }

    // binária
    if (uniqueValues.length === 2) {
      return {
        name,
        type: 'binary',
        categories: uniqueValues,
        description: 'Variável binária/dicotômica'
      };
    }

    // ordinal
    if (this.isOrdinalVariable(uniqueValues)) {
      return {
        name,
        type: 'ordinal',
        categories: uniqueValues,
        uniqueCount: uniqueValues.length,
        description: 'Variável ordinal'
      };
    }

    // qualitativa
    return {
      name,
      type: 'qualitative',
      subtype: uniqueValues.length > 10 ? 'nominal_many' : 'nominal',
      categories: uniqueValues.slice(0, 20),
      uniqueCount: uniqueValues.length,
      description: `Variável qualitativa nominal (${uniqueValues.length} categorias)`
    };
  }

  // =======================
  // Descritiva
  // =======================
performDescriptiveAnalysis(dataset, variableTypes) {
  const results = {};

  // Análise quantitativa
  variableTypes.quantitative.forEach(variable => {
    const values = dataset.data
      .map(row => row[variable.name])
      .filter(v => typeof v === 'number' && !isNaN(v));

    if (values.length > 0) {
      const n = values.length;
      const canSkew = n >= 3;
      const canKurt = n >= 4;

      results[variable.name] = {
        type: 'quantitative',
        count: n,
        mean: this.stats.mean(values),
        median: this.stats.median(values),
        standardDeviation: this.stats.standardDeviation(values),
        min: Math.min(...values),
        max: Math.max(...values),
        quartiles: this.stats.quartiles(values),
        skewness: canSkew ? this.stats.skewness(values) : null,
        kurtosis: canKurt ? this.stats.kurtosis(values) : null
      };
    }
  });

  // Análise qualitativa (inclui variáveis binárias)
  [...variableTypes.qualitative, ...variableTypes.binary].forEach(variable => {
    const values = dataset.data
      .map(row => row[variable.name])
      .filter(v => v != null);

    if (values.length > 0) {
      const frequencyTable = this.stats.frequencyTable(values);

      results[variable.name] = {
        type: 'qualitative',
        count: values.length,
        uniqueValues: variable.uniqueCount,
        frequencyTable: frequencyTable.slice(0, 10), // top 10 categorias
        mostFrequent: frequencyTable[0],
        concentration: this.calculateConcentration(frequencyTable)
      };
    }
  });

  return results;
}


  // =======================
  // Correlação
  // =======================
  performCorrelationAnalysis(dataset, quantitativeVars, config) {
    if (quantitativeVars.length < 2) {
      return { message: 'Insuficientes variáveis quantitativas para análise de correlação' };
    }

    // Usa a StatLibrary -> correlationMatrix(dataset)
    const correlationMatrix = this.stats.correlationMatrix(dataset);
    const strongCorrelations = (correlationMatrix.strongCorrelations || [])
      .filter(corr => Math.abs(corr.correlation) >= config.minCorrelationThreshold);

    const insights = strongCorrelations.map(corr => {
      const strength = this.getCorrelationStrength(Math.abs(corr.correlation));
      const direction = corr.correlation > 0 ? 'positiva' : 'negativa';

      return {
        type: 'correlation',
        priority: Math.abs(corr.correlation) > 0.7 ? 'high' : 'medium',
        title: `Correlação ${strength} entre ${corr.variable1} e ${corr.variable2}`,
        description: `Correlação ${direction} de ${corr.correlation.toFixed(3)}`,
        variables: [corr.variable1, corr.variable2],
        correlation: corr.correlation,
        significance: corr.pValue != null ? (corr.pValue < config.significanceLevel) : undefined
      };
    });

    return {
      matrix: correlationMatrix.correlations || correlationMatrix,
      strongCorrelations,
      insights,
      summary: `Encontradas ${strongCorrelations.length} correlações ≥ ${config.minCorrelationThreshold}`
    };
  }

  // =======================
  // Regressão
  // =======================
  performRegressionAnalysis(dataset, quantitativeVars, correlationAnalysis, config) {
    const regressionResults = [];

    if (correlationAnalysis.strongCorrelations) {
      correlationAnalysis.strongCorrelations
        .filter(corr => Math.abs(corr.correlation) > 0.5)
        .slice(0, 5)
        .forEach(corr => {
          try {
            const xValues = dataset.data.map(row => row[corr.variable1])
              .filter(v => typeof v === 'number' && !isNaN(v));
            const yValues = dataset.data.map(row => row[corr.variable2])
              .filter(v => typeof v === 'number' && !isNaN(v));

            if (xValues.length === yValues.length && xValues.length > 10) {
              const regression = this.stats.linearRegression(xValues, yValues);

              regressionResults.push({
                independent: corr.variable1,
                dependent: corr.variable2,
                equation: regression.equation,
                rSquared: regression.rSquared,
                significant: regression.pValueModel < config.significanceLevel,
                interpretation: this.interpretRegressionResult(regression),
                details: regression
              });
            }
          } catch (error) {
            console.warn(`Erro na regressão ${corr.variable1} -> ${corr.variable2}:`, error.message);
          }
        });
    }

    return {
      models: regressionResults,
      summary: `${regressionResults.length} modelos de regressão analisados`
    };
  }

  // =======================
  // Distribuições / Normalidade
  // =======================
  performDistributionAnalysis(dataset, variableTypes) {
    const results = {};

    variableTypes.quantitative.forEach(variable => {
      const values = dataset.data
        .map(row => row[variable.name])
        .filter(v => typeof v === 'number' && !isNaN(v));

      if (values.length > 10) {
        try {
          const normalityTest = this.stats.shapiroWilkTest(values);
          const skewness = this.stats.skewness(values);
          const kurtosis = this.stats.kurtosis(values);

          results[variable.name] = {
            isNormal: normalityTest.isNormal,
            normalityPValue: normalityTest.pValue,
            skewness,
            kurtosis,
            distributionType: this.classifyDistributionType(skewness, kurtosis, normalityTest.isNormal),
            recommendation: this.getDistributionRecommendation(skewness, kurtosis, normalityTest.isNormal)
          };
        } catch (error) {
          results[variable.name] = {
            error: 'Não foi possível analisar a distribuição',
            reason: error.message
          };
        }
      }
    });

    return results;
  }

  // =======================
  // Outliers
  // =======================
  performOutlierAnalysis(dataset, quantitativeVars) {
    const results = {};

    quantitativeVars.forEach(variable => {
      const values = dataset.data
        .map(row => row[variable.name])
        .filter(v => typeof v === 'number' && !isNaN(v));

      if (values.length > 5) {
        const outliers = this.stats.detectOutliers(values, 'iqr');
        results[variable.name] = {
          count: outliers.count,
          percentage: outliers.percentage,
          severity: this.classifyOutlierSeverity(outliers.percentage),
          values: outliers.outliers.slice(0, 10),
          recommendation: this.getOutlierRecommendation(outliers.percentage)
        };
      }
    });

    return results;
  }

  // =======================
  // Temporal
  // =======================
  performTemporalAnalysis(dataset, variableTypes) {
    if (variableTypes.datetime.length === 0) {
      return { message: 'Nenhuma variável temporal detectada' };
    }

    const results = {};

    variableTypes.datetime.forEach(dateVar => {
      const dates = dataset.data
        .map(row => new Date(row[dateVar.name]))
        .filter(date => !isNaN(date.getTime()))
        .sort((a, b) => a - b);

      if (dates.length > 2) {
        const timeSpan = dates[dates.length - 1] - dates[0];
        const avgInterval = timeSpan / (dates.length - 1);

        results[dateVar.name] = {
          span: `${Math.floor(timeSpan / (1000 * 60 * 60 * 24))} dias`,
          frequency: this.determineFrequency(avgInterval),
          earliest: dates[0].toISOString().split('T')[0],
          latest: dates[dates.length - 1].toISOString().split('T')[0],
          dataPoints: dates.length
        };
      }
    });

    return results;
  }

  // =======================
  // Insights / Visualizações / Sumário
  // =======================
  generateAutoInsights(dataset, analyses, config) {
    const insights = [];

    const { quantitative, qualitative } = analyses.variableTypes;
    insights.push({
      category: 'overview',
      priority: 'high',
      title: 'Composição do Dataset',
      description: `Dataset com ${dataset.length} registros, ${quantitative.length} variáveis numéricas e ${qualitative.length} categóricas`,
      icon: '📊'
    });

    if (analyses.correlationAnalysis.insights) {
      insights.push(...analyses.correlationAnalysis.insights);
    }

    Object.entries(analyses.distributionAnalysis).forEach(([variable, analysis]) => {
      if (analysis.distributionType && analysis.distributionType !== 'normal') {
        insights.push({
          category: 'distribution',
          priority: 'medium',
          title: `Distribuição não-normal: ${variable}`,
          description: analysis.recommendation,
          variable,
          icon: '📈'
        });
      }
    });

    Object.entries(analyses.outlierAnalysis).forEach(([variable, analysis]) => {
      if (analysis.severity === 'high') {
        insights.push({
          category: 'quality',
          priority: 'high',
          title: `Outliers significativos em ${variable}`,
          description: `${analysis.count} outliers (${analysis.percentage.toFixed(1)}%) detectados`,
          recommendation: analysis.recommendation,
          variable,
          icon: '⚠️'
        });
      }
    });

    analyses.regressionAnalysis.models?.forEach(model => {
      if (model.significant && model.rSquared > 0.5) {
        insights.push({
          category: 'modeling',
          priority: 'high',
          title: `Modelo preditivo viável: ${model.dependent}`,
          description: `${model.independent} explica ${(model.rSquared * 100).toFixed(1)}% da variação em ${model.dependent}`,
          variables: [model.independent, model.dependent],
          rSquared: model.rSquared,
          icon: '🎯'
        });
      }
    });

    const priorityOrder = { high: 3, medium: 2, low: 1 };
    return insights.sort((a, b) => priorityOrder[b.priority] - priorityOrder[a.priority]);
  }

  suggestVisualizations(variableTypes, correlationAnalysis, distributionAnalysis) {
    const suggestions = [];

    variableTypes.quantitative.forEach(variable => {
      suggestions.push({
        type: 'histogram',
        variable: variable.name,
        title: `Distribuição de ${variable.name}`,
        description: 'Histogram mostrando a distribuição dos valores',
        priority: 'medium'
      });
    });

    if (correlationAnalysis.strongCorrelations) {
      correlationAnalysis.strongCorrelations
        .filter(corr => Math.abs(corr.correlation) > 0.5)
        .slice(0, 3)
        .forEach(corr => {
          suggestions.push({
            type: 'scatter',
            variables: [corr.variable1, corr.variable2],
            title: `${corr.variable1} vs ${corr.variable2}`,
            description: `Scatter plot mostrando correlação ${corr.correlation > 0 ? 'positiva' : 'negativa'}`,
            priority: 'high'
          });
        });
    }

    [...variableTypes.qualitative, ...variableTypes.binary].forEach(variable => {
      if (variable.uniqueCount <= 20) {
        suggestions.push({
          type: 'bar',
          variable: variable.name,
          title: `Frequência de ${variable.name}`,
          description: 'Gráfico de barras mostrando a distribuição das categorias',
          priority: 'medium'
        });
      }
    });

    variableTypes.quantitative.forEach(variable => {
      suggestions.push({
        type: 'boxplot',
        variable: variable.name,
        title: `Box Plot de ${variable.name}`,
        description: 'Box plot para identificar outliers e quartis',
        priority: 'low'
      });
    });

    const priorityOrder = { high: 3, medium: 2, low: 1 };
    return suggestions.sort((a, b) => priorityOrder[b.priority] - priorityOrder[a.priority]);
  }

  generateExecutiveSummary(insights) {
    const highPriority = insights.filter(i => i.priority === 'high');
    const categories = [...new Set(insights.map(i => i.category))];

    return {
      totalInsights: insights.length,
      highPriorityInsights: highPriority.length,
      categoriesCovered: categories,
      keyFindings: highPriority.slice(0, 3).map(i => ({
        title: i.title,
        description: i.description
      })),
      recommendations: this.generateTopRecommendations(insights)
    };
  }

  // =======================
  // Helpers
  // =======================
  isDateTimeColumn(values) {
    const sampleSize = Math.min(values.length, 20);
    const sample = values.slice(0, sampleSize);
    const dateCount = sample.filter(val => {
      if (typeof val === 'string') {
        const date = new Date(val);
        return !isNaN(date.getTime());
      }
      return false;
    }).length;

    return dateCount / sampleSize > 0.7;
  }

  determineQuantitativeSubtype(values) {
    const integers = values.filter(v => Number.isInteger(v));
    const integerRatio = integers.length / values.length;
    return integerRatio > 0.9 ? 'discrete' : 'continuous';
  }

  isOrdinalVariable(uniqueValues) {
    const ordinalPatterns = [
      /^(baixo|médio|alto)$/i,
      /^(pequeno|grande)$/i,
      /^(ruim|regular|bom|ótimo)$/i,
      /^[1-5]$/,
      /^(primeiro|segundo|terceiro)$/i
    ];
    return ordinalPatterns.some(pattern =>
      uniqueValues.every(val => pattern.test(String(val)))
    );
  }

  calculateConcentration(frequencyTable) {
    if (frequencyTable.length === 0) return 0;
    return frequencyTable[0].percentage;
  }

  getCorrelationStrength(correlation) {
    if (correlation >= 0.8) return 'muito forte';
    if (correlation >= 0.6) return 'forte';
    if (correlation >= 0.4) return 'moderada';
    if (correlation >= 0.2) return 'fraca';
    return 'muito fraca';
  }

  interpretRegressionResult(regression) {
    const r2Percent = (regression.rSquared * 100).toFixed(1);
    const significant = regression.pValueModel < 0.05;
    return {
      quality: regression.rSquared > 0.7 ? 'excelente'
        : regression.rSquared > 0.5 ? 'boa'
        : regression.rSquared > 0.3 ? 'moderada' : 'fraca',
      explanation: `O modelo explica ${r2Percent}% da variação`,
      isSignificant: significant
    };
  }

  classifyDistributionType(skewness, kurtosis, isNormal) {
    if (isNormal) return 'normal';
    if (Math.abs(skewness) > 1) {
      return skewness > 0 ? 'assimétrica_direita' : 'assimétrica_esquerda';
    }
    if (Math.abs(kurtosis) > 1) {
      return kurtosis > 0 ? 'leptocúrtica' : 'platicúrtica';
    }
    return 'aproximadamente_normal';
  }

  getDistributionRecommendation(skewness, kurtosis, isNormal) {
    if (isNormal) return 'Distribuição normal - ideal para testes paramétricos';
    if (Math.abs(skewness) > 1) return 'Considere transformação logarítmica para normalizar';
    if (Math.abs(kurtosis) > 1) return 'Distribuição com caudas atípicas - use testes robustos';
    return 'Distribuição aproximadamente normal';
  }

  classifyOutlierSeverity(percentage) {
    if (percentage > 10) return 'high';
    if (percentage > 5) return 'medium';
    return 'low';
  }

  getOutlierRecommendation(percentage) {
    if (percentage > 10) return 'Investigar e possivelmente remover outliers';
    if (percentage > 5) return 'Verificar se outliers são valores legítimos';
    return 'Poucos outliers - monitorar';
  }

  determineFrequency(avgInterval) {
    const day = 24 * 60 * 60 * 1000;
    if (avgInterval < day) return 'diária';
    if (avgInterval < day * 7) return 'semanal';
    if (avgInterval < day * 30) return 'mensal';
    return 'anual';
  }

  generateTopRecommendations(insights) {
    const recommendations = [];

    const correlationInsights = insights.filter(i => i.category === 'correlation');
    if (correlationInsights.length > 0) {
      recommendations.push('Explore as correlações identificadas para possível modelagem preditiva');
    }

    const qualityInsights = insights.filter(i => i.category === 'quality' && i.priority === 'high');
    if (qualityInsights.length > 0) {
      recommendations.push('Trate os outliers identificados antes de prosseguir com análises');
    }

    const distributionInsights = insights.filter(i => i.category === 'distribution');
    if (distributionInsights.length > 0) {
      recommendations.push('Considere transformações para normalizar distribuições assimétricas');
    }

    return recommendations;
  }
}

export default AutoAnalyzer;
