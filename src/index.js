import {
  plotHistogram,
  plotBoxplot,
  plotScatter,
  plotLine,
  plotBar,
  plotPie,
  plotHeatmap,
  plotViolin,
  plotDensity,
  plotQQ,
  plotParallel,
  plotPairplot,
  plotMultiline,
} from "./plot.js";

import {
// Loaders
  df_from_csv,
  df_from_json,
  df_from_array,
  df_from_structured_json,
  df_from_object,

  // Manipulação básica
  df_get_column,
  df_get_columns,
  df_filter,
  df_sort,
  df_select,
  df_info,
  df_head,
  df_tail,

  // Avançadas
  df_concat,
  df_merge,
  df_dropna,
  df_fillna,
  df_groupby,
  df_aggregate,

  // Utilitárias
  df_apply,
  df_to_csv,
  df_sample,
  df_unique,
  df_rename,
  df_add_column,
  df_drop,
  df_describe,
  df_missing_report,
  df_corr,
  df_explode,

  eda_overview,
  // stats
  mean,
  stddeviation,
  variance,
  median,
  quantile,
  minv,
  maxv,
  skewness,
  kurtosis,
  corr_pearson,
  corr_spearman,
  // distributions
  normal_pdf,
  normal_cdf,
  normal_ppf,
  binomial_pmf,
  binomial_cdf,
  poisson_pmf,
  poisson_cdf,
  // hypothesis tests
  t_test_independent,
  z_test_one_sample,
  chi_square_independence,
  anova_oneway,
  // ml
  train_test_split,
  train_linear_regression,
  train_logistic_regression,
  predict_linear,
  predict_logistic,
  metrics_classification,
  metrics_regression,
  // additional statistical tests
  t_test_paired,
  t_test_one_sample,
  shapiro_wilk,
  jarque_bera,
  levene_test,
  kruskal_wallis,
  mann_whitney,
  wilcoxon_signed_rank,
  chi_square_goodness,
  // confidence intervals
  confidence_interval_mean,
  confidence_interval_proportion,
  confidence_interval_variance,
  confidence_interval_difference,
  // additional correlations
  corr_kendall,
  corr_partial,
  corr_matrix_all,
  // knn
  train_knn_classifier,
  predict_knn_classifier,
  train_knn_regressor,
  predict_knn_regressor,
  // decision trees
  train_decision_tree_classifier,
  train_decision_tree_regressor,
  predict_decision_tree,
  // random forest
  train_random_forest_classifier,
  train_random_forest_regressor,
  predict_random_forest_classifier,
  predict_random_forest_regressor,
  // naive bayes
  train_naive_bayes,
  predict_naive_bayes,
  // feature scaling
  standard_scaler_fit,
  standard_scaler_transform,
  minmax_scaler_fit,
  minmax_scaler_transform,
  // dimensionality reduction
  train_pca,
  transform_pca,
  // clustering
  train_kmeans,
  predict_kmeans,
  // ensemble
  ensemble_voting_classifier,
  ensemble_voting_regressor,
  // cross-validation
  cross_validate,
  // feature importance
  feature_importance_tree,
  // outlier detection
  outliers_iqr,
  outliers_zscore,
  // time series
  moving_average,
  exponential_smoothing,
  autocorrelation,
} from "./code.js";

const datly = {
  // Loaders
  df_from_csv,
  df_from_json,
  df_from_array,
  df_from_structured_json,
  df_from_object,

  // Manipulação básica
  df_get_column,
  df_get_columns,
  df_filter,
  df_sort,
  df_select,
  df_info,
  df_head,
  df_tail,

  // Avançadas
  df_concat,
  df_merge,
  df_dropna,
  df_fillna,
  df_groupby,
  df_aggregate,

  // Utilitárias
  df_apply,
  df_to_csv,
  df_sample,
  df_unique,
  df_rename,
  df_add_column,
  df_drop,
  df_describe,
  df_missing_report,
  df_corr,
  df_explode,
  eda_overview,
  // stats
  mean,
  stddeviation,
  variance,
  median,
  quantile,
  minv,
  maxv,
  skewness,
  kurtosis,
  corr_pearson,
  corr_spearman,
  // distributions
  normal_pdf,
  normal_cdf,
  normal_ppf,
  binomial_pmf,
  binomial_cdf,
  poisson_pmf,
  poisson_cdf,
  // hypothesis tests
  t_test_independent,
  z_test_one_sample,
  chi_square_independence,
  anova_oneway,
  // ml
  train_test_split,
  train_linear_regression,
  train_logistic_regression,
  predict_linear,
  predict_logistic,
  metrics_classification,
  metrics_regression,
  // additional statistical tests
  t_test_paired,
  t_test_one_sample,
  shapiro_wilk,
  jarque_bera,
  levene_test,
  kruskal_wallis,
  mann_whitney,
  wilcoxon_signed_rank,
  chi_square_goodness,
  // confidence intervals
  confidence_interval_mean,
  confidence_interval_proportion,
  confidence_interval_variance,
  confidence_interval_difference,
  // additional correlations
  corr_kendall,
  corr_partial,
  corr_matrix_all,
  // knn
  train_knn_classifier,
  predict_knn_classifier,
  train_knn_regressor,
  predict_knn_regressor,
  // decision trees
  train_decision_tree_classifier,
  train_decision_tree_regressor,
  predict_decision_tree,
  // random forest
  train_random_forest_classifier,
  train_random_forest_regressor,
  predict_random_forest_classifier,
  predict_random_forest_regressor,
  // naive bayes
  train_naive_bayes,
  predict_naive_bayes,
  // feature scaling
  standard_scaler_fit,
  standard_scaler_transform,
  minmax_scaler_fit,
  minmax_scaler_transform,
  // dimensionality reduction
  train_pca,
  transform_pca,
  // clustering
  train_kmeans,
  predict_kmeans,
  // ensemble
  ensemble_voting_classifier,
  ensemble_voting_regressor,
  // cross-validation
  cross_validate,
  // feature importance
  feature_importance_tree,
  // outlier detection
  outliers_iqr,
  outliers_zscore,
  // time series
  moving_average,
  exponential_smoothing,
  autocorrelation,
  // plots
  plotHistogram,
  plotBoxplot,
  plotScatter,
  plotLine,
  plotBar,
  plotPie,
  plotHeatmap,
  plotViolin,
  plotDensity,
  plotQQ,
  plotParallel,
  plotPairplot,
  plotMultiline,
};

export default datly;
