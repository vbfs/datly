// import (node esm)
import {
  mean, stddeviation, dataframe_from_json, eda_overview,
  train_test_split, train_logistic_regression, predict_logistic,
  metrics_classification, normal_cdf, t_test_independent
} from './src/code.js';

// 1) estatística básica
console.log( mean([1,2,3,4]) );            // texto: type/statistic/name/value/...
console.log( stddeviation([1,2,3,4]) );

// 2) dataframe + eda
const rows = [
  { renda: 4000, idade: 29, margem: 800, default: 0 },
  { renda: 2500, idade: 41, margem: 500, default: 1 },
  { renda: 5200, idade: 33, margem: 1100, default: 0 },
];
console.log( dataframe_from_json(rows) );   // texto com colunas, dtypes e preview
console.log( eda_overview(rows) );         // texto com describe/missing/correlation

// 3) modelo de classificação (logística)
const X = rows.map(r => [r.renda, r.idade, r.margem]);
const y = rows.map(r => r.default);
const split = train_test_split(X, y, 0.33);
console.log(split); // mostra tamanhos e índices em texto

const modelText = train_logistic_regression(X, y, { learning_rate: 0.1, iterations: 800 });
console.log(modelText); // texto com pesos e acurácia

const predsText = predict_logistic(modelText, [[3000, 30, 700], [6000, 45, 1200]], 0.5);
console.log(predsText); // texto com probabilities/classes

// supondo y_true para avaliar:
const y_true = [0, 1];
const y_pred = [0, 0]; // (exemplo)
console.log( metrics_classification(y_true, y_pred) );

// 4) distribuições
console.log( normal_cdf([ -1.96, 0, 1.96 ]) );

// 5) testes de hipóteses
console.log( t_test_independent([10,12,9,11], [8,7,9,10]) );
