// ========================
// 📊 D3 Core Imports
// ========================
// d3-selection
import { select, selectAll } from "d3-selection";
// d3-scale
import {
  scaleLinear,
  scaleBand,
  scalePoint,
  scaleOrdinal,
  scaleSequential,
} from "d3-scale";
// d3-array
import {
  extent,
  max,
  min,
  sum,
  range,
  mean,
  deviation,
  histogram as d3Histogram,
  quantile,
} from "d3-array";
// d3-axis
import { axisBottom, axisLeft } from "d3-axis";
// d3-colors
import {
  schemeCategory10,
  interpolateRdYlBu,
  interpolateViridis,
} from "d3-scale-chromatic";
// d3-shape
import { line as d3Line, curveBasis, pie as d3Pie, arc as d3Arc } from "d3-shape";

let plotCounter = 0;

const defaultConfig = {
  width: 400,
  height: 400,
  color: "#000",
  background: "#fff",
  title: "",
  xlabel: "",
  ylabel: "",
  axisColor: "#000000",
  titleColor: "#000000",
  xAxisColor: null,
  yAxisColor: null
};

function createSvg(userSelector, opts) {
  const config = { ...defaultConfig, ...opts };
  let selector = userSelector;
  let container;

  if (!selector) {
    selector = `#datly-plot-${plotCounter++}`;
    const div = document.createElement("div");
    div.id = selector.replace("#", "");
    document.body.appendChild(div);
  }

  container = select(selector);
  container.html("");
  container.style("background", config.background).style("display", "inline-block");

  if (config.title) {
    container
      .append("h3")
      .style("text-align", "center")
      .style("font-family", "sans-serif")
      .style("margin-bottom", "5px")
      .style("color", config.titleColor || defaultConfig.titleColor)
      .text(config.title);
  }

  const svg = container
    .append("svg")
    .attr("width", config.width)
    .attr("height", config.height)
    .style("background", config.background);

  // Adicionar xlabel
  if (config.xlabel) {
    svg.append("text")
      .attr("x", config.width / 2)
      .attr("y", config.height - 10) // 10px da borda inferior
      .attr("text-anchor", "middle")
      .style("font-family", "sans-serif")
      .style("font-size", "14px")
      .style("fill", config.xAxisColor || config.axisColor || defaultConfig.axisColor)
      .text(config.xlabel);
  }

  // Adicionar ylabel
  if (config.ylabel) {
    svg.append("text")
      .attr("transform", `translate(15, ${config.height / 2}) rotate(-90)`)
      .attr("text-anchor", "middle")
      .style("font-family", "sans-serif")
      .style("font-size", "14px")
      .style("fill", config.yAxisColor || config.axisColor || defaultConfig.axisColor)
      .text(config.ylabel);
  }

  return { svg, config };
}

// ✅ Função para aplicar cor nos eixos
function styleAxis(axisSelection, color) {
  axisSelection.selectAll("path").attr("stroke", color);
  axisSelection.selectAll("line").attr("stroke", color);
  axisSelection.selectAll("text").attr("fill", color);
}

// =======================================================
// HISTOGRAM
// =======================================================
export function plotHistogram(data, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleLinear().domain(extent(data)).nice().range([0, width]);
  const bins = d3Histogram().domain(x.domain()).thresholds(options.bins || 10)(data);
  const y = scaleLinear().domain([0, max(bins, (d) => d.length)]).nice().range([height, 0]);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  g.selectAll("rect")
    .data(bins)
    .enter()
    .append("rect")
    .attr("x", (d) => x(d.x0))
    .attr("y", (d) => y(d.length))
    .attr("width", (d) => x(d.x1) - x(d.x0) - 1)
    .attr("height", (d) => height - y(d.length))
    .attr("fill", config.color);

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

// =======================================================
// BOXPLOT
// =======================================================
export function plotBoxplot(data, options = {}, selector) {
  const groups = Array.isArray(data[0]) ? data : [data];
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleBand()
    .domain(groups.map((_, i) => options.labels ? options.labels[i] : `Group ${i+1}`))
    .range([0, width])
    .padding(0.5);

  const allValues = groups.flat();
  const y = scaleLinear().domain(extent(allValues)).nice().range([height, 0]);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  groups.forEach((group, i) => {
    const sorted = [...group].sort((a, b) => a - b);
    const q1 = quantile(sorted, 0.25);
    const median = quantile(sorted, 0.5);
    const q3 = quantile(sorted, 0.75);
    const minVal = min(sorted);
    const maxVal = max(sorted);
    const xPos = x(options.labels ? options.labels[i] : `Group ${i+1}`) + x.bandwidth()/2;
    const boxWidth = x.bandwidth()/2;

    g.append("line")
      .attr("x1", xPos)
      .attr("x2", xPos)
      .attr("y1", y(minVal))
      .attr("y2", y(maxVal))
      .attr("stroke", config.color);

    g.append("rect")
      .attr("x", xPos - boxWidth / 2)
      .attr("y", y(q3))
      .attr("width", boxWidth)
      .attr("height", y(q1) - y(q3))
      .attr("stroke", config.color)
      .attr("fill", "none");

    g.append("line")
      .attr("x1", xPos - boxWidth / 2)
      .attr("x2", xPos + boxWidth / 2)
      .attr("y1", y(median))
      .attr("y2", y(median))
      .attr("stroke", config.color);
  });

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

// =======================================================
// SCATTER
// =======================================================
export function plotScatter(xData, yData, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleLinear().domain(extent(xData)).nice().range([0, width]);
  const y = scaleLinear().domain(extent(yData)).nice().range([height, 0]);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  g.selectAll("circle")
    .data(xData)
    .enter()
    .append("circle")
    .attr("cx", (_, i) => x(xData[i]))
    .attr("cy", (_, i) => y(yData[i]))
    .attr("r", options.size || 4)
    .attr("fill", config.color);

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

// =======================================================
// LINE
// =======================================================
export function plotLine(xData, yData, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleLinear().domain(extent(xData)).range([0, width]);
  const y = scaleLinear().domain(extent(yData)).range([height, 0]);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const path = d3Line()
    .x((_, i) => x(xData[i]))
    .y((_, i) => y(yData[i]))
    .curve(curveBasis);

  g.append("path")
    .datum(xData)
    .attr("fill", "none")
    .attr("stroke", config.color)
    .attr("stroke-width", options.lineWidth || 2)
    .attr("d", path);

  if (options.showPoints) {
    g.selectAll("circle")
      .data(xData)
      .enter()
      .append("circle")
      .attr("cx", (_, i) => x(xData[i]))
      .attr("cy", (_, i) => y(yData[i]))
      .attr("r", 3)
      .attr("fill", config.color);
  }

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

// =======================================================
// BAR
// =======================================================
export function plotBar(categories, values, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleBand().domain(categories).range([0, width]).padding(0.2);
  const y = scaleLinear().domain([0, max(values)]).nice().range([height, 0]);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  g.selectAll("rect")
    .data(values)
    .enter()
    .append("rect")
    .attr("x", (_, i) => x(categories[i]))
    .attr("y", (d) => y(d))
    .attr("width", x.bandwidth())
    .attr("height", (d) => height - y(d))
    .attr("fill", config.color);

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

// =======================================================
// PIE
// =======================================================
export function plotPie(labels, values, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const radius = Math.min(config.width, config.height) / 2;
  const g = svg.append("g").attr("transform", `translate(${config.width/2},${config.height/2})`);
  const color = scaleOrdinal(schemeCategory10);
  const pieGen = d3Pie();
  const arcs = pieGen(values);
  const arcGen = d3Arc().innerRadius(0).outerRadius(radius);

  g.selectAll("path")
    .data(arcs)
    .enter()
    .append("path")
    .attr("d", arcGen)
    .attr("fill", (d, i) => color(i));

  if (options.showLabels) {
    g.selectAll("text")
      .data(arcs)
      .enter()
      .append("text")
      .attr("transform", (d) => `translate(${arcGen.centroid(d)})`)
      .attr("text-anchor", "middle")
      .text((d, i) => labels[i]);
  }
}

// =======================================================
// HEATMAP
// =======================================================
export function plotHeatmap(matrix, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const labels = options.labels || matrix.map((_, i) => `Var${i+1}`);
  const margin = { top: 40, right: 20, bottom: 40, left: 60 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleBand().domain(labels).range([0, width]).padding(0.05);
  const y = scaleBand().domain(labels).range([0, height]).padding(0.05);
  const color = scaleSequential(interpolateRdYlBu).domain([1, -1]);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const cells = [];
  matrix.forEach((row, i) => {
    row.forEach((value, j) => cells.push({ x: labels[j], y: labels[i], value }));
  });

  g.selectAll("rect")
    .data(cells)
    .enter()
    .append("rect")
    .attr("x", d => x(d.x))
    .attr("y", d => y(d.y))
    .attr("width", x.bandwidth())
    .attr("height", y.bandwidth())
    .attr("fill", d => color(d.value));

  if (options.showValues) {
    g.selectAll("text")
      .data(cells)
      .enter()
      .append("text")
      .attr("x", d => x(d.x) + x.bandwidth()/2)
      .attr("y", d => y(d.y) + y.bandwidth()/2)
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .text(d => d.value.toFixed(2));
  }

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

// =======================================================
// VIOLIN
// =======================================================
export function plotViolin(groups, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const dataGroups = Array.isArray(groups[0]) ? groups : [groups];
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleBand()
    .domain(dataGroups.map((_, i) => options.labels ? options.labels[i] : `Group ${i+1}`))
    .range([0, width])
    .padding(0.5);
  const allValues = dataGroups.flat();
  const y = scaleLinear().domain(extent(allValues)).nice().range([height, 0]);
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  dataGroups.forEach((group, i) => {
    const bins = d3Histogram().domain(y.domain()).thresholds(20)(group);
    const maxLen = max(bins, d => d.length);
    const xPos = x(options.labels ? options.labels[i] : `Group ${i+1}`);
    const scaleW = scaleLinear().domain([0, maxLen]).range([0, x.bandwidth()/2]);

    const areaGen = d3Line()
      .x(d => scaleW(d.length))
      .y(d => y((d.x0 + d.x1)/2));

    const mirrored = d3Line()
      .x(d => -scaleW(d.length))
      .y(d => y((d.x0 + d.x1)/2));

    const g2 = g.append("g").attr("transform", `translate(${xPos + x.bandwidth()/2},0)`);

    g2.append("path")
      .datum(bins)
      .attr("fill", options.color || config.color)
      .attr("fill-opacity", 0.3)
      .attr("stroke", config.color)
      .attr("d", areaGen);

    g2.append("path")
      .datum(bins)
      .attr("fill", options.color || config.color)
      .attr("fill-opacity", 0.3)
      .attr("stroke", config.color)
      .attr("d", mirrored);
  });

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

// =======================================================
// DENSITY
// =======================================================
export function plotDensity(data, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scaleLinear().domain(extent(data)).nice().range([0, width]);
  const kde = kernelDensityEstimator(epanechnikovKernel(options.bandwidth || 5), x.ticks(50));
  const density = kde(data);
  const y = scaleLinear().domain([0, max(density, d => d[1])]).range([height, 0]);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const path = d3Line().curve(curveBasis).x(d => x(d[0])).y(d => y(d[1]));

  g.append("path")
    .datum(density)
    .attr("fill", "none")
    .attr("stroke", config.color)
    .attr("stroke-width", 2)
    .attr("d", path);

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}

function kernelDensityEstimator(kernel, X) {
  return function (V) {
    return X.map(function (x) {
      return [x, mean(V, v => kernel(x - v))];
    });
  };
}
function epanechnikovKernel(bandwidth) {
  return function (u) {
    u /= bandwidth;
    return Math.abs(u) <= 1 ? 0.75 * (1 - u * u) / bandwidth : 0;
  };
}

// =======================================================
// QQ PLOT
// =======================================================
export function plotQQ(data, options = {}, selector) {
  const sorted = [...data].sort((a,b)=>a-b);
  const n = sorted.length;
  const quantiles = sorted.map((_,i)=>(i+0.5)/n);
  const theoretical = quantiles.map(q => normalQuantile(q));
  plotScatter(theoretical, sorted, options, selector);
}

function normalQuantile(p) {
  const a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
  const a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
  const b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
  const b4 = 66.8013118877197, b5 = -13.2806815528857;
  const c1 = -0.00778489400243029, c2 = -0.322396458041136;
  const c3 = -2.40075827716184, c4 = -2.54973253934373;
  const c5 = 4.37466414146497, c6 = 2.93816398269878;
  const d1 = 0.00778469570904146, d2 = 0.32246712907004;
  const d3 = 2.445134137143, d4 = 3.75440866190742;
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q, r;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1);
  } else if (phigh < p) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1);
  } else {
    q = p - 0.5;
    r = q * q;
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/((((b1*r+b2)*r+b3)*r+b4)*r+b5)+1;
  }
}

// =======================================================
// PARALLEL COORDINATES
// =======================================================
export function plotParallel(data, dimensions, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 30, right: 30, bottom: 10, left: 30 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const x = scalePoint().range([0, width]).padding(1).domain(dimensions);
  const y = {};
  dimensions.forEach(dim => {
    y[dim] = scaleLinear()
      .domain(extent(data, d => d[dim]))
      .range([height, 0]);
  });

  const lineGen = d3Line();
  const path = d => lineGen(dimensions.map(p => [x(p), y[p](d[p])]));

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  g.selectAll("path")
    .data(data)
    .enter().append("path")
    .attr("d", path)
    .attr("fill", "none")
    .attr("stroke", (d, i) => options.colors ? options.colors[i % options.colors.length] : config.color)
    .attr("stroke-width", 1)
    .attr("opacity", 0.6);

  dimensions.forEach(dim => {
    const axis = g.append("g")
      .attr("transform", `translate(${x(dim)},0)`)
      .call(axisLeft(y[dim]));
    styleAxis(axis, config.yAxisColor || config.axisColor);
    axis.append("text")
      .style("text-anchor", "middle")
      .attr("y", -9)
      .text(dim);
  });
}

// =======================================================
// PAIRPLOT
// =======================================================
export function plotPairplot(data, columns, options = {}, selector) {
  const n = columns.length;
  const size = options.size || 120;
  const gap = 10;
  const totalSize = n * (size + gap);
  const container = selector || `#datly-plot-${plotCounter++}`;
  const div = document.createElement("div");
  div.id = container.replace("#", "");
  document.body.appendChild(div);

  const containerSel = select(container);
  containerSel.html("");
  containerSel.style("display", "inline-block");

  const svg = containerSel
    .append("svg")
    .attr("width", totalSize)
    .attr("height", totalSize)
    .style("background", "#fff");

  const x = {};
  const y = {};
  columns.forEach(col => {
    x[col] = scaleLinear().domain(extent(data, d => d[col])).range([gap, size - gap]);
    y[col] = scaleLinear().domain(extent(data, d => d[col])).range([size - gap, gap]);
  });

  columns.forEach((colX, i) => {
    columns.forEach((colY, j) => {
      const g = svg.append("g")
        .attr("transform", `translate(${i * (size + gap)},${j * (size + gap)})`);
      g.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", d => x[colX](d[colX]))
        .attr("cy", d => y[colY](d[colY]))
        .attr("r", 2)
        .attr("fill", options.color || "#000");
    });
  });
}

// =======================================================
// MULTILINE
// =======================================================
export function plotMultiline(series, options = {}, selector) {
  const { svg, config } = createSvg(selector, options);
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const width = config.width - margin.left - margin.right;
  const height = config.height - margin.top - margin.bottom;

  const allX = series.flatMap(s => s.data.map(d => d.x));
  const allY = series.flatMap(s => s.data.map(d => d.y));
  const x = scaleLinear().domain(extent(allX)).range([0, width]);
  const y = scaleLinear().domain(extent(allY)).range([height, 0]);
  const color = scaleOrdinal(schemeCategory10);

  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  series.forEach((s, i) => {
    const path = d3Line()
      .x(d => x(d.x))
      .y(d => y(d.y));

    g.append("path")
      .datum(s.data)
      .attr("fill", "none")
      .attr("stroke", color(i))
      .attr("stroke-width", 2)
      .attr("d", path);
  });

  if (options.legend) {
    const legend = svg.append("g").attr("transform", `translate(${width - 100},20)`);
    series.forEach((s, i) => {
      legend.append("rect")
        .attr("x", 0)
        .attr("y", i * 20)
        .attr("width", 12)
        .attr("height", 12)
        .attr("fill", color(i));
      legend.append("text")
        .attr("x", 20)
        .attr("y", i * 20 + 10)
        .text(s.name)
        .style("font-size", "12px");
    });
  }

  const xAxis = g.append("g").attr("transform", `translate(0,${height})`).call(axisBottom(x));
  const yAxis = g.append("g").call(axisLeft(y));
  styleAxis(xAxis, config.xAxisColor || config.axisColor);
  styleAxis(yAxis, config.yAxisColor || config.axisColor);
}
