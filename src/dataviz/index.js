// d3-selection
import { select, selectAll } from "d3-selection";
import {
  scaleLinear,
  scaleBand,
  scalePoint,
  scaleOrdinal,
  scaleSequential,
} from "d3-scale";
import {
  extent,
  max,
  min,
  sum,
  range,
  mean,
  deviation,
  histogram,
  quantile,
} from "d3-array";
import { axisBottom, axisLeft } from "d3-axis";
import {
  schemeCategory10,
  interpolateRdYlBu,
  interpolateViridis,
} from "d3-scale-chromatic";
import { line, area, curveBasis, pie, arc } from "d3-shape";

const d3 = {
  select,
  selectAll,
  scaleLinear,
  scaleBand,
  scalePoint,
  scaleOrdinal,
  scaleSequential,
  extent,
  max,
  min,
  sum,
  deviation,
  mean,
  quantile,
  histogram,
  range,
  axisBottom,
  axisLeft,
  line,
  area,
  curveBasis,
  pie,
  arc,
  schemeCategory10,
  interpolateViridis,
  interpolateRdYlBu,
};

class DataViz {
  constructor(containerId = "dataviz-container") {
    this.containerId = containerId;
    this.defaultWidth = 800;
    this.defaultHeight = 600;
    this.defaultMargin = { top: 40, right: 40, bottom: 60, left: 60 };
    this.colors = d3.schemeCategory10;
  }

  /**
   * Cria ou atualiza um container para visualização
   * @param {string} containerId - ID do elemento container
   * @param {number} width - Largura do SVG
   * @param {number} height - Altura do SVG
   */
  createContainer(
    containerId,
    width = this.defaultWidth,
    height = this.defaultHeight
  ) {
    const targetId = containerId || this.containerId;
    let container = d3.select(`#${targetId}`);

    if (container.empty()) {
      container = d3
        .select("body")
        .append("div")
        .attr("id", targetId)
        .style("margin", "20px");
    }

    container.selectAll("*").remove();

    const svg = container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("background", "#fff")
      .style("border", "1px solid #ddd")
      .style("border-radius", "8px");

    return { container, svg };
  }

  // ============================================
  // HISTOGRAMA
  // ============================================
  histogram(data, options = {}) {
    const {
      title = "Histogram",
      xlabel = "Value",
      ylabel = "Frequency",
      bins = 30,
      color = "#4299e1",
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain(d3.extent(data)).range([0, innerWidth]);

    const histogram = d3
      .histogram()
      .domain(x.domain())
      .thresholds(x.ticks(bins));

    const histData = histogram(data);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(histData, (d) => d.length)])
      .range([innerHeight, 0]);

    g.selectAll("rect")
      .data(histData)
      .join("rect")
      .attr("x", (d) => x(d.x0) + 1)
      .attr("width", (d) => Math.max(0, x(d.x1) - x(d.x0) - 2))
      .attr("y", (d) => y(d.length))
      .attr("height", (d) => innerHeight - y(d.length))
      .attr("fill", color)
      .attr("opacity", 0.8)
      .on("mouseover", function () {
        d3.select(this).attr("opacity", 1);
      })
      .on("mouseout", function () {
        d3.select(this).attr("opacity", 0.8);
      });

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // BOX PLOT
  // ============================================
  boxplot(data, options = {}) {
    const {
      title = "Box Plot",
      xlabel = "Category",
      ylabel = "Value",
      labels = null,
      color = "#4299e1",
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const datasets = Array.isArray(data[0]) ? data : [data];
    const categoryLabels = labels || datasets.map((_, i) => `Group ${i + 1}`);

    const boxData = datasets.map((dataset, i) => {
      const sorted = [...dataset].sort((a, b) => a - b);
      const q1 = d3.quantile(sorted, 0.25);
      const median = d3.quantile(sorted, 0.5);
      const q3 = d3.quantile(sorted, 0.75);
      const iqr = q3 - q1;
      const min = Math.max(d3.min(sorted), q1 - 1.5 * iqr);
      const max = Math.min(d3.max(sorted), q3 + 1.5 * iqr);
      const outliers = sorted.filter((d) => d < min || d > max);

      return { label: categoryLabels[i], q1, median, q3, min, max, outliers };
    });

    const x = d3
      .scaleBand()
      .domain(categoryLabels)
      .range([0, innerWidth])
      .padding(0.3);

    const y = d3
      .scaleLinear()
      .domain([d3.min(boxData, (d) => d.min), d3.max(boxData, (d) => d.max)])
      .nice()
      .range([innerHeight, 0]);

    boxData.forEach((d, i) => {
      const center = x(d.label) + x.bandwidth() / 2;
      const boxWidth = x.bandwidth();

      g.append("line")
        .attr("x1", center)
        .attr("x2", center)
        .attr("y1", y(d.min))
        .attr("y2", y(d.max))
        .attr("stroke", "#000")
        .attr("stroke-width", 1);

      g.append("rect")
        .attr("x", x(d.label))
        .attr("y", y(d.q3))
        .attr("width", boxWidth)
        .attr("height", y(d.q1) - y(d.q3))
        .attr("fill", color)
        .attr("stroke", "#000")
        .attr("opacity", 0.7);

      g.append("line")
        .attr("x1", x(d.label))
        .attr("x2", x(d.label) + boxWidth)
        .attr("y1", y(d.median))
        .attr("y2", y(d.median))
        .attr("stroke", "#000")
        .attr("stroke-width", 2);

      [d.min, d.max].forEach((val) => {
        g.append("line")
          .attr("x1", center - boxWidth / 4)
          .attr("x2", center + boxWidth / 4)
          .attr("y1", y(val))
          .attr("y2", y(val))
          .attr("stroke", "#000")
          .attr("stroke-width", 1);
      });

      d.outliers.forEach((outlier) => {
        g.append("circle")
          .attr("cx", center)
          .attr("cy", y(outlier))
          .attr("r", 3)
          .attr("fill", "red")
          .attr("opacity", 0.6);
      });
    });

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // SCATTER PLOT
  // ============================================
  scatter(xData, yData, options = {}) {
    const {
      title = "Scatter Plot",
      xlabel = "X",
      ylabel = "Y",
      color = "#4299e1",
      size = 5,
      labels = null,
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const data = xData.map((x, i) => ({
      x,
      y: yData[i],
      label: labels ? labels[i] : null,
    }));

    const x = d3
      .scaleLinear()
      .domain(d3.extent(xData))
      .nice()
      .range([0, innerWidth]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(yData))
      .nice()
      .range([innerHeight, 0]);

    const tooltip = d3
      .select("body")
      .append("div")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.8)")
      .style("color", "#fff")
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0);

    g.selectAll("circle")
      .data(data)
      .join("circle")
      .attr("cx", (d) => x(d.x))
      .attr("cy", (d) => y(d.y))
      .attr("r", size)
      .attr("fill", color)
      .attr("opacity", 0.7)
      .on("mouseover", function (event, d) {
        d3.select(this)
          .attr("r", size * 1.5)
          .attr("opacity", 1);
        tooltip
          .style("opacity", 1)
          .html(
            `X: ${d.x.toFixed(2)}<br>Y: ${d.y.toFixed(2)}${
              d.label ? "<br>" + d.label : ""
            }`
          )
          .style("left", event.pageX + 10 + "px")
          .style("top", event.pageY - 10 + "px");
      })
      .on("mouseout", function () {
        d3.select(this).attr("r", size).attr("opacity", 0.7);
        tooltip.style("opacity", 0);
      });

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // LINE CHART
  // ============================================
  line(xData, yData, options = {}) {
    const {
      title = "Line Chart",
      xlabel = "X",
      ylabel = "Y",
      color = "#4299e1",
      lineWidth = 2,
      showPoints = true,
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const data = xData.map((x, i) => ({ x, y: yData[i] }));

    const x = d3.scaleLinear().domain(d3.extent(xData)).range([0, innerWidth]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(yData))
      .nice()
      .range([innerHeight, 0]);

    const line = d3
      .line()
      .x((d) => x(d.x))
      .y((d) => y(d.y));

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", lineWidth)
      .attr("d", line);

    if (showPoints) {
      g.selectAll("circle")
        .data(data)
        .join("circle")
        .attr("cx", (d) => x(d.x))
        .attr("cy", (d) => y(d.y))
        .attr("r", 4)
        .attr("fill", color);
    }

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // BAR CHART
  // ============================================
  bar(categories, values, options = {}) {
    const {
      title = "Bar Chart",
      xlabel = "Category",
      ylabel = "Value",
      color = "#4299e1",
      horizontal = false,
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const data = categories.map((cat, i) => ({
      category: cat,
      value: values[i],
    }));

    if (!horizontal) {
      const x = d3
        .scaleBand()
        .domain(categories)
        .range([0, innerWidth])
        .padding(0.2);

      const y = d3
        .scaleLinear()
        .domain([0, d3.max(values)])
        .nice()
        .range([innerHeight, 0]);

      g.selectAll("rect")
        .data(data)
        .join("rect")
        .attr("x", (d) => x(d.category))
        .attr("y", (d) => y(d.value))
        .attr("width", x.bandwidth())
        .attr("height", (d) => innerHeight - y(d.value))
        .attr("fill", color)
        .attr("opacity", 0.8)
        .on("mouseover", function () {
          d3.select(this).attr("opacity", 1);
        })
        .on("mouseout", function () {
          d3.select(this).attr("opacity", 0.8);
        });

      g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "rotate(-45)")
        .style("text-anchor", "end");

      g.append("g").call(d3.axisLeft(y));
    } else {
      const x = d3
        .scaleLinear()
        .domain([0, d3.max(values)])
        .nice()
        .range([0, innerWidth]);

      const y = d3
        .scaleBand()
        .domain(categories)
        .range([0, innerHeight])
        .padding(0.2);

      g.selectAll("rect")
        .data(data)
        .join("rect")
        .attr("x", 0)
        .attr("y", (d) => y(d.category))
        .attr("width", (d) => x(d.value))
        .attr("height", y.bandwidth())
        .attr("fill", color)
        .attr("opacity", 0.8)
        .on("mouseover", function () {
          d3.select(this).attr("opacity", 1);
        })
        .on("mouseout", function () {
          d3.select(this).attr("opacity", 0.8);
        });

      g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

      g.append("g").call(d3.axisLeft(y));
    }

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height - 10)
      .attr("text-anchor", "middle")
      .text(xlabel);

    svg
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // PIE CHART
  // ============================================
  pie(labels, values, options = {}) {
    const {
      title = "Pie Chart",
      width = this.defaultWidth,
      height = this.defaultHeight,
      showLabels = true,
      showPercentage = true,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const radius = Math.min(width, height) / 2 - 40;

    const g = svg
      .append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`);

    const data = labels.map((label, i) => ({ label, value: values[i] }));
    const total = d3.sum(values);

    const color = d3.scaleOrdinal().domain(labels).range(this.colors);

    const pie = d3.pie().value((d) => d.value);

    const arc = d3.arc().innerRadius(0).outerRadius(radius);

    const labelArc = d3
      .arc()
      .innerRadius(radius * 0.7)
      .outerRadius(radius * 0.7);

    const arcs = g
      .selectAll("arc")
      .data(pie(data))
      .join("g")
      .attr("class", "arc");

    arcs
      .append("path")
      .attr("d", arc)
      .attr("fill", (d) => color(d.data.label))
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("opacity", 0.8)
      .on("mouseover", function () {
        d3.select(this).attr("opacity", 1);
      })
      .on("mouseout", function () {
        d3.select(this).attr("opacity", 0.8);
      });

    if (showLabels) {
      arcs
        .append("text")
        .attr("transform", (d) => `translate(${labelArc.centroid(d)})`)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .style("fill", "#fff")
        .text((d) => {
          if (showPercentage) {
            const percentage = ((d.data.value / total) * 100).toFixed(1);
            return `${d.data.label}\n${percentage}%`;
          }
          return d.data.label;
        });
    }

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // HEATMAP
  // ============================================
  heatmap(matrix, options = {}) {
    const {
      title = "Heatmap",
      labels = null,
      colorScheme = "RdYlBu",
      showValues = true,
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = { top: 80, right: 40, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const n = matrix.length;
    const rowLabels =
      labels || Array.from({ length: n }, (_, i) => `Var${i + 1}`);
    const colLabels =
      labels || Array.from({ length: n }, (_, i) => `Var${i + 1}`);

    const data = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        data.push({
          row: i,
          col: j,
          value: matrix[i][j],
          rowLabel: rowLabels[i],
          colLabel: colLabels[j],
        });
      }
    }

    const x = d3
      .scaleBand()
      .domain(colLabels)
      .range([0, innerWidth])
      .padding(0.05);

    const y = d3
      .scaleBand()
      .domain(rowLabels)
      .range([0, innerHeight])
      .padding(0.05);

    const colorScale = d3
      .scaleSequential()
      .domain([-1, 1])
      .interpolator(d3[`interpolate${colorScheme}`]);

    g.selectAll("rect")
      .data(data)
      .join("rect")
      .attr("x", (d) => x(d.colLabel))
      .attr("y", (d) => y(d.rowLabel))
      .attr("width", x.bandwidth())
      .attr("height", y.bandwidth())
      .attr("fill", (d) => colorScale(d.value))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1);

    if (showValues) {
      g.selectAll("text.value")
        .data(data)
        .join("text")
        .attr("class", "value")
        .attr("x", (d) => x(d.colLabel) + x.bandwidth() / 2)
        .attr("y", (d) => y(d.rowLabel) + y.bandwidth() / 2)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "10px")
        .style("fill", (d) => (Math.abs(d.value) > 0.5 ? "#fff" : "#000"))
        .text((d) => d.value.toFixed(2));
    }

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end");

    g.append("g").call(d3.axisLeft(y));

    const legendWidth = 200;
    const legendHeight = 20;
    const legendG = svg
      .append("g")
      .attr(
        "transform",
        `translate(${width - margin.right - legendWidth},${margin.top - 40})`
      );

    const legendScale = d3
      .scaleLinear()
      .domain([-1, 1])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale).ticks(5);

    legendG
      .selectAll("rect")
      .data(d3.range(-1, 1, 0.01))
      .join("rect")
      .attr("x", (d) => legendScale(d))
      .attr("y", 0)
      .attr("width", legendWidth / 200)
      .attr("height", legendHeight)
      .attr("fill", (d) => colorScale(d));

    legendG
      .append("g")
      .attr("transform", `translate(0,${legendHeight})`)
      .call(legendAxis);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // VIOLIN PLOT
  // ============================================
  violin(data, options = {}) {
    const {
      title = "Violin Plot",
      xlabel = "Category",
      ylabel = "Value",
      labels = null,
      color = "#4299e1",
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const datasets = Array.isArray(data[0]) ? data : [data];
    const categoryLabels = labels || datasets.map((_, i) => `Group ${i + 1}`);

    const x = d3
      .scaleBand()
      .domain(categoryLabels)
      .range([0, innerWidth])
      .padding(0.3);

    const allValues = datasets.flat();
    const y = d3
      .scaleLinear()
      .domain(d3.extent(allValues))
      .nice()
      .range([innerHeight, 0]);

    const kde = (data, bandwidth = 0.5) => {
      const thresholds = y.ticks(50);
      return thresholds.map((t) => {
        const density = d3.mean(data, (d) => {
          return (
            Math.exp(-0.5 * Math.pow((d - t) / bandwidth, 2)) /
            (bandwidth * Math.sqrt(2 * Math.PI))
          );
        });
        return [t, density];
      });
    };

    datasets.forEach((dataset, i) => {
      const density = kde(dataset);
      const maxDensity = d3.max(density, (d) => d[1]);

      const xScale = d3
        .scaleLinear()
        .domain([0, maxDensity])
        .range([0, x.bandwidth() / 2]);

      const area = d3
        .area()
        .x0((d) => x(categoryLabels[i]) + x.bandwidth() / 2 - xScale(d[1]))
        .x1((d) => x(categoryLabels[i]) + x.bandwidth() / 2 + xScale(d[1]))
        .y((d) => y(d[0]))
        .curve(d3.curveBasis);

      g.append("path")
        .datum(density)
        .attr("fill", color)
        .attr("opacity", 0.6)
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .attr("d", area);
    });

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // QQ PLOT
  // ============================================
  qqplot(data, options = {}) {
    const {
      title = "Q-Q Plot",
      xlabel = "Theoretical Quantiles",
      ylabel = "Sample Quantiles",
      color = "#4299e1",
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const sorted = [...data].sort((a, b) => a - b);
    const n = sorted.length;

    const theoretical = sorted.map((_, i) => {
      const p = (i + 0.5) / n;
      return this.invNormalCDF(p);
    });

    const qqData = theoretical.map((t, i) => ({ x: t, y: sorted[i] }));

    const x = d3
      .scaleLinear()
      .domain(d3.extent(theoretical))
      .nice()
      .range([0, innerWidth]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(sorted))
      .nice()
      .range([innerHeight, 0]);

    const minVal = Math.max(x.domain()[0], y.domain()[0]);
    const maxVal = Math.min(x.domain()[1], y.domain()[1]);

    g.append("line")
      .attr("x1", x(minVal))
      .attr("y1", y(minVal))
      .attr("x2", x(maxVal))
      .attr("y2", y(maxVal))
      .attr("stroke", "red")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    g.selectAll("circle")
      .data(qqData)
      .join("circle")
      .attr("cx", (d) => x(d.x))
      .attr("cy", (d) => y(d.y))
      .attr("r", 4)
      .attr("fill", color)
      .attr("opacity", 0.7);

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  invNormalCDF(p) {
    const a1 = -39.6968302866538;
    const a2 = 220.946098424521;
    const a3 = -275.928510446969;
    const a4 = 138.357751867269;
    const a5 = -30.6647980661472;
    const a6 = 2.50662827745924;

    const b1 = -54.4760987982241;
    const b2 = 161.585836858041;
    const b3 = -155.698979859887;
    const b4 = 66.8013118877197;
    const b5 = -13.2806815528857;

    const c1 = -0.00778489400243029;
    const c2 = -0.322396458041136;
    const c3 = -2.40075827716184;
    const c4 = -2.54973253934373;
    const c5 = 4.37466414146497;
    const c6 = 2.93816398269878;

    const d1 = 0.00778469570904146;
    const d2 = 0.32246712907004;
    const d3 = 2.445134137143;
    const d4 = 3.75440866190742;

    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q, r, x;

    if (p < pLow) {
      q = Math.sqrt(-2 * Math.log(p));
      x =
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    } else if (p <= pHigh) {
      q = p - 0.5;
      r = q * q;
      x =
        ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x =
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    }

    return x;
  }

  // ============================================
  // DENSITY PLOT
  // ============================================
  density(data, options = {}) {
    const {
      title = "Density Plot",
      xlabel = "Value",
      ylabel = "Density",
      color = "#4299e1",
      bandwidth = null,
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const bw =
      bandwidth || 1.06 * d3.deviation(data) * Math.pow(data.length, -1 / 5);

    const extent = d3.extent(data);
    const range = extent[1] - extent[0];
    const xPoints = d3.range(
      extent[0] - range * 0.1,
      extent[1] + range * 0.1,
      range / 200
    );

    const densityData = xPoints.map((xi) => {
      const density = d3.mean(data, (d) => {
        return (
          Math.exp(-0.5 * Math.pow((d - xi) / bw, 2)) /
          (bw * Math.sqrt(2 * Math.PI))
        );
      });
      return { x: xi, y: density };
    });

    const x = d3
      .scaleLinear()
      .domain([xPoints[0], xPoints[xPoints.length - 1]])
      .range([0, innerWidth]);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(densityData, (d) => d.y)])
      .nice()
      .range([innerHeight, 0]);

    const area = d3
      .area()
      .x((d) => x(d.x))
      .y0(innerHeight)
      .y1((d) => y(d.y))
      .curve(d3.curveBasis);

    const line = d3
      .line()
      .x((d) => x(d.x))
      .y((d) => y(d.y))
      .curve(d3.curveBasis);

    g.append("path")
      .datum(densityData)
      .attr("fill", color)
      .attr("opacity", 0.3)
      .attr("d", area);

    g.append("path")
      .datum(densityData)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2)
      .attr("d", line);

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // PARALLEL COORDINATES
  // ============================================
  parallel(data, dimensions, options = {}) {
    const {
      title = "Parallel Coordinates",
      colors = null,
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = { top: 60, right: 40, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const y = {};
    dimensions.forEach((dim) => {
      y[dim] = d3
        .scaleLinear()
        .domain(d3.extent(data, (d) => d[dim]))
        .range([innerHeight, 0]);
    });

    const x = d3.scalePoint().domain(dimensions).range([0, innerWidth]);

    const line = d3.line();

    const path = (d) => {
      return line(dimensions.map((dim) => [x(dim), y[dim](d[dim])]));
    };

    const colorScale = colors
      ? d3.scaleOrdinal().domain(d3.range(data.length)).range(colors)
      : d3.scaleSequential(d3.interpolateViridis).domain([0, data.length]);

    g.selectAll("path.line")
      .data(data)
      .join("path")
      .attr("class", "line")
      .attr("d", path)
      .attr("fill", "none")
      .attr("stroke", (d, i) => colorScale(i))
      .attr("opacity", 0.3)
      .attr("stroke-width", 2)
      .on("mouseover", function () {
        d3.select(this).attr("opacity", 1).attr("stroke-width", 3);
      })
      .on("mouseout", function () {
        d3.select(this).attr("opacity", 0.3).attr("stroke-width", 2);
      });

    dimensions.forEach((dim) => {
      const axis = g
        .append("g")
        .attr("transform", `translate(${x(dim)},0)`)
        .call(d3.axisLeft(y[dim]));

      axis
        .append("text")
        .attr("y", -10)
        .attr("text-anchor", "middle")
        .style("fill", "#000")
        .style("font-weight", "bold")
        .text(dim);
    });

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // PAIR PLOT
  // ============================================
  pairplot(data, variables, options = {}) {
    const {
      title = "Pair Plot",
      color = "#4299e1",
      size = 3,
      width = 900,
      height = 900,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const n = variables.length;
    const padding = 60;
    const cellSize = (Math.min(width, height) - padding * 2) / n;

    const scales = {};
    variables.forEach((variable) => {
      scales[variable] = d3
        .scaleLinear()
        .domain(d3.extent(data, (d) => d[variable]))
        .range([0, cellSize - 20]);
    });

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const xVar = variables[j];
        const yVar = variables[i];

        const cellG = svg
          .append("g")
          .attr(
            "transform",
            `translate(${padding + j * cellSize},${padding + i * cellSize})`
          );

        if (i === j) {
          const values = data.map((d) => d[xVar]);
          const histogram = d3
            .histogram()
            .domain(scales[xVar].domain())
            .thresholds(20);

          const bins = histogram(values);
          const yScale = d3
            .scaleLinear()
            .domain([0, d3.max(bins, (d) => d.length)])
            .range([cellSize - 20, 0]);

          cellG
            .selectAll("rect")
            .data(bins)
            .join("rect")
            .attr("x", (d) => scales[xVar](d.x0))
            .attr("y", (d) => yScale(d.length))
            .attr("width", (d) => scales[xVar](d.x1) - scales[xVar](d.x0) - 1)
            .attr("height", (d) => cellSize - 20 - yScale(d.length))
            .attr("fill", color)
            .attr("opacity", 0.7);
        } else {
          cellG
            .selectAll("circle")
            .data(data)
            .join("circle")
            .attr("cx", (d) => scales[xVar](d[xVar]))
            .attr("cy", (d) => scales[yVar](d[yVar]))
            .attr("r", size)
            .attr("fill", color)
            .attr("opacity", 0.5);
        }

        if (i === n - 1) {
          cellG
            .append("g")
            .attr("transform", `translate(0,${cellSize - 20})`)
            .call(d3.axisBottom(scales[xVar]).ticks(3));
        }

        if (j === 0) {
          cellG.append("g").call(d3.axisLeft(scales[yVar]).ticks(3));
        }

        if (i === n - 1) {
          cellG
            .append("text")
            .attr("x", (cellSize - 20) / 2)
            .attr("y", cellSize - 5)
            .attr("text-anchor", "middle")
            .style("font-size", "10px")
            .text(xVar);
        }

        if (j === 0) {
          cellG
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -(cellSize - 20) / 2)
            .attr("y", -25)
            .attr("text-anchor", "middle")
            .style("font-size", "10px")
            .text(yVar);
        }
      }
    }

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 30)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }

  // ============================================
  // MULTI-LINE CHART
  // ============================================
  multiline(series, options = {}) {
    const {
      title = "Multi-Line Chart",
      xlabel = "X",
      ylabel = "Y",
      legend = true,
      width = this.defaultWidth,
      height = this.defaultHeight,
      containerId = null,
    } = options;

    const { svg } = this.createContainer(containerId, width, height);
    const margin = this.defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const allX = series.flatMap((s) => s.data.map((d) => d.x));
    const allY = series.flatMap((s) => s.data.map((d) => d.y));

    const x = d3.scaleLinear().domain(d3.extent(allX)).range([0, innerWidth]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(allY))
      .nice()
      .range([innerHeight, 0]);

    const color = d3
      .scaleOrdinal()
      .domain(series.map((s) => s.name))
      .range(this.colors);

    const line = d3
      .line()
      .x((d) => x(d.x))
      .y((d) => y(d.y));

    series.forEach((s) => {
      g.append("path")
        .datum(s.data)
        .attr("fill", "none")
        .attr("stroke", color(s.name))
        .attr("stroke-width", 2)
        .attr("d", line);
    });

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(xlabel);

    g.append("g")
      .call(d3.axisLeft(y))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -40)
      .attr("fill", "#000")
      .attr("text-anchor", "middle")
      .text(ylabel);

    if (legend) {
      const legendG = svg
        .append("g")
        .attr(
          "transform",
          `translate(${width - margin.right - 100},${margin.top})`
        );

      series.forEach((s, i) => {
        const legendItem = legendG
          .append("g")
          .attr("transform", `translate(0,${i * 25})`);

        legendItem
          .append("line")
          .attr("x1", 0)
          .attr("x2", 30)
          .attr("y1", 10)
          .attr("y2", 10)
          .attr("stroke", color(s.name))
          .attr("stroke-width", 2);

        legendItem
          .append("text")
          .attr("x", 35)
          .attr("y", 15)
          .style("font-size", "12px")
          .text(s.name);
      });
    }

    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);

    return this;
  }
}

export default DataViz;
