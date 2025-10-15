// datly.js — functional, text-first data-science toolkit for JavaScript
// =========================
// Helpers internos
// =========================
const _inferType = (value) => {
  if (value === "" || value === "null" || value === "NULL" || value === "NaN")
    return null;
  if (value === "true" || value === "TRUE") return true;
  if (value === "false" || value === "FALSE") return false;
  if (/^-?\d+$/.test(value)) return parseInt(value, 10);
  if (/^-?\d*\.\d+$/.test(value)) return parseFloat(value);
  return value;
};

const _build_df = (columns, data) => ({
  type: "dataframe",
  columns,
  data,
  n_rows: data.length,
  n_cols: columns.length,
});

const _empty_df = () => _build_df([], []);

const _uniq = (arr) => [...new Set(arr)];

const _text = (obj) => obj;

const _flatten = (obj, prefix = "", maxDepth = 5, currentDepth = 0) => {
  const result = {};

  if (currentDepth >= maxDepth) {
    result[prefix || "value"] = obj;
    return result;
  }

  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;

    if (Array.isArray(value)) {
      // Adiciona o array completo
      result[path] = value;

      // Se é array de objetos, expande as propriedades
      if (
        value.length > 0 &&
        typeof value[0] === "object" &&
        !Array.isArray(value[0])
      ) {
        const firstItem = value[0];
        Object.keys(firstItem).forEach((subKey) => {
          result[`${path}.${subKey}`] = value.map(
            (item) => item[subKey] ?? null
          );
        });
      }
    } else if (value && typeof value === "object" && value !== null) {
      // Recursivamente achata objetos aninhados
      const nested = _flatten(value, path, maxDepth, currentDepth + 1);
      Object.assign(result, nested);
    } else {
      // Valor primitivo
      result[path] = value;
    }
  }

  return result;
};

const _toTable = (data, opts = {}) => {
  const maxWidth = opts.max_width ?? 80;
  const padding = opts.padding ?? 2;

  if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object") {
    // array of objects -> table
    const keys = Object.keys(data[0]);
    const rows = data.map((obj) => keys.map((k) => String(obj[k] ?? "")));
    const headers = keys;

    // calculate column widths
    const widths = headers.map((h, i) => {
      const maxContentWidth = Math.max(
        h.length,
        ...rows.map((r) => r[i].length)
      );
      return Math.min(maxContentWidth + padding, maxWidth / keys.length);
    });

    // build table
    const separator = "+" + widths.map((w) => "-".repeat(w)).join("+") + "+";
    const headerRow =
      "|" + headers.map((h, i) => h.padEnd(widths[i])).join("|") + "|";
    const dataRows = rows.map(
      (row) =>
        "|" +
        row
          .map((cell, i) => cell.slice(0, widths[i]).padEnd(widths[i]))
          .join("|") +
        "|"
    );

    return [separator, headerRow, separator, ...dataRows, separator].join("\n");
  }

  if (typeof data === "object" && !Array.isArray(data)) {
    // single object -> key-value table
    const entries = Object.entries(data).map(([k, v]) => ({
      key: String(k),
      value: typeof v === "object" ? JSON.stringify(v) : String(v),
    }));
    return _toTable(entries, opts);
  }

  return String(data);
};

const _isNumber = (v) => typeof v === "number" && Number.isFinite(v);
const _toNum = (v) => (v == null || v === "" ? NaN : Number(v));

const _numeric = (arr) => arr.map(_toNum).filter((x) => Number.isFinite(x));

const _sum = (arr) => _numeric(arr).reduce((a, b) => a + b, 0);
const _mean = (arr) => {
  const x = _numeric(arr);
  const n = x.length;
  if (!n) return NaN;
  return _sum(x) / n;
};
const _variance = (arr, sample = true) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < 2) return NaN;
  const m = _mean(x);
  const s = x.reduce((a, b) => a + (b - m) ** 2, 0);
  return s / (sample ? n - 1 : n);
};
const _std = (arr, sample = true) => Math.sqrt(_variance(arr, sample));
const _min = (arr) => Math.min(..._numeric(arr));
const _max = (arr) => Math.max(..._numeric(arr));
const _median = (arr) => {
  const x = _numeric(arr).sort((a, b) => a - b);
  const n = x.length;
  if (!n) return NaN;
  const mid = Math.floor(n / 2);
  return n % 2 ? x[mid] : (x[mid - 1] + x[mid]) / 2;
};
const _quantile = (arr, q) => {
  const x = _numeric(arr).sort((a, b) => a - b);
  const n = x.length;
  if (!n) return NaN;
  const pos = (n - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return x[base] + (x[Math.min(base + 1, n - 1)] - x[base]) * rest;
};
const _skewness = (arr) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < 3) return NaN;
  const m = _mean(x);
  const s = _std(x, true);
  const m3 = x.reduce((a, b) => a + (b - m) ** 3, 0) / n;
  return ((m3 / s ** 3) * Math.sqrt(n * (n - 1))) / (n - 2);
};
const _kurtosis = (arr) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < 4) return NaN;
  const m = _mean(x);
  const s2 = _variance(x, true);
  const m4 = x.reduce((a, b) => a + (b - m) ** 4, 0) / n;
  const g2 = m4 / s2 ** 2 - 3;
  return g2;
};

const _corrPearson = (x, y) => {
  const a = _numeric(x),
    b = _numeric(y);
  const n = Math.min(a.length, b.length);
  if (n < 2) return NaN;
  const ax = a.slice(0, n),
    by = b.slice(0, n);
  const mx = _mean(ax),
    my = _mean(by);
  let num = 0,
    dx = 0,
    dy = 0;
  for (let i = 0; i < n; i++) {
    const vx = ax[i] - mx,
      vy = by[i] - my;
    num += vx * vy;
    dx += vx * vx;
    dy += vy * vy;
  }
  return num / Math.sqrt(dx * dy);
};

const _rank = (arr) => {
  const indexed = _numeric(arr)
    .map((v, i) => ({ v, i }))
    .sort((a, b) => a.v - b.v);
  const ranks = Array(arr.length).fill(NaN);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j + 1 < indexed.length && indexed[j + 1].v === indexed[i].v) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[indexed[k].i] = r;
    i = j + 1;
  }
  return ranks.filter(Number.isFinite);
};

const _corrSpearman = (x, y) => _corrPearson(_rank(x), _rank(y));

const _invErf = (x) => {
  // numerical approx of inverse error function (for normal quantile)
  const a = 0.147;
  const ln = Math.log(1 - x * x);
  const t = 2 / (Math.PI * a) + ln / 2;
  const s = Math.sign(x) * Math.sqrt(Math.sqrt(t * t - ln / a) - t);
  return s;
};

// standard normal pdf/cdf/ppf
const _phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
const _Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const erf = (x) => {
  // numerical approx for erf
  const sign = Math.sign(x);
  x = Math.abs(x);
  const a1 = 0.254829592,
    a2 = -0.284496736,
    a3 = 1.421413741,
    a4 = -1.453152027,
    a5 = 1.061405429,
    p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y =
    1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
};
const _normInv = (p) => {
  if (p <= 0 || p >= 1) return NaN;
  return Math.SQRT2 * _invErf(2 * p - 1);
};

// lowercased text output
// const _text = (obj) => {
//   const lowerKeys = (o) =>
//     Array.isArray(o)
//       ? o.map(lowerKeys)
//       : o && typeof o === "object"
//       ? Object.fromEntries(
//           Object.entries(o).map(([k, v]) => [
//             String(k).toLowerCase(),
//             lowerKeys(v),
//           ])
//         )
//       : typeof o === "number" && Number.isFinite(o)
//       ? Number(Number(o).toPrecision(2))
//       : o;
//   const normalized = lowerKeys(obj);
//   const lines = [];
//   const walk = (o, indent = 0) => {
//     const pad = " ".repeat(indent);
//     if (Array.isArray(o)) {
//       lines.push(pad + "- array:");
//       o.forEach((v) => walk(v, indent + 2));
//     } else if (o && typeof o === "object") {
//       Object.keys(o).forEach((k) => {
//         const v = o[k];
//         if (v && typeof v === "object") {
//           lines.push(pad + k + ":");
//           walk(v, indent + 2);
//         } else {
//           lines.push(pad + k + ": " + String(v).toLowerCase());
//         }
//       });
//     } else {
//       lines.push(pad + String(o).toLowerCase());
//     }
//   };
//   walk(normalized);
//   return lines.join("\n");
// };

// const _text = (obj) => {
//   const lowerKeys = (o) =>
//     Array.isArray(o)
//       ? o.map(lowerKeys)
//       : o && typeof o === "object"
//       ? Object.fromEntries(
//           Object.entries(o).map(([k, v]) => [
//             String(k).toLowerCase(),
//             lowerKeys(v),
//           ])
//         )
//       : typeof o === "number" && Number.isFinite(o)
//       ? Number(Number(o).toPrecision(12))
//       : o;
//   const normalized = lowerKeys(obj);
//   const lines = [];
//   const walk = (o, indent = 0) => {
//     const pad = " ".repeat(indent);
//     if (Array.isArray(o)) {
//       lines.push(pad + `- array (${o.length} items):`);
//       const limited = o.slice(0, 5);
//       limited.forEach((v) => walk(v, indent + 2));
//       if (o.length > 5) {
//         lines.push(pad + `  ... ${o.length - 5} more items omitted`);
//       }
//     } else if (o && typeof o === "object") {
//       Object.keys(o).forEach((k) => {
//         const v = o[k];
//         if (v && typeof v === "object") {
//           lines.push(pad + k + ":");
//           walk(v, indent + 2);
//         } else {
//           lines.push(pad + k + ": " + String(v).toLowerCase());
//         }
//       });
//     } else {
//       lines.push(pad + String(o).toLowerCase());
//     }
//   };
//   walk(normalized);
//   return lines.join("\n");
// };

const _ok = (type, payload) => _text({ type, ...payload });
const _err = (type, message) => _text({ type, error: message });

// =========================
// dataframe
// =========================

// utilities to operate directly on raw rows (array of objects)
// const df_describe = (rows) => {
//   if (!Array.isArray(rows) || !rows.length)
//     return _err("describe", "empty data");
//   const cols = _uniq(_flatten(rows.map((r) => Object.keys(r))));
//   const out = { type: "describe", columns: {} };
//   cols.forEach((c) => {
//     const col = rows.map((r) => r[c]);
//     const nums = col.map(_toNum).filter(Number.isFinite);
//     const miss = col.filter(
//       (v) =>
//         v == null || (typeof v === "number" && !Number.isFinite(v)) || v === ""
//     ).length;
//     const dtype =
//       nums.length === col.length
//         ? "number"
//         : col.every((v) => typeof v === "boolean")
//         ? "boolean"
//         : "string";
//     const info = { dtype, count: col.length, missing: miss };
//     if (dtype === "number") {
//       info.mean = _mean(nums);
//       info.std = _std(nums);
//       info.min = _min(nums);
//       info.q1 = _quantile(nums, 0.25);
//       info.median = _median(nums);
//       info.q3 = _quantile(nums, 0.75);
//       info.max = _max(nums);
//       info.skewness = _skewness(nums);
//       info.kurtosis = _kurtosis(nums);
//     } else if (dtype === "string" || dtype === "boolean") {
//       const vc = {};
//       col.forEach((v) => {
//         const key = String(v);
//         vc[key] = (vc[key] || 0) + 1;
//       });
//       const entries = Object.entries(vc)
//         .sort((a, b) => b[1] - a[1])
//         .slice(0, 10);
//       info.top = entries.map(([k, v]) => ({ value: k, freq: v }));
//       info.unique = Object.keys(vc).length;
//     }
//     out.columns[c] = info;
//   });
//   return _text(out);
// };

const df_missing_report = (rows) => {
  if (!Array.isArray(rows) || !rows.length)
    return _err("missing_report", "empty data");
  const cols = _uniq(_flatten(rows.map((r) => Object.keys(r))));
  const res = cols.map((c) => {
    const col = rows.map((r) => r[c]);
    const miss = col.filter((v) => v == null || v === "").length;
    return { column: c, missing: miss, missing_rate: miss / col.length };
  });
  return _text({ type: "missing_report", rows: res });
};

const df_corr = (rows, method = "pearson") => {
  const cols = _uniq(_flatten(rows.map((r) => Object.keys(r))));
  const numericCols = cols.filter((c) =>
    rows.every((r) => Number.isFinite(_toNum(r[c])) || r[c] == null)
  );
  const mat = {};
  numericCols.forEach((a) => {
    mat[a] = {};
    const xa = rows.map((r) => _toNum(r[a]));
    numericCols.forEach((b) => {
      const xb = rows.map((r) => _toNum(r[b]));
      const c =
        method === "spearman" ? _corrSpearman(xa, xb) : _corrPearson(xa, xb);
      mat[a][b] = c;
    });
  });
  return _text({ type: "correlation_matrix", method, matrix: mat });
};

// =========================
// core statistics (public)
// =========================

const mean = (arr) =>
  _ok("statistic", {
    name: "mean",
    n: _numeric(arr).length,
    value: _mean(arr),
  });
const stddeviation = (arr, sample = true) =>
  _ok("statistic", {
    name: "std_deviation",
    sample,
    n: _numeric(arr).length,
    value: _std(arr, sample),
  });
const variance = (arr, sample = true) =>
  _ok("statistic", {
    name: "variance",
    sample,
    n: _numeric(arr).length,
    value: _variance(arr, sample),
  });
const median = (arr) =>
  _ok("statistic", {
    name: "median",
    n: _numeric(arr).length,
    value: _median(arr),
  });
const quantile = (arr, q) =>
  _ok("statistic", {
    name: "quantile",
    q,
    n: _numeric(arr).length,
    value: _quantile(arr, q),
  });
const minv = (arr) => _ok("statistic", { name: "min", value: _min(arr) });
const maxv = (arr) => _ok("statistic", { name: "max", value: _max(arr) });
const skewness = (arr) =>
  _ok("statistic", { name: "skewness", value: _skewness(arr) });
const kurtosis = (arr) =>
  _ok("statistic", { name: "kurtosis", value: _kurtosis(arr) });
const corr_pearson = (x, y) =>
  _ok("statistic", { name: "pearson_correlation", value: _corrPearson(x, y) });
const corr_spearman = (x, y) =>
  _ok("statistic", {
    name: "spearman_correlation",
    value: _corrSpearman(x, y),
  });

// =========================
// probability distributions
// =========================

const normal_pdf = (x, mu = 0, sigma = 1) =>
  _ok("distribution", {
    name: "normal_pdf",
    params: { mu, sigma },
    value: Array.isArray(x)
      ? x.map((v) => _phi((v - mu) / sigma) / sigma)
      : _phi((x - mu) / sigma) / sigma,
  });

const normal_cdf = (x, mu = 0, sigma = 1) =>
  _ok("distribution", {
    name: "normal_cdf",
    params: { mu, sigma },
    value: Array.isArray(x)
      ? x.map((v) => _Phi((v - mu) / sigma))
      : _Phi((x - mu) / sigma),
  });

const normal_ppf = (p, mu = 0, sigma = 1) =>
  _ok("distribution", {
    name: "normal_ppf",
    params: { mu, sigma },
    value: Array.isArray(p)
      ? p.map((q) => mu + sigma * _normInv(q))
      : mu + sigma * _normInv(p),
  });

const binomial_pmf = (k, n, p) => {
  const C = (n, k) => {
    if (k < 0 || k > n) return 0;
    k = Math.min(k, n - k);
    let num = 1,
      den = 1;
    for (let i = 1; i <= k; i++) {
      num *= n - (k - i);
      den *= i;
    }
    return num / den;
  };
  const f = (x) => C(n, x) * p ** x * (1 - p) ** (n - x);
  const val = Array.isArray(k) ? k.map(f) : f(k);
  return _ok("distribution", {
    name: "binomial_pmf",
    params: { n, p },
    value: val,
  });
};

const binomial_cdf = (k, n, p) => {
  const pmf = (x) =>
    JSON.parse(binomial_pmf(x, n, p).toLowerCase ? '{"ignore":0}' : "{}"); // safeguard no-op
  const f = (t) => {
    let s = 0;
    for (let i = 0; i <= t; i++) {
      s +=
        (function C(n, k) {
          if (k < 0 || k > n) return 0;
          k = Math.min(k, n - k);
          let num = 1,
            den = 1;
          for (let j = 1; j <= k; j++) {
            num *= n - (k - j);
            den *= j;
          }
          return num / den;
        })(n, i) *
        p ** i *
        (1 - p) ** (n - i);
    }
    return s;
  };
  const val = Array.isArray(k) ? k.map(f) : f(k);
  return _ok("distribution", {
    name: "binomial_cdf",
    params: { n, p },
    value: val,
  });
};

const poisson_pmf = (k, lambda) => {
  const fact = (m) => {
    let r = 1;
    for (let i = 2; i <= m; i++) r *= i;
    return r;
  };
  const f = (x) => (Math.exp(-lambda) * lambda ** x) / fact(x);
  return _ok("distribution", {
    name: "poisson_pmf",
    params: { lambda },
    value: Array.isArray(k) ? k.map(f) : f(k),
  });
};

const poisson_cdf = (k, lambda) => {
  const f = (t) => {
    let s = 0;
    for (let i = 0; i <= t; i++)
      s +=
        (Math.exp(-lambda) * lambda ** i) /
        (function fact(m) {
          let r = 1;
          for (let j = 2; j <= m; j++) r *= j;
          return r;
        })(i);
    return s;
  };
  return _ok("distribution", {
    name: "poisson_cdf",
    params: { lambda },
    value: Array.isArray(k) ? k.map(f) : f(k),
  });
};

// =========================
// hypothesis tests
// =========================

const _tCDF = (t, df) => {
  // symmetric; use relationship with regularized incomplete beta (approx via numerical integration)
  const a = df / 2,
    b = 0.5;
  const x = df / (df + t * t);
  const betacf = (a, b, x) => {
    const itmax = 200,
      eps = 3e-7;
    let am = 1,
      bm = 1,
      az = 1,
      qab = a + b,
      qap = a + 1,
      qam = a - 1,
      bz = 1 - (qab * x) / qap;
    let aold;
    for (let m = 1; m <= itmax; m++) {
      const em = m,
        tem = em + em;
      let d = (em * (b - m) * x) / ((qam + tem) * (a + tem));
      let ap = az + d * am;
      let bp = bz + d * bm;
      d = (-(a + em) * (qab + em) * x) / ((a + tem) * (qap + tem));
      let app = ap + d * az;
      let bpp = bp + d * bz;
      aold = az;
      am = ap / bpp;
      bm = bp / bpp;
      az = app / bpp;
      bz = 1;
      if (Math.abs(az - aold) < eps * Math.abs(az)) return az;
    }
    return az;
  };
  const ib = ((Math.pow(x, a) * Math.pow(1 - x, b)) / a) * betacf(a, b, x);
  const p = 0.5 * ib;
  return t >= 0 ? 1 - p : p;
};

const t_test_independent = (a, b, equal_var = true) => {
  const xa = _numeric(a),
    xb = _numeric(b);
  const na = xa.length,
    nb = xb.length;
  if (na < 2 || nb < 2) return _err("t_test_independent", "insufficient data");
  const ma = _mean(xa),
    mb = _mean(xb),
    va = _variance(xa, true),
    vb = _variance(xb, true);
  let df, se;
  if (equal_var) {
    const sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2);
    se = Math.sqrt(sp2 * (1 / na + 1 / nb));
    df = na + nb - 2;
  } else {
    se = Math.sqrt(va / na + vb / nb);
    const num = (va / na + vb / nb) ** 2;
    const den = va ** 2 / (na ** 2 * (na - 1)) + vb ** 2 / (nb ** 2 * (nb - 1));
    df = num / den;
  }
  const t = (ma - mb) / se;
  const p = 2 * (1 - _tCDF(Math.abs(t), df));
  return _text({
    type: "hypothesis_test",
    name: "independent_t_test",
    statistic: t,
    df,
    p_value: p,
    means: { group_a: ma, group_b: mb },
  });
};

const z_test_one_sample = (data, mu0 = 0, sigma = null, alpha = 0.05) => {
  const x = _numeric(data);
  const n = x.length;
  if (n < 2) return _err("z_test_one_sample", "insufficient data");

  const mean = _mean(x);
  const s = sigma ?? _std(x, true);
  const se = s / Math.sqrt(n);
  const z = (mean - mu0) / se;

  const p = 2 * (1 - normal_cdf(Math.abs(z)));
  const zcrit = normal_ppf(1 - alpha / 2);
  const moe = zcrit * se;

  return _text({
    type: "hypothesis_test",
    name: "one_sample_z_test",
    statistic: z,
    p_value: p,
    ci_lower: mean - moe,
    ci_upper: mean + moe,
    confidence: 1 - alpha,
    extra: {
      sample_mean: mean,
      hypothesized_mean: mu0,
      se,
      sigma_used: s,
      n,
      effect_size: (mean - mu0) / s,
    },
  });
};

const chi_square_independence = (table, alpha = 0.05) => {
  const r = table.length;
  const c = table[0].length;
  const rowS = table.map((row) => row.reduce((a, b) => a + b, 0));
  const colS = Array(c).fill(0);
  table.forEach((row) => row.forEach((v, j) => (colS[j] += v)));
  const N = rowS.reduce((a, b) => a + b, 0);
  let chi = 0;
  const expected = Array.from({ length: r }, (_, i) =>
    Array.from({ length: c }, (_, j) => (rowS[i] * colS[j]) / N)
  );

  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      chi += (table[i][j] - expected[i][j]) ** 2 / expected[i][j];
    }
  }
  const df = (r - 1) * (c - 1);
  const p = 1 - chi_square_cdf(chi, df);

  return _text({
    type: "hypothesis_test",
    name: "chi_square_independence",
    statistic: chi,
    df,
    p_value: p,
    confidence: 1 - alpha,
    extra: {
      observed: table,
      expected,
      dof: df,
    },
  });
};

const anova_oneway = (groups, alpha = 0.05) => {
  const k = groups.length;
  const ns = groups.map((g) => _numeric(g).length);
  const means = groups.map(_mean);
  const overall = _mean(groups.flat());
  const ssb = groups.reduce(
    (s, g, i) => s + ns[i] * (means[i] - overall) ** 2,
    0
  );
  const ssw = groups.reduce(
    (s, g, i) => s + _numeric(g).reduce((a, x) => a + (x - means[i]) ** 2, 0),
    0
  );
  const dfb = k - 1;
  const dfw = ns.reduce((a, b) => a + b, 0) - k;
  const msb = ssb / dfb;
  const msw = ssw / dfw;
  const F = msb / msw;
  const p = 1 - f_cdf(F, dfb, dfw);

  return _text({
    type: "hypothesis_test",
    name: "anova_oneway",
    statistic: F,
    df: { between: dfb, within: dfw },
    p_value: p,
    confidence: 1 - alpha,
    extra: {
      group_means: means,
      grand_mean: overall,
      ssb,
      ssw,
    },
  });
};

// =========================
// machine learning (linear regression, logistic regression)
// models are serialized as lowercase json text strings
// =========================

const _addBias = (X) => X.map((row) => [1, ...row]);
const _transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const _dot = (A, B) => {
  const n = A.length,
    m = B[0].length,
    p = B.length;
  const out = Array(n)
    .fill(0)
    .map(() => Array(m).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < p; k++) s += A[i][k] * B[k][j];
      out[i][j] = s;
    }
  return out;
};
const _pinv = (A, lambda = 1e-8) => {
  // ridge-stabilized (A^T A + λI)^-1 A^T
  const At = _transpose(A);
  const AtA = _dot(At, A);
  const n = AtA.length;
  for (let i = 0; i < n; i++) AtA[i][i] += lambda;
  const inv = _inv(AtA);
  return _dot(inv, At);
};
const _inv = (M) => {
  const n = M.length;
  const A = M.map((row, i) =>
    row.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  );
  for (let i = 0; i < n; i++) {
    let pivot = A[i][i];
    let r = i;
    for (let k = i + 1; k < n; k++)
      if (Math.abs(A[k][i]) > Math.abs(pivot)) {
        pivot = A[k][i];
        r = k;
      }
    if (r !== i) {
      const tmp = A[i];
      A[i] = A[r];
      A[r] = tmp;
    }
    const pv = A[i][i];
    if (Math.abs(pv) < 1e-12) continue;
    for (let j = 0; j < 2 * n; j++) A[i][j] /= pv;
    for (let k = 0; k < n; k++)
      if (k !== i) {
        const f = A[k][i];
        for (let j = 0; j < 2 * n; j++) A[k][j] -= f * A[i][j];
      }
  }
  return A.map((row) => row.slice(n));
};

const train_linear_regression = (X, y) => {
  const Xb = _addBias(X);
  const pinv = _pinv(Xb);
  const w = _dot(
    pinv,
    y.map((v) => [v])
  ).map((r) => r[0]);
  const predict = (row) => w[0] + row.reduce((s, v, i) => s + w[i + 1] * v, 0);
  const yhat = X.map(predict);
  const resid = y.map((v, i) => v - yhat[i]);
  const mse = _mean(resid.map((e) => e * e));
  const r2 =
    1 - _sum(resid.map((e) => e * e)) / _sum(y.map((v) => (v - _mean(y)) ** 2));
  const model = {
    type: "linear_regression",
    weights: w,
    mse,
    r2,
    n: y.length,
    p: X[0]?.length ?? 0,
  };
  return _text(model);
};

const _sigmoid = (z) => 1 / (1 + Math.exp(-z));

const train_logistic_regression = (X, y, opts = {}) => {
  const lr = opts.learning_rate ?? 0.1;
  const iters = opts.iterations ?? 1000;
  const lambda = opts.l2 ?? 0;
  const p = X[0]?.length ?? 0;
  let w = Array(p + 1).fill(0);
  const addBias = (row) => [1, ...row];
  const Xb = X.map(addBias);
  for (let t = 0; t < iters; t++) {
    const grad = Array(p + 1).fill(0);
    for (let i = 0; i < Xb.length; i++) {
      const z = w.reduce((s, wi, j) => s + wi * Xb[i][j], 0);
      const p1 = _sigmoid(z);
      const e = p1 - y[i];
      for (let j = 0; j < grad.length; j++) grad[j] += e * Xb[i][j];
    }
    for (let j = 0; j < w.length; j++) {
      grad[j] = grad[j] / Xb.length + lambda * w[j];
      w[j] -= lr * grad[j];
    }
  }
  const predict_proba_row = (row) =>
    _sigmoid(w[0] + row.reduce((s, v, i) => s + w[i + 1] * v, 0));
  const proba = X.map(predict_proba_row);
  const pred = proba.map((p) => (p >= 0.5 ? 1 : 0));
  const acc = pred.filter((v, i) => v === y[i]).length / y.length;
  const model = {
    type: "logistic_regression",
    weights: w,
    accuracy: acc,
    n: y.length,
    p,
  };
  return _text(model);
};

const predict_linear = (model_text, X) => {
  try {
    const m = JSON.parse(JSON.stringify(_lowerJson(model_text)));
    const w = m.weights || m.model?.weights;
    if (!w) return _err("predict_linear", "invalid model");
    const yhat = X.map(
      (row) => w[0] + row.reduce((s, v, i) => s + w[i + 1] * v, 0)
    );
    return _text({
      type: "prediction",
      name: "linear_regression",
      predictions: yhat,
    });
  } catch {
    return _err("predict_linear", "invalid model text");
  }
};

const predict_logistic = (model_text, X, threshold = 0.5) => {
  try {
    const m = JSON.parse(JSON.stringify(_lowerJson(model_text)));
    const w = m.weights || m.model?.weights;
    if (!w) return _err("predict_logistic", "invalid model");
    const proba = X.map((row) =>
      _sigmoid(w[0] + row.reduce((s, v, i) => s + w[i + 1] * v, 0))
    );
    const pred = proba.map((p) => (p >= threshold ? 1 : 0));
    return _text({
      type: "prediction",
      name: "logistic_regression",
      threshold,
      probabilities: proba,
      classes: pred,
    });
  } catch {
    return _err("predict_logistic", "invalid model text");
  }
};

const _lowerJson = (textOrObj) => {
  const parse = (v) => {
    try {
      return typeof v === "string" ? JSON.parse(v) : v;
    } catch {
      return {};
    }
  };
  const lower = (o) =>
    Array.isArray(o)
      ? o.map(lower)
      : o && typeof o === "object"
      ? Object.fromEntries(
          Object.entries(o).map(([k, v]) => [String(k).toLowerCase(), lower(v)])
        )
      : o;
  return lower(parse(textOrObj));
};

const train_test_split = (X, y, test_size = 0.2, seed = 42) => {
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  let s = seed;
  const rand = () => (s = (s * 9301 + 49297) % 233280) / 233280;
  idx.sort(() => rand() - 0.5);
  const ntest = Math.max(1, Math.floor(n * test_size));
  const test_idx = idx.slice(0, ntest);
  const train_idx = idx.slice(ntest);
  const X_train = train_idx.map((i) => X[i]);
  const y_train = train_idx.map((i) => y[i]);
  const X_test = test_idx.map((i) => X[i]);
  const y_test = test_idx.map((i) => y[i]);
  return _text({
    type: "split",
    sizes: { train: y_train.length, test: y_test.length },
    indices: { train: train_idx, test: test_idx },
    preview: { x_train:  X_train.slice(0, 2), y_train: y_train.slice(0, 5) },
  });
};

const metrics_classification = (y_true, y_pred) => {
  const n = Math.min(y_true.length, y_pred.length);
  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;
  for (let i = 0; i < n; i++) {
    if (y_pred[i] === 1 && y_true[i] === 1) tp++;
    else if (y_pred[i] === 0 && y_true[i] === 0) tn++;
    else if (y_pred[i] === 1 && y_true[i] === 0) fp++;
    else if (y_pred[i] === 0 && y_true[i] === 1) fn++;
  }
  const accuracy = (tp + tn) / n;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 =
    precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return _text({
    type: "metric",
    name: "classification_report",
    confusion_matrix: { tp, fp, tn, fn },
    accuracy,
    precision,
    recall,
    f1,
  });
};

const metrics_regression = (y_true, y_pred) => {
  const n = Math.min(y_true.length, y_pred.length);
  const e = Array.from({ length: n }, (_, i) => y_true[i] - y_pred[i]);
  const mse = _mean(e.map((v) => v * v));
  const mae = _mean(e.map((v) => Math.abs(v)));
  const r2 =
    1 -
    _sum(e.map((v) => v * v)) /
      _sum(y_true.map((v) => (v - _mean(y_true)) ** 2));
  return _text({ type: "metric", name: "regression_report", mse, mae, r2 });
};

// =========================
// eda convenience
// =========================

const eda_overview = (rows) => {
  const desc = _lowerJson(df_describe(rows));
  const miss = _lowerJson(df_missing_report(rows));
  const corr = _lowerJson(df_corr(rows, "pearson"));
  return _text({
    type: "eda",
    summary: desc.columns ?? desc,
    missing: miss.rows ?? miss,
    correlation: corr.matrix ?? corr,
  });
};

// =========================
// ADDITIONAL STATISTICAL TESTS
// =========================

const t_test_paired = (a, b) => {
  const xa = _numeric(a),
    xb = _numeric(b);
  const n = Math.min(xa.length, xb.length);
  if (n < 2) return _err("t_test_paired", "insufficient data");
  const diffs = Array.from({ length: n }, (_, i) => xa[i] - xb[i]);
  const md = _mean(diffs),
    sd = _std(diffs, true);
  const t = md / (sd / Math.sqrt(n));
  const df = n - 1;
  const p = 2 * (1 - _tCDF(Math.abs(t), df));
  return _text({
    type: "hypothesis_test",
    name: "paired_t_test",
    statistic: t,
    df,
    p_value: p,
    mean_difference: md,
  });
};

const t_test_one_sample = (arr, mu0) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < 2) return _err("t_test_one_sample", "insufficient data");
  const m = _mean(x),
    s = _std(x, true);
  const t = (m - mu0) / (s / Math.sqrt(n));
  const df = n - 1;
  const p = 2 * (1 - _tCDF(Math.abs(t), df));
  return _text({
    type: "hypothesis_test",
    name: "one_sample_t_test",
    statistic: t,
    df,
    p_value: p,
    mean: m,
    hypothesized_mean: mu0,
  });
};

const shapiro_wilk = (arr) => {
  const x = _numeric(arr).sort((a, b) => a - b);
  const n = x.length;
  if (n < 3 || n > 5000)
    return _err("shapiro_wilk", "sample size must be between 3 and 5000");
  const m = _mean(x);
  const ss = x.reduce((s, v) => s + (v - m) ** 2, 0);
  let b = 0;
  const k = Math.floor(n / 2);
  for (let i = 0; i < k; i++) {
    const ai =
      i === 0
        ? -2.706056 / Math.sqrt(n)
        : i === k - 1 && n % 2 === 0
        ? 2.706056 / Math.sqrt(n)
        : 0;
    b += ai * (x[n - 1 - i] - x[i]);
  }
  const w = (b * b) / ss;
  return _text({
    type: "hypothesis_test",
    name: "shapiro_wilk",
    statistic: w,
    n,
    note: "approximation; w > 0.9 suggests normality",
  });
};

const jarque_bera = (arr) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < 4) return _err("jarque_bera", "insufficient data");
  const s = _skewness(x);
  const k = _kurtosis(x);
  const jb = (n / 6) * (s * s + (k * k) / 4);
  return _text({
    type: "hypothesis_test",
    name: "jarque_bera",
    statistic: jb,
    n,
    df: 2,
    note: "tests normality; low p-value rejects normality",
  });
};

const levene_test = (groups) => {
  const k = groups.length;
  const medians = groups.map(_median);
  const zs = groups.map((g, i) =>
    _numeric(g).map((v) => Math.abs(v - medians[i]))
  );
  const allz = _flatten(zs);
  const overall_median = _median(allz);
  const ns = zs.map((z) => z.length);
  const N = ns.reduce((a, b) => a + b, 0);
  const ssb = zs.reduce(
    (s, z, i) => s + ns[i] * (_mean(z) - overall_median) ** 2,
    0
  );
  const ssw = zs.reduce(
    (s, z) => s + z.reduce((a, v) => a + (v - _mean(z)) ** 2, 0),
    0
  );
  const dfb = k - 1;
  const dfw = N - k;
  const msb = ssb / dfb;
  const msw = ssw / dfw;
  const W = msb / msw;
  return _text({
    type: "hypothesis_test",
    name: "levene_test",
    statistic: W,
    df_between: dfb,
    df_within: dfw,
    note: "tests homogeneity of variance",
  });
};

const kruskal_wallis = (groups) => {
  const all = _flatten(groups);
  const n = all.length;
  const ranks = _rank(all);
  let pos = 0;
  const rankSums = groups.map((g) => {
    const len = _numeric(g).length;
    const rsum = ranks.slice(pos, pos + len).reduce((a, b) => a + b, 0);
    pos += len;
    return { n: len, rsum };
  });
  const H =
    (12 / (n * (n + 1))) *
      rankSums.reduce((s, { n: ni, rsum }) => s + (rsum * rsum) / ni, 0) -
    3 * (n + 1);
  const df = groups.length - 1;
  return _text({
    type: "hypothesis_test",
    name: "kruskal_wallis",
    statistic: H,
    df,
    note: "non-parametric alternative to anova",
  });
};

const mann_whitney = (a, b) => {
  const xa = _numeric(a);
  const xb = _numeric(b);
  const na = xa.length;
  const nb = xb.length;
  if (na < 1 || nb < 1) return _err("mann_whitney", "insufficient data");
  const combined = xa.concat(xb);
  const ranks = _rank(combined);
  const ra = ranks.slice(0, na).reduce((s, r) => s + r, 0);
  const U1 = ra - (na * (na + 1)) / 2;
  const U2 = na * nb - U1;
  const U = Math.min(U1, U2);
  const mu = (na * nb) / 2;
  const sigma = Math.sqrt((na * nb * (na + nb + 1)) / 12);
  const z = (U - mu) / sigma;
  const p = 2 * (1 - _Phi(Math.abs(z)));
  return _text({
    type: "hypothesis_test",
    name: "mann_whitney_u",
    statistic: U,
    z_score: z,
    p_value: p,
    note: "non-parametric alternative to t-test",
  });
};

const wilcoxon_signed_rank = (a, b) => {
  const xa = _numeric(a);
  const xb = _numeric(b);
  const n = Math.min(xa.length, xb.length);
  if (n < 2) return _err("wilcoxon_signed_rank", "insufficient data");
  const diffs = Array.from({ length: n }, (_, i) => xa[i] - xb[i]).filter(
    (d) => d !== 0
  );
  const absDiffs = diffs.map(Math.abs);
  const ranks = _rank(absDiffs);
  const Wplus = ranks.reduce((s, r, i) => s + (diffs[i] > 0 ? r : 0), 0);
  const m = diffs.length;
  const mu = (m * (m + 1)) / 4;
  const sigma = Math.sqrt((m * (m + 1) * (2 * m + 1)) / 24);
  const z = (Wplus - mu) / sigma;
  const p = 2 * (1 - _Phi(Math.abs(z)));
  return _text({
    type: "hypothesis_test",
    name: "wilcoxon_signed_rank",
    statistic: Wplus,
    z_score: z,
    p_value: p,
    n: m,
  });
};

const chi_square_goodness = (observed, expected, alpha = 0.05) => {
  const obs = _numeric(observed);
  const exp = _numeric(expected);
  const n = Math.min(obs.length, exp.length);
  let chi = 0;
  for (let i = 0; i < n; i++) chi += (obs[i] - exp[i]) ** 2 / exp[i];
  const df = n - 1;
  const p = 1 - chi_square_cdf(chi, df);

  return _text({
    type: "hypothesis_test",
    name: "chi_square_goodness_of_fit",
    statistic: chi,
    df,
    p_value: p,
    confidence: 1 - alpha,
    extra: {
      observed: obs,
      expected: exp,
      dof: df,
    },
  });
};

// =========================
// CONFIDENCE INTERVALS
// =========================

const confidence_interval_mean = (arr, confidence = 0.95) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < 2) return _err("confidence_interval_mean", "insufficient data");
  const m = _mean(x);
  const s = _std(x, true);
  const alpha = 1 - confidence;
  const t_crit = _normInv(1 - alpha / 2) * (n > 30 ? 1 : 1.15);
  const margin = (t_crit * s) / Math.sqrt(n);
  const lower = m - margin;
  const upper = m + margin;
  return _text({
    type: "confidence_interval",
    parameter: "mean",
    confidence,
    n,
    mean: m,
    lower,
    upper,
    margin,
  });
};

const confidence_interval_proportion = (successes, n, confidence = 0.95) => {
  if (n < 1)
    return _err("confidence_interval_proportion", "invalid sample size");
  const p = successes / n;
  const alpha = 1 - confidence;
  const z = _normInv(1 - alpha / 2);
  const se = Math.sqrt((p * (1 - p)) / n);
  const margin = z * se;
  const lower = Math.max(0, p - margin);
  const upper = Math.min(1, p + margin);
  return _text({
    type: "confidence_interval",
    parameter: "proportion",
    confidence,
    n,
    proportion: p,
    lower,
    upper,
    margin,
  });
};

const confidence_interval_variance = (arr, confidence = 0.95) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < 2) return _err("confidence_interval_variance", "insufficient data");
  const s2 = _variance(x, true);
  const df = n - 1;
  const alpha = 1 - confidence;
  const chi_lower = df / (1 + _normInv(1 - alpha / 2) * Math.sqrt(2 / df));
  const chi_upper = df / (1 - _normInv(1 - alpha / 2) * Math.sqrt(2 / df));
  const lower = (df * s2) / chi_upper;
  const upper = (df * s2) / chi_lower;
  return _text({
    type: "confidence_interval",
    parameter: "variance",
    confidence,
    n,
    variance: s2,
    lower,
    upper,
  });
};

const confidence_interval_difference = (a, b, confidence = 0.95) => {
  const xa = _numeric(a);
  const xb = _numeric(b);
  const na = xa.length;
  const nb = xb.length;
  if (na < 2 || nb < 2)
    return _err("confidence_interval_difference", "insufficient data");
  const ma = _mean(xa);
  const mb = _mean(xb);
  const va = _variance(xa, true);
  const vb = _variance(xb, true);
  const diff = ma - mb;
  const se = Math.sqrt(va / na + vb / nb);
  const alpha = 1 - confidence;
  const z = _normInv(1 - alpha / 2);
  const margin = z * se;
  const lower = diff - margin;
  const upper = diff + margin;
  return _text({
    type: "confidence_interval",
    parameter: "difference_of_means",
    confidence,
    difference: diff,
    lower,
    upper,
    margin,
    means: { group_a: ma, group_b: mb },
  });
};

// =========================
// ADDITIONAL CORRELATIONS
// =========================

const corr_kendall = (x, y) => {
  const ax = _numeric(x);
  const by = _numeric(y);
  const n = Math.min(ax.length, by.length);
  if (n < 2) return _err("corr_kendall", "insufficient data");
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = ax[j] - ax[i];
      const dy = by[j] - by[i];
      if (dx * dy > 0) concordant++;
      else if (dx * dy < 0) discordant++;
    }
  }
  const tau = (concordant - discordant) / (0.5 * n * (n - 1));
  return _ok("statistic", {
    name: "kendall_tau",
    value: tau,
    concordant,
    discordant,
    n,
  });
};

const corr_partial = (x, y, z) => {
  const rxy = _corrPearson(x, y);
  const rxz = _corrPearson(x, z);
  const ryz = _corrPearson(y, z);
  const rxy_z =
    (rxy - rxz * ryz) / Math.sqrt((1 - rxz * rxz) * (1 - ryz * ryz));
  return _ok("statistic", {
    name: "partial_correlation",
    value: rxy_z,
    controlling_for: "third_variable",
  });
};

const corr_matrix_all = (rows, method = "pearson") => {
  const pearson = _lowerJson(df_corr(rows, "pearson")).matrix;
  const spearman = _lowerJson(df_corr(rows, "spearman")).matrix;
  const cols = Object.keys(pearson);
  const kendall = {};
  cols.forEach((a) => {
    kendall[a] = {};
    cols.forEach((b) => {
      const xa = rows.map((r) => _toNum(r[a]));
      const xb = rows.map((r) => _toNum(r[b]));
      const tau = _lowerJson(corr_kendall(xa, xb)).value ?? NaN;
      kendall[a][b] = tau;
    });
  });
  return _text({ type: "correlation_analysis", pearson, spearman, kendall });
};

// =========================
// K-NEAREST NEIGHBORS
// =========================

const _euclidean = (a, b) =>
  Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));

const train_knn_classifier = (X, y, k = 5) => {
  if (X.length !== y.length)
    return _err("train_knn_classifier", "X and y length mismatch");
  const model = {
    type: "knn_classifier",
    k,
    X,
    y,
    n: y.length,
    p: X[0]?.length ?? 0,
  };
  return _text(model);
};

const predict_knn_classifier = (model_text, X_test) => {
  try {
    const m = _lowerJson(model_text);
    if (m.type !== "knn_classifier")
      return _err("predict_knn_classifier", "invalid model type");
    const { k, x: X_train, y: y_train } = m;
    const predictions = X_test.map((x) => {
      const distances = X_train.map((xt, i) => ({
        dist: _euclidean(x, xt),
        label: y_train[i],
      }));
      distances.sort((a, b) => a.dist - b.dist);
      const neighbors = distances.slice(0, k);
      const votes = {};
      neighbors.forEach(({ label }) => {
        votes[label] = (votes[label] || 0) + 1;
      });
      const pred = Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
      return Number(pred);
    });
    return _text({
      type: "prediction",
      name: "knn_classifier",
      k,
      predictions,
    });
  } catch {
    return _err("predict_knn_classifier", "invalid model text");
  }
};

const train_knn_regressor = (X, y, k = 5) => {
  if (X.length !== y.length)
    return _err("train_knn_regressor", "X and y length mismatch");
  const model = {
    type: "knn_regressor",
    k,
    X,
    y,
    n: y.length,
    p: X[0]?.length ?? 0,
  };
  return _text(model);
};

const predict_knn_regressor = (model_text, X_test) => {
  try {
    const m = _lowerJson(model_text);
    if (m.type !== "knn_regressor")
      return _err("predict_knn_regressor", "invalid model type");
    const { k, x: X_train, y: y_train } = m;
    const predictions = X_test.map((x) => {
      const distances = X_train.map((xt, i) => ({
        dist: _euclidean(x, xt),
        value: y_train[i],
      }));
      distances.sort((a, b) => a.dist - b.dist);
      const neighbors = distances.slice(0, k);
      return _mean(neighbors.map((n) => n.value));
    });
    return _text({ type: "prediction", name: "knn_regressor", k, predictions });
  } catch {
    return _err("predict_knn_regressor", "invalid model text");
  }
};

// =========================
// DECISION TREE (CART)
// =========================

const _gini = (y) => {
  const counts = {};
  y.forEach((v) => {
    counts[v] = (counts[v] || 0) + 1;
  });
  const total = y.length;
  return 1 - Object.values(counts).reduce((s, c) => s + (c / total) ** 2, 0);
};

const _mse_split = (y) => {
  const m = _mean(y);
  return _mean(y.map((v) => (v - m) ** 2));
};

const _best_split = (X, y, task = "classification") => {
  let best = { feature: -1, threshold: 0, score: Infinity };
  const n_features = X[0].length;
  for (let f = 0; f < n_features; f++) {
    const values = _uniq(X.map((row) => row[f])).sort((a, b) => a - b);
    for (let i = 0; i < values.length - 1; i++) {
      const thresh = (values[i] + values[i + 1]) / 2;
      const left_idx = [];
      const right_idx = [];
      X.forEach((row, idx) => {
        if (row[f] <= thresh) left_idx.push(idx);
        else right_idx.push(idx);
      });
      if (left_idx.length === 0 || right_idx.length === 0) continue;
      const left_y = left_idx.map((i) => y[i]);
      const right_y = right_idx.map((i) => y[i]);
      let score;
      if (task === "classification") {
        score =
          (left_y.length / y.length) * _gini(left_y) +
          (right_y.length / y.length) * _gini(right_y);
      } else {
        score =
          (left_y.length / y.length) * _mse_split(left_y) +
          (right_y.length / y.length) * _mse_split(right_y);
      }
      if (score < best.score) best = { feature: f, threshold: thresh, score };
    }
  }
  return best;
};

const _build_tree = (X, y, depth, max_depth, min_samples, task) => {
  if (depth >= max_depth || y.length < min_samples) {
    const pred =
      task === "classification"
        ? Object.entries(
            y.reduce((a, v) => {
              a[v] = (a[v] || 0) + 1;
              return a;
            }, {})
          ).sort((a, b) => b[1] - a[1])[0][0]
        : _mean(y);
    return { leaf: true, prediction: Number(pred), n: y.length };
  }
  const split = _best_split(X, y, task);
  if (split.feature === -1) {
    const pred =
      task === "classification"
        ? Object.entries(
            y.reduce((a, v) => {
              a[v] = (a[v] || 0) + 1;
              return a;
            }, {})
          ).sort((a, b) => b[1] - a[1])[0][0]
        : _mean(y);
    return { leaf: true, prediction: Number(pred), n: y.length };
  }
  const left_idx = [];
  const right_idx = [];
  X.forEach((row, i) => {
    if (row[split.feature] <= split.threshold) left_idx.push(i);
    else right_idx.push(i);
  });
  const left_X = left_idx.map((i) => X[i]);
  const left_y = left_idx.map((i) => y[i]);
  const right_X = right_idx.map((i) => X[i]);
  const right_y = right_idx.map((i) => y[i]);
  return {
    leaf: false,
    feature: split.feature,
    threshold: split.threshold,
    left: _build_tree(left_X, left_y, depth + 1, max_depth, min_samples, task),
    right: _build_tree(
      right_X,
      right_y,
      depth + 1,
      max_depth,
      min_samples,
      task
    ),
  };
};

const train_decision_tree_classifier = (X, y, opts = {}) => {
  const max_depth = opts.max_depth ?? 5;
  const min_samples = opts.min_samples_split ?? 2;
  const tree = _build_tree(X, y, 0, max_depth, min_samples, "classification");
  const model = {
    type: "decision_tree_classifier",
    tree,
    max_depth,
    min_samples,
    n: y.length,
    p: X[0]?.length ?? 0,
  };
  return _text(model);
};

const train_decision_tree_regressor = (X, y, opts = {}) => {
  const max_depth = opts.max_depth ?? 5;
  const min_samples = opts.min_samples_split ?? 2;
  const tree = _build_tree(X, y, 0, max_depth, min_samples, "regression");
  const model = {
    type: "decision_tree_regressor",
    tree,
    max_depth,
    min_samples,
    n: y.length,
    p: X[0]?.length ?? 0,
  };
  return _text(model);
};

const _predict_tree = (tree, x) => {
  if (tree.leaf) return tree.prediction;
  return x[tree.feature] <= tree.threshold
    ? _predict_tree(tree.left, x)
    : _predict_tree(tree.right, x);
};

const predict_decision_tree = (model_text, X_test) => {
  try {
    const m = _lowerJson(model_text);
    const predictions = X_test.map((x) => _predict_tree(m.tree, x));
    return _text({ type: "prediction", name: m.type, predictions });
  } catch {
    return _err("predict_decision_tree", "invalid model text");
  }
};

// =========================
// RANDOM FOREST
// =========================

const _bootstrap_sample = (X, y, seed) => {
  const n = X.length;
  let s = seed;
  const rand = () => (s = (s * 9301 + 49297) % 233280) / 233280;
  const indices = Array.from({ length: n }, () => Math.floor(rand() * n));
  const X_boot = indices.map((i) => X[i]);
  const y_boot = indices.map((i) => y[i]);
  return { X_boot, y_boot };
};

const train_random_forest_classifier = (X, y, opts = {}) => {
  const n_trees = opts.n_estimators ?? 10;
  const max_depth = opts.max_depth ?? 5;
  const min_samples = opts.min_samples_split ?? 2;
  const seed = opts.seed ?? 42;
  const trees = [];
  for (let i = 0; i < n_trees; i++) {
    const { X_boot, y_boot } = _bootstrap_sample(X, y, seed + i);
    const tree_model = _lowerJson(
      train_decision_tree_classifier(X_boot, y_boot, { max_depth, min_samples })
    );
    trees.push(tree_model.tree);
  }
  const model = {
    type: "random_forest_classifier",
    trees,
    n_trees,
    max_depth,
    min_samples,
    n: y.length,
    p: X[0]?.length ?? 0,
  };
  return _text(model);
};

const train_random_forest_regressor = (X, y, opts = {}) => {
  const n_trees = opts.n_estimators ?? 10;
  const max_depth = opts.max_depth ?? 5;
  const min_samples = opts.min_samples_split ?? 2;
  const seed = opts.seed ?? 42;
  const trees = [];
  for (let i = 0; i < n_trees; i++) {
    const { X_boot, y_boot } = _bootstrap_sample(X, y, seed + i);
    const tree_model = _lowerJson(
      train_decision_tree_regressor(X_boot, y_boot, { max_depth, min_samples })
    );
    trees.push(tree_model.tree);
  }
  const model = {
    type: "random_forest_regressor",
    trees,
    n_trees,
    max_depth,
    min_samples,
    n: y.length,
    p: X[0]?.length ?? 0,
  };
  return _text(model);
};

const predict_random_forest_classifier = (model_text, X_test) => {
  try {
    const m = _lowerJson(model_text);
    const predictions = X_test.map((x) => {
      const votes = m.trees.map((tree) => _predict_tree(tree, x));
      const counts = {};
      votes.forEach((v) => {
        counts[v] = (counts[v] || 0) + 1;
      });
      return Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]);
    });
    return _text({
      type: "prediction",
      name: "random_forest_classifier",
      n_trees: m.n_trees,
      predictions,
    });
  } catch {
    return _err("predict_random_forest_classifier", "invalid model text");
  }
};

const predict_random_forest_regressor = (model_text, X_test) => {
  try {
    const m = _lowerJson(model_text);
    const predictions = X_test.map((x) => {
      const preds = m.trees.map((tree) => _predict_tree(tree, x));
      return _mean(preds);
    });
    return _text({
      type: "prediction",
      name: "random_forest_regressor",
      n_trees: m.n_trees,
      predictions,
    });
  } catch {
    return _err("predict_random_forest_regressor", "invalid model text");
  }
};

// =========================
// NAIVE BAYES
// =========================

const train_naive_bayes = (X, y) => {
  const classes = _uniq(y);
  const n = y.length;
  const p = X[0]?.length ?? 0;
  const priors = {};
  const stats = {};
  classes.forEach((c) => {
    const indices = y.map((v, i) => (v === c ? i : -1)).filter((i) => i >= 0);
    priors[c] = indices.length / n;
    stats[c] = Array.from({ length: p }, (_, j) => {
      const col = indices.map((i) => X[i][j]);
      return { mean: _mean(col), std: _std(col, true) };
    });
  });
  const model = { type: "naive_bayes", classes, priors, stats, n, p };
  return _text(model);
};

const predict_naive_bayes = (model_text, X_test) => {
  try {
    const m = _lowerJson(model_text);
    const predictions = X_test.map((x) => {
      const scores = {};
      m.classes.forEach((c) => {
        let log_prob = Math.log(m.priors[c]);
        m.stats[c].forEach((s, j) => {
          const val = (x[j] - s.mean) / s.std;
          log_prob +=
            -0.5 * val * val - Math.log(s.std) - 0.5 * Math.log(2 * Math.PI);
        });
        scores[c] = log_prob;
      });
      return Number(Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0]);
    });
    return _text({ type: "prediction", name: "naive_bayes", predictions });
  } catch {
    return _err("predict_naive_bayes", "invalid model text");
  }
};

// =========================
// FEATURE SCALING
// =========================

const standard_scaler_fit = (X) => {
  const p = X[0]?.length ?? 0;
  const params = Array.from({ length: p }, (_, j) => {
    const col = X.map((row) => row[j]);
    return { mean: _mean(col), std: _std(col, true) };
  });
  return _text({ type: "standard_scaler", params, n: X.length, p });
};

const standard_scaler_transform = (scaler_text, X) => {
  try {
    const m = _lowerJson(scaler_text);
    const X_scaled = X.map((row) =>
      row.map((v, j) => (v - m.params[j].mean) / m.params[j].std)
    );
    return _text({
      type: "scaled_data",
      method: "standard",
      data: X_scaled,
      preview: X_scaled.slice(0, 5),
    });
  } catch {
    return _err("standard_scaler_transform", "invalid scaler text");
  }
};

const minmax_scaler_fit = (X) => {
  const p = X[0]?.length ?? 0;
  const params = Array.from({ length: p }, (_, j) => {
    const col = X.map((row) => row[j]);
    return { min: _min(col), max: _max(col) };
  });
  return _text({ type: "minmax_scaler", params, n: X.length, p });
};

const minmax_scaler_transform = (scaler_text, X) => {
  try {
    const m = _lowerJson(scaler_text);
    const X_scaled = X.map((row) =>
      row.map((v, j) => {
        const range = m.params[j].max - m.params[j].min;
        return range === 0 ? 0 : (v - m.params[j].min) / range;
      })
    );
    return _text({
      type: "scaled_data",
      method: "minmax",
      preview: X_scaled.slice(0, 5),
    });
  } catch {
    return _err("minmax_scaler_transform", "invalid scaler text");
  }
};

// =========================
// DIMENSIONALITY REDUCTION (PCA)
// =========================

const train_pca = (X, n_components = 2) => {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  if (n_components > p)
    return _err("train_pca", "n_components cannot exceed number of features");

  // center data
  const means = Array.from({ length: p }, (_, j) =>
    _mean(X.map((row) => row[j]))
  );
  const X_centered = X.map((row) => row.map((v, j) => v - means[j]));

  // covariance matrix
  const cov = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += X_centered[k][i] * X_centered[k][j];
      return sum / (n - 1);
    })
  );

  // simple power iteration for first n_components eigenvectors
  const components = [];
  for (let c = 0; c < n_components; c++) {
    let v = Array.from({ length: p }, () => Math.random());
    for (let iter = 0; iter < 100; iter++) {
      const v_new = Array.from({ length: p }, (_, i) =>
        cov[i].reduce((s, val, j) => s + val * v[j], 0)
      );
      const norm = Math.sqrt(v_new.reduce((s, val) => s + val * val, 0));
      v = v_new.map((val) => val / norm);
    }
    components.push(v);

    // deflate covariance matrix
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        cov[i][j] -=
          v[i] * v[j] * cov[i].reduce((s, val, k) => s + val * v[k], 0);
      }
    }
  }

  const model = { type: "pca", n_components, means, components, n, p };
  return _text(model);
};

const transform_pca = (model_text, X) => {
  try {
    const m = _lowerJson(model_text);
    const X_centered = X.map((row) => row.map((v, j) => v - m.means[j]));
    const X_transformed = X_centered.map((row) =>
      m.components.map((comp) => row.reduce((s, v, i) => s + v * comp[i], 0))
    );
    return _text({
      type: "pca_transform",
      n_components: m.n_components,
      preview: X_transformed.slice(0, 5),
    });
  } catch {
    return _err("transform_pca", "invalid model text");
  }
};

// =========================
// CLUSTERING (K-MEANS)
// =========================

const train_kmeans = (X, k = 3, opts = {}) => {
  const max_iter = opts.max_iterations ?? 100;
  const seed = opts.seed ?? 42;
  let s = seed;
  const rand = () => (s = (s * 9301 + 49297) % 233280) / 233280;

  const n = X.length;
  const p = X[0]?.length ?? 0;

  // initialize centroids randomly
  const indices = Array.from({ length: k }, () => Math.floor(rand() * n));
  let centroids = indices.map((i) => [...X[i]]);
  let labels = Array(n).fill(0);

  for (let iter = 0; iter < max_iter; iter++) {
    // assign points to nearest centroid
    const new_labels = X.map((x) => {
      const distances = centroids.map((c) => _euclidean(x, c));
      return distances.indexOf(Math.min(...distances));
    });

    // check convergence
    if (labels.every((l, i) => l === new_labels[i])) break;
    labels = new_labels;

    // update centroids
    centroids = Array.from({ length: k }, (_, c) => {
      const cluster_points = X.filter((_, i) => labels[i] === c);
      if (cluster_points.length === 0) return centroids[c];
      return Array.from({ length: p }, (_, j) =>
        _mean(cluster_points.map((pt) => pt[j]))
      );
    });
  }

  // calculate inertia
  const inertia = X.reduce(
    (s, x, i) => s + _euclidean(x, centroids[labels[i]]) ** 2,
    0
  );

  const model = { type: "kmeans", k, centroids, inertia, n, p };
  return _text(model);
};

const predict_kmeans = (model_text, X_test) => {
  try {
    const m = _lowerJson(model_text);
    const predictions = X_test.map((x) => {
      const distances = m.centroids.map((c) => _euclidean(x, c));
      return distances.indexOf(Math.min(...distances));
    });
    return _text({
      type: "prediction",
      name: "kmeans",
      k: m.k,
      cluster_labels: predictions,
    });
  } catch {
    return _err("predict_kmeans", "invalid model text");
  }
};

// =========================
// ENSEMBLE VOTING
// =========================

const ensemble_voting_classifier = (models_text, X_test, voting = "hard") => {
  try {
    const models = models_text.map((mt) => _lowerJson(mt));

    if (voting === "hard") {
      const all_preds = models.map((m) => {
        if (m.type === "logistic_regression") {
          const pred_result = _lowerJson(predict_logistic(_text(m), X_test));
          return pred_result.classes;
        } else if (m.type === "knn_classifier") {
          const pred_result = _lowerJson(
            predict_knn_classifier(_text(m), X_test)
          );
          return pred_result.predictions;
        } else if (m.type === "decision_tree_classifier") {
          const pred_result = _lowerJson(
            predict_decision_tree(_text(m), X_test)
          );
          return pred_result.predictions;
        } else if (m.type === "random_forest_classifier") {
          const pred_result = _lowerJson(
            predict_random_forest_classifier(_text(m), X_test)
          );
          return pred_result.predictions;
        } else if (m.type === "naive_bayes") {
          const pred_result = _lowerJson(predict_naive_bayes(_text(m), X_test));
          return pred_result.predictions;
        }
        return [];
      });

      const ensemble_preds = X_test.map((_, i) => {
        const votes = {};
        all_preds.forEach((preds) => {
          const v = preds[i];
          votes[v] = (votes[v] || 0) + 1;
        });
        return Number(Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]);
      });

      return _text({
        type: "ensemble_prediction",
        method: "voting_hard",
        n_models: models.length,
        predictions: ensemble_preds,
      });
    } else {
      return _err(
        "ensemble_voting_classifier",
        "soft voting not yet implemented"
      );
    }
  } catch {
    return _err("ensemble_voting_classifier", "invalid models");
  }
};

const ensemble_voting_regressor = (models_text, X_test) => {
  try {
    const models = models_text.map((mt) => _lowerJson(mt));

    const all_preds = models.map((m) => {
      if (m.type === "linear_regression") {
        const pred_result = _lowerJson(predict_linear(_text(m), X_test));
        return pred_result.predictions;
      } else if (m.type === "knn_regressor") {
        const pred_result = _lowerJson(predict_knn_regressor(_text(m), X_test));
        return pred_result.predictions;
      } else if (m.type === "decision_tree_regressor") {
        const pred_result = _lowerJson(predict_decision_tree(_text(m), X_test));
        return pred_result.predictions;
      } else if (m.type === "random_forest_regressor") {
        const pred_result = _lowerJson(
          predict_random_forest_regressor(_text(m), X_test)
        );
        return pred_result.predictions;
      }
      return [];
    });

    const ensemble_preds = X_test.map((_, i) => {
      const values = all_preds.map((preds) => preds[i]);
      return _mean(values);
    });

    return _text({
      type: "ensemble_prediction",
      method: "voting_average",
      n_models: models.length,
      predictions: ensemble_preds,
    });
  } catch {
    return _err("ensemble_voting_regressor", "invalid models");
  }
};

// =========================
// CROSS-VALIDATION
// =========================

const cross_validate = (X, y, model_type, opts = {}) => {
  const k_folds = opts.k_folds ?? 5;
  const normalize = opts.normalize ?? false;
  const shuffle = opts.shuffle ?? true;  // ← Embaralhar antes de dividir
  const seed = opts.seed ?? 42;
  const n = X.length;

  // ✅ Embaralhar índices se solicitado
  let indices = Array.from({ length: n }, (_, i) => i);
  if (shuffle) {
    let s = seed;
    const rand = () => (s = (s * 9301 + 49297) % 233280) / 233280;
    indices.sort(() => rand() - 0.5);
  }

  const fold_size = Math.floor(n / k_folds);
  const scores = [];

  for (let fold = 0; fold < k_folds; fold++) {
    const test_start = fold * fold_size;
    const test_end = fold === k_folds - 1 ? n : (fold + 1) * fold_size;

    // Usar índices embaralhados
    const train_idx = [...indices.slice(0, test_start), ...indices.slice(test_end)];
    const test_idx = indices.slice(test_start, test_end);

    let X_train = train_idx.map(i => X[i]);
    let y_train = train_idx.map(i => y[i]);
    let X_test = test_idx.map(i => X[i]);
    const y_test = test_idx.map(i => y[i]);

    // Normalização dentro do fold
    if (normalize) {
      const scaler = standard_scaler_fit(X_train);
      const train_scaled = standard_scaler_transform(scaler, X_train);
      const test_scaled = standard_scaler_transform(scaler, X_test);

      X_train = _lowerJson(train_scaled).data || train_scaled.data;
      X_test = _lowerJson(test_scaled).data || test_scaled.data;
    }

    let model_text;
    let predictions;

    // Treinar e avaliar baseado no tipo de modelo
    if (model_type === "linear_regression") {
      model_text = train_linear_regression(X_train, y_train);
      const pred_result = _lowerJson(predict_linear(model_text, X_test));
      predictions = pred_result.predictions;
      const metrics = _lowerJson(metrics_regression(y_test, predictions));
      scores.push(metrics.r2);

    } else if (model_type === "logistic_regression") {
      model_text = train_logistic_regression(X_train, y_train, opts);
      const pred_result = _lowerJson(predict_logistic(model_text, X_test));
      predictions = pred_result.classes;
      const metrics = _lowerJson(metrics_classification(y_test, predictions));
      scores.push(metrics.accuracy);

    } else if (model_type === "knn_classifier") {
      model_text = train_knn_classifier(X_train, y_train, opts.k ?? 5);
      const pred_result = _lowerJson(predict_knn_classifier(model_text, X_test));
      predictions = pred_result.predictions;
      const metrics = _lowerJson(metrics_classification(y_test, predictions));
      scores.push(metrics.accuracy);

    } else if (model_type === "decision_tree_classifier") {
      model_text = train_decision_tree_classifier(X_train, y_train, opts);
      const pred_result = _lowerJson(predict_decision_tree(model_text, X_test));
      predictions = pred_result.predictions;
      const metrics = _lowerJson(metrics_classification(y_test, predictions));
      scores.push(metrics.accuracy);

    } else if (model_type === "random_forest_classifier") {
      model_text = train_random_forest_classifier(X_train, y_train, opts);
      const pred_result = _lowerJson(predict_random_forest_classifier(model_text, X_test));
      predictions = pred_result.predictions;
      const metrics = _lowerJson(metrics_classification(y_test, predictions));
      scores.push(metrics.accuracy);

    } else if (model_type === "knn_regressor") {
      model_text = train_knn_regressor(X_train, y_train, opts.k ?? 5);
      const pred_result = _lowerJson(predict_knn_regressor(model_text, X_test));
      predictions = pred_result.predictions;
      const metrics = _lowerJson(metrics_regression(y_test, predictions));
      scores.push(metrics.r2);

    } else if (model_type === "decision_tree_regressor") {
      model_text = train_decision_tree_regressor(X_train, y_train, opts);
      const pred_result = _lowerJson(predict_decision_tree(model_text, X_test));
      predictions = pred_result.predictions;
      const metrics = _lowerJson(metrics_regression(y_test, predictions));
      scores.push(metrics.r2);

    } else if (model_type === "random_forest_regressor") {
      model_text = train_random_forest_regressor(X_train, y_train, opts);
      const pred_result = _lowerJson(predict_random_forest_regressor(model_text, X_test));
      predictions = pred_result.predictions;
      const metrics = _lowerJson(metrics_regression(y_test, predictions));
      scores.push(metrics.r2);
    }
  }

  return _text({
    type: "cross_validation",
    model_type,
    k_folds,
    scores,
    mean_score: _mean(scores),
    std_score: _std(scores, true),
    min_score: _min(scores),
    max_score: _max(scores),
    normalized: normalize,
    shuffled: shuffle,
  });
};

// =========================
// FEATURE IMPORTANCE (for tree-based models)
// =========================

const _tree_feature_importance = (tree, n_features) => {
  const importance = Array(n_features).fill(0);

  const traverse = (node, n_samples) => {
    if (node.leaf) return;
    importance[node.feature] += n_samples;
    traverse(node.left, n_samples);
    traverse(node.right, n_samples);
  };

  traverse(tree, 1);
  const total = importance.reduce((a, b) => a + b, 0);
  return importance.map((v) => (total > 0 ? v / total : 0));
};

const feature_importance_tree = (model_text) => {
  try {
    const m = _lowerJson(model_text);
    if (
      m.type === "decision_tree_classifier" ||
      m.type === "decision_tree_regressor"
    ) {
      const importance = _tree_feature_importance(m.tree, m.p);
      return _text({ type: "feature_importance", model: m.type, importance });
    } else if (
      m.type === "random_forest_classifier" ||
      m.type === "random_forest_regressor"
    ) {
      const all_importance = m.trees.map((tree) =>
        _tree_feature_importance(tree, m.p)
      );
      const avg_importance = Array.from({ length: m.p }, (_, i) =>
        _mean(all_importance.map((imp) => imp[i]))
      );
      return _text({
        type: "feature_importance",
        model: m.type,
        n_trees: m.n_trees,
        importance: avg_importance,
      });
    } else {
      return _err("feature_importance_tree", "model must be tree-based");
    }
  } catch {
    return _err("feature_importance_tree", "invalid model text");
  }
};

// =========================
// OUTLIER DETECTION
// =========================

const outliers_iqr = (arr) => {
  const x = _numeric(arr);
  const q1 = _quantile(x, 0.25);
  const q3 = _quantile(x, 0.75);
  const iqr = q3 - q1;
  const lower = q1 - 1.5 * iqr;
  const upper = q3 + 1.5 * iqr;
  const outliers = x.filter((v) => v < lower || v > upper);
  const indices = arr
    .map((v, i) => (_isNumber(v) && (v < lower || v > upper) ? i : -1))
    .filter((i) => i >= 0);
  return _text({
    type: "outlier_detection",
    method: "iqr",
    lower_bound: lower,
    upper_bound: upper,
    n_outliers: outliers.length,
    outlier_indices: indices,
    outlier_values: outliers,
  });
};

const outliers_zscore = (arr, threshold = 3) => {
  const x = _numeric(arr);
  const m = _mean(x);
  const s = _std(x, true);
  const zscores = x.map((v) => Math.abs((v - m) / s));
  const outliers = x.filter((_, i) => zscores[i] > threshold);
  const indices = arr
    .map((v, i) => (_isNumber(v) && Math.abs((v - m) / s) > threshold ? i : -1))
    .filter((i) => i >= 0);
  return _text({
    type: "outlier_detection",
    method: "zscore",
    threshold,
    n_outliers: outliers.length,
    outlier_indices: indices,
    outlier_values: outliers,
  });
};

// =========================
// TIME SERIES BASICS
// =========================

const moving_average = (arr, window = 3) => {
  const x = _numeric(arr);
  const ma = [];
  for (let i = 0; i < x.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = x.slice(start, i + 1);
    ma.push(_mean(slice));
  }
  return _text({
    type: "time_series",
    method: "moving_average",
    window,
    values: ma,
  });
};

const exponential_smoothing = (arr, alpha = 0.3) => {
  const x = _numeric(arr);
  if (x.length === 0) return _err("exponential_smoothing", "empty data");
  const smoothed = [x[0]];
  for (let i = 1; i < x.length; i++) {
    smoothed.push(alpha * x[i] + (1 - alpha) * smoothed[i - 1]);
  }
  return _text({
    type: "time_series",
    method: "exponential_smoothing",
    alpha,
    values: smoothed,
  });
};

const autocorrelation = (arr, lag = 1) => {
  const x = _numeric(arr);
  const n = x.length;
  if (n < lag + 1) return _err("autocorrelation", "insufficient data for lag");
  const m = _mean(x);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    den += (x[i] - m) ** 2;
    if (i >= lag) num += (x[i] - m) * (x[i - lag] - m);
  }
  const acf = num / den;
  return _ok("statistic", { name: "autocorrelation", lag, value: acf });
};

// =========================
// Loaders
// =========================
const df_from_csv = (content, opts = {}) => {
  const config = {
    delimiter: ",",
    header: true,
    skipEmptyLines: true,
    ...opts,
  };
  const lines = content
    .split("\n")
    .filter((l) => (config.skipEmptyLines ? l.trim() !== "" : true));
  if (lines.length === 0) return _empty_df();

  const headers = config.header
    ? lines[0].split(config.delimiter).map((h) => h.trim().replace(/['"]/g, ""))
    : Array.from(
        { length: lines[0].split(config.delimiter).length },
        (_, i) => `col_${i}`
      );

  const startIndex = config.header ? 1 : 0;
  const data = [];

  for (let i = startIndex; i < lines.length; i++) {
    const values = lines[i].split(config.delimiter);
    if (values.length === headers.length) {
      const row = {};
      headers.forEach((header, idx) => {
        row[header] = _inferType(values[idx].trim().replace(/['"]/g, ""));
      });
      data.push(row);
    }
  }

  return _build_df(headers, data);
};

const df_from_json = (input) => {
  let jsonData;
  if (typeof input === "string") jsonData = JSON.parse(input);
  else jsonData = input;

  if (Array.isArray(jsonData)) return df_from_array(jsonData);
  if (jsonData.headers && jsonData.data)
    return df_from_structured_json(jsonData);
  if (typeof jsonData === "object") return df_from_object(jsonData);
  return _empty_df();
};

const df_from_array = (arr) => {
  if (!arr.length) return _empty_df();
  const headers = _uniq(arr.flatMap((obj) => Object.keys(obj)));
  const data = arr.map((o) => {
    const row = {};
    headers.forEach((h) => (row[h] = _inferType(o[h])));
    return row;
  });
  return _build_df(headers, data);
};

const df_from_structured_json = (jsonData) => {
  const headers = jsonData.headers;
  const data = jsonData.data.map((row) => {
    const obj = {};
    headers.forEach((h, i) => (obj[h] = _inferType(row[i])));
    return obj;
  });
  return _build_df(headers, data);
};

// MODIFICADO: Agora suporta objetos aninhados
const df_from_object = (obj, opts = {}) => {
  const config = {
    flatten: true, // Se true, achata objetos aninhados
    maxDepth: 10, // Profundidade máxima para achatamento
    ...opts,
  };

  if (config.flatten) {
    // Achata o objeto em uma única linha com múltiplas colunas
    const flattened = _flatten(obj, "", config.maxDepth);
    const columns = Object.keys(flattened);
    const data = [flattened];
    return _build_df(columns, data);
  } else {
    // Formato antigo: key-value pairs
    const headers = ["key", "value"];
    const data = Object.entries(obj).map(([key, val]) => ({
      key,
      value: _inferType(val),
    }));
    return _build_df(headers, data);
  }
};

// =========================
// Manipulação básica
// =========================
const df_get_column = (df, col) => {
  if (!df.columns.includes(col)) {
    throw new Error(
      `Column '${col}' not found. Available: ${df.columns.join(", ")}`
    );
  }
  return df.data.map((row) => row[col]);
};

// NOVA: Pegar valor único (útil para dataframes de objeto achatado)
const df_get_value = (df, col) => {
  const column = df_get_column(df, col);
  return column[0];
};

const df_get_columns = (df, cols) => {
  const result = {};
  cols.forEach((c) => (result[c] = df_get_column(df, c)));
  return result;
};

const df_filter = (df, predicate) => {
  const filtered = df.data.filter(predicate);
  return _build_df(df.columns, filtered);
};

const df_sort = (df, col, order = "asc") => {
  if (!df.columns.includes(col)) throw new Error(`Column '${col}' not found`);
  const sorted = [...df.data].sort((a, b) => {
    const av = a[col],
      bv = b[col];
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string" && typeof bv === "string")
      return order === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
    return order === "asc" ? av - bv : bv - av;
  });
  return _build_df(df.columns, sorted);
};

const df_select = (df, cols) => {
  cols.forEach((c) => {
    if (!df.columns.includes(c)) throw new Error(`Column '${c}' not found`);
  });
  const data = df.data.map((row) => {
    const newRow = {};
    cols.forEach((c) => (newRow[c] = row[c]));
    return newRow;
  });
  return _build_df(cols, data);
};

// MODIFICADO: Retorna string formatada com _text
const df_info = (df) => {
  const types = {};
  const nulls = {};
  const uniques = {};

  df.columns.forEach((c) => {
    const colVals = df.data.map((r) => r[c]);
    const nonNull = colVals.filter((v) => v != null);

    // Coletar os tipos únicos
    const typeSet = new Set(nonNull.map((v) => typeof v));

    // Converter Set para array e pegar o tipo apropriado
    if (typeSet.size === 0) {
      types[c] = "empty";
    } else if (typeSet.size === 1) {
      types[c] = Array.from(typeSet)[0]; // ou [...typeSet][0]
    } else {
      types[c] = "mixed";
    }

    nulls[c] = colVals.length - nonNull.length;
    uniques[c] = new Set(nonNull).size;
  });

  const info = {
    n_rows: df.n_rows,
    n_cols: df.n_cols,
    columns: df.columns,
    types,
    null_counts: nulls,
    unique_counts: uniques,
  };

  return _text(info);
};

const df_head = (df, n = 5) => _build_df(df.columns, df.data.slice(0, n));
const df_tail = (df, n = 5) => _build_df(df.columns, df.data.slice(-n));

// =========================
// Avançadas
// =========================
const df_concat = (...dfs) => {
  if (!dfs.length) return _empty_df();
  const allColumns = _uniq(dfs.flatMap((df) => df.columns));
  const allRows = dfs.flatMap((df) => {
    return df.data.map((row) => {
      const newRow = {};
      allColumns.forEach((c) => (newRow[c] = row[c] ?? null));
      return newRow;
    });
  });
  return _build_df(allColumns, allRows);
};

const df_merge = (df1, df2, { on, how = "inner" }) => {
  if (!Array.isArray(on)) on = [on];
  const colSet = _uniq([...df1.columns, ...df2.columns]);
  const merged = [];

  for (const row1 of df1.data) {
    const matches = df2.data.filter((row2) =>
      on.every((key) => row1[key] === row2[key])
    );

    if (matches.length > 0) {
      for (const row2 of matches) merged.push({ ...row1, ...row2 });
    } else if (how === "left" || how === "outer") {
      const newRow = { ...row1 };
      df2.columns.forEach((c) => {
        if (!on.includes(c)) newRow[c] = null;
      });
      merged.push(newRow);
    }
  }

  if (how === "right" || how === "outer") {
    for (const row2 of df2.data) {
      const match = df1.data.some((row1) =>
        on.every((key) => row1[key] === row2[key])
      );
      if (!match) {
        const newRow = { ...row2 };
        df1.columns.forEach((c) => {
          if (!on.includes(c)) newRow[c] = null;
        });
        merged.push(newRow);
      }
    }
  }

  return _build_df(colSet, merged);
};

const df_dropna = (df, subset = null) => {
  const cols = subset || df.columns;
  const filtered = df.data.filter((row) =>
    cols.every((c) => row[c] !== null && row[c] !== undefined)
  );
  return _build_df(df.columns, filtered);
};

const df_fillna = (df, value, subset = null) => {
  const cols = subset || df.columns;
  const filled = df.data.map((row) => {
    const newRow = { ...row };
    cols.forEach((c) => {
      if (newRow[c] === null || newRow[c] === undefined) newRow[c] = value;
    });
    return newRow;
  });
  return _build_df(df.columns, filled);
};

const df_groupby = (df, keys) => {
  if (!Array.isArray(keys)) keys = [keys];
  const groups = new Map();

  df.data.forEach((row) => {
    const key = keys.map((k) => row[k]).join("|");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });

  return { keys, groups };
};

const df_aggregate = (grouped, aggMap) => {
  const result = [];

  grouped.groups.forEach((rows, key) => {
    const keyParts = key.split("|");
    const rowOut = {};
    grouped.keys.forEach((k, idx) => (rowOut[k] = keyParts[idx]));

    Object.entries(aggMap).forEach(([col, fn]) => {
      const vals = rows.map((r) => r[col]).filter((v) => v != null);
      rowOut[col] = fn(vals);
    });

    result.push(rowOut);
  });

  const allCols = _uniq([...grouped.keys, ...Object.keys(aggMap)]);
  return _build_df(allCols, result);
};

// =========================
// UTILITÁRIAS
// =========================

// 📉 aplicar função em uma coluna
const df_apply = (df, col, fn) => {
  if (!df.columns.includes(col)) throw new Error(`Column '${col}' not found`);
  const newData = df.data.map((row) => ({ ...row, [col]: fn(row[col], row) }));
  return _build_df(df.columns, newData);
};

// 🧾 exportar dataframe para CSV
const df_to_csv = (df, delimiter = ",") => {
  const header = df.columns.join(delimiter);
  const rows = df.data.map((row) =>
    df.columns
      .map((c) => {
        const val = row[c];
        // Serializa arrays e objetos como JSON
        if (Array.isArray(val) || (typeof val === "object" && val !== null)) {
          return JSON.stringify(val);
        }
        return val ?? "";
      })
      .join(delimiter)
  );
  return [header, ...rows].join("\n");
};

// 🧭 amostragem aleatória
const df_sample = (df, n = 5, seed = null) => {
  const data = [...df.data];
  if (seed !== null) {
    // seed opcional
    let m = seed;
    data.sort(() => (Math.sin(m++) > 0 ? 1 : -1));
  } else {
    data.sort(() => Math.random() - 0.5);
  }
  return _build_df(df.columns, data.slice(0, n));
};

// 🧪 valores únicos de uma coluna
const df_unique = (df, col) => {
  if (!df.columns.includes(col)) throw new Error(`Column '${col}' not found`);
  return _uniq(df.data.map((row) => row[col]));
};

// 🧼 renomear coluna(s)
const df_rename = (df, renameMap) => {
  const newColumns = df.columns.map((c) => renameMap[c] || c);
  const newData = df.data.map((row) => {
    const newRow = {};
    df.columns.forEach((c) => {
      const newName = renameMap[c] || c;
      newRow[newName] = row[c];
    });
    return newRow;
  });
  return _build_df(newColumns, newData);
};

// 🧱 adicionar coluna derivada
const df_add_column = (df, colName, fn) => {
  if (df.columns.includes(colName))
    throw new Error(`Column '${colName}' already exists`);
  const newData = df.data.map((row) => ({ ...row, [colName]: fn(row) }));
  return _build_df([...df.columns, colName], newData);
};

// 🧹 remover colunas
const df_drop = (df, cols) => {
  if (!Array.isArray(cols)) cols = [cols];
  const newColumns = df.columns.filter((c) => !cols.includes(c));
  const newData = df.data.map((row) => {
    const newRow = {};
    newColumns.forEach((c) => (newRow[c] = row[c]));
    return newRow;
  });
  return _build_df(newColumns, newData);
};

// =========================
// NOVAS: Funções para análise de dados aninhados
// =========================

// 📊 Expandir array de objetos em múltiplas linhas (explode)
const df_explode = (df, col) => {
  if (!df.columns.includes(col)) throw new Error(`Column '${col}' not found`);

  const newData = [];
  df.data.forEach((row) => {
    const value = row[col];
    if (Array.isArray(value) && value.length > 0) {
      value.forEach((item) => {
        newData.push({ ...row, [col]: item });
      });
    } else {
      newData.push(row);
    }
  });

  return _build_df(df.columns, newData);
};

// 🔍 Buscar colunas por padrão - RETORNA STRING
const df_find_columns = (df, pattern) => {
  const regex = new RegExp(pattern, "i");
  const matched = df.columns.filter((col) => regex.test(col));

  return _text({
    pattern,
    matches_found: matched.length,
    columns: matched,
  });
};

// 📈 Estatísticas descritivas de colunas numéricas - RETORNA STRING
const df_describe = (df, cols = null) => {
  const targetCols =
    cols ||
    df.columns.filter((col) => {
      const sample = df.data[0]?.[col];
      return typeof sample === "number";
    });

  const stats = {};

  targetCols.forEach((col) => {
    const values = df.data
      .map((row) => row[col])
      .filter((v) => typeof v === "number" && !isNaN(v));

    if (values.length === 0) {
      stats[col] = { error: "no numeric values" };
      return;
    }

    const sorted = [...values].sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / values.length;
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const median =
      sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)];

    stats[col] = { count: values.length, mean, median, min, max, sum };
  });

  return _text({
    description: "Descriptive Statistics",
    analyzed_columns: targetCols.length,
    statistics: stats,
  });
};

// =========================
// ADDITIONAL EXPORTS
// =========================

export {
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
};