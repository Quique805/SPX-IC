// ---- Regime config (edit defaults here, or tweak in the on-page table) ----
// Each row: VIX <= maxVix selects this regime.
// upperMult / lowerMult / k determine band width.
// trade=false → the day is graphed but marked as "no-trade" (Iron Condor not opened).
const DEFAULT_REGIMES = [
  { maxVix: 16.0, upperMult: 1.4, lowerMult: 0.9, k: 1.2, trade: true  },
  { maxVix: 19.5, upperMult: 1.2, lowerMult: 0.9, k: 0.9, trade: true  },
  { maxVix: 20.5, upperMult: 1.2, lowerMult: 0.9, k: 0.9, trade: false }, // gap regime
  { maxVix: 24.0, upperMult: 1.2, lowerMult: 1.2, k: 1.3, trade: true  },
  { maxVix: 999,  upperMult: 1.2, lowerMult: 1.2, k: 1.3, trade: false }, // VIX > 24
];

// Minimum sample size required inside the lookback window before we trust σ.
const MIN_SAMPLES = 20;

// Trend window & threshold (mean of last N closes vs previous N).
const TREND_WINDOW = 20;
const TREND_THRESHOLD = 0.5; // %

// ---- Dynamic compression factor C_t (variable selection model) ----------
// Each variable has a standardized form (centered ~0, scale ~1) and a weight.
// Z = sum(weight × standardized_value) over ACTIVE variables only.
// C_t = sigmoid(λ × Z), clamped to [Cmin, Cmax].
// Bands_t = Close_prev × (1 ± C_t × k_regime × mult × σ)
//
// Sign convention: a positive standardized value pushes toward LESS compression
// (wider bands). A user can flip this by entering a negative weight.
const COMPRESSION_VARS = [
  { id: 'iv_hv',   label: 'IV/HV',         group: 'A', defaultWeight: 1.0, defaultActive: true,
    desc: 'log(IV/HV) — positivo cuando IV > HV (mercado descuenta más vol)' },
  { id: 'iv_rank', label: 'IV Rank',       group: 'B', defaultWeight: 0.5, defaultActive: false,
    desc: '(IVR-50)/50 — positivo cuando IV está sobre la media del histórico' },
  { id: 'iv_pctl', label: 'IV Percentile', group: 'B', defaultWeight: 0.5, defaultActive: false,
    desc: '(IVP-50)/50 — positivo cuando IV está sobre la mediana histórica' },
  { id: 'vix',     label: 'VIX',           group: 'B', defaultWeight: 0.7, defaultActive: true,
    desc: '(Vref-VIX)/Vref — positivo cuando VIX < Vref' },
  { id: 'iv_chg',  label: 'IV Change',     group: 'C', defaultWeight: 0.3, defaultActive: false,
    desc: '(IV-IV_prev)/IV_prev — positivo cuando IV está subiendo' },
  { id: 'pcv',     label: 'P/C Volume',    group: 'D', defaultWeight: 0.4, defaultActive: false,
    desc: 'log(PCV) — positivo cuando puts dominan (sentimiento miedoso)' },
];
const COMPRESSION_GROUP_LABEL = {
  A: 'Mispricing',
  B: 'Nivel/régimen vol (REDUNDANTES — elige idealmente uno)',
  C: 'Momentum vol',
  D: 'Sentimiento',
};

// Mutable runtime state — initialized from defaults, edited via UI.
const compressionVars = COMPRESSION_VARS.map(v => ({
  ...v, active: v.defaultActive, weight: v.defaultWeight,
}));
const compressionParams = { lambda: 1.5, Cmin: 0.5, Cmax: 0.95, Vref: 18, shiftFactor: 0.25 };

// Standardize a variable value to a centered ~[-1, 1] scale.
// Returns null if data is missing for that variable.
function standardizeVar(varId, prevRow, params) {
  switch (varId) {
    case 'iv_hv':
      if (!isFinite(prevRow.iv) || !isFinite(prevRow.hv) || prevRow.iv <= 0 || prevRow.hv <= 0) return null;
      return Math.log(prevRow.iv / prevRow.hv);
    case 'iv_rank':
      return isFinite(prevRow.ivRank) ? (prevRow.ivRank - 50) / 50 : null;
    case 'iv_pctl':
      return isFinite(prevRow.ivPctl) ? (prevRow.ivPctl - 50) / 50 : null;
    case 'vix':
      return isFinite(prevRow.vix) ? (params.Vref - prevRow.vix) / params.Vref : null;
    case 'iv_chg':
      return isFinite(prevRow.ivChg) ? prevRow.ivChg : null;
    case 'pcv':
      if (!isFinite(prevRow.pcv) || prevRow.pcv <= 0) return null;
      return Math.log(prevRow.pcv);
    default:
      return null;
  }
}

// Returns { Ccall, Cput, base, trend } so the caller can apply asymmetric
// strikes per side. Cbase is the value of C_t straight from the sigmoid;
// Ccall and Cput are after the trend shift Δ = shiftFactor × (Cmax − Cmin).
function compressionFactor(prevRow) {
  if (!prevRow) return { Ccall: 1.0, Cput: 1.0, base: 1.0, trend: null };
  let Z = 0;
  for (let i = 0; i < compressionVars.length; i++) {
    const v = compressionVars[i];
    if (!v.active) continue;
    const s = standardizeVar(v.id, prevRow, compressionParams);
    if (s === null || !isFinite(s)) continue;
    Z += v.weight * s;
  }
  const Craw = 1 / (1 + Math.exp(-compressionParams.lambda * Z));
  const Cmin = compressionParams.Cmin, Cmax = compressionParams.Cmax;
  const base = Math.max(Cmin, Math.min(Cmax, Craw));

  const trend = prevRow.trend || null;
  const delta = compressionParams.shiftFactor * (Cmax - Cmin);
  let Ccall = base, Cput = base;
  if (trend === 'up')   { Ccall = base + delta; Cput = base - delta; }
  else if (trend === 'down') { Ccall = base - delta; Cput = base + delta; }
  // Final clamp (in case shift took us outside)
  Ccall = Math.max(Cmin, Math.min(Cmax, Ccall));
  Cput  = Math.max(Cmin, Math.min(Cmax, Cput));
  return { Ccall, Cput, base, trend };
}

// For UI: produce a breakdown including base, Ccall, Cput, trend.
function compressionBreakdown(prevRow) {
  if (!prevRow) return null;
  const terms = [];
  let Z = 0;
  for (const v of compressionVars) {
    if (!v.active) continue;
    const s = standardizeVar(v.id, prevRow, compressionParams);
    if (s === null || !isFinite(s)) {
      terms.push({ label: v.label, std: null, weighted: 0, missing: true });
      continue;
    }
    const w = v.weight * s;
    Z += w;
    terms.push({ label: v.label, std: s, weight: v.weight, weighted: w });
  }
  const C_raw = 1 / (1 + Math.exp(-compressionParams.lambda * Z));
  const Cmin = compressionParams.Cmin, Cmax = compressionParams.Cmax;
  const base = Math.max(Cmin, Math.min(Cmax, C_raw));
  const trend = prevRow.trend || null;
  const delta = compressionParams.shiftFactor * (Cmax - Cmin);
  let Ccall = base, Cput = base;
  if (trend === 'up')   { Ccall = base + delta; Cput = base - delta; }
  else if (trend === 'down') { Ccall = base - delta; Cput = base + delta; }
  Ccall = Math.max(Cmin, Math.min(Cmax, Ccall));
  Cput  = Math.max(Cmin, Math.min(Cmax, Cput));
  return { Z, C_raw, base, Ccall, Cput, trend, delta, terms };
}

// ---- CSV loader -----------------------------------------------------------
function parseCSV(text) {
  // Strip UTF-8 BOM if present (Excel often adds it)
  if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);

  const rawLines = text.split(/\r?\n/).filter(l => l.trim() !== '');
  if (rawLines.length === 0) return [];

  // Auto-detect delimiter: prefer the one that gives more columns in line 1
  const tryDelims = [',', ';', '\t'];
  const delim = tryDelims.reduce((best, d) =>
    rawLines[0].split(d).length > rawLines[0].split(best).length ? d : best
  );

  const headers = rawLines[0].split(delim).map(h => h.trim());

  // Header aliases — map common variants to internal IDs
  const findHeader = (...aliases) => {
    for (const a of aliases) {
      const i = headers.findIndex(h => h.toLowerCase().replace(/[\s\/\-]/g, '') === a.toLowerCase().replace(/[\s\/\-]/g, ''));
      if (i >= 0) return i;
    }
    return -1;
  };
  const idx = {
    Date:  findHeader('Date', 'Fecha'),
    Close: findHeader('Close', 'Cierre'),
    Open:  findHeader('Open', 'Apertura', 'Op'),
    High:  findHeader('High', 'Maximo', 'Máximo'),
    Low:   findHeader('Low', 'Minimo', 'Mínimo'),
    VIX:   findHeader('VIX'),
    IV:    findHeader('IV', 'Implied Volatility'),
    HV:    findHeader('HV', 'Historical Volatility'),
    IVR:   findHeader('IVR', 'IV Rank'),
    IVP:   findHeader('IVP', 'IV Pctl', 'IV Percentile'),
    IVCHG: findHeader('IVCHG', '1D IV Chg', 'IV Change', 'IV Chg', '1D Chg'),
    PCV:   findHeader('PCV', 'P/C Vol', 'Put/Call', 'PC Vol', 'P C Vol'),
  };

  const missing = ['Date','Close','High','Low','VIX'].filter(h => idx[h] < 0);
  if (missing.length) {
    console.error('CSV: faltan columnas obligatorias:', missing, '· headers detectados:', headers);
    alert('CSV inválido: faltan columnas obligatorias ' + missing.join(', ') +
          '\n\nDelimitador detectado: "' + (delim === '\t' ? 'TAB' : delim) + '"' +
          '\nHeaders detectados: ' + headers.join(' | '));
    return [];
  }

  // Date parser: accepts YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY (Spanish/European)
  const parseDate = (s) => {
    if (!s) return null;
    s = s.trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
    const m = s.match(/^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})$/);
    if (m) {
      const dd = m[1].padStart(2, '0');
      const mm = m[2].padStart(2, '0');
      return `${m[3]}-${mm}-${dd}`;
    }
    return null;
  };

  // Number parser: strips %, spaces, and accepts both . and , as decimal
  const parseNum = (s) => {
    if (s === undefined || s === null) return NaN;
    s = String(s).trim().replace(/[%\s]/g, '');
    if (s === '') return NaN;
    // If has both comma and dot, assume comma is thousands → strip commas
    if (s.includes(',') && s.includes('.')) s = s.replace(/,/g, '');
    // If only commas (no dots), treat comma as decimal separator
    else if (s.includes(',') && !s.includes('.')) s = s.replace(/,/g, '.');
    return parseFloat(s);
  };

  const get = (c, i) => i >= 0 ? parseNum(c[i]) : NaN;
  const rows = [];
  let badDates = 0;

  for (let i = 1; i < rawLines.length; i++) {
    const c = rawLines[i].split(delim);
    const date = parseDate(c[idx.Date]);
    if (!date) { badDates++; continue; }
    rows.push({
      date,
      close:  parseNum(c[idx.Close]),
      open:   get(c, idx.Open),
      high:   parseNum(c[idx.High]),
      low:    parseNum(c[idx.Low]),
      vix:    parseNum(c[idx.VIX]),
      iv:     get(c, idx.IV),
      hv:     get(c, idx.HV),
      ivRank: get(c, idx.IVR),
      ivPctl: get(c, idx.IVP),
      ivChg:  get(c, idx.IVCHG),
      pcv:    get(c, idx.PCV),
    });
  }

  if (badDates > 0) {
    console.warn(`CSV: ${badDates} filas con fecha no parseable descartadas.`);
  }
  console.log(`CSV: ${rows.length} filas cargadas. Delimitador: "${delim === '\t' ? 'TAB' : delim}". Headers:`, headers, 'Mapping:', idx);

  rows.sort((a, b) => new Date(a.date) - new Date(b.date));
  return rows;
}

// Fill in derived metrics ONLY when the user didn't provide them in the CSV/form.
// User-provided values (from Barchart) win because they're computed against the
// full underlying option history (~252 days), not our small accumulated window.
function enrichRows(rows, ivWindow = 252, hvWindow = 30) {
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];

    // Auto-compute HV from realized log-returns of close prices when missing.
    // Standard formula: stdev(log returns) × √252 × 100, annualized %.
    if (!isFinite(row.hv) && isFinite(row.close)) {
      const returns = [];
      const start = Math.max(1, i - hvWindow + 1);
      for (let j = start; j <= i; j++) {
        const c0 = rows[j - 1] ? rows[j - 1].close : NaN;
        const c1 = rows[j] ? rows[j].close : NaN;
        if (isFinite(c0) && isFinite(c1) && c0 > 0 && c1 > 0) {
          returns.push(Math.log(c1 / c0));
        }
      }
      if (returns.length >= 10) {
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / (returns.length - 1);
        row.hv = Math.sqrt(variance) * Math.sqrt(252) * 100;
      }
    }

    // Compute IV Rank / Percentile from local history only if missing
    if ((!isFinite(row.ivRank) || !isFinite(row.ivPctl)) && isFinite(row.iv)) {
      const start = Math.max(0, i - ivWindow);
      const ivHist = [];
      for (let j = start; j < i; j++) {
        if (isFinite(rows[j].iv)) ivHist.push(rows[j].iv);
      }
      if (ivHist.length >= 10) {
        if (!isFinite(row.ivRank)) {
          let mn = Infinity, mx = -Infinity;
          for (let k = 0; k < ivHist.length; k++) {
            if (ivHist[k] < mn) mn = ivHist[k];
            if (ivHist[k] > mx) mx = ivHist[k];
          }
          row.ivRank = mx > mn ? (row.iv - mn) / (mx - mn) * 100 : 50;
        }
        if (!isFinite(row.ivPctl)) {
          let below = 0;
          for (let k = 0; k < ivHist.length; k++) if (ivHist[k] < row.iv) below++;
          row.ivPctl = below / ivHist.length * 100;
        }
      }
    }

    // Compute IV Change only if missing
    if (!isFinite(row.ivChg) && isFinite(row.iv)) {
      for (let j = i - 1; j >= 0; j--) {
        if (isFinite(rows[j].iv) && rows[j].iv > 0) {
          row.ivChg = (row.iv - rows[j].iv) / rows[j].iv;
          break;
        }
      }
    }

    // Pre-compute trend AT this row (uses only data ≤ i, no look-ahead).
    row.trend = null;
    if (i >= TREND_WINDOW * 2 - 1) {
      let rsum = 0, rn = 0, psum = 0, pn = 0;
      for (let j = i - TREND_WINDOW + 1; j <= i; j++) {
        if (isFinite(rows[j].close)) { rsum += rows[j].close; rn++; }
      }
      for (let j = i - TREND_WINDOW * 2 + 1; j <= i - TREND_WINDOW; j++) {
        if (isFinite(rows[j].close)) { psum += rows[j].close; pn++; }
      }
      if (rn > 0 && pn > 0) {
        const rm = rsum / rn, pm = psum / pn;
        if (pm > 0) {
          const diffPct = (rm - pm) / pm * 100;
          if (diffPct > TREND_THRESHOLD)       row.trend = 'up';
          else if (diffPct < -TREND_THRESHOLD) row.trend = 'down';
          else                                 row.trend = 'flat';
        }
      }
    }
  }
}

// ---- Regime lookup --------------------------------------------------------
// Plain for loop — much faster than for-of in hot paths (called millions of
// times during optimization).
function regimeFor(vix, regimes) {
  const n = regimes.length;
  for (let i = 0; i < n; i++) {
    if (vix <= regimes[i].maxVix) return regimes[i];
  }
  return regimes[n - 1];
}

// ---- Band math ------------------------------------------------------------
// σ from a rolling window of intraday range moves vs previous close:
//   σ_high = std(High / Close_prev − 1)
//   σ_low  = std(1 − Low  / Close_prev)
// Bands for day t use Close[t-1] as base and the regime picked by VIX[t-1].
// Sample standard deviation (Bessel-corrected, ddof=1) — matches pandas/numpy default.
function std(arr) {
  const n = arr.length;
  if (n < 2) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / n;
  const v = arr.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1);
  return Math.sqrt(v);
}

function computeBands(rows, regimes, lookback) {
  // Pre-compute daily range moves vs previous close
  const upMoves = new Array(rows.length).fill(NaN);
  const dnMoves = new Array(rows.length).fill(NaN);
  for (let i = 1; i < rows.length; i++) {
    const prevClose = rows[i - 1].close;
    upMoves[i] = rows[i].high / prevClose - 1;
    dnMoves[i] = 1 - rows[i].low / prevClose;
  }

  const bandAt = (t) => {
    const winStart = Math.max(1, t - lookback);
    const upWin = upMoves.slice(winStart, t).filter(Number.isFinite);
    const dnWin = dnMoves.slice(winStart, t).filter(Number.isFinite);
    if (upWin.length < MIN_SAMPLES || dnWin.length < MIN_SAMPLES) return null;

    const sigUp = std(upWin);
    const sigDn = std(dnWin);
    const prev = rows[t - 1];
    const today = rows[t]; // may be undefined for the next-day band
    const reg = regimeFor(prev.vix, regimes);
    const cf = compressionFactor(prev);
    const trade = isDayAccepted(today, reg);
    return {
      upper: prev.close * (1 + cf.Ccall * reg.k * reg.upperMult * sigUp),
      lower: prev.close * (1 - cf.Cput  * reg.k * reg.lowerMult * sigDn),
      sigUp, sigDn,
      Cbase: cf.base, Ccall: cf.Ccall, Cput: cf.Cput, trend: cf.trend,
      regime: reg, trade,
    };
  };

  const bands = new Array(rows.length).fill(null);
  for (let t = 1; t < rows.length; t++) bands[t] = bandAt(t);
  const nextBand = bandAt(rows.length); // one step beyond the last row
  return { bands, nextBand };
}

// ---- Backtest stats -------------------------------------------------------
// A "loss" = close outside the band on a trading day.
// No-trade days are excluded from the denominator.
function computeStats(rows, bands) {
  const overall = {
    total: 0, wins: 0, lossUp: 0, lossDn: 0, noTrade: 0,
    cleanWins: 0, touched: 0, callTouches: 0, putTouches: 0,
    firstDate: null, lastDate: null,
  };
  const byRegime = new Map();

  for (let i = 1; i < rows.length; i++) {
    const b = bands[i];
    const r = rows[i];
    if (!b || !isFinite(r.close)) continue;
    if (overall.firstDate === null) overall.firstDate = r.date;
    overall.lastDate = r.date;

    if (!b.trade) { overall.noTrade++; continue; }

    overall.total++;
    if (!byRegime.has(b.regime)) {
      byRegime.set(b.regime, { total: 0, wins: 0, lossUp: 0, lossDn: 0, cleanWins: 0, touched: 0 });
    }
    const s = byRegime.get(b.regime);
    s.total++;

    // Strikes the model would actually sell (rounded outside)
    const callStrike = Math.ceil(b.upper / 5) * 5;
    const putStrike  = Math.floor(b.lower / 5) * 5;
    const callTouched = isFinite(r.high) && r.high >= callStrike;
    const putTouched  = isFinite(r.low)  && r.low  <= putStrike;
    if (callTouched) overall.callTouches++;
    if (putTouched)  overall.putTouches++;

    if (r.close >= b.upper)      { overall.lossUp++; s.lossUp++; }
    else if (r.close <= b.lower) { overall.lossDn++; s.lossDn++; }
    else {
      overall.wins++; s.wins++;
      if (callTouched || putTouched) { overall.touched++;   s.touched++;   }
      else                           { overall.cleanWins++; s.cleanWins++; }
    }
  }
  return { overall, byRegime };
}

// ---- P&L engine (modelo conservador +150 / -600) -------------------------
// Modelo simplificado: cada día ganador suma WIN_PNL, cada día perdedor resta LOSS_PNL.
// No-trade days: P&L = 0. Break-even win rate: 80%.
const WIN_PNL  = 150;
const LOSS_PNL = -600;

function computePnL(rows, bands, initialCapital) {
  const trades = [];
  let equity = initialCapital;
  let peak = initialCapital;
  let maxDD = 0;

  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const b = bands[i];
    const prev = rows[i - 1];
    if (!b) continue;

    const baseRow = {
      date: r.date, prevClose: prev.close, close: r.close,
      regime: b.regime, upper: b.upper, lower: b.lower,
    };

    if (!b.trade) {
      trades.push({ ...baseRow, status: 'no-trade', distUp: null, distDn: null, pnl: 0, equity });
      continue;
    }

    const distUp = (b.upper / prev.close - 1) * 100;
    const distDn = (1 - b.lower / prev.close) * 100;

    let pnl, status;
    if (r.close >= b.upper)      { pnl = LOSS_PNL; status = 'loss-up'; }
    else if (r.close <= b.lower) { pnl = LOSS_PNL; status = 'loss-dn'; }
    else                         { pnl = WIN_PNL;  status = 'win';     }

    equity += pnl;
    if (equity > peak) peak = equity;
    const dd = (peak - equity) / peak * 100;
    if (dd > maxDD) maxDD = dd;

    trades.push({ ...baseRow, status, distUp, distDn, pnl, equity });
  }

  return { trades, finalEquity: equity, peak, maxDD, initialCapital };
}

function pnlSummary(pnl) {
  const traded = pnl.trades.filter(t => t.status !== 'no-trade');
  const wins   = traded.filter(t => t.status === 'win');
  const losses = traded.filter(t => t.status !== 'win');
  const sumWins   = wins.reduce((a, t) => a + t.pnl, 0);
  const sumLosses = losses.reduce((a, t) => a + Math.abs(t.pnl), 0);
  const best  = traded.reduce((a, t) => t.pnl > (a?.pnl ?? -Infinity) ? t : a, null);
  const worst = traded.reduce((a, t) => t.pnl < (a?.pnl ??  Infinity) ? t : a, null);
  const totalPnL  = pnl.finalEquity - pnl.initialCapital;
  const returnPct = totalPnL / pnl.initialCapital * 100;
  const expectancy = traded.length ? totalPnL / traded.length : 0;
  const profitFactor = sumLosses > 0 ? sumWins / sumLosses : Infinity;
  return {
    tradedDays: traded.length, wins: wins.length, losses: losses.length,
    totalPnL, returnPct, expectancy, profitFactor,
    best, worst, maxDD: pnl.maxDD, finalEquity: pnl.finalEquity,
  };
}

// ---- Equity chart --------------------------------------------------------
function drawEquityChart(pnl) {
  const dates = pnl.trades.map(t => t.date);
  const equity = pnl.trades.map(t => t.equity);

  const lossUpX = [], lossUpY = [], lossDnX = [], lossDnY = [];
  pnl.trades.forEach(t => {
    if (t.status === 'loss-up') { lossUpX.push(t.date); lossUpY.push(t.equity); }
    if (t.status === 'loss-dn') { lossDnX.push(t.date); lossDnY.push(t.equity); }
  });

  const traces = [
    { x: dates, y: equity, mode: 'lines', name: 'Equity',
      line: { color: '#143a64', width: 2.2 },
      fill: 'tozeroy', fillcolor: 'rgba(44, 111, 163, 0.08)' },
    { x: lossUpX, y: lossUpY, mode: 'markers', name: 'Loss (call hit)',
      marker: { color: '#b73232', size: 9, symbol: 'x', line: { width: 1, color: '#fff' } } },
    { x: lossDnX, y: lossDnY, mode: 'markers', name: 'Loss (put hit)',
      marker: { color: '#c46a35', size: 9, symbol: 'x', line: { width: 1, color: '#fff' } } },
  ];

  const layout = {
    margin: { t: 20, r: 20, b: 50, l: 70 },
    paper_bgcolor: '#ffffff', plot_bgcolor: '#f3f8fc',
    font: { family: 'Segoe UI, sans-serif', color: '#0c1f33' },
    xaxis: { title: 'Date', type: 'date', gridcolor: '#dce7f1', linecolor: '#143a64' },
    yaxis: {
      title: 'Equity ($)', gridcolor: '#dce7f1', linecolor: '#143a64',
      rangemode: 'tozero',
    },
    legend: { orientation: 'h', y: -0.22,
              bgcolor: 'rgba(255,255,255,0.6)', bordercolor: '#cfe1f2', borderwidth: 1 },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#0a2540', font: { color: '#fff' }, bordercolor: '#c9a227' },
    shapes: [{
      type: 'line', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: pnl.initialCapital, y1: pnl.initialCapital,
      line: { color: '#c9a227', width: 1.2, dash: 'dash' },
    }],
    annotations: [{
      xref: 'paper', yref: 'y', x: 0, y: pnl.initialCapital,
      xanchor: 'left', yanchor: 'bottom',
      text: ` Capital inicial $${pnl.initialCapital.toFixed(0)} `,
      font: { size: 10, color: '#8a6d10' },
      bgcolor: 'rgba(255,255,255,0.7)', showarrow: false,
    }],
  };
  safePlotly('equityChart', traces, layout, { responsive: true });
}

function renderEquityStats(s, pnl) {
  const fmt   = v => (v >= 0 ? '+' : '−') + '$' + Math.abs(v).toFixed(2);
  const fmtPc = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) + '%';
  const goodBad = v => v >= 0 ? 'good' : 'bad';
  const bestStr  = s.best  ? `${fmt(s.best.pnl)} (${s.best.date})`   : '—';
  const worstStr = s.worst ? `${fmt(s.worst.pnl)} (${s.worst.date})` : '—';

  document.getElementById('equityStats').innerHTML = `
    <div class="stats-card">
      <h3>P&amp;L total</h3>
      <div class="big ${goodBad(s.totalPnL)}">${fmt(s.totalPnL)}</div>
      <div style="font-size:11px;color:#888;margin-top:4px">
        Equity ${pnl.initialCapital} → <b>${pnl.finalEquity.toFixed(2)}</b> (${fmtPc(s.returnPct)})
      </div>
    </div>
    <div class="stats-card" style="min-width:240px">
      <h3>Métricas económicas</h3>
      <div class="row"><span>Días operados</span><span><b>${s.tradedDays}</b></span></div>
      <div class="row"><span>Expectancia / día</span><span class="${goodBad(s.expectancy)}"><b>${fmt(s.expectancy)}</b></span></div>
      <div class="row"><span>Profit factor</span><span><b>${isFinite(s.profitFactor) ? s.profitFactor.toFixed(2) : '∞'}</b></span></div>
      <div class="row"><span>Max drawdown</span><span class="bad"><b>−${s.maxDD.toFixed(2)}%</b></span></div>
      <div class="row" style="border-top:1px solid #eee;margin-top:4px;padding-top:4px">
        <span style="color:#080">Mejor día</span><span style="color:#080">${bestStr}</span>
      </div>
      <div class="row"><span class="loss-up">Peor día</span><span class="loss-up">${worstStr}</span></div>
    </div>
  `;
}

// ---- Trade log table -----------------------------------------------------
function renderTradesTable(pnl) {
  const status2label = {
    'win':      '<span style="color:#1e7a4d">✓ Win</span>',
    'loss-up':  '<span class="loss-up">✗ Call hit</span>',
    'loss-dn':  '<span class="loss-dn">✗ Put hit</span>',
    'no-trade': '<span style="color:#9b8a4b">— No-trade</span>',
  };
  const fmt$ = v => '$' + v.toFixed(2);
  const rows = pnl.trades.map(t => `
    <tr class="${t.status}">
      <td>${t.date}</td>
      <td>${t.prevClose.toFixed(2)}</td>
      <td>${t.lower.toFixed(2)} – ${t.upper.toFixed(2)}</td>
      <td>${t.distDn !== null ? '−'+t.distDn.toFixed(2)+'%' : '—'} / ${t.distUp !== null ? '+'+t.distUp.toFixed(2)+'%' : '—'}</td>
      <td>${t.close.toFixed(2)}</td>
      <td>${status2label[t.status]}</td>
      <td class="pnl-${t.pnl >= 0 ? 'pos' : 'neg'}"><b>${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(0)}</b></td>
      <td><b>${fmt$(t.equity)}</b></td>
    </tr>
  `).join('');

  document.getElementById('tradesTable').innerHTML = `
    <thead><tr>
      <th>Date</th><th>Prev close</th><th>Bands (low–up)</th><th>Dist (put / call)</th>
      <th>Close</th><th>Outcome</th><th>P&amp;L</th><th>Equity</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  `;
}

// ---- Quant Agent: optimizer ---------------------------------------------
// Per-regime grid search. Each regime is independent because a given day
// belongs to exactly one regime, so its params don't affect other regimes.
// Score = expectancy (= +150 per win, −600 per loss) over the training window.
const OPT_RANGE = { min: 0.5, max: 2.0, step: 0.1 };

// Pre-computed grid array (way faster than re-iterating a generator each pass).
const OPT_GRID = (() => {
  const arr = [];
  for (let v = OPT_RANGE.min; v <= OPT_RANGE.max + 1e-9; v += OPT_RANGE.step) {
    arr.push(Math.round(v * 100) / 100);
  }
  return arr;
})();
const OPT_GRID_N = OPT_GRID.length;

// Backwards-compat for the standalone Quant Agent (still uses gridValues).
function* gridValues({ min, max, step }) {
  for (let v = min; v <= max + 1e-9; v += step) yield Math.round(v * 100) / 100;
}

// Indexed access — no iterator object per call.
// Applies asymmetric Ccall/Cput compression factors per side.
function scoreParams(days, upperMult, lowerMult, k) {
  let score = 0;
  const n = days.length;
  for (let i = 0; i < n; i++) {
    const d = days[i];
    const cf = compressionFactor(d.prev);
    const upper = d.prev.close * (1 + cf.Ccall * k * upperMult * d.sigUp);
    const lower = d.prev.close * (1 - cf.Cput  * k * lowerMult * d.sigDn);
    if (d.row.close >= upper || d.row.close <= lower) score += LOSS_PNL;
    else score += WIN_PNL;
  }
  return score;
}

function optimizeRegimes(rows, baseBands, baseRegimes, windowDays) {
  const trainStart = Math.max(1, rows.length - windowDays);
  // Group days by regime within the training window
  const regimeDays = new Map();
  for (let i = trainStart; i < rows.length; i++) {
    const b = baseBands[i];
    if (!b || !b.trade) continue;
    if (!regimeDays.has(b.regime)) regimeDays.set(b.regime, []);
    regimeDays.get(b.regime).push({
      row: rows[i], prev: rows[i - 1], sigUp: b.sigUp, sigDn: b.sigDn,
    });
  }

  const newRegimes = baseRegimes.map(r => ({ ...r }));
  const perRegimeStats = [];

  for (let regIdx = 0; regIdx < baseRegimes.length; regIdx++) {
    const reg = baseRegimes[regIdx];
    if (!reg.trade) {
      perRegimeStats.push({ regIdx, sample: 0, baseScore: 0, optScore: 0 });
      continue;
    }
    const days = regimeDays.get(reg) || [];
    const baseScore = scoreParams(days, reg.upperMult, reg.lowerMult, reg.k);
    if (days.length === 0) {
      perRegimeStats.push({ regIdx, sample: 0, baseScore: 0, optScore: 0 });
      continue;
    }

    let bestScore = baseScore;
    let bestParams = { upperMult: reg.upperMult, lowerMult: reg.lowerMult, k: reg.k };
    for (const u of gridValues(OPT_RANGE)) {
      for (const l of gridValues(OPT_RANGE)) {
        for (const k of gridValues(OPT_RANGE)) {
          const s = scoreParams(days, u, l, k);
          if (s > bestScore) {
            bestScore = s;
            bestParams = { upperMult: u, lowerMult: l, k };
          }
        }
      }
    }
    Object.assign(newRegimes[regIdx], bestParams);
    perRegimeStats.push({ regIdx, sample: days.length, baseScore, optScore: bestScore });
  }

  const totalSample = [...regimeDays.values()].reduce((a, d) => a + d.length, 0);
  return {
    newRegimes,
    trainingStartDate: rows[trainStart].date,
    trainingEndDate: rows[rows.length - 1].date,
    totalSample,
    perRegimeStats,
  };
}

// Recompute bands for the full history with new regime params,
// reusing pre-computed sigmas from the base bands (cheap, O(N)).
function rebandWithNewRegimes(rows, baseBands, baseRegimes, newRegimes) {
  const map = new Map();
  baseRegimes.forEach((r, i) => map.set(r, newRegimes[i]));
  return baseBands.map((b, i) => {
    if (!b) return null;
    const newReg = map.get(b.regime);
    const prev = rows[i - 1];
    const cf = compressionFactor(prev);
    return {
      ...b,
      upper: prev.close * (1 + cf.Ccall * newReg.k * newReg.upperMult * b.sigUp),
      lower: prev.close * (1 - cf.Cput  * newReg.k * newReg.lowerMult * b.sigDn),
      regime: newReg, trade: newReg.trade,
      Cbase: cf.base, Ccall: cf.Ccall, Cput: cf.Cput, trend: cf.trend,
    };
  });
}

// ---- Quant Agent: proposal rendering ------------------------------------
const _proposalCache = new Map(); // periodKey → newRegimes (for Apply button)

function inSampleStats(pnl, fromDate) {
  const traded = pnl.trades.filter(t => t.date >= fromDate && t.status !== 'no-trade');
  const wins   = traded.filter(t => t.status === 'win').length;
  const total  = traded.length;
  const sum    = traded.reduce((a, t) => a + t.pnl, 0);
  return { wins, total, winRate: total ? wins / total * 100 : 0, pnl: sum };
}

function fullStats(pnl) {
  const traded = pnl.trades.filter(t => t.status !== 'no-trade');
  const wins   = traded.filter(t => t.status === 'win').length;
  return { winRate: traded.length ? wins / traded.length * 100 : 0 };
}

function renderProposal(containerId, periodKey, periodLabel, baseRegimes, optResult, basePnL, optPnL) {
  _proposalCache.set(periodKey, optResult.newRegimes);

  const fromDate = optResult.trainingStartDate;
  const inBase = inSampleStats(basePnL, fromDate);
  const inOpt  = inSampleStats(optPnL,  fromDate);
  const fullBase = fullStats(basePnL);
  const fullOpt  = fullStats(optPnL);

  const showWarn = optResult.totalSample < 30;
  const fmt$ = v => (v >= 0 ? '+$' : '−$') + Math.abs(v).toFixed(0);
  const arrow = (a, b, betterIsHigher = true) => {
    const diff = b - a;
    if (Math.abs(diff) < 1e-6) return '<span class="arrow-eq">=</span>';
    const isUp = diff > 0;
    const isGood = (isUp && betterIsHigher) || (!isUp && !betterIsHigher);
    return `<span class="arrow-${isUp ? 'up' : 'down'}-${isGood ? 'good' : 'bad'}">${isUp ? '?' : '?'}</span>`;
  };

  const paramRows = baseRegimes.map((r, i) => {
    if (!r.trade) {
      return `<tr class="notrade"><td>${regimeLabel(baseRegimes, i)}</td><td colspan="3" style="font-style:italic">no-trade · sin optimizar</td></tr>`;
    }
    const n = optResult.newRegimes[i];
    const sampleStat = optResult.perRegimeStats[i] || { sample: 0 };
    const sampleHint = sampleStat.sample === 0
      ? '<div class="sample-warn">0 días en ventana</div>'
      : `<div class="sample-hint">${sampleStat.sample} días</div>`;
    return `<tr>
      <td>${regimeLabel(baseRegimes, i)}${sampleHint}</td>
      <td>${r.k.toFixed(2)} → <b>${n.k.toFixed(2)}</b> ${arrow(r.k, n.k, false)}</td>
      <td>${r.upperMult.toFixed(2)} → <b>${n.upperMult.toFixed(2)}</b> ${arrow(r.upperMult, n.upperMult, false)}</td>
      <td>${r.lowerMult.toFixed(2)} → <b>${n.lowerMult.toFixed(2)}</b> ${arrow(r.lowerMult, n.lowerMult, false)}</td>
    </tr>`;
  }).join('');

  const html = `
    <h3>${periodLabel}</h3>
    <div class="period-info">
      <span><b>Training:</b> ${optResult.trainingStartDate} → ${optResult.trainingEndDate}</span>
      <span><b>${optResult.totalSample}</b> trades en ventana</span>
    </div>
    ${showWarn ? `<div class="warn-box">⚠ Muestra pequeña (${optResult.totalSample} trades) — alto riesgo de sobreajuste.</div>` : ''}
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Win rate (in-sample)</div>
        <div class="metric-value">
          ${inBase.winRate.toFixed(1)}% → <b class="${inOpt.winRate >= inBase.winRate ? 'd-good' : 'd-bad'}">${inOpt.winRate.toFixed(1)}%</b>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">P&amp;L (in-sample)</div>
        <div class="metric-value">
          ${fmt$(inBase.pnl)} → <b class="${inOpt.pnl >= inBase.pnl ? 'd-good' : 'd-bad'}">${fmt$(inOpt.pnl)}</b>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Equity final (full history)</div>
        <div class="metric-value">
          $${basePnL.finalEquity.toFixed(0)} → <b class="${optPnL.finalEquity >= basePnL.finalEquity ? 'd-good' : 'd-bad'}">$${optPnL.finalEquity.toFixed(0)}</b>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Win rate (full history)</div>
        <div class="metric-value">
          ${fullBase.winRate.toFixed(1)}% → <b class="${fullOpt.winRate >= fullBase.winRate ? 'd-good' : 'd-bad'}">${fullOpt.winRate.toFixed(1)}%</b>
        </div>
      </div>
    </div>
    <div class="params-block">
      <div class="block-label">Parámetros propuestos por régimen</div>
      <table class="agent-params">
        <thead><tr><th>Régimen</th><th>k</th><th>Upper mult</th><th>Lower mult</th></tr></thead>
        <tbody>${paramRows}</tbody>
      </table>
    </div>
    <div class="equity-cmp" id="equity-cmp-${periodKey}"></div>
    <button class="apply-btn" data-period="${periodKey}">Aplicar esta optimización</button>
  `;

  document.getElementById(containerId).innerHTML = html;
  drawAgentEquity(`equity-cmp-${periodKey}`, basePnL, optPnL, optResult.trainingStartDate);
}

function drawAgentEquity(elementId, basePnL, optPnL, trainStart) {
  const baseTrace = {
    x: basePnL.trades.map(t => t.date),
    y: basePnL.trades.map(t => t.equity),
    mode: 'lines', name: 'Base',
    line: { color: '#143a64', width: 2 },
  };
  const optTrace = {
    x: optPnL.trades.map(t => t.date),
    y: optPnL.trades.map(t => t.equity),
    mode: 'lines', name: 'Agent',
    line: { color: '#c9a227', width: 2.2 },
  };
  const lastDate = basePnL.trades[basePnL.trades.length - 1].date;
  const layout = {
    margin: { t: 14, r: 12, b: 44, l: 56 },
    paper_bgcolor: '#ffffff', plot_bgcolor: '#f3f8fc',
    font: { family: 'Segoe UI, sans-serif', color: '#0c1f33', size: 11 },
    xaxis: { type: 'date', gridcolor: '#dce7f1' },
    yaxis: { title: 'Equity ($)', gridcolor: '#dce7f1' },
    legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#0a2540', font: { color: '#fff' } },
    shapes: [{
      type: 'rect', xref: 'x', yref: 'paper',
      x0: trainStart, x1: lastDate, y0: 0, y1: 1,
      fillcolor: 'rgba(201, 162, 39, 0.14)',
      line: { width: 0 }, layer: 'below',
    }],
    annotations: [{
      xref: 'x', yref: 'paper', x: trainStart, y: 1,
      xanchor: 'left', yanchor: 'top',
      text: ' Training window ',
      bgcolor: 'rgba(255,255,255,0.85)',
      font: { size: 9, color: '#8a6d10' },
      showarrow: false,
    }],
  };
  safePlotly(elementId, [baseTrace, optTrace], layout, { responsive: true, displayModeBar: false });
}

// Trading-day window estimation (rough): 21 / 63 / 126 trading days
const AGENT_PERIODS = [
  { key: '1m', label: 'Mejora 1 — Último mes',     days:  21 },
  { key: '3m', label: 'Mejora 2 — Últimos 3 meses', days:  63 },
  { key: '6m', label: 'Mejora 3 — Últimos 6 meses', days: 126 },
];

function runAgent() {
  if (!currentRows) return;
  const lookback = parseInt(document.getElementById('lookback').value, 10);
  const initialCapital = parseFloat(document.getElementById('initialCapital').value) || 1000;
  const baseRegimes = readRegimes();
  const { bands: baseBands } = computeBands(currentRows, baseRegimes, lookback);
  const basePnL = computePnL(currentRows, baseBands, initialCapital);

  const allProposals = [];
  for (const period of AGENT_PERIODS) {
    const optResult = optimizeRegimes(currentRows, baseBands, baseRegimes, period.days);
    const optBands  = rebandWithNewRegimes(currentRows, baseBands, baseRegimes, optResult.newRegimes);
    const optPnL    = computePnL(currentRows, optBands, initialCapital);
    renderProposal(`proposal-${period.key}`, period.key, period.label, baseRegimes, optResult, basePnL, optPnL);
    allProposals.push({ ...period, optResult, optPnL });
  }

  renderAgentLegend(currentRows, baseRegimes, basePnL, allProposals);
}

function renderAgentLegend(rows, baseRegimes, basePnL, proposals) {
  const lastRow = rows[rows.length - 1];
  const currentRegime = regimeFor(lastRow.vix, baseRegimes);
  const currentRegimeIdx = baseRegimes.indexOf(currentRegime);
  const currentRegimeLbl = regimeLabel(baseRegimes, currentRegimeIdx);

  const ranked = proposals.map(p => ({
    key: p.key, label: p.label, optResult: p.optResult, optPnL: p.optPnL,
    finalEquity: p.optPnL.finalEquity,
    deltaVsBase: p.optPnL.finalEquity - basePnL.finalEquity,
  })).sort((a, b) => b.finalEquity - a.finalEquity);

  const best = ranked[0];
  const noneBeatBase = ranked.every(r => r.deltaVsBase <= 0);

  const fmt$ = v => (v >= 0 ? '+$' : '−$') + Math.abs(v).toFixed(0);

  // Direction agreement check: count how many proposals push each parameter same direction
  // (useful for the 1m-wins case to decide if it's robust or noise)
  const agreementHint = (() => {
    let agreeCount = 0, paramCount = 0;
    baseRegimes.forEach((reg, i) => {
      if (!reg.trade) return;
      ['k', 'upperMult', 'lowerMult'].forEach(f => {
        paramCount++;
        const dirs = proposals.map(p => Math.sign(p.optResult.newRegimes[i][f] - reg[f]));
        if (dirs.every(d => d === dirs[0]) && dirs[0] !== 0) agreeCount++;
      });
    });
    return paramCount ? agreeCount / paramCount : 0;
  })();

  let recommendation = '';
  if (noneBeatBase) {
    recommendation = `<p>Ninguna de las tres propuestas mejora la equity histórica del modelo base.
      <span class="pill">Mantener parámetros actuales</span>.
      Esto puede indicar que tu config ya está bien calibrada, o que las tres ventanas
      reflejan regímenes de mercado distintos y el agente no encuentra una mejora robusta común.</p>`;
  } else if (best.key === '6m') {
    recommendation = `<p>La propuesta de <b>6 meses</b> es la que más equity acumula en el histórico completo
      (<b>${fmt$(best.deltaVsBase)}</b> vs base) y se basa en la muestra más amplia
      (${best.optResult.totalSample} trades). Es la opción <b>más robusta estadísticamente</b>.
      <span class="pill gold">Aplicar Mejora 3 (6m)</span></p>`;
  } else if (best.key === '3m') {
    const sixM = ranked.find(p => p.key === '6m');
    recommendation = `<p>La propuesta de <b>3 meses</b> rinde mejor en histórico completo
      (<b>${fmt$(best.deltaVsBase)}</b> vs base) que la de 6 meses
      (${fmt$(sixM.deltaVsBase)}). Sugiere que el mercado <b>reciente</b> tiene un perfil distinto
      al del semestre. <span class="pill gold">Aplicar Mejora 2 (3m)</span> y vigilar las próximas semanas.</p>`;
  } else {
    // 1m wins
    const agreementHigh = agreementHint >= 0.6;
    recommendation = `<p>La propuesta de <b>1 mes</b> es la más rentable
      (<b>${fmt$(best.deltaVsBase)}</b> vs base), pero está entrenada con sólo
      ${best.optResult.totalSample} trades — <b>alto riesgo de sobreajuste</b>.</p>
      ${agreementHigh
        ? `<p>Las tres propuestas coinciden bastante en la dirección de los cambios
            (~${(agreementHint*100).toFixed(0)}% de los parámetros mueven en el mismo sentido),
            así que el patrón parece real. <span class="pill gold">Puedes aplicar Mejora 1</span>,
            pero monitoriza de cerca.</p>`
        : `<p>Las propuestas discrepan bastante entre sí — sólo el ${(agreementHint*100).toFixed(0)}%
            de los parámetros mueve en la misma dirección. El "ganador" es probablemente ruido del mes.
            <span class="pill">Recomiendo aplicar Mejora 2 o 3 en su lugar</span>.</p>`}`;
  }

  const regimeNote = currentRegime.trade
    ? `<p>El cierre más reciente es <b>VIX ${lastRow.vix.toFixed(2)}</b>, dentro del régimen
       <b>${currentRegimeLbl}</b>. La próxima sesión se opera con los parámetros de ese régimen
       — fíjate especialmente en cómo lo modifica cada propuesta.</p>`
    : `<p>El cierre más reciente es <b>VIX ${lastRow.vix.toFixed(2)}</b>, en una zona
       <b>${currentRegimeLbl} (no-trade)</b>. La próxima sesión no se opera, así que
       cualquier ajuste sólo afectará cuando el VIX vuelva a una zona operable.</p>`;

  document.getElementById('agentLegend').innerHTML = `
    <h3>Cómo leer este panel</h3>

    <div class="legend-section">
      <div class="legend-title"><span class="badge">↔</span>Trading window y franja dorada</div>
      <p>Cada propuesta entrena los parámetros sobre una <b>ventana del histórico</b>:
        <b>Mejora 1</b> usa los últimos 21 días de trading,
        <b>Mejora 2</b> los últimos 63, y <b>Mejora 3</b> los últimos 126.</p>
      <p>Esa ventana es la <b>franja dorada</b> que ves en el gráfico de equity. Dentro de la franja,
        el agente <i>vio</i> los datos al optimizar — por eso la línea dorada (propuesta) suele
        superar siempre a la navy (base) ahí.</p>
      <p><b>Lo que importa de verdad es lo que pasa fuera de la franja</b>: ahí la línea dorada
        está corriendo sobre datos que el agente nunca vio (validación <i>out-of-sample</i>).
        Si la línea dorada sigue por encima de la navy fuera de la franja, la mejora es robusta.
        Si se desploma, era curva-fitting del período.</p>
    </div>

    <div class="legend-section">
      <div class="legend-title"><span class="badge">②</span>Modelo de optimización</div>
      <p>Búsqueda exhaustiva por régimen (<b>grid search</b>) sobre k, upper mult y lower mult.
        Maximiza expectancia con el modelo simplificado <b>+$150</b> por win / <b>−$600</b> por loss.</p>
    </div>

    <div class="legend-section">
      <div class="legend-title"><span class="badge">③</span>Recomendación del agente</div>
      ${recommendation}
      ${regimeNote}
    </div>
  `;
}

function applyAgentProposal(periodKey) {
  const newRegimes = _proposalCache.get(periodKey);
  if (!newRegimes) return;
  newRegimes.forEach((reg, i) => {
    const set = (f, v) => {
      const inp = document.querySelector(`#regimeTable input[data-i="${i}"][data-f="${f}"]`);
      if (inp) inp.value = v;
    };
    set('upperMult', reg.upperMult);
    set('lowerMult', reg.lowerMult);
    set('k',         reg.k);
  });
  closeAgentModal();
  recalc();
}

function openAgentModal()  { document.getElementById('agentModal').style.display = 'flex'; runAgent(); }
function closeAgentModal() { document.getElementById('agentModal').style.display = 'none'; }

// ==========================================================================
// JEFE DE MESA — random-search optimizer constrained on avg strike distance
// ==========================================================================
// Generates random configurations, filters to those whose average strike
// distance over the last 3 months matches the user's target (within tolerance),
// then ranks the survivors by strict win rate.
const JEFE_WINDOW_DAYS = 63; // ~3 months
const _jefeCache = new Map(); // rank → config (for Apply button)

function rand(a, b) { return a + Math.random() * (b - a); }
function randStep(a, b, step) {
  const v = rand(a, b);
  return Math.round(v / step) * step;
}

// Generate a random configuration. Keeps regime maxVix and trade flags fixed
// (we don't reclassify days), but varies k, upperMult, lowerMult per operable regime.
function randomJefeConfig(baseRegimes) {
  let Cmin = randStep(0.30, 0.55, 0.05);
  let Cmax = randStep(0.55, 0.85, 0.05);
  if (Cmax <= Cmin + 0.05) Cmax = Math.min(0.85, Cmin + 0.10);
  return {
    compressionParams: {
      lambda:      randStep(0.5, 3.0, 0.25),
      Cmin, Cmax,
      Vref:        Math.round(rand(14, 22)),
      shiftFactor: randStep(0, 0.45, 0.05),
    },
    compressionVars: [
      { id: 'iv_hv',   active: Math.random() > 0.30, weight: randStep(0,    2.0, 0.25) },
      { id: 'iv_rank', active: Math.random() > 0.55, weight: randStep(0,    1.5, 0.25) },
      { id: 'iv_pctl', active: Math.random() > 0.55, weight: randStep(0,    1.5, 0.25) },
      { id: 'vix',     active: Math.random() > 0.30, weight: randStep(0,    1.5, 0.25) },
      { id: 'iv_chg',  active: Math.random() > 0.55, weight: randStep(0,    1.5, 0.25) },
      { id: 'pcv',     active: Math.random() > 0.55, weight: randStep(0,    1.5, 0.25) },
    ],
    regimes: baseRegimes.map(reg => reg.trade ? {
      ...reg,
      k:         randStep(0.6, 1.8, 0.1),
      upperMult: randStep(0.7, 1.8, 0.1),
      lowerMult: randStep(0.7, 1.8, 0.1),
    } : { ...reg }),
  };
}

// Score a config inline over the last JEFE_WINDOW_DAYS days.
// Returns avgDistCall, avgDistPut, win rate (strict), counts, etc.
function scoreJefeConfig(rows, baseBands, config) {
  const start = Math.max(1, rows.length - JEFE_WINDOW_DAYS);
  const cp = config.compressionParams;
  let cleanWins = 0, touched = 0, losses = 0;
  let sumDistCall = 0, sumDistPut = 0, count = 0;

  for (let i = start; i < rows.length; i++) {
    const b = baseBands[i];
    if (!b) continue;
    const prev = rows[i - 1];
    const r = rows[i];
    const reg = regimeFor(prev.vix, config.regimes);
    if (!isDayAccepted(r, reg)) continue;

    // Z = Σ wᵢ × sᵢ
    let Z = 0;
    for (let j = 0; j < config.compressionVars.length; j++) {
      const v = config.compressionVars[j];
      if (!v.active) continue;
      const s = standardizeVar(v.id, prev, cp);
      if (s === null || !isFinite(s)) continue;
      Z += v.weight * s;
    }
    const Craw = 1 / (1 + Math.exp(-cp.lambda * Z));
    const base = Math.max(cp.Cmin, Math.min(cp.Cmax, Craw));
    const delta = cp.shiftFactor * (cp.Cmax - cp.Cmin);
    let Ccall = base, Cput = base;
    if (prev.trend === 'up')        { Ccall = base + delta; Cput = base - delta; }
    else if (prev.trend === 'down') { Ccall = base - delta; Cput = base + delta; }
    Ccall = Math.max(cp.Cmin, Math.min(cp.Cmax, Ccall));
    Cput  = Math.max(cp.Cmin, Math.min(cp.Cmax, Cput));

    const upper = prev.close * (1 + Ccall * reg.k * reg.upperMult * b.sigUp);
    const lower = prev.close * (1 - Cput  * reg.k * reg.lowerMult * b.sigDn);
    const callStrike = Math.ceil(upper / 5) * 5;
    const putStrike  = Math.floor(lower / 5) * 5;

    sumDistCall += (callStrike - prev.close) / prev.close * 100;
    sumDistPut  += (prev.close - putStrike)  / prev.close * 100;
    count++;

    const callTouched = isFinite(r.high) && r.high >= callStrike;
    const putTouched  = isFinite(r.low)  && r.low  <= putStrike;
    if (r.close >= upper)        losses++;
    else if (r.close <= lower)   losses++;
    else if (callTouched || putTouched) touched++;
    else                         cleanWins++;
  }
  const total = cleanWins + touched + losses;
  return {
    cleanWins, touched, losses,
    winRateStrict: total > 0 ? cleanWins / total * 100 : 0,
    winRateClose:  total > 0 ? (cleanWins + touched) / total * 100 : 0,
    avgDistCall:   count > 0 ? sumDistCall / count : 0,
    avgDistPut:    count > 0 ? sumDistPut / count  : 0,
    sampleCount:   count,
  };
}

async function runJefeDeMesa() {
  if (!currentRows) {
    alert('Carga el CSV primero.');
    return;
  }
  const btn = document.getElementById('jefeRunBtn');
  const status = document.getElementById('jefeStatus');
  const bar = document.getElementById('jefeProgressBar');
  const fill = document.getElementById('jefeProgressFill');
  const resultsEl = document.getElementById('jefeResults');

  const targetCall = parseFloat(document.getElementById('jefeTargetCall').value);
  const targetPut  = parseFloat(document.getElementById('jefeTargetPut').value);
  const tolerance  = parseFloat(document.getElementById('jefeTolerance').value);
  const samples    = Math.max(1000, parseInt(document.getElementById('jefeSamples').value, 10) || 10000);

  if (!isFinite(targetCall) || !isFinite(targetPut) || !isFinite(tolerance)) {
    alert('Revisa los valores numéricos del formulario.');
    return;
  }

  btn.disabled = true;
  bar.style.display = 'block';
  fill.style.width = '0%';
  resultsEl.innerHTML = '';
  status.style.color = '';

  try {
    const lookback = parseInt(document.getElementById('lookback').value, 10);
    const baseRegimes = readRegimes();
    const { bands: baseBands } = computeBands(currentRows, baseRegimes, lookback);

    // Always treat targets as positive magnitudes (handle if user types -1.80)
    const targetCallAbs = Math.abs(targetCall);
    const targetPutAbs  = Math.abs(targetPut);

    const matches = [];
    let minCall = Infinity, maxCall = -Infinity;
    let minPut  = Infinity, maxPut  = -Infinity;
    let bestNear = null, bestNearScore = Infinity;
    const t0 = performance.now();

    for (let i = 0; i < samples; i++) {
      const cfg = randomJefeConfig(baseRegimes);
      const stats = scoreJefeConfig(currentRows, baseBands, cfg);
      if (stats.sampleCount === 0) continue;

      // Range tracking
      if (stats.avgDistCall < minCall) minCall = stats.avgDistCall;
      if (stats.avgDistCall > maxCall) maxCall = stats.avgDistCall;
      if (stats.avgDistPut  < minPut)  minPut  = stats.avgDistPut;
      if (stats.avgDistPut  > maxPut)  maxPut  = stats.avgDistPut;

      // Best near-match (combined distance to target)
      const nearScore = Math.abs(stats.avgDistCall - targetCallAbs) + Math.abs(stats.avgDistPut - targetPutAbs);
      if (nearScore < bestNearScore) { bestNearScore = nearScore; bestNear = { config: cfg, stats }; }

      // Strict match
      if (Math.abs(stats.avgDistCall - targetCallAbs) <= tolerance &&
          Math.abs(stats.avgDistPut  - targetPutAbs)  <= tolerance) {
        matches.push({ config: cfg, stats });
      }
      if (i % 500 === 0) {
        fill.style.width = (i / samples * 100).toFixed(1) + '%';
        status.textContent = `Probando ${i.toLocaleString()}/${samples.toLocaleString()} · ${matches.length} matches…`;
        await new Promise(r => setTimeout(r, 0));
      }
    }

    fill.style.width = '100%';
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    if (matches.length === 0) {
      // No exact match — show diagnostic + near-best
      status.style.color = '#b06000';
      status.textContent = `⚠ Ningún match exacto en ${elapsed}s. Mostrando la mejor aproximación.`;

      _jefeCache.clear();
      _jefeCache.set(0, bestNear.config);

      const callDelta = bestNear.stats.avgDistCall - targetCallAbs;
      const putDelta  = bestNear.stats.avgDistPut  - targetPutAbs;

      resultsEl.innerHTML = `
        <div style="grid-column:1/-1;background:#fff7df;border:1px solid var(--gold-500);border-left:4px solid var(--gold-500);border-radius:8px;padding:14px;margin-bottom:12px">
          <h4 style="margin:0 0 8px;color:#6b5b2e">📊 Diagnóstico de la búsqueda</h4>
          <div style="font-size:12px;line-height:1.7;color:#5a4a18">
            <b>Targets pedidos</b>: call <b>+${targetCallAbs.toFixed(2)}%</b> · put <b>−${targetPutAbs.toFixed(2)}%</b> · tolerancia <b>±${tolerance.toFixed(2)}%</b><br>
            <b>Rango factible probado</b>: call <b>+${minCall.toFixed(2)}% a +${maxCall.toFixed(2)}%</b> · put <b>−${minPut.toFixed(2)}% a −${maxPut.toFixed(2)}%</b><br>
            ${(targetCallAbs < minCall || targetCallAbs > maxCall)
              ? `<span style="color:#b73232">⚠ Tu target call está FUERA del rango factible</span><br>` : ''}
            ${(targetPutAbs < minPut || targetPutAbs > maxPut)
              ? `<span style="color:#b73232">⚠ Tu target put está FUERA del rango factible</span><br>` : ''}
            <b>Mejor aproximación</b>: avg call <b>+${bestNear.stats.avgDistCall.toFixed(3)}%</b>
            (${callDelta >= 0 ? '+' : ''}${callDelta.toFixed(3)} del target),
            avg put <b>−${bestNear.stats.avgDistPut.toFixed(3)}%</b>
            (${putDelta >= 0 ? '+' : ''}${putDelta.toFixed(3)} del target)
          </div>
          <div style="margin-top:10px;font-size:11px;color:#6b5b2e">
            <b>Sugerencias</b>:
            ${(targetCallAbs < minCall || targetCallAbs > maxCall || targetPutAbs < minPut || targetPutAbs > maxPut)
              ? `prueba targets dentro del rango factible (ej. <b>${((minCall + maxCall)/2).toFixed(2)}%</b> / <b>${((minPut + maxPut)/2).toFixed(2)}%</b>)`
              : `aumenta tolerancia a <b>±${Math.max(0.15, bestNearScore.toFixed(2))}%</b> o sube las muestras a 25.000-50.000`}
          </div>
        </div>
        ${renderJefeResult(bestNear, 0)}
      `;
      setTimeout(() => bar.style.display = 'none', 1500);
      btn.disabled = false;
      return;
    }

    // Sort by strict win rate (highest first)
    matches.sort((a, b) => b.stats.winRateStrict - a.stats.winRateStrict);
    const top = matches.slice(0, 5);
    _jefeCache.clear();
    top.forEach((m, i) => _jefeCache.set(i, m.config));

    status.textContent = `✓ Completado en ${elapsed}s · ${matches.length} configs cumplen el target · top 5 ordenadas por win rate estricto`;
    resultsEl.innerHTML = top.map((m, i) => renderJefeResult(m, i)).join('');
    setTimeout(() => { bar.style.display = 'none'; fill.style.width = '0%'; }, 1500);
  } catch (err) {
    console.error('Jefe de mesa error:', err);
    status.style.color = '#b73232';
    status.textContent = `✗ Error: ${err.message || err}`;
  } finally {
    btn.disabled = false;
  }
}

function renderJefeResult(m, rank) {
  const s = m.stats;
  const cp = m.config.compressionParams;
  const activeVars = m.config.compressionVars.filter(v => v.active);
  const isBest = rank === 0;
  const fmtPct = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3) + '%';

  return `
    <div class="jefe-result-card ${isBest ? 'best' : ''}">
      <div class="jefe-rank">${isBest ? '🏆 #1 — MEJOR' : `#${rank + 1}`}</div>
      <div class="jefe-metrics">
        <div class="jefe-metric-card">
          <div class="jefe-metric-lbl">Win rate (estricto)</div>
          <div class="jefe-metric-val" style="color:var(--good)">${s.winRateStrict.toFixed(1)}%</div>
        </div>
        <div class="jefe-metric-card">
          <div class="jefe-metric-lbl">Win rate (close)</div>
          <div class="jefe-metric-val">${s.winRateClose.toFixed(1)}%</div>
        </div>
        <div class="jefe-metric-card">
          <div class="jefe-metric-lbl">Dist call media</div>
          <div class="jefe-metric-val">${fmtPct(s.avgDistCall)}</div>
        </div>
        <div class="jefe-metric-card">
          <div class="jefe-metric-lbl">Dist put media</div>
          <div class="jefe-metric-val">${fmtPct(-s.avgDistPut)}</div>
        </div>
        <div class="jefe-metric-card">
          <div class="jefe-metric-lbl">Limpios / Tocados / Loss</div>
          <div class="jefe-metric-val" style="font-size:12px">
            <span style="color:var(--good)">${s.cleanWins}</span> ·
            <span style="color:#b06000">${s.touched}</span> ·
            <span style="color:var(--bad)">${s.losses}</span>
          </div>
        </div>
        <div class="jefe-metric-card">
          <div class="jefe-metric-lbl">Días en muestra</div>
          <div class="jefe-metric-val">${s.sampleCount}</div>
        </div>
      </div>
      <div class="threshold-block" style="margin-bottom:8px">
        <div class="th-label">Compresión</div>
        <div class="th-vals">
          λ=${cp.lambda} · C<sub>min</sub>=${cp.Cmin} · C<sub>max</sub>=${cp.Cmax} · V<sub>ref</sub>=${cp.Vref} · Δ=${cp.shiftFactor}
        </div>
      </div>
      <div class="threshold-block" style="margin-bottom:8px">
        <div class="th-label">Variables (${activeVars.length}/6)</div>
        <div class="th-vals" style="font-size:11px">
          ${activeVars.length === 0 ? '<i>ninguna</i>' :
            activeVars.map(v => `<b>${v.id}</b>=${v.weight}`).join(' · ')}
        </div>
      </div>
      <div class="threshold-block" style="margin-bottom:8px">
        <div class="th-label">Regímenes operables (k / up / lo)</div>
        <table class="fav-regime-table" style="font-size:10px">
          ${m.config.regimes.filter(r => r.trade).map(r =>
            `<tr><td>VIX≤${r.maxVix}</td><td>${r.k.toFixed(1)}</td><td>${r.upperMult.toFixed(1)}</td><td>${r.lowerMult.toFixed(1)}</td></tr>`
          ).join('')}
        </table>
      </div>
      <button class="apply-btn" data-jefe-rank="${rank}">✓ Aplicar esta configuración</button>
    </div>`;
}

function applyJefeConfig(rank) {
  const cfg = _jefeCache.get(parseInt(rank, 10));
  if (!cfg) return;
  // Compression params
  Object.assign(compressionParams, cfg.compressionParams);
  document.getElementById('cParamLambda').value = cfg.compressionParams.lambda;
  document.getElementById('cParamCmin').value   = cfg.compressionParams.Cmin;
  document.getElementById('cParamCmax').value   = cfg.compressionParams.Cmax;
  document.getElementById('cParamVref').value   = cfg.compressionParams.Vref;
  document.getElementById('cParamShift').value  = cfg.compressionParams.shiftFactor;
  // Compression vars
  cfg.compressionVars.forEach((v, i) => {
    if (compressionVars[i]) {
      compressionVars[i].active = v.active;
      compressionVars[i].weight = v.weight;
    }
  });
  renderCompressionPanel();
  // Regime k/up/lo
  cfg.regimes.forEach((reg, i) => {
    const set = (f, val) => {
      const inp = document.querySelector(`#regimeTable input[data-i="${i}"][data-f="${f}"]`);
      if (!inp) return;
      if (inp.type === 'checkbox') inp.checked = !!val; else inp.value = val;
    };
    set('k', reg.k);
    set('upperMult', reg.upperMult);
    set('lowerMult', reg.lowerMult);
  });
  closeJefeModal();
  recalc();
}

function openJefeModal()  { document.getElementById('jefeModal').style.display = 'flex'; }
function closeJefeModal() { document.getElementById('jefeModal').style.display = 'none'; }

// ==========================================================================
// OPTION CHAINS — parse Barchart CSV, persist, and view stored chains
// ==========================================================================
const CHAINS_KEY = 'spx-vix-chains-v1';

function loadChains() {
  try { return JSON.parse(localStorage.getItem(CHAINS_KEY) || '{}'); }
  catch (_) { return {}; }
}
function saveChains(chains) { localStorage.setItem(CHAINS_KEY, JSON.stringify(chains)); }

// Flexible CSV parser — autodetects delimiter, headers, and column aliases.
// Works with Barchart's wide format (Strike + Call cols + Put cols).
function parseChainCSV(text) {
  if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
  const lines = text.split(/\r?\n/).filter(l => l.trim() !== '');
  if (lines.length < 2) throw new Error('CSV vacío o sin filas de datos.');

  // Detect delimiter
  const delims = [',', ';', '\t'];
  const delim = delims.reduce((b, d) => lines[0].split(d).length > lines[0].split(b).length ? d : b);

  // CSV-split that respects quoted strings
  const splitCSV = (line) => {
    const out = [];
    let cur = '', inQuote = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') { inQuote = !inQuote; continue; }
      if (ch === delim && !inQuote) { out.push(cur); cur = ''; continue; }
      cur += ch;
    }
    out.push(cur);
    return out;
  };

  const headers = splitCSV(lines[0]).map(h => h.trim());
  const norm = s => s.toLowerCase().replace(/[\s_\-\/]/g, '');
  const findCol = (...aliases) => {
    for (const a of aliases) {
      const i = headers.findIndex(h => norm(h) === norm(a));
      if (i >= 0) return i;
    }
    return -1;
  };

  const idx = {
    strike:   findCol('Strike', 'Strike Price', 'K'),
    callBid:  findCol('Call Bid', 'CallBid', 'Bid Call', 'C Bid'),
    callAsk:  findCol('Call Ask', 'CallAsk', 'Ask Call', 'C Ask'),
    callMid:  findCol('Call Mid', 'CallMid'),
    callLast: findCol('Call Last', 'CallLast', 'Last Call', 'Call'),
    callIV:   findCol('Call IV', 'CallIV', 'IV Call', 'Call Implied Volatility', 'C IV'),
    callVol:  findCol('Call Volume', 'CallVolume', 'Volume Call', 'C Vol'),
    callOI:   findCol('Call Open Interest', 'CallOI', 'Call OI', 'C OI'),
    callDelta:findCol('Call Delta', 'CallDelta', 'C Delta'),
    putBid:   findCol('Put Bid', 'PutBid', 'Bid Put', 'P Bid'),
    putAsk:   findCol('Put Ask', 'PutAsk', 'Ask Put', 'P Ask'),
    putMid:   findCol('Put Mid', 'PutMid'),
    putLast:  findCol('Put Last', 'PutLast', 'Last Put', 'Put'),
    putIV:    findCol('Put IV', 'PutIV', 'IV Put', 'Put Implied Volatility', 'P IV'),
    putVol:   findCol('Put Volume', 'PutVolume', 'Volume Put', 'P Vol'),
    putOI:    findCol('Put Open Interest', 'PutOI', 'Put OI', 'P OI'),
    putDelta: findCol('Put Delta', 'PutDelta', 'P Delta'),
  };

  if (idx.strike < 0) {
    throw new Error(`No se encontró la columna "Strike". Headers: ${headers.join(' | ')}`);
  }

  const cleanNum = (s) => {
    if (s === undefined) return NaN;
    s = String(s).trim().replace(/[%$"\s]/g, '');
    if (s === '' || s === '-' || s === 'N/A') return NaN;
    if (s.includes(',') && s.includes('.')) s = s.replace(/,/g, '');
    else if (s.includes(',') && !s.includes('.')) s = s.replace(',', '.');
    return parseFloat(s);
  };
  const get = (cells, i) => i < 0 ? NaN : cleanNum(cells[i]);

  const strikes = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = splitCSV(lines[i]);
    const strike = get(cells, idx.strike);
    if (!isFinite(strike)) continue;
    strikes.push({
      strike,
      callBid: get(cells, idx.callBid),  callAsk: get(cells, idx.callAsk),
      callMid: get(cells, idx.callMid),  callLast: get(cells, idx.callLast),
      callIV:  get(cells, idx.callIV),   callVol: get(cells, idx.callVol),
      callOI:  get(cells, idx.callOI),   callDelta: get(cells, idx.callDelta),
      putBid:  get(cells, idx.putBid),   putAsk:  get(cells, idx.putAsk),
      putMid:  get(cells, idx.putMid),   putLast: get(cells, idx.putLast),
      putIV:   get(cells, idx.putIV),    putVol:  get(cells, idx.putVol),
      putOI:   get(cells, idx.putOI),    putDelta: get(cells, idx.putDelta),
    });
  }
  if (strikes.length === 0) throw new Error('No se encontraron filas válidas con strike numérico.');
  strikes.sort((a, b) => a.strike - b.strike);

  // Detected columns (for showing the user what we found)
  const detected = {};
  for (const k of Object.keys(idx)) if (idx[k] >= 0) detected[k] = headers[idx[k]];

  return { strikes, detectedHeaders: headers, detectedFields: detected };
}

// Find the strike closest to a target (for ATM detection).
function strikeClosestTo(strikes, target) {
  let best = null, bestDiff = Infinity;
  for (const s of strikes) {
    const d = Math.abs(s.strike - target);
    if (d < bestDiff) { best = s; bestDiff = d; }
  }
  return best;
}

let _pendingChain = null;

function handleChainUpload(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = parseChainCSV(reader.result);
      _pendingChain = parsed;
      renderChainPreview(parsed, file.name);
    } catch (err) {
      alert('Error al parsear el CSV:\n\n' + err.message);
      console.error('[Chain CSV]', err);
    }
  };
  reader.readAsText(file);
}

function renderChainPreview(parsed, filename) {
  const preview = document.getElementById('chainPreview');
  preview.style.display = '';

  // Try to extract date from filename (formats: 2026-05-06 or 06-05-2026 or 06_05_2026 etc.)
  let detectedDate = '';
  const dateMatch = filename && filename.match(/(\d{4})[-_](\d{2})[-_](\d{2})|(\d{2})[-_](\d{2})[-_](\d{4})/);
  if (dateMatch) {
    if (dateMatch[1]) detectedDate = `${dateMatch[1]}-${dateMatch[2]}-${dateMatch[3]}`;
    else              detectedDate = `${dateMatch[6]}-${dateMatch[5]}-${dateMatch[4]}`;
  }
  if (!detectedDate) detectedDate = new Date().toISOString().slice(0, 10);

  // Try to detect spot — use median strike as a rough fallback
  const middleIdx = Math.floor(parsed.strikes.length / 2);
  const detectedSpot = parsed.strikes[middleIdx]?.strike || '';

  // Detected fields summary
  const fieldList = Object.entries(parsed.detectedFields)
    .map(([k, h]) => `<span style="background:var(--blue-50);border:1px solid var(--blue-200);padding:1px 6px;border-radius:3px;margin:0 3px 3px 0;display:inline-block;font-size:10px"><b>${k}</b>: ${h}</span>`)
    .join('');

  const sample = parsed.strikes.slice(0, 5).concat(parsed.strikes.slice(-5));
  const sampleRows = sample.map(s => `
    <tr>
      <td>${s.strike}</td>
      <td class="call-cell">${isFinite(s.callBid) ? s.callBid.toFixed(2) : '—'}</td>
      <td class="call-cell">${isFinite(s.callAsk) ? s.callAsk.toFixed(2) : '—'}</td>
      <td class="put-cell">${isFinite(s.putBid) ? s.putBid.toFixed(2) : '—'}</td>
      <td class="put-cell">${isFinite(s.putAsk) ? s.putAsk.toFixed(2) : '—'}</td>
    </tr>`).join('');

  preview.innerHTML = `
    <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px 14px;margin-bottom:12px">
      <div style="font-weight:600;margin-bottom:6px;color:var(--navy-700)">
        ✓ ${parsed.strikes.length} strikes detectados
      </div>
      <div style="font-size:11px">${fieldList}</div>
    </div>

    <div class="chain-preview-grid">
      <div>
        <label>Fecha de la cadena</label>
        <input type="date" id="chainPreviewDate" value="${detectedDate}">
      </div>
      <div>
        <label>DTE</label>
        <select id="chainPreviewDTE">
          <option value="0DTE">0DTE (mismo día)</option>
          <option value="1DTE">1DTE (siguiente sesión)</option>
          <option value="2DTE">2DTE</option>
          <option value="weekly">Weekly</option>
          <option value="monthly">Monthly</option>
        </select>
      </div>
      <div>
        <label>Spot SPX</label>
        <input type="number" id="chainPreviewSpot" step="0.01" value="${detectedSpot}">
      </div>
    </div>

    <details style="margin-bottom:10px">
      <summary style="cursor:pointer;font-size:12px;color:var(--ink-soft)">Ver muestra (primeros y últimos 5 strikes)</summary>
      <table class="chain-table" style="margin-top:8px">
        <thead><tr>
          <th>Strike</th>
          <th>Call Bid</th><th>Call Ask</th>
          <th>Put Bid</th><th>Put Ask</th>
        </tr></thead>
        <tbody>${sampleRows}</tbody>
      </table>
    </details>

    <div class="chain-preview-actions">
      <button class="chain-save-btn"  id="chainSaveBtn">💾 Guardar cadena</button>
      <button class="chain-cancel-btn" id="chainCancelBtn">Cancelar</button>
    </div>
  `;

  document.getElementById('chainSaveBtn').addEventListener('click', confirmSaveChain);
  document.getElementById('chainCancelBtn').addEventListener('click', () => {
    preview.style.display = 'none';
    _pendingChain = null;
    document.getElementById('chainCSV').value = '';
  });
}

function confirmSaveChain() {
  if (!_pendingChain) return;
  const date = document.getElementById('chainPreviewDate').value;
  const dte  = document.getElementById('chainPreviewDTE').value;
  const spot = parseFloat(document.getElementById('chainPreviewSpot').value);
  if (!date) { alert('Falta la fecha.'); return; }

  const chains = loadChains();
  if (chains[date] && !confirm(`Ya existe una cadena guardada para ${date}. ¿Sobrescribir?`)) return;

  chains[date] = {
    date, dte, spot: isFinite(spot) ? spot : null,
    capturedAt: new Date().toISOString(),
    strikes: _pendingChain.strikes,
    detectedFields: _pendingChain.detectedFields,
  };
  saveChains(chains);
  _pendingChain = null;
  document.getElementById('chainPreview').style.display = 'none';
  document.getElementById('chainCSV').value = '';
  renderChainsList();
}

function renderChainsList() {
  const chains = loadChains();
  const dates = Object.keys(chains).sort().reverse();
  const container = document.getElementById('chainSavedList');
  if (dates.length === 0) {
    container.innerHTML = `<div style="font-size:12px;color:var(--ink-soft);font-style:italic">No hay cadenas guardadas todavía.</div>`;
    return;
  }
  const rows = dates.map(d => {
    const c = chains[d];
    return `<tr>
      <td><b>${d}</b></td>
      <td>${c.dte || '—'}</td>
      <td>${c.spot ? c.spot.toFixed(2) : '—'}</td>
      <td>${c.strikes.length}</td>
      <td>
        <button class="view-btn" data-chain-date="${d}">👁 Ver</button>
        <button class="del-btn" data-chain-date="${d}">🗑</button>
      </td>
    </tr>`;
  }).join('');
  container.innerHTML = `
    <div style="font-size:11px;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px">
      Cadenas guardadas (${dates.length})
    </div>
    <table class="chain-list-table">
      <thead><tr>
        <th>Fecha</th><th>DTE</th><th>Spot</th><th>Strikes</th><th></th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function viewChain(date) {
  const c = loadChains()[date];
  if (!c) return;
  // Build a full table view in the preview area
  const preview = document.getElementById('chainPreview');
  preview.style.display = '';
  const spot = c.spot;
  const rows = c.strikes.map(s => {
    const isATM = spot && Math.abs(s.strike - spot) < 5;
    const fmt = v => isFinite(v) ? v.toFixed(2) : '—';
    const fmtIV = v => isFinite(v) ? (v * 100 < 100 ? (v * 100).toFixed(1) + '%' : v.toFixed(1) + '%') : '—';
    return `<tr class="${isATM ? 'atm' : ''}">
      <td>${s.strike}</td>
      <td class="call-cell">${fmt(s.callBid)}</td>
      <td class="call-cell">${fmt(s.callAsk)}</td>
      <td class="call-cell">${fmtIV(s.callIV)}</td>
      <td class="put-cell">${fmt(s.putBid)}</td>
      <td class="put-cell">${fmt(s.putAsk)}</td>
      <td class="put-cell">${fmtIV(s.putIV)}</td>
    </tr>`;
  }).join('');
  preview.innerHTML = `
    <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px 14px;margin-bottom:12px">
      <b>Cadena ${c.date}</b> · DTE: ${c.dte || '—'} · Spot: ${c.spot ? c.spot.toFixed(2) : '—'} · ${c.strikes.length} strikes
    </div>
    <table class="chain-table">
      <thead><tr>
        <th>Strike</th>
        <th>Call Bid</th><th>Call Ask</th><th>Call IV</th>
        <th>Put Bid</th><th>Put Ask</th><th>Put IV</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="chain-preview-actions">
      <button class="chain-cancel-btn" id="chainCancelBtn">Cerrar vista</button>
    </div>`;
  document.getElementById('chainCancelBtn').addEventListener('click', () => {
    preview.style.display = 'none';
  });
}

function deleteChain(date) {
  if (!confirm(`¿Eliminar la cadena del ${date}?`)) return;
  const chains = loadChains();
  delete chains[date];
  saveChains(chains);
  renderChainsList();
}

// ==========================================================================
// FAVORITES — save/load/apply full configurations with summary stats
// ==========================================================================
const FAVORITES_KEY = 'spx-vix-favorites-v1';

function loadFavorites() {
  try { return JSON.parse(localStorage.getItem(FAVORITES_KEY) || '[]'); }
  catch (_) { return []; }
}
function saveFavorites(favs) { localStorage.setItem(FAVORITES_KEY, JSON.stringify(favs)); }

// Capture EVERYTHING editable from the page into one config object.
function captureCurrentConfig() {
  return {
    compressionParams: { ...compressionParams },
    compressionVars: compressionVars.map(v => ({ id: v.id, active: v.active, weight: v.weight })),
    regimes: readRegimes(),
    initialCapital: parseFloat(document.getElementById('initialCapital').value) || 1000,
    lookback:       parseInt(document.getElementById('lookback').value, 10) || 90,
    displayWindow:  parseInt(document.getElementById('window').value, 10)   || 30,
  };
}

// Compute summary stats at save time for the current bands (full history).
function computeFavoriteStats(rows, bands) {
  let wins = 0, losses = 0, noTrade = 0, totalPnL = 0;
  // Last 3 months distance averages (~63 trading days)
  const cutoff = Math.max(1, rows.length - 63);
  let distSumCall = 0, distSumPut = 0, distCount = 0;

  for (let i = 1; i < rows.length; i++) {
    const b = bands[i];
    const r = rows[i];
    const prev = rows[i - 1];
    if (!b) continue;
    if (!b.trade) { noTrade++; continue; }

    const callStrike = Math.ceil(b.upper / 5) * 5;
    const putStrike  = Math.floor(b.lower / 5) * 5;

    if (i >= cutoff) {
      distSumCall += (callStrike - prev.close) / prev.close * 100;
      distSumPut  += (prev.close - putStrike)  / prev.close * 100;
      distCount++;
    }

    if (r.close >= b.upper)      { losses++; totalPnL += LOSS_PNL; }
    else if (r.close <= b.lower) { losses++; totalPnL += LOSS_PNL; }
    else                         { wins++;   totalPnL += WIN_PNL;  }
  }

  const total = wins + losses;
  return {
    winRate: total > 0 ? wins / total * 100 : 0,
    wins, losses, noTrade,
    avgDistCallPct: distCount > 0 ? distSumCall / distCount : 0,
    avgDistPutPct:  distCount > 0 ? distSumPut  / distCount : 0,
    totalPnL,
  };
}

function saveCurrentAsFavorite(name) {
  if (!currentRows) {
    alert('No hay datos cargados — carga el CSV primero.');
    return;
  }
  const lookback = parseInt(document.getElementById('lookback').value, 10);
  const regimes = readRegimes();
  const { bands } = computeBands(currentRows, regimes, lookback);
  const stats = computeFavoriteStats(currentRows, bands);

  const fav = {
    id: 'fav_' + Date.now(),
    name: (name || '').trim() || `Config ${new Date().toLocaleString('es-ES', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}`,
    date: new Date().toISOString().slice(0, 10),
    config: captureCurrentConfig(),
    stats,
  };
  const favs = loadFavorites();
  favs.unshift(fav); // newest first
  saveFavorites(favs);
  document.getElementById('newFavName').value = '';
  renderFavoritesList();
  updateFavoritesCardCount();
}

function deleteFavorite(id) {
  if (!confirm('¿Eliminar esta configuración guardada?')) return;
  saveFavorites(loadFavorites().filter(f => f.id !== id));
  renderFavoritesList();
  updateFavoritesCardCount();
}

function renameFavorite(id, newName) {
  const favs = loadFavorites();
  const fav = favs.find(f => f.id === id);
  if (fav) {
    fav.name = newName.trim() || fav.name;
    saveFavorites(favs);
  }
}

function applyFavorite(id) {
  const fav = loadFavorites().find(f => f.id === id);
  if (!fav) return;
  const c = fav.config;

  // Compression params (state + DOM inputs)
  Object.assign(compressionParams, c.compressionParams);
  document.getElementById('cParamLambda').value = c.compressionParams.lambda;
  document.getElementById('cParamCmin').value   = c.compressionParams.Cmin;
  document.getElementById('cParamCmax').value   = c.compressionParams.Cmax;
  document.getElementById('cParamVref').value   = c.compressionParams.Vref;
  document.getElementById('cParamShift').value  = c.compressionParams.shiftFactor || 0.25;

  // Compression variables (active flags + weights)
  c.compressionVars.forEach((v, i) => {
    if (compressionVars[i]) {
      compressionVars[i].active = v.active;
      compressionVars[i].weight = v.weight;
    }
  });

  // Regime parameters → re-render the table with saved values, then write to DOM inputs
  c.regimes.forEach((reg, i) => {
    const set = (f, val) => {
      const inp = document.querySelector(`#regimeTable input[data-i="${i}"][data-f="${f}"]`);
      if (!inp) return;
      if (inp.type === 'checkbox') inp.checked = !!val;
      else inp.value = val;
    };
    set('maxVix',    reg.maxVix);
    set('upperMult', reg.upperMult);
    set('lowerMult', reg.lowerMult);
    set('k',         reg.k);
    set('trade',     reg.trade);
  });

  // Other params
  document.getElementById('initialCapital').value = c.initialCapital;
  document.getElementById('lookback').value       = c.lookback;
  document.getElementById('window').value         = c.displayWindow;

  closeFavoritesModal();
  renderCompressionPanel();
  recalc();
}

function renderFavoritesList() {
  const container = document.getElementById('favoritesList');
  const favs = loadFavorites();
  if (favs.length === 0) {
    container.innerHTML = `
      <div style="text-align:center;padding:40px;color:var(--ink-soft)">
        <div style="font-size:36px">⭐</div>
        <div style="margin-top:8px;font-size:14px">Aún no has guardado ninguna configuración.</div>
        <div style="font-size:11px;margin-top:6px">Usa el formulario de arriba para guardar la configuración actual.</div>
      </div>`;
    return;
  }
  container.innerHTML = favs.map(renderFavoriteCard).join('');
}

function renderFavoriteCard(fav) {
  const fmt$ = v => (v >= 0 ? '+$' : '−$') + Math.abs(v).toFixed(0);
  const wrColor = fav.stats.winRate >= 80 ? 'var(--good)' : 'var(--bad)';
  const pnlColor = fav.stats.totalPnL >= 0 ? 'var(--good)' : 'var(--bad)';
  return `
    <div class="favorite-card" data-id="${fav.id}">
      <div class="fav-header">
        <span class="fav-star">⭐</span>
        <input type="text" class="fav-name" value="${fav.name.replace(/"/g, '&quot;')}" data-id="${fav.id}">
        <span class="fav-date">${fav.date}</span>
        <button class="fav-apply"  data-id="${fav.id}">✓ Aplicar</button>
        <button class="fav-delete" data-id="${fav.id}">🗑</button>
      </div>
      <div class="fav-stats">
        <div class="fav-stat">
          <div class="fav-stat-lbl">Win rate</div>
          <div class="fav-stat-val" style="color:${wrColor}">${fav.stats.winRate.toFixed(1)}%</div>
        </div>
        <div class="fav-stat">
          <div class="fav-stat-lbl">Operaciones</div>
          <div class="fav-stat-val" style="font-size:12px">
            <span style="color:var(--good)">${fav.stats.wins} W</span> ·
            <span style="color:var(--bad)">${fav.stats.losses} L</span> ·
            <span style="color:var(--ink-soft)">${fav.stats.noTrade} NT</span>
          </div>
        </div>
        <div class="fav-stat">
          <div class="fav-stat-lbl">Dist media (3m)</div>
          <div class="fav-stat-val" style="font-size:12px">
            <span style="color:var(--good)">+${fav.stats.avgDistCallPct.toFixed(2)}%</span> /
            <span style="color:var(--bad)">−${fav.stats.avgDistPutPct.toFixed(2)}%</span>
          </div>
        </div>
        <div class="fav-stat">
          <div class="fav-stat-lbl">P&amp;L total</div>
          <div class="fav-stat-val" style="color:${pnlColor}">${fmt$(fav.stats.totalPnL)}</div>
        </div>
      </div>
      <button class="fav-expand" data-id="${fav.id}">? Ver todas las variables</button>
      <div class="fav-details" id="fav-details-${fav.id}" style="display:none">${renderFavoriteDetails(fav)}</div>
    </div>`;
}

function renderFavoriteDetails(fav) {
  const cp = fav.config.compressionParams;
  const vars = fav.config.compressionVars;
  const regs = fav.config.regimes;
  const activeVars = vars.filter(v => v.active);
  return `
    <div class="fav-detail-block">
      <div class="fav-detail-title">Compression params</div>
      λ = ${cp.lambda} · C<sub>min</sub> = ${cp.Cmin} · C<sub>max</sub> = ${cp.Cmax} · V<sub>ref</sub> = ${cp.Vref} · Δ<sub>trend</sub> = ${cp.shiftFactor || 0.25}
    </div>
    <div class="fav-detail-block">
      <div class="fav-detail-title">Variables activas (${activeVars.length}/${vars.length})</div>
      ${activeVars.length === 0
        ? '<i style="color:var(--ink-soft)">Ninguna</i>'
        : activeVars.map(v => `<b>${v.id}</b>=${v.weight}`).join(' · ')}
      ${vars.filter(v => !v.active).length > 0
        ? `<div style="margin-top:3px;font-size:10px;color:var(--ink-soft)">Inactivas: ${vars.filter(v => !v.active).map(v => v.id).join(', ')}</div>`
        : ''}
    </div>
    <div class="fav-detail-block">
      <div class="fav-detail-title">Regímenes</div>
      <table class="fav-regime-table">
        <thead><tr><th>VIX≤</th><th>k</th><th>Up</th><th>Low</th><th>Trade</th></tr></thead>
        <tbody>${regs.map(r => `<tr><td>${r.maxVix}</td><td>${r.k}</td><td>${r.upperMult}</td><td>${r.lowerMult}</td><td>${r.trade ? '✓' : '✗'}</td></tr>`).join('')}</tbody>
      </table>
    </div>
    <div class="fav-detail-block">
      <div class="fav-detail-title">Otros</div>
      Capital inicial = $${fav.config.initialCapital} · Lookback σ = ${fav.config.lookback}d · Display window = ${fav.config.displayWindow}d
    </div>`;
}

function updateFavoritesCardCount() {
  const n = loadFavorites().length;
  const el = document.getElementById('favoritesCount');
  if (el) el.textContent = n === 0 ? 'Sin guardar' : `${n} guardada${n === 1 ? '' : 's'}`;
}

function openFavoritesModal()  {
  document.getElementById('favoritesModal').style.display = 'flex';
  renderFavoritesList();
}
function closeFavoritesModal() {
  document.getElementById('favoritesModal').style.display = 'none';
}

// ==========================================================================
// VOL ANALYST AGENT
// ==========================================================================
const EVENTS_UPCOMING_KEY  = 'spx-vix-events-upcoming-v1';
const EVENTS_HISTORICAL_KEY = 'spx-vix-events-historical-v1';
const EVENT_DISMISSED_KEY  = 'spx-vix-event-dismissed-v1';

function loadEvents(key)  {
  try { return JSON.parse(localStorage.getItem(key) || '[]'); }
  catch (_) { return []; }
}
function saveEvents(key, events) { localStorage.setItem(key, JSON.stringify(events)); }

// CSV parser for events: date,event,category,importance
function parseEventsCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const idx = {
    date: headers.indexOf('date'),
    event: headers.indexOf('event'),
    category: headers.indexOf('category'),
    importance: headers.indexOf('importance'),
  };
  const events = [];
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i].split(',').map(s => s.trim());
    if (!c[idx.date]) continue;
    events.push({
      date: c[idx.date],
      event: c[idx.event] || '',
      category: c[idx.category] || '',
      importance: (c[idx.importance] || 'medium').toLowerCase(),
    });
  }
  events.sort((a, b) => a.date.localeCompare(b.date));
  return events;
}

// ---- Tab 1: VIX chart for analyst (full history) ------------------------
function drawVixAnalystChart() {
  if (!currentRows) return;
  const lookback = parseInt(document.getElementById('lookback').value, 10);
  const regimes = readRegimes();
  const { bands } = computeBands(currentRows, regimes, lookback);

  const dates = currentRows.map(r => r.date);
  const vix   = currentRows.map(r => r.vix);

  const lossX = [], lossY = [];
  for (let i = 1; i < currentRows.length; i++) {
    const b = bands[i];
    if (!b || !b.trade) continue;
    const r = currentRows[i];
    if (r.close >= b.upper || r.close <= b.lower) {
      lossX.push(r.date); lossY.push(r.vix);
    }
  }

  const vMax = Math.max(...vix, ...regimes.map(r => r.maxVix < 998 ? r.maxVix : 0)) * 1.05;
  const shapes = [];
  regimes.forEach(reg => {
    if (reg.maxVix >= 998) return;
    shapes.push({
      type: 'line', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: reg.maxVix, y1: reg.maxVix,
      line: { color: '#c9a227', width: 1, dash: 'dot' },
    });
  });
  for (let i = 0; i < regimes.length; i++) {
    if (regimes[i].trade) continue;
    const yLo = i > 0 ? regimes[i - 1].maxVix : 0;
    const yHi = regimes[i].maxVix < 998 ? regimes[i].maxVix : vMax;
    shapes.push({
      type: 'rect', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: yLo, y1: yHi,
      fillcolor: 'rgba(201, 162, 39, 0.13)',
      line: { width: 0 }, layer: 'below',
    });
  }

  const annotations = regimes.filter(r => r.maxVix < 998).map(reg => ({
    xref: 'paper', yref: 'y', x: 1, y: reg.maxVix,
    xanchor: 'right', yanchor: 'bottom',
    text: ` ≤ ${reg.maxVix} `,
    font: { size: 10, color: '#8a6d10' },
    bgcolor: 'rgba(255,255,255,0.7)', showarrow: false,
  }));

  const traces = [
    { x: dates, y: vix, mode: 'lines', name: 'VIX',
      line: { color: '#143a64', width: 1.8 } },
    { x: lossX, y: lossY, mode: 'markers', name: 'Pérdidas',
      marker: { color: '#b73232', size: 10, symbol: 'x', line: { width: 1, color: '#fff' } } },
  ];
  safePlotly('vixAnalystChart', traces, {
    margin: { t: 20, r: 20, b: 50, l: 60 },
    paper_bgcolor: '#ffffff', plot_bgcolor: '#f3f8fc',
    font: { family: 'Segoe UI, sans-serif', color: '#0c1f33' },
    xaxis: { title: 'Date', type: 'date', gridcolor: '#dce7f1' },
    yaxis: { title: 'VIX', gridcolor: '#dce7f1', range: [Math.max(0, Math.min(...vix) * 0.9), vMax] },
    legend: { orientation: 'h', y: -0.18 },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#0a2540', font: { color: '#fff' } },
    shapes, annotations,
  }, { responsive: true });
}

// ---- Tab 2: Upcoming events list + main-page banner --------------------
function renderUpcomingEvents() {
  const events = loadEvents(EVENTS_UPCOMING_KEY);
  const today = new Date(); today.setHours(0,0,0,0);
  const future = events.filter(e => new Date(e.date) >= today);

  const cls = imp => imp === 'high' ? 'imp-high' : imp === 'medium' ? 'imp-medium' : 'imp-low';
  const isImminent = (date) => {
    const d = new Date(date); d.setHours(0,0,0,0);
    const diff = (d - today) / 86400000;
    return diff >= 0 && diff <= 1;
  };

  const rowsHtml = future.length === 0
    ? '<tr><td colspan="4" style="text-align:center;color:var(--ink-soft);padding:20px">No hay eventos próximos cargados.</td></tr>'
    : future.slice(0, 50).map(e => `
        <tr class="${isImminent(e.date) ? 'imminent' : ''}">
          <td><b>${e.date}</b></td>
          <td>${e.event}</td>
          <td>${e.category}</td>
          <td class="${cls(e.importance)}">${e.importance.toUpperCase()}</td>
        </tr>
      `).join('');

  document.getElementById('upcomingEventsList').innerHTML = `
    <table class="events-table">
      <thead><tr><th>Date</th><th>Event</th><th>Category</th><th>Importance</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>
    <div style="font-size:11px;color:var(--ink-soft);margin-top:8px">
      ${future.length} eventos cargados. Las filas resaltadas son hoy o mañana.
    </div>
  `;

  // Update tab badge
  const imminent = future.filter(e => isImminent(e.date));
  const badge = document.getElementById('upcomingBadge');
  if (imminent.length > 0) {
    badge.style.display = 'inline-block';
    badge.textContent = imminent.length;
  } else {
    badge.style.display = 'none';
  }
}

function checkEventBanner() {
  const events = loadEvents(EVENTS_UPCOMING_KEY);
  if (!events.length) { document.getElementById('eventBanner').style.display = 'none'; return; }

  const today = new Date(); today.setHours(0,0,0,0);
  const tomorrow = new Date(today); tomorrow.setDate(tomorrow.getDate() + 1);
  const dismissed = new Set(JSON.parse(localStorage.getItem(EVENT_DISMISSED_KEY) || '[]'));

  const todayISO = today.toISOString().slice(0, 10);
  const tomorrowISO = tomorrow.toISOString().slice(0, 10);
  const imminent = events.filter(e =>
    (e.date === todayISO || e.date === tomorrowISO) &&
    e.importance === 'high' &&
    !dismissed.has(`${e.date}|${e.event}`)
  );

  const banner = document.getElementById('eventBanner');
  if (imminent.length === 0) { banner.style.display = 'none'; return; }

  const list = imminent.map(e => `<b>${e.date === todayISO ? 'HOY' : 'MAÑANA'}</b>: ${e.event} (${e.category})`).join(' &nbsp;·&nbsp; ');
  banner.style.display = 'flex';
  banner.className = 'event-banner';
  banner.innerHTML = `
    <div class="icon">?</div>
    <div class="body">
      <div class="title">Evento macro de alta importancia</div>
      <div class="desc">${list} — Esperar volatilidad superior a la media.</div>
    </div>
    <button class="dismiss" type="button">Descartar</button>
  `;
  banner.querySelector('.dismiss').onclick = () => {
    imminent.forEach(e => dismissed.add(`${e.date}|${e.event}`));
    localStorage.setItem(EVENT_DISMISSED_KEY, JSON.stringify([...dismissed]));
    banner.style.display = 'none';
  };
}

// ---- Tab 3: Historical correlation (loss days vs macro events) ---------
function renderHistoricalCorrelation() {
  const events = loadEvents(EVENTS_HISTORICAL_KEY);
  const container = document.getElementById('historicalCorrelation');

  if (!events.length) {
    container.innerHTML = `<div style="color:var(--ink-soft);text-align:center;padding:20px">
      Carga el CSV histórico de eventos macro para analizar la correlación con tus días de pérdida.</div>`;
    return;
  }
  if (!currentRows) return;

  const lookback = parseInt(document.getElementById('lookback').value, 10);
  const initialCapital = parseFloat(document.getElementById('initialCapital').value) || 1000;
  const regimes = readRegimes();
  const { bands } = computeBands(currentRows, regimes, lookback);
  const pnl = computePnL(currentRows, bands, initialCapital);
  const losses = pnl.trades.filter(t => t.status === 'loss-up' || t.status === 'loss-dn');

  const eventByDate = new Map();
  events.forEach(e => eventByDate.set(e.date, e));

  const shiftDays = (iso, n) => { const d = new Date(iso); d.setDate(d.getDate() + n); return d.toISOString().slice(0, 10); };

  // For each loss, look ±1 day
  let withEvent = 0;
  const eventCounter = new Map();
  losses.forEach(loss => {
    const candidates = [shiftDays(loss.date, -1), loss.date, shiftDays(loss.date, 1)];
    let nearest = null;
    for (const dt of candidates) {
      if (eventByDate.has(dt)) { nearest = eventByDate.get(dt); break; }
    }
    if (nearest) {
      withEvent++;
      const key = `${nearest.event} (${nearest.category})`;
      eventCounter.set(key, (eventCounter.get(key) || 0) + 1);
    }
  });

  const pct = losses.length ? (withEvent / losses.length * 100).toFixed(1) : '—';
  const topEvents = [...eventCounter.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  const eventsRows = topEvents.length === 0
    ? '<tr><td colspan="2" style="color:var(--ink-soft);text-align:center">Ningún evento del histórico coincide con días de pérdida.</td></tr>'
    : topEvents.map(([name, n]) => `<tr><td>${name}</td><td><b>${n}</b></td></tr>`).join('');

  container.innerHTML = `
    <div class="corr-summary">
      <div class="corr-card">
        <div class="lbl">Pérdidas totales</div>
        <div class="val bad">${losses.length}</div>
      </div>
      <div class="corr-card">
        <div class="lbl">Coincidieron con evento (±1 día)</div>
        <div class="val">${withEvent}</div>
      </div>
      <div class="corr-card">
        <div class="lbl">% pérdidas con evento</div>
        <div class="val">${pct}%</div>
      </div>
      <div class="corr-card">
        <div class="lbl">Eventos cargados</div>
        <div class="val">${events.length}</div>
      </div>
    </div>
    <div style="margin-top:14px">
      <div style="font-size:11px;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px">
        Eventos que más coinciden con pérdidas
      </div>
      <table class="events-table">
        <thead><tr><th>Evento</th><th>Veces que coincidió</th></tr></thead>
        <tbody>${eventsRows}</tbody>
      </table>
    </div>
    <div style="font-size:11px;color:var(--ink-soft);margin-top:8px">
      Ventana de correlación: el día del evento o ±1 día respecto al día de la pérdida.
    </div>
  `;
}

// ---- Tab 4: MIX iterative optimizer (Quant ↔ Vol Analyst) -------------
function rebandWithNewThresholds(rows, baseBands, regimes) {
  return baseBands.map((b, i) => {
    if (!b) return null;
    const prev = rows[i - 1];
    const newReg = regimeFor(prev.vix, regimes);
    const cf = compressionFactor(prev);
    return {
      ...b,
      upper: prev.close * (1 + cf.Ccall * newReg.k * newReg.upperMult * b.sigUp),
      lower: prev.close * (1 - cf.Cput  * newReg.k * newReg.lowerMult * b.sigDn),
      regime: newReg, trade: newReg.trade,
      Cbase: cf.base, Ccall: cf.Ccall, Cput: cf.Cput, trend: cf.trend,
    };
  });
}

function scorePnLOnWindow(rows, bands, windowDays) {
  const start = Math.max(1, rows.length - windowDays);
  let score = 0;
  for (let i = start; i < rows.length; i++) {
    const b = bands[i];
    if (!b || !b.trade) continue;
    if (rows[i].close >= b.upper || rows[i].close <= b.lower) score += LOSS_PNL;
    else score += WIN_PNL;
  }
  return score;
}

// Inline scorer — computes upper/lower on the fly with no allocations,
// and inlines regimeFor (called millions of times during threshold search).
function scoreRegimesOnWindow(rows, baseBands, regimes, windowDays) {
  const start = Math.max(1, rows.length - windowDays);
  const N = rows.length;
  const regN = regimes.length;
  let score = 0;
  for (let i = start; i < N; i++) {
    const b = baseBands[i];
    if (!b) continue;
    const prev = rows[i - 1];
    const vix = prev.vix;

    let reg = regimes[regN - 1];
    for (let j = 0; j < regN; j++) {
      if (vix <= regimes[j].maxVix) { reg = regimes[j]; break; }
    }
    if (!isDayAccepted(rows[i], reg)) continue;

    const cf = compressionFactor(prev);
    const upper = prev.close * (1 + cf.Ccall * reg.k * reg.upperMult * b.sigUp);
    const lower = prev.close * (1 - cf.Cput  * reg.k * reg.lowerMult * b.sigDn);
    if (rows[i].close >= upper || rows[i].close <= lower) score += LOSS_PNL;
    else score += WIN_PNL;
  }
  return score;
}

// Re-bucket days by regime using the CURRENT regimes (handles threshold changes).
function bucketDaysForRegimes(rows, baseBands, regimes, windowDays) {
  const start = Math.max(1, rows.length - windowDays);
  const buckets = new Map();
  for (let i = start; i < rows.length; i++) {
    const b = baseBands[i];
    if (!b) continue;
    const reg = regimeFor(rows[i - 1].vix, regimes);
    if (!isDayAccepted(rows[i], reg)) continue;
    if (!buckets.has(reg)) buckets.set(reg, []);
    buckets.get(reg).push({ row: rows[i], prev: rows[i - 1], sigUp: b.sigUp, sigDn: b.sigDn });
  }
  return buckets;
}

// MIX-aware mult optimizer: re-buckets each call so it works after threshold changes.
// Uses indexed loops over the pre-computed grid (no generators).
function optimizeMultsForMix(rows, baseBands, regimes, windowDays) {
  const buckets = bucketDaysForRegimes(rows, baseBands, regimes, windowDays);
  const newRegimes = regimes.map(r => ({ ...r }));
  const G = OPT_GRID_N;
  for (let i = 0; i < regimes.length; i++) {
    const reg = regimes[i];
    if (!reg.trade) continue;
    const days = buckets.get(reg) || [];
    if (days.length === 0) continue;
    let bestS = scoreParams(days, reg.upperMult, reg.lowerMult, reg.k);
    let bestU = reg.upperMult, bestL = reg.lowerMult, bestK = reg.k;
    for (let ui = 0; ui < G; ui++) {
      const u = OPT_GRID[ui];
      for (let li = 0; li < G; li++) {
        const l = OPT_GRID[li];
        for (let ki = 0; ki < G; ki++) {
          const k = OPT_GRID[ki];
          const s = scoreParams(days, u, l, k);
          if (s > bestS) { bestS = s; bestU = u; bestL = l; bestK = k; }
        }
      }
    }
    newRegimes[i].upperMult = bestU;
    newRegimes[i].lowerMult = bestL;
    newRegimes[i].k = bestK;
  }
  return newRegimes;
}

// Optimize VIX thresholds via COORDINATE DESCENT:
//   For each threshold (T1,T2,T3,T4) in turn, find the best value while
//   holding the others fixed. Repeat until a full pass produces no change.
// ~150 evaluations vs ~3000 for full grid — converges in milliseconds.
function optimizeThresholds(rows, baseBands, regimes, windowDays) {
  const innerRegimes = regimes.filter(r => r.maxVix < 998);
  const N = innerRegimes.length;
  if (N === 0) return { newRegimes: regimes.map(r => ({ ...r })), score: 0 };

  const STEP = 0.5;
  const SPAN = 2.5;

  const work = regimes.map(r => ({ ...r }));
  const current = innerRegimes.map(r => r.maxVix);
  let bestScore = scoreRegimesOnWindow(rows, baseBands, work, windowDays);

  const MAX_ROUNDS = 6;
  for (let round = 0; round < MAX_ROUNDS; round++) {
    let changed = false;
    for (let idx = 0; idx < N; idx++) {
      const lower = idx > 0     ? current[idx - 1] + 0.1 : -Infinity;
      const upper = idx < N - 1 ? current[idx + 1] - 0.1 :  Infinity;
      const original = innerRegimes[idx].maxVix;
      let bestV = current[idx];

      for (let v = original - SPAN; v <= original + SPAN + 1e-9; v += STEP) {
        const vR = Math.round(v * 10) / 10;
        if (vR < lower || vR > upper) continue;
        work[idx].maxVix = vR;
        const score = scoreRegimesOnWindow(rows, baseBands, work, windowDays);
        if (score > bestScore) { bestScore = score; bestV = vR; }
      }

      if (Math.abs(bestV - current[idx]) > 1e-6) {
        current[idx] = bestV;
        changed = true;
      }
      work[idx].maxVix = current[idx]; // ensure committed
    }
    if (!changed) break;
  }

  const newRegimes = regimes.map((r, i) => i < N ? { ...r, maxVix: current[i] } : { ...r });
  return { newRegimes, score: bestScore };
}

// Iterative: alternate Quant (mults+k) ↔ VolAnalyst (thresholds) until no improvement.
// Uses inline scoring throughout — no allocations during the search.
async function runMixIteration(rows, baseRegimes, windowDays, maxIters, onProgress) {
  const lookback = parseInt(document.getElementById('lookback').value, 10);
  const baseBands = computeBands(rows, baseRegimes, lookback).bands;

  let regimes = baseRegimes.map(r => ({ ...r }));
  let lastScore = scoreRegimesOnWindow(rows, baseBands, regimes, windowDays);
  let actualIters = 0;

  for (let iter = 1; iter <= maxIters; iter++) {
    actualIters = iter;

    onProgress && onProgress({ stage: 'quant', iter });
    await new Promise(r => setTimeout(r, 0));
    regimes = optimizeMultsForMix(rows, baseBands, regimes, windowDays);

    onProgress && onProgress({ stage: 'vol', iter });
    await new Promise(r => setTimeout(r, 0));
    const volResult = optimizeThresholds(rows, baseBands, regimes, windowDays);
    regimes = volResult.newRegimes;

    if (volResult.score <= lastScore + 1e-6) break;
    lastScore = volResult.score;
  }

  // One-shot rebanding for the final equity comparison & PnL display
  const finalBands = rebandWithNewThresholds(rows, baseBands, regimes);
  return { finalRegimes: regimes, finalBands, finalScore: lastScore, iters: actualIters };
}

async function runMix() {
  if (!currentRows) return;
  const btn = document.getElementById('mixRunBtn');
  const status = document.getElementById('mixStatus');
  const bar = document.getElementById('mixProgressBar');
  const fill = document.getElementById('mixProgressFill');
  btn.disabled = true;
  bar.style.display = 'block';
  fill.style.width = '0%';
  status.style.color = '';

  try {
    const lookback = parseInt(document.getElementById('lookback').value, 10);
    const initialCapital = parseFloat(document.getElementById('initialCapital').value) || 1000;
    const baseRegimes = readRegimes();
    const { bands: baseBands } = computeBands(currentRows, baseRegimes, lookback);
    const basePnL = computePnL(currentRows, baseBands, initialCapital);

    const periods = [
      { key: '1m', label: 'Mejora 1 — 1 mes',   days: 21  },
      { key: '3m', label: 'Mejora 2 — 3 meses', days: 63  },
      { key: '6m', label: 'Mejora 3 — 6 meses', days: 126 },
    ];

    document.getElementById('mixResults').innerHTML = '';

    const MAX_ITERS = 8;
    const totalSlots = periods.length * MAX_ITERS * 2;
    let consumedSlots = 0;
    const setProgress = (pct, label) => {
      fill.style.width = Math.min(99, pct).toFixed(1) + '%';
      if (label) status.textContent = label;
    };

    const t0 = performance.now();
    const results = [];
    for (let pIdx = 0; pIdx < periods.length; pIdx++) {
      const p = periods[pIdx];
      const tWin = performance.now();
      const baseSlot = pIdx * MAX_ITERS * 2;
      const onProgress = ({ stage, iter }) => {
        consumedSlots = baseSlot + (iter - 1) * 2 + (stage === 'vol' ? 1 : 0) + 1;
        const pct = consumedSlots / totalSlots * 100;
        setProgress(pct, `[${pIdx + 1}/3 · ${p.label}] iter ${iter} · ${stage === 'quant' ? 'Quant (mults+k)' : 'Vol (umbrales)'}`);
      };
      const mix = await runMixIteration(currentRows, baseRegimes, p.days, MAX_ITERS, onProgress);
      const finalPnL = computePnL(currentRows, mix.finalBands, initialCapital);
      results.push({ period: p, mix, finalPnL });

      consumedSlots = (pIdx + 1) * MAX_ITERS * 2;
      setProgress(consumedSlots / totalSlots * 100,
        `${p.label} listo (${((performance.now() - tWin) / 1000).toFixed(1)}s, ${mix.iters} iter).`);

      renderMixCard(p, baseRegimes, mix, basePnL, finalPnL);
      await new Promise(r => setTimeout(r, 0));
    }

    renderMixSummary(basePnL, results);
    fill.style.width = '100%';

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent = `✓ Completado en ${elapsed}s · iteraciones (1m/3m/6m): ${results.map(r => r.mix.iters).join(' / ')}`;
    setTimeout(() => { bar.style.display = 'none'; fill.style.width = '0%'; }, 1800);
  } catch (err) {
    console.error('runMix error:', err);
    status.style.color = '#b73232';
    status.textContent = `✗ Error: ${err.message || err}`;
    fill.style.background = '#b73232';
  } finally {
    btn.disabled = false;
  }
}

function renderMixCard(period, baseRegimes, mix, basePnL, finalPnL) {
  const container = document.getElementById('mixResults');
  const card = document.createElement('div');
  card.className = 'mix-card';

  const trainStart = currentRows[Math.max(1, currentRows.length - period.days)].date;
  const inBase = inSampleStats(basePnL, trainStart);
  const inOpt  = inSampleStats(finalPnL, trainStart);
  const fmt$ = v => (v >= 0 ? '+$' : '−$') + Math.abs(v).toFixed(0);

  // Threshold changes
  const baseTh = baseRegimes.filter(r => r.maxVix < 998).map(r => r.maxVix);
  const optTh  = mix.finalRegimes.filter(r => r.maxVix < 998).map(r => r.maxVix);
  const thHtml = baseTh.map((t, i) => {
    const changed = Math.abs(t - optTh[i]) > 1e-6;
    return `<span class="${changed ? 'changed' : ''}">${t} → ${optTh[i]}</span>`;
  }).join(' &nbsp;|&nbsp; ');

  // Param changes per regime
  const paramRows = baseRegimes.map((r, i) => {
    if (!r.trade) return `<tr class="notrade"><td>${regimeLabel(baseRegimes, i)}</td><td colspan="3" style="font-style:italic">no-trade</td></tr>`;
    const n = mix.finalRegimes[i];
    const arrow = (a, b) => Math.abs(a - b) < 1e-6 ? '<span class="arrow-eq">=</span>' :
      b > a ? '<span class="arrow-up-bad">?</span>' : '<span class="arrow-down-good">?</span>';
    return `<tr>
      <td>${regimeLabel(baseRegimes, i)}</td>
      <td>${r.k.toFixed(2)} → <b>${n.k.toFixed(2)}</b> ${arrow(r.k, n.k)}</td>
      <td>${r.upperMult.toFixed(2)} → <b>${n.upperMult.toFixed(2)}</b> ${arrow(r.upperMult, n.upperMult)}</td>
      <td>${r.lowerMult.toFixed(2)} → <b>${n.lowerMult.toFixed(2)}</b> ${arrow(r.lowerMult, n.lowerMult)}</td>
    </tr>`;
  }).join('');

  const iters = mix.iters;
  const chartId = `mix-chart-${period.key}`;
  card.innerHTML = `
    <h4>${period.label}</h4>
    <div class="iter-info">
      <span>Convergió en <b>${iters} iteración${iters === 1 ? '' : 'es'}</b></span>
      <span>P&L in-sample: ${fmt$(inBase.pnl)} → <b>${fmt$(inOpt.pnl)}</b></span>
    </div>
    <div class="threshold-block">
      <div class="th-label">Umbrales VIX (base → optimizado)</div>
      <div class="th-vals">${thHtml}</div>
    </div>
    <table class="agent-params">
      <thead><tr><th>Régimen</th><th>k</th><th>Upper</th><th>Lower</th></tr></thead>
      <tbody>${paramRows}</tbody>
    </table>
    <div class="equity-cmp" id="${chartId}"></div>
    <button class="apply-btn" data-mix-period="${period.key}">Aplicar esta optimización</button>
  `;
  container.appendChild(card);

  // Cache for Apply button
  _mixCache.set(period.key, mix.finalRegimes);

  drawAgentEquity(chartId, basePnL, finalPnL, trainStart);
}

const _mixCache = new Map();

function applyMixProposal(periodKey) {
  const newRegimes = _mixCache.get(periodKey);
  if (!newRegimes) return;
  newRegimes.forEach((reg, i) => {
    const set = (f, v) => {
      const inp = document.querySelector(`#regimeTable input[data-i="${i}"][data-f="${f}"]`);
      if (inp) inp.value = v;
    };
    set('maxVix', reg.maxVix);
    set('upperMult', reg.upperMult);
    set('lowerMult', reg.lowerMult);
    set('k', reg.k);
  });
  closeAnalystModal();
  recalc();
}

function renderMixSummary(basePnL, results) {
  const container = document.getElementById('mixResults');
  const card = document.createElement('div');
  card.className = 'mix-card summary';

  const fmt$ = v => (v >= 0 ? '+$' : '−$') + Math.abs(v).toFixed(0);
  const ranked = results.map(r => ({
    key: r.period.key, label: r.period.label,
    finalEquity: r.finalPnL.finalEquity,
    delta: r.finalPnL.finalEquity - basePnL.finalEquity,
  })).sort((a, b) => b.finalEquity - a.finalEquity);

  const best = ranked[0];
  const noneBeats = ranked.every(r => r.delta <= 0);

  let recommendation;
  if (noneBeats) {
    recommendation = `<p>Ninguna optimización iterativa supera la base en equity histórica completa.
      <span class="pill">Mantener parámetros actuales</span>.</p>`;
  } else if (best.key === '6m') {
    recommendation = `<p>La iteración a <b>6 meses</b> es la que más equity histórica acumula
      (${fmt$(best.delta)} vs base) y la más robusta.
      <span class="pill gold">Aplicar Mejora 3 (MIX 6m)</span></p>`;
  } else if (best.key === '3m') {
    recommendation = `<p>La iteración a <b>3 meses</b> rinde mejor que la de 6 meses (${fmt$(best.delta)} vs base).
      Sugiere que el mercado reciente tiene un perfil distinto.
      <span class="pill gold">Aplicar Mejora 2 (MIX 3m)</span></p>`;
  } else {
    recommendation = `<p>La iteración a <b>1 mes</b> es la mejor en cifras pero usa muestra muy pequeña
      → riesgo alto de sobreajuste. <span class="pill">Recomendado: contrastar con MIX 3m antes de aplicar</span></p>`;
  }

  card.innerHTML = `
    <h4>📌 Recomendación del análisis combinado</h4>
    <div style="font-size:12px;line-height:1.5">
      ${recommendation}
      <p style="font-size:11px;color:var(--ink-soft);margin-top:8px">
        El MIX optimiza <b>al mismo tiempo</b> los multiplicadores y los umbrales del VIX,
        alternando Quant ↔ Analista hasta convergencia. Comparado con el Quant solo,
        suele encontrar una mejora extra ajustando dónde caen las fronteras de régimen.
      </p>
    </div>
  `;
  container.appendChild(card);
}

// ---- CSV upload handlers (events) ---------------------------------------
function wireEventUpload(inputId, storageKey, onDone) {
  document.getElementById(inputId).addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const events = parseEventsCSV(reader.result);
      saveEvents(storageKey, events);
      onDone();
    };
    reader.readAsText(file);
  });
}

// ---- Modal handlers -----------------------------------------------------
function openAnalystModal() {
  document.getElementById('analystModal').style.display = 'flex';
  // Render whichever tab is active (default = vix)
  setTimeout(() => {
    drawVixAnalystChart();
    renderUpcomingEvents();
    renderHistoricalCorrelation();
  }, 50);
}
function closeAnalystModal() { document.getElementById('analystModal').style.display = 'none'; }

function regimeLabel(regimes, idx) {
  const cur = regimes[idx];
  const prev = idx > 0 ? regimes[idx - 1].maxVix : null;
  if (cur.maxVix >= 998) return `VIX > ${prev}`;
  if (prev === null)     return `VIX ≤ ${cur.maxVix}`;
  return `VIX ${prev}–${cur.maxVix}`;
}

function renderStats(stats, regimes) {
  const o = stats.overall;
  const winRateClose  = o.total ? (o.wins / o.total * 100) : 0;
  const winRateStrict = o.total ? (o.cleanWins / o.total * 100) : 0;
  const period = (o.firstDate && o.lastDate) ? `${o.firstDate} → ${o.lastDate}` : '—';

  const rowsHtml = regimes.map((reg, i) => {
    const s = stats.byRegime.get(reg) || { total: 0, wins: 0, lossUp: 0, lossDn: 0, cleanWins: 0, touched: 0 };
    const wrStrict = s.total ? (s.cleanWins / s.total * 100).toFixed(1) + '%' : '—';
    const cls = reg.trade ? '' : ' class="notrade"';
    return `<tr${cls}>
      <td>${regimeLabel(regimes, i)}${reg.trade ? '' : ' (no-trade)'}</td>
      <td>${s.total}</td>
      <td>${s.cleanWins || 0}</td>
      <td style="color:#b06000">${s.touched || 0}</td>
      <td class="loss-up">${s.lossUp}</td>
      <td class="loss-dn">${s.lossDn}</td>
      <td><b>${wrStrict}</b></td>
    </tr>`;
  }).join('');

  // Compute next-day strikes from the latest band
  let strikesHtml = `
    <div class="stats-card" id="strikesCard">
      <h3>Next-day strikes</h3>
      <div style="font-size:12px;color:var(--ink-soft)">Esperando datos…</div>
    </div>`;
  if (window._lastNextBand && window._lastPrevRow) {
    const nb = window._lastNextBand;
    const prev = window._lastPrevRow;
    const sellCallStrike = Math.ceil(nb.upper / 5) * 5;   // outside-rounded
    const sellPutStrike  = Math.floor(nb.lower / 5) * 5;
    const distCall = (sellCallStrike - prev.close) / prev.close * 100;
    const distPut  = (prev.close - sellPutStrike) / prev.close * 100;
    const tradeFlag = nb.trade
      ? '<span style="color:var(--good)">✓ TRADE</span>'
      : '<span class="loss-up">✗ NO-TRADE</span>';
    strikesHtml = `
      <div class="stats-card" id="strikesCard">
        <h3>Next-day strikes</h3>
        <div style="font-size:11px;color:var(--ink-soft);margin-bottom:6px">
          Cierre previo: <b>${prev.close.toFixed(2)}</b> · ${tradeFlag}
        </div>
        <div class="row">
          <span style="color:var(--good)"><b>↑ Vender CALL</b></span>
          <span><b>${sellCallStrike}</b> <small style="color:var(--ink-soft)">(banda ${nb.upper.toFixed(2)})</small></span>
        </div>
        <div class="row">
          <span style="color:var(--ink-soft)">Distancia</span>
          <span><b>+${distCall.toFixed(3)}%</b></span>
        </div>
        <div class="row" style="border-top:1px solid #eee;margin-top:6px;padding-top:6px">
          <span class="loss-dn"><b>↓ Vender PUT</b></span>
          <span><b>${sellPutStrike}</b> <small style="color:var(--ink-soft)">(banda ${nb.lower.toFixed(2)})</small></span>
        </div>
        <div class="row">
          <span style="color:var(--ink-soft)">Distancia</span>
          <span><b>−${distPut.toFixed(3)}%</b></span>
        </div>
        <div style="font-size:10px;color:var(--ink-soft);margin-top:6px;font-style:italic">
          Strikes redondeados al múltiplo de $5 más cercano hacia OUTSIDE
        </div>
      </div>`;
  }

  document.getElementById('stats').innerHTML = `
    ${strikesHtml}
    <div class="stats-card">
      <h3>Win rate (full history)</h3>
      <div class="big ${winRateStrict >= 80 ? '' : 'bad'}">${winRateStrict.toFixed(1)}%</div>
      <div style="font-size:11px;color:var(--ink-soft);margin-top:2px"><b>estricto</b> (sin toques intradía)</div>
      <div style="font-size:11px;color:var(--ink-soft);margin-top:6px">
        Close: <b>${winRateClose.toFixed(1)}%</b>
      </div>
      <div style="font-size:11px;color:#888;margin-top:4px">${period}</div>
    </div>
    <div class="stats-card" style="min-width:240px">
      <h3>Trade outcomes</h3>
      <div class="row"><span>Trading days</span><span><b>${o.total}</b></span></div>
      <div class="row"><span style="color:var(--good)">✓ Limpios</span><span style="color:var(--good)"><b>${o.cleanWins}</b></span></div>
      <div class="row"><span style="color:#b06000">? Tocados (recuperaron)</span><span style="color:#b06000"><b>${o.touched}</b></span></div>
      <div class="row"><span class="loss-up">✗ Loss — close &gt; upper</span><span class="loss-up"><b>${o.lossUp}</b></span></div>
      <div class="row"><span class="loss-dn">✗ Loss — close &lt; lower</span><span class="loss-dn"><b>${o.lossDn}</b></span></div>
      <div class="row" style="border-top:1px solid #eee;margin-top:4px;padding-top:4px">
        <span>Fallos estrictos</span><span><b>${o.touched + o.lossUp + o.lossDn}</b> (${(100 - winRateStrict).toFixed(1)}%)</span>
      </div>
      <div class="row"><span style="font-size:11px;color:var(--ink-soft)">Toques: ${o.callTouches} call · ${o.putTouches} put</span><span></span></div>
      <div class="row"><span>No-trade days</span><span>${o.noTrade}</span></div>
    </div>
    <div class="stats-card">
      <h3>Breakdown by regime</h3>
      <table class="regbreak">
        <thead><tr>
          <th>Regime</th><th>Días</th>
          <th style="color:#5dc080">✓</th>
          <th style="color:#b06000">?</th>
          <th class="loss-up">↑</th><th class="loss-dn">↓</th>
          <th>WR estricto</th>
        </tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
  `;
}

// ---- Plotting -------------------------------------------------------------
function drawChart(rows, bands, nextBand, windowDays) {
  const start = Math.max(1, rows.length - windowDays);
  const slice = rows.slice(start);
  const sliceBands = bands.slice(start);

  const dates  = slice.map(r => r.date);
  const closes = slice.map(r => r.close);
  const upper  = sliceBands.map(b => b ? b.upper : null);
  const lower  = sliceBands.map(b => b ? b.lower : null);
  // Palette — keep in sync with index.html :root vars
  const NAVY = '#143a64', GOLD = '#c9a227', BLUE = '#2c6fa3',
        BAD = '#b73232',  BAD_SOFT = '#c46a35', MUTED = '#9aa9bb';
  const closeColors = sliceBands.map(b => b && !b.trade ? MUTED : NAVY);

  // Losses for the iron condor: close outside the band on a trading day.
  // Skip no-trade days (they don't count as losses since we wouldn't open a position).
  const lossUpX = [], lossUpY = [], lossDnX = [], lossDnY = [];
  slice.forEach((r, i) => {
    const b = sliceBands[i];
    if (!b || !b.trade) return;
    if (r.close >= b.upper) { lossUpX.push(r.date); lossUpY.push(r.close); }
    if (r.close <= b.lower) { lossDnX.push(r.date); lossDnY.push(r.close); }
  });

  // Background shading on no-trade days (merge consecutive runs)
  const shapes = [];
  let runStart = null;
  for (let i = 0; i <= slice.length; i++) {
    const noTrade = i < slice.length && sliceBands[i] && !sliceBands[i].trade;
    if (noTrade && runStart === null) runStart = i;
    if (!noTrade && runStart !== null) {
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: shiftDate(slice[runStart].date, -0.5),
        x1: shiftDate(slice[i - 1].date,    +0.5),
        y0: 0, y1: 1,
        fillcolor: 'rgba(201, 162, 39, 0.12)', line: { width: 0 },
        layer: 'below',
      });
      runStart = null;
    }
  }

  // Next-day extension (dashed)
  const lastStr = slice[slice.length - 1].date;
  const nextStr = shiftDate(lastStr, +1);
  const lastUpper = upper[upper.length - 1];
  const lastLower = lower[lower.length - 1];

  const traces = [
    { x: dates, y: closes, mode: 'lines+markers', name: 'Close',
      line: { color: NAVY, width: 2.2 },
      marker: { color: closeColors, size: 6, line: { color: NAVY, width: 1 } } },
    { x: dates, y: upper, mode: 'lines', name: 'Upper band',
      line: { color: GOLD, width: 1.8 } },
    { x: dates, y: lower, mode: 'lines', name: 'Lower band',
      line: { color: BLUE, width: 1.8 },
      fill: 'tonexty', fillcolor: 'rgba(44, 111, 163, 0.07)' },
    { x: lossUpX, y: lossUpY, mode: 'markers', name: 'Loss — close > upper',
      marker: { color: BAD, size: 12, symbol: 'x', line: { width: 1, color: '#fff' } } },
    { x: lossDnX, y: lossDnY, mode: 'markers', name: 'Loss — close < lower',
      marker: { color: BAD_SOFT, size: 12, symbol: 'x', line: { width: 1, color: '#fff' } } },
  ];

  if (nextBand) {
    traces.push({
      x: [lastStr, nextStr], y: [lastUpper, nextBand.upper],
      mode: 'lines+markers', name: 'Next upper',
      line: { color: GOLD, width: 1.8, dash: 'dash' }, marker: { size: 9, color: GOLD },
    });
    traces.push({
      x: [lastStr, nextStr], y: [lastLower, nextBand.lower],
      mode: 'lines+markers', name: 'Next lower',
      line: { color: BLUE, width: 1.8, dash: 'dash' }, marker: { size: 9, color: BLUE },
    });
  }

  const layout = {
    margin: { t: 20, r: 20, b: 50, l: 60 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#f3f8fc',
    font: { family: 'Segoe UI, sans-serif', color: '#0c1f33' },
    xaxis: {
      title: 'Date', type: 'date',
      gridcolor: '#dce7f1', linecolor: '#143a64', zerolinecolor: '#dce7f1',
    },
    yaxis: {
      title: 'SPX',
      gridcolor: '#dce7f1', linecolor: '#143a64', zerolinecolor: '#dce7f1',
    },
    legend: {
      orientation: 'h', y: -0.2,
      bgcolor: 'rgba(255,255,255,0.6)', bordercolor: '#cfe1f2', borderwidth: 1,
    },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#0a2540', font: { color: '#fff' }, bordercolor: '#c9a227' },
    shapes,
  };
  safePlotly('chart', traces, layout, { responsive: true });

  // Info panel — uses Number.isFinite (strict) to avoid null→0 coercion bugs
  const tradeLabel = nextBand && nextBand.trade ? 'TRADE' : 'NO-TRADE';
  const reg = nextBand ? nextBand.regime : null;
  const lastRow = rows[rows.length - 1];
  const safe = (v, dec) => Number.isFinite(v) ? v.toFixed(dec) : '—';
  const safePct = (v, dec) => Number.isFinite(v) ? (v * 100).toFixed(dec) + '%' : '—';
  const ivHvNote = (Number.isFinite(lastRow.iv) && Number.isFinite(lastRow.hv))
    ? `IV=${safe(lastRow.iv, 3)} · HV=${safe(lastRow.hv, 3)} · `
    : `<span style="color:#ffb39b">IV/HV no introducidos</span> · `;
  const trendArrow = { up: '↑', down: '↓', flat: '↔' };
  const trendLabel = nextBand && nextBand.trend
    ? `${trendArrow[nextBand.trend]} ${nextBand.trend === 'up' ? 'Alcista' : nextBand.trend === 'down' ? 'Bajista' : 'Rango'}`
    : '—';
  const ctNote = nextBand
    ? (nextBand.Ccall !== nextBand.Cput
        ? `<b style="color:#e8cf78">C<sub>call</sub>=${(nextBand.Ccall*100).toFixed(1)}% · C<sub>put</sub>=${(nextBand.Cput*100).toFixed(1)}%</b>`
        : `<b style="color:#e8cf78">C<sub>t</sub>=${(nextBand.Ccall*100).toFixed(1)}%</b>`)
    : '';
  document.getElementById('info').innerHTML = nextBand
    ? `Showing last ${slice.length} days. ` +
      `Next-day (${nextStr}) — VIX<sub>prev</sub>=${safe(lastRow.vix, 2)} · ` +
      ivHvNote +
      `Tendencia: <b>${trendLabel}</b> · ` +
      ctNote + ` · ` +
      `σ<sub>high</sub>=${safePct(nextBand.sigUp, 2)} · ` +
      `σ<sub>low</sub>=${safePct(nextBand.sigDn, 2)} · ` +
      `k=${reg ? reg.k : '—'} · mults (${reg ? reg.upperMult : '—'}/${reg ? reg.lowerMult : '—'}) → ` +
      `<b>Upper ${safe(nextBand.upper, 2)} · Lower ${safe(nextBand.lower, 2)}</b> · ` +
      `<span style="color:${nextBand.trade ? '#7fdca8' : '#ffb39b'}"><b>${tradeLabel}</b></span>`
    : 'No data — not enough history for the chosen lookback yet.';
}

// ---- VIX context chart ----------------------------------------------------
function drawVixChart(rows, bands, regimes, windowDays) {
  const start = Math.max(1, rows.length - windowDays);
  const slice = rows.slice(start);
  const sliceBands = bands.slice(start);

  const dates = slice.map(r => r.date);
  const vix   = slice.map(r => r.vix);

  // Loss days, plotted at their VIX value (so you see "this loss happened with VIX=X")
  const lossUpX = [], lossUpY = [], lossDnX = [], lossDnY = [];
  slice.forEach((r, i) => {
    const b = sliceBands[i];
    if (!b || !b.trade) return;
    if (r.close >= b.upper) { lossUpX.push(r.date); lossUpY.push(r.vix); }
    if (r.close <= b.lower) { lossDnX.push(r.date); lossDnY.push(r.vix); }
  });

  // Y-axis range with a little headroom; cap "open-ended" no-trade band against this.
  const vMax = Math.max(...vix, ...regimes.map(r => r.maxVix < 998 ? r.maxVix : 0)) * 1.08;

  const shapes = [];
  // Threshold lines (every regime's upper bound except the open-ended last)
  regimes.forEach(reg => {
    if (reg.maxVix >= 998) return;
    shapes.push({
      type: 'line', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: reg.maxVix, y1: reg.maxVix,
      line: { color: '#c9a227', width: 1, dash: 'dot' },
    });
  });
  // Shade no-trade VIX bands across the whole width
  for (let i = 0; i < regimes.length; i++) {
    if (regimes[i].trade) continue;
    const yLo = i > 0 ? regimes[i - 1].maxVix : 0;
    const yHi = regimes[i].maxVix < 998 ? regimes[i].maxVix : vMax;
    shapes.push({
      type: 'rect', xref: 'paper', yref: 'y',
      x0: 0, x1: 1, y0: yLo, y1: yHi,
      fillcolor: 'rgba(201, 162, 39, 0.13)',
      line: { width: 0 }, layer: 'below',
    });
  }

  // Annotations for thresholds (right edge, small)
  const annotations = regimes
    .filter(reg => reg.maxVix < 998)
    .map(reg => ({
      xref: 'paper', yref: 'y',
      x: 1, y: reg.maxVix,
      xanchor: 'right', yanchor: 'bottom',
      text: ` ≤ ${reg.maxVix} `,
      font: { size: 10, color: '#8a6d10' },
      bgcolor: 'rgba(255,255,255,0.7)',
      showarrow: false,
    }));

  const traces = [
    { x: dates, y: vix, mode: 'lines+markers', name: 'VIX',
      line: { color: '#143a64', width: 2 },
      marker: { color: '#143a64', size: 5 } },
    { x: lossUpX, y: lossUpY, mode: 'markers', name: 'Loss (close > upper)',
      marker: { color: '#b73232', size: 11, symbol: 'x', line: { width: 1, color: '#fff' } } },
    { x: lossDnX, y: lossDnY, mode: 'markers', name: 'Loss (close < lower)',
      marker: { color: '#c46a35', size: 11, symbol: 'x', line: { width: 1, color: '#fff' } } },
  ];

  const layout = {
    margin: { t: 20, r: 20, b: 50, l: 60 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#f3f8fc',
    font: { family: 'Segoe UI, sans-serif', color: '#0c1f33' },
    xaxis: {
      title: 'Date', type: 'date',
      gridcolor: '#dce7f1', linecolor: '#143a64',
    },
    yaxis: {
      title: 'VIX',
      gridcolor: '#dce7f1', linecolor: '#143a64',
      range: [Math.max(0, Math.min(...vix) * 0.9), vMax],
    },
    legend: {
      orientation: 'h', y: -0.25,
      bgcolor: 'rgba(255,255,255,0.6)', bordercolor: '#cfe1f2', borderwidth: 1,
    },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#0a2540', font: { color: '#fff' }, bordercolor: '#c9a227' },
    shapes, annotations,
  };
  safePlotly('vixChart', traces, layout, { responsive: true });
}

// Shift an ISO date string by N days (fractional allowed)
function shiftDate(iso, days) {
  const d = new Date(iso);
  d.setDate(d.getDate() + Math.floor(days));
  const frac = days - Math.floor(days);
  if (frac) d.setHours(d.getHours() + frac * 24);
  return d.toISOString().slice(0, frac ? 19 : 10);
}

function isDayAccepted(todayRow, regime) {
  return Boolean(regime && regime.trade);
}

// ---- Trend C/P indicator ------------------------------------------------
// Compares mean(close last 20) vs mean(close last 20–40 ago).
// > +0.5% → Alcista; < −0.5% → Bajista; en medio → Rango.
// (TREND_WINDOW and TREND_THRESHOLD declared near MIN_SAMPLES at top of file)

function computeTrend(rows) {
  if (!rows || rows.length < TREND_WINDOW * 2) return null;
  const recent = rows.slice(-TREND_WINDOW);
  const prev   = rows.slice(-TREND_WINDOW * 2, -TREND_WINDOW);
  const meanClose = arr => {
    const valid = arr.filter(r => isFinite(r.close));
    return valid.length ? valid.reduce((s, r) => s + r.close, 0) / valid.length : NaN;
  };
  const recentMean = meanClose(recent);
  const prevMean   = meanClose(prev);
  if (!isFinite(recentMean) || !isFinite(prevMean) || prevMean === 0) return null;
  const diffPct = (recentMean - prevMean) / prevMean * 100;
  let label, kind, arrow;
  if (diffPct >  TREND_THRESHOLD) { label = 'Alcista'; kind = 'up';   arrow = '↑'; }
  else if (diffPct < -TREND_THRESHOLD) { label = 'Bajista'; kind = 'down'; arrow = '↓'; }
  else                                 { label = 'Rango';   kind = 'flat'; arrow = '↔'; }
  return { recentMean, prevMean, diffPct, label, kind, arrow,
           recentRange: [recent[0].date, recent[recent.length-1].date],
           prevRange:   [prev[0].date,   prev[prev.length-1].date] };
}

function renderTrendCard() {
  const card = document.getElementById('trendCard');
  if (!card) return;
  console.log('[Trend] renderTrendCard · currentRows:', currentRows ? currentRows.length : 'null');
  try {
    if (!currentRows || currentRows.length === 0) {
      card.innerHTML = `
        <div class="trend-title">Tendencia C/P</div>
        <div style="font-size:11px;color:var(--ink-soft);text-align:center;margin-top:8px">
          Sin datos cargados
        </div>`;
      window._lastTrend = null;
      return;
    }
    if (currentRows.length < TREND_WINDOW * 2) {
      card.innerHTML = `
        <div class="trend-title">Tendencia C/P</div>
        <div style="font-size:11px;color:var(--ink-soft);text-align:center;margin-top:8px">
          ${currentRows.length}/${TREND_WINDOW * 2} días<br>(insuficientes)
        </div>`;
      window._lastTrend = null;
      return;
    }
    const t = computeTrend(currentRows);
    console.log('[Trend] computeTrend result:', t);
    if (!t) {
      card.innerHTML = `<div class="trend-title">Tendencia C/P</div>
        <div style="font-size:11px;color:var(--bad);text-align:center;margin-top:8px">
          Error en cálculo (cierres no numéricos)
        </div>`;
      window._lastTrend = null;
      return;
    }
    window._lastTrend = t;
    const sign = t.diffPct >= 0 ? '+' : '';
    card.innerHTML = `
      <div class="trend-title">Tendencia C/P</div>
      <div class="trend-label ${t.kind}">${t.arrow} ${t.label}</div>
      <div class="trend-detail">${sign}${t.diffPct.toFixed(2)}% (20d)</div>
      <div class="trend-means">
        μ<sub>20d</sub> = ${t.recentMean.toFixed(0)}<br>
        μ<sub>20-40d</sub> = ${t.prevMean.toFixed(0)}
      </div>`;
  } catch (err) {
    console.error('[Trend] renderTrendCard error:', err);
    card.innerHTML = `<div class="trend-title">Tendencia C/P</div>
      <div style="font-size:11px;color:var(--bad)">Error: ${err.message}</div>`;
  }
}

// ---- Compression panel UI -----------------------------------------------
function readCompressionParamsFromUI() {
  compressionParams.lambda      = parseFloat(document.getElementById('cParamLambda').value) || 1.5;
  compressionParams.Cmin        = parseFloat(document.getElementById('cParamCmin').value)   || 0.5;
  compressionParams.Cmax        = parseFloat(document.getElementById('cParamCmax').value)   || 0.95;
  compressionParams.Vref        = parseFloat(document.getElementById('cParamVref').value)   || 18;
  const shift                   = parseFloat(document.getElementById('cParamShift').value);
  compressionParams.shiftFactor = isFinite(shift) ? shift : 0.25;
}
function readCompressionVarsFromUI() {
  document.querySelectorAll('#compressionTable input').forEach(inp => {
    const i = +inp.dataset.i, f = inp.dataset.f;
    if (f === 'active') compressionVars[i].active = inp.checked;
    if (f === 'weight') compressionVars[i].weight = parseFloat(inp.value) || 0;
  });
}
function renderCompressionPanel() {
  const lastPrev = currentRows && currentRows.length ? currentRows[currentRows.length - 1] : null;
  const tbody = document.querySelector('#compressionTable tbody');
  tbody.innerHTML = compressionVars.map((v, i) => {
    const std = lastPrev ? standardizeVar(v.id, lastPrev, compressionParams) : null;
    const stdStr = (std === null || !isFinite(std)) ? '<span style="color:#b73232">no data</span>' : std.toFixed(3);
    const ws    = (std === null || !isFinite(std)) ? '—' : (v.weight * std).toFixed(3);
    const groupClass = `group-tag ${v.group}`;
    const groupTitle = COMPRESSION_GROUP_LABEL[v.group] || '';
    return `<tr title="${v.desc}">
      <td><input type="checkbox" data-i="${i}" data-f="active" ${v.active ? 'checked' : ''}></td>
      <td><b>${v.label}</b></td>
      <td><span class="${groupClass}" title="${groupTitle}">${v.group}</span></td>
      <td><input type="number" step="0.05" data-i="${i}" data-f="weight" value="${v.weight}"></td>
      <td>${stdStr}</td>
      <td>${ws}</td>
    </tr>`;
  }).join('');

  // Show breakdown of Z and Ccall/Cput for the latest day
  if (lastPrev) {
    const bd = compressionBreakdown(lastPrev);
    const pieces = bd.terms.map(t => {
      if (t.missing) return `<span class="breakdown-pill" style="color:#b73232">${t.label}: n/a</span>`;
      const sign = t.weighted >= 0 ? '+' : '−';
      return `<span class="breakdown-pill"><b>${t.label}</b>: ${sign}${Math.abs(t.weighted).toFixed(3)}</span>`;
    }).join(' ');
    const trendLbl = bd.trend ? (bd.trend === 'up' ? '↑ Alcista' : bd.trend === 'down' ? '↓ Bajista' : '↔ Rango') : '—';
    const trendColor = bd.trend === 'up' ? 'var(--good)' : bd.trend === 'down' ? 'var(--bad)' : 'var(--ink-soft)';
    const shiftNote = (bd.trend === 'up' || bd.trend === 'down')
      ? `, desplazamiento Δ = ±${bd.delta.toFixed(3)}`
      : '';
    document.getElementById('cOptBreakdown').innerHTML = `
      <b>Cierre más reciente (${lastPrev.date}):</b>
      ${pieces}<br>
      Z = ${bd.Z.toFixed(3)} →
      sigmoid(λ·Z) = ${bd.C_raw.toFixed(3)} →
      C<sub>base</sub> = ${bd.base.toFixed(3)}
      &nbsp;·&nbsp; Tendencia: <b style="color:${trendColor}">${trendLbl}</b>${shiftNote} →
      <b style="color:var(--gold-500)">C<sub>call</sub> = ${bd.Ccall.toFixed(3)} · C<sub>put</sub> = ${bd.Cput.toFixed(3)}</b>
    `;
  } else {
    document.getElementById('cOptBreakdown').innerHTML = '';
  }
}

// "Run Optimization" — coordinate descent on weights of active variables.
// Maximizes total P&L (= win_rate × WIN + loss_rate × LOSS over trading days).
async function runCompressionOptimization() {
  if (!currentRows) return;
  const btn = document.getElementById('cRunOpt');
  const status = document.getElementById('cOptStatus');
  btn.disabled = true;
  status.style.color = '';
  const t0 = performance.now();
  try {
    readCompressionParamsFromUI();
    readCompressionVarsFromUI();
    const active = compressionVars.filter(v => v.active);
    if (active.length === 0) {
      status.style.color = '#b73232';
      status.textContent = '✗ Activa al menos una variable.';
      btn.disabled = false; return;
    }

    const lookback = parseInt(document.getElementById('lookback').value, 10);
    const initCap = parseFloat(document.getElementById('initialCapital').value) || 1000;
    const regimes = readRegimes();

    const scoreCurrent = () => {
      const { bands } = computeBands(currentRows, regimes, lookback);
      const pnl = computePnL(currentRows, bands, initCap);
      return pnl.finalEquity - initCap;
    };

    let bestScore = scoreCurrent();
    const WEIGHTS = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0];
    const MAX_ROUNDS = 4;

    for (let round = 0; round < MAX_ROUNDS; round++) {
      let changed = false;
      for (const v of active) {
        const orig = v.weight;
        let bestW = orig;
        for (const w of WEIGHTS) {
          v.weight = w;
          const s = scoreCurrent();
          if (s > bestScore) { bestScore = s; bestW = w; }
        }
        v.weight = bestW;
        if (Math.abs(bestW - orig) > 1e-6) changed = true;
        status.textContent = `Optimizando ${v.label}…`;
        await new Promise(r => setTimeout(r, 0));
      }
      if (!changed) break;
    }

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent = `✓ Completado en ${elapsed}s · score: $${bestScore.toFixed(0)}`;
    renderCompressionPanel();
    recalc();
  } catch (err) {
    console.error(err);
    status.style.color = '#b73232';
    status.textContent = `✗ Error: ${err.message || err}`;
  } finally {
    btn.disabled = false;
  }
}

// ---- Regime table UI -----------------------------------------------------
function renderRegimeTable() {
  const tbody = document.querySelector('#regimeTable tbody');
  tbody.innerHTML = '';
  DEFAULT_REGIMES.forEach((r, i) => {
    const tr = document.createElement('tr');
    if (!r.trade) tr.className = 'notrade';
    tr.innerHTML = `
      <td><input type="number" step="0.1" data-i="${i}" data-f="maxVix"    value="${r.maxVix}"></td>
      <td><input type="number" step="0.1" data-i="${i}" data-f="upperMult" value="${r.upperMult}"></td>
      <td><input type="number" step="0.1" data-i="${i}" data-f="lowerMult" value="${r.lowerMult}"></td>
      <td><input type="number" step="0.1" data-i="${i}" data-f="k"         value="${r.k}"></td>
      <td><input type="checkbox"          data-i="${i}" data-f="trade"     ${r.trade ? 'checked' : ''}></td>
    `;
    tbody.appendChild(tr);
  });
}

function readRegimes() {
  const regimes = DEFAULT_REGIMES.map(r => ({ ...r }));
  document.querySelectorAll('#regimeTable input').forEach(inp => {
    const i = +inp.dataset.i, f = inp.dataset.f;
    regimes[i][f] = inp.type === 'checkbox' ? inp.checked : parseFloat(inp.value);
  });
  return regimes;
}

// ---- CSV cache (so the user doesn't have to re-pick it on every refresh) -
const CSV_CACHE_KEY = 'spx-vix-csv-cache-v1';

function saveCSVCache(text, filename) {
  try {
    localStorage.setItem(CSV_CACHE_KEY, JSON.stringify({
      text, filename, savedAt: Date.now(), bytes: text.length,
    }));
    return true;
  } catch (e) {
    console.warn('No se pudo cachear el CSV (¿localStorage lleno?):', e);
    return false;
  }
}
function loadCSVCache() {
  try { return JSON.parse(localStorage.getItem(CSV_CACHE_KEY) || 'null'); }
  catch (_) { return null; }
}
function clearCSVCache() { localStorage.removeItem(CSV_CACHE_KEY); }

function updateCacheStatus() {
  const status = document.getElementById('csvCacheStatus');
  if (!status) return;
  const cache = loadCSVCache();
  if (!cache) {
    status.innerHTML = '';
    return;
  }
  const ageDays = Math.floor((Date.now() - cache.savedAt) / 86400000);
  const ageLabel = ageDays === 0 ? 'hoy' : ageDays === 1 ? 'ayer' : `hace ${ageDays}d`;
  const rowsCount = currentRows ? currentRows.length : '?';
  status.innerHTML = `<span style="color:var(--good)">✓</span> en caché: <b>${cache.filename || 'csv'}</b> (${rowsCount} filas, ${ageLabel}) <button id="clearCsvCacheBtn" type="button" title="Borrar caché y obligar a re-subir CSV" style="background:transparent;border:1px solid var(--blue-200);color:var(--ink-soft);border-radius:3px;padding:1px 6px;font-size:10px;cursor:pointer;margin-left:6px">✕ limpiar</button>`;
  const btn = document.getElementById('clearCsvCacheBtn');
  if (btn) btn.addEventListener('click', () => {
    if (!confirm('¿Borrar el CSV cacheado? Tendrás que volver a subirlo para ver los datos.')) return;
    clearCSVCache();
    currentRows = null;
    recalc();
    updateCacheStatus();
  });
}

// ---- Manual entries (persisted in localStorage) --------------------------
// Stored as { 'YYYY-MM-DD': {close, high, low, vix} }. Overlaid on top of CSV.
const STORAGE_KEY = 'spx-vix-entries-v1';

function loadEntries() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
  catch (_) { return {}; }
}
function saveEntries(entries) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

function applyEntries(rows, entries) {
  const byDate = new Map(rows.map(r => [r.date, r]));
  for (const [date, vals] of Object.entries(entries)) {
    // JSON.stringify convierte NaN → null, JSON.parse no lo devuelve a NaN.
    // Reconvertimos los nulls a NaN para que isFinite() funcione correctamente.
    const cleaned = {};
    for (const [k, v] of Object.entries(vals)) {
      cleaned[k] = (v === null) ? NaN : v;
    }
    byDate.set(date, { date, ...cleaned });
  }
  const merged = Array.from(byDate.values()).sort((a, b) => new Date(a.date) - new Date(b.date));
  enrichRows(merged);
  return merged;
}

function upsertRow(row) {
  const entries = loadEntries();
  entries[row.date] = {
    open: row.open,
    close: row.close, high: row.high, low: row.low, vix: row.vix,
    iv: row.iv, hv: row.hv,
    ivRank: row.ivRank, ivPctl: row.ivPctl, ivChg: row.ivChg,
    pcv: row.pcv,
  };
  saveEntries(entries);
  if (!currentRows) {
    currentRows = [row];
  } else {
    const i = currentRows.findIndex(r => r.date === row.date);
    if (i >= 0) currentRows[i] = row;
    else currentRows.push(row);
    currentRows.sort((a, b) => new Date(a.date) - new Date(b.date));
  }
  enrichRows(currentRows);
  recalc();
}

const NYSE_FULL_CLOSE_HOLIDAYS = new Set([
  '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
  '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
  '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31',
  '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24'
]);

function isMarketHoliday(dateStr) {
  return NYSE_FULL_CLOSE_HOLIDAYS.has(String(dateStr || ''));
}

function isMarketTradingDate(dateStr) {
  const [y, m, d] = String(dateStr || '').split('-').map(Number);
  if (!y || !m || !d) return false;
  const dt = new Date(Date.UTC(y, m - 1, d));
  const dow = dt.getUTCDay();
  return dow !== 0 && dow !== 6 && !isMarketHoliday(dateStr);
}

// ---- Missing-day detection -----------------------------------------------
// Weekdays between (last data date + 1) and yesterday that aren't in the data.
// Holidays will get falsely flagged — user can dismiss them with "Skip".
function detectMissing(rows) {
  if (!rows || rows.length === 0) return [];
  const dataDates = new Set(rows.map(r => r.date));
  const skipped = new Set(JSON.parse(localStorage.getItem('spx-vix-skip-v1') || '[]'));
  const last = new Date(rows[rows.length - 1].date);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const missing = [];
  const d = new Date(last);
  while (true) {
    d.setDate(d.getDate() + 1);
    if (d >= today) break;
    const iso = d.toISOString().slice(0, 10);
    if (!isMarketTradingDate(iso)) continue;
    if (!dataDates.has(iso) && !skipped.has(iso)) missing.push(iso);
  }
  return missing;
}

function dismissMissing(date) {
  const skipped = new Set(JSON.parse(localStorage.getItem('spx-vix-skip-v1') || '[]'));
  skipped.add(date);
  localStorage.setItem('spx-vix-skip-v1', JSON.stringify([...skipped]));
}

function renderMissingBanner(missing) {
  const banner = document.getElementById('missingBanner');
  if (missing.length === 0) { banner.style.display = 'none'; return; }
  banner.style.display = '';
  document.getElementById('missingTitle').textContent =
    `⚠ ${missing.length} sesión${missing.length === 1 ? '' : 'es'} sin datos`;
  const rows = missing.map(date => `
    <tr data-date="${date}">
      <td><b>${date}</b></td>
      <td><input type="number" step="0.01"  data-f="open"   placeholder="Open"></td>
      <td><input type="number" step="0.01"  data-f="close"  placeholder="Close"></td>
      <td><input type="number" step="0.01"  data-f="high"   placeholder="High"></td>
      <td><input type="number" step="0.01"  data-f="low"    placeholder="Low"></td>
      <td><input type="number" step="0.01"  data-f="vix"    placeholder="VIX"></td>
      <td><input type="number" step="0.001" data-f="iv"     placeholder="IV"></td>
      <td><input type="number" step="0.001" data-f="hv"     placeholder="HV"></td>
      <td><input type="number" step="0.1"   data-f="ivRank" placeholder="IVR"></td>
      <td><input type="number" step="0.1"   data-f="ivPctl" placeholder="IVP"></td>
      <td><input type="number" step="0.001" data-f="ivChg"  placeholder="IVCHG"></td>
      <td><input type="number" step="0.01"  data-f="pcv"    placeholder="PCV"></td>
      <td>
        <button class="add" type="button">Add</button>
        <button class="skip" type="button" title="Mark as non-trading day (e.g. holiday)">Skip</button>
      </td>
    </tr>
  `).join('');
  document.getElementById('missingContent').innerHTML = `
    <div style="font-size:12px;color:#6b5b2e;margin-bottom:6px">
      Días laborables entre el último cierre y hoy que no están en los datos.
      <b>IV/HV/IVR/IVP/IVCHG/PCV son opcionales</b> — déjalos vacíos si no los tienes y se omiten del cálculo.
      Si alguno fue festivo de mercado, pulsa <b>Skip</b> para ignorarlo.
    </div>
    <table class="missing">
      <thead><tr>
        <th>Date</th><th>Open</th><th>Close</th><th>High</th><th>Low</th><th>VIX</th>
        <th>IV</th><th>HV</th><th>IVR</th><th>IVP</th><th>IVCHG</th><th>PCV</th><th></th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ---- Wire up --------------------------------------------------------------
let currentRows = null;

function recalc() {
  const info = document.getElementById('info');
  if (!currentRows) {
    info.textContent = 'No data loaded yet — pick a CSV file with the "CSV file" selector above.';
    info.style.color = '#b00';
    renderTrendCard(); // also update on the empty case
    return;
  }
  info.style.color = '#444';
  // ALWAYS render the trend card first — it has its own try/catch and depends only on currentRows.
  // If anything later in recalc throws, at least the trend is updated.
  renderTrendCard();
  try {
    _recalcInner();
  } catch (err) {
    console.error('[recalc] error en pipeline:', err);
    info.style.color = '#b00';
    info.textContent = `? Error en recalc: ${err.message} (revisa F12 Console)`;
  }
}

function _recalcInner() {
  const displayWindow = parseInt(document.getElementById('window').value, 10);
  const lookback      = parseInt(document.getElementById('lookback').value, 10);
  const regimes = readRegimes();
  const { bands, nextBand } = computeBands(currentRows, regimes, lookback);
  // Cache next-day band + prev row for the strikes panel
  window._lastNextBand = nextBand;
  window._lastPrevRow  = currentRows[currentRows.length - 1];

  const stats = computeStats(currentRows, bands);
  renderStats(stats, regimes);
  drawChart(currentRows, bands, nextBand, displayWindow);
  drawVixChart(currentRows, bands, regimes, displayWindow);

  const initialCapital = parseFloat(document.getElementById('initialCapital').value) || 1000;
  const pnl = computePnL(currentRows, bands, initialCapital);
  drawEquityChart(pnl);
  renderEquityStats(pnlSummary(pnl), pnl);
  // Only re-render the (potentially huge) trades table if it's visible
  if (document.getElementById('tradesPanel').style.display !== 'none') {
    renderTradesTable(pnl);
  }
  window._lastPnL = pnl; // cache for the toggle handler

  renderMissingBanner(detectMissing(currentRows));
  prefillNextDate();
  checkEventBanner();
  renderCompressionPanel();
  renderTrendCard();
  updateCacheStatus();
  // Auto-refresh desglose if visible
  if (document.getElementById('desglosePanel').style.display !== 'none') {
    renderDesglose();
  }
}

// Pre-fill the entry-form date with the next missing weekday (or today).
function prefillNextDate() {
  const dateInput = document.getElementById('entryDate');
  if (!dateInput || dateInput.value) return; // don't overwrite user's pick
  const missing = detectMissing(currentRows || []);
  if (missing.length) {
    dateInput.value = missing[0];
  } else if (currentRows && currentRows.length) {
    const last = new Date(currentRows[currentRows.length - 1].date);
    do { last.setDate(last.getDate() + 1); }
    while (last.getDay() === 0 || last.getDay() === 6);
    dateInput.value = last.toISOString().slice(0, 10);
  }
}

renderRegimeTable();
renderCompressionPanel();
renderTrendCard();
updateFavoritesCardCount();

document.getElementById('csvFile').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    saveCSVCache(reader.result, file.name);
    currentRows = applyEntries(parseCSV(reader.result), loadEntries());
    recalc();
    updateCacheStatus();
  };
  reader.readAsText(file);
});

// ---- Yahoo Finance fetch (for auto-fill of Open) ------------------------
// Returns {open, high, low, close} for a given date and symbol.
// Tries direct fetch first, falls back to a CORS proxy.
async function fetchYahooOHLC(symbol, dateStr) {
  const date = new Date(dateStr + 'T00:00:00Z');
  const period1 = Math.floor(date.getTime() / 1000) - 5 * 86400;
  const period2 = Math.floor(date.getTime() / 1000) + 86400;
  const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${period1}&period2=${period2}&interval=1d`;

  let data;
  try {
    const r = await fetch(yahooUrl);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    data = await r.json();
  } catch (e) {
    console.log('[Yahoo] direct fetch failed, trying proxy:', e.message);
    const r = await fetch('https://corsproxy.io/?url=' + encodeURIComponent(yahooUrl));
    if (!r.ok) throw new Error('Proxy HTTP ' + r.status);
    data = await r.json();
  }

  if (data.chart && data.chart.error) throw new Error(data.chart.error.description || 'Error de Yahoo');
  const result = data.chart && data.chart.result && data.chart.result[0];
  if (!result) throw new Error('Respuesta vacía de Yahoo');
  const ts = result.timestamp || [];
  const q  = result.indicators && result.indicators.quote && result.indicators.quote[0];
  if (!q) throw new Error('Sin datos OHLC');

  for (let i = 0; i < ts.length; i++) {
    const d = new Date(ts[i] * 1000).toISOString().slice(0, 10);
    if (d === dateStr) {
      return { open: q.open[i], high: q.high[i], low: q.low[i], close: q.close[i] };
    }
  }
  // Find closest before
  for (let i = ts.length - 1; i >= 0; i--) {
    const d = new Date(ts[i] * 1000).toISOString().slice(0, 10);
    if (d <= dateStr) {
      return { open: q.open[i], high: q.high[i], low: q.low[i], close: q.close[i],
               actualDate: d, requested: dateStr };
    }
  }
  throw new Error('No hay datos para ' + dateStr + ' (¿es fin de semana o festivo?)');
}

// ---- Top-right "Add session data" form ----------------------------------
document.getElementById('entryForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const fb = document.getElementById('entryFeedback');
  const num = id => {
    const v = document.getElementById(id).value;
    return v === '' ? NaN : parseFloat(v);
  };
  const row = {
    date:   document.getElementById('entryDate').value,
    open:   num('entryOpen'),
    close:  num('entryClose'),
    high:   num('entryHigh'),
    low:    num('entryLow'),
    vix:    num('entryVix'),
    iv:     num('entryIv'),
    hv:     num('entryHv'),
    ivRank: num('entryIvr'),
    ivPctl: num('entryIvp'),
    ivChg:  num('entryIvchg'),
    pcv:    num('entryPcv'),
  };
  if (!row.date || [row.close, row.high, row.low, row.vix].some(v => !isFinite(v))) {
    fb.className = 'feedback err'; fb.textContent = 'Faltan campos básicos o no son numéricos (IV/HV/IVR/IVP/IVCHG/PCV son opcionales).';
    return;
  }
  if (row.high < row.low) {
    fb.className = 'feedback err'; fb.textContent = 'High < Low — revisa los valores.';
    return;
  }
  upsertRow(row);
  const optionals = [];
  if (isFinite(row.iv))     optionals.push('IV');
  if (isFinite(row.hv))     optionals.push('HV');
  if (isFinite(row.ivRank)) optionals.push('IVR');
  if (isFinite(row.ivPctl)) optionals.push('IVP');
  if (isFinite(row.ivChg))  optionals.push('IVCHG');
  if (isFinite(row.pcv))    optionals.push('PCV');
  const note = optionals.length ? ` · con ${optionals.join(' + ')}` : ' · sin variables opcionales';
  fb.className = 'feedback'; fb.textContent = `✓ ${row.date} guardado${note}.`;
  ['entryOpen','entryClose','entryHigh','entryLow','entryVix','entryIv','entryHv','entryIvr','entryIvp','entryIvchg','entryPcv']
    .forEach(id => document.getElementById(id).value = '');
  document.getElementById('entryDate').value = '';
  prefillNextDate();
});

// ---- Inline fill buttons in the missing-days banner ---------------------
document.getElementById('missingBanner').addEventListener('click', (e) => {
  const tr = e.target.closest('tr[data-date]');
  if (!tr) return;
  const date = tr.dataset.date;
  if (e.target.classList.contains('add')) {
    const get = f => {
      const v = tr.querySelector(`input[data-f="${f}"]`).value;
      return v === '' ? NaN : parseFloat(v);
    };
    const row = {
      date,
      open: get('open'),
      close: get('close'), high: get('high'), low: get('low'), vix: get('vix'),
      iv:     get('iv'),     hv:     get('hv'),
      ivRank: get('ivRank'), ivPctl: get('ivPctl'), ivChg: get('ivChg'),
      pcv:    get('pcv'),
    };
    if ([row.close, row.high, row.low, row.vix].some(v => !isFinite(v))) {
      tr.style.background = 'rgba(180, 50, 50, 0.15)';
      setTimeout(() => tr.style.background = '', 800);
      return;
    }
    upsertRow(row);
  } else if (e.target.classList.contains('skip')) {
    dismissMissing(date);
    recalc();
  }
});

// Auto-fetch Open from Yahoo Finance
document.getElementById('fetchOpenBtn').addEventListener('click', async () => {
  const dateInput = document.getElementById('entryDate');
  const date = dateInput.value;
  const fb = document.getElementById('entryFeedback');
  if (!date) {
    fb.className = 'feedback err';
    fb.textContent = 'Selecciona primero una fecha.';
    return;
  }
  const btn = document.getElementById('fetchOpenBtn');
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = '⏳';
  fb.className = 'feedback';
  fb.textContent = 'Consultando Yahoo Finance…';
  try {
    const ohlc = await fetchYahooOHLC('^GSPC', date);
    if (!isFinite(ohlc.open)) throw new Error('Open no disponible');
    document.getElementById('entryOpen').value = ohlc.open.toFixed(2);
    if (ohlc.actualDate && ohlc.actualDate !== date) {
      fb.className = 'feedback err';
      fb.textContent = `⚠ ${date} no tiene datos (¿festivo?). Usado el más reciente: ${ohlc.actualDate} → Open ${ohlc.open.toFixed(2)}`;
    } else {
      fb.textContent = `✓ Open de ${date} obtenido de Yahoo: ${ohlc.open.toFixed(2)}`;
    }
  } catch (e) {
    console.error('[Yahoo fetch]', e);
    fb.className = 'feedback err';
    fb.textContent = `✗ ${e.message}. Métela manualmente desde Barchart.`;
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
});

document.getElementById('reload').addEventListener('click', recalc);
// Auto-recalc when any regime input changes
document.getElementById('regimeTable').addEventListener('change', recalc);
document.getElementById('initialCapital').addEventListener('change', recalc);

// Compression panel: any change → read state → recalc
document.getElementById('compressionTable').addEventListener('change', () => {
  readCompressionVarsFromUI();
  recalc();
});
['cParamLambda','cParamCmin','cParamCmax','cParamVref','cParamShift'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    readCompressionParamsFromUI();
    recalc();
  });
});
document.getElementById('cRunOpt').addEventListener('click', runCompressionOptimization);

// Data view modal
function openDataModal() {
  const modal = document.getElementById('dataModal');
  if (!modal) {
    console.error('[Data view] #dataModal not found in DOM!');
    alert('Error: el modal #dataModal no existe. Probablemente caché del navegador. Pulsa Ctrl+F5.');
    return;
  }
  console.log('[Data view] Modal element found, currentRows:', currentRows ? currentRows.length + ' filas' : 'null');
  modal.style.display = 'flex';
  renderDataView();
}
function closeDataModal() {
  const modal = document.getElementById('dataModal');
  if (modal) modal.style.display = 'none';
}
function renderDataView() {
  const content = document.getElementById('dataModalContent');
  const countEl = document.getElementById('dataModalCount');
  try {
    _renderDataViewInner(content, countEl);
  } catch (err) {
    console.error('[Data view] FATAL render error:', err);
    content.innerHTML = `
      <div style="background:#fee2e2;color:#7a1010;padding:16px;border:2px solid #b73232;border-radius:6px;font-family:monospace;font-size:12px">
        <b style="font-size:14px">❌ Error al renderizar la tabla:</b><br><br>
        <b>${err.name}:</b> ${err.message}<br><br>
        <b>Stack:</b>
        <pre style="white-space:pre-wrap;background:#fff;padding:10px;border-radius:3px;margin-top:6px;font-size:10px">${err.stack || '(sin stack)'}</pre>
      </div>`;
    countEl.textContent = `Error: ${err.message}`;
  }
}

function _renderDataViewInner(content, countEl) {
  console.log('[Data view] renderDataView called');
  console.log('[Data view] typeof currentRows:', typeof currentRows);
  console.log('[Data view] currentRows is null?', currentRows === null);
  console.log('[Data view] currentRows length:', currentRows ? currentRows.length : 'N/A');
  if (currentRows && currentRows.length > 0) {
    console.log('[Data view] First row:', currentRows[0]);
    console.log('[Data view] Last row:', currentRows[currentRows.length - 1]);
  }

  const manual = loadEntries();

  // ALWAYS show a diagnostic banner so we can see the state regardless
  const diag = `
    <div style="padding:10px 14px;background:#fff7df;color:#5a4a18;border:1px solid #c9a227;border-radius:5px;margin-bottom:10px;font-size:12px">
      <b>📊 Estado:</b>
      <code style="background:#fff;padding:1px 5px;border-radius:3px">currentRows = ${currentRows === null ? 'null' : (currentRows === undefined ? 'undefined' : currentRows.length + ' filas')}</code> ·
      <code style="background:#fff;padding:1px 5px;border-radius:3px">localStorage = ${Object.keys(manual).length} entradas manuales</code>
    </div>
  `;

  if (!currentRows || currentRows.length === 0) {
    let extra = '';
    if (Object.keys(manual).length > 0) {
      extra = `
        <div style="margin-top:14px;padding:12px;background:#e8f1fa;border-radius:5px;font-size:12px;color:#0c1f33">
          <b>Tienes ${Object.keys(manual).length} entradas guardadas en localStorage:</b>
          <pre style="margin-top:6px;white-space:pre-wrap;font-size:11px;background:#fff;padding:8px;border-radius:3px;max-height:300px;overflow:auto">${JSON.stringify(manual, null, 2)}</pre>
        </div>
      `;
    }
    content.innerHTML = diag + `
      <div style="padding:30px;text-align:center;color:#444;background:#fff;border-radius:5px">
        <div style="font-size:30px">📭</div>
        <div style="margin-top:6px;font-size:14px"><b>currentRows está vacío.</b></div>
        <div style="font-size:12px;margin-top:6px;color:#666">Esto significa que el CSV no se ha cargado en memoria, aunque las entradas manuales sigan guardadas en el navegador.</div>
        <div style="font-size:12px;margin-top:6px;color:#666">Recarga la página y vuelve a seleccionar el CSV con el selector "CSV file" en Controls.</div>
      </div>
      ${extra}`;
    countEl.textContent = '0 filas';
    return;
  }

  const filter = document.getElementById('dataModalFilter').value;
  const lastDate = currentRows[currentRows.length - 1].date;
  const cutoffDate = new Date(lastDate); cutoffDate.setDate(cutoffDate.getDate() - 30);
  const cutoffISO = cutoffDate.toISOString().slice(0, 10);
  console.log('[Data view] filter:', filter, '· cutoff:', cutoffISO);

  let rows = currentRows.slice().reverse();
  if (filter === 'recent')        rows = rows.filter(r => r.date >= cutoffISO);
  else if (filter === 'manual')   rows = rows.filter(r => manual[r.date]);
  else if (filter === 'missing')  rows = rows.filter(r =>
    !isFinite(r.iv) || !isFinite(r.hv) || !isFinite(r.ivPctl) || !isFinite(r.ivChg) || !isFinite(r.pcv));
  console.log('[Data view] rows after filter:', rows.length);

  const total = currentRows.length;
  const shown = rows.length;
  const manualCount = Object.keys(manual).length;
  countEl.textContent = `${shown} de ${total} filas mostradas · ${manualCount} entradas manuales`;

  if (rows.length === 0) {
    content.innerHTML = diag + `
      <div style="padding:30px;text-align:center;color:#444;background:#fff;border-radius:5px">
        <div style="font-size:30px">🔎</div>
        <div style="margin-top:6px">El filtro <b>"${filter}"</b> no devuelve ninguna fila.</div>
        <div style="font-size:11px;margin-top:6px;color:#666">Cambia a "Todas las filas" para ver el dataset completo.</div>
      </div>`;
    return;
  }

  const cell = (v, dec = 2) => {
    // Defensive: coerce strings/null/undefined to number safely
    const num = (v === null || v === undefined || v === '') ? NaN : Number(v);
    return isFinite(num)
      ? `<td style="background:inherit;color:#000;padding:3px 8px;border-bottom:1px solid #ddd;text-align:right;font-variant-numeric:tabular-nums">${num.toFixed(dec)}</td>`
      : `<td style="background:inherit;color:#bbb;padding:3px 8px;border-bottom:1px solid #ddd;text-align:right">—</td>`;
  };

  const bodyParts = [];
  let badRows = 0;
  for (let i = 0; i < rows.length; i++) {
    try {
      const r = rows[i];
      if (!r || typeof r !== 'object') {
        badRows++;
        console.error('[Data view] Row', i, 'is not an object:', r);
        continue;
      }
      const isManual = !!manual[r.date];
      const num = v => (v === null || v === undefined || v === '') ? NaN : Number(v);
      const hasMissing = !isFinite(num(r.iv)) || !isFinite(num(r.hv)) || !isFinite(num(r.ivPctl)) || !isFinite(num(r.ivChg)) || !isFinite(num(r.pcv));
      const rowBg = isManual ? '#fff7df' : (hasMissing ? '#fdf3ec' : '#fff');
      const dateColor = isManual ? '#8a6d10' : '#0c1f33';
      // Botón editar SIEMPRE disponible (filas CSV o manuales).
      // Al editar una fila de CSV, se crea un override en localStorage.
      const editBtn = ` <button class="edit-entry-btn" data-date="${r.date}" style="margin-left:6px;background:${isManual ? '#c9a227' : '#cfe1f2'};color:${isManual ? '#0a0a0a' : '#0c1f33'};border:none;border-radius:3px;padding:1px 6px;font-size:10px;cursor:pointer;font-weight:600" title="${isManual ? 'Editar entrada manual' : 'Crear override de fila CSV'}">✏️ Editar</button>`;
      bodyParts.push(`<tr style="background:${rowBg}">
        <td style="background:${rowBg};color:${dateColor};padding:3px 8px;border-bottom:1px solid #ddd;font-weight:500">${r.date || '(sin fecha)'}${isManual ? ' ✎' : ''}${editBtn}</td>
        ${cell(r.close)}${cell(r.high)}${cell(r.low)}${cell(r.vix)}
        ${cell(r.iv, 3)}${cell(r.hv, 3)}
        ${cell(r.ivRank, 1)}${cell(r.ivPctl, 1)}${cell(r.ivChg, 4)}
        ${cell(r.pcv, 2)}
      </tr>`);
    } catch (rowErr) {
      badRows++;
      console.error('[Data view] Error rendering row', i, ':', rowErr, rows[i]);
    }
  }
  const body = bodyParts.join('');
  console.log('[Data view] Rendered', bodyParts.length, 'rows ·', badRows, 'bad rows skipped · body length:', body.length);

  const errBanner = badRows > 0
    ? `<div style="padding:8px 12px;background:#fee2e2;color:#7a1010;border:1px solid #b73232;border-radius:5px;margin-bottom:8px;font-size:12px">⚠ ${badRows} filas no pudieron renderizarse — ver consola.</div>`
    : '';

  content.innerHTML = diag + errBanner + `
    <table style="border-collapse:separate;border-spacing:0;width:100%;font-size:12px;background:#fff">
      <thead>
        <tr>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:left;position:sticky;top:0">Date</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">Close</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">High</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">Low</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">VIX</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">IV</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">HV</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">IVR</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">IVP</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">IVCHG</th>
          <th style="background:#143a64;color:#fff;padding:6px 8px;text-align:right;position:sticky;top:0">PCV</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}
// Pre-fill the entry form with an existing row (manual override OR CSV-only).
// Manual overrides take priority; otherwise falls back to the row in currentRows.
function populateFormFromEntry(date) {
  const entries = loadEntries();
  const entry = entries[date];
  let source = 'manual';
  let data = entry;
  if (!data) {
    // Try to find the row in currentRows (CSV-only row)
    const csvRow = currentRows && currentRows.find(r => r.date === date);
    if (!csvRow) {
      console.warn('[Edit] No data found for', date);
      return;
    }
    data = csvRow;
    source = 'csv';
  }
  const setVal = (id, v) => {
    document.getElementById(id).value = isFinite(v) ? v : '';
  };
  document.getElementById('entryDate').value = date;
  setVal('entryOpen',   data.open);
  setVal('entryClose',  data.close);
  setVal('entryHigh',   data.high);
  setVal('entryLow',    data.low);
  setVal('entryVix',    data.vix);
  setVal('entryIv',     data.iv);
  setVal('entryHv',     data.hv);
  setVal('entryIvr',    data.ivRank);
  setVal('entryIvp',    data.ivPctl);
  setVal('entryIvchg',  data.ivChg);
  setVal('entryPcv',    data.pcv);
  // Visual feedback
  const fb = document.getElementById('entryFeedback');
  fb.className = 'feedback';
  fb.textContent = source === 'manual'
    ? `✏️ Editando entrada manual ${date}. Modifica y pulsa Add para sobrescribir.`
    : `✏️ Editando fila CSV ${date}. Al pulsar Add, se creará un override en localStorage que prevalecerá sobre el CSV.`;
  document.querySelector('header').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Event delegation — robust against any DOM-loading order issues
document.addEventListener('click', (e) => {
  if (e.target.closest('#openDataView')) {
    console.log('[Data view] Click detected, opening modal');
    openDataModal();
    return;
  }
  if (e.target.closest('#dataModalClose')) {
    closeDataModal();
    return;
  }
  if (e.target.closest('.edit-entry-btn')) {
    const btn = e.target.closest('.edit-entry-btn');
    populateFormFromEntry(btn.dataset.date);
    closeDataModal();
    return;
  }
  if (e.target.id === 'dataModal') {
    closeDataModal();
  }
});
document.addEventListener('change', (e) => {
  if (e.target.id === 'dataModalFilter') renderDataView();
});

// Quant Agent handlers
document.getElementById('agentCard').addEventListener('click', openAgentModal);
document.getElementById('agentModalClose').addEventListener('click', closeAgentModal);

// Vol Analyst handlers
document.getElementById('analystCard').addEventListener('click', openAnalystModal);
document.getElementById('analystModalClose').addEventListener('click', closeAnalystModal);

// Option chain CSV upload
document.getElementById('chainCSV').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleChainUpload(file);
});
document.getElementById('chainSavedList').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-chain-date]');
  if (!btn) return;
  const date = btn.dataset.chainDate;
  if (btn.classList.contains('view-btn')) viewChain(date);
  if (btn.classList.contains('del-btn'))  deleteChain(date);
});

// Render the saved chains list at startup
renderChainsList();

// Favorites handlers
document.getElementById('favoritesCard').addEventListener('click', openFavoritesModal);
document.getElementById('favoritesModalClose').addEventListener('click', closeFavoritesModal);
document.getElementById('saveFavBtn').addEventListener('click', () => {
  const name = document.getElementById('newFavName').value;
  saveCurrentAsFavorite(name);
});
document.getElementById('newFavName').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    saveCurrentAsFavorite(e.target.value);
  }
});
document.getElementById('favoritesModal').addEventListener('click', (e) => {
  if (e.target.id === 'favoritesModal') closeFavoritesModal();
  if (e.target.classList.contains('fav-apply'))   applyFavorite(e.target.dataset.id);
  if (e.target.classList.contains('fav-delete'))  deleteFavorite(e.target.dataset.id);
  if (e.target.classList.contains('fav-expand')) {
    const id = e.target.dataset.id;
    const det = document.getElementById('fav-details-' + id);
    const isHidden = det.style.display === 'none';
    det.style.display = isHidden ? '' : 'none';
    e.target.textContent = isHidden ? '? Ocultar variables' : '? Ver todas las variables';
  }
});
document.getElementById('favoritesModal').addEventListener('change', (e) => {
  if (e.target.classList.contains('fav-name')) renameFavorite(e.target.dataset.id, e.target.value);
});

// Jefe de mesa handlers
document.getElementById('jefeCard').addEventListener('click', openJefeModal);
document.getElementById('jefeModalClose').addEventListener('click', closeJefeModal);
document.getElementById('jefeRunBtn').addEventListener('click', runJefeDeMesa);
document.getElementById('jefeModal').addEventListener('click', (e) => {
  if (e.target.id === 'jefeModal') closeJefeModal();
  if (e.target.classList.contains('apply-btn') && e.target.dataset.jefeRank !== undefined) {
    applyJefeConfig(e.target.dataset.jefeRank);
  }
});

// Compressor Analyst handlers
document.getElementById('compressorCard').addEventListener('click', openCompressorModal);
document.getElementById('compressorModalClose').addEventListener('click', closeCompressorModal);
document.getElementById('compressorRunBtn').addEventListener('click', runCompressorAgent);
document.getElementById('compressorModal').addEventListener('click', (e) => {
  if (e.target.id === 'compressorModal') closeCompressorModal();
  if (e.target.classList.contains('apply-btn') && e.target.dataset.compressorPeriod) {
    applyCompressorProposal(e.target.dataset.compressorPeriod);
  }
});
document.getElementById('analystModal').addEventListener('click', (e) => {
  if (e.target.id === 'analystModal') closeAnalystModal();
  if (e.target.classList.contains('apply-btn') && e.target.dataset.mixPeriod) {
    applyMixProposal(e.target.dataset.mixPeriod);
  }
});
// Tab navigation
document.querySelectorAll('.modal-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.modal-tab').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.dataset.tab === tab));
    if (tab === 'vix') drawVixAnalystChart();
  });
});
// MIX run button
document.getElementById('mixRunBtn').addEventListener('click', runMix);
// CSV uploads
wireEventUpload('upcomingCSV',   EVENTS_UPCOMING_KEY,   () => { renderUpcomingEvents(); checkEventBanner(); });
wireEventUpload('historicalCSV', EVENTS_HISTORICAL_KEY, () => { renderHistoricalCorrelation(); });
// Esc closes whichever modal is open
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (document.getElementById('analystModal').style.display === 'flex')    closeAnalystModal();
  if (document.getElementById('dataModal').style.display === 'flex')       closeDataModal();
  if (document.getElementById('compressorModal').style.display === 'flex') closeCompressorModal();
  if (document.getElementById('favoritesModal').style.display === 'flex')  closeFavoritesModal();
  if (document.getElementById('jefeModal').style.display === 'flex')       closeJefeModal();
});
document.getElementById('agentModal').addEventListener('click', (e) => {
  if (e.target.id === 'agentModal') closeAgentModal();
  if (e.target.classList.contains('apply-btn')) applyAgentProposal(e.target.dataset.period);
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('agentModal').style.display === 'flex') closeAgentModal();
});

// Render the desglose table — historical operations with strikes & outcome.
function renderDesglose() {
  const table = document.getElementById('desgloseTable');
  const summary = document.getElementById('desgloseSummary');
  if (!currentRows) {
    table.innerHTML = '';
    summary.textContent = 'No hay datos cargados.';
    return;
  }

  const lookback = parseInt(document.getElementById('lookback').value, 10);
  const regimes = readRegimes();
  const { bands, nextBand } = computeBands(currentRows, regimes, lookback);

  const rows = [];
  let cleanWins = 0, touched = 0, losses = 0, noTrade = 0;

  // Historical operations
  for (let i = 1; i < currentRows.length; i++) {
    const b = bands[i];
    const r = currentRows[i];
    const prev = currentRows[i - 1];
    if (!b) continue;

    const callStrike = Math.ceil(b.upper / 5) * 5;
    const putStrike  = Math.floor(b.lower / 5) * 5;
    const distCall = (callStrike - prev.close) / prev.close * 100;
    const distPut  = (prev.close - putStrike)  / prev.close * 100;
    const callTouched = isFinite(r.high) && r.high >= callStrike;
    const putTouched  = isFinite(r.low)  && r.low  <= putStrike;

    let cls, status;
    if (!b.trade) { cls = 'notrade-row'; status = 'NO-TRADE'; noTrade++; }
    else if (r.close >= b.upper) { cls = 'loss-row'; status = '✗ Call hit'; losses++; }
    else if (r.close <= b.lower) { cls = 'loss-row'; status = '✗ Put hit';  losses++; }
    else if (callTouched && putTouched) { cls = 'touched-row'; status = '⚠ Tocó ambos'; touched++; }
    else if (callTouched)               { cls = 'touched-row'; status = '⚠ Tocó call';  touched++; }
    else if (putTouched)                { cls = 'touched-row'; status = '⚠ Tocó put';   touched++; }
    else                                { cls = 'win-row';     status = '✓ Limpio';     cleanWins++; }

    rows.push({ cls, date: r.date, prevClose: prev.close, callStrike, distCall, putStrike, distPut, close: r.close, status,
                callTouched, putTouched });
  }

  // Pending next-day operation
  let pendingHtml = '';
  if (nextBand && nextBand.trade) {
    const last = currentRows[currentRows.length - 1];
    const callStrike = Math.ceil(nextBand.upper / 5) * 5;
    const putStrike  = Math.floor(nextBand.lower / 5) * 5;
    const distCall = (callStrike - last.close) / last.close * 100;
    const distPut  = (last.close - putStrike)  / last.close * 100;
    // Compute "tomorrow" date (next weekday for display)
    const d = new Date(last.date); do { d.setDate(d.getDate() + 1); } while (d.getDay() === 0 || d.getDay() === 6);
    const tomorrow = d.toISOString().slice(0, 10);
    pendingHtml = `<tr class="pending-row">
      <td><b>${tomorrow}</b> ⏳</td>
      <td>${last.close.toFixed(2)}</td>
      <td class="strike-call">${callStrike}</td>
      <td>+${distCall.toFixed(3)}%</td>
      <td class="strike-put">${putStrike}</td>
      <td>−${distPut.toFixed(3)}%</td>
      <td><i>pendiente</i></td>
      <td><i>pendiente</i></td>
    </tr>`;
  }

  // Render newest first (excluding pending which goes on top)
  rows.reverse();
  const bodyRows = rows.map(r => `
    <tr class="${r.cls}">
      <td>${r.date}</td>
      <td>${r.prevClose.toFixed(2)}</td>
      <td class="strike-call">${r.callStrike}</td>
      <td>+${r.distCall.toFixed(3)}%</td>
      <td class="strike-put">${r.putStrike}</td>
      <td>−${r.distPut.toFixed(3)}%</td>
      <td>${r.close.toFixed(2)}</td>
      <td>${r.status}</td>
    </tr>`).join('');

  table.innerHTML = `
    <thead><tr>
      <th>Fecha</th>
      <th>Cierre anterior</th>
      <th>Call vendida</th>
      <th>% dist call</th>
      <th>Put vendida</th>
      <th>% dist put</th>
      <th>Cierre día</th>
      <th>Resultado</th>
    </tr></thead>
    <tbody>${pendingHtml}${bodyRows}</tbody>`;

  const totalOp = cleanWins + touched + losses;
  const wrStrict = totalOp > 0 ? (cleanWins / totalOp * 100).toFixed(1) : '—';
  const wrClose  = totalOp > 0 ? ((cleanWins + touched) / totalOp * 100).toFixed(1) : '—';
  summary.innerHTML = `
    <b>${rows.length}</b> días totales ·
    <b style="color:var(--good)">${cleanWins} limpios</b> ·
    <b style="color:#b06000">${touched} tocados</b> ·
    <b style="color:var(--bad)">${losses} losses</b> ·
    <b>${noTrade} no-trade</b> ·
    WR estricto: <b>${wrStrict}%</b> · WR close: <b>${wrClose}%</b>
    ${pendingHtml ? '· operación pendiente para próxima sesión arriba en amarillo' : ''}
  `;
}

// Toggle handler for the desglose panel
document.getElementById('toggleDesglose').addEventListener('click', () => {
  const panel = document.getElementById('desglosePanel');
  const btn = document.getElementById('toggleDesglose');
  if (panel.style.display === 'none') {
    panel.style.display = '';
    btn.textContent = '📋 Ocultar desglose de operaciones';
    renderDesglose();
  } else {
    panel.style.display = 'none';
    btn.textContent = '📋 Mostrar desglose de operaciones';
  }
});

// Toggle for trade log
document.getElementById('toggleTrades').addEventListener('click', () => {
  const panel = document.getElementById('tradesPanel');
  const btn   = document.getElementById('toggleTrades');
  if (panel.style.display === 'none') {
    panel.style.display = '';
    btn.textContent = 'Hide trade log';
    if (window._lastPnL) renderTradesTable(window._lastPnL);
  } else {
    panel.style.display = 'none';
    btn.textContent = 'Show trade log';
  }
});

// ==========================================================================
// COMPRESSOR ANALYST — exhaustive search of compression model parameters
// ==========================================================================
// Grid (deterministic order):
//   λ:    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]                     6 values
//   Cmin: [0.3, 0.4, 0.5, 0.6]                               4 values (≥ 0.3)
//   Cmax: [0.55, 0.65, 0.75, 0.85]                           4 values (≤ 0.85)
//   Vref: [14, 16, 18, 20, 22]                               5 values
//   Variable subsets: 2^6 = 64 (binary mask, bit i = COMPRESSION_VARS[i])
// With Cmax > Cmin constraint: ≈ 21,000 combinations per window.
const COMPRESSOR_GRID = {
  lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
  cmin:   [0.3, 0.4, 0.5, 0.6],
  cmax:   [0.55, 0.65, 0.75, 0.85],
  vref:   [14, 16, 18, 20, 22],
};

// Score a single (λ, Cmin, Cmax, Vref, mask) config on a window.
// Inline, no allocations. Reads sigmas from baseBands and current weights from compressionVars.
function scoreCompressorConfig(rows, baseBands, baseRegimes, windowDays, lambda, cmin, cmax, vref, mask) {
  const start = Math.max(1, rows.length - windowDays);
  const N = rows.length;
  let score = 0;
  for (let i = start; i < N; i++) {
    const b = baseBands[i];
    if (!b) continue;
    const prev = rows[i - 1];
    const reg = regimeFor(prev.vix, baseRegimes);
    if (!isDayAccepted(rows[i], reg)) continue;

    // Compute Z over active variables (per mask)
    let Z = 0;
    if (mask & 1) {  // iv_hv
      if (isFinite(prev.iv) && isFinite(prev.hv) && prev.iv > 0 && prev.hv > 0) {
        Z += compressionVars[0].weight * Math.log(prev.iv / prev.hv);
      }
    }
    if (mask & 2) {  // iv_rank
      if (isFinite(prev.ivRank)) Z += compressionVars[1].weight * (prev.ivRank - 50) / 50;
    }
    if (mask & 4) {  // iv_pctl
      if (isFinite(prev.ivPctl)) Z += compressionVars[2].weight * (prev.ivPctl - 50) / 50;
    }
    if (mask & 8) {  // vix
      if (isFinite(prev.vix)) Z += compressionVars[3].weight * (vref - prev.vix) / vref;
    }
    if (mask & 16) {  // iv_chg
      if (isFinite(prev.ivChg)) Z += compressionVars[4].weight * prev.ivChg;
    }
    if (mask & 32) {  // pcv
      if (isFinite(prev.pcv) && prev.pcv > 0) Z += compressionVars[5].weight * Math.log(prev.pcv);
    }

    const Craw = 1 / (1 + Math.exp(-lambda * Z));
    const base = Math.max(cmin, Math.min(cmax, Craw));
    const delta = compressionParams.shiftFactor * (cmax - cmin);
    let Ccall = base, Cput = base;
    if (prev.trend === 'up')        { Ccall = base + delta; Cput = base - delta; }
    else if (prev.trend === 'down') { Ccall = base - delta; Cput = base + delta; }
    Ccall = Math.max(cmin, Math.min(cmax, Ccall));
    Cput  = Math.max(cmin, Math.min(cmax, Cput));

    const upper = prev.close * (1 + Ccall * reg.k * reg.upperMult * b.sigUp);
    const lower = prev.close * (1 - Cput  * reg.k * reg.lowerMult * b.sigDn);
    if (rows[i].close >= upper || rows[i].close <= lower) score += LOSS_PNL;
    else score += WIN_PNL;
  }
  return score;
}

async function searchCompressorWindow(rows, baseBands, baseRegimes, windowDays, baselineScore, onProgress) {
  let bestScore = baselineScore;
  let bestConfig = null;
  const Lg = COMPRESSOR_GRID.lambda, Mg = COMPRESSOR_GRID.cmin, Xg = COMPRESSOR_GRID.cmax, Vg = COMPRESSOR_GRID.vref;
  const totalOuter = Lg.length * Mg.length * Xg.length * Vg.length;
  let outerDone = 0;

  for (const lambda of Lg) {
    for (const cmin of Mg) {
      for (const cmax of Xg) {
        if (cmax <= cmin + 0.04) { outerDone += Vg.length; continue; }
        for (const vref of Vg) {
          for (let mask = 0; mask < 64; mask++) {
            const s = scoreCompressorConfig(rows, baseBands, baseRegimes, windowDays, lambda, cmin, cmax, vref, mask);
            if (s > bestScore) { bestScore = s; bestConfig = { lambda, cmin, cmax, vref, mask }; }
          }
          outerDone++;
        }
        if (onProgress) onProgress(outerDone / totalOuter);
        await new Promise(r => setTimeout(r, 0));
      }
    }
  }
  return { bestScore, bestConfig };
}

async function runCompressorAgent() {
  if (!currentRows) return;
  const btn = document.getElementById('compressorRunBtn');
  const status = document.getElementById('compressorStatus');
  const bar = document.getElementById('compressorProgressBar');
  const fill = document.getElementById('compressorProgressFill');
  btn.disabled = true;
  bar.style.display = 'block';
  fill.style.width = '0%';
  status.style.color = '';

  try {
    const lookback = parseInt(document.getElementById('lookback').value, 10);
    const baseRegimes = readRegimes();
    const { bands: baseBands } = computeBands(currentRows, baseRegimes, lookback);

    // Baseline (current state) per window
    const baseScore = (windowDays) => {
      const start = Math.max(1, currentRows.length - windowDays);
      let s = 0;
      for (let i = start; i < currentRows.length; i++) {
        const b = baseBands[i];
        if (!b || !b.trade) continue;
        if (currentRows[i].close >= b.upper || currentRows[i].close <= b.lower) s += LOSS_PNL;
        else s += WIN_PNL;
      }
      return s;
    };

    const periods = [
      { key: '1m', label: 'Mejora 1 — 1 mes',   days: 21  },
      { key: '3m', label: 'Mejora 2 — 3 meses', days: 63  },
      { key: '6m', label: 'Mejora 3 — 6 meses', days: 126 },
    ];

    document.getElementById('compressorResults').innerHTML = '';
    _compressorCache.clear();

    const t0 = performance.now();
    for (let pIdx = 0; pIdx < periods.length; pIdx++) {
      const p = periods[pIdx];
      const windowBaseline = baseScore(p.days);
      status.textContent = `[${pIdx + 1}/3 · ${p.label}] explorando ~21.500 combos…`;
      const tWin = performance.now();
      const result = await searchCompressorWindow(
        currentRows, baseBands, baseRegimes, p.days, windowBaseline,
        (pct) => {
          const overall = (pIdx + pct) / periods.length * 100;
          fill.style.width = overall.toFixed(1) + '%';
        }
      );
      const elapsed = ((performance.now() - tWin) / 1000).toFixed(1);
      status.textContent = `${p.label} listo (${elapsed}s, mejor: $${result.bestScore.toFixed(0)} vs base $${windowBaseline.toFixed(0)})`;
      renderCompressorCard(p, windowBaseline, result);
      await new Promise(r => setTimeout(r, 0));
    }
    fill.style.width = '100%';
    status.textContent = `✓ Completado en ${((performance.now() - t0) / 1000).toFixed(1)}s.`;
    setTimeout(() => { bar.style.display = 'none'; fill.style.width = '0%'; }, 1800);
  } catch (err) {
    console.error('runCompressorAgent error:', err);
    status.style.color = '#b73232';
    status.textContent = `✗ Error: ${err.message || err}`;
  } finally {
    btn.disabled = false;
  }
}

const _compressorCache = new Map();

function renderCompressorCard(period, baselineScore, result) {
  const container = document.getElementById('compressorResults');
  const card = document.createElement('div');
  card.className = 'mix-card';

  const fmt$ = v => (v >= 0 ? '+$' : '−$') + Math.abs(v).toFixed(0);

  if (!result.bestConfig) {
    card.innerHTML = `
      <h4>${period.label}</h4>
      <div style="font-size:12px;color:var(--ink-soft)">
        Ninguna combinación supera la configuración actual (baseline: ${fmt$(baselineScore)}).
        <br>El modelo actual ya es óptimo para esta ventana, o necesitas tunear pesos antes.
      </div>`;
    container.appendChild(card);
    return;
  }

  const cfg = result.bestConfig;
  const delta = result.bestScore - baselineScore;
  const activeVars = COMPRESSION_VARS
    .map((v, i) => (cfg.mask & (1 << i)) ? v.label : null)
    .filter(Boolean);

  _compressorCache.set(period.key, cfg);

  card.innerHTML = `
    <h4>${period.label}</h4>
    <div class="iter-info">
      <span>Baseline → Mejor</span>
      <span>${fmt$(baselineScore)} → <b style="color:var(--good)">${fmt$(result.bestScore)}</b> (${fmt$(delta)})</span>
    </div>
    <div class="threshold-block">
      <div class="th-label">Parámetros del modelo</div>
      <div class="th-vals">
        λ = <b>${cfg.lambda}</b> · C<sub>min</sub> = <b>${cfg.cmin}</b> · C<sub>max</sub> = <b>${cfg.cmax}</b> · V<sub>ref</sub> = <b>${cfg.vref}</b>
      </div>
    </div>
    <div class="threshold-block">
      <div class="th-label">Variables activas (${activeVars.length}/6)</div>
      <div class="th-vals">
        ${activeVars.length === 0
          ? '<i style="color:var(--ink-soft)">Ninguna — sólo aplica el clamp [Cmin, Cmax]</i>'
          : activeVars.map(v => `<span class="changed">${v}</span>`).join(' · ')}
      </div>
    </div>
    <button class="apply-btn" data-compressor-period="${period.key}">Aplicar esta configuración</button>
  `;
  container.appendChild(card);
}

function applyCompressorProposal(periodKey) {
  const cfg = _compressorCache.get(periodKey);
  if (!cfg) return;
  // Update DOM inputs
  document.getElementById('cParamLambda').value = cfg.lambda;
  document.getElementById('cParamCmin').value   = cfg.cmin;
  document.getElementById('cParamCmax').value   = cfg.cmax;
  document.getElementById('cParamVref').value   = cfg.vref;
  // Update internal state
  compressionParams.lambda = cfg.lambda;
  compressionParams.Cmin   = cfg.cmin;
  compressionParams.Cmax   = cfg.cmax;
  compressionParams.Vref   = cfg.vref;
  // Update active flags per mask
  for (let i = 0; i < 6; i++) {
    compressionVars[i].active = (cfg.mask & (1 << i)) !== 0;
  }
  renderCompressionPanel();
  closeCompressorModal();
  recalc();
}

function openCompressorModal()  { document.getElementById('compressorModal').style.display = 'flex'; }
function closeCompressorModal() { document.getElementById('compressorModal').style.display = 'none'; }

// Safe wrapper around Plotly — never crashes the rest of the pipeline
// if Plotly failed to load from CDN.
function safePlotly(elementId, traces, layout, config) {
  if (typeof Plotly === 'undefined') {
    const el = document.getElementById(elementId);
    if (el && !el.dataset.plotlyWarned) {
      el.dataset.plotlyWarned = '1';
      el.innerHTML = '<div style="padding:20px;text-align:center;color:#b73232;background:#fee;border:1px solid #b73232;border-radius:5px;font-size:13px"><b>⚠ Plotly no disponible</b><br>Refresca la página cuando vuelvas a tener internet.</div>';
    }
    return;
  }
  try {
    Plotly.newPlot(elementId, traces, layout, config);
  } catch (e) {
    console.error('[Plotly]', elementId, 'render error:', e);
  }
}

// ---- Theme toggle (light/dark) ------------------------------------------
const THEME_KEY = 'spx-vix-theme-v1';
function applyTheme(theme) {
  const t = 'light';
  document.documentElement.dataset.theme = t;
  document.querySelectorAll('.theme-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.theme === t);
  });
  try { localStorage.setItem(THEME_KEY, t); } catch (_) {}
}
applyTheme(localStorage.getItem(THEME_KEY) || 'light');
document.querySelectorAll('.theme-btn').forEach(b => {
  b.addEventListener('click', () => applyTheme(b.dataset.theme));
});

// Auto-load priority: cached CSV → fetch (http only) → manual entries only
// ---- Development notes ---------------------------------------------------
const DEV_NOTES_KEY = 'spx-dev-notes-v1';
let devNotesState = { notes: [], selectedId: null };

function devNotesEscape(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function loadDevNotes() {
  try {
    const parsed = JSON.parse(localStorage.getItem(DEV_NOTES_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function saveDevNotes(notes) {
  localStorage.setItem(DEV_NOTES_KEY, JSON.stringify(notes, null, 2));
}

function devNoteStatusLabel(status) {
  return {
    idea: 'Idea',
    pending: 'Pendiente',
    in_progress: 'En desarrollo',
    done: 'Hecho'
  }[status] || 'Pendiente';
}

function createEmptyDevNote() {
  const now = new Date().toISOString();
  return {
    id: `note-${Date.now()}`,
    status: 'idea',
    title: '',
    body: '',
    codex: '',
    createdAt: now,
    updatedAt: now
  };
}

function getSelectedDevNote() {
  return devNotesState.notes.find(note => note.id === devNotesState.selectedId) || null;
}

function renderDevNotesList() {
  const list = document.getElementById('devNotesList');
  if (!list) return;
  const notes = devNotesState.notes.slice().sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')));
  if (!notes.length) {
    list.innerHTML = `
      <div class="dev-notes-empty">
        Todavía no hay pendientes. Crea una nota para guardar ideas de ML, mejoras del IC, reglas futuras o dudas que tengamos que revisar.
      </div>
    `;
    return;
  }
  list.innerHTML = notes.map(note => `
    <button type="button" class="dev-note-item ${note.id === devNotesState.selectedId ? 'active' : ''}" data-note-id="${devNotesEscape(note.id)}">
      <strong>${devNotesEscape(note.title || 'Sin título')}</strong>
      <span>${devNoteStatusLabel(note.status)} · ${note.updatedAt ? new Date(note.updatedAt).toLocaleDateString('es-ES') : ''}</span>
    </button>
  `).join('');
  list.querySelectorAll('.dev-note-item').forEach(btn => {
    btn.addEventListener('click', () => {
      devNotesState.selectedId = btn.dataset.noteId;
      renderDevNotes();
    });
  });
}

function fillDevNotesForm() {
  const note = getSelectedDevNote();
  const id = document.getElementById('devNoteId');
  const status = document.getElementById('devNoteStatus');
  const title = document.getElementById('devNoteTitleInput');
  const body = document.getElementById('devNoteBodyInput');
  const codex = document.getElementById('devNoteCodexInput');
  const meta = document.getElementById('devNotesMeta');
  const del = document.getElementById('devNoteDeleteBtn');
  if (!id || !status || !title || !body || !codex || !meta || !del) return;

  if (!note) {
    id.value = '';
    status.value = 'idea';
    title.value = '';
    body.value = '';
    codex.value = '';
    meta.textContent = 'Crea una nota nueva o selecciona una pendiente existente.';
    del.disabled = true;
    return;
  }
  id.value = note.id;
  status.value = note.status || 'idea';
  title.value = note.title || '';
  body.value = note.body || '';
  codex.value = note.codex || '';
  meta.textContent = `Creada: ${note.createdAt ? new Date(note.createdAt).toLocaleString('es-ES') : '-'} · Actualizada: ${note.updatedAt ? new Date(note.updatedAt).toLocaleString('es-ES') : '-'}`;
  del.disabled = false;
}

function renderDevNotes() {
  renderDevNotesList();
  fillDevNotesForm();
}

function openDevNotes() {
  const overlay = document.getElementById('devNotesOverlay');
  if (!overlay) return;
  devNotesState.notes = loadDevNotes();
  if (!devNotesState.selectedId && devNotesState.notes.length) {
    devNotesState.selectedId = devNotesState.notes[0].id;
  }
  renderDevNotes();
  overlay.classList.add('open');
  overlay.setAttribute('aria-hidden', 'false');
}

function closeDevNotes() {
  const overlay = document.getElementById('devNotesOverlay');
  if (!overlay) return;
  overlay.classList.remove('open');
  overlay.setAttribute('aria-hidden', 'true');
}

function saveDevNotesForm(event) {
  event.preventDefault();
  const id = document.getElementById('devNoteId');
  const status = document.getElementById('devNoteStatus');
  const title = document.getElementById('devNoteTitleInput');
  const body = document.getElementById('devNoteBodyInput');
  const codex = document.getElementById('devNoteCodexInput');
  if (!id || !status || !title || !body || !codex) return;
  const now = new Date().toISOString();
  let note = devNotesState.notes.find(item => item.id === id.value);
  if (!note) {
    note = createEmptyDevNote();
    devNotesState.notes.unshift(note);
  }
  note.status = status.value || 'idea';
  note.title = title.value.trim() || 'Sin título';
  note.body = body.value.trim();
  note.codex = codex.value.trim();
  note.updatedAt = now;
  if (!note.createdAt) note.createdAt = now;
  devNotesState.selectedId = note.id;
  saveDevNotes(devNotesState.notes);
  renderDevNotes();
}

function newDevNote() {
  const note = createEmptyDevNote();
  devNotesState.notes.unshift(note);
  devNotesState.selectedId = note.id;
  saveDevNotes(devNotesState.notes);
  renderDevNotes();
  const title = document.getElementById('devNoteTitleInput');
  if (title) title.focus();
}

function deleteDevNote() {
  const note = getSelectedDevNote();
  if (!note) return;
  if (!confirm(`¿Eliminar la nota "${note.title || 'Sin título'}"?`)) return;
  devNotesState.notes = devNotesState.notes.filter(item => item.id !== note.id);
  devNotesState.selectedId = devNotesState.notes[0]?.id || null;
  saveDevNotes(devNotesState.notes);
  renderDevNotes();
}

async function exportDevNotesSummary() {
  const notes = loadDevNotes();
  const text = notes.length
    ? notes.map((note, idx) => [
        `${idx + 1}. ${note.title || 'Sin título'} [${devNoteStatusLabel(note.status)}]`,
        `Usuario: ${note.body || '-'}`,
        `Codex: ${note.codex || '-'}`,
      ].join('\n')).join('\n\n')
    : 'No hay pendientes de desarrollo guardados.';
  try {
    await navigator.clipboard.writeText(text);
    const meta = document.getElementById('devNotesMeta');
    if (meta) meta.textContent = 'Resumen copiado al portapapeles.';
  } catch (_) {
    alert(text);
  }
}

function initDevNotes() {
  const open = document.getElementById('devNotesOpenBtn');
  const close = document.getElementById('devNotesCloseBtn');
  const overlay = document.getElementById('devNotesOverlay');
  const form = document.getElementById('devNotesForm');
  const add = document.getElementById('devNoteNewBtn');
  const del = document.getElementById('devNoteDeleteBtn');
  const exp = document.getElementById('devNotesExportBtn');
  if (!open || !close || !overlay || !form || !add || !del || !exp) return;
  open.addEventListener('click', openDevNotes);
  close.addEventListener('click', closeDevNotes);
  overlay.addEventListener('click', event => {
    if (event.target === overlay) closeDevNotes();
  });
  form.addEventListener('submit', saveDevNotesForm);
  add.addEventListener('click', newDevNote);
  del.addEventListener('click', deleteDevNote);
  exp.addEventListener('click', exportDevNotesSummary);
}

initDevNotes();

const AUTO_LOAD_NAMES = ['SPX-VIX.fin.csv', 'data.csv'];
(async () => {
  // 1. Cached CSV — works regardless of file:// vs http
  const cache = loadCSVCache();
  if (cache && cache.text) {
    try {
      currentRows = applyEntries(parseCSV(cache.text), loadEntries());
      console.log(`[CSV cache] Cargadas ${currentRows.length} filas de "${cache.filename}" (${((Date.now() - cache.savedAt)/86400000).toFixed(1)} días en caché)`);
      recalc();
      updateCacheStatus();
      return;
    } catch (e) {
      console.warn('[CSV cache] Cache corrupto, ignorando:', e);
      clearCSVCache();
    }
  }

  // 2. Auto-fetch (only works via http(s))
  for (const name of AUTO_LOAD_NAMES) {
    try {
      const r = await fetch(name);
      if (r.ok) {
        const text = await r.text();
        saveCSVCache(text, name);
        currentRows = applyEntries(parseCSV(text), loadEntries());
        recalc();
        updateCacheStatus();
        return;
      }
    } catch (_) { /* file:// or missing */ }
  }

  // 3. Manual entries only (no CSV available)
  const stored = loadEntries();
  if (Object.keys(stored).length) {
    currentRows = applyEntries([], stored);
  }
  recalc();
  updateCacheStatus();
})();

// ---- Auto-load OHLC fetched by GitHub Action --------------------------
const GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/Quique805/SPX-IC/main';

async function loadAutoFetchedOHLC() {
  if (!currentRows) {
    console.log('[Auto OHLC] currentRows no disponible aún, reintentando en 2s');
    setTimeout(loadAutoFetchedOHLC, 2000);
    return;
  }
  try {
    const url = `${GITHUB_RAW_BASE}/data/daily-ohlc.json?t=${Date.now()}`;
    const r = await fetch(url);
    if (!r.ok) {
      console.log('[Auto OHLC] No disponible aún (status ' + r.status + '). Esperando primer run del Action.');
      return;
    }
    const data = await r.json();
    if (!data.byDate) return;
    const hasOHLCValue = value =>
      value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
    let added = 0, filled = 0;
    const byDate = new Map(currentRows.map(row => [row.date, row]));
    for (const [d, info] of Object.entries(data.byDate)) {
      if (byDate.has(d)) {
        const row = byDate.get(d);
        if (!hasOHLCValue(row.open)  && hasOHLCValue(info.spx_open))  { row.open  = Number(info.spx_open);  filled++; }
        if (!hasOHLCValue(row.high)  && hasOHLCValue(info.spx_high))  { row.high  = Number(info.spx_high);  filled++; }
        if (!hasOHLCValue(row.low)   && hasOHLCValue(info.spx_low))   { row.low   = Number(info.spx_low);   filled++; }
        if (!hasOHLCValue(row.close) && hasOHLCValue(info.spx_close)) { row.close = Number(info.spx_close); filled++; }
        if (!hasOHLCValue(row.vix)   && hasOHLCValue(info.vix_close)) { row.vix   = Number(info.vix_close); filled++; }
      } else if (hasOHLCValue(info.spx_close)) {
        currentRows.push({
          date: d,
          open: hasOHLCValue(info.spx_open) ? Number(info.spx_open) : NaN,
          high: hasOHLCValue(info.spx_high) ? Number(info.spx_high) : NaN,
          low: hasOHLCValue(info.spx_low) ? Number(info.spx_low) : NaN,
          close: Number(info.spx_close),
          vix: hasOHLCValue(info.vix_close) ? Number(info.vix_close) : NaN,
          iv: NaN, hv: NaN, ivRank: NaN, ivPctl: NaN, ivChg: NaN, pcv: NaN,
        });
        added++;
      }
    }
    if (added > 0 || filled > 0) {
      currentRows.sort((a, b) => new Date(a.date) - new Date(b.date));
      enrichRows(currentRows);
      console.log(`[Auto OHLC] ${added} días nuevos, ${filled} campos rellenados desde GitHub`);
      recalc();
    } else {
      console.log('[Auto OHLC] Sin cambios respecto a datos existentes');
    }
    const remoteDates = Object.keys(data.byDate).sort();
    const localDates = currentRows.map(row => row.date).filter(Boolean).sort();
    console.log(`[Auto OHLC] Última fecha remota: ${remoteDates.at(-1) || '—'} · Última fecha dashboard: ${localDates.at(-1) || '—'}`);
  } catch (e) {
    console.warn('[Auto OHLC] Error:', e.message);
  }
}

setTimeout(loadAutoFetchedOHLC, 3000);

// ---- Auto-loaded chains from GitHub Action ------------------------------
async function loadAutoFetchedChains(kind = 'entry') {
  const isClose = kind === 'close';
  const listEl = document.getElementById(isClose ? 'autoCloseChainsList' : 'autoChainsList');
  const lastEl = document.getElementById('chainsLastUpdate');
  if (!listEl) return;

  const indexFile = isClose ? 'chains-close-index.json' : 'chains-index.json';
  const folder = isClose ? 'chains-close' : 'chains';
  const label = isClose ? 'Cierre CBOE (22:32 Madrid)' : 'Entrada CBOE delayed (16:07 Madrid)';

  try {
    const indexUrl = `${GITHUB_RAW_BASE}/data/${indexFile}?t=${Date.now()}`;
    const r = await fetch(indexUrl);
    if (!r.ok) {
      listEl.innerHTML = `<div style="font-size:12px;color:var(--ink-soft);text-align:center;padding:14px">Aún no hay cadenas de ${isClose ? 'cierre' : 'entrada'} auto-descargadas.</div>`;
      if (!isClose && lastEl) lastEl.textContent = 'aún no disponible';
      return;
    }
    const idx = await r.json();
    if (!isClose && lastEl) lastEl.textContent = idx.lastUpdated ? formatMadridTime(idx.lastUpdated) : '—';
    const dates = (idx.dates || []).slice().reverse();
    if (dates.length === 0) {
      listEl.innerHTML = `<div style="font-size:12px;color:var(--ink-soft);text-align:center;padding:14px">Sin cadenas de ${isClose ? 'cierre' : 'entrada'} todavía.</div>`;
      return;
    }
    listEl.innerHTML = `
      <div style="font-size:11px;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px">
        ${label} (${dates.length})
      </div>
      ${dates.map(d => `
        <div class="auto-chain-item">
          <div>
            <span class="ac-date">${d}</span>
            <span class="ac-meta"> · ${isClose ? 'snapshot cierre' : 'snapshot entrada'}</span>
          </div>
          <button class="auto-chain-view" data-date="${d}" data-kind="${kind}">👁 Ver</button>
        </div>`).join('')}`;
    listEl.querySelectorAll('.auto-chain-view').forEach(btn => {
      btn.addEventListener('click', () => viewAutoChain(btn.dataset.date, btn.dataset.kind));
    });
  } catch (e) {
    console.warn('[Auto chains] Error:', e.message);
    listEl.innerHTML = `<div style="color:var(--bad);font-size:12px">Error: ${e.message}</div>`;
  }
}

async function viewAutoChain(date, kind = 'entry') {
  const preview = document.getElementById('chainPreview');
  if (!preview) return;
  preview.style.display = '';
  preview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  preview.innerHTML = '<div style="padding:14px;text-align:center">Cargando cadena…</div>';
  try {
    const folder = kind === 'close' ? 'chains-close' : 'chains';
    const url = `${GITHUB_RAW_BASE}/data/${folder}/${date}.json?t=${Date.now()}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const c = await r.json();
    const expirations = Object.entries(c.expirations || {});
    const fmt = v => isFinite(v) && v !== null ? Number(v).toFixed(2) : '—';
    const fmtIV = v => isFinite(v) && v !== null ? (Number(v) * 100).toFixed(1) + '%' : '—';
    const badge = (c.kind || kind) === 'close' ? 'Cierre sesión' : 'Entrada sesión';
    let html = `<div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px 14px;margin-bottom:12px;font-size:12px">
      <b>${badge} · Cadena ${c.date}</b> · Spot: <b>${fmt(c.spot)}</b> · Capturada: <b>${formatMadridTime(c.capturedAt)} (Madrid)</b>
    </div>`;
    for (const [exp, data] of expirations) {
      const strikes = Array.isArray(data.strikes) ? data.strikes : [];
      const rows = strikes.map(s => `<tr>
        <td>${s.strike}</td>
        <td class="call-cell">${fmt(s.call_bid)}</td>
        <td class="call-cell">${fmt(s.call_ask)}</td>
        <td class="call-cell">${fmtIV(s.call_iv)}</td>
        <td class="put-cell">${fmt(s.put_bid)}</td>
        <td class="put-cell">${fmt(s.put_ask)}</td>
        <td class="put-cell">${fmtIV(s.put_iv)}</td>
      </tr>`).join('');
      html += `
        <details ${data.dte === 0 ? 'open' : ''} style="margin-bottom:8px">
          <summary style="cursor:pointer;font-size:13px;font-weight:600;color:var(--navy-700);padding:6px 0">
            Vencimiento ${exp} (${data.dte} DTE) · ${strikes.length} strikes
          </summary>
          <table class="chain-table" style="margin-top:6px">
            <thead><tr>
              <th>Strike</th>
              <th>Call Bid</th><th>Call Ask</th><th>Call IV</th>
              <th>Put Bid</th><th>Put Ask</th><th>Put IV</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </details>`;
    }
    html += `<div class="chain-preview-actions"><button class="chain-cancel-btn" id="chainCancelBtn">Cerrar vista</button></div>`;
    preview.innerHTML = html;
    document.getElementById('chainCancelBtn').addEventListener('click', () => {
      preview.style.display = 'none';
    });
  } catch (e) {
    preview.innerHTML = `<div style="color:var(--bad);padding:14px">Error cargando cadena: ${e.message}</div>`;
  }
}

async function fetchFirstJson(candidates) {
  let lastError = null;
  for (const url of candidates) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError || new Error('No se pudo cargar el JSON');
}

async function loadVolSurfaceDates() {
  const index = await fetchFirstJson([
    `data/chains-close-index.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/chains-close-index.json?t=${Date.now()}`
  ]);
  const dates = (index.dates || []).filter(Boolean).sort();
  const completeDates = [];
  for (const date of dates) {
    try {
      const chain = await loadVolSurfaceChain(date);
      if (countVolSurfaceExpirations(chain) >= 5) completeDates.push(date);
    } catch (_) {}
  }
  return completeDates.length ? completeDates : dates;
}

async function loadVolSurfaceChain(date) {
  return fetchFirstJson([
    `data/chains-close/${date}.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/chains-close/${date}.json?t=${Date.now()}`,
    `data/chains/${date}.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/chains/${date}.json?t=${Date.now()}`
  ]);
}

function countVolSurfaceExpirations(chain) {
  return Object.values(chain && chain.expirations || {})
    .filter(data => Array.isArray(data.strikes) && data.strikes.length > 0)
    .length;
}

function surfaceIvValue(row, side) {
  const key = side === 'put' ? 'put_iv' : 'call_iv';
  const value = Number(row[key]);
  return Number.isFinite(value) && value > 0 ? value * 100 : null;
}

function buildVolSurfaceGrid(chain, side = 'call') {
  const expirations = Object.entries(chain.expirations || {})
    .map(([exp, data]) => ({ exp, dte: Number(data.dte), strikes: Array.isArray(data.strikes) ? data.strikes : [] }))
    .filter(item => Number.isFinite(item.dte) && item.dte > 0 && item.strikes.length)
    .sort((a, b) => a.dte - b.dte);
  if (!expirations.length) return null;

  const spot = Number(chain.spot);
  let strikes = [...new Set(expirations.flatMap(item => item.strikes.map(row => Number(row.strike)).filter(Number.isFinite)))]
    .sort((a, b) => a - b);
  if (Number.isFinite(spot) && spot > 0) {
    strikes = strikes.filter(strike => strike >= spot * 0.95 && strike <= spot * 1.05);
  }
  const maxStrikes = 115;
  if (strikes.length > maxStrikes) {
    const step = Math.ceil(strikes.length / maxStrikes);
    strikes = strikes.filter((_, idx) => idx % step === 0);
  }

  const z = expirations.map(item => {
    const byStrike = new Map();
    item.strikes.forEach(row => {
      const strike = Number(row.strike);
      const iv = surfaceIvValue(row, side);
      if (Number.isFinite(strike) && iv !== null) byStrike.set(strike, iv);
    });
    return strikes.map(strike => byStrike.get(strike) ?? null);
  });

  return {
    date: chain.date,
    capturedAt: chain.capturedAt,
    spot,
    side,
    excludedZeroDte: countVolSurfaceExpirations(chain) > expirations.length,
    strikes,
    dtes: expirations.map(item => item.dte),
    expirations: expirations.map(item => item.exp),
    z
  };
}

function renderVolSurfaceChart(elementId, grid, totalCompleteDates = null) {
  const status = document.getElementById('volSurfaceStatus');
  const chart = document.getElementById(elementId);
  if (!grid || !grid.strikes.length || !grid.dtes.length) {
    if (chart) chart.innerHTML = '<div style="padding:24px;color:#8a3a3a">No hay suficientes datos de IV para dibujar la superficie.</div>';
    return;
  }
  if (status) {
    const spot = Number.isFinite(grid.spot) ? grid.spot.toFixed(2) : '-';
    const sessions = Number.isFinite(Number(totalCompleteDates)) ? ` | ${totalCompleteDates} sesiones completas` : '';
    const zeroDteNote = grid.excludedZeroDte ? ' | 0DTE excluido' : '';
    status.textContent = `${grid.dtes.length} vencimientos usados | ${grid.strikes.length} strikes | spot ${spot}${sessions}${zeroDteNote}`;
  }

  const trace = {
    type: 'surface',
    x: grid.strikes,
    y: grid.dtes,
    z: grid.z,
    colorscale: [
      [0, '#f7f7f3'],
      [0.35, '#cfcfc8'],
      [0.62, '#10b981'],
      [1, '#050505']
    ],
    colorbar: { title: `${grid.side === 'put' ? 'Put' : 'Call'} IV %`, thickness: 12 },
    contours: {
      z: { show: true, usecolormap: true, highlightcolor: '#050505', project: { z: true } }
    },
    hovertemplate: `${grid.side === 'put' ? 'Put' : 'Call'}<br>Strike %{x}<br>DTE %{y}<br>IV %{z:.2f}%<extra></extra>`
  };
  const layout = {
    margin: { l: 0, r: 0, t: 22, b: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      xaxis: { title: 'Strike', backgroundcolor: 'rgba(255,255,255,0.68)', gridcolor: '#d6d6d1' },
      yaxis: { title: 'Dias a expirar', backgroundcolor: 'rgba(255,255,255,0.68)', gridcolor: '#d6d6d1' },
      zaxis: { title: 'Volatilidad implicita (%)', backgroundcolor: 'rgba(255,255,255,0.68)', gridcolor: '#d6d6d1' },
      camera: { eye: { x: 1.7, y: 1.45, z: 0.9 } }
    }
  };
  safePlotly(elementId, [trace], layout, { responsive: true, displayModeBar: true });
}

async function renderVolSurfaceLab() {
  const content = document.getElementById('researchDetailContent');
  if (!content) return;
  content.innerHTML = `
    <h3>Volatility surface</h3>
    <p class="vol-surface-note">
      Reconstruccion 3D desde las cadenas guardadas: eje X = strike, eje Y = dias a expirar, eje Z = IV.
      El slider inferior mueve la fecha de captura, excluye sesiones con una sola cadena descargada y no dibuja el 0DTE para evitar IV anualizada distorsionada.
    </p>
    <div class="vol-surface-toolbar">
      <input id="volSurfaceDateRange" type="range" min="0" max="0" value="0" step="1" aria-label="Fecha de cadena">
      <div id="volSurfaceDateLabel" class="vol-surface-date">Cargando</div>
    </div>
    <div id="volSurfaceStatus" class="vol-surface-note">Cargando sesiones completas...</div>
    <div class="vol-surface-grid">
      <section class="vol-surface-panel">
        <p class="vol-surface-panel-title">Calls</p>
        <div id="volSurfaceCallChart" class="vol-surface-chart"></div>
      </section>
      <section class="vol-surface-panel">
        <p class="vol-surface-panel-title">Puts</p>
        <div id="volSurfacePutChart" class="vol-surface-chart"></div>
      </section>
    </div>
  `;

  try {
    const dates = await loadVolSurfaceDates();
    if (!dates.length) throw new Error('No hay indice de cadenas de cierre');
    const slider = document.getElementById('volSurfaceDateRange');
    const label = document.getElementById('volSurfaceDateLabel');
    if (!slider || !label) return;
    slider.max = String(dates.length - 1);
    slider.value = String(dates.length - 1);

    async function drawSelected() {
      const idx = Math.max(0, Math.min(dates.length - 1, Number(slider.value)));
      const date = dates[idx];
      label.textContent = date;
      const status = document.getElementById('volSurfaceStatus');
      if (status) status.textContent = `Cargando ${date}... (${dates.length} sesiones con 5+ vencimientos)`;
      const chain = await loadVolSurfaceChain(date);
      renderVolSurfaceChart('volSurfaceCallChart', buildVolSurfaceGrid(chain, 'call'), dates.length);
      renderVolSurfaceChart('volSurfacePutChart', buildVolSurfaceGrid(chain, 'put'), dates.length);
    }

    slider.addEventListener('input', drawSelected);
    await drawSelected();
  } catch (e) {
    content.innerHTML += `<div style="margin-top:18px;padding:18px;border-radius:18px;background:#fff2f2;color:#8a3a3a;border:1px solid #e5b8b8">Error cargando superficie: ${e.message}</div>`;
  }
}

const GEX_MODEL_ORDER = ['pure_gex', 'liquidity_weighted', 'spotgamma_fit'];
const GEX_MODE_ORDER = [
  { id: 'next', label: 'GEX proximo vencimiento' },
  { id: 'multi', label: 'GEX multi-vencimiento' }
];
const GEX_LEVEL_META = {
  callWall: { label: 'Call Wall', color: '#7CFF9A' },
  putWall: { label: 'Put Wall', color: '#ff4d5e' },
  gammaFlip: { label: 'Gamma Flip', color: '#ffb347' },
  volTrigger: { label: 'VT', color: '#b388ff' }
};
const GEX_HELP_TEXT = {
  bestError: {
    title: 'Error medio mejor',
    body: 'Es el menor error medio historico entre todos los modelos y modos. Compara los niveles estimados contra tus niveles SpotGamma manuales usando una ponderacion: Call Wall 35%, Put Wall 35%, Gamma Flip 15% y VT 15%. Sirve para ver que modelo se esta acercando mas a SpotGamma en el historico disponible.'
  },
  avgError: {
    title: 'Error SG medio',
    body: 'Es el error medio historico de ese modelo y modo frente a SpotGamma. Para cada sesion con referencia manual, calcula la distancia en puntos de los niveles estimados contra SpotGamma y despues promedia esas diferencias. Cuanto menor sea, mas parecido esta siendo el modelo a SpotGamma.'
  },
  sessionError: {
    title: 'Error SG sesion',
    body: 'Es el error solo de la sesion seleccionada. Mide cuantos puntos se alejan los niveles del modelo frente a los niveles SpotGamma introducidos para esa fecha aplicada. Si no hay referencia SpotGamma para esa sesion, aparece vacio.'
  }
};

async function loadGexModelIndex() {
  return fetchFirstJson([
    `data/gex-models-index.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/gex-models-index.json?t=${Date.now()}`
  ]);
}

async function loadGexModelRecord(date) {
  return fetchFirstJson([
    `data/gex-models/${date}.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/gex-models/${date}.json?t=${Date.now()}`
  ]);
}

function gexNum(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function gexStatusLabel(status) {
  const map = {
    win: 'Acierto',
    loss: 'Tocado',
    no_trade: 'No opera',
    pending: 'Pendiente'
  };
  return map[status] || '-';
}

function gexOperationClass(status) {
  if (status === 'win') return '#0b7f3a';
  if (status === 'loss') return '#a20f2d';
  if (status === 'no_trade') return '#7a4a00';
  return '#555';
}

function summarizeGexRecords(records) {
  const summary = {};
  for (const modelId of GEX_MODEL_ORDER) {
    summary[modelId] = {};
    for (const mode of GEX_MODE_ORDER) {
      const stats = { errors: [], wins: 0, losses: 0, noTrade: 0, pending: 0, totalSignals: 0 };
      for (const record of records) {
        const diag = record?.diagnostics?.[modelId]?.[mode.id];
        const err = Number(diag?.spotgammaError?.weightedError);
        if (Number.isFinite(err)) stats.errors.push(err);
        const op = diag?.operation || {};
        if (op.status === 'win') {
          stats.wins += 1;
          stats.totalSignals += 1;
        } else if (op.status === 'loss') {
          stats.losses += 1;
          stats.totalSignals += 1;
        } else if (op.status === 'no_trade') {
          stats.noTrade += 1;
        } else {
          stats.pending += 1;
        }
      }
      const avgError = stats.errors.length ? stats.errors.reduce((a, b) => a + b, 0) / stats.errors.length : null;
      const wr = stats.totalSignals ? stats.wins / stats.totalSignals * 100 : null;
      summary[modelId][mode.id] = { ...stats, avgError, wr };
    }
  }
  return summary;
}

function levelValueNear(level, strike, tolerance = 12.5) {
  const a = Number(level);
  const b = Number(strike);
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tolerance;
}

function gexBarColor(row, levels) {
  const strike = row.strike;
  if (levelValueNear(levels.callWall, strike)) return GEX_LEVEL_META.callWall.color;
  if (levelValueNear(levels.putWall, strike)) return GEX_LEVEL_META.putWall.color;
  if (levelValueNear(levels.gammaFlip, strike)) return GEX_LEVEL_META.gammaFlip.color;
  if (levelValueNear(levels.volTrigger, strike)) return GEX_LEVEL_META.volTrigger.color;
  return '#d9d9d4';
}

function gexHistogramRange(levels, spot) {
  const callWall = Number(levels?.callWall);
  const putWall = Number(levels?.putWall);
  if (Number.isFinite(callWall) && Number.isFinite(putWall) && callWall > putWall) {
    return [putWall - 100, callWall + 100];
  }
  const center = Number(spot);
  if (Number.isFinite(center)) return [center - 300, center + 300];
  return undefined;
}

function toggleGexChartExpanded(element) {
  if (!element) return;
  const expanded = element.classList.toggle('is-expanded');
  if (typeof Plotly !== 'undefined') {
    setTimeout(() => {
      try { Plotly.Plots.resize(element); } catch (_) {}
    }, 80);
  }
  if (expanded) {
    element.setAttribute('aria-label', 'Histograma GEX ampliado. Pulsa Escape o haz clic para cerrar.');
  } else {
    element.removeAttribute('aria-label');
  }
}

function showGexHelp(key) {
  const help = GEX_HELP_TEXT[key];
  if (!help) return;
  const overlay = document.getElementById('gexHelpOverlay');
  const title = document.getElementById('gexHelpTitle');
  const body = document.getElementById('gexHelpBody');
  if (!overlay || !title || !body) return;
  title.textContent = help.title;
  body.textContent = help.body;
  overlay.classList.add('open');
  overlay.setAttribute('aria-hidden', 'false');
}

function closeGexHelp() {
  const overlay = document.getElementById('gexHelpOverlay');
  if (!overlay) return;
  overlay.classList.remove('open');
  overlay.setAttribute('aria-hidden', 'true');
}

function gexModelKey(modelId, date) {
  return `${modelId}_${String(date || '').replaceAll('-', '')}`;
}

function toggleGexModelView(modelId, date) {
  const key = gexModelKey(modelId, date);
  const card = document.getElementById(`gexModelCard_${key}`);
  if (!card) return;
  const modes = card.querySelector('.gex-mode-grid');
  const evolution = card.querySelector('.gex-evolution-panel');
  const button = card.querySelector('.gex-evolution-btn');
  const open = !evolution?.classList.contains('open');
  if (modes) modes.style.display = open ? 'none' : '';
  if (evolution) evolution.classList.toggle('open', open);
  if (button) {
    button.classList.toggle('active', open);
    button.textContent = open ? 'Histogramas' : 'Evolución';
  }
  const chart = document.getElementById(`gexEvolution_${key}`);
  if (chart && typeof Plotly !== 'undefined') {
    setTimeout(() => {
      try { Plotly.Plots.resize(chart); } catch (_) {}
    }, 80);
  }
}

function renderGexHistogram(elementId, record, modelId, modeId) {
  const element = document.getElementById(elementId);
  const levels = record?.models?.[modelId]?.[modeId];
  const rows = levels?.histogram || [];
  if (!element || !levels || !rows.length) {
    if (element) element.innerHTML = '<div class="gex-empty">No hay histograma GEX para esta sesion.</div>';
    return;
  }

  const x = rows.map(row => row.strike);
  const y = rows.map(row => Math.abs(Number(row.callGex || 0)) + Math.abs(Number(row.putGex || 0)));
  const colors = rows.map(row => gexBarColor(row, levels));
  const custom = rows.map(row => [row.callGex, row.putGex, row.netGex, row.callScore, row.putScore]);
  const maxY = Math.max(...y.filter(Number.isFinite), 1);
  const xRange = gexHistogramRange(levels, record.spot);
  const shapes = [];
  const annotations = [];
  for (const [key, meta] of Object.entries(GEX_LEVEL_META)) {
    const value = Number(levels[key]);
    if (!Number.isFinite(value)) continue;
    shapes.push({
      type: 'line',
      x0: value,
      x1: value,
      y0: 0,
      y1: maxY * 1.08,
      xref: 'x',
      yref: 'y',
      line: { color: meta.color, width: 2 }
    });
    annotations.push({
      x: value,
      y: maxY * 1.1,
      xref: 'x',
      yref: 'y',
      text: meta.label,
      showarrow: false,
      font: { color: meta.color, size: 10 },
      textangle: -90,
      yanchor: 'bottom'
    });
  }
  const spot = Number(record.spot);
  if (Number.isFinite(spot)) {
    shapes.push({
      type: 'line',
      x0: spot,
      x1: spot,
      y0: 0,
      y1: maxY * 1.08,
      xref: 'x',
      yref: 'y',
      line: { color: '#ffffff', width: 1.5, dash: 'dot' }
    });
  }

  const trace = {
    type: 'bar',
    x,
    y,
    marker: { color: colors },
    customdata: custom,
    hovertemplate:
      'Strike %{x}<br>Abs GEX %{y:,.0f}<br>Call GEX %{customdata[0]:,.0f}<br>Put GEX %{customdata[1]:,.0f}<br>Net GEX %{customdata[2]:,.0f}<extra></extra>'
  };
  const layout = {
    margin: { l: 54, r: 18, t: 42, b: 46 },
    paper_bgcolor: '#050505',
    plot_bgcolor: '#050505',
    bargap: 0.08,
    shapes,
    annotations,
    font: { color: '#e8e8e2' },
    xaxis: { title: 'Strike', gridcolor: 'rgba(255,255,255,0.08)', zeroline: false, fixedrange: true, range: xRange },
    yaxis: { title: 'Abs GEX', gridcolor: 'rgba(255,255,255,0.08)', zeroline: false, fixedrange: true },
    dragmode: false,
    showlegend: false
  };
  safePlotly(elementId, [trace], layout, { responsive: true, displayModeBar: false, scrollZoom: false, doubleClick: false });
  element.onclick = () => toggleGexChartExpanded(element);
}

function renderGexEvolutionChart(elementId, records, modelId) {
  const element = document.getElementById(elementId);
  if (!element) return;
  const cleanRecords = (records || []).filter(record => record?.models?.[modelId]?.next);
  if (!cleanRecords.length) {
    element.innerHTML = '<div class="gex-empty">No hay datos historicos para este modelo.</div>';
    return;
  }
  const traces = Object.entries(GEX_LEVEL_META).map(([key, meta]) => ({
    type: 'scatter',
    mode: 'lines+markers',
    name: meta.label,
    x: cleanRecords.map(record => record.targetSession || record.date),
    y: cleanRecords.map(record => {
      const value = Number(record.models?.[modelId]?.next?.[key]);
      return Number.isFinite(value) ? value : null;
    }),
    line: { color: meta.color, width: 3 },
    marker: { color: meta.color, size: 7 },
    hovertemplate: `${meta.label}<br>Sesion %{x}<br>Nivel %{y:,.0f}<extra></extra>`
  }));
  const layout = {
    margin: { l: 58, r: 18, t: 32, b: 74 },
    paper_bgcolor: '#050505',
    plot_bgcolor: '#050505',
    font: { color: '#e8e8e2' },
    hovermode: 'x unified',
    legend: { orientation: 'h', x: 0, y: 1.12, font: { color: '#e8e8e2' } },
    xaxis: {
      title: 'Sesion aplicada',
      gridcolor: 'rgba(255,255,255,0.08)',
      zeroline: false,
      fixedrange: true,
      tickangle: -35
    },
    yaxis: {
      title: 'Nivel SPX (strike)',
      gridcolor: 'rgba(255,255,255,0.08)',
      zeroline: false,
      fixedrange: true
    }
  };
  safePlotly(elementId, traces, layout, { responsive: true, displayModeBar: false, scrollZoom: false, doubleClick: false });
}

function renderGexLevelChips(levels) {
  return Object.entries(GEX_LEVEL_META).map(([key, meta]) => `
    <div class="gex-level-chip">
      <span style="color:${meta.color}">${meta.label}</span>
      <strong>${gexNum(levels?.[key])}</strong>
    </div>
  `).join('');
}

function renderGexDiagnosticChips(diag, globalStats) {
  const op = diag?.operation || {};
  const err = diag?.spotgammaError?.weightedError;
  const wr = globalStats?.wr;
  const wrText = wr === null || wr === undefined
    ? 'Sin ops.'
    : `${gexNum(wr, 1)}%`;
  const opColor = gexOperationClass(op.status);
  return `
    <div class="gex-diagnostic-chip">
      <button type="button" class="gex-info-trigger" onclick="showGexHelp('avgError')">
        <span>Error SG medio</span>
        <strong>${gexNum(globalStats?.avgError, 1)} pts</strong>
      </button>
    </div>
    <div class="gex-diagnostic-chip">
      <button type="button" class="gex-info-trigger" onclick="showGexHelp('sessionError')">
        <span>Error SG sesion</span>
        <strong>${gexNum(err, 1)} pts</strong>
      </button>
    </div>
    <div class="gex-diagnostic-chip">
      <span>WR historico</span>
      <strong>${wrText}</strong>
    </div>
    <div class="gex-diagnostic-chip">
      <span>Sesion objetivo</span>
      <strong style="color:${opColor}">${gexStatusLabel(op.status)}</strong>
    </div>
  `;
}

function renderGexModelCard(record, summary, modelId, modelIndex) {
  const model = record.models[modelId];
  const chartBase = `gexChart_${modelId}_${record.date.replaceAll('-', '')}`;
  const key = gexModelKey(modelId, record.date);
  const evolutionId = `gexEvolution_${key}`;
  const modes = GEX_MODE_ORDER.map(mode => {
    const levels = model[mode.id];
    const diag = record.diagnostics?.[modelId]?.[mode.id];
    const stats = summary?.[modelId]?.[mode.id];
    const chartId = `${chartBase}_${mode.id}`;
    const expirations = levels?.expirationsUsed?.length || 0;
    return `
      <section class="gex-mode-panel">
        <span class="gex-mode-label">${mode.label} · ${expirations} venc.</span>
        <div class="gex-level-row">${renderGexLevelChips(levels)}</div>
        <div class="gex-diagnostic-row">${renderGexDiagnosticChips(diag, stats)}</div>
        <div id="${chartId}" class="gex-chart"></div>
      </section>
    `;
  }).join('');
  return `
    <article id="gexModelCard_${key}" class="gex-model-card">
      <div class="gex-model-head">
        <div>
          <h4>${model.name || modelId}</h4>
          <p>${model.description || ''}</p>
        </div>
        <div class="gex-model-actions">
          <span class="gex-model-badge">Modelo ${modelIndex + 1}</span>
          <button type="button" class="gex-evolution-btn" onclick="toggleGexModelView('${modelId}', '${record.date}')">Evolución</button>
        </div>
      </div>
      <div class="gex-mode-grid">${modes}</div>
      <section class="gex-evolution-panel">
        <span class="gex-mode-label">Evolución histórica · lectura 1 vencimiento</span>
        <div id="${evolutionId}" class="gex-evolution-chart"></div>
      </section>
    </article>
  `;
}

function renderGexSummary(record, records, summary) {
  const ref = record.reference?.spotgamma;
  const best = [];
  for (const modelId of GEX_MODEL_ORDER) {
    for (const mode of GEX_MODE_ORDER) {
      const stats = summary?.[modelId]?.[mode.id];
      if (Number.isFinite(stats?.avgError)) {
        best.push({ modelId, mode: mode.id, avgError: stats.avgError });
      }
    }
  }
  best.sort((a, b) => a.avgError - b.avgError);
  const bestLabel = best.length
    ? `${record.models[best[0].modelId]?.name || best[0].modelId} · ${best[0].mode === 'next' ? 'next' : 'multi'}`
    : '-';
  return `
    <div class="gex-summary-grid">
      <div class="gex-summary-card"><span>Cadena cierre</span><strong>${record.date}</strong></div>
      <div class="gex-summary-card"><span>Sesion aplicada</span><strong>${record.targetSession}</strong></div>
      <div class="gex-summary-card"><span>Spot cierre</span><strong>${gexNum(record.spot, 2)}</strong></div>
      <div class="gex-summary-card"><span>Referencia SG</span><strong>${ref ? `${gexNum(ref.callWall)} / ${gexNum(ref.putWall)}` : 'Sin referencia'}</strong></div>
      <div class="gex-summary-card"><span>Sesiones JSON</span><strong>${records.length}</strong></div>
      <div class="gex-summary-card"><span>Mejor cercania</span><strong>${bestLabel}</strong></div>
      <div class="gex-summary-card">
        <button type="button" class="gex-info-trigger" onclick="showGexHelp('bestError')">
          <span>Error medio mejor</span>
          <strong>${best.length ? gexNum(best[0].avgError, 1) + ' pts' : '-'}</strong>
        </button>
      </div>
      <div class="gex-summary-card"><span>Version</span><strong>${record.version || '-'}</strong></div>
    </div>
  `;
}

async function renderGexLab() {
  const content = document.getElementById('researchDetailContent');
  if (!content) return;
  content.innerHTML = `
    <h3>Cálculos GEX</h3>
    <p class="gex-lab-note">
      Tres modelos experimentales calculados con las cadenas guardadas al cierre. Cada uno ofrece dos lecturas:
      proximo vencimiento y multi-vencimiento. El JSON queda persistido para comparar contra SpotGamma y alimentar
      futuros estudios de machine learning.
    </p>
    <div class="gex-toolbar">
      <input id="gexDateRange" type="range" min="0" max="0" value="0" step="1" aria-label="Fecha GEX">
      <div id="gexDateLabel" class="gex-date-pill">Cargando</div>
      <div id="gexTargetLabel" class="gex-target-pill">Sesion -</div>
    </div>
    <div id="gexLabStatus" class="gex-lab-note">Cargando modelos GEX...</div>
    <div id="gexSummary"></div>
    <div id="gexModelGrid" class="gex-model-grid"></div>
    <div id="gexHelpOverlay" class="gex-help-overlay" aria-hidden="true" onclick="if (event.target === this) closeGexHelp()">
      <section class="gex-help-card" role="dialog" aria-modal="true" aria-labelledby="gexHelpTitle">
        <h4 id="gexHelpTitle"></h4>
        <p id="gexHelpBody"></p>
        <button type="button" onclick="closeGexHelp()">Cerrar</button>
      </section>
    </div>
  `;

  try {
    const index = await loadGexModelIndex();
    const dates = (index.dates || []).filter(Boolean).sort();
    if (!dates.length) throw new Error('No hay JSON GEX generado todavia');
    const records = await Promise.all(dates.map(date => loadGexModelRecord(date)));
    const summary = summarizeGexRecords(records);
    const slider = document.getElementById('gexDateRange');
    const dateLabel = document.getElementById('gexDateLabel');
    const targetLabel = document.getElementById('gexTargetLabel');
    const status = document.getElementById('gexLabStatus');
    const summaryEl = document.getElementById('gexSummary');
    const grid = document.getElementById('gexModelGrid');
    if (!slider || !dateLabel || !targetLabel || !status || !summaryEl || !grid) return;
    slider.max = String(dates.length - 1);
    slider.value = String(dates.length - 1);

    function drawSelected() {
      const idx = Math.max(0, Math.min(dates.length - 1, Number(slider.value)));
      const record = records[idx];
      dateLabel.textContent = record.date;
      targetLabel.textContent = `Aplica ${record.targetSession}`;
      status.textContent = `${dates.length} sesiones utiles | Fuente ${record.sourceChain} | Generado ${record.generatedAt || '-'}`;
      summaryEl.innerHTML = renderGexSummary(record, records, summary);
      grid.innerHTML = GEX_MODEL_ORDER.map((modelId, modelIndex) => renderGexModelCard(record, summary, modelId, modelIndex)).join('');
      for (const modelId of GEX_MODEL_ORDER) {
        for (const mode of GEX_MODE_ORDER) {
          const chartId = `gexChart_${modelId}_${record.date.replaceAll('-', '')}_${mode.id}`;
          renderGexHistogram(chartId, record, modelId, mode.id);
        }
        renderGexEvolutionChart(`gexEvolution_${gexModelKey(modelId, record.date)}`, records, modelId);
      }
    }

    slider.addEventListener('input', drawSelected);
    drawSelected();
  } catch (e) {
    content.innerHTML += `<div class="gex-empty">Error cargando Cálculos GEX: ${e.message}</div>`;
  }
}

const DEX_LEVEL_META = {
  callDeltaWall: { label: 'Call Delta Wall', color: '#7CFF9A' },
  putDeltaWall: { label: 'Put Delta Wall', color: '#ff4d5e' },
  dexFlip: { label: 'DEX Flip', color: '#60a5fa' },
  maxPositiveDex: { label: 'Max + DEX', color: '#b388ff' },
  maxNegativeDex: { label: 'Max - DEX', color: '#ffb347' }
};

async function loadDexModelIndex() {
  return fetchFirstJson([
    `data/dex-models-index.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/dex-models-index.json?t=${Date.now()}`
  ]);
}

async function loadDexModelRecord(date) {
  return fetchFirstJson([
    `data/dex-models/${date}.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/dex-models/${date}.json?t=${Date.now()}`
  ]);
}

function dexNum(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function dexLarge(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  return dexNum(n, 0);
}

function dexHistogramRange(levels, spot) {
  const callWall = Number(levels?.callDeltaWall);
  const putWall = Number(levels?.putDeltaWall);
  if (Number.isFinite(callWall) && Number.isFinite(putWall) && callWall > putWall) {
    return [putWall - 150, callWall + 150];
  }
  const center = Number(spot);
  if (Number.isFinite(center)) return [center - 350, center + 350];
  return undefined;
}

function renderDexSummary(record, records) {
  const levels = record.levels || {};
  const regime = record.diagnostics?.regime || 'neutral';
  const regimeLabel = regime === 'positive' ? 'DEX positivo' : regime === 'negative' ? 'DEX negativo' : 'DEX neutral';
  return `
    <div class="dex-summary-grid">
      <div class="dex-summary-card"><span>Cadena cierre</span><strong>${record.date}</strong></div>
      <div class="dex-summary-card"><span>Sesion aplicada</span><strong>${record.targetSession}</strong></div>
      <div class="dex-summary-card"><span>Spot cierre</span><strong>${dexNum(record.spot, 2)}</strong></div>
      <div class="dex-summary-card"><span>Regimen DEX</span><strong>${regimeLabel}</strong></div>
      <div class="dex-summary-card"><span>Net DEX spot</span><strong>${dexLarge(levels.netDexAtSpot)}</strong></div>
      <div class="dex-summary-card"><span>Vencimientos</span><strong>${levels.expirationsUsed?.length || 0}</strong></div>
      <div class="dex-summary-card"><span>Sesiones JSON</span><strong>${records.length}</strong></div>
      <div class="dex-summary-card"><span>Version</span><strong>${record.version || '-'}</strong></div>
    </div>
  `;
}

function renderDexLevels(record) {
  const levels = record.levels || {};
  return `
    <div class="dex-level-grid">
      ${Object.entries(DEX_LEVEL_META).map(([key, meta]) => `
        <div class="dex-level-card">
          <span style="color:${meta.color}">${meta.label}</span>
          <strong>${dexNum(levels[key])}</strong>
        </div>
      `).join('')}
      <div class="dex-level-card">
        <span>Modelo</span>
        <strong>${record.model?.name || 'Weighted DEX'}</strong>
      </div>
      <div class="dex-level-card">
        <span>Fuente</span>
        <strong>${record.sourceChain || '-'}</strong>
      </div>
      <div class="dex-level-card">
        <span>Captura</span>
        <strong>${record.capturedAt ? formatMadridTime(record.capturedAt) : '-'}</strong>
      </div>
    </div>
  `;
}

function renderDexHistogram(elementId, record) {
  const element = document.getElementById(elementId);
  const levels = record?.levels || {};
  const rows = levels.histogram || [];
  if (!element || !rows.length) {
    if (element) element.innerHTML = '<div class="gex-empty">No hay histograma DEX para esta sesion.</div>';
    return;
  }
  const x = rows.map(row => row.strike);
  const y = rows.map(row => Number(row.netDex || 0));
  const colors = y.map(value => value >= 0 ? '#d9d9d4' : '#888883');
  const custom = rows.map(row => [row.callDex, row.putDex, row.absDex]);
  const maxAbs = Math.max(...y.map(value => Math.abs(value)).filter(Number.isFinite), 1);
  const xRange = dexHistogramRange(levels, record.spot);
  const shapes = [];
  const annotations = [];
  for (const [key, meta] of Object.entries(DEX_LEVEL_META)) {
    const value = Number(levels[key]);
    if (!Number.isFinite(value)) continue;
    shapes.push({
      type: 'line',
      x0: value,
      x1: value,
      y0: -maxAbs * 1.05,
      y1: maxAbs * 1.05,
      xref: 'x',
      yref: 'y',
      line: { color: meta.color, width: 2 }
    });
    annotations.push({
      x: value,
      y: maxAbs * 1.08,
      xref: 'x',
      yref: 'y',
      text: meta.label,
      showarrow: false,
      font: { color: meta.color, size: 10 },
      textangle: -90,
      yanchor: 'bottom'
    });
  }
  const spot = Number(record.spot);
  if (Number.isFinite(spot)) {
    shapes.push({
      type: 'line',
      x0: spot,
      x1: spot,
      y0: -maxAbs * 1.05,
      y1: maxAbs * 1.05,
      xref: 'x',
      yref: 'y',
      line: { color: '#ffffff', width: 1.5, dash: 'dot' }
    });
  }
  safePlotly(elementId, [{
    type: 'bar',
    x,
    y,
    marker: { color: colors },
    customdata: custom,
    hovertemplate: 'Strike %{x}<br>Net DEX %{y:,.0f}<br>Call DEX %{customdata[0]:,.0f}<br>Put DEX %{customdata[1]:,.0f}<br>Abs DEX %{customdata[2]:,.0f}<extra></extra>'
  }], {
    margin: { l: 62, r: 18, t: 42, b: 48 },
    paper_bgcolor: '#050505',
    plot_bgcolor: '#050505',
    bargap: 0.08,
    shapes,
    annotations,
    font: { color: '#e8e8e2' },
    xaxis: { title: 'Strike', gridcolor: 'rgba(255,255,255,0.08)', zeroline: false, fixedrange: true, range: xRange },
    yaxis: { title: 'Net DEX', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.32)', fixedrange: true },
    dragmode: false,
    showlegend: false
  }, { responsive: true, displayModeBar: false, scrollZoom: false, doubleClick: false });
}

function renderDexEvolutionChart(elementId, records) {
  const element = document.getElementById(elementId);
  if (!element) return;
  const traces = Object.entries(DEX_LEVEL_META).map(([key, meta]) => ({
    type: 'scatter',
    mode: 'lines+markers',
    name: meta.label,
    x: records.map(record => record.targetSession || record.date),
    y: records.map(record => {
      const value = Number(record.levels?.[key]);
      return Number.isFinite(value) ? value : null;
    }),
    line: { color: meta.color, width: 3 },
    marker: { color: meta.color, size: 7 },
    hovertemplate: `${meta.label}<br>Sesion %{x}<br>Nivel %{y:,.0f}<extra></extra>`
  }));
  safePlotly(elementId, traces, {
    margin: { l: 58, r: 18, t: 32, b: 74 },
    paper_bgcolor: '#050505',
    plot_bgcolor: '#050505',
    font: { color: '#e8e8e2' },
    hovermode: 'x unified',
    legend: { orientation: 'h', x: 0, y: 1.12, font: { color: '#e8e8e2' } },
    xaxis: { title: 'Sesion aplicada', gridcolor: 'rgba(255,255,255,0.08)', zeroline: false, fixedrange: true, tickangle: -35 },
    yaxis: { title: 'Nivel SPX (strike)', gridcolor: 'rgba(255,255,255,0.08)', zeroline: false, fixedrange: true }
  }, { responsive: true, displayModeBar: false, scrollZoom: false, doubleClick: false });
}

async function renderDexLab() {
  const content = document.getElementById('researchDetailContent');
  if (!content) return;
  content.innerHTML = `
    <h3>Cálculos DEX</h3>
    <p class="dex-lab-note">
      Lectura inicial de Delta Exposure desde las cadenas guardadas al cierre. Usa una ponderacion decreciente por vencimiento
      para que el vencimiento cercano domine, sin perder el contexto de las siguientes expiraciones.
    </p>
    <div class="dex-toolbar">
      <input id="dexDateRange" type="range" min="0" max="0" value="0" step="1" aria-label="Fecha DEX">
      <div id="dexDateLabel" class="dex-date-pill">Cargando</div>
      <div id="dexTargetLabel" class="dex-target-pill">Sesion -</div>
    </div>
    <div id="dexLabStatus" class="dex-lab-note">Cargando modelos DEX...</div>
    <div id="dexSummary"></div>
    <div id="dexLevels"></div>
    <section class="dex-panel">
      <span class="dex-panel-title">Mapa DEX por strike</span>
      <div id="dexHistogramChart" class="dex-chart"></div>
    </section>
    <section class="dex-panel">
      <span class="dex-panel-title">Evolución histórica de niveles DEX</span>
      <div id="dexEvolutionChart" class="dex-chart"></div>
    </section>
  `;

  try {
    const index = await loadDexModelIndex();
    const dates = (index.dates || []).filter(Boolean).sort();
    if (!dates.length) throw new Error('No hay JSON DEX generado todavia');
    const records = await Promise.all(dates.map(date => loadDexModelRecord(date)));
    const slider = document.getElementById('dexDateRange');
    const dateLabel = document.getElementById('dexDateLabel');
    const targetLabel = document.getElementById('dexTargetLabel');
    const status = document.getElementById('dexLabStatus');
    const summaryEl = document.getElementById('dexSummary');
    const levelsEl = document.getElementById('dexLevels');
    if (!slider || !dateLabel || !targetLabel || !status || !summaryEl || !levelsEl) return;
    slider.max = String(dates.length - 1);
    slider.value = String(dates.length - 1);

    function drawSelected() {
      const idx = Math.max(0, Math.min(dates.length - 1, Number(slider.value)));
      const record = records[idx];
      dateLabel.textContent = record.date;
      targetLabel.textContent = `Aplica ${record.targetSession}`;
      status.textContent = `${dates.length} sesiones utiles | Fuente ${record.sourceChain} | Generado ${record.generatedAt || '-'}`;
      summaryEl.innerHTML = renderDexSummary(record, records);
      levelsEl.innerHTML = renderDexLevels(record);
      renderDexHistogram('dexHistogramChart', record);
      renderDexEvolutionChart('dexEvolutionChart', records);
    }

    slider.addEventListener('input', drawSelected);
    drawSelected();
  } catch (e) {
    content.innerHTML += `<div class="gex-empty">Error cargando Cálculos DEX: ${e.message}</div>`;
  }
}

async function loadMlDatasetIndex() {
  return fetchFirstJson([
    `data/ml-dataset-index.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/ml-dataset-index.json?t=${Date.now()}`
  ]);
}

async function loadMlUnderlyingDataset() {
  return fetchFirstJson([
    `data/ml-dataset/subyacente.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/ml-dataset/subyacente.json?t=${Date.now()}`
  ]);
}

function mlNum(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function renderMlBlocks(index) {
  const labels = {
    subyacente: 'Subyacente',
    spotgamma: 'SpotGamma',
    gex: 'GEX',
    dex: 'DEX',
    vol_surface: 'Vol Surface',
    premiums_labels: 'Primas / Labels'
  };
  const blocks = index.blocks || {};
  return `
    <div class="ml-block-grid">
      ${Object.entries(labels).map(([key, label]) => {
        const block = blocks[key] || { status: 'pending' };
        const available = block.status === 'available';
        return `
          <div class="ml-block-card ${available ? 'available' : ''}">
            <span>${label}</span>
            <strong>${available ? `${block.rows || 0} filas` : 'Pendiente'}</strong>
            <div class="ml-status-pill ${available ? '' : 'pending'}">${available ? 'Disponible' : 'Por construir'}</div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function renderMlUnderlyingTable(rows) {
  const latest = (rows || []).slice(-14).reverse();
  if (!latest.length) {
    return '<div class="gex-empty">No hay filas de subyacente todavía.</div>';
  }
  return `
    <div class="ml-table-panel">
      <div class="ml-table-head">
        <strong>Últimas sesiones · Bloque Subyacente</strong>
        <span class="ml-lab-note" style="margin:0">Variables base para el primer modelo de venta de volatilidad.</span>
      </div>
      <div class="ml-table-wrap">
        <table class="ml-feature-table">
          <thead>
            <tr>
              <th>Fecha</th>
              <th>Open</th>
              <th>Close</th>
              <th>Gap %</th>
              <th>Open % rango prev.</th>
              <th>Range % 1D</th>
              <th>Range % 5D</th>
              <th>RV 5D</th>
              <th>Trend 5D</th>
              <th>VIX Open</th>
              <th>VIX Chg %</th>
            </tr>
          </thead>
          <tbody>
            ${latest.map(row => `
              <tr>
                <td>${row.date}</td>
                <td>${mlNum(row.open, 2)}</td>
                <td>${mlNum(row.close, 2)}</td>
                <td>${mlNum(row.gapPct, 2)}</td>
                <td>${mlNum(row.openPositionPrevRange, 1)}</td>
                <td>${mlNum(row.rangePct1d, 2)}</td>
                <td>${mlNum(row.rangePct5dAvg, 2)}</td>
                <td>${mlNum(row.realizedVol5d, 4)}</td>
                <td>${mlNum(row.trend5d, 4)}</td>
                <td>${mlNum(row.vixOpen, 2)}</td>
                <td>${mlNum(row.vixChangePct, 2)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

async function renderMlLab() {
  const content = document.getElementById('researchDetailContent');
  if (!content) return;
  content.innerHTML = `
    <h3>Machine Learning Lab</h3>
    <p class="ml-lab-note">
      Primer ladrillo del dataset maestro para ML. De momento no entrenamos modelos: construimos datos limpios,
      diarios y ampliables para mejorar la venta de volatilidad 0DTE.
    </p>
    <div id="mlDatasetStatus" class="ml-lab-note">Cargando dataset maestro...</div>
    <div id="mlBlocks"></div>
    <div id="mlUnderlyingPreview"></div>
  `;

  try {
    const index = await loadMlDatasetIndex();
    const underlying = await loadMlUnderlyingDataset();
    const rows = underlying.rows || [];
    const status = document.getElementById('mlDatasetStatus');
    const blocks = document.getElementById('mlBlocks');
    const preview = document.getElementById('mlUnderlyingPreview');
    if (status) {
      const sub = index.blocks?.subyacente || {};
      status.textContent = `Dataset ${index.version || ''} | Subyacente ${sub.rows || rows.length} filas | ${sub.firstDate || '-'} → ${sub.lastDate || '-'} | Actualizado ${index.lastUpdated || '-'}`;
    }
    if (blocks) blocks.innerHTML = renderMlBlocks(index);
    if (preview) preview.innerHTML = renderMlUnderlyingTable(rows);
  } catch (e) {
    content.innerHTML += `<div class="gex-empty">Error cargando Machine Learning Lab: ${e.message}</div>`;
  }
}

function renderResearchPlaceholder(section) {
  const content = document.getElementById('researchDetailContent');
  if (!content) return;
  const title = section === 'gex' ? 'Cálculos GEX' : 'Cálculos DEX';
  content.innerHTML = `
    <h3>${title}</h3>
    <p class="vol-surface-note">
      Entrada preparada. En la siguiente fase conectaremos aqui las metricas especificas y sus graficos.
    </p>
  `;
}
// ---- Madrid timezone helpers --------------------------------------------
function formatMadridTime(isoUtc) {
  if (!isoUtc) return '—';
  try {
    const d = new Date(isoUtc);
    return d.toLocaleString('es-ES', {
      timeZone: 'Europe/Madrid',
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    }) + ' h';
  } catch (_) { return isoUtc; }
}

function getMadridDST() {
  const parts = new Intl.DateTimeFormat('es-ES', {
    timeZone: 'Europe/Madrid', timeZoneName: 'short'
  }).formatToParts(new Date());
  const tz = parts.find(p => p.type === 'timeZoneName');
  return tz ? tz.value : '—';
}

function updateScheduleBar() {
  const dst = getMadridDST();
  const dstLabel = dst === 'CEST' ? '☀ Verano (CEST, UTC+2)' :
                   dst === 'CET'  ? '❄ Invierno (CET, UTC+1)' : dst;
  const dstEl = document.getElementById('chainsCurrentDST');
  if (dstEl) dstEl.textContent = dstLabel;

  // Cron hardcoded en el YAML — actualízalo aquí también si cambias el YAML
  // Verano (CEST):  cron 0 14 * * 1-5  →  cronUTC = 14
  // Invierno (CET): cron 0 15 * * 1-5  →  cronUTC = 15
  const cronUTC = dst === 'CEST' ? 14 : 15;
  const madridHour = dst === 'CEST' ? cronUTC + 2 : cronUTC + 1;
  const nyHour     = dst === 'CEST' ? cronUTC - 4 : cronUTC - 5;
  const minutesAfterOpen = (nyHour - 9) * 60 - 30; // NY abre 9:30
  const delayedDataMin = minutesAfterOpen - 15;     // CBOE delay 15min
  const nextEl = document.getElementById('chainsNextRun');
  if (nextEl) {
    nextEl.innerHTML = `<b>${String(madridHour).padStart(2,'0')}:00 Madrid</b> (= ${String(nyHour).padStart(2,'0')}:00 NY, datos efectivos +${delayedDataMin} min tras apertura)`;
  }
}

const editBtn = document.getElementById('editScheduleBtn');
if (editBtn) {
  editBtn.addEventListener('click', () => {
    window.open('https://github.com/Quique805/SPX-IC/edit/main/.github/workflows/daily-fetch.yml', '_blank');
    alert('Abriendo el YAML en GitHub.\n\nObjetivo: que la captura siempre sea a las 16:00 Madrid (= 10:00 NY = 30 min tras apertura).\n\n• VERANO (CEST):  - cron: \'0 14 * * 1-5\'  ← actualmente activo\n• INVIERNO (CET): - cron: \'0 15 * * 1-5\'\n\nCuando España cambie de hora (último domingo de marzo y de octubre), comenta una línea y descomenta la otra. Después actualiza también la línea cronUTC en script.js (función updateScheduleBar) para que coincida.\n\nDespués pulsa "Commit changes" abajo.');
  });
}

// ---- Gamma levels from close chains --------------------------------------
const GAMMA_LEGACY_CLOSE_DATES = ['2026-05-11', '2026-05-13', '2026-05-14', '2026-05-15', '2026-05-18'];
const GAMMA_RISK_FREE_RATE = 0.045;
const GAMMA_CONTRACT_MULT = 100;
const GAMMA_MIN_T = 1 / 252;
const SPOTGAMMA_STORAGE_KEY = 'spx-ic-spotgamma-levels-v1';
const SPOTGAMMA_ENDPOINT_KEY = 'spx-ic-spotgamma-endpoint-v1';
const QUARTERLY_OPEX_DATES = [
  { date: '2026-06-18', note: 'jueves, ajustado por festivo' },
  { date: '2026-09-18' }, { date: '2026-12-18' },
  { date: '2027-03-19' }, { date: '2027-06-18' }, { date: '2027-09-17' }, { date: '2027-12-17' },
  { date: '2028-03-17' }, { date: '2028-06-16' }, { date: '2028-09-15' }, { date: '2028-12-15' },
  { date: '2029-03-16' }, { date: '2029-06-15' }, { date: '2029-09-21' }, { date: '2029-12-21' },
  { date: '2030-03-15' }, { date: '2030-06-21' }, { date: '2030-09-20' }, { date: '2030-12-20' },
  { date: '2031-03-21' }, { date: '2031-06-20' }, { date: '2031-09-19' }, { date: '2031-12-19' },
  { date: '2032-03-19' }, { date: '2032-06-18' }, { date: '2032-09-17' }, { date: '2032-12-17' },
  { date: '2033-03-18' }, { date: '2033-06-17' }, { date: '2033-09-16' }, { date: '2033-12-16' },
  { date: '2034-03-17' }, { date: '2034-06-16' }, { date: '2034-09-15' }, { date: '2034-12-15' },
  { date: '2035-03-16' }, { date: '2035-06-15' }, { date: '2035-09-21' }, { date: '2035-12-21' }
];

function normalizeSpotGammaData(data) {
  const byDate = {};
  const raw = data && data.byDate ? data.byDate : {};
  for (const [date, row] of Object.entries(raw)) {
    const callWall = Number(row && (row.callWall ?? row.call_wall));
    const putWall = Number(row && (row.putWall ?? row.put_wall));
    if (!date || !Number.isFinite(callWall) || !Number.isFinite(putWall)) continue;
    byDate[date] = {
      callWall,
      putWall,
      volTrigger: row.volTrigger ?? row.vol_trigger ?? null,
      gammaFlip: row.gammaFlip ?? row.gamma_flip ?? null,
      source: row.source || 'SpotGamma manual',
      updatedAt: row.updatedAt || null
    };
  }
  return { lastUpdated: data && data.lastUpdated || null, byDate };
}

function getLocalSpotGammaData() {
  try {
    return normalizeSpotGammaData(JSON.parse(localStorage.getItem(SPOTGAMMA_STORAGE_KEY) || '{}'));
  } catch (_) {
    return { lastUpdated: null, byDate: {} };
  }
}

function saveLocalSpotGammaData(data) {
  localStorage.setItem(SPOTGAMMA_STORAGE_KEY, JSON.stringify(normalizeSpotGammaData(data), null, 2));
}

function getSpotGammaEndpoint() {
  try {
    return localStorage.getItem(SPOTGAMMA_ENDPOINT_KEY) || '';
  } catch (_) {
    return '';
  }
}

function saveSpotGammaEndpoint(url) {
  try {
    localStorage.setItem(SPOTGAMMA_ENDPOINT_KEY, String(url || '').trim());
  } catch (_) {}
}

async function postSpotGammaLevel(endpoint, pin, payload) {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pin, ...payload })
  });
  const text = await response.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) {}
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || text || `HTTP ${response.status}`);
  }
  return data;
}

async function fetchSpotGammaData() {
  const merged = { lastUpdated: null, byDate: {} };
  const candidates = [
    `data/spotgamma-levels.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/spotgamma-levels.json?t=${Date.now()}`
  ];
  for (const url of candidates) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const remote = normalizeSpotGammaData(await r.json());
      Object.assign(merged.byDate, remote.byDate);
      merged.lastUpdated = remote.lastUpdated || merged.lastUpdated;
      break;
    } catch (e) {
      console.warn('[SpotGamma] No se pudo cargar', url, e.message);
    }
  }
  const local = getLocalSpotGammaData();
  Object.assign(merged.byDate, local.byDate);
  merged.lastUpdated = local.lastUpdated || merged.lastUpdated;
  return normalizeSpotGammaData(merged);
}

function spotGammaEntryToLevels(date, row) {
  const sourceDate = gammaPreviousSessionDate(date) || date;
  const callWall = { strike: Number(row.callWall) };
  const putWall = { strike: Number(row.putWall) };
  return {
    date: sourceDate,
    sessionDate: date,
    capturedAt: row.updatedAt || null,
    sourceLabel: row.source || 'SpotGamma manual',
    spot: NaN,
    expiration: 'SpotGamma',
    dte: null,
    effectiveDte: null,
    callWall,
    putWall,
    volTrigger: row.volTrigger,
    gammaFlip: row.gammaFlip,
    netAtSpot: NaN,
    topRows: []
  };
}

function spotGammaSortedDates(data, desc = true) {
  const dates = Object.keys((data && data.byDate) || {}).sort();
  return desc ? dates.reverse() : dates;
}

function normalPdf(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function bsGamma(spot, strike, sigma, yearsToExpiry, rate = GAMMA_RISK_FREE_RATE) {
  const S = Number(spot);
  const K = Number(strike);
  const rawSigma = Number(sigma);
  const T = Math.max(Number(yearsToExpiry) || 0, GAMMA_MIN_T);
  if (!(S > 0) || !(K > 0)) return 0;
  const vol = Math.min(Math.max(rawSigma > 0 ? rawSigma : 0.18, 0.03), 5);
  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (rate + 0.5 * vol * vol) * T) / (vol * sqrtT);
  return normalPdf(d1) / (S * vol * sqrtT);
}

function gammaExposure(spot, strike, sigma, yearsToExpiry, openInterest, sideSign) {
  const oi = Number(openInterest) || 0;
  if (oi <= 0) return 0;
  const gamma = bsGamma(spot, strike, sigma, yearsToExpiry);
  return sideSign * gamma * oi * GAMMA_CONTRACT_MULT * spot * spot * 0.01;
}

function getBestExpiration(chain) {
  const expirations = Object.entries(chain.expirations || {});
  if (!expirations.length) return null;
  return expirations
    .map(([date, data]) => ({ date, data, dte: Number(data && data.dte) || 0 }))
    .sort((a, b) => Math.abs(a.dte) - Math.abs(b.dte))[0];
}

function computeGammaLevels(chain) {
  const spot = Number(chain && chain.spot);
  const exp = getBestExpiration(chain);
  if (!(spot > 0) || !exp || !Array.isArray(exp.data.strikes)) return null;
  const yearsToExpiry = Math.max((Number(exp.data.dte) || 0) / 252, GAMMA_MIN_T);
  const strikes = exp.data.strikes
    .map(s => ({
      strike: Number(s.strike),
      callIv: Number(s.call_iv) || Number(s.put_iv) || 0.18,
      putIv: Number(s.put_iv) || Number(s.call_iv) || 0.18,
      callOi: Number(s.call_oi) || 0,
      putOi: Number(s.put_oi) || 0,
      callVolume: Number(s.call_volume) || 0,
      putVolume: Number(s.put_volume) || 0
    }))
    .filter(s => s.strike > 0)
    .sort((a, b) => a.strike - b.strike);
  if (!strikes.length) return null;

  const rows = strikes.map(s => {
    const callGex = gammaExposure(spot, s.strike, s.callIv, yearsToExpiry, s.callOi, 1);
    const putGex = gammaExposure(spot, s.strike, s.putIv, yearsToExpiry, s.putOi, -1);
    return {
      ...s,
      callGex,
      putGex,
      netGex: callGex + putGex,
      absTotalGex: Math.abs(callGex) + Math.abs(putGex)
    };
  });

  const callWall = rows.reduce((best, r) => !best || r.callGex > best.callGex ? r : best, null);
  const putWall = rows.reduce((best, r) => !best || Math.abs(r.putGex) > Math.abs(best.putGex) ? r : best, null);
  const netAtSpot = rows.reduce((sum, r) => sum + r.netGex, 0);

  return {
    date: chain.date,
    capturedAt: chain.capturedAt,
    sourceLabel: chain._sourceLabel || 'cadena cargada',
    spot,
    expiration: exp.date,
    dte: exp.data.dte,
    effectiveDte: Math.max(Number(exp.data.dte) || 0, 1),
    callWall,
    putWall,
    netAtSpot,
    topRows: rows
      .slice()
      .sort((a, b) => b.absTotalGex - a.absTotalGex)
      .slice(0, 8)
  };
}

async function fetchGammaChain(date) {
  const candidates = [
    { url: `data/chains-close/${date}.json?t=${Date.now()}`, label: 'cierre local/Pages' },
    { url: `data/chains/${date}.json?t=${Date.now()}`, label: 'histórico local/Pages' },
    { url: `${GITHUB_RAW_BASE}/data/chains-close/${date}.json?t=${Date.now()}`, label: 'cierre GitHub' },
    { url: `${GITHUB_RAW_BASE}/data/chains/${date}.json?t=${Date.now()}`, label: 'histórico GitHub' }
  ];
  let lastErr = null;
  for (const src of candidates) {
    try {
      const r = await fetch(src.url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const chain = await r.json();
      chain._sourceLabel = src.label;
      return chain;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('cadena no encontrada');
}

async function fetchEntryChain(date) {
  const candidates = [
    { url: `data/chains/${date}.json?t=${Date.now()}`, label: 'entrada local/Pages' },
    { url: `${GITHUB_RAW_BASE}/data/chains/${date}.json?t=${Date.now()}`, label: 'entrada GitHub' }
  ];
  let lastErr = null;
  for (const src of candidates) {
    try {
      const r = await fetch(src.url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const chain = await r.json();
      chain._sourceLabel = src.label;
      return chain;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('cadena de entrada no encontrada');
}

function findOptionStrike(chain, targetStrike) {
  const exp = getBestExpiration(chain);
  const strikes = exp && Array.isArray(exp.data.strikes) ? exp.data.strikes : [];
  const target = Number(targetStrike);
  if (!strikes.length || !Number.isFinite(target)) return null;
  return strikes.reduce((best, s) => {
    const dist = Math.abs(Number(s.strike) - target);
    return !best || dist < best.dist ? { row: s, dist } : best;
  }, null);
}

function getGammaOpenWallSetup(levels) {
  const { target, row } = getGammaSessionRow(levels);
  const callWall = Number(levels.callWall && levels.callWall.strike);
  const putWall = Number(levels.putWall && levels.putWall.strike);
  const wallRange = callWall - putWall;
  if (!(wallRange > 0)) {
    return { ok: false, target, reason: 'Rango de walls no válido', callWall, putWall };
  }
  const open = Number(row && row.open);
  if (!Number.isFinite(open)) {
    return { ok: false, target, reason: 'Open no disponible', callWall, putWall };
  }
  const openPct = ((open - putWall) / wallRange) * 100;
  if (openPct < 10 || openPct > 90) {
    return {
      ok: false,
      target,
      reason: `Open fuera del rango operable (${openPct.toFixed(2)}%)`,
      open,
      openPct,
      callWall,
      putWall
    };
  }
  let sellCall = callWall;
  let sellPut = putWall;
  let adjustment = 'Sin ajuste';
  if (openPct >= 10 && openPct <= 20) {
    sellPut = putWall - 35;
    adjustment = 'Put Wall -35';
  } else if (openPct > 20 && openPct <= 30) {
    sellPut = putWall - 20;
    adjustment = 'Put Wall -20';
  } else if (openPct > 80 && openPct <= 90) {
    sellCall = callWall + 15;
    adjustment = 'Call Wall +15';
  }
  return {
    ok: true,
    target,
    reason: null,
    open,
    openPct,
    callWall,
    putWall,
    sellCall,
    sellPut,
    adjustment
  };
}

async function computeEntryPremiumsForLevels(levels) {
  const entryDate = gammaTargetSessionDate(levels);
  if (!entryDate) return { entryDate, ok: false, error: 'fecha objetivo no válida' };
  const setup = getGammaOpenWallSetup(levels);
  if (!setup.ok) return { entryDate, ok: false, error: setup.reason, openWall: setup };
  try {
    const chain = await fetchEntryChain(entryDate);
    const callHit = findOptionStrike(chain, setup.sellCall);
    const putHit = findOptionStrike(chain, setup.sellPut);
    const callProtectTarget = setup.sellCall + 20;
    const putProtectTarget = setup.sellPut - 20;
    const callProtectHit = findOptionStrike(chain, callProtectTarget);
    const putProtectHit = findOptionStrike(chain, putProtectTarget);
    if (!callHit || !putHit || !callProtectHit || !putProtectHit) throw new Error('strikes no encontrados');
    return {
      ok: true,
      entryDate,
      openWall: setup,
      sourceLabel: chain._sourceLabel,
      capturedAt: chain.capturedAt,
      spot: Number(chain.spot),
      call: {
        targetStrike: setup.sellCall,
        strike: Number(callHit.row.strike),
        bid: Number(callHit.row.call_bid),
        ask: Number(callHit.row.call_ask),
        dist: callHit.dist
      },
      callProtection: {
        targetStrike: callProtectTarget,
        strike: Number(callProtectHit.row.strike),
        bid: Number(callProtectHit.row.call_bid),
        ask: Number(callProtectHit.row.call_ask),
        dist: callProtectHit.dist
      },
      put: {
        targetStrike: setup.sellPut,
        strike: Number(putHit.row.strike),
        bid: Number(putHit.row.put_bid),
        ask: Number(putHit.row.put_ask),
        dist: putHit.dist
      },
      putProtection: {
        targetStrike: putProtectTarget,
        strike: Number(putProtectHit.row.strike),
        bid: Number(putProtectHit.row.put_bid),
        ask: Number(putProtectHit.row.put_ask),
        dist: putProtectHit.dist
      }
    };
  } catch (e) {
    return { ok: false, entryDate, error: e.message };
  }
}

function optionQuote(row, key) {
  const value = Number(row && row[key]);
  return Number.isFinite(value) && value >= 0 ? value : 0;
}

const MIN_WING_CREDIT_POINTS = 0.05;

function gammaWingCredits(p) {
  return {
    call: optionQuote(p.call, 'bid') - optionQuote(p.callProtection, 'ask'),
    put: optionQuote(p.put, 'bid') - optionQuote(p.putProtection, 'ask')
  };
}

function gammaSkippedWings(p) {
  const credits = gammaWingCredits(p);
  const opex = getQuarterlyOpexStatus(p.entryDate);
  const skipped = [];
  if (credits.call <= MIN_WING_CREDIT_POINTS) {
    skipped.push({ wing: 'call', label: 'Call Spread', credit: credits.call });
  }
  if (!opex.isOpexDay && credits.put <= MIN_WING_CREDIT_POINTS) {
    skipped.push({ wing: 'put', label: 'Put Spread', credit: credits.put });
  }
  return skipped;
}

function computeEntryNetCredit(levels) {
  const p = levels.entryPremiums;
  if (!p || !p.ok) return null;
  const opex = getQuarterlyOpexStatus(p.entryDate);
  const credits = gammaWingCredits(p);
  const callCredit = credits.call > MIN_WING_CREDIT_POINTS ? credits.call : 0;
  const putCredit = !opex.isOpexDay && credits.put > MIN_WING_CREDIT_POINTS ? credits.put : 0;
  return callCredit + putCredit;
}

async function fetchGammaIndexDates() {
  const candidates = [
    `data/chains-close-index.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/chains-close-index.json?t=${Date.now()}`
  ];
  const dates = [];
  for (const url of candidates) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const idx = await r.json();
      if (Array.isArray(idx.dates)) dates.push(...idx.dates);
    } catch (e) {
      console.warn('[Gamma index] No se pudo cargar', url, e.message);
    }
  }
  return [...new Set([...dates, ...GAMMA_LEGACY_CLOSE_DATES])]
    .filter(Boolean)
    .sort((a, b) => String(b).localeCompare(String(a)));
}

function fmtGammaNum(v, digits = 0) {
  return Number.isFinite(Number(v)) ? Number(v).toLocaleString('es-ES', { maximumFractionDigits: digits }) : '—';
}

function gammaNextSessionDate(dateStr) {
  if (!dateStr) return null;
  const [y, m, d] = String(dateStr).split('-').map(Number);
  if (!y || !m || !d) return null;
  const dt = new Date(Date.UTC(y, m - 1, d));
  do {
    dt.setUTCDate(dt.getUTCDate() + 1);
  } while (!isMarketTradingDate(dt.toISOString().slice(0, 10)));
  return dt.toISOString().slice(0, 10);
}

function gammaPreviousSessionDate(dateStr) {
  if (!dateStr) return null;
  const [y, m, d] = String(dateStr).split('-').map(Number);
  if (!y || !m || !d) return null;
  const dt = new Date(Date.UTC(y, m - 1, d));
  do {
    dt.setUTCDate(dt.getUTCDate() - 1);
  } while (!isMarketTradingDate(dt.toISOString().slice(0, 10)));
  return dt.toISOString().slice(0, 10);
}

function gammaTargetSessionDate(levelOrDate) {
  if (levelOrDate && typeof levelOrDate === 'object') {
    return levelOrDate.sessionDate || gammaNextSessionDate(levelOrDate.date);
  }
  return gammaNextSessionDate(levelOrDate);
}

function gammaNextSessionLabel(dateStr) {
  if (!dateStr) return 'PARA EL DÍA —';
  const nextDate = gammaNextSessionDate(dateStr);
  if (!nextDate) return 'PARA EL DÍA —';
  const [y, m, d] = nextDate.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  const months = [
    'ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO',
    'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE'
  ];
  return `PARA EL DÍA ${dt.getUTCDate()} DE ${months[dt.getUTCMonth()]}`;
}

function isoWeekMonday(dateStr) {
  const [y, m, d] = String(dateStr || '').split('-').map(Number);
  if (!y || !m || !d) return null;
  const dt = new Date(Date.UTC(y, m - 1, d));
  const day = dt.getUTCDay() || 7;
  dt.setUTCDate(dt.getUTCDate() - day + 1);
  return dt.toISOString().slice(0, 10);
}

function getQuarterlyOpexStatus(sessionDate) {
  if (!sessionDate) return { isOpexDay: false, isOpexWeek: false, event: null };
  const exact = QUARTERLY_OPEX_DATES.find(e => e.date === sessionDate) || null;
  const week = isoWeekMonday(sessionDate);
  const event = exact || QUARTERLY_OPEX_DATES.find(e => isoWeekMonday(e.date) === week) || null;
  return { isOpexDay: Boolean(exact), isOpexWeek: Boolean(event), event };
}

function formatSpanishLongDate(dateStr) {
  const [y, m, d] = String(dateStr || '').split('-').map(Number);
  if (!y || !m || !d) return dateStr || '—';
  const months = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre'];
  return `${d} de ${months[m - 1]} de ${y}`;
}

function renderGammaDataInputPanel() {
  const panel = document.getElementById('gammaDataInputPanel');
  if (!panel) return;
  const byYear = QUARTERLY_OPEX_DATES.reduce((acc, event) => {
    const year = event.date.slice(0, 4);
    (acc[year] ||= []).push(event);
    return acc;
  }, {});
  panel.innerHTML = `
    <div style="display:flex;gap:8px;border-bottom:1px solid var(--blue-200);margin-bottom:12px">
      <button type="button" class="chain-auto-tab active" style="margin-bottom:-1px">Hora Bruja Trimestral</button>
    </div>
    <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px 14px;margin-bottom:12px;font-size:12px">
      <b>Hora Bruja Trimestral</b><br>
      Durante la semana se muestra el aviso OPEX. En la fecha exacta, la estrategia ejecuta únicamente el call spread: venta CW y compra de protección call.
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:10px">
      ${Object.entries(byYear).map(([year, events]) => `
        <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px">
          <div style="font-weight:800;color:var(--navy-700);margin-bottom:7px">${year}</div>
          ${events.map(event => `
            <div style="padding:5px 0;border-top:1px solid rgba(12,45,78,0.10);font-size:12px">
              <b>${formatSpanishLongDate(event.date)}</b>${event.note ? `<br><span style="font-size:10px;color:var(--bad)">${event.note}</span>` : ''}
            </div>`).join('')}
        </div>`).join('')}
    </div>`;
}

async function renderSpotGammaPanel() {
  const panel = document.getElementById('spotGammaPanel');
  if (!panel) return;
  panel.innerHTML = '<div style="padding:14px;text-align:center">Cargando niveles SpotGamma...</div>';
  const data = await fetchSpotGammaData();
  const dates = spotGammaSortedDates(data, true);
  const latest = dates[0] || new Date().toISOString().slice(0, 10);
  const current = data.byDate[latest] || {};
  const endpoint = getSpotGammaEndpoint();
  const rows = dates.map(date => {
    const row = data.byDate[date];
    return `<tr>
      <td><b>${date}</b></td>
      <td>${fmtGammaNum(row.callWall, 0)}</td>
      <td>${fmtGammaNum(row.putWall, 0)}</td>
      <td>${row.volTrigger === null || row.volTrigger === undefined ? '—' : fmtGammaNum(row.volTrigger, 0)}</td>
      <td>${row.gammaFlip === null || row.gammaFlip === undefined ? '—' : fmtGammaNum(row.gammaFlip, 0)}</td>
      <td>${row.source || 'SpotGamma manual'}</td>
      <td><button type="button" class="spotgamma-edit-btn" data-date="${date}">Editar</button></td>
    </tr>`;
  }).join('');
  const exportJson = JSON.stringify(data, null, 2);
  panel.innerHTML = `
    <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px 14px;margin-bottom:12px;font-size:12px">
      <b>SpotGamma manual</b><br>
      Introduce aqui las walls que se aplicaran a la siguiente sesion. El dashboard usa estos niveles para Niveles Gamma, Resumen y reglas de entrada.
      <br><span style="color:var(--bad)">Nota:</span> si configuras el puente externo, el boton Guardar nivel hara commit en <code>data/spotgamma-levels.json</code> y GitHub Actions lo vera con el PC apagado.
    </div>
    <details style="margin-bottom:12px">
      <summary style="cursor:pointer;font-size:12px;color:var(--ink-soft);font-weight:700">Configuracion de guardado remoto</summary>
      <div class="summary-range-controls" style="margin-top:10px;align-items:end">
        <label>Endpoint Worker<input type="url" id="spotGammaEndpoint" placeholder="https://..." value="${endpoint}"></label>
        <label>PIN<input type="password" id="spotGammaPin" placeholder="PIN privado"></label>
        <span style="font-size:11px;color:var(--ink-soft)">La URL se guarda en este navegador. El PIN no se guarda.</span>
      </div>
    </details>
    <div class="summary-range-controls" style="align-items:end">
      <label>Fecha aplicada<input type="date" id="spotGammaDate" value="${latest}"></label>
      <label>Call Wall<input type="number" id="spotGammaCallWall" step="5" value="${current.callWall ?? ''}"></label>
      <label>Put Wall<input type="number" id="spotGammaPutWall" step="5" value="${current.putWall ?? ''}"></label>
      <label>VT<input type="number" id="spotGammaVT" step="5" value="${current.volTrigger ?? ''}"></label>
      <label>Gamma Flip<input type="number" id="spotGammaFlip" step="5" value="${current.gammaFlip ?? ''}"></label>
      <button type="button" id="saveSpotGammaBtn">Guardar nivel</button>
    </div>
    <div style="display:flex;gap:8px;align-items:center;margin:8px 0 12px">
      <button type="button" id="copySpotGammaJsonBtn">Copiar JSON para data/spotgamma-levels.json</button>
      <span id="spotGammaFeedback" style="font-size:12px;color:var(--ink-soft)"></span>
    </div>
    <textarea id="spotGammaJsonExport" readonly style="width:100%;min-height:120px;font-family:monospace;font-size:11px;margin-bottom:12px">${exportJson}</textarea>
    <table class="summary-period-table">
      <thead><tr><th>Fecha</th><th>Call Wall</th><th>Put Wall</th><th>VT</th><th>Gamma Flip</th><th>Fuente</th><th></th></tr></thead>
      <tbody>${rows || '<tr><td colspan="7">No hay niveles cargados.</td></tr>'}</tbody>
    </table>`;

  const feedback = document.getElementById('spotGammaFeedback');
  const fillForm = date => {
    const row = data.byDate[date] || {};
    document.getElementById('spotGammaDate').value = date;
    document.getElementById('spotGammaCallWall').value = row.callWall ?? '';
    document.getElementById('spotGammaPutWall').value = row.putWall ?? '';
    document.getElementById('spotGammaVT').value = row.volTrigger ?? '';
    document.getElementById('spotGammaFlip').value = row.gammaFlip ?? '';
  };
  panel.querySelectorAll('.spotgamma-edit-btn').forEach(btn => {
    btn.addEventListener('click', () => fillForm(btn.dataset.date));
  });
  document.getElementById('saveSpotGammaBtn').addEventListener('click', async () => {
    const date = document.getElementById('spotGammaDate').value;
    const callWall = Number(document.getElementById('spotGammaCallWall').value);
    const putWall = Number(document.getElementById('spotGammaPutWall').value);
    const vtRaw = document.getElementById('spotGammaVT').value;
    const flipRaw = document.getElementById('spotGammaFlip').value;
    const endpoint = document.getElementById('spotGammaEndpoint').value.trim();
    const pin = document.getElementById('spotGammaPin').value;
    if (!date || !Number.isFinite(callWall) || !Number.isFinite(putWall)) {
      if (feedback) feedback.textContent = 'Fecha, Call Wall y Put Wall son obligatorios.';
      return;
    }
    if (endpoint) saveSpotGammaEndpoint(endpoint);
    const payload = {
      date,
      callWall,
      putWall,
      volTrigger: vtRaw === '' ? null : Number(vtRaw),
      gammaFlip: flipRaw === '' ? null : Number(flipRaw),
      source: 'SpotGamma manual'
    };
    if (endpoint) {
      if (!pin) {
        if (feedback) feedback.textContent = 'Introduce el PIN para guardar en GitHub.';
        return;
      }
      if (feedback) feedback.textContent = 'Guardando en GitHub...';
      try {
        await postSpotGammaLevel(endpoint, pin, payload);
        if (feedback) feedback.textContent = `Guardado ${date} en GitHub.`;
      } catch (error) {
        if (feedback) feedback.textContent = `Error guardando en GitHub: ${error.message}`;
        return;
      }
    }
    const local = getLocalSpotGammaData();
    local.byDate[date] = {
      ...payload,
      updatedAt: new Date().toISOString()
    };
    local.lastUpdated = new Date().toISOString();
    saveLocalSpotGammaData(local);
    if (feedback && !endpoint) feedback.textContent = `Guardado ${date} en este navegador.`;
    renderSpotGammaPanel();
  });
  document.getElementById('copySpotGammaJsonBtn').addEventListener('click', async () => {
    const text = document.getElementById('spotGammaJsonExport').value;
    try {
      await navigator.clipboard.writeText(text);
      if (feedback) feedback.textContent = 'JSON copiado.';
    } catch (_) {
      document.getElementById('spotGammaJsonExport').select();
      if (feedback) feedback.textContent = 'Selecciona y copia el JSON manualmente.';
    }
  });
}

function formatSummaryMoney(value) {
  return Number(value).toLocaleString('es-ES', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

async function buildGammaSummaryRows(startDate, endDate) {
  const spotGamma = await fetchSpotGammaData();
  const results = [];
  for (const sessionDate of spotGammaSortedDates(spotGamma, false)) {
    if (!sessionDate || sessionDate < startDate || sessionDate > endDate) continue;
    try {
      const levels = spotGammaEntryToLevels(sessionDate, spotGamma.byDate[sessionDate]);
      const risk = getGammaRiskFlags(levels);
      if (risk.noTradeDay) {
        results.push({ date: sessionDate, status: 'No operable', cls: 'summary-no-trade', pnl: 0 });
        continue;
      }
      levels.entryPremiums = await computeEntryPremiumsForLevels(levels);
      const entryCredit = computeEntryNetCredit(levels);
      if (!Number.isFinite(entryCredit)) continue;
      const skippedWings = levels.entryPremiums && levels.entryPremiums.ok ? gammaSkippedWings(levels.entryPremiums) : [];
      const callActive = !skippedWings.some(item => item.wing === 'call');
      const putActive = !skippedWings.some(item => item.wing === 'put');
      if (!callActive && !putActive) {
        results.push({ date: sessionDate, status: 'No operable', cls: 'summary-no-trade', pnl: 0, source: 'NO COMPENSA' });
        continue;
      }
      const premiumResult = await getPremiumHistoryResult(sessionDate);
      if (premiumResult) {
        const losingDay = premiumResult.pnl < 0;
        results.push({
          date: sessionDate,
          status: losingDay ? 'Malo' : 'Bueno',
          cls: losingDay ? 'summary-bad' : 'summary-good',
          pnl: premiumResult.pnl,
          source: 'Primas monitorizadas'
        });
        continue;
      }
      const { row } = getGammaSessionRow(levels);
      const sellCall = Number(risk.openWall && risk.openWall.sellCall);
      const sellPut = Number(risk.openWall && risk.openWall.sellPut);
      const high = Number(row && row.high);
      const low = Number(row && row.low);
      const opex = getQuarterlyOpexStatus(sessionDate);
      const bad = callActive && [sellCall, high].every(Number.isFinite) && high >= sellCall
        || (putActive && !opex.isOpexDay && [sellPut, low].every(Number.isFinite) && low <= sellPut);
      const losingDay = bad || entryCredit < 0;
      results.push({
        date: sessionDate,
        status: losingDay ? 'Malo' : 'Bueno',
        cls: losingDay ? 'summary-bad' : 'summary-good',
        pnl: (bad ? -(20 - entryCredit) : entryCredit) * 100,
        source: bad ? 'Fallback OHLC: toque de strike' : 'Fallback OHLC: sin toque'
      });
    } catch (e) {
      console.warn('[Resumen] No se pudo calcular', sessionDate, e.message);
    }
  }
  return results;
}

async function generateGammaSummary() {
  const panel = document.getElementById('gammaSummaryPanel');
  const start = document.getElementById('summaryStartDate');
  const end = document.getElementById('summaryEndDate');
  const output = document.getElementById('summaryPeriodOutput');
  if (!panel || !start || !end || !output) return;
  if (!start.value || !end.value || start.value > end.value) {
    output.innerHTML = '<div style="color:var(--bad);padding:10px">Selecciona un periodo válido.</div>';
    return;
  }
  output.innerHTML = '<div style="padding:14px;text-align:center">Generando resumen…</div>';
  const rows = await buildGammaSummaryRows(start.value, end.value);
  const total = rows.reduce((sum, row) => sum + (Number.isFinite(row.pnl) ? row.pnl : 0), 0);
  const body = rows.map(row => `<tr class="${row.cls}">
    <td>${row.date}</td>
    <td>${row.status}</td>
    <td>${row.source || '—'}</td>
    <td>${formatSummaryMoney(row.pnl)}</td>
  </tr>`).join('');
  output.innerHTML = rows.length ? `
    <table class="summary-period-table">
      <thead><tr><th>Fecha</th><th>Resultado</th><th>Fuente</th><th>Primas ganadas / perdidas</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
    <div class="summary-period-total">
      <span>Total del periodo · 1 contrato</span>
      <span style="color:${total >= 0 ? 'var(--good)' : 'var(--bad)'}">${formatSummaryMoney(total)}</span>
    </div>` : '<div style="padding:14px;text-align:center;color:var(--ink-soft)">No hay sesiones disponibles en el periodo seleccionado.</div>';
}

async function renderGammaSummaryPanel() {
  const panel = document.getElementById('gammaSummaryPanel');
  if (!panel) return;
  panel.innerHTML = '<div style="padding:14px;text-align:center">Preparando resumen…</div>';
  const spotGamma = await fetchSpotGammaData();
  const dates = spotGammaSortedDates(spotGamma, false);
  const first = dates[0] || '';
  const last = dates.at(-1) || '';
  panel.innerHTML = `
    <div class="summary-range-controls">
      <label>Desde<input type="date" id="summaryStartDate" value="${first}" min="${first}" max="${last}"></label>
      <label>Hasta<input type="date" id="summaryEndDate" value="${last}" min="${first}" max="${last}"></label>
      <button type="button" id="generateSummaryBtn">Generar resumen</button>
    </div>
    <div style="font-size:11px;color:var(--ink-soft);margin-bottom:10px">
      Resultado por un contrato: si existe histórico de primas monitorizadas se usa el último P&L capturado. Si no existe, se usa el fallback OHLC por toque de strike vendido.
    </div>
    <div id="summaryPeriodOutput"></div>`;
  document.getElementById('generateSummaryBtn').addEventListener('click', generateGammaSummary);
  if (first && last) generateGammaSummary();
}

async function fetchPremiumHistoryIndex() {
  const candidates = [
    `data/premium-history-index.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/premium-history-index.json?t=${Date.now()}`
  ];
  for (const url of candidates) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
    } catch (_) {}
  }
  return { dates: [] };
}

async function fetchPremiumHistory(date) {
  const candidates = [
    `data/premium-history/${date}.json?t=${Date.now()}`,
    `${GITHUB_RAW_BASE}/data/premium-history/${date}.json?t=${Date.now()}`
  ];
  let lastError = null;
  for (const url of candidates) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error('histórico no disponible');
}

async function getPremiumHistoryResult(date) {
  try {
    const history = await fetchPremiumHistory(date);
    const snapshots = Array.isArray(history.snapshots) ? history.snapshots : [];
    if (history.status !== 'active' || !snapshots.length) return null;
    const last = snapshots.at(-1);
    const pnl = Number(last && last.pnl);
    if (!Number.isFinite(pnl)) return null;
    return {
      pnl,
      closeCost: Number(last.closeCost),
      multiple: last.multiple,
      capturedAt: last.capturedAt,
      snapshots: snapshots.length,
      source: 'premium-history'
    };
  } catch (_) {
    return null;
  }
}

function premiumLegLabel(name, leg) {
  const action = leg.side === 'sell' ? 'Venta' : 'Compra';
  const type = leg.type === 'call' ? 'Call' : 'Put';
  return `${action} ${type} ${fmtGammaNum(leg.strike, 0)}`;
}

async function drawPremiumHistory(date) {
  const output = document.getElementById('premiumHistoryOutput');
  if (!output) return;
  output.innerHTML = '<div style="padding:14px;text-align:center">Cargando evolución de primas…</div>';
  try {
    const history = await fetchPremiumHistory(date);
    if (history.status !== 'active') {
      output.innerHTML = `<div style="padding:14px;background:#ececec;border-radius:6px;color:#666">
        <b>${date} · No operable</b><br>${history.reason || 'Sesión excluida por los filtros de riesgo.'}
      </div>`;
      return;
    }
    const snapshots = history.snapshots || [];
    if (!snapshots.length) {
      output.innerHTML = '<div style="padding:14px;text-align:center;color:var(--ink-soft)">Todavía no hay capturas para esta sesión.</div>';
      return;
    }
    const last = snapshots.at(-1);
    const pnlColor = Number(last.pnl) >= 0 ? 'var(--good)' : 'var(--bad)';
    output.innerHTML = `
      <div class="premium-history-stats">
        <div class="premium-history-stat"><div>Crédito inicial</div><b>${fmtGammaNum(history.entryCredit, 2)} puntos</b></div>
        <div class="premium-history-stat"><div>Coste de cierre actual</div><b>${fmtGammaNum(last.closeCost, 2)} puntos</b></div>
        <div class="premium-history-stat"><div>P&L estimado · 1 contrato</div><b style="color:${pnlColor}">${formatSummaryMoney(last.pnl)}</b></div>
        <div class="premium-history-stat"><div>Múltiplo sobre crédito</div><b>${Number.isFinite(Number(last.multiple)) ? Number(last.multiple).toFixed(2) + '×' : '—'}</b></div>
        <div class="premium-history-stat"><div>Capturas</div><b>${snapshots.length}</b></div>
      </div>
      <div id="premiumHistoryChart" style="width:100%;height:460px"></div>
      <div style="font-size:10px;color:var(--ink-soft);margin-top:8px">
        Cada línea representa el midpoint bid/ask de una pata. Las horas efectivas descuentan aproximadamente 15 minutos por el retraso de Cboe.
      </div>`;
    const colors = {
      sell_call: '#2f6fb0',
      buy_call: '#65a9e8',
      sell_put: '#b73232',
      buy_put: '#e58b8b'
    };
    const traces = Object.entries(history.legs || {}).map(([name, leg]) => ({
      x: snapshots.map(snapshot => snapshot.effectiveAt),
      y: snapshots.map(snapshot => Number(snapshot.quotes?.[name]?.mid)),
      type: 'scatter',
      mode: 'lines+markers',
      name: premiumLegLabel(name, leg),
      line: { color: colors[name], width: leg.side === 'sell' ? 3 : 2, dash: leg.side === 'sell' ? 'solid' : 'dot' },
      marker: { size: 5 },
      hovertemplate: '%{x|%H:%M}<br>%{y:.2f} puntos<extra>%{fullData.name}</extra>'
    }));
    traces.push({
      x: snapshots.map(snapshot => snapshot.effectiveAt),
      y: snapshots.map(snapshot => Number(snapshot.closeCost)),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Coste cierre IC',
      line: { color: '#0a2540', width: 4 },
      marker: { size: 6 },
      hovertemplate: '%{x|%H:%M}<br>%{y:.2f} puntos<extra>%{fullData.name}</extra>'
    });
    safePlotly('premiumHistoryChart', traces, {
      margin: { t: 25, r: 25, b: 55, l: 65 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#f3f8fc',
      font: { family: 'Segoe UI, sans-serif', color: '#0c1f33' },
      xaxis: { title: 'Hora efectiva aproximada', type: 'date', gridcolor: '#dce7f1' },
      yaxis: { title: 'Prima · midpoint bid/ask', gridcolor: '#dce7f1', rangemode: 'tozero' },
      legend: { orientation: 'h', y: 1.12 },
      hovermode: 'x unified'
    }, { responsive: true, displayModeBar: false });
  } catch (error) {
    output.innerHTML = `<div style="padding:14px;color:var(--bad)">No se pudo cargar ${date}: ${error.message}</div>`;
  }
}

async function renderPremiumHistoryPanel() {
  const panel = document.getElementById('premiumHistoryPanel');
  if (!panel) return;
  panel.innerHTML = '<div style="padding:14px;text-align:center">Buscando sesiones monitorizadas…</div>';
  const index = await fetchPremiumHistoryIndex();
  const dates = (index.dates || []).slice().sort().reverse();
  if (!dates.length) {
    panel.innerHTML = `<div style="padding:14px;background:var(--blue-50);border:1px solid var(--blue-200);border-radius:6px;color:var(--ink-soft)">
      Todavía no hay sesiones monitorizadas. El histórico empezará a generarse automáticamente en la próxima sesión de mercado.
    </div>`;
    return;
  }
  panel.innerHTML = `
    <div class="premium-history-controls">
      <label>Sesión
        <select id="premiumHistoryDate">${dates.map(date => `<option value="${date}">${date}</option>`).join('')}</select>
      </label>
      <button type="button" id="loadPremiumHistoryBtn">Ver gráfico</button>
    </div>
    <div id="premiumHistoryOutput"></div>`;
  const select = document.getElementById('premiumHistoryDate');
  document.getElementById('loadPremiumHistoryBtn').addEventListener('click', () => drawPremiumHistory(select.value));
  drawPremiumHistory(select.value);
}

function getGammaSessionRow(chainDate) {
  const target = gammaTargetSessionDate(chainDate);
  if (!target || !currentRows) return { target, row: null };
  return { target, row: currentRows.find(r => r.date === target) || null };
}

function getGammaOpeningGap(chainDate) {
  const target = gammaNextSessionDate(chainDate);
  if (!target || !currentRows) return { target, ok: false, error: 'sin fecha objetivo' };
  const idx = currentRows.findIndex(r => r.date === target);
  if (idx < 1) return { target, ok: false, error: 'sin cierre previo' };
  const row = currentRows[idx];
  const prev = currentRows[idx - 1];
  const open = Number(row.open);
  const prevClose = Number(prev.close);
  if (!Number.isFinite(open) || !Number.isFinite(prevClose) || prevClose <= 0) {
    return { target, ok: false, error: 'open o cierre previo no disponible' };
  }
  const pct = (open - prevClose) / prevClose * 100;
  return {
    ok: true,
    target,
    open,
    prevDate: prev.date,
    prevClose,
    pct,
    alert: Math.abs(pct) >= 0.50
  };
}

function renderGammaSessionChart(levels) {
  const { target, row } = getGammaSessionRow(levels);
  const callWall = levels.callWall ? Number(levels.callWall.strike) : NaN;
  const putWall = levels.putWall ? Number(levels.putWall.strike) : NaN;
  if (!target) return '';
  if (!row) {
    return `<div style="background:var(--blue-50);border:1px dashed var(--blue-200);border-radius:5px;padding:10px;font-size:12px;color:var(--ink-soft)">
      <b>Sesión ${target}</b><br>No hay OHLC cargado todavía para comprobar el día.
    </div>`;
  }

  const o = Number(row.open), h = Number(row.high), l = Number(row.low), c = Number(row.close);
  if (![o, h, l, c, callWall, putWall].every(Number.isFinite)) {
    return `<div style="background:var(--blue-50);border:1px dashed var(--blue-200);border-radius:5px;padding:10px;font-size:12px;color:var(--ink-soft)">
      <b>Sesión ${target}</b><br>OHLC incompleto para dibujar el gráfico.
    </div>`;
  }

  const vals = [o, h, l, c, callWall, putWall];
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const pad = Math.max((maxV - minV) * 0.12, 10);
  const yMin = minV - pad;
  const yMax = maxV + pad;
  const y = v => 116 - ((v - yMin) / (yMax - yMin || 1)) * 92;
  const xs = [28, 92, 156, 220];
  const points = [[xs[0], y(o)], [xs[1], y(h)], [xs[2], y(l)], [xs[3], y(c)]];
  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  const yCall = y(callWall);
  const yPut = y(putWall);
  const opex = getQuarterlyOpexStatus(target);
  const closeInside = opex.isOpexDay ? c <= callWall : (c <= callWall && c >= putWall);
  const rangeInside = opex.isOpexDay ? h <= callWall : (h <= callWall && l >= putWall);
  const statusColor = rangeInside ? 'var(--good)' : 'var(--bad)';
  const statusText = opex.isOpexDay
    ? (rangeInside ? 'Call spread dentro' : 'Tocó Call Wall')
    : (rangeInside ? 'Sesión dentro' : 'Tocó wall');
  const closeText = opex.isOpexDay
    ? (closeInside ? 'Cierre bajo CW' : 'Cierre sobre CW')
    : (closeInside ? 'Cierre dentro' : 'Cierre fuera');

  return `<div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px">
    <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;margin-bottom:6px">
      <div>
        <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Comprobación OHLC</div>
        <b style="font-size:13px">${target}</b>
      </div>
      <div style="text-align:right;font-size:10px;color:${statusColor};font-weight:700">
        ${statusText}<br><span style="color:${closeInside ? 'var(--good)' : 'var(--bad)'}">${closeText}</span>
      </div>
    </div>
    <svg viewBox="0 0 248 132" width="100%" height="132" role="img" aria-label="OHLC contra Call Wall y Put Wall">
      <line x1="18" x2="232" y1="${yCall.toFixed(1)}" y2="${yCall.toFixed(1)}" stroke="#2f6fb0" stroke-width="2" stroke-dasharray="5 4"/>
      <line x1="18" x2="232" y1="${yPut.toFixed(1)}" y2="${yPut.toFixed(1)}" stroke="#7b4ab8" stroke-width="2" stroke-dasharray="5 4"/>
      <path d="${path}" fill="none" stroke="#c9a227" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      ${points.map((p, i) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${i === 3 ? 4 : 3}" fill="${i === 3 ? (closeInside ? '#23824d' : '#b73232') : '#c9a227'}"/>`).join('')}
      <text x="20" y="${Math.max(10, yCall - 5).toFixed(1)}" font-size="9" fill="#2f6fb0">Call Wall ${fmtGammaNum(callWall, 0)}</text>
      <text x="20" y="${Math.min(126, yPut + 12).toFixed(1)}" font-size="9" fill="#7b4ab8">Put Wall ${fmtGammaNum(putWall, 0)}</text>
      ${['O','H','L','C'].map((label, i) => `<text x="${xs[i]}" y="128" text-anchor="middle" font-size="9" fill="#496276">${label}</text>`).join('')}
    </svg>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;font-size:10px;color:var(--ink-soft);text-align:center">
      <span>O ${fmtGammaNum(o, 0)}</span><span>H ${fmtGammaNum(h, 0)}</span><span>L ${fmtGammaNum(l, 0)}</span><span>C ${fmtGammaNum(c, 0)}</span>
    </div>
  </div>`;
}

function renderGammaPremiumCard(levels) {
  const p = levels.entryPremiums;
  if (!p) return '';
  if (!p.ok) {
    return `<div style="background:var(--blue-50);border:1px dashed var(--blue-200);border-radius:5px;padding:10px;font-size:12px;color:var(--ink-soft);margin-top:8px">
      <b>Primas entrada ${p.entryDate || '—'}</b><br>No hay cadena de entrada para calcular ventas: ${p.error || 'sin datos'}.
    </div>`;
  }
  const callWarn = p.call.dist > 0 ? ` <span style="color:var(--bad);font-size:10px">(aprox. ${fmtGammaNum(p.call.targetStrike, 0)})</span>` : '';
  const putWarn = p.put.dist > 0 ? ` <span style="color:var(--bad);font-size:10px">(aprox. ${fmtGammaNum(p.put.targetStrike, 0)})</span>` : '';
  const callProtWarn = p.callProtection.dist > 0 ? ` <span style="color:var(--bad);font-size:10px">(aprox. ${fmtGammaNum(p.callProtection.targetStrike, 0)})</span>` : '';
  const putProtWarn = p.putProtection.dist > 0 ? ` <span style="color:var(--bad);font-size:10px">(aprox. ${fmtGammaNum(p.putProtection.targetStrike, 0)})</span>` : '';
  const opex = getQuarterlyOpexStatus(p.entryDate);
  const skippedWings = gammaSkippedWings(p);
  const callSkipped = skippedWings.some(item => item.wing === 'call');
  const putSkipped = skippedWings.some(item => item.wing === 'put');
  const callCredit = optionQuote(p.call, 'bid') - optionQuote(p.callProtection, 'ask');
  const putCredit = optionQuote(p.put, 'bid') - optionQuote(p.putProtection, 'ask');
  const grossCredit = (callSkipped ? 0 : (Number(p.call.bid) || 0)) + (opex.isOpexDay || putSkipped ? 0 : (Number(p.put.bid) || 0));
  const protectionCost = (callSkipped ? 0 : (Number(p.callProtection.ask) || 0)) + (opex.isOpexDay || putSkipped ? 0 : (Number(p.putProtection.ask) || 0));
  const netCredit = grossCredit - protectionCost;
  const noCompensaRows = skippedWings.map(item => `
    <div style="background:rgba(183,50,50,0.10);border:1px solid rgba(183,50,50,0.35);border-radius:4px;padding:6px 8px;color:var(--bad);font-weight:800">
      ${item.label}: NO COMPENSA (${fmtGammaNum(item.credit * 100, 2)} USD)
    </div>`).join('');
  return `<div style="background:var(--blue-50);border:1px solid var(--gold-500);border-radius:5px;padding:10px;font-size:12px;margin-top:8px">
    <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start;margin-bottom:8px">
      <div>
        <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">${opex.isOpexDay ? 'Call spread OPEX en cadena de entrada' : 'Iron Condor en cadena de entrada'}</div>
        <b>${p.entryDate}</b>
      </div>
      <div style="text-align:right;color:var(--gold-700);font-weight:700">
        Neto ${fmtGammaNum(netCredit, 2)}
      </div>
    </div>
    <div style="display:grid;gap:6px">
      <div style="display:flex;justify-content:space-between;gap:8px">
        <span>Venta de call ${fmtGammaNum(p.call.strike, 0)}${callWarn}</span>
        <b class="call-cell">${callSkipped ? 'NO COMPENSA' : fmtGammaNum(p.call.bid, 2)}</b>
      </div>
      <div style="display:flex;justify-content:space-between;gap:8px">
        <span>Compra call ${fmtGammaNum(p.callProtection.strike, 0)}${callProtWarn}</span>
        <b class="call-cell">${callSkipped ? `Crédito ${fmtGammaNum(callCredit * 100, 2)} USD` : '-' + fmtGammaNum(p.callProtection.ask, 2)}</b>
      </div>
      ${noCompensaRows}
      ${opex.isOpexDay ? `
        <div style="background:rgba(201,162,39,0.12);border:1px solid var(--gold-500);border-radius:4px;padding:6px 8px;color:var(--gold-700);font-weight:700">
          Hora Bruja Trimestral: no se abre el lado put.
        </div>` : `
        <div style="display:flex;justify-content:space-between;gap:8px">
          <span>Venta de put ${fmtGammaNum(p.put.strike, 0)}${putWarn}</span>
          <b class="put-cell">${putSkipped ? 'NO COMPENSA' : fmtGammaNum(p.put.bid, 2)}</b>
        </div>
        <div style="display:flex;justify-content:space-between;gap:8px">
          <span>Compra put ${fmtGammaNum(p.putProtection.strike, 0)}${putProtWarn}</span>
          <b class="put-cell">${putSkipped ? `Crédito ${fmtGammaNum(putCredit * 100, 2)} USD` : '-' + fmtGammaNum(p.putProtection.ask, 2)}</b>
        </div>`}
      <div style="border-top:1px solid var(--blue-200);margin-top:2px;padding-top:6px;display:grid;gap:4px">
        <div style="display:flex;justify-content:space-between;gap:8px;color:var(--ink-soft)">
          <span>Prima recibida ventas</span><b>${fmtGammaNum(grossCredit, 2)}</b>
        </div>
        <div style="display:flex;justify-content:space-between;gap:8px;color:var(--ink-soft)">
          <span>Coste protecciones</span><b>-${fmtGammaNum(protectionCost, 2)}</b>
        </div>
        <div style="display:flex;justify-content:space-between;gap:8px;font-size:13px;color:${netCredit >= 0 ? 'var(--good)' : 'var(--bad)'}">
          <span><b>Ingresado neto</b></span><b>${fmtGammaNum(netCredit, 2)}</b>
        </div>
      </div>
    </div>
    <div style="font-size:10px;color:var(--ink-soft);margin-top:8px">
      ${opex.isOpexDay ? 'Operativa especial OPEX: solo lado call · ' : ''}Protección: 20 puntos más OTM · Fuente: ${p.sourceLabel} · Spot entrada: ${fmtGammaNum(p.spot, 2)} · Captura: ${formatMadridTime(p.capturedAt)}
    </div>
  </div>`;
}

function getGammaRiskFlags(levels) {
  const openWall = getGammaOpenWallSetup(levels);
  return {
    openWall,
    openWallPctRaw: openWall.openPct,
    noTradeDay: !openWall.ok,
    reason: openWall.reason
  };
}

function computeGammaHitStats(results, startChainDate = '2026-06-01') {
  const stats = {
    startDate: startChainDate,
    sessionsAnalyzed: 0,
    total: 0,
    wins: 0,
    losses: 0,
    upperTouches: 0,
    lowerTouches: 0,
    pending: 0
  };
  for (const r of results) {
    if (!r.ok || !r.levels) continue;
    if (!r.levels.date || r.levels.date < startChainDate) continue;
    const { row } = getGammaSessionRow(r.levels);
    const callWall = r.levels.callWall ? Number(r.levels.callWall.strike) : NaN;
    const putWall = r.levels.putWall ? Number(r.levels.putWall.strike) : NaN;
    const high = row ? Number(row.high) : NaN;
    const low = row ? Number(row.low) : NaN;
    if (![callWall, putWall, high, low].every(Number.isFinite)) {
      stats.pending++;
      continue;
    }
    stats.sessionsAnalyzed++;
    const risk = getGammaRiskFlags(r.levels);
    if (risk.noTradeDay) continue;

    stats.total++;
    const sellCall = Number(risk.openWall && risk.openWall.sellCall);
    const sellPut = Number(risk.openWall && risk.openWall.sellPut);
    const upperTouch = Number.isFinite(sellCall) && high >= sellCall;
    const opex = getQuarterlyOpexStatus(gammaTargetSessionDate(r.levels));
    const lowerTouch = !opex.isOpexDay && Number.isFinite(sellPut) && low <= sellPut;
    if (upperTouch) stats.upperTouches++;
    if (lowerTouch) stats.lowerTouches++;
    if (upperTouch || lowerTouch) stats.losses++;
    else stats.wins++;
  }
  return stats;
}

function applyPostLossCapitalWarnings(results, startChainDate = '2026-06-01') {
  const chronological = results
    .filter(r => r.ok && r.levels && r.levels.date >= startChainDate)
    .slice()
    .sort((a, b) => String(a.levels.date).localeCompare(String(b.levels.date)));
  let previousOperableFailed = false;

  for (const r of chronological) {
    const levels = r.levels;
    levels.reduceCapitalAfterLoss = false;
    const risk = getGammaRiskFlags(levels);
    if (risk.noTradeDay) continue;

    // NO OPERAR sessions between two trades do not consume the warning.
    levels.reduceCapitalAfterLoss = previousOperableFailed;

    const { row } = getGammaSessionRow(levels);
    const sellCall = Number(risk.openWall && risk.openWall.sellCall);
    const sellPut = Number(risk.openWall && risk.openWall.sellPut);
    const high = row ? Number(row.high) : NaN;
    const low = row ? Number(row.low) : NaN;
    if (![sellCall, sellPut, high, low].every(Number.isFinite)) {
      // The next operable session has been identified, but its result is pending.
      break;
    }
    const opex = getQuarterlyOpexStatus(gammaTargetSessionDate(levels));
    const upperTouch = high >= sellCall;
    const lowerTouch = !opex.isOpexDay && low <= sellPut;
    previousOperableFailed = upperTouch || lowerTouch;
  }
}

function renderGammaHitStats(stats) {
  const winRate = stats.total > 0 ? (stats.wins / stats.total * 100).toFixed(1) + '%' : '—';
  return `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:10px 0 4px">
    <div style="background:#fff;border:1px solid var(--blue-200);border-radius:5px;padding:8px">
      <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Sesiones analizadas</div>
      <b>${stats.sessionsAnalyzed}</b>
    </div>
    <div style="background:#fff;border:1px solid var(--blue-200);border-radius:5px;padding:8px">
      <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Operaciones</div>
      <b>${stats.total}</b>
    </div>
    <div style="background:#fff;border:1px solid var(--blue-200);border-radius:5px;padding:8px">
      <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Aciertos</div>
      <b style="color:var(--good)">${stats.wins}</b>
    </div>
    <div style="background:#fff;border:1px solid var(--blue-200);border-radius:5px;padding:8px">
      <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Fallos</div>
      <b style="color:var(--bad)">${stats.losses}</b>
    </div>
    <div style="background:#fff;border:1px solid var(--blue-200);border-radius:5px;padding:8px">
      <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Toques arriba</div>
      <b style="color:var(--bad)">${stats.upperTouches}</b>
    </div>
    <div style="background:#fff;border:1px solid var(--blue-200);border-radius:5px;padding:8px">
      <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Toques abajo</div>
      <b style="color:var(--bad)">${stats.lowerTouches}</b>
    </div>
    <div style="background:#fff;border:1px solid var(--blue-200);border-radius:5px;padding:8px">
      <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Win rate</div>
      <b>${winRate}</b>
    </div>
  </div>
  <div style="font-size:10px;color:var(--ink-soft);margin-top:6px">
    Contador desde cadena ${stats.startDate}. Las sesiones marcadas NO OPERAR cuentan como analizadas, pero no como operaciones. Pendientes sin OHLC: ${stats.pending}.
  </div>`;
}

function renderGammaCard(levels) {
  const cw = levels.callWall;
  const pw = levels.putWall;
  const sessionLabel = gammaNextSessionLabel(levels.date);
  const tone = levels.netAtSpot >= 0 ? 'var(--good)' : 'var(--bad)';
  const risk = getGammaRiskFlags(levels);
  const openWall = risk.openWall;
  const openWallPct = Number(openWall && openWall.openPct);
  const openWallPctClamped = Number.isFinite(openWallPct)
    ? Math.max(0, Math.min(100, openWallPct))
    : 0;
  const openWallLabel = Number.isFinite(openWallPct) ? `${openWallPct.toFixed(2)}%` : '—';
  const noTradeDay = risk.noTradeDay;
  const sessionDate = gammaTargetSessionDate(levels);
  const opex = getQuarterlyOpexStatus(sessionDate);
  const opexMessage = opex.isOpexDay
    ? `<div style="background:rgba(201,162,39,0.16);border:1px solid var(--gold-500);border-left:5px solid var(--gold-500);border-radius:5px;padding:8px 10px;margin-bottom:10px;font-size:12px;color:var(--gold-700)"><b>HORA BRUJA TRIMESTRAL:</b> hoy solo se vende la CW con su protección. No se abre la PW.</div>`
    : opex.isOpexWeek
      ? `<div style="background:rgba(201,162,39,0.10);border:1px solid var(--gold-500);border-radius:5px;padding:7px 10px;margin-bottom:10px;font-size:12px;color:var(--gold-700)"><b>Semana de vencimiento trimestral OPEX</b> · Fecha principal: ${formatSpanishLongDate(opex.event.date)}</div>`
      : '';
  const capitalWarning = levels.reduceCapitalAfterLoss && !noTradeDay
    ? `<div style="background:rgba(207,76,76,0.12);border:1px solid var(--bad);border-left:5px solid var(--bad);border-radius:5px;padding:8px 10px;margin-bottom:10px;font-size:12px;color:var(--bad)"><b>GESTIÓN DE RIESGO:</b> solo operar con el 50% del capital destinado. Es la siguiente operación ejecutable después de un fallo.</div>`
    : '';
  const dayBorder = noTradeDay ? '2px solid var(--bad)' : '1px solid var(--blue-200)';
  const dayShadow = noTradeDay ? '0 0 0 2px rgba(207,76,76,0.08)' : 'none';
  const sessionChart = renderGammaSessionChart(levels);
  const premiumCard = renderGammaPremiumCard(levels);
  const topRows = (levels.topRows || []).map(r => `<tr>
    <td>${fmtGammaNum(r.strike, 0)}</td>
    <td class="call-cell">${fmtGammaNum(r.callOi, 0)}</td>
    <td class="call-cell">${fmtGammaNum(r.callGex, 0)}</td>
    <td class="put-cell">${fmtGammaNum(r.putOi, 0)}</td>
    <td class="put-cell">${fmtGammaNum(r.putGex, 0)}</td>
    <td>${fmtGammaNum(r.netGex, 0)}</td>
  </tr>`).join('');
  return `
    <details class="auto-chain-item" open style="display:block;padding:0;margin-bottom:12px;border:${dayBorder};box-shadow:${dayShadow}">
      <summary style="cursor:pointer;padding:10px 12px;font-weight:700;color:var(--navy-700)">
        ${noTradeDay ? 'NO OPERAR · ' : ''}${sessionLabel} · Niveles ${levels.sourceLabel} · Call Wall ${fmtGammaNum(cw && cw.strike, 0)} · Put Wall ${fmtGammaNum(pw && pw.strike, 0)}
      </summary>
      <div style="padding:0 12px 12px">
        ${opexMessage}
        ${capitalWarning}
        <div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(260px,330px);gap:12px;align-items:start">
          <div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px;margin-bottom:10px">
              <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:8px">
                <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Call Wall</div>
                <b>${fmtGammaNum(cw && cw.strike, 0)}</b>
              </div>
              <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:8px">
                <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Put Wall</div>
                <b>${fmtGammaNum(pw && pw.strike, 0)}</b>
              </div>
              <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:8px">
                <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">Net GEX spot</div>
                <b style="color:${tone}">${fmtGammaNum(levels.netAtSpot, 0)}</b>
              </div>
              <div style="background:${noTradeDay ? 'rgba(207,76,76,0.12)' : 'var(--blue-50)'};border:1px solid ${noTradeDay ? 'var(--bad)' : 'var(--gold-500)'};border-radius:5px;padding:8px">
                <div style="font-size:10px;color:var(--ink-soft);text-transform:uppercase">% Open en rango</div>
                <b style="color:${noTradeDay ? 'var(--bad)' : 'inherit'}">${noTradeDay ? '⚠ ' : ''}${openWallLabel}</b>
                <div style="height:7px;background:rgba(12,45,78,0.14);border-radius:99px;margin-top:6px;overflow:hidden">
                  <div style="height:100%;width:${openWallPctClamped}%;background:${noTradeDay ? 'var(--bad)' : 'linear-gradient(90deg,var(--blue-500),var(--gold-500))'};border-radius:99px"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--ink-soft);margin-top:3px">
                  <span>Put Wall</span><span>Call Wall</span>
                </div>
                <div style="font-size:9px;color:var(--ink-soft);margin-top:4px">
                  ${openWall.ok
                    ? `Open ${fmtGammaNum(openWall.open, 2)} · Ajuste: ${openWall.adjustment} · Venta C ${fmtGammaNum(openWall.sellCall, 0)} / P ${fmtGammaNum(openWall.sellPut, 0)}`
                    : openWall.reason}
                </div>
              </div>
            </div>
            <div style="font-size:11px;color:var(--ink-soft);margin-bottom:8px">
              Fuente: ${levels.sourceLabel}${levels.sessionDate ? ` · Sesión aplicada: ${levels.sessionDate}` : ''}${levels.volTrigger ? ` · VT ${fmtGammaNum(levels.volTrigger, 0)}` : ''}${levels.gammaFlip ? ` · Gamma Flip ${fmtGammaNum(levels.gammaFlip, 0)}` : ''}
            </div>
            ${topRows ? `<table class="chain-table">
              <thead><tr>
                <th>Strike</th><th>Call OI</th><th>Call GEX</th><th>Put OI</th><th>Put GEX</th><th>Net GEX</th>
              </tr></thead>
              <tbody>${topRows}</tbody>
            </table>` : ''}
          </div>
          <div>${sessionChart}${premiumCard}</div>
        </div>
      </div>
    </details>`;
}

async function renderGammaLevelsPanel() {
  const panel = document.getElementById('gammaLevelsPanel');
  if (!panel) return;
  panel.innerHTML = '<div style="padding:14px;text-align:center">Cargando niveles SpotGamma...</div>';
  const spotGamma = await fetchSpotGammaData();
  const gammaDates = spotGammaSortedDates(spotGamma, true);
  if (!gammaDates.length) {
    panel.innerHTML = '<div style="padding:14px;text-align:center;color:var(--bad)">No hay niveles SpotGamma cargados.</div>';
    return;
  }
  const results = [];
  for (const date of gammaDates) {
    try {
      const levels = spotGammaEntryToLevels(date, spotGamma.byDate[date]);
      levels.entryPremiums = await computeEntryPremiumsForLevels(levels);
      results.push({ ok: true, date, levels });
    } catch (e) {
      results.push({ ok: false, date, error: e.message });
    }
  }
  applyPostLossCapitalWarnings(results, '2026-06-01');
  const cards = results.map(r => r.ok
    ? renderGammaCard(r.levels)
    : `<div class="auto-chain-item" style="color:var(--bad)">No se pudo calcular ${r.date}: ${r.error}</div>`
  ).join('');
  const datePills = gammaDates.map(date => {
    const ok = results.find(r => r.date === date && r.ok);
    const color = ok ? 'var(--good)' : 'var(--bad)';
    const bg = ok ? 'rgba(39,174,96,0.12)' : 'rgba(207,76,76,0.12)';
    return `<span style="display:inline-block;border:1px solid ${color};background:${bg};color:${color};border-radius:4px;padding:3px 7px;margin:2px;font-weight:700">${date}</span>`;
  }).join('');
  const gammaStats = computeGammaHitStats(results, '2026-06-01');
  panel.innerHTML = `
    <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px 14px;margin-bottom:12px;font-size:12px">
      <b>Niveles SpotGamma manuales</b> · Estas walls ya no se calculan desde CBOE; se cargan desde la pestaña SpotGamma.
      Filtro operativo: el Open debe quedar entre el 10% y el 90% del rango Put Wall → Call Wall.
      <div style="margin-top:8px">
        <b>Sesiones cargadas:</b> ${datePills}
      </div>
      ${renderGammaHitStats(gammaStats)}
    </div>
    ${cards}`;
}

async function renderGammaChartsPanel() {
  const panel = document.getElementById('gammaChartsPanel');
  if (!panel) return;
  panel.innerHTML = '<div style="padding:14px;text-align:center">Calculando evolución NET GEX SPOT…</div>';

  const gammaDates = (await fetchGammaIndexDates())
    .filter(date => date >= '2026-06-01')
    .sort((a, b) => String(a).localeCompare(String(b)));
  const points = [];
  for (const date of gammaDates) {
    try {
      const chain = await fetchGammaChain(date);
      const levels = computeGammaLevels(chain);
      if (levels && Number.isFinite(Number(levels.netAtSpot))) {
        points.push({
          date,
          value: Number(levels.netAtSpot),
          sessionDate: gammaNextSessionDate(date)
        });
      }
    } catch (e) {
      console.warn('[Gamma charts] No se pudo calcular', date, e.message);
    }
  }

  if (!points.length) {
    panel.innerHTML = '<div style="padding:14px;text-align:center;color:var(--bad)">No hay cadenas de cierre desde el 1 de junio para graficar NET GEX SPOT.</div>';
    return;
  }

  panel.innerHTML = `
    <div style="background:var(--blue-50);border:1px solid var(--blue-200);border-radius:5px;padding:10px 14px;margin-bottom:12px;font-size:12px">
      <b>Evolución NET GEX SPOT</b> · ${points.length} cadenas de cierre desde el 1 de junio de 2026.
    </div>
    <div id="netGexSpotChart" style="width:100%;height:430px"></div>
    <div style="background:rgba(201,162,39,0.12);border:1px solid var(--gold-500);border-left:4px solid var(--gold-500);border-radius:5px;padding:11px 14px;margin-top:12px;font-size:12px;color:var(--ink)">
      <b>Aviso:</b> Pendiente de encontrar el nivel de gamma positiva para aplicar el filtro de las celdas BE1:BJ8 del excel
    </div>`;

  const trace = {
    x: points.map(p => p.date),
    y: points.map(p => p.value),
    type: 'scatter',
    mode: 'lines+markers',
    name: 'NET GEX SPOT',
    customdata: points.map(p => p.sessionDate),
    line: { color: '#6b4bb6', width: 3 },
    marker: {
      size: 9,
      color: points.map(p => p.value >= 0 ? '#23824d' : '#b73232'),
      line: { color: '#ffffff', width: 1 }
    },
    hovertemplate: 'Cadena: %{x}<br>Para sesión: %{customdata}<br>NET GEX SPOT: %{y:,.0f}<extra></extra>'
  };
  const layout = {
    margin: { t: 25, r: 25, b: 55, l: 85 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#f3f8fc',
    font: { family: 'Segoe UI, sans-serif', color: '#0c1f33' },
    xaxis: { title: 'Fecha cadena de cierre', type: 'date', gridcolor: '#dce7f1' },
    yaxis: { title: 'NET GEX SPOT', zeroline: true, zerolinecolor: '#b73232', zerolinewidth: 2, gridcolor: '#dce7f1' },
    shapes: [{
      type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 0, y1: 0,
      line: { color: '#b73232', width: 2, dash: 'dash' }
    }],
    showlegend: false,
    hovermode: 'x unified'
  };
  safePlotly('netGexSpotChart', [trace], layout, { responsive: true, displayModeBar: false });
}

function initChainTabs() {
  let activeHistoryTab = 'entry';

  function activateChainTab(tab, trigger) {
      const mainTab = tab === 'entry' || tab === 'close' ? 'history' : tab;
      document.querySelectorAll('.chain-auto-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.chainTab === mainTab);
      });
      document.querySelectorAll('.chain-history-tab').forEach(b => {
        b.classList.toggle('active', b.dataset.chainTab === tab);
      });
      document.querySelectorAll('.chain-history-node').forEach(node => {
        node.classList.toggle('open', mainTab === 'history');
      });
      const entry = document.getElementById('autoChainsList');
      const close = document.getElementById('autoCloseChainsList');
      const spotGamma = document.getElementById('spotGammaPanel');
      const gamma = document.getElementById('gammaLevelsPanel');
      const charts = document.getElementById('gammaChartsPanel');
      const dataInput = document.getElementById('gammaDataInputPanel');
      const summary = document.getElementById('gammaSummaryPanel');
      const premiumHistory = document.getElementById('premiumHistoryPanel');
      if (entry) entry.style.display = tab === 'entry' ? '' : 'none';
      if (close) close.style.display = tab === 'close' ? '' : 'none';
      if (spotGamma) spotGamma.style.display = tab === 'spotgamma' ? '' : 'none';
      if (gamma) gamma.style.display = tab === 'gamma' ? '' : 'none';
      if (charts) charts.style.display = tab === 'charts' ? '' : 'none';
      if (dataInput) dataInput.style.display = tab === 'data-input' ? '' : 'none';
      if (summary) summary.style.display = tab === 'summary' ? '' : 'none';
      if (premiumHistory) premiumHistory.style.display = tab === 'premium-history' ? '' : 'none';
      const preview = document.getElementById('chainPreview');
      if (preview) preview.style.display = 'none';
      if (tab === 'gamma' && gamma) {
        renderGammaLevelsPanel();
      }
      if (tab === 'spotgamma' && spotGamma) {
        renderSpotGammaPanel();
      }
      if (tab === 'charts' && charts) {
        renderGammaChartsPanel();
      }
      if (tab === 'data-input' && dataInput) {
        renderGammaDataInputPanel();
      }
      if (tab === 'summary' && summary) {
        renderGammaSummaryPanel();
      }
      if (tab === 'premium-history' && premiumHistory) {
        renderPremiumHistoryPanel();
      }
  }

  document.querySelectorAll('.chain-auto-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.chainTab;
      activateChainTab(tab === 'history' ? activeHistoryTab : tab, btn);
    });
  });

  document.querySelectorAll('.chain-history-tab').forEach(btn => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      activeHistoryTab = btn.dataset.chainTab;
      activateChainTab(activeHistoryTab, btn);
    });
  });
}

function initResearchCore() {
  const button = document.getElementById('researchCoreBtn');
  const overlay = document.getElementById('researchOverlay');
  const close = document.getElementById('researchCloseBtn');
  const detail = document.getElementById('researchDetail');
  const back = document.getElementById('researchBackBtn');
  if (!button || !overlay || !close) return;

  function openResearch() {
    overlay.classList.add('open');
    overlay.setAttribute('aria-hidden', 'false');
    overlay.scrollTop = 0;
    close.focus();
  }

  function closeDetail() {
    if (!detail) return;
    detail.classList.remove('open');
    detail.setAttribute('aria-hidden', 'true');
  }

  function closeResearch() {
    closeDetail();
    overlay.classList.remove('open');
    overlay.setAttribute('aria-hidden', 'true');
    button.focus();
  }

  function openDetail(section) {
    if (!detail) return;
    detail.classList.add('open');
    detail.setAttribute('aria-hidden', 'false');
    overlay.scrollTop = 0;
    detail.scrollTop = 0;
    if (detail.parentElement) detail.parentElement.scrollTop = 0;
    if (section === 'gex') {
      renderGexLab();
    } else if (section === 'dex') {
      renderDexLab();
    } else if (section === 'ml') {
      renderMlLab();
    } else if (section === 'volsurface') {
      renderVolSurfaceLab();
    } else {
      renderResearchPlaceholder(section);
    }
    if (back) back.focus();
  }

  button.addEventListener('click', openResearch);
  close.addEventListener('click', closeResearch);
  if (back) back.addEventListener('click', closeDetail);
  document.querySelectorAll('.research-card[data-research-section]').forEach(card => {
    card.addEventListener('click', () => openDetail(card.dataset.researchSection));
  });
  const mlGateway = document.getElementById('mlGatewayCard');
  if (mlGateway) {
    mlGateway.addEventListener('click', () => openDetail('ml'));
  }
  overlay.addEventListener('click', event => {
    if (event.target === overlay) closeResearch();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      const helpOverlay = document.getElementById('gexHelpOverlay');
      if (helpOverlay?.classList.contains('open')) {
        closeGexHelp();
        return;
      }
      const expandedChart = document.querySelector('.gex-chart.is-expanded');
      if (expandedChart) {
        expandedChart.classList.remove('is-expanded');
        if (typeof Plotly !== 'undefined') {
          setTimeout(() => {
            try { Plotly.Plots.resize(expandedChart); } catch (_) {}
          }, 80);
        }
        return;
      }
      if (overlay.classList.contains('open')) closeResearch();
    }
  });
}

setTimeout(() => {
  loadAutoFetchedChains('entry');
  loadAutoFetchedChains('close');
  initChainTabs();
  initResearchCore();
  updateScheduleBar();
}, 1500);
