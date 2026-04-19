/* ═══════════════════════════════════════════════════════════════════
   StockSense LSTM — Frontend Logic
   Handles stock selection, API calls, Plotly chart rendering
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  stocks:      {},
  ticker:      null,
  jobId:       null,
  pollTimer:   null,
  sector:      'all',
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dom = {
  stockGrid:      $('stock-grid'),
  filterBtns:     document.querySelectorAll('.filter-btn'),
  startDate:      $('start-date'),
  endDate:        $('end-date'),
  epochsSlider:   $('epochs-slider'),
  epochsVal:      $('epochs-val'),
  windowSlider:   $('window-slider'),
  windowVal:      $('window-val'),
  forecastToggle: $('forecast-toggle'),
  forecastRow:    $('forecast-days-row'),
  forecastSlider: $('forecast-slider'),
  forecastVal:    $('forecast-val'),
  trainBtn:       $('train-btn'),
  btnSpinner:     $('btn-spinner'),
  btnText:        $('btn-text'),
  progressWrap:   $('progress-wrap'),
  progressFill:   $('progress-fill'),
  progressPct:    $('progress-pct'),
  epochInfo:      $('epoch-info'),
  lossPreview:    $('loss-preview'),
  welcome:        $('welcome'),
  resultsArea:    $('results-area'),
  rhTicker:       $('rh-ticker'),
  rhName:         $('rh-name'),
  mMAE:           $('m-mae'),
  mRMSE:          $('m-rmse'),
  mMAPE:          $('m-mape'),
  mR2:            $('m-r2'),
  toastArea:      $('toast-area'),
  presetHint:     $('preset-hint'),
  presetTag:      $('preset-tag'),
  presetParams:   $('preset-params'),
  presetApplyBtn: $('preset-apply-btn'),
  trainingSummary:  $('training-summary'),
  mainChartTitle:  $('main-chart-title'),
};

// ── Plotly config ─────────────────────────────────────────────────────────
const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: 'hover',
  scrollZoom: true,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'lasso2d', 'select2d'],
  displaylogo: false,
};

const PHASE_LABELS = {
  initializing: 'Inicializando…',
  download:     'Descargando datos…',
  preparing:    'Preparando datos…',
  building:     'Construyendo modelo…',
  training:     'Entrenando modelo…',
  evaluating:   'Evaluando modelo…',
  forecasting:  'Generando pronóstico…',
};

const LAYOUT_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font: { family: "'Inter', sans-serif", color: '#9898b8', size: 11 },
  margin: { t: 10, r: 16, b: 40, l: 56 },
  dragmode: 'zoom',
  xaxis: {
    gridcolor:  'rgba(255,255,255,0.05)',
    linecolor:  'rgba(255,255,255,0.08)',
    zeroline:   false,
    tickfont:   { size: 10 },
  },
  yaxis: {
    gridcolor:  'rgba(255,255,255,0.05)',
    linecolor:  'rgba(255,255,255,0.08)',
    zeroline:   false,
    tickfont:   { size: 10 },
    tickprefix: '$',
  },
  legend: {
    bgcolor:      'rgba(14,14,42,0.8)',
    bordercolor:  'rgba(255,255,255,0.08)',
    borderwidth:  1,
    font:         { size: 11 },
    x: 0.01, y: 0.99,
  },
  hovermode: 'x unified',
  hoverlabel: {
    bgcolor:     'rgba(14,14,42,0.95)',
    bordercolor: 'rgba(0,212,255,0.4)',
    font:        { size: 11, color: '#e8e8f5' },
  },
};

// ── Per-ticker recommended presets ────────────────────────────────
// label: category shown to user
// color: 'amber' | 'cyan' | 'purple' | 'green'  (maps to CSS vars)
// startYears: how many years of history to load
const PRESETS = {
  // ── Alta volatilidad ─────────────────────────────────────────────
  TSLA:  { label:'Alta volatilidad',     color:'amber',  epochs:60, window:50, forecast_days:7,  startYears:5, hint:'Swings bruscos · ventana larga mejora la captura de tendencia' },
  NVDA:  { label:'Alta volatilidad',     color:'amber',  epochs:60, window:50, forecast_days:7,  startYears:4, hint:'Ciclos GPU · crecimiento explosivo con correcciones bruscas' },
  AMD:   { label:'Alta volatilidad',     color:'amber',  epochs:60, window:45, forecast_days:7,  startYears:4, hint:'Semiconductores · alta dispersión entre ciclos' },
  NFLX:  { label:'Volatilidad media-alta',color:'amber', epochs:55, window:45, forecast_days:10, startYears:5, hint:'Impulsado por resultados trimestrales · patrón irregular' },
  SPOT:  { label:'Alta volatilidad',     color:'amber',  epochs:60, window:50, forecast_days:7,  startYears:4, hint:'Sector streaming · mucha incertidumbre en corto plazo' },
  // ── Tech blue-chip ───────────────────────────────────────────────
  AAPL:  { label:'Tech blue-chip',       color:'cyan',   epochs:50, window:40, forecast_days:10, startYears:5, hint:'Tendencia clara y suave · buen candidato para LSTM' },
  MSFT:  { label:'Tech blue-chip',       color:'cyan',   epochs:50, window:40, forecast_days:14, startYears:5, hint:'Crecimiento sostenido · baja dispersión intra-trend' },
  GOOGL: { label:'Tech blue-chip',       color:'cyan',   epochs:50, window:40, forecast_days:10, startYears:5, hint:'Consolidado · patrón relativamente estable' },
  META:  { label:'Tech consolidado',     color:'cyan',   epochs:50, window:40, forecast_days:10, startYears:4, hint:'Recuperación fuerte post-2022 · tomar 4 años de datos' },
  ADBE:  { label:'Tech blue-chip',       color:'cyan',   epochs:50, window:40, forecast_days:10, startYears:4, hint:'SaaS · tendencia suave a largo plazo' },
  CRM:   { label:'Tech blue-chip',       color:'cyan',   epochs:50, window:40, forecast_days:10, startYears:4, hint:'SaaS · correlacionado con ciclo tech global' },
  INTC:  { label:'Semiconductores',      color:'cyan',   epochs:50, window:40, forecast_days:10, startYears:5, hint:'Ciclos de inventario largos · ventana mayor ayuda' },
  AMZN:  { label:'Consumo large-cap',    color:'cyan',   epochs:55, window:45, forecast_days:10, startYears:5, hint:'Alta capitalización con picos puntuales · ventana media-larga' },
  // ── Sector financiero ────────────────────────────────────────────
  JPM:   { label:'Sector financiero',    color:'purple', epochs:40, window:30, forecast_days:10, startYears:5, hint:'Correlacionado con macro y tipos de interés' },
  GS:    { label:'Sector financiero',    color:'purple', epochs:40, window:30, forecast_days:10, startYears:5, hint:'Cíclico · sensible a ciclos crediticios' },
  V:     { label:'Finanzas estable',     color:'purple', epochs:40, window:30, forecast_days:14, startYears:5, hint:'Tendencia alcista constante · buen comportamiento del modelo' },
  MA:    { label:'Finanzas estable',     color:'purple', epochs:40, window:30, forecast_days:14, startYears:5, hint:'Tendencia alcista constante · muy similar a Visa' },
  // ── Salud / defensivo ────────────────────────────────────────────
  JNJ:   { label:'Defensivo',            color:'green',  epochs:35, window:25, forecast_days:14, startYears:5, hint:'Muy baja volatilidad · ideal para ver LSTM en acción' },
  PFE:   { label:'Salud',               color:'green',  epochs:40, window:30, forecast_days:14, startYears:5, hint:'Eventos regulatorios crean picos aislados' },
  UNH:   { label:'Salud growth',        color:'green',  epochs:45, window:35, forecast_days:14, startYears:5, hint:'Crecimiento constante · buen perfil temporal para LSTM' },
  MCD:   { label:'Defensivo',            color:'green',  epochs:35, window:25, forecast_days:14, startYears:5, hint:'Muy estable · dividendos regulares · poca sorpresa' },
  NKE:   { label:'Consumo estacional',  color:'green',  epochs:40, window:30, forecast_days:14, startYears:5, hint:'Patrones estacionales suaves · ventana media suficiente' },
  // ── Energía ──────────────────────────────────────────────────────
  XOM:   { label:'Energía / petróleo',  color:'amber',  epochs:45, window:35, forecast_days:10, startYears:5, hint:'Muy correlacionado con precio del crudo WTI' },
  CVX:   { label:'Energía / petróleo',  color:'amber',  epochs:45, window:35, forecast_days:10, startYears:5, hint:'Muy correlacionado con precio del crudo Brent' },
  // ── Media / entretenimiento ──────────────────────────────────────
  DIS:   { label:'Media / consumo',     color:'amber',  epochs:45, window:35, forecast_days:10, startYears:5, hint:'Recuperación post-pandemia · dispersión media' },
};

const PRESET_COLORS = { amber:'#ffb347', cyan:'#00d4ff', purple:'#7c5cfc', green:'#00e5a0' };

// ── Toast helpers ─────────────────────────────────────────────────────────
function toast(msg, type = 'info', dur = 3500) {
  const MAX_TOASTS = 3;
  while (dom.toastArea.children.length >= MAX_TOASTS) {
    dom.toastArea.firstChild.remove();
  }
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  dom.toastArea.appendChild(el);
  setTimeout(() => el.remove(), dur);
}

// ── Sliders ───────────────────────────────────────────────────────────────
function initSliders() {
  const pairs = [
    [dom.epochsSlider,   dom.epochsVal],
    [dom.windowSlider,   dom.windowVal],
    [dom.forecastSlider, dom.forecastVal],
  ];
  pairs.forEach(([slider, label]) => {
    if (!slider) return;
    slider.addEventListener('input', () => { label.textContent = slider.value; });
  });
}

// ── Date defaults ─────────────────────────────────────────────────────────
function initDates() {
  const today  = new Date();
  const start  = new Date();
  start.setFullYear(today.getFullYear() - 4);
  const fmt = d => d.toISOString().split('T')[0];
  dom.startDate.value = fmt(start);
  dom.endDate.value   = fmt(today);
}

// ── Stock grid ────────────────────────────────────────────────────────────
async function loadStocks() {
  dom.stockGrid.innerHTML =
    '<div style="grid-column:1/-1;text-align:center;color:var(--text-2);font-size:0.75rem;padding:20px 0;">Cargando acciones…</div>';
  try {
    const res = await fetch('/api/stocks');
    state.stocks = await res.json();
    renderStockGrid();
  } catch {
    dom.stockGrid.innerHTML = '';
    toast('No se pudieron cargar las acciones', 'error');
  }
}

function renderStockGrid() {
  dom.stockGrid.innerHTML = '';
  const entries = Object.entries(state.stocks).filter(([, info]) =>
    state.sector === 'all' || info.sector === state.sector
  );

  entries.forEach(([ticker, info]) => {
    const card = document.createElement('div');
    card.className = 'stock-card' + (ticker === state.ticker ? ' selected' : '');
    card.dataset.ticker = ticker;
    card.innerHTML = `
      <div class="sc-ticker">${ticker}</div>
      <div class="sc-name">${info.name}</div>
      <div class="sc-sector">${info.sector}</div>
    `;
    card.addEventListener('click', () => selectStock(ticker));
    dom.stockGrid.appendChild(card);
  });
}

function selectStock(ticker) {
  state.ticker = ticker;
  document.querySelectorAll('.stock-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.ticker === ticker);
  });
  const info = state.stocks[ticker];
  dom.rhTicker.textContent = ticker;
  dom.rhName.textContent   = info?.name ?? '';

  // Show preset recommendation if available
  const preset = PRESETS[ticker];
  if (preset && dom.presetHint) {
    const col = PRESET_COLORS[preset.color] || '#9898b8';
    dom.presetTag.textContent = preset.label;
    dom.presetTag.style.cssText =
      `color:${col}; background:${col}22; padding:2px 7px; border-radius:99px;` +
      `font-size:0.6rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;`;
    dom.presetParams.textContent =
      `${preset.epochs} épocas · ventana ${preset.window} días · ${preset.startYears} años de datos · pronóstico ${preset.forecast_days} días`;
    dom.presetHint.querySelector('.preset-hint-note').textContent = preset.hint;
    dom.presetHint.style.borderColor  = col + '33';
    dom.presetApplyBtn.style.color    = col;
    dom.presetApplyBtn.style.borderColor = col + '44';
    dom.presetHint.style.display      = 'block';
    dom.presetApplyBtn.onclick        = () => applyPreset(preset);
  } else if (dom.presetHint) {
    dom.presetHint.style.display = 'none';
  }

  toast(`${ticker} seleccionado`, 'info', 1800);
}

function applyPreset(preset) {
  dom.epochsSlider.value      = preset.epochs;
  dom.epochsVal.textContent   = preset.epochs;
  dom.windowSlider.value      = preset.window;
  dom.windowVal.textContent   = preset.window;
  dom.forecastSlider.value    = preset.forecast_days;
  dom.forecastVal.textContent = preset.forecast_days;

  const today = new Date();
  const start = new Date();
  start.setFullYear(today.getFullYear() - preset.startYears);
  const fmt = d => d.toISOString().split('T')[0];
  dom.startDate.value = fmt(start);

  toast('Configuración recomendada aplicada ✓', 'success', 2200);
}

// ── Sector filters ────────────────────────────────────────────────────────
function initFilters() {
  dom.filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      dom.filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.sector = btn.dataset.sector;
      renderStockGrid();
    });
  });
}

// ── Training ──────────────────────────────────────────────────────────────
async function startTraining() {
  if (!state.ticker) {
    toast('Selecciona una acción primero', 'error');
    return;
  }

  const includeForecast = dom.forecastToggle.checked;
  const body = {
    ticker:           state.ticker,
    start_date:       dom.startDate.value,
    end_date:         dom.endDate.value,
    epochs:           parseInt(dom.epochsSlider.value),
    window:           parseInt(dom.windowSlider.value),
    forecast_days:    parseInt(dom.forecastSlider.value),
    include_forecast: includeForecast,
  };

  // UI: loading state
  dom.trainBtn.disabled     = true;
  dom.btnSpinner.style.display = 'block';
  dom.btnText.textContent   = 'Entrenando…';
  dom.progressWrap.style.display = 'block';
  dom.progressFill.style.width   = '0%';
  dom.progressPct.textContent    = '0%';
  dom.epochInfo.textContent      = `Época 0 / ${body.epochs}`;
  dom.lossPreview.textContent    = '';

  try {
    const res  = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Error iniciando entrenamiento');
    state.jobId = data.job_id;
    toast(`Iniciando entrenamiento para ${state.ticker}…`, 'info');
    pollStatus();
  } catch (err) {
    resetTrainBtn();
    toast(err.message, 'error');
  }
}

function pollStatus() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const res  = await fetch(`/api/status/${state.jobId}`);
      const job  = await res.json();

      // Update progress bar
      const pct  = job.progress ?? 0;
      dom.progressFill.style.width = pct + '%';
      dom.progressPct.textContent  = pct + '%';

      const phase     = job.phase || 'initializing';
      const phaseText = PHASE_LABELS[phase] || phase;
      if (phase === 'training') {
        dom.epochInfo.textContent =
          `${phaseText} Época ${job.current_epoch} / ${job.total_epochs}`;
      } else if (phase === 'evaluating' || phase === 'forecasting') {
        const stoppedEarly = job.current_epoch > 0 && job.current_epoch < job.total_epochs;
        dom.epochInfo.textContent = stoppedEarly
          ? `${phaseText} (early stop época ${job.current_epoch})`
          : phaseText;
      } else {
        dom.epochInfo.textContent = phaseText;
      }

      // Show latest losses
      if (job.loss && job.loss.length > 0) {
        const last    = job.loss[job.loss.length - 1];
        const lastVal = job.val_loss?.[job.val_loss.length - 1];
        dom.lossPreview.innerHTML =
          `loss: <b>${last.toFixed(6)}</b>` +
          (lastVal != null ? `  •  val_loss: <b>${lastVal.toFixed(6)}</b>` : '');
      }

      if (job.status === 'completed') {
        clearInterval(state.pollTimer);
        fetchAndRender();
      } else if (job.status === 'error') {
        clearInterval(state.pollTimer);
        resetTrainBtn();
        toast(`Error: ${job.error}`, 'error', 7000);
      }
    } catch {
      // network hiccup — keep polling
    }
  }, 800);
}

async function fetchAndRender() {
  try {
    const res    = await fetch(`/api/result/${state.jobId}`);
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail);
    renderResults(result);
    toast(`¡Entrenamiento completado! ${state.ticker} listo`, 'success', 5000);
  } catch (err) {
    toast(err.message, 'error', 5000);
  } finally {
    resetTrainBtn();
    setTimeout(() => { dom.progressWrap.style.display = 'none'; }, 600);
  }
}

function resetTrainBtn() {
  dom.trainBtn.disabled        = false;
  dom.btnSpinner.style.display = 'none';
  dom.btnText.textContent      = 'Entrenar Modelo';
}

// ── Results rendering ─────────────────────────────────────────────────────
function renderResults(data) {
  dom.welcome.style.display     = 'none';
  dom.resultsArea.style.display = 'flex';

  renderTrainingSummary(data);
  renderMetrics(data.metrics);
  dom.mainChartTitle.textContent = data.forecast
    ? 'Serie histórica · Predicciones test · Pronóstico futuro'
    : 'Serie histórica · Predicciones test';
  renderMainChart(data);
  renderLossChart(data.training_history);
  renderErrorChart(data.test.errors);
}

function renderTrainingSummary(data) {
  const h   = data.training_history;
  const m   = data.metrics;
  const early = h.epochs_run < parseInt(dom.epochsSlider.value);

  const items = [
    `<span class="ts-item">Épocas: <strong>${h.epochs_run}</strong> / ${dom.epochsSlider.value}</span>`,
    `<span class="ts-item">Train: <strong>${m.train_size.toLocaleString()}</strong> muestras</span>`,
    `<span class="ts-item">Test: <strong>${m.test_size.toLocaleString()}</strong> muestras</span>`,
  ];
  if (early) {
    items.push('<span class="ts-item ts-early-stop">Early Stopping activo</span>');
  }
  dom.trainingSummary.innerHTML = items.join('');
}

function renderMetrics(m) {
  animateCount(dom.mMAE,  m.mae,  v => `$${v.toFixed(2)}`);
  animateCount(dom.mRMSE, m.rmse, v => `$${v.toFixed(2)}`);
  animateCount(dom.mMAPE, m.mape, v => `${v.toFixed(2)}%`);
  animateCount(dom.mR2,   m.r2,   v => v.toFixed(4));
}

function animateCount(el, target, fmt, dur = 800) {
  const start = performance.now();
  const from  = 0;
  function step(now) {
    const t   = Math.min((now - start) / dur, 1);
    const val = from + (target - from) * easeOut(t);
    el.textContent = fmt(val);
    if (t < 1) requestAnimationFrame(step);
    else        el.textContent = fmt(target);
  }
  requestAnimationFrame(step);
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

function renderMainChart(data) {
  const hist     = data.historical;
  const test     = data.test;
  const forecast = data.forecast;

  const traces = [
    // Historical prices
    {
      x: hist.dates,
      y: hist.prices,
      type: 'scatter',
      mode: 'lines',
      name: 'Precio histórico',
      line: { color: 'rgba(100,181,246,0.75)', width: 1.5 },
      hovertemplate: '%{x}<br><b>$%{y:.2f}</b><extra></extra>',
    },
    // Test predictions
    {
      x: test.dates,
      y: test.predictions,
      type: 'scatter',
      mode: 'lines',
      name: 'Predicción (test)',
      line: { color: '#ffb347', width: 2, dash: 'dot' },
      hovertemplate: '%{x}<br><b>$%{y:.2f}</b><extra></extra>',
    },
  ];

  // Forecast traces (only if forecast was requested)
  if (forecast) {
    traces.push(
      // 95% band
      {
        x: [...forecast.dates, ...forecast.dates.slice().reverse()],
        y: [...forecast.upper_95, ...forecast.lower_95.slice().reverse()],
        type: 'scatter',
        mode: 'none',
        fill: 'toself',
        fillcolor: 'rgba(0,229,160,0.08)',
        name: 'IC 95%',
        hoverinfo: 'skip',
      },
      // 68% band
      {
        x: [...forecast.dates, ...forecast.dates.slice().reverse()],
        y: [...forecast.upper_68, ...forecast.lower_68.slice().reverse()],
        type: 'scatter',
        mode: 'none',
        fill: 'toself',
        fillcolor: 'rgba(0,229,160,0.14)',
        name: 'IC 68%',
        hoverinfo: 'skip',
      },
      // Mean forecast line
      {
        x: forecast.dates,
        y: forecast.mean,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Pronóstico',
        line: { color: '#00e5a0', width: 2.5 },
        marker: { size: 5, color: '#00e5a0', symbol: 'circle' },
        hovertemplate: '%{x}<br><b>$%{y:.2f}</b><extra></extra>',
      },
    );
  }

  // Vertical separator at last historical date
  const allPrices = forecast
    ? [...hist.prices, ...forecast.mean]
    : hist.prices;
  traces.push({
    x: [hist.dates.at(-1), hist.dates.at(-1)],
    y: [
      hist.prices.reduce((a, b) => Math.min(a, b), Infinity) * 0.95,
      allPrices.reduce((a, b) => Math.max(a, b), -Infinity) * 1.05,
    ],
    type: 'scatter',
    mode: 'lines',
    name: 'Hoy',
    line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dot' },
    hoverinfo: 'skip',
    showlegend: false,
  });

  const layout = {
    ...LAYOUT_BASE,
    height: 380,
    margin: { t: 10, r: 20, b: 50, l: 64 },
  };

  Plotly.newPlot('main-chart', traces, layout, PLOTLY_CONFIG);
}

function renderLossChart(history) {
  const epochs = Array.from({ length: history.loss.length }, (_, i) => i + 1);

  const traces = [
    {
      x: epochs,
      y: history.loss,
      type: 'scatter',
      mode: 'lines',
      name: 'Train loss',
      line: { color: '#7c5cfc', width: 2 },
    },
    {
      x: Array.from({ length: history.val_loss.length }, (_, i) => i + 1),
      y: history.val_loss,
      type: 'scatter',
      mode: 'lines',
      name: 'Val loss',
      line: { color: '#00d4ff', width: 2, dash: 'dot' },
    },
  ];

  const layout = {
    ...LAYOUT_BASE,
    height: 220,
    margin: { t: 8, r: 16, b: 36, l: 56 },
    yaxis: { ...LAYOUT_BASE.yaxis, tickprefix: '', type: 'linear' },
    xaxis: {
      ...LAYOUT_BASE.xaxis,
      type: 'linear',
      dtick: Math.max(1, Math.floor(history.loss.length / 10)),
      title: { text: 'Época', font: { size: 10 } },
    },
  };

  Plotly.newPlot('loss-chart', traces, layout, PLOTLY_CONFIG);
}

function renderErrorChart(errors) {
  const trace = {
    x:          errors,
    type:       'histogram',
    nbinsx:     30,
    marker: {
      color:   'rgba(124,92,252,0.6)',
      line:    { color: 'rgba(124,92,252,0.9)', width: 1 },
    },
    name: 'Error (pred − real)',
    hovertemplate: 'Error: %{x:.2f}<br>Frecuencia: %{y}<extra></extra>',
  };

  const layout = {
    ...LAYOUT_BASE,
    height: 220,
    margin: { t: 8, r: 16, b: 36, l: 44 },
    yaxis: { ...LAYOUT_BASE.yaxis, tickprefix: '', type: 'linear', title: { text: 'Frecuencia', font: { size: 10 } } },
    xaxis: { ...LAYOUT_BASE.xaxis, type: 'linear', title: { text: 'Error ($)', font: { size: 10 } } },
    bargap: 0.05,
  };

  Plotly.newPlot('error-chart', [trace], layout, PLOTLY_CONFIG);
}

// ── Init ──────────────────────────────────────────────────────────────────
async function init() {
  initSliders();
  initDates();
  await loadStocks();
  initFilters();

  dom.forecastToggle.addEventListener('change', () => {
    dom.forecastRow.style.display = dom.forecastToggle.checked ? '' : 'none';
  });

  dom.trainBtn.addEventListener('click', startTraining);
}

document.addEventListener('DOMContentLoaded', init);
