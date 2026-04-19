# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
pip install -r requirements.txt
python app.py
# Server starts at http://localhost:8000
```

## Architecture

Two-layer application:

**`lstm_core.py` — ML pipeline (`LSTMPredictor` class)**
- `run()` is the end-to-end entry point: fetch → prepare → build → train → evaluate → forecast
- Data comes from Yahoo Finance (`yfinance`), scaled with `MinMaxScaler`
- Model: 2-layer stacked LSTM (100 units → 50 units) → Dense(25, relu) → Dropout(0.15) → Dense(1)
- Optimizer: Adam(lr=0.0005, clipnorm=1.0) for stable training curves
- Uncertainty estimation uses Monte Carlo Dropout: keeps `training=True` at inference time across 40 simulations to produce 68% and 95% confidence bands
- EarlyStopping (patience=10) and ReduceLROnPlateau (patience=5) prevent overfitting

**`app.py` — FastAPI web backend**
- Training is async: `POST /api/train` spawns a daemon thread and returns a `job_id`
- Poll `GET /api/status/{job_id}` for progress (epoch count, loss history); `GET /api/result/{job_id}` for the full result once complete
- In-memory job store (`_jobs` dict with `threading.Lock`) — auto-cleanup after 30 min, max 1 concurrent training
- `TrainRequest` validators clamp inputs: window [10–60], epochs [10–100], forecast_days [5–30]; date format validated
- `phase_callback` reports pipeline phases (download → preparing → building → training → evaluating → forecasting) for real-time progress
- Supported tickers are hardcoded in `STOCKS` dict in `app.py`; unknown tickers are rejected at validation

**`static/js/app.js` — frontend logic**
- `PLOTLY_CONFIG` uses `displayModeBar: 'hover'` and `scrollZoom: true`; double-click resets zoom
- `PRESETS` object holds per-ticker recommended parameters (epochs, window, forecast_days, startYears) grouped by volatility category (alta volatilidad / tech blue-chip / financiero / defensivo / energía)
- `PRESET_COLORS` maps category names to hex colors used for the hint UI
- When a stock is selected, `selectStock()` shows a preset hint card; "Aplicar" calls `applyPreset()` which updates all sliders and the date range
- Training flow: `startTraining()` → `POST /api/train` → `pollStatus()` every 800 ms → `fetchAndRender()` on completion

**`static/css/style.css`** — dark glassmorphism theme; CSS vars defined in `:root`

**`templates/index.html`** — Jinja2 shell; all dynamic content injected by `app.js`

**`prediccion_acciones_lstm.py`** — original standalone script; `lstm_core.py` is the web-adapted version of it.

## Key constraints

- Adding a new ticker requires updating **both** the `STOCKS` dict in `app.py` (for API validation) and the `PRESETS` object in `app.js` (for recommendations)
- yfinance ≥0.2 may return MultiIndex columns; `lstm_core.py:68-69` flattens them
- TensorFlow logging is suppressed via env var `TF_CPP_MIN_LOG_LEVEL=3` and `tf.get_logger().setLevel("ERROR")`
- Preset parameters are UI-only; the backend clamps all values independently (`TrainRequest` validators)
