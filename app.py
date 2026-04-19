"""
LSTM Stock Predictor — FastAPI Backend
Start: python app.py
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import date

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator, model_validator

from lstm_core import LSTMPredictor

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="LSTM Stock Predictor", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Stock catalogue ──────────────────────────────────────────────────────────

STOCKS: dict[str, dict] = {
    # Technology
    "AAPL":  {"name": "Apple Inc.",              "sector": "Tecnología"},
    "MSFT":  {"name": "Microsoft Corp.",         "sector": "Tecnología"},
    "GOOGL": {"name": "Alphabet Inc.",           "sector": "Tecnología"},
    "NVDA":  {"name": "NVIDIA Corp.",            "sector": "Tecnología"},
    "META":  {"name": "Meta Platforms",          "sector": "Tecnología"},
    "ADBE":  {"name": "Adobe Inc.",              "sector": "Tecnología"},
    "CRM":   {"name": "Salesforce Inc.",         "sector": "Tecnología"},
    "INTC":  {"name": "Intel Corp.",             "sector": "Tecnología"},
    "AMD":   {"name": "Advanced Micro Devices",  "sector": "Tecnología"},
    # Consumer / Disruptors
    "TSLA":  {"name": "Tesla Inc.",              "sector": "Consumo"},
    "AMZN":  {"name": "Amazon.com Inc.",         "sector": "Consumo"},
    "NFLX":  {"name": "Netflix Inc.",            "sector": "Consumo"},
    "NKE":   {"name": "Nike Inc.",               "sector": "Consumo"},
    "MCD":   {"name": "McDonald's Corp.",        "sector": "Consumo"},
    # Finance
    "JPM":   {"name": "JPMorgan Chase",          "sector": "Finanzas"},
    "GS":    {"name": "Goldman Sachs",           "sector": "Finanzas"},
    "V":     {"name": "Visa Inc.",               "sector": "Finanzas"},
    "MA":    {"name": "Mastercard Inc.",         "sector": "Finanzas"},
    # Healthcare
    "JNJ":   {"name": "Johnson & Johnson",       "sector": "Salud"},
    "PFE":   {"name": "Pfizer Inc.",             "sector": "Salud"},
    "UNH":   {"name": "UnitedHealth Group",      "sector": "Salud"},
    # Energy
    "XOM":   {"name": "Exxon Mobil Corp.",       "sector": "Energía"},
    "CVX":   {"name": "Chevron Corp.",           "sector": "Energía"},
    # Communication / Media
    "DIS":   {"name": "Walt Disney Co.",         "sector": "Entretenimiento"},
    "SPOT":  {"name": "Spotify Technology",      "sector": "Entretenimiento"},
}

# ── Job store ────────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_JOB_TTL = 1800  # 30 min


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL
    to_remove = [
        jid for jid, j in _jobs.items()
        if j.get("created_at", 0) < cutoff and j["status"] != "running"
    ]
    for jid in to_remove:
        del _jobs[jid]

# ── Request / response models ────────────────────────────────────────────────

class TrainRequest(BaseModel):
    ticker:           str
    start_date:       str = "2020-01-01"
    end_date:         str | None = None
    window:           int = 30
    epochs:           int = 25
    forecast_days:    int = 10
    include_forecast: bool = True

    @field_validator("ticker")
    @classmethod
    def ticker_valid(cls, v: str) -> str:
        v = v.upper()
        if v not in STOCKS:
            raise ValueError(f"Ticker desconocido: {v}")
        return v

    @field_validator("start_date")
    @classmethod
    def start_date_valid(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except (ValueError, TypeError):
            raise ValueError(f"Formato de fecha inválido: {v}. Use YYYY-MM-DD")
        return v

    @field_validator("end_date", mode="before")
    @classmethod
    def end_date_valid(cls, v: str | None) -> str:
        if not v:
            return str(date.today())
        try:
            date.fromisoformat(str(v))
        except (ValueError, TypeError):
            raise ValueError(f"Formato de fecha inválido: {v}. Use YYYY-MM-DD")
        return str(v)

    @model_validator(mode="after")
    def dates_order(self) -> "TrainRequest":
        if self.start_date >= self.end_date:
            raise ValueError("start_date debe ser anterior a end_date")
        return self

    @field_validator("window")
    @classmethod
    def window_range(cls, v: int) -> int:
        return max(10, min(60, v))

    @field_validator("epochs")
    @classmethod
    def epochs_range(cls, v: int) -> int:
        return max(10, min(100, v))

    @field_validator("forecast_days")
    @classmethod
    def forecast_range(cls, v: int) -> int:
        return max(5, min(30, v))


# ── Background training ───────────────────────────────────────────────────────

def _train_worker(job_id: str, req: TrainRequest) -> None:
    try:
        predictor = LSTMPredictor(
            ticker=req.ticker,
            window=req.window,
            epochs=req.epochs,
            units=100,
        )

        def on_epoch(epoch: int, logs: dict) -> None:
            with _lock:
                if job_id not in _jobs:
                    return
                job = _jobs[job_id]
                job["current_epoch"] = epoch
                job["progress"] = 10 + int(epoch / req.epochs * 65)
                job["loss"].append(round(float(logs.get("loss", 0)), 6))
                val = logs.get("val_loss")
                if val is not None:
                    job["val_loss"].append(round(float(val), 6))

        def on_phase(phase: str, progress: int) -> None:
            with _lock:
                if job_id not in _jobs:
                    return
                _jobs[job_id]["phase"] = phase
                _jobs[job_id]["progress"] = progress

        result = predictor.run(
            start_date=req.start_date,
            end_date=req.end_date,
            progress_callback=on_epoch,
            phase_callback=on_phase,
            forecast_days=req.forecast_days,
            include_forecast=req.include_forecast,
        )

        with _lock:
            _jobs[job_id].update(
                status="completed",
                progress=100,
                phase="completed",
                result=result,
            )

    except Exception as exc:  # noqa: BLE001
        with _lock:
            _jobs[job_id].update(status="error", error=str(exc))


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stocks")
async def list_stocks() -> dict:
    return STOCKS


@app.post("/api/train")
async def start_training(req: TrainRequest) -> dict:
    job_id = str(uuid.uuid4())
    with _lock:
        _cleanup_old_jobs()
        running = sum(1 for j in _jobs.values() if j["status"] == "running")
        if running > 0:
            raise HTTPException(
                status_code=429,
                detail="Ya hay un entrenamiento en curso. Espera a que termine.",
            )
        _jobs[job_id] = {
            "status":        "running",
            "ticker":        req.ticker,
            "progress":      0,
            "current_epoch": 0,
            "total_epochs":  req.epochs,
            "phase":         "initializing",
            "loss":          [],
            "val_loss":      [],
            "result":        None,
            "error":         None,
            "created_at":    time.time(),
        }
    thread = threading.Thread(
        target=_train_worker, args=(job_id, req), daemon=True
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str) -> dict:
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job no encontrado")
        job = dict(_jobs[job_id])
    job.pop("result", None)
    job.pop("created_at", None)
    return job


@app.get("/api/result/{job_id}")
async def job_result(job_id: str) -> dict:
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job no encontrado")
        job = dict(_jobs[job_id])
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Error desconocido"))
    if job["status"] != "completed":
        raise HTTPException(status_code=202, detail="Entrenamiento en curso")
    return job["result"]


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
