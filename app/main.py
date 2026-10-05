"""RasoiIQ web app: orders CSV -> TabPFN forecast -> prep plan, chart and accuracy check."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import asdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import CITY, FRIEND_NAME, ROOT, SAMPLE_COMPLEX, SAMPLE_ORDERS, THEME
from .data import MAX_UPLOAD_BYTES, DataError, load_orders
from .festivals import festival_name
from .forecast import run_forecast
from .planner import make_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("rasoiiq")
STATIC = ROOT / "app" / "static"

_cache: OrderedDict[str, dict] = OrderedDict()
_lock = threading.Lock()  # one TabPFN run at a time keeps an 8 GB laptop comfortable


def _compute(raw: bytes, is_sample: bool, source_name: str, tomorrow_special: bool) -> dict:
    key = hashlib.sha256(raw + f"|{tomorrow_special}".encode()).hexdigest()
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        data = load_orders(raw)
        result = run_forecast(data, tomorrow_special=tomorrow_special)
        fest = festival_name(result.target_date) or ("a festival or party day (you marked it)" if tomorrow_special else None)
        plan = make_plan(result, festival=fest)
        bt = {k: v for k, v in result.backtest.items() if k != "rows"} if result.backtest else None
        body = {
            "is_sample": is_sample,
            "source_name": source_name,
            "data_range": {"start": data.start.isoformat(), "end": data.end.isoformat(),
                           "days": (data.end - data.start).days + 1, "items": data.items,
                           "rows": len(data.daily)},
            "target_date": result.target_date.isoformat(),
            "target_day": result.target_date.strftime("%A"),
            "festival": fest,
            "tomorrow_special": tomorrow_special,
            "model_used": result.model_used,
            "model_note": result.model_note,
            "items": [asdict(i) for i in result.items],
            "backtest": bt,
            "chart": result.chart,
            "plan": {"text": plan.text, "source": plan.source, "note": plan.note, "shopping": plan.shopping},
            "warnings": data.warnings,
            "timings": result.timings,
        }
        if result.model_note is None:  # don't cache a fallback result
            _cache[key] = body
            while len(_cache) > 8:
                _cache.popitem(last=False)
        return body


app = FastAPI(title="RasoiIQ", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/config")
def config() -> dict:
    return {"friend": FRIEND_NAME, "city": CITY, "theme": THEME,
            "gemma_enabled": bool(os.getenv("GEMMA_API_KEY"))}


@app.get("/api/sample.csv")
def sample_csv() -> FileResponse:
    return FileResponse(SAMPLE_ORDERS, media_type="text/csv", filename="sample_orders.csv")


@app.get("/api/sample-complex.csv")
def sample_complex_csv() -> FileResponse:
    return FileResponse(SAMPLE_COMPLEX, media_type="text/csv", filename="sample_orders_complex.csv")


@app.post("/api/forecast")
async def forecast(file: UploadFile | None = File(None), tomorrow_special: bool = Form(False)) -> JSONResponse:
    # Nothing is forecast until a CSV is uploaded.
    if file is None or not file.filename:
        raise HTTPException(400, "Please upload your orders CSV first.")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    name = file.filename
    is_sample = name.startswith("sample_orders")
    try:
        body = await run_in_threadpool(_compute, raw, is_sample, name, tomorrow_special)
    except DataError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(body)
