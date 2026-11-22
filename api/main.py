"""Serve nightly Seattle Airbnb list-price estimates."""

from contextlib import asynccontextmanager
from math import expm1
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    room_type: str
    neighborhood_group: str
    accommodates: Annotated[int, Field(ge=1, le=16)]
    bedrooms: Annotated[int, Field(ge=0, le=7)]
    bathrooms: Annotated[float, Field(ge=0, le=8, multiple_of=0.5)]


class PredictionOutput(BaseModel):
    nightly_price_usd: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text())
    model_path = root / config["models_dir"] / f"{config['serving']['model_name']}.joblib"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Serving model missing at {model_path}. Run python -m src.train_serving."
        ) from exc

    encoders = model.named_steps["features"].named_transformers_
    app.state.model = model
    app.state.categories = {
        field: encoders[field].categories_[0].tolist()
        for field in ("room_type", "neighborhood_group")
    }
    app.state.snapshot_date = config["data"]["snapshot_date"]
    yield


app = FastAPI(title="Seattle Airbnb Price Prediction", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata(request: Request):
    return {
        "snapshot_date": request.app.state.snapshot_date,
        "room_types": request.app.state.categories["room_type"],
        "neighborhood_groups": request.app.state.categories["neighborhood_group"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request):
    fields = payload.model_dump()
    for field, choices in request.app.state.categories.items():
        if fields[field] not in choices:
            raise HTTPException(
                status_code=422,
                detail=[{"loc": ["body", field], "msg": "Select a value from /metadata."}],
            )

    log_price = request.app.state.model.predict(pd.DataFrame([fields]))[0]
    return PredictionOutput(nightly_price_usd=round(expm1(log_price), 2))
