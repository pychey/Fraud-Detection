from __future__ import annotations

import os
from contextlib import suppress
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from predict import predict

app = FastAPI(
    title="Fake Review Detection API",
    version="1.0.0",
    description="Production-ready API for ML + rule-based fake review detection.",
)


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/fake_reviews"
)
DATABASE_REQUIRED = os.getenv("DATABASE_REQUIRED", "false").lower() == "true"
DB_ENABLED = False


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text)
    review_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    account_age: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_between_reviews: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    report_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    label: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Review text")
    review_count: Optional[int] = Field(default=None, ge=0)
    account_age: Optional[float] = Field(default=None, ge=0, description="Account age in days")
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    time_between_reviews: Optional[float] = Field(default=None, ge=0)
    report_count: Optional[int] = Field(default=None, ge=0)


class SignalsInfo(BaseModel):
    applied_rules: list[str]


class PredictResponse(BaseModel):
    risk_level: str
    confidence: float
    fake_probability: float
    decision_source: str
    signals: SignalsInfo
    explanation: str
    timestamp: str


@app.on_event("startup")
def on_startup() -> None:
    global DB_ENABLED
    try:
        Base.metadata.create_all(bind=engine)
        DB_ENABLED = True
    except Exception as exc:
        DB_ENABLED = False
        if DATABASE_REQUIRED:
            raise RuntimeError(f"Database startup failed and is required: {exc}") from exc
        print(f"[WARN] Database unavailable, continuing without persistence: {exc}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request payload",
            "errors": exc.errors(),
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_review(payload: PredictRequest) -> PredictResponse:
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    cleaned_text = payload.text.strip()

    try:
        result = predict(
            text=cleaned_text,
            review_count=payload.review_count,
            account_age=payload.account_age,
            rating=payload.rating,
            time_between_reviews=payload.time_between_reviews,
            report_count=payload.report_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    response = PredictResponse(**result)

    if DB_ENABLED:
        db = SessionLocal()
        try:
            db_record = PredictionLog(
                text=payload.text,
                review_count=payload.review_count,
                account_age=payload.account_age,
                rating=payload.rating,
                time_between_reviews=payload.time_between_reviews,
                report_count=payload.report_count,
                label=response.risk_level,
                confidence=response.confidence,
                reason=response.explanation,
            )
            db.add(db_record)
            db.commit()
        except Exception as exc:
            db.rollback()
            if DATABASE_REQUIRED:
                raise HTTPException(status_code=500, detail=f"Database write failed: {exc}") from exc
            print(f"[WARN] Database write skipped: {exc}")
        finally:
            with suppress(Exception):
                db.close()

    return response
