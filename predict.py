from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"
META_PATH = BASE_DIR / "model_meta.pkl"

DEFAULT_SUSPICIOUS_THRESHOLD = 0.50
DEFAULT_FAKE_THRESHOLD = 0.80
MAX_LENGTH = 256

RULE_REPORT_COUNT_WEIGHT = 0.3
RULE_ACCOUNT_AGE_WEIGHT = 0.3
RULE_REVIEW_GAP_WEIGHT = 0.3
RULE_FIVE_STAR_WEIGHT = 0.1

RULE_REPORT_COUNT = "report_count > 10"
RULE_ACCOUNT_AGE = "account_age < 3"
RULE_REVIEW_GAP = "time_between_reviews < 1"
RULE_FIVE_STAR_MODEL = "rating == 5 and fake_prob > 0.6"

FAKE_SCORE_THRESHOLD = 0.8
SUSPICIOUS_SCORE_THRESHOLD = 0.6


class Predictor:
    """Loads model artifacts once and serves text predictions."""

    def __init__(self) -> None:
        self.meta = self._load_meta(META_PATH)
        self.suspicious_threshold = float(
            self.meta.get("suspicious_threshold", DEFAULT_SUSPICIOUS_THRESHOLD)
        )
        self.fake_threshold = float(self.meta.get("fake_threshold", DEFAULT_FAKE_THRESHOLD))

        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        self.model.eval()

    @staticmethod
    def _load_meta(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            meta = pickle.load(handle)
            return meta if isinstance(meta, dict) else {}

    def predict(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        fake_prob = self._extract_fake_probability(logits)
        label = self._label_from_probability(fake_prob)

        return {
            "label": label,
            "fake_prob": fake_prob,
        }

    @staticmethod
    def _extract_fake_probability(logits: torch.Tensor) -> float:
        if logits.shape[-1] == 1:
            probability = torch.sigmoid(logits.squeeze()).item()
            return float(probability)

        probabilities = torch.softmax(logits, dim=-1)
        return float(probabilities[0, 1].item())

    def _label_from_probability(self, fake_prob: float) -> str:
        if fake_prob >= self.fake_threshold:
            return "fake"
        if fake_prob >= self.suspicious_threshold:
            return "suspicious"
        return "real"


_predictor = Predictor()


def predict(
    text: str,
    review_count: Optional[int] = None,
    account_age: Optional[float] = None,
    rating: Optional[float] = None,
    time_between_reviews: Optional[float] = None,
    report_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Public function used by the API layer."""
    del review_count

    ml_result = _predictor.predict(text)
    fake_prob = float(ml_result.get("fake_prob", 0.0))
    model_label = str(ml_result.get("label", "real"))

    rule_score, applied_rules = _compute_rule_score(
        fake_prob=fake_prob,
        rating=rating,
        account_age=account_age,
        time_between_reviews=time_between_reviews,
        report_count=report_count,
    )
    final_score = _clamp_score(fake_prob + rule_score)
    behavior_risk = _has_behavior_risk(account_age=account_age, time_between_reviews=time_between_reviews)
    label = _label_from_score(final_score=final_score, applied_rules=applied_rules)
    explanation = _build_explanation(
        label=label,
        behavior_risk=behavior_risk,
        model_label=model_label,
        fake_prob=fake_prob,
    )
    rounded_final_score = round(final_score, 2)
    decision_source = "hybrid" if applied_rules else "model"

    risk_level = _label_to_risk_level(label)
    timestamp = datetime.utcnow().isoformat() + "Z"

    return {
        "risk_level": risk_level,
        "confidence": rounded_final_score,
        "fake_probability": round(fake_prob, 4),
        "decision_source": decision_source,
        "signals": {
            "applied_rules": applied_rules,
        },
        "explanation": explanation,
        "timestamp": timestamp,
    }


def _compute_rule_score(
    fake_prob: float,
    rating: Optional[float],
    account_age: Optional[float],
    time_between_reviews: Optional[float],
    report_count: Optional[int],
) -> tuple[float, list[str]]:
    score = 0.0
    applied_rules: list[str] = []

    if report_count is not None and report_count > 10:
        score += RULE_REPORT_COUNT_WEIGHT
        applied_rules.append(RULE_REPORT_COUNT)

    if account_age is not None and account_age < 3:
        score += RULE_ACCOUNT_AGE_WEIGHT
        applied_rules.append(RULE_ACCOUNT_AGE)

    if time_between_reviews is not None and time_between_reviews < 1:
        score += RULE_REVIEW_GAP_WEIGHT
        applied_rules.append(RULE_REVIEW_GAP)

    if rating is not None and float(rating) == 5.0 and fake_prob > 0.6:
        score += RULE_FIVE_STAR_WEIGHT
        applied_rules.append(RULE_FIVE_STAR_MODEL)

    return score, applied_rules


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _label_to_risk_level(label: str) -> str:
    mapping = {
        "fake": "high",
        "suspicious": "medium",
        "real": "low",
    }
    return mapping.get(str(label).lower(), "low")


def _label_from_score(final_score: float, applied_rules: list[str]) -> str:
    if final_score >= FAKE_SCORE_THRESHOLD:
        return "fake"
    if final_score >= SUSPICIOUS_SCORE_THRESHOLD:
        return "suspicious"
    if applied_rules:
        return "suspicious"
    return "real"


def _has_behavior_risk(account_age: Optional[float], time_between_reviews: Optional[float]) -> bool:
    return (account_age is not None and account_age < 3) or (
        time_between_reviews is not None and time_between_reviews < 1
    )


def _build_explanation(
    label: str,
    behavior_risk: bool,
    model_label: str,
    fake_prob: float,
) -> str:
    if behavior_risk:
        return "Suspicious behavior detected: new account with rapid review activity"
    if label == "fake":
        return f"Model indicated high likelihood of fake review (probability={fake_prob:.2f})"
    if label == "suspicious":
        return f"Model predicted suspicious behavior (probability={fake_prob:.2f})"
    return f"Review appears genuine based on model analysis (probability={fake_prob:.2f})"
