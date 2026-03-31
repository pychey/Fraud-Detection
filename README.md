# Fake Review Detection

A hybrid ML + rule-based fake review detection system built with DistilBERT, FastAPI, and Streamlit.

---

## How It Works

Each review is processed in two stages:

1. **ML Model** — A fine-tuned DistilBERT model scores the review text and returns a raw fake probability (0–1)
2. **Rule Engine** — Behavioral signals (account age, report count, review frequency, rating) are evaluated on top of the ML score to produce a final risk level

The final output combines both signals into a `risk_level` of `high`, `medium`, or `low`.

---

## Project Structure

```
fake-review-detection/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints, DB logging
│   └── predict.py       # ML model + rule engine
├── model/
│   ├── model_meta.pkl   # Tuned thresholds
│   └── distilbert/      # Model weights and tokenizer files
├── dashboard/
│   └── app.py           # Streamlit dashboard
├── data/
│   └── fake reviews dataset.csv
├── docs/
│   ├── confusion_test.png
│   └── model_comparison.png
├── deploy/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.12
- pip
- (Optional) PostgreSQL — for prediction logging

### Mac — Install Python 3.12 via Homebrew

```bash
brew install python@3.12
python3.12 --version
```

### Windows — Install Python 3.12

Download and install from [python.org](https://www.python.org/downloads/). During installation, check **Add Python to PATH**.

```powershell
python --version
```

---

## Running Locally

### 1. Clone the repo and navigate to project root

```bash
# Mac
cd /path/to/fake-review-detection

# Windows
cd C:\path\to\fake-review-detection
```

### 2. Create and activate a virtual environment

**Mac:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If you get a PowerShell execution policy error on Windows, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:
- Base URL: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

### 5. Run the Dashboard

Open a **second terminal**, activate the venv again, then:

**Mac:**
```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

Dashboard will be available at `http://localhost:8501`

---

## Database Setup (Optional)

By default the API runs without a database — predictions are not persisted. To enable PostgreSQL logging:

### 1. Create the database

**Mac:**
```bash
psql postgres
```

**Windows** (via psql or pgAdmin):
```bash
psql -U postgres
```

Then inside psql:
```sql
CREATE DATABASE fake_reviews;
\q
```

> The `prediction_logs` table is created automatically on API startup — no manual schema setup needed.

### 2. Create a `.env` file in the project root

**Mac (no password):**
```
DATABASE_URL=postgresql+psycopg2://your_username@localhost:5432/fake_reviews
DATABASE_REQUIRED=true
```

**Windows (with password):**
```
DATABASE_URL=postgresql+psycopg2://postgres:your_password@localhost:5432/fake_reviews
DATABASE_REQUIRED=true
```

Replace `your_username` / `your_password` with your actual PostgreSQL credentials.

### 3. Install dotenv

```bash
pip install python-dotenv
```

Restart the API — you should see no `[WARN] Database unavailable` message on startup.

---

## API Contract

### GET /health

```json
{ "status": "ok" }
```

---

### POST /predict

**Request:**

```json
{
  "text": "This product is amazing!",
  "review_count": 6,
  "account_age": 1,
  "rating": 5,
  "time_between_reviews": 0.2,
  "report_count": 0
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | The review text to analyze |
| `review_count` | int | ❌ | Total number of reviews posted by this account |
| `account_age` | float | ❌ | Account age in years |
| `rating` | float (0–5) | ❌ | Star rating given with the review |
| `time_between_reviews` | float | ❌ | Average days between this account's reviews |
| `report_count` | int | ❌ | Number of times this account has been reported |

**Response:**

```json
{
  "risk_level": "medium",
  "confidence": 0.6,
  "fake_probability": 0.0022,
  "decision_source": "hybrid",
  "signals": {
    "applied_rules": [
      "account_age < 3",
      "time_between_reviews < 1"
    ]
  },
  "explanation": "Suspicious behavior detected: new account with rapid review activity",
  "timestamp": "2026-03-31T02:26:35.982693Z"
}
```

| Field | Description |
|---|---|
| `risk_level` | Final verdict: `high` (fake), `medium` (suspicious), `low` (genuine) |
| `confidence` | Final combined score (0–1) after ML + rules. Higher = more likely fake |
| `fake_probability` | Raw ML model score only, before rules are applied (0–1) |
| `decision_source` | `model` if only ML was used, `hybrid` if rules also applied |
| `signals.applied_rules` | List of rules that triggered. Empty if none |
| `explanation` | Human-readable summary of the decision |
| `timestamp` | UTC timestamp of when the prediction was made |

> **Note:** `fake_probability` and `confidence` are different. `fake_probability` is what the ML model thinks based on text alone. `confidence` is the final score after behavioral rules are factored in.

---

## Rule Engine

Rules are evaluated after the ML score and can push the final `confidence` score higher:

| Rule | Weight Added | Description |
|---|---|---|
| `report_count > 10` | +0.3 | Account has been reported more than 10 times |
| `account_age < 3` | +0.3 | Account is less than 3 years old |
| `time_between_reviews < 1` | +0.3 | Reviews posted less than 1 day apart on average |
| `rating == 5 and fake_prob > 0.6` | +0.1 | 5-star rating combined with high ML fake score |

Final score is clamped to a maximum of 1.0.