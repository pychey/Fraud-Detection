# Fake Review Detection API

Production-ready FastAPI service for fake review detection with:
- ML prediction from `predict.py`
- Hybrid rule-based adjustments
- PostgreSQL logging for every prediction request

## API Contract

### POST /predict

Request JSON:

```json
{
  "text": "This product is amazing",
  "review_count": 6,
  "account_age": 1,
  "rating": 5,
  "time_between_reviews": 0.2,
  "report_count": 0
}
```

Required field:
- `text` (string)

Optional fields:
- `review_count` (int)
- `account_age` (float)
- `rating` (float, 0-5)
- `time_between_reviews` (float)
- `report_count` (int)

Response JSON:

```json
{
  "label": "suspicious",
  "confidence": 0.8712,
  "reason": "ML prediction: suspicious (0.87) + Rule applied: account_age < 2 and review_count > 5"
}
```

### GET /health

Health endpoint:

```json
{
  "status": "ok"
}
```

## Rule-Based Logic

Rules are applied after ML output:
- If `report_count > 3` then output label becomes `suspicious` (unless already `fake`)
- If `account_age < 2` and `review_count > 5` then output label becomes `suspicious` (unless already `fake`)

## Local Run (without Docker)

1. Create and activate virtual env:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set database URL:

```powershell
$env:DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/fake_reviews"
```

4. Start API:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Deploy with Docker Compose (API + PostgreSQL)

From project root:

```powershell
docker compose up -d --build
```

API:
- http://localhost:8000
- Swagger: http://localhost:8000/docs

Stop services:

```powershell
docker compose down
```

Stop and delete DB volume too:

```powershell
docker compose down -v
```

## Team Integration Checklist

Share these with your team app developers:
- Base URL (example: `https://api.yourdomain.com`)
- Endpoint: `POST /predict`
- Auth method (if enabled in your deployment)
- Request and response schema above
- Sample payload and response
- Retry strategy (for example: retry on 5xx with exponential backoff)

## Production Notes

- Keep model files on server; do not ship model weights inside client apps.
- Put API behind Nginx or cloud load balancer with HTTPS.
- Add auth (API key or JWT) before internet exposure.
- Add monitoring/log aggregation (for example, Grafana/Prometheus or cloud APM).
