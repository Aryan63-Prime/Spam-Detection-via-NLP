# SpamShield AI — API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication

All protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <access_token>
```

### Get Tokens
```bash
# Register
curl -X POST /api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "username": "john", "password": "SecurePass1!"}'

# Login
curl -X POST /api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass1!"}'
# Response: { "access_token": "...", "refresh_token": "...", "token_type": "bearer" }
```

## Prediction API

### Classify Single Text
```bash
POST /api/v1/predict
Content-Type: application/json

{
  "text": "You won a free iPhone! Click to claim!",
  "model": "logistic_regression",
  "explain": true,
  "explain_method": "lime",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "text": "You won a free iPhone! Click to claim!",
  "prediction": "spam",
  "confidence": 0.97,
  "spam_probability": 0.97,
  "ham_probability": 0.03,
  "model_used": "logistic_regression",
  "processing_time_ms": 12.5,
  "explanation": {
    "method": "lime",
    "word_importances": [
      {"word": "won", "importance": 0.85},
      {"word": "free", "importance": 0.78},
      {"word": "click", "importance": 0.65}
    ]
  }
}
```

### Batch Classify (Authenticated)
```bash
POST /api/v1/predict/batch
Authorization: Bearer <token>

{
  "texts": ["Hello, how are you?", "FREE MONEY NOW!!!"],
  "model": "logistic_regression"
}
```

### Available Models
```bash
GET /api/v1/predict/models
```

## Feedback API

### Submit Correction
```bash
POST /api/v1/feedback
Authorization: Bearer <token>

{
  "prediction_id": "uuid-here",
  "correct_label": "ham",
  "comment": "This was a legitimate message"
}
```

## Analytics API

### Dashboard Summary
```bash
GET /api/v1/analytics/summary
Authorization: Bearer <token>
```

### Model Benchmarks
```bash
GET /api/v1/analytics/benchmarks
```

### Prediction Timeline
```bash
GET /api/v1/analytics/timeline?days=7
Authorization: Bearer <token>
```

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| Global | 120 requests/minute |
| `/api/v1/predict` | 60 requests/minute |
| `/api/v1/predict/batch` | 10 requests/minute |

## Error Responses

```json
{
  "detail": "Error description here"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request / validation error |
| 401 | Unauthorized / invalid token |
| 403 | Forbidden / insufficient role |
| 404 | Resource not found |
| 409 | Conflict (duplicate) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable |

## Interactive Docs

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json
