<div align="center">

# 🛡️ SpamShield AI

### AI-Powered Communication Defense Platform

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?logo=github)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Enterprise-grade spam detection powered by transformers, explainable AI, and real-time analytics.*

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Nginx (Reverse Proxy)                │
├─────────────────────────────┬───────────────────────────────┤
│    Next.js Frontend (:3000) │    FastAPI Backend (:8000)     │
│    - Landing Page           │    - REST API (v1)             │
│    - Dashboard              │    - JWT Auth + RBAC           │
│    - AI Demo                │    - Prediction Service        │
│    - Analytics              │    - Analytics Service         │
├─────────────────────────────┴───────────────────────────────┤
│                     ML Inference Engine                      │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│   │ Naive    │ │ Logistic │ │ XGBoost  │ │ DistilBERT   │  │
│   │ Bayes    │ │ Reg.     │ │ + LGBM   │ │ + RoBERTa    │  │
│   └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│   ┌──────────┐ ┌──────────┐ ┌──────────────────────────┐   │
│   │ TextCNN  │ │ BiLSTM   │ │ LIME + SHAP (XAI)        │   │
│   └──────────┘ └──────────┘ └──────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL    │    Redis     │  MLflow   │  Prometheus     │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| **13 ML Models** | NB, LR, SVM, RF, XGBoost, LightGBM, TextCNN, LSTM, BiLSTM, GRU, BERT, DistilBERT, RoBERTa |
| **Explainable AI** | LIME & SHAP word-level explanations for every prediction |
| **Real-Time API** | FastAPI async REST API with sub-50ms inference |
| **JWT Auth** | Secure authentication with access/refresh tokens and RBAC |
| **Analytics** | Dashboard with threat statistics, timelines, and model benchmarks |
| **Batch Processing** | Classify up to 100 messages per API call |
| **12-Step NLP** | Cleaning, tokenization, normalization, slang/Hinglish support |
| **ONNX Export** | Transformer models exportable to ONNX for edge deployment |
| **Full MLOps** | MLflow tracking, DVC versioning, Prometheus monitoring |
| **K8s Ready** | Auto-scaling deployments with health probes and ingress |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (optional)

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/your-username/spamshield-ai.git
cd spamshield-ai
cp configs/.env.example configs/.env
docker compose up -d
```
- **Frontend:** http://localhost
- **API Docs:** http://localhost/docs
- **Health:** http://localhost/health

### Option 2: Manual Setup

```bash
# Backend
pip install -r requirements.txt
cp configs/.env.example configs/.env
uvicorn backend.app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Option 3: Train Models First
```bash
python -m ml.training.train_traditional
```

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/v1/auth/register` | ❌ | Create account |
| `POST` | `/api/v1/auth/login` | ❌ | Get JWT tokens |
| `POST` | `/api/v1/auth/refresh` | ❌ | Refresh token |
| `GET` | `/api/v1/auth/me` | ✅ | User profile |
| `POST` | `/api/v1/predict` | Optional | Classify text |
| `POST` | `/api/v1/predict/batch` | ✅ | Batch classify |
| `GET` | `/api/v1/predict/history` | ✅ | History |
| `GET` | `/api/v1/predict/models` | ❌ | List models |
| `POST` | `/api/v1/feedback` | ✅ | Submit correction |
| `GET` | `/api/v1/analytics/summary` | ✅ | Dashboard stats |
| `GET` | `/api/v1/analytics/benchmarks` | ❌ | Model benchmarks |
| `GET` | `/api/v1/analytics/timeline` | ✅ | Prediction timeline |
| `GET` | `/health` | ❌ | Liveness probe |
| `GET` | `/ready` | ❌ | Readiness probe |

### Example: Classify a message
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You won a free iPhone! Click now!", "explain": true}'
```

## 📁 Project Structure

```
spamshield-ai/
├── backend/            # FastAPI REST API
│   └── app/
│       ├── api/        # Routes (auth, predict, feedback, analytics)
│       ├── core/       # DB, security, logging
│       ├── middleware/  # Rate limiting, security headers
│       ├── models/     # SQLAlchemy ORM models
│       ├── schemas/    # Pydantic DTOs
│       └── services/   # Business logic
├── frontend/           # Next.js 15 landing page
│   └── src/
│       ├── app/        # Pages, layout, CSS
│       └── components/ # 13 premium UI components
├── ml/                 # ML pipeline
│   ├── preprocessing/  # NLP pipeline (clean → tokenize → normalize)
│   ├── features/       # TF-IDF feature extraction
│   ├── models/         # Traditional, DL, Transformer models
│   ├── training/       # Training scripts
│   ├── evaluation/     # Metrics and benchmarking
│   ├── explainability/ # LIME + SHAP
│   ├── inference/      # Production inference engine
│   └── tracking/       # MLflow experiment tracking
├── datasets/           # Data pipeline (download, load, augment)
├── docker/             # Dockerfiles + Nginx config
├── kubernetes/         # K8s manifests (deploy, HPA, ingress)
├── monitoring/         # Prometheus + Grafana config
├── alembic/            # Database migrations
├── configs/            # Environment configuration
├── docker-compose.yml  # Full stack orchestration
└── .github/workflows/  # CI/CD pipeline
```

## 🧠 ML Pipeline

### Preprocessing (12 steps)
Lowercasing → URL removal → HTML cleaning → Emoji handling → Unicode normalization → Tokenization → Stopword removal → Lemmatization → Slang normalization → Hinglish normalization → Regex cleaning → Spell correction

### Models

| Category | Models | Use Case |
|----------|--------|----------|
| **Traditional** | NB, LR, SVM, RF, XGBoost, LightGBM | Fast inference, baseline |
| **Deep Learning** | TextCNN, LSTM, BiLSTM, GRU | Sequence understanding |
| **Transformers** | BERT, DistilBERT, RoBERTa | State-of-the-art accuracy |

## 🔒 Security

- JWT access + refresh tokens (HS256)
- Bcrypt password hashing (12 rounds)
- Rate limiting (120 req/min)
- Security headers (HSTS, CSP, X-Frame-Options)
- Input validation via Pydantic
- Non-root Docker containers
- RBAC (user/admin roles)

## 📊 Monitoring

- **Prometheus:** HTTP latency, prediction throughput, confidence distribution
- **Grafana:** Pre-built dashboards for API and ML metrics
- **Structured Logging:** JSON format compatible with ELK Stack

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by SpamShield AI Team**

</div>
