# SpamShield AI — Deployment Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker | 24+ | Container runtime |
| Docker Compose | 2.20+ | Multi-service orchestration |
| kubectl | 1.28+ | Kubernetes CLI (optional) |
| Helm | 3.x | K8s package manager (optional) |

## 1. Environment Setup

```bash
# Copy and edit environment variables
cp configs/.env.example configs/.env

# Required changes for production:
# - SEC_JWT_SECRET_KEY → generate with: openssl rand -hex 32
# - DB_POSTGRES_PASSWORD → strong random password
# - ENVIRONMENT=production
# - DEBUG=false
```

## 2. Docker Compose Deployment

### Start all services
```bash
docker compose up -d
```

### Verify health
```bash
curl http://localhost/health
curl http://localhost/ready
```

### View logs
```bash
docker compose logs -f backend
docker compose logs -f frontend
```

### Scale backend
```bash
docker compose up -d --scale backend=3
```

## 3. Kubernetes Deployment

### Apply manifests
```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/backend.yaml
kubectl apply -f kubernetes/frontend.yaml
kubectl apply -f kubernetes/ingress.yaml
```

### Verify
```bash
kubectl get pods -n spamshield
kubectl get svc -n spamshield
kubectl get hpa -n spamshield
```

## 4. Database Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "initial tables"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 5. Model Training

```bash
# Train traditional models (CPU)
python -m ml.training.train_traditional

# Train on Google Colab (GPU)
# Upload notebooks/training.ipynb to Colab
```

## 6. Monitoring Setup

```bash
# Start Prometheus + Grafana
docker compose -f docker-compose.monitoring.yml up -d

# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
```

## 7. SSL/TLS (Production)

For K8s with cert-manager:
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.0/cert-manager.yaml
```

The ingress manifest already includes cert-manager annotations.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | Check `docker compose logs backend` for DB connection errors |
| Models not loading | Run training pipeline first: `python -m ml.training.train_traditional` |
| Frontend 502 | Wait for backend health check to pass |
| Slow predictions | Enable model caching in configs/.env |
