"""
Traditional ML Training Script
================================
End-to-end training pipeline:
1. Download/load dataset
2. Preprocess text
3. Extract TF-IDF features
4. Train all 6 traditional models
5. Evaluate and benchmark
6. Save best model

Usage:
    python -m ml.training.train_traditional
    python -m ml.training.train_traditional --models naive_bayes logistic_regression
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import MODELS_DIR, ModelType
from ml.evaluation.metrics import ModelEvaluator
from ml.features.tfidf import TfidfFeatureExtractor
from ml.models.traditional.classifiers import (
    TRADITIONAL_MODEL_REGISTRY,
    create_traditional_model,
)
from ml.preprocessing.pipeline import PreprocessingPipeline
from datasets.download import DatasetDownloader
from datasets.loader import DatasetLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_traditional_models(
    model_names: list[str] | None = None,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> None:
    """
    Full training pipeline for traditional ML models.

    Steps:
    1. Download SMS Spam Collection (if not exists)
    2. Load and split dataset
    3. Preprocess all texts
    4. Extract TF-IDF features
    5. Train each model
    6. Evaluate on test set
    7. Print benchmark table
    8. Save all models
    """
    total_start = time.perf_counter()

    # ── Step 1: Download Dataset ─────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Dataset Download")
    logger.info("=" * 60)

    downloader = DatasetDownloader()
    dataset_path = downloader.download_sms_spam()
    logger.info("Dataset path: %s", dataset_path)

    # ── Step 2: Load & Split ─────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Load & Split Dataset")
    logger.info("=" * 60)

    loader = DatasetLoader()
    df = loader.load_sms_spam(dataset_path)
    split = loader.create_split(df, test_size=test_size, val_size=val_size)

    logger.info("Split summary:")
    for split_name, stats in split.summary().items():
        logger.info("  %s: %s", split_name, stats)

    # ── Step 3: Preprocess ───────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Text Preprocessing")
    logger.info("=" * 60)

    pipeline = PreprocessingPipeline()

    logger.info("Preprocessing training texts...")
    train_texts = pipeline.process_batch_to_texts(split.X_train.tolist())
    logger.info("Preprocessing validation texts...")
    val_texts = pipeline.process_batch_to_texts(split.X_val.tolist())
    logger.info("Preprocessing test texts...")
    test_texts = pipeline.process_batch_to_texts(split.X_test.tolist())

    # ── Step 4: Feature Extraction ───────────────
    logger.info("=" * 60)
    logger.info("STEP 4: TF-IDF Feature Extraction")
    logger.info("=" * 60)

    tfidf = TfidfFeatureExtractor(
        max_features=max_features,
        ngram_range=ngram_range,
    )

    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)

    logger.info("Feature matrix shapes: train=%s, val=%s, test=%s",
                X_train.shape, X_val.shape, X_test.shape)

    # Save TF-IDF vectorizer
    tfidf.save("tfidf_spam")

    # Top features
    top_features = tfidf.get_top_features(20)
    logger.info("Top 20 TF-IDF features by IDF:")
    for feat, score in top_features:
        logger.info("  %s: %.4f", feat, score)

    # ── Step 5: Train Models ─────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Model Training")
    logger.info("=" * 60)

    # Select models to train
    if model_names is None:
        model_names = list(TRADITIONAL_MODEL_REGISTRY.keys())

    trained_models = []
    for name in model_names:
        logger.info("Training: %s", name)
        try:
            model = create_traditional_model(name)
            model.train(
                X_train, split.y_train,
                X_val=X_val, y_val=split.y_val,
            )
            trained_models.append(model)
            model.save()
            logger.info("  ✓ %s trained and saved.", name)
        except Exception as e:
            logger.error("  ✗ Failed to train %s: %s", name, e)

    # ── Step 6: Evaluate & Benchmark ─────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluation & Benchmark")
    logger.info("=" * 60)

    evaluator = ModelEvaluator()
    results = evaluator.benchmark(trained_models, X_test, split.y_test)

    # Print benchmark table
    evaluator.print_benchmark_table(results)

    # Print detailed report for best model
    if results:
        best = results[0]
        logger.info("Best model: %s (F1=%.4f)", best.model_name, best.f1)
        logger.info("\nDetailed Classification Report:")
        print(best.classification_report)
        logger.info("Confusion Matrix: %s", best.confusion_matrix)

    # Save benchmark results
    results_path = MODELS_DIR / "benchmark_traditional.json"
    evaluator.save_results(results, results_path)

    total_time = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — Total time: %.1fs", total_time)
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train traditional ML models for spam detection."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(TRADITIONAL_MODEL_REGISTRY.keys()),
        help="Models to train. Default: all.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Max TF-IDF features.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram size.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size.",
    )

    args = parser.parse_args()

    train_traditional_models(
        model_names=args.models,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
    )


if __name__ == "__main__":
    main()
