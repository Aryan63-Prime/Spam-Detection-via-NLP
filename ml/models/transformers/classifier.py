"""
Transformer-Based Spam Classifiers
=====================================
Fine-tuning BERT, DistilBERT, RoBERTa for spam classification
using the Hugging Face Transformers library.

Architecture Decision:
- Hugging Face Trainer API for robust training
- DistilBERT as the default production model (speed vs accuracy)
- CPU-friendly with optional GPU acceleration via Colab
- ONNX export support for optimized inference

Model Comparison:
┌─────────────┬────────┬───────────┬─────────────────┐
│ Model       │ Params │ Speed     │ Accuracy (Spam)  │
├─────────────┼────────┼───────────┼─────────────────┤
│ BERT-base   │ 110M   │ Slow      │ ~99.3%          │
│ DistilBERT  │ 66M    │ 2x faster │ ~99.1%          │
│ RoBERTa     │ 125M   │ Slow      │ ~99.4%          │
└─────────────┴────────┴───────────┴─────────────────┘

For production: DistilBERT (best speed/accuracy tradeoff).
For Colab training: BERT or RoBERTa (more accurate, needs GPU).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ml.config import MODELS_DIR, TRANSFORMER_DEFAULTS
from ml.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class TransformerSpamClassifier(BaseModel):
    """
    Unified transformer-based spam classifier.

    Supports BERT, DistilBERT, RoBERTa, DeBERTa via a single class.
    Uses Hugging Face's AutoModelForSequenceClassification.

    Usage:
        # For local CPU training (small dataset)
        model = TransformerSpamClassifier(model_name="distilbert")
        model.train(train_texts, train_labels, val_texts, val_labels)

        # For inference
        model = TransformerSpamClassifier(model_name="distilbert")
        model.load()
        predictions = model.predict(["free money click here!"])
    """

    def __init__(
        self,
        model_name: str = "distilbert",
        max_length: int = 256,
        batch_size: int = 16,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(name=model_name, model_type="transformer")

        self.model_key = model_name
        self.pretrained_name = TRANSFORMER_DEFAULTS.model_names.get(
            model_name, model_name
        )
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        self.metadata.hyperparameters = {
            "pretrained_model": self.pretrained_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
        }

        logger.info(
            "TransformerSpamClassifier initialized: %s (device=%s)",
            self.pretrained_name, self.device,
        )

    def _load_pretrained(self) -> None:
        """Lazy-load pretrained model and tokenizer."""
        if self.model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading pretrained model: %s", self.pretrained_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_name,
            num_labels=2,
            problem_type="single_label_classification",
        )
        self.model = self.model.to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded. Parameters: %s", f"{param_count:,}")

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        """
        Fine-tune the transformer model.

        Args:
            X_train: List of training text strings.
            y_train: Training labels (0=ham, 1=spam).
            X_val: Optional validation texts.
            y_val: Optional validation labels.

        Returns:
            Training metrics dict.
        """
        from transformers import Trainer, TrainingArguments
        from datasets import Dataset

        start = time.perf_counter()
        self._load_pretrained()

        # Create HuggingFace datasets
        train_dataset = self._create_hf_dataset(X_train, y_train)
        eval_dataset = None
        if X_val is not None and y_val is not None:
            eval_dataset = self._create_hf_dataset(X_val, y_val)

        # Training arguments
        output_dir = MODELS_DIR / f"{self.model_key}_training"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.lr,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            logging_steps=50,
            save_total_limit=2,
            report_to="none",  # Disable WandB/MLflow during initial training
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Windows compatibility
        )

        # Compute metrics function
        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds),
                "precision": precision_score(labels, preds),
                "recall": recall_score(labels, preds),
            }

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset else None,
        )

        # Train
        logger.info("Starting fine-tuning: %s", self.pretrained_name)
        train_result = trainer.train()

        self._is_trained = True
        train_time = time.perf_counter() - start
        self.metadata.training_time_seconds = train_time
        self.metadata.training_samples = len(X_train)

        logger.info(
            "%s fine-tuned in %.1fs. Loss: %.4f",
            self.model_key.upper(), train_time,
            train_result.training_loss,
        )

        return {
            "training_time": train_time,
            "training_loss": train_result.training_loss,
        }

    def predict(self, X) -> np.ndarray:
        """Predict class labels for input texts."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for input texts."""
        self._load_pretrained()
        self.model.eval()

        # Handle single string input
        if isinstance(X, str):
            X = [X]

        all_probs = []

        for i in range(0, len(X), self.batch_size):
            batch_texts = list(X[i:i + self.batch_size])

            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def _create_hf_dataset(self, texts, labels):
        """Create a Hugging Face Dataset with tokenized inputs."""
        from datasets import Dataset

        texts_list = list(texts) if not isinstance(texts, list) else texts
        labels_list = list(labels) if not isinstance(labels, list) else labels

        dataset = Dataset.from_dict({
            "text": texts_list,
            "label": [int(l) for l in labels_list],
        })

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        dataset.set_format("torch")
        return dataset

    def save(self, directory: Optional[Path] = None) -> Path:
        """Save model and tokenizer."""
        save_dir = directory or MODELS_DIR / self.model_key
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(str(save_dir))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(save_dir))

        # Save metadata
        meta_path = save_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata.__dict__, f, indent=2, default=str)

        logger.info("Transformer model saved: %s", save_dir)
        return save_dir

    def load(self, directory: Optional[Path] = None) -> TransformerSpamClassifier:
        """Load model and tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        load_dir = directory or MODELS_DIR / self.model_key
        if not load_dir.exists():
            raise FileNotFoundError(f"Model not found: {load_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(load_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(load_dir))
        self.model = self.model.to(self.device)
        self._is_trained = True

        logger.info("Transformer model loaded: %s", load_dir)
        return self

    def export_onnx(self, directory: Optional[Path] = None) -> Path:
        """
        Export model to ONNX format for optimized inference.
        ONNX Runtime provides 2-3x speedup on CPU.
        """
        self._load_pretrained()
        self.model.eval()

        save_dir = directory or MODELS_DIR / f"{self.model_key}_onnx"
        save_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = save_dir / "model.onnx"

        # Create dummy input
        dummy_input = self.tokenizer(
            "test input",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}

        # Export
        torch.onnx.export(
            self.model,
            tuple(dummy_input.values()),
            str(onnx_path),
            input_names=list(dummy_input.keys()),
            output_names=["logits"],
            dynamic_axes={
                name: {0: "batch_size", 1: "sequence"}
                for name in dummy_input.keys()
            },
            opset_version=14,
        )

        # Save tokenizer alongside
        self.tokenizer.save_pretrained(str(save_dir))

        logger.info("ONNX model exported: %s", onnx_path)
        return onnx_path

    def _get_model_object(self) -> Any:
        return self.model

    def _set_model_object(self, model: Any) -> None:
        self.model = model
