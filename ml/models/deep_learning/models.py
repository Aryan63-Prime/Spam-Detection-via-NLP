"""
Deep Learning Models for Spam Classification
==============================================
Implements CNN, LSTM, BiLSTM, and GRU text classifiers using PyTorch.

Architecture Decisions:
- All models inherit from nn.Module AND BaseModel
- Shared embedding layer with optional pretrained weights
- Consistent train/predict/save/load API
- CPU-optimized (user has no local GPU, uses Colab for training)
- Export-ready for ONNX conversion

Model Comparison:
┌──────────┬─────────────┬────────────────────────────────────────┐
│ Model    │ Strengths   │ Best For                               │
├──────────┼─────────────┼────────────────────────────────────────┤
│ TextCNN  │ Fast, local │ Short texts (SMS), fixed patterns      │
│ LSTM     │ Sequential  │ Long-range dependencies                │
│ BiLSTM   │ Bi-context  │ Best DL baseline for text              │
│ GRU      │ Efficient   │ Similar to LSTM, fewer parameters      │
└──────────┴─────────────┴────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import DEEP_LEARNING_DEFAULTS, MODELS_DIR
from ml.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Device Selection (CPU-first, GPU if available)
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference.")
    return device


# ──────────────────────────────────────────────
# Text CNN
# ──────────────────────────────────────────────

class TextCNNModule(nn.Module):
    """
    CNN for text classification (Kim 2014).

    Architecture:
    - Embedding → Multiple parallel conv filters → MaxPool → FC → Sigmoid
    - Filter sizes [3, 4, 5] capture 3/4/5-gram patterns
    - Each filter produces feature maps that capture local patterns

    Why CNN for spam:
    - Captures local n-gram patterns ("click here", "free money")
    - Parallel filters capture multiple pattern sizes
    - Fast inference (parallelizable, no sequential dependency)
    - Works well for short texts like SMS
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: tuple = (3, 4, 5),
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Parallel convolutional layers for different n-gram sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs,
                padding=fs // 2,
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        # Apply each conv filter + ReLU + MaxPool
        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(embedded))  # (batch, num_filters, seq_len)
            c = torch.max(c, dim=2)[0]      # (batch, num_filters) — global max pool
            conv_outputs.append(c)

        # Concatenate all filter outputs
        combined = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * n_filters)
        combined = self.dropout(combined)

        logits = self.fc(combined)  # (batch, num_classes)
        return logits


# ──────────────────────────────────────────────
# LSTM / BiLSTM
# ──────────────────────────────────────────────

class LSTMModule(nn.Module):
    """
    LSTM / BiLSTM for text classification.

    Architecture:
    - Embedding → LSTM(s) → Last hidden state → FC → Sigmoid

    LSTM gates:
    - Forget gate: What to discard from cell state
    - Input gate: What new info to store
    - Output gate: What to output from cell state

    BiLSTM: Processes text left→right AND right→left,
    capturing both forward and backward context.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # (batch, seq_len, hidden*directions)

        # Use last hidden state from both directions
        if self.bidirectional:
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_cat = hidden[-1]

        output = self.dropout(hidden_cat)
        logits = self.fc(output)
        return logits


# ──────────────────────────────────────────────
# GRU
# ──────────────────────────────────────────────

class GRUModule(nn.Module):
    """
    GRU (Gated Recurrent Unit) for text classification.

    Simpler than LSTM (2 gates vs 3):
    - Update gate: Controls how much past info to keep
    - Reset gate: Controls how to combine new input with past

    Why GRU:
    - Fewer parameters than LSTM → faster training
    - Comparable accuracy for most text tasks
    - Better for smaller datasets (less overfitting)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)

        if self.bidirectional:
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_cat = hidden[-1]

        output = self.dropout(hidden_cat)
        logits = self.fc(output)
        return logits


# ──────────────────────────────────────────────
# Unified Deep Learning Wrapper (BaseModel)
# ──────────────────────────────────────────────

class DeepLearningModel(BaseModel):
    """
    Wrapper that makes PyTorch modules conform to the BaseModel interface.
    Handles training loop, prediction, save/load for all DL architectures.
    """

    ARCHITECTURES = {
        "text_cnn": TextCNNModule,
        "lstm": lambda **kw: LSTMModule(bidirectional=False, **kw),
        "bilstm": lambda **kw: LSTMModule(bidirectional=True, **kw),
        "gru": GRUModule,
    }

    def __init__(
        self,
        architecture: str = "bilstm",
        vocab_size: int = 30000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        batch_size: int = 32,
        epochs: int = 15,
        learning_rate: float = 1e-3,
        patience: int = 3,
    ) -> None:
        super().__init__(name=architecture, model_type="deep_learning")

        self.architecture = architecture
        self.device = get_device()
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.patience = patience
        self.vocab_size = vocab_size

        # Build model
        arch_cls = self.ARCHITECTURES.get(architecture)
        if arch_cls is None:
            raise ValueError(f"Unknown architecture: {architecture}")

        if architecture == "text_cnn":
            self.model = TextCNNModule(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        else:
            self.model = arch_cls(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.model = self.model.to(self.device)
        self.metadata.hyperparameters = {
            "architecture": architecture,
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "lr": learning_rate,
        }

        logger.info(
            "%s model created. Parameters: %s",
            architecture.upper(),
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        """
        Train with early stopping.
        X_train/X_val should be integer-encoded sequences (numpy arrays).
        """
        start = time.perf_counter()

        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validate
            if val_loader:
                val_loss = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                logger.info(
                    "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f",
                    epoch + 1, self.epochs, avg_train_loss, val_loss,
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info("Early stopping at epoch %d.", epoch + 1)
                        break
            else:
                logger.info("Epoch %d/%d — train_loss: %.4f", epoch + 1, self.epochs, avg_train_loss)

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        self._is_trained = True
        train_time = time.perf_counter() - start
        self.metadata.training_time_seconds = train_time

        logger.info("%s trained in %.1fs.", self.architecture.upper(), train_time)
        return {"history": history, "training_time": train_time}

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        loader = self._create_dataloader(X, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def _validate(self, val_loader, criterion) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def _create_dataloader(self, X, y=None, shuffle=False):
        """Create a PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.LongTensor(np.array(X))
        if y is not None:
            y_tensor = torch.LongTensor(np.array(y))
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def save(self, directory: Optional[Path] = None) -> Path:
        """Save PyTorch model state dict."""
        save_dir = directory or MODELS_DIR / self.name
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "metadata": self.metadata.__dict__,
        }, model_path)

        logger.info("DL model saved: %s", model_path)
        return save_dir

    def load(self, directory: Optional[Path] = None) -> DeepLearningModel:
        """Load PyTorch model state dict."""
        load_dir = directory or MODELS_DIR / self.name
        model_path = load_dir / "model.pt"

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._is_trained = True

        logger.info("DL model loaded: %s", model_path)
        return self

    def _get_model_object(self) -> Any:
        return self.model.state_dict()

    def _set_model_object(self, model: Any) -> None:
        self.model.load_state_dict(model)
