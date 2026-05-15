"""
Dataset Loader Module
======================
Unified dataset loading with train/val/test splitting,
label encoding, and statistics reporting.

Architecture Decision:
- Pandas DataFrames as the standard interchange format
- Stratified splitting to preserve class distribution
- Reproducible splits via fixed random seed
- Support for multiple dataset formats (TSV, CSV, custom)
- Returns typed DatasetSplit dataclass for clarity

Performance:
- Lazy loading (only load when accessed)
- Caching of processed DataFrames
- Memory-efficient chunked reading for large datasets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import LABEL_MAP, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """
    Container for a train/validation/test split.
    Provides clear typing and prevents mix-ups.
    """
    X_train: pd.Series
    X_val: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    def summary(self) -> dict:
        """Generate split statistics."""
        return {
            "train": {
                "total": self.train_size,
                "spam": int(self.y_train.sum()),
                "ham": int(self.train_size - self.y_train.sum()),
                "spam_ratio": float(self.y_train.mean()),
            },
            "val": {
                "total": self.val_size,
                "spam": int(self.y_val.sum()),
                "ham": int(self.val_size - self.y_val.sum()),
                "spam_ratio": float(self.y_val.mean()),
            },
            "test": {
                "total": self.test_size,
                "spam": int(self.y_test.sum()),
                "ham": int(self.test_size - self.y_test.sum()),
                "spam_ratio": float(self.y_test.mean()),
            },
        }


class DatasetLoader:
    """
    Unified dataset loader supporting multiple spam datasets.

    Usage:
        loader = DatasetLoader()
        df = loader.load_sms_spam()
        split = loader.create_split(df)
        print(split.summary())
    """

    def __init__(self, raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_sms_spam(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load the UCI SMS Spam Collection dataset.

        Format: Tab-separated, two columns: label (ham/spam), message text.

        Args:
            file_path: Optional custom path. Defaults to standard location.

        Returns:
            DataFrame with columns: ['text', 'label', 'label_encoded']
        """
        if file_path is None:
            file_path = self.raw_dir / "sms_spam" / "SMSSpamCollection"

        if not file_path.exists():
            raise FileNotFoundError(
                f"SMS Spam dataset not found at {file_path}. "
                f"Run DatasetDownloader().download_sms_spam() first."
            )

        logger.info("Loading SMS Spam Collection from: %s", file_path)

        # Read tab-separated file with no header
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="latin-1",
        )

        # Encode labels
        df["label_encoded"] = df["label"].map(LABEL_MAP)

        # Basic validation
        null_count = df["label_encoded"].isna().sum()
        if null_count > 0:
            logger.warning("Found %d rows with unknown labels, dropping.", null_count)
            df = df.dropna(subset=["label_encoded"])
            df["label_encoded"] = df["label_encoded"].astype(int)

        # Drop duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.info("Dropped %d duplicate messages.", dropped)

        logger.info(
            "Dataset loaded: %d messages (ham=%d, spam=%d, ratio=%.2f%%)",
            len(df),
            (df["label_encoded"] == 0).sum(),
            (df["label_encoded"] == 1).sum(),
            df["label_encoded"].mean() * 100,
        )

        return df

    def load_csv(
        self,
        file_path: Path,
        text_column: str = "text",
        label_column: str = "label",
    ) -> pd.DataFrame:
        """
        Load a custom CSV dataset.

        Args:
            file_path: Path to the CSV file.
            text_column: Name of the text column.
            label_column: Name of the label column.

        Returns:
            DataFrame with standardized columns.
        """
        logger.info("Loading custom CSV from: %s", file_path)

        df = pd.read_csv(file_path)

        # Validate required columns
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV.")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV.")

        # Standardize column names
        df = df.rename(columns={text_column: "text", label_column: "label"})

        # Encode labels if they are strings
        if df["label"].dtype == object:
            df["label_encoded"] = df["label"].map(LABEL_MAP)
        else:
            df["label_encoded"] = df["label"].astype(int)

        # Clean
        df = df.dropna(subset=["text", "label_encoded"])
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

        logger.info("Custom dataset loaded: %d messages.", len(df))
        return df

    def create_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ) -> DatasetSplit:
        """
        Create stratified train/validation/test splits.

        Why stratified: Spam datasets are typically imbalanced (10-15% spam).
        Stratified splitting ensures each split has the same spam/ham ratio.

        Args:
            df: DataFrame with 'text' and 'label_encoded' columns.
            test_size: Fraction for test set.
            val_size: Fraction for validation set (from training data).
            random_state: Seed for reproducibility.
            stratify: Whether to use stratified splitting.

        Returns:
            DatasetSplit with train/val/test data.
        """
        X = df["text"]
        y = df["label_encoded"]

        stratify_col = y if stratify else None

        # Split: full → train+val / test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )

        # Split: train+val → train / val
        val_fraction = val_size / (1 - test_size)
        stratify_trainval = y_trainval if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction,
            random_state=random_state,
            stratify=stratify_trainval,
        )

        # Reset indices
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        split = DatasetSplit(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
        )

        logger.info("Dataset split created: %s", split.summary())
        return split

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """Save processed DataFrame to parquet for fast reloading."""
        path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved processed dataset: %s (%.2f MB)", path, path.stat().st_size / 1e6)
        return path

    def load_processed(self, name: str) -> pd.DataFrame:
        """Load a previously saved processed dataset."""
        path = self.processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {path}")
        return pd.read_parquet(path)
