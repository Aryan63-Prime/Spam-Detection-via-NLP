"""
Preprocessing Pipeline Orchestrator
=====================================
Composes cleaner, tokenizer, and normalizer into a single unified pipeline.
This is the main entry point for all text preprocessing in the project.

Architecture Decision:
- Facade pattern: single clean API over multiple subsystems
- Configurable pipeline stages (can skip any stage)
- Consistent interface for training AND inference (no skew)
- Supports both single-text and batch processing
- Includes metadata extraction (spam keywords, text stats)

Usage:
    from ml.preprocessing.pipeline import PreprocessingPipeline, PipelineConfig

    # Default pipeline
    pipeline = PreprocessingPipeline()
    result = pipeline.process("FREE PRIZE! Click http://spam.com NOW!!!")
    print(result.cleaned_text)    # "free prize click __URL__ now"
    print(result.tokens)          # ["free", "prize", "click", "__url__", "now"]
    print(result.spam_keywords)   # ["free", "prize"]

    # Batch processing
    results = pipeline.process_batch(["spam msg 1", "ham msg 2"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ml.preprocessing.cleaner import CleanerConfig, TextCleaner
from ml.preprocessing.normalizer import NormalizerConfig, TextNormalizer
from ml.preprocessing.tokenizer import BaseTokenizer, TokenizerFactory

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Master configuration for the preprocessing pipeline.
    Aggregates configs for each sub-component.
    """
    # Sub-component configs
    cleaner_config: CleanerConfig = field(default_factory=CleanerConfig)
    normalizer_config: NormalizerConfig = field(default_factory=NormalizerConfig)
    tokenizer_backend: str = "regex"  # "nltk" | "spacy" | "whitespace" | "regex"

    # Pipeline behavior
    enable_cleaning: bool = True
    enable_tokenization: bool = True
    enable_normalization: bool = True
    extract_metadata: bool = True

    # Output options
    join_tokens: bool = True  # Whether to rejoin tokens into string
    token_separator: str = " "


@dataclass
class ProcessedText:
    """
    Result of preprocessing a single text.
    Contains all intermediate and final outputs.
    """
    original_text: str
    cleaned_text: str
    tokens: list[str]
    normalized_tokens: list[str]
    final_text: str  # Rejoined normalized tokens

    # Metadata (useful as features)
    original_length: int = 0
    cleaned_length: int = 0
    token_count: int = 0
    spam_keyword_count: int = 0
    spam_keywords: list[str] = field(default_factory=list)
    has_url: bool = False
    has_email: bool = False
    has_phone: bool = False
    has_currency: bool = False
    uppercase_ratio: float = 0.0
    digit_ratio: float = 0.0
    special_char_ratio: float = 0.0
    processing_time_ms: float = 0.0


class PreprocessingPipeline:
    """
    Production-grade NLP preprocessing pipeline.

    Orchestrates text cleaning, tokenization, and normalization
    into a single, configurable pipeline. Extracts useful metadata
    for downstream feature engineering.

    Thread-safety: Each pipeline instance is thread-safe as long as
    the underlying components are (NLTK lemmatizer is thread-safe,
    spaCy is thread-safe for tokenization).

    Performance:
    - Single text: ~1ms on CPU
    - Batch of 10K: ~5s on CPU (with regex tokenizer)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        # Initialize sub-components
        self._cleaner = TextCleaner(self.config.cleaner_config)
        self._tokenizer: BaseTokenizer = TokenizerFactory.create(
            self.config.tokenizer_backend
        )
        self._normalizer = TextNormalizer(self.config.normalizer_config)

        logger.info(
            "PreprocessingPipeline initialized: "
            "cleaning=%s, tokenization=%s, normalization=%s, "
            "tokenizer=%s",
            self.config.enable_cleaning,
            self.config.enable_tokenization,
            self.config.enable_normalization,
            self.config.tokenizer_backend,
        )

    def process(self, text: str) -> ProcessedText:
        """
        Process a single text through the full pipeline.

        Args:
            text: Raw input text.

        Returns:
            ProcessedText with all intermediate and final outputs.
        """
        start_time = time.perf_counter()

        # Extract metadata from original text
        metadata = self._extract_metadata(text) if self.config.extract_metadata else {}

        # Stage 1: Cleaning
        if self.config.enable_cleaning:
            cleaned = self._cleaner.clean(text)
        else:
            cleaned = text

        # Stage 2: Tokenization
        if self.config.enable_tokenization:
            tokens = self._tokenizer.tokenize(cleaned)
        else:
            tokens = cleaned.split()

        # Stage 3: Normalization
        if self.config.enable_normalization:
            normalized = self._normalizer.normalize(tokens)
        else:
            normalized = tokens

        # Build final text
        if self.config.join_tokens:
            final_text = self.config.token_separator.join(normalized)
        else:
            final_text = cleaned

        # Extract spam keywords
        spam_keywords = self._normalizer.extract_spam_keywords(normalized)

        processing_time = (time.perf_counter() - start_time) * 1000  # ms

        return ProcessedText(
            original_text=text,
            cleaned_text=cleaned,
            tokens=tokens,
            normalized_tokens=normalized,
            final_text=final_text,
            original_length=len(text),
            cleaned_length=len(cleaned),
            token_count=len(normalized),
            spam_keyword_count=len(spam_keywords),
            spam_keywords=spam_keywords,
            has_url=metadata.get("has_url", False),
            has_email=metadata.get("has_email", False),
            has_phone=metadata.get("has_phone", False),
            has_currency=metadata.get("has_currency", False),
            uppercase_ratio=metadata.get("uppercase_ratio", 0.0),
            digit_ratio=metadata.get("digit_ratio", 0.0),
            special_char_ratio=metadata.get("special_char_ratio", 0.0),
            processing_time_ms=processing_time,
        )

    def process_batch(self, texts: list[str]) -> list[ProcessedText]:
        """
        Process a batch of texts through the pipeline.

        Args:
            texts: List of raw text strings.

        Returns:
            List of ProcessedText results.
        """
        start_time = time.perf_counter()
        results = [self.process(t) for t in texts]
        total_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Batch preprocessing complete: %d texts in %.1fms (%.2fms/text)",
            len(texts),
            total_time,
            total_time / max(len(texts), 1),
        )
        return results

    def process_to_text(self, text: str) -> str:
        """
        Convenience method: process text and return only the final string.
        Used when you just need the cleaned text for model input.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text string.
        """
        return self.process(text).final_text

    def process_batch_to_texts(self, texts: list[str]) -> list[str]:
        """
        Convenience method: process batch and return only final strings.

        Args:
            texts: List of raw text strings.

        Returns:
            List of preprocessed text strings.
        """
        return [r.final_text for r in self.process_batch(texts)]

    @staticmethod
    def _extract_metadata(text: str) -> dict:
        """
        Extract statistical metadata from original text.
        These serve as additional features for the classifier.

        Features extracted:
        - has_url: Whether text contains URLs
        - has_email: Whether text contains email addresses
        - has_phone: Whether text contains phone numbers
        - has_currency: Whether text mentions currency
        - uppercase_ratio: Ratio of uppercase characters
        - digit_ratio: Ratio of digit characters
        - special_char_ratio: Ratio of special characters
        """
        import re
        from ml.preprocessing.cleaner import (
            CURRENCY_PATTERN,
            EMAIL_PATTERN,
            PHONE_PATTERN,
            URL_PATTERN,
        )

        alpha_chars = [c for c in text if c.isalpha()]
        total_chars = max(len(text), 1)
        alpha_count = max(len(alpha_chars), 1)

        return {
            "has_url": bool(URL_PATTERN.search(text)),
            "has_email": bool(EMAIL_PATTERN.search(text)),
            "has_phone": bool(PHONE_PATTERN.search(text)),
            "has_currency": bool(CURRENCY_PATTERN.search(text)),
            "uppercase_ratio": sum(1 for c in alpha_chars if c.isupper()) / alpha_count,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / total_chars,
            "special_char_ratio": sum(
                1 for c in text if not c.isalnum() and not c.isspace()
            ) / total_chars,
        }

    def __repr__(self) -> str:
        return (
            f"PreprocessingPipeline("
            f"cleaning={self.config.enable_cleaning}, "
            f"tokenizer={self.config.tokenizer_backend}, "
            f"normalization={self.config.enable_normalization})"
        )
