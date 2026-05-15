"""
Explainable AI Module
======================
Implements SHAP and LIME for model interpretability.
Users should understand WHY a message was classified as spam.

Architecture Decision:
- Unified ExplainabilityEngine wrapping SHAP and LIME
- Returns structured ExplanationResult for frontend rendering
- Supports both traditional ML and transformer models
- Caches explanations for performance

Why XAI for Spam Detection:
- Users need trust in AI decisions (false positives are costly)
- Compliance requirements (GDPR right to explanation)
- Debugging: understand model failures
- Feature discovery: find new spam patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WordImportance:
    """A single word with its importance score."""
    word: str
    score: float
    is_spam_indicator: bool = False  # True if word pushes toward spam


@dataclass
class ExplanationResult:
    """
    Structured explanation result for a single prediction.
    Designed to be directly consumed by the frontend.
    """
    text: str
    prediction: str  # "spam" or "ham"
    confidence: float
    method: str  # "shap" or "lime"
    word_importances: list[WordImportance] = field(default_factory=list)
    top_spam_words: list[str] = field(default_factory=list)
    top_ham_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "method": self.method,
            "word_importances": [
                {"word": wi.word, "score": wi.score, "is_spam_indicator": wi.is_spam_indicator}
                for wi in self.word_importances
            ],
            "top_spam_words": self.top_spam_words,
            "top_ham_words": self.top_ham_words,
        }


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations).

    How LIME works:
    1. Perturb the input text by randomly removing words
    2. Get predictions for all perturbations
    3. Fit a simple linear model on perturbations vs predictions
    4. Linear model coefficients = word importance

    Pros: Model-agnostic, intuitive, works with any classifier
    Cons: Slow (requires many prediction calls), stochastic
    """

    def __init__(self, class_names: list[str] = None) -> None:
        self.class_names = class_names or ["ham", "spam"]

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 15,
        num_samples: int = 500,
    ) -> ExplanationResult:
        """
        Generate LIME explanation for a single text.

        Args:
            text: Input text to explain.
            predict_fn: Function that takes list[str] → np.ndarray of probabilities.
            num_features: Number of top features to return.
            num_samples: Number of perturbation samples.

        Returns:
            ExplanationResult with word importances.
        """
        try:
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=self.class_names)

            explanation = explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

            # Get prediction
            proba = predict_fn([text])[0]
            pred_label = "spam" if proba[1] > proba[0] else "ham"
            confidence = float(max(proba))

            # Extract word importances
            word_importances = []
            top_spam = []
            top_ham = []

            for word, score in explanation.as_list():
                is_spam = score > 0
                word_importances.append(WordImportance(
                    word=word,
                    score=float(score),
                    is_spam_indicator=is_spam,
                ))
                if is_spam:
                    top_spam.append(word)
                else:
                    top_ham.append(word)

            return ExplanationResult(
                text=text,
                prediction=pred_label,
                confidence=confidence,
                method="lime",
                word_importances=word_importances,
                top_spam_words=top_spam[:5],
                top_ham_words=top_ham[:5],
            )

        except ImportError:
            logger.warning("LIME not installed. Install with: pip install lime")
            return ExplanationResult(
                text=text, prediction="unknown", confidence=0.0, method="lime"
            )
        except Exception as e:
            logger.error("LIME explanation failed: %s", e)
            return ExplanationResult(
                text=text, prediction="error", confidence=0.0, method="lime"
            )


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations).

    Based on game theory: each feature's contribution is calculated
    as a Shapley value — the average marginal contribution.

    Pros: Theoretically grounded, consistent, global + local explanations
    Cons: Computationally expensive for large models
    """

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        max_evals: int = 500,
    ) -> ExplanationResult:
        """Generate SHAP explanation for a single text."""
        try:
            import shap

            # Use the text masker for SHAP
            masker = shap.maskers.Text(tokenizer=r"\W+")
            explainer = shap.Explainer(predict_fn, masker, output_names=["ham", "spam"])

            shap_values = explainer([text], max_evals=max_evals)

            # Get prediction
            proba = predict_fn([text])[0]
            pred_label = "spam" if proba[1] > proba[0] else "ham"
            confidence = float(max(proba))

            # Extract word importances (for spam class)
            values = shap_values.values[0][:, 1]  # Spam class contributions
            words = shap_values.data[0]

            word_importances = []
            top_spam = []
            top_ham = []

            for word, score in zip(words, values):
                if not word.strip():
                    continue
                is_spam = score > 0
                word_importances.append(WordImportance(
                    word=word.strip(),
                    score=float(score),
                    is_spam_indicator=is_spam,
                ))
                if is_spam:
                    top_spam.append(word.strip())
                else:
                    top_ham.append(word.strip())

            # Sort by absolute importance
            word_importances.sort(key=lambda x: abs(x.score), reverse=True)

            return ExplanationResult(
                text=text,
                prediction=pred_label,
                confidence=confidence,
                method="shap",
                word_importances=word_importances[:15],
                top_spam_words=top_spam[:5],
                top_ham_words=top_ham[:5],
            )

        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return ExplanationResult(
                text=text, prediction="unknown", confidence=0.0, method="shap"
            )
        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            return ExplanationResult(
                text=text, prediction="error", confidence=0.0, method="shap"
            )


class ExplainabilityEngine:
    """
    Unified explainability engine.

    Provides a single interface to get explanations from
    either LIME or SHAP, depending on configuration.

    Usage:
        engine = ExplainabilityEngine()
        result = engine.explain(text, model.predict_proba, method="lime")
        print(result.top_spam_words)
    """

    def __init__(self) -> None:
        self._lime = LIMEExplainer()
        self._shap = SHAPExplainer()

    def explain(
        self,
        text: str,
        predict_fn: Callable,
        method: str = "lime",
        **kwargs,
    ) -> ExplanationResult:
        """
        Generate an explanation for a prediction.

        Args:
            text: Input text.
            predict_fn: Prediction function (list[str] → np.ndarray).
            method: "lime" or "shap".

        Returns:
            ExplanationResult with word-level importance scores.
        """
        if method == "lime":
            return self._lime.explain(text, predict_fn, **kwargs)
        elif method == "shap":
            return self._shap.explain(text, predict_fn, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lime' or 'shap'.")
