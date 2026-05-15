"""
Data Augmentation Module
=========================
Implements text augmentation techniques to address class imbalance
in spam detection datasets.

Why Augmentation:
- SMS Spam Collection has ~13% spam — severely imbalanced
- Class imbalance leads to high ham accuracy but poor spam recall
- Augmentation creates synthetic spam examples
- Combined with class weights for robust training

Techniques:
1. Synonym Replacement — Replace words with WordNet synonyms
2. Random Insertion — Insert random synonyms into text
3. Random Swap — Swap positions of two random words
4. Random Deletion — Randomly remove words
5. Back-Translation Placeholder — For production, use external API

Architecture:
- Each technique is a standalone function (composable)
- AugmentationPipeline orchestrates multiple techniques
- Configurable augmentation factor per class
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Cached WordNet (Lazy-loaded)
# ──────────────────────────────────────────────
_wordnet = None


def _get_wordnet():
    """Lazy-load WordNet for synonym lookup."""
    global _wordnet
    if _wordnet is None:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet
        _wordnet = wordnet
    return _wordnet


def _get_synonyms(word: str) -> list[str]:
    """Get synonyms for a word from WordNet."""
    wn = _get_wordnet()
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


# ──────────────────────────────────────────────
# Augmentation Techniques
# ──────────────────────────────────────────────

def synonym_replacement(text: str, n: int = 2) -> str:
    """
    Replace n random words with their synonyms.

    Why: Creates semantically similar text with different surface forms.
    This helps the model learn that spam intent matters, not exact words.
    """
    words = text.split()
    if len(words) < 2:
        return text

    # Find words that have synonyms
    candidates = [(i, w) for i, w in enumerate(words) if len(w) > 3]
    random.shuffle(candidates)

    replacements = 0
    for idx, word in candidates:
        if replacements >= n:
            break
        syns = _get_synonyms(word)
        if syns:
            words[idx] = random.choice(syns)
            replacements += 1

    return " ".join(words)


def random_insertion(text: str, n: int = 1) -> str:
    """
    Insert n random synonyms of existing words at random positions.

    Why: Adds noise while preserving semantic content.
    Helps model be robust to extra words in messages.
    """
    words = text.split()
    if not words:
        return text

    for _ in range(n):
        # Pick a random word to find synonym for
        random_word = random.choice(words)
        syns = _get_synonyms(random_word)
        if syns:
            synonym = random.choice(syns)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, synonym)

    return " ".join(words)


def random_swap(text: str, n: int = 1) -> str:
    """
    Swap the positions of n pairs of random words.

    Why: Tests model robustness to word order changes.
    Important since spam detection shouldn't rely on exact order.
    """
    words = text.split()
    if len(words) < 2:
        return text

    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)


def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly delete words with probability p.

    Why: Creates shorter variants. Spam often has key phrases that
    remain detectable even with missing context words.
    """
    words = text.split()
    if len(words) <= 1:
        return text

    remaining = [w for w in words if random.random() > p]

    # Ensure at least one word remains
    if not remaining:
        return random.choice(words)

    return " ".join(remaining)


def char_level_noise(text: str, p: float = 0.05) -> str:
    """
    Add character-level noise (typos, case changes).

    Why: Spammers often use character obfuscation ("FR33", "w1nner").
    This helps the model handle obfuscated text.
    """
    result = []
    for char in text:
        if random.random() < p:
            action = random.choice(["swap_case", "duplicate", "skip"])
            if action == "swap_case" and char.isalpha():
                result.append(char.swapcase())
            elif action == "duplicate":
                result.append(char)
                result.append(char)
            elif action == "skip":
                continue  # Delete the character
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


# ──────────────────────────────────────────────
# Augmentation Pipeline
# ──────────────────────────────────────────────

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    techniques: list[str] = field(
        default_factory=lambda: ["synonym_replacement", "random_swap", "random_deletion"]
    )
    augment_minority_class: bool = True
    minority_label: int = 1  # spam
    augmentation_factor: int = 3  # Generate 3x synthetic samples per minority sample
    max_augmented_samples: int = 5000
    random_seed: int = 42


class AugmentationPipeline:
    """
    Orchestrates text augmentation to balance the dataset.

    Usage:
        pipeline = AugmentationPipeline()
        augmented_df = pipeline.augment(df)
    """

    TECHNIQUE_MAP = {
        "synonym_replacement": synonym_replacement,
        "random_insertion": random_insertion,
        "random_swap": random_swap,
        "random_deletion": random_deletion,
        "char_noise": char_level_noise,
    }

    def __init__(self, config: Optional[AugmentationConfig] = None) -> None:
        self.config = config or AugmentationConfig()
        random.seed(self.config.random_seed)

        # Validate techniques
        for tech in self.config.techniques:
            if tech not in self.TECHNIQUE_MAP:
                raise ValueError(
                    f"Unknown augmentation technique: '{tech}'. "
                    f"Supported: {list(self.TECHNIQUE_MAP.keys())}"
                )

        logger.info(
            "AugmentationPipeline initialized: techniques=%s, factor=%d",
            self.config.techniques,
            self.config.augmentation_factor,
        )

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the dataset to balance class distribution.

        Args:
            df: DataFrame with 'text', 'label', and 'label_encoded' columns.

        Returns:
            Augmented DataFrame with synthetic samples appended.
        """
        if not self.config.augment_minority_class:
            return df

        # Get minority class samples
        minority_mask = df["label_encoded"] == self.config.minority_label
        minority_df = df[minority_mask]
        majority_count = (~minority_mask).sum()
        minority_count = minority_mask.sum()

        logger.info(
            "Class distribution before augmentation: "
            "majority=%d, minority=%d (ratio=%.2f)",
            majority_count, minority_count,
            minority_count / max(majority_count, 1),
        )

        # Calculate how many synthetic samples to create
        target_augmented = min(
            minority_count * self.config.augmentation_factor,
            self.config.max_augmented_samples,
        )

        # Generate synthetic samples
        synthetic_texts = []
        synthetic_labels = []
        synthetic_encoded = []

        for _ in range(target_augmented):
            # Pick a random minority sample
            sample = minority_df.sample(n=1).iloc[0]
            text = sample["text"]

            # Apply a random augmentation technique
            technique_name = random.choice(self.config.techniques)
            technique_fn = self.TECHNIQUE_MAP[technique_name]

            try:
                augmented_text = technique_fn(text)
                if augmented_text and augmented_text != text:
                    synthetic_texts.append(augmented_text)
                    synthetic_labels.append(sample["label"])
                    synthetic_encoded.append(sample["label_encoded"])
            except Exception as e:
                logger.debug("Augmentation failed for text: %s", e)
                continue

        # Create augmented DataFrame
        if synthetic_texts:
            augmented_df = pd.DataFrame({
                "text": synthetic_texts,
                "label": synthetic_labels,
                "label_encoded": synthetic_encoded,
            })

            result = pd.concat([df, augmented_df], ignore_index=True)

            logger.info(
                "Augmentation complete: +%d synthetic samples. "
                "New total: %d (spam=%d, ham=%d)",
                len(synthetic_texts),
                len(result),
                (result["label_encoded"] == 1).sum(),
                (result["label_encoded"] == 0).sum(),
            )
            return result

        logger.warning("No synthetic samples generated.")
        return df

    def augment_text(self, text: str, num_variants: int = 3) -> list[str]:
        """
        Generate multiple augmented variants of a single text.

        Args:
            text: Input text to augment.
            num_variants: Number of variants to generate.

        Returns:
            List of augmented text variants.
        """
        variants = []
        for _ in range(num_variants):
            tech_name = random.choice(self.config.techniques)
            tech_fn = self.TECHNIQUE_MAP[tech_name]
            try:
                variant = tech_fn(text)
                if variant and variant != text:
                    variants.append(variant)
            except Exception:
                continue
        return variants
