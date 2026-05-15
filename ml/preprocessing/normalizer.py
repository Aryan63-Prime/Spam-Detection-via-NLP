"""
Text Normalizer Module
=======================
Handles linguistic normalization: stopword removal, lemmatization,
stemming, slang normalization, and Hinglish text handling.

Architecture Decision:
- Modular normalizers that can be composed in any order
- Extensible slang/Hinglish dictionaries (loaded from JSON)
- Configurable via NormalizerConfig dataclass
- Supports multilingual text processing

Why Lemmatization over Stemming:
- Lemmatization produces actual words ("running" → "run")
- Stemming can produce non-words ("running" → "runn")
- We support both; lemmatization is the default
- Stemming is ~5x faster, useful for high-throughput pipelines
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Set

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Cached Resources (Lazy-loaded)
# ──────────────────────────────────────────────
_lemmatizer = None
_stemmer = None
_stopwords: Optional[Set[str]] = None


def _get_lemmatizer():
    """Lazy-load WordNet lemmatizer."""
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        for pkg in ["wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
        logger.info("WordNet lemmatizer loaded.")
    return _lemmatizer


def _get_stemmer():
    """Lazy-load Porter stemmer."""
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
        logger.info("Porter stemmer loaded.")
    return _stemmer


def _get_stopwords(language: str = "english") -> Set[str]:
    """Lazy-load NLTK stopwords."""
    global _stopwords
    if _stopwords is None:
        import nltk
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as sw
        _stopwords = set(sw.words(language))
        logger.info("Loaded %d stopwords for '%s'.", len(_stopwords), language)
    return _stopwords


# ──────────────────────────────────────────────
# Slang / Abbreviation Dictionary
# ──────────────────────────────────────────────
# Common SMS/internet slang used in spam messages.
# This is extensible — in production, load from a JSON config file.

SLANG_DICT: dict[str, str] = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "k": "okay",
    "ok": "okay",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "thnx": "thanks",
    "txt": "text",
    "msg": "message",
    "msgs": "messages",
    "b4": "before",
    "2day": "today",
    "2moro": "tomorrow",
    "2nite": "tonight",
    "4u": "for you",
    "gr8": "great",
    "l8r": "later",
    "l8": "late",
    "w8": "wait",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "brb": "be right back",
    "btw": "by the way",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "idk": "i don't know",
    "fyi": "for your information",
    "asap": "as soon as possible",
    "lmk": "let me know",
    "nvm": "never mind",
    "smh": "shaking my head",
    "gg": "good game",
    "fomo": "fear of missing out",
    "dm": "direct message",
    "rn": "right now",
    "fr": "for real",
    "ngl": "not gonna lie",
    "w/": "with",
    "w/o": "without",
    "b/c": "because",
    "bc": "because",
    "cuz": "because",
    "coz": "because",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "ya": "you",
    "yall": "you all",
    "aint": "is not",
    "dont": "do not",
    "doesnt": "does not",
    "cant": "cannot",
    "wont": "will not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "couldnt": "could not",
    "didnt": "did not",
    "hadnt": "had not",
    "hasnt": "has not",
    "havent": "have not",
    "isnt": "is not",
    "wasnt": "was not",
    "werent": "were not",
}

# ──────────────────────────────────────────────
# Hinglish (Hindi-English) Dictionary
# ──────────────────────────────────────────────
# Common Hinglish words found in spam messages (India market).
# Critical for detecting spam targeting Hindi-speaking audiences.

HINGLISH_DICT: dict[str, str] = {
    "paisa": "money",
    "paise": "money",
    "rupaye": "rupees",
    "rupya": "rupee",
    "jeetiye": "win",
    "jeeto": "win",
    "jeet": "win",
    "inam": "prize",
    "inaam": "prize",
    "muft": "free",
    "mushkil": "difficult",
    "badhai": "congratulations",
    "badhiya": "excellent",
    "suniye": "listen",
    "abhi": "now",
    "turant": "immediately",
    "jaldi": "quickly",
    "call": "call",
    "karo": "do",
    "kare": "do",
    "karein": "do",
    "kariye": "do",
    "dijiye": "give",
    "bhejiye": "send",
    "bhejo": "send",
    "dhamaka": "blast",
    "offer": "offer",
    "mauka": "chance",
    "avsar": "opportunity",
    "sasta": "cheap",
    "mehnga": "expensive",
    "aaj": "today",
    "kal": "tomorrow",
    "abhi": "now",
    "yahan": "here",
    "wahan": "there",
    "kab": "when",
    "kaise": "how",
    "kyon": "why",
    "kyunki": "because",
    "lekin": "but",
    "aur": "and",
    "ya": "or",
    "nahi": "no",
    "haan": "yes",
    "accha": "good",
    "bura": "bad",
    "bahut": "very",
    "zyada": "more",
    "kam": "less",
    "sab": "all",
    "kuch": "some",
    "matlab": "meaning",
    "samjho": "understand",
    "dekho": "look",
    "batao": "tell",
    "chalo": "let's go",
    "ruko": "wait",
    "suno": "listen",
}

# Spam-specific keywords for feature enhancement
SPAM_KEYWORDS: set[str] = {
    "free", "winner", "prize", "congratulations", "claim",
    "urgent", "limited", "offer", "discount", "deal",
    "cash", "money", "credit", "loan", "debt",
    "click", "subscribe", "unsubscribe", "opt-in",
    "guarantee", "risk-free", "no-obligation",
    "act now", "don't miss", "expires", "today only",
    "earn", "income", "profit", "investment",
    "viagra", "pharmacy", "weight loss", "diet",
    "nigeria", "prince", "inheritance", "transfer",
    "password", "verify", "account", "suspended",
    "lottery", "jackpot", "selected", "chosen",
}


@dataclass
class NormalizerConfig:
    """Configuration for text normalization operations."""
    remove_stopwords: bool = True
    stopword_language: str = "english"
    custom_stopwords: set[str] = field(default_factory=set)
    preserve_words: set[str] = field(
        default_factory=lambda: {"not", "no", "nor", "never", "neither", "nobody", "nothing"}
    )
    lemmatize: bool = True
    stem: bool = False  # Use lemmatization by default
    normalize_slang: bool = True
    normalize_hinglish: bool = True
    min_word_length: int = 2
    max_word_length: int = 50


class TextNormalizer:
    """
    Production-grade text normalizer.
    
    Handles stopword removal, lemmatization, slang normalization,
    and Hinglish text processing.

    Usage:
        normalizer = TextNormalizer()
        tokens = ["u", "won", "free", "prizes", "!!!"]
        normalized = normalizer.normalize(tokens)
        # Output: ["you", "won", "free", "prize"]
    """

    def __init__(self, config: Optional[NormalizerConfig] = None) -> None:
        self.config = config or NormalizerConfig()

        # Build stopword set
        self._stopwords: Set[str] = set()
        if self.config.remove_stopwords:
            self._stopwords = _get_stopwords(self.config.stopword_language)
            self._stopwords = self._stopwords.union(self.config.custom_stopwords)
            # Never remove negation words (critical for sentiment/spam)
            self._stopwords -= self.config.preserve_words

        # Build combined slang dictionary
        self._slang_dict: dict[str, str] = {}
        if self.config.normalize_slang:
            self._slang_dict.update(SLANG_DICT)
        if self.config.normalize_hinglish:
            self._slang_dict.update(HINGLISH_DICT)

        logger.info(
            "TextNormalizer initialized: stopwords=%d, slang_entries=%d, "
            "lemmatize=%s, stem=%s",
            len(self._stopwords),
            len(self._slang_dict),
            self.config.lemmatize,
            self.config.stem,
        )

    def normalize(self, tokens: list[str]) -> list[str]:
        """
        Normalize a list of tokens.

        Pipeline:
        1. Slang/Hinglish normalization
        2. Stopword removal
        3. Lemmatization or stemming
        4. Length filtering

        Args:
            tokens: List of word tokens (already tokenized).

        Returns:
            List of normalized tokens.
        """
        result = []
        for token in tokens:
            # Skip empty tokens
            if not token:
                continue

            word = token.lower().strip()

            # Step 1: Slang/Hinglish normalization
            if word in self._slang_dict:
                word = self._slang_dict[word]

            # Step 2: Stopword removal
            if self.config.remove_stopwords and word in self._stopwords:
                continue

            # Step 3: Length filtering
            if len(word) < self.config.min_word_length:
                continue
            if len(word) > self.config.max_word_length:
                continue

            # Step 4: Lemmatization or stemming
            if self.config.lemmatize:
                word = _get_lemmatizer().lemmatize(word, pos="v")
                word = _get_lemmatizer().lemmatize(word, pos="n")
            elif self.config.stem:
                word = _get_stemmer().stem(word)

            result.append(word)

        return result

    def normalize_batch(self, token_lists: list[list[str]]) -> list[list[str]]:
        """Normalize a batch of token lists."""
        return [self.normalize(tokens) for tokens in token_lists]

    def extract_spam_keywords(self, tokens: list[str]) -> list[str]:
        """
        Extract known spam keywords from token list.
        Useful as additional features for the classifier.

        Args:
            tokens: List of word tokens.

        Returns:
            List of spam keywords found in the text.
        """
        return [t for t in tokens if t.lower() in SPAM_KEYWORDS]

    def get_spam_keyword_count(self, tokens: list[str]) -> int:
        """Count the number of spam keywords in a token list."""
        return len(self.extract_spam_keywords(tokens))

    def __repr__(self) -> str:
        return (
            f"TextNormalizer(stopwords={len(self._stopwords)}, "
            f"slang={len(self._slang_dict)}, "
            f"lemmatize={self.config.lemmatize})"
        )
