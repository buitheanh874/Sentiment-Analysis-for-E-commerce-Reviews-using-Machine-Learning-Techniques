import re
import string
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# ----------- Lexicons & defaults -----------
NEGATION_CUES: List[str] = [
    "not",
    "no",
    "never",
    "n't",
    "cannot",
    "can't",
    "dont",
    "don't",
    "didn't",
    "didnt",
    "won't",
    "wont",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "shouldn't",
    "shouldnt",
    "couldn't",
    "couldnt",
    "wouldn't",
    "wouldnt",
]

CONTRAST_MARKERS: List[str] = ["but", "however", "although", "though", "yet"]

DEFAULT_ABBREV_MAP: Dict[str, str] = {
    "gr8": "great",
    "thx": "thanks",
    "u": "you",
    "idk": "i do not know",
    "imo": "in my opinion",
    "w/": "with",
    "w/o": "without",
    "cant": "cannot",
    "can't": "cannot",
    "cant.": "cannot",
    "didnt": "did not",
    "didn't": "did not",
}

DEFAULT_NEGATION_WINDOW = 3


@dataclass(frozen=True)
class VariantSpec:
    """Configuration for a context-aware feature variant."""

    name: str
    word_ngram: Tuple[int, int] = (1, 2)
    use_negation: bool = False
    use_clause: bool = False
    use_char: bool = False
    use_lexicon: bool = False  # NEW: AFINN sentiment lexicon features
    description: str = ""


# Fixed set of variants used in step06b (keeps baseline V0 intact)
CONTEXT_VARIANTS: List[VariantSpec] = [
    VariantSpec("V0", word_ngram=(1, 2), description="Baseline word 1-2"),
    VariantSpec("V1", word_ngram=(1, 3), description="Word 1-3"),
    VariantSpec("V2", word_ngram=(1, 2), use_negation=True, description="Word 1-2 + negation tagging"),
    VariantSpec("V3", word_ngram=(1, 2), use_clause=True, description="Word 1-2 + clause split after contrast"),
    VariantSpec(
        "V4",
        word_ngram=(1, 2),
        use_negation=True,
        use_clause=True,
        description="Word 1-2 + negation + clause split",
    ),
    VariantSpec(
        "V5",
        word_ngram=(1, 2),
        use_char=True,
        description="Word 1-2 + char 3-5",
    ),
    VariantSpec(
        "V6",
        word_ngram=(1, 2),
        use_negation=True,
        use_char=True,
        description="Word 1-2 + char 3-5 + negation",
    ),
    # V7: TF-IDF + AFINN lexicon features (4 columns: raw, negated, intensified, contrast)
    VariantSpec(
        "V7",
        word_ngram=(1, 2),
        use_negation=True,
        use_lexicon=True,
        description="Word 1-2 + negation + AFINN lexicon (4 features)",
    ),
]


# ----------- Text normalization helpers -----------
def normalize_abbrev(text: str, mapping: Dict[str, str]) -> str:
    """Expand common chat/typo abbreviations in a case-insensitive way."""
    if not mapping:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in mapping.keys()) + r")\b", re.IGNORECASE
    )

    def _replace(match: re.Match) -> str:
        key = match.group(0).lower()
        return mapping.get(key, key)

    return pattern.sub(_replace, text)


_PUNCT_NO_UNDERSCORE = re.sub("_", "", re.escape(string.punctuation))
_URL_PATTERN = re.compile(r"http\S+|www\.\S+")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[.!?,;:]")


def apply_negation_tagging(text: str, window: int = DEFAULT_NEGATION_WINDOW) -> str:
    """
    Prefix tokens within a small window after a negation cue.
    Stops at punctuation boundaries to keep scope short.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    output_tokens: List[str] = []
    countdown = 0
    for tok in tokens:
        if tok in ".!?,;:":
            countdown = 0
            continue
        if tok in NEGATION_CUES:
            countdown = window
            output_tokens.append(tok)
            continue
        if countdown > 0:
            output_tokens.append(f"NOT_{tok}")
            countdown -= 1
        else:
            output_tokens.append(tok)
    return " ".join(output_tokens)


def clean_text(
    text: str,
    enable_abbrev_norm: bool = False,
    enable_negation: bool = False,
    negation_window: int = DEFAULT_NEGATION_WINDOW,
    abbrev_map: Dict[str, str] = DEFAULT_ABBREV_MAP,
) -> str:
    """
    Lowercase, optional abbreviation expansion, optional negation tagging,
    strip URLs/punctuation, whitespace normalize.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    if enable_abbrev_norm:
        text = normalize_abbrev(text, abbrev_map)
    if enable_negation:
        text = apply_negation_tagging(text, window=negation_window)
    # Remove punctuation but keep underscores so NOT_ tokens survive
    text = re.sub(f"[{_PUNCT_NO_UNDERSCORE}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------- Contrast handling -----------
def split_at_contrast_marker(text: str) -> Tuple[str, str]:
    """
    Split cleaned text at the first contrast marker.
    Returns (left_clause, right_clause). If none found, right is empty.
    """
    tokens = text.split()
    for idx, tok in enumerate(tokens):
        if tok in CONTRAST_MARKERS:
            left = " ".join(tokens[:idx]).strip()
            right = " ".join(tokens[idx + 1 :]).strip()
            return left, right
    return text.strip(), ""


def contrast_flags(text: str) -> List[int]:
    """Binary presence flags for each contrast marker."""
    tokens = set(text.split())
    return [1 if marker in tokens else 0 for marker in CONTRAST_MARKERS]


# ----------- Transformers for sklearn pipelines -----------
def _prep_texts(
    texts: Sequence[str],
    enable_abbrev_norm: bool,
    enable_negation: bool,
    negation_window: int,
) -> List[str]:
    return [
        clean_text(
            t,
            enable_abbrev_norm=enable_abbrev_norm,
            enable_negation=enable_negation,
            negation_window=negation_window,
        )
        for t in texts
    ]


def _prep_left(texts: Sequence[str], **kwargs) -> List[str]:
    cleaned = _prep_texts(texts, **kwargs)
    return [split_at_contrast_marker(t)[0] for t in cleaned]


def _prep_right(texts: Sequence[str], **kwargs) -> List[str]:
    cleaned = _prep_texts(texts, **kwargs)
    return [split_at_contrast_marker(t)[1] for t in cleaned]


def _prep_flags(texts: Sequence[str], **kwargs) -> csr_matrix:
    cleaned = _prep_texts(texts, **kwargs)
    arr = np.array([contrast_flags(t) for t in cleaned], dtype=np.int32)
    return csr_matrix(arr)


def build_word_vectorizer(ngram_range: Tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.9,
        max_features=50000,
        lowercase=False,  # already cleaned
        preprocessor=None,
        tokenizer=str.split,
    )


def build_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        lowercase=False,
    )


def build_vectorizer_from_spec(
    spec: VariantSpec,
    enable_abbrev_norm: bool = False,
    negation_window: int = DEFAULT_NEGATION_WINDOW,
) -> FeatureUnion:
    """
    Build a FeatureUnion transformer that maps raw text -> sparse features
    according to the requested variant spec.
    """
    kw_args = {
        "enable_abbrev_norm": enable_abbrev_norm,
        "enable_negation": spec.use_negation,
        "negation_window": negation_window,
    }

    transformers = []

    if spec.use_clause:
        left_pipe = Pipeline(
            steps=[
                (
                    "left",
                    FunctionTransformer(
                        _prep_left, kw_args=kw_args, validate=False
                    ),
                ),
                ("tfidf_left", build_word_vectorizer(spec.word_ngram)),
            ]
        )
        right_pipe = Pipeline(
            steps=[
                (
                    "right",
                    FunctionTransformer(
                        _prep_right, kw_args=kw_args, validate=False
                    ),
                ),
                ("tfidf_right", build_word_vectorizer(spec.word_ngram)),
            ]
        )
        flag_pipe = FunctionTransformer(_prep_flags, kw_args=kw_args, validate=False)
        transformers.extend(
            [
                ("left_clause", left_pipe),
                ("right_clause", right_pipe),
                ("contrast_flags", flag_pipe),
            ]
        )
    else:
        transformers.append(
            (
                "word_tfidf",
                Pipeline(
                    steps=[
                        (
                            "clean",
                            FunctionTransformer(
                                _prep_texts, kw_args=kw_args, validate=False
                            ),
                        ),
                        ("tfidf", build_word_vectorizer(spec.word_ngram)),
                    ]
                ),
            )
        )

    if spec.use_char:
        transformers.append(
            (
                "char_tfidf",
                Pipeline(
                    steps=[
                        (
                            "clean",
                            FunctionTransformer(
                                _prep_texts, kw_args=kw_args, validate=False
                            ),
                        ),
                        ("tfidf_char", build_char_vectorizer()),
                    ]
                ),
            )
        )

    # NEW: Add AFINN lexicon features (4 separate columns)
    if spec.use_lexicon:
        from src.sentiment_lexicon import SentimentFeatureTransformer
        transformers.append(("sentiment_lexicon", SentimentFeatureTransformer()))

    return FeatureUnion(transformer_list=transformers)


# ----------- Small unit-like sanity checks -----------
def negation_sanity_tests(window: int = DEFAULT_NEGATION_WINDOW) -> List[Tuple[str, str]]:
    """
    Return a list of (input, output) pairs for quick verification.
    Used by step02 to log deterministic examples.
    """
    examples = [
        "not bad at all",
        "this is not good",
        "never really helpful",
        "good but not great",
        "can't recommend",
        "no issues whatsoever",
        "not good, not bad",
    ]
    return [(ex, clean_text(ex, enable_negation=True, negation_window=window)) for ex in examples]


__all__ = [
    "CONTEXT_VARIANTS",
    "VariantSpec",
    "NEGATION_CUES",
    "CONTRAST_MARKERS",
    "DEFAULT_ABBREV_MAP",
    "DEFAULT_NEGATION_WINDOW",
    "clean_text",
    "normalize_abbrev",
    "apply_negation_tagging",
    "split_at_contrast_marker",
    "contrast_flags",
    "build_vectorizer_from_spec",
    "build_word_vectorizer",
    "build_char_vectorizer",
    "negation_sanity_tests",
]
