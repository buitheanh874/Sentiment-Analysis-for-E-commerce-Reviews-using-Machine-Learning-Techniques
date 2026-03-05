# -*- coding: utf-8 -*-
"""
Sentiment Lexicon Features for NLP Project
==============================================
AFINN-111 embedded lexicon with negation, intensifier handling.
Features are extracted as 4 separate columns (not injected into TF-IDF).

NLP Techniques:
- Lexicon-based sentiment scoring
- Negation scope detection with polarity flip
- Intensifier/diminisher multipliers  
- Contrast clause weighting
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


# ============== AFINN-111 Lexicon (Partial - Most Common Sentiment Words) ==============
# Full AFINN has ~2500 words; we embed the most impactful ones for gift card reviews
AFINN_LEXICON: Dict[str, int] = {
    # Strong Positive (+4, +5)
    "outstanding": 5, "superb": 5, "excellent": 3, "amazing": 4, "fantastic": 4,
    "wonderful": 4, "awesome": 4, "brilliant": 4, "perfect": 3, "love": 3,
    "loved": 3, "loving": 2, "best": 3, "great": 3, "happy": 3,
    "thrilled": 5, "delighted": 3, "excited": 3, "satisfied": 2,
    
    # Moderate Positive (+1, +2, +3)
    "good": 3, "nice": 3, "fine": 2, "pleased": 3, "glad": 3,
    "enjoy": 2, "enjoyed": 2, "like": 2, "liked": 2, "recommend": 2,
    "recommended": 2, "helpful": 2, "useful": 2, "easy": 1, "fast": 1,
    "quick": 1, "smooth": 1, "simple": 1, "convenient": 2, "reliable": 2,
    "thanks": 2, "thank": 2, "appreciate": 2, "worth": 2, "valuable": 2,
    "beautiful": 3, "lovely": 3, "impressed": 3, "impressive": 3,
    
    # Weak/Neutral
    "ok": 1, "okay": 1, "decent": 1, "average": 0, "fair": 1,
    "alright": 1, "acceptable": 1, "adequate": 1, "meh": 0,
    
    # Moderate Negative (-1, -2, -3)
    "bad": -3, "poor": -2, "disappointing": -2, "disappointed": -2,
    "unfortunate": -2, "unhappy": -2, "upset": -2, "frustrated": -2,
    "frustrating": -2, "annoying": -2, "annoyed": -2, "difficult": -1,
    "hard": -1, "slow": -1, "late": -1, "delayed": -1, "wait": -1,
    "waiting": -1, "wrong": -2, "broken": -2, "failed": -2, "fail": -2,
    "issue": -1, "issues": -1, "problem": -2, "problems": -2,
    "trouble": -2, "error": -2, "confusing": -2, "confused": -2,
    "useless": -2, "waste": -2, "wasted": -2, "regret": -2,
    "dislike": -2, "hate": -3, "hated": -3, "boring": -2,
    "mediocre": -1, "lackluster": -2, "lacking": -1, "missing": -1,
    "cheap": -2, "flimsy": -2, "defective": -3, "faulty": -2,
    
    # Strong Negative (-4, -5)
    "terrible": -4, "horrible": -4, "awful": -4, "worst": -4,
    "disgusting": -4, "dreadful": -4, "pathetic": -3, "ridiculous": -2,
    "scam": -5, "fraud": -5, "fake": -4, "stolen": -5, "steal": -5,
    "angry": -3, "furious": -4, "outraged": -4, "nightmare": -4,
    "atrocious": -5, "abysmal": -5, "horrendous": -4, "appalling": -4,
    
    # Gift card / e-commerce specific
    "redeem": 1, "redeemed": 1, "balance": 0, "gift": 1, "card": 0,
    "delivery": 0, "delivered": 1, "arrived": 1, "received": 1,
    "works": 2, "worked": 2, "working": 1, "shipping": 0, "shipped": 1,
    "refund": -1, "refunded": 0, "return": -1, "returned": 0,
    "instant": 2, "immediately": 2, "promptly": 2, "quickly": 1,
    
    # Colloquial/informal
    "sucks": -3, "suck": -3, "sucked": -3, "crap": -3, "crappy": -3,
    "lame": -2, "rubbish": -3, "trash": -3, "junk": -3,
    "legit": 2, "solid": 2, "dope": 3, "clutch": 3,
}

# ============== Intensifiers & Diminishers ==============
INTENSIFIERS: Dict[str, float] = {
    "very": 1.5,
    "really": 1.5,
    "extremely": 2.0,
    "absolutely": 2.0,
    "completely": 1.8,
    "totally": 1.8,
    "highly": 1.5,
    "incredibly": 2.0,
    "amazingly": 2.0,
    "exceptionally": 2.0,
    "super": 1.5,
    "so": 1.3,
    "too": 1.3,
    "quite": 1.2,
}

DIMINISHERS: Dict[str, float] = {
    "slightly": 0.5,
    "somewhat": 0.6,
    "barely": 0.3,
    "hardly": 0.3,
    "a bit": 0.5,
    "a little": 0.5,
    "kind of": 0.6,
    "sort of": 0.6,
    "fairly": 0.7,
    "rather": 0.8,
}

# Negation cues (expanded from text_features.py)
NEGATION_CUES: List[str] = [
    "not", "no", "never", "n't", "cannot", "can't", "dont", "don't",
    "didn't", "didnt", "won't", "wont", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent", "shouldn't", "shouldnt",
    "couldn't", "couldnt", "wouldn't", "wouldnt", "neither", "nor",
    "hardly", "barely", "nobody", "nothing", "nowhere", "without",
]

CONTRAST_MARKERS: List[str] = ["but", "however", "although", "though", "yet", "except"]

# Punctuation that ends negation scope
SCOPE_ENDERS = set(".,;:!?")


# ============== Core Functions ==============
def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for sentiment analysis."""
    text = text.lower()
    # Keep apostrophes for contractions
    tokens = re.findall(r"[a-z']+|[.,;:!?]", text)
    return tokens


def compute_afinn_score(text: str) -> float:
    """
    Compute raw AFINN sentiment score (sum of word scores).
    
    Args:
        text: Input text
        
    Returns:
        Sum of sentiment scores for all recognized words
    """
    tokens = _tokenize(text)
    score = 0.0
    for token in tokens:
        if token in AFINN_LEXICON:
            score += AFINN_LEXICON[token]
    return score


def compute_negated_afinn_score(text: str, window: int = 3) -> float:
    """
    Compute AFINN score with negation handling.
    Words within 'window' tokens after a negation cue have their polarity flipped.
    Negation scope stops at punctuation or contrast markers.
    
    Args:
        text: Input text
        window: Number of tokens affected by negation
        
    Returns:
        Sentiment score with negation-aware polarity
    """
    tokens = _tokenize(text)
    score = 0.0
    negation_countdown = 0
    
    for token in tokens:
        # Check scope enders
        if token in SCOPE_ENDERS or token in CONTRAST_MARKERS:
            negation_countdown = 0
            continue
        
        # Check negation cues
        if token in NEGATION_CUES or token.endswith("n't"):
            negation_countdown = window
            continue
        
        # Score word (with potential negation flip)
        if token in AFINN_LEXICON:
            word_score = AFINN_LEXICON[token]
            if negation_countdown > 0:
                word_score = -word_score  # Flip polarity
            score += word_score
        
        # Decrement countdown
        if negation_countdown > 0:
            negation_countdown -= 1
    
    return score


def compute_intensified_score(text: str, window: int = 3) -> float:
    """
    Compute AFINN score with both negation and intensifier/diminisher handling.
    
    Args:
        text: Input text
        window: Negation window size
        
    Returns:
        Score with negation flips and intensity multipliers applied
    """
    tokens = _tokenize(text)
    score = 0.0
    negation_countdown = 0
    intensity_multiplier = 1.0
    
    for i, token in enumerate(tokens):
        # Check scope enders
        if token in SCOPE_ENDERS or token in CONTRAST_MARKERS:
            negation_countdown = 0
            intensity_multiplier = 1.0
            continue
        
        # Check negation cues
        if token in NEGATION_CUES or token.endswith("n't"):
            negation_countdown = window
            continue
        
        # Check intensifiers
        if token in INTENSIFIERS:
            intensity_multiplier = INTENSIFIERS[token]
            continue
        
        # Check diminishers
        if token in DIMINISHERS:
            intensity_multiplier = DIMINISHERS[token]
            continue
        
        # Score word
        if token in AFINN_LEXICON:
            word_score = AFINN_LEXICON[token] * intensity_multiplier
            if negation_countdown > 0:
                word_score = -word_score  # Flip polarity
            score += word_score
            intensity_multiplier = 1.0  # Reset after use
        
        # Decrement countdown
        if negation_countdown > 0:
            negation_countdown -= 1
    
    return score


def compute_contrast_weighted_score(text: str, contrast_weight: float = 2.0) -> float:
    """
    Compute score giving more weight to post-contrast clause.
    "good but late" â†’ score("good") + contrast_weight * score("late")
    
    Args:
        text: Input text
        contrast_weight: Multiplier for post-contrast sentiment
        
    Returns:
        Contrast-weighted sentiment score
    """
    tokens = _tokenize(text)
    pre_contrast_tokens = []
    post_contrast_tokens = []
    found_contrast = False
    
    for token in tokens:
        if token in CONTRAST_MARKERS:
            found_contrast = True
            continue
        if found_contrast:
            post_contrast_tokens.append(token)
        else:
            pre_contrast_tokens.append(token)
    
    # Score each part
    pre_score = sum(AFINN_LEXICON.get(t, 0) for t in pre_contrast_tokens)
    post_score = sum(AFINN_LEXICON.get(t, 0) for t in post_contrast_tokens)
    
    if found_contrast:
        return pre_score + contrast_weight * post_score
    else:
        return pre_score


def extract_sentiment_features(text: str) -> Dict[str, float]:
    """
    Extract all 4 sentiment lexicon features for a single text.
    
    Returns:
        Dict with keys: afinn_raw, afinn_negated, afinn_intensified, afinn_contrast
    """
    return {
        "afinn_raw": compute_afinn_score(text),
        "afinn_negated": compute_negated_afinn_score(text),
        "afinn_intensified": compute_intensified_score(text),
        "afinn_contrast": compute_contrast_weighted_score(text),
    }


# ============== sklearn Transformer ==============
class SentimentFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that extracts 6 NON-NEGATIVE sentiment features.
    
    Chi2 feature selection requires non-negative values, so we split each score
    into positive and negative (absolute) components:
    - pos_raw: max(0, raw_score)
    - neg_raw: abs(min(0, raw_score))
    - pos_negated: max(0, negated_score)
    - neg_negated: abs(min(0, negated_score))
    - pos_intensified: max(0, intensified_score)
    - neg_intensified: abs(min(0, intensified_score))
    
    Usage in FeatureUnion:
        FeatureUnion([
            ("tfidf", TfidfVectorizer(...)),
            ("sentiment", SentimentFeatureTransformer()),
        ])
    """
    
    def __init__(self):
        self.feature_names_ = [
            "sent_pos_raw", "sent_neg_raw",
            "sent_pos_negated", "sent_neg_negated",
            "sent_pos_intensified", "sent_neg_intensified",
        ]
    
    def fit(self, X, y=None):
        """No fitting required - lexicon is fixed."""
        return self
    
    def transform(self, X) -> csr_matrix:
        """
        Transform texts to sentiment feature matrix (6 non-negative columns).
        
        Args:
            X: Iterable of strings (raw or cleaned text)
            
        Returns:
            Sparse matrix of shape (n_samples, 6) with all values >= 0
        """
        rows = []
        for text in X:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            raw = compute_afinn_score(text)
            negated = compute_negated_afinn_score(text)
            intensified = compute_intensified_score(text)
            
            # Split into positive and negative (absolute) components
            rows.append([
                max(0.0, raw),           # pos_raw
                abs(min(0.0, raw)),      # neg_raw
                max(0.0, negated),       # pos_negated
                abs(min(0.0, negated)),  # neg_negated
                max(0.0, intensified),   # pos_intensified
                abs(min(0.0, intensified)),  # neg_intensified
            ])
        return csr_matrix(np.array(rows, dtype=np.float32))
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for downstream use."""
        return np.array(self.feature_names_)


# ============== Sanity Tests ==============
def lexicon_sanity_tests() -> List[Tuple[str, Dict[str, float]]]:
    """
    Return test cases for verification.
    Used to validate lexicon behavior on hard cases.
    """
    test_cases = [
        "not bad",
        "not good", 
        "not great",
        "not terrible",
        "very good",
        "extremely bad",
        "good but late delivery",
        "great product but support is awful",
        "love it",
        "hate it",
        "ok",
        "slightly disappointed",
        "absolutely amazing",
        "not bad at all",
    ]
    return [(text, extract_sentiment_features(text)) for text in test_cases]


if __name__ == "__main__":
    # Quick demo
    print("=== AFINN Lexicon Sanity Tests ===\n")
    for text, features in lexicon_sanity_tests():
        print(f"Text: '{text}'")
        print(f"  Raw: {features['afinn_raw']:+.1f}")
        print(f"  Negated: {features['afinn_negated']:+.1f}")
        print(f"  Intensified: {features['afinn_intensified']:+.1f}")
        print(f"  Contrast: {features['afinn_contrast']:+.1f}")
        print()

