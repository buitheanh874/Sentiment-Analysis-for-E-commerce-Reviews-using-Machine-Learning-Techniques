# -*- coding: utf-8 -*-
"""
NLP Demo: Customer Review Sentiment Classification
===================================================
Type a review to see the model's sentiment prediction.
Model returns: Positive, Negative, or Uncertain (if confidence is low).

Usage:
    python demo.py                           # Interactive mode
    python demo.py "review text here"        # Batch mode
    python demo.py --json "review text"      # JSON output (slide-friendly)
"""

import argparse
import json
import os
import sys
import io
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from src.dm2_steps.common import DEFAULT_THRESHOLDS, MIN_NNZ_DEFAULT, apply_uncertainty_rule
from src.issue_steps import load_issue_bundle, predict_issue_labels
from src.text_features import clean_text

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============== Configuration ==============
VERSION = "1.0.0"
THRESHOLDS = DEFAULT_THRESHOLDS  # default fallback
MIN_NNZ = MIN_NNZ_DEFAULT  # default fallback

# ============== Load Models ==============
def load_models(base_dir: Path, verbose: bool = True):
    """Load vectorizer, selector, and classifier from models/ directory."""
    models_dir = base_dir / "models"
    
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    selector_path = models_dir / "chi2_selector.joblib"
    model_path = models_dir / "best_lr_model.joblib"
    meta_path = models_dir / "variant_meta.json"
    
    # Check if models exist
    missing = []
    for path in [vectorizer_path, selector_path, model_path]:
        if not path.exists():
            missing.append(path.name)
    
    if missing:
        print(f"[ERROR] Missing model files: {', '.join(missing)}")
        print(f"        Run 'python -m src.run_all --data_path data/Gift_Cards.jsonl' first.")
        sys.exit(1)
    
    vectorizer = joblib.load(vectorizer_path)
    selector = joblib.load(selector_path)
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    
    # Model metadata
    model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
    k_features = selector.k if hasattr(selector, 'k') else selector.get_support().sum()
    
    # Get vocab size - handle both TfidfVectorizer and FeatureUnion
    if hasattr(vectorizer, 'vocabulary_'):
        vocab_size = len(vectorizer.vocabulary_)
    elif hasattr(vectorizer, 'transformer_list'):
        # FeatureUnion: sum vocab sizes from TF-IDF components
        vocab_size = 0
        for name, trans in vectorizer.transformer_list:
            if hasattr(trans, 'vocabulary_'):
                vocab_size += len(trans.vocabulary_)
            elif hasattr(trans, 'named_steps'):
                # Pipeline: look for tfidf step
                for step_name, step in trans.named_steps.items():
                    if hasattr(step, 'vocabulary_'):
                        vocab_size += len(step.vocabulary_)
            elif hasattr(trans, 'feature_names_'):
                vocab_size += len(trans.feature_names_)
    else:
        vocab_size = k_features  # Fallback
    
    class_weight = getattr(model, 'class_weight', None)
    thresholds = tuple(meta.get("thresholds", DEFAULT_THRESHOLDS))
    min_nnz = int(meta.get("min_nnz", MIN_NNZ_DEFAULT))
    
    model_info = {
        "version": VERSION,
        "trained_at": model_mtime.strftime("%Y-%m-%d %H:%M"),
        "k_features": int(k_features),
        "vocab_size": vocab_size,
        "class_weight": str(class_weight) if class_weight else "none",
        "thresholds": f"{thresholds[0]:.2f}/{thresholds[1]:.2f}",
        "min_nnz": min_nnz,
        "variant": meta.get("variant", "unknown"),
    }
    
    if verbose:
        print(f"[*] NLP Sentiment Demo v{VERSION}")
        print(f"    Model trained: {model_info['trained_at']}")
        print(f"    Config: K*={k_features}, vocab={vocab_size}, thresholds={thresholds[0]:.2f}/{thresholds[1]:.2f}")
        print(f"    Class weight: {model_info['class_weight']}, min_nnz={min_nnz}, variant={model_info['variant']}")
        print(f"[OK] Models loaded!\n")
    
    return vectorizer, selector, model, meta, model_info


def load_issue_model(base_dir: Path, verbose: bool = True):
    """Optionally load trained Stage-2 issue classifier artifacts."""
    model_dir = base_dir / "models" / "issue_classifier"
    issue_bundle = load_issue_bundle(model_dir)
    if verbose:
        if issue_bundle is None:
            print("[INFO] Learned issue model not found (models/issue_classifier).")
            print("      Fallback to rule-based issue_tags is active.\n")
        else:
            print(
                f"[OK] Learned issue model loaded: {len(issue_bundle.label_list)} labels "
                f"from {model_dir}\n"
            )
    return issue_bundle


# ============== Prediction ==============
def predict_sentiment(text: str, vectorizer, selector, model, meta, issue_bundle=None):
    """
    Predict sentiment for a given review text.
    
    Returns:
        dict with keys: 'label', 'confidence', 'probability', 'preprocessed_text', 'fallback_reason'
    """
    thresholds = tuple(meta.get("thresholds", THRESHOLDS))
    min_nnz = int(meta.get("min_nnz", MIN_NNZ))
    enable_abbrev = bool(meta.get("enable_abbrev_norm", False))
    enable_neg = bool(meta.get("negation", False))
    neg_window = int(meta.get("negation_window", 3))

    def _attach_issue_outputs(result: dict, fallback_issue_tags=None):
        label = result.get("label", "")
        if label in {"NEGATIVE", "NEEDS_ATTENTION"}:
            if issue_bundle is not None:
                issue_pred = predict_issue_labels(text, issue_bundle)
                result["issue_model_status"] = "trained"
                result["issue_labels"] = issue_pred.get("predicted_labels", [])
                result["issue_confidences"] = issue_pred.get("confidences", {})
                result["issue_thresholds"] = issue_pred.get("thresholds", {})
                if fallback_issue_tags is not None:
                    result["issue_tags"] = fallback_issue_tags
            else:
                result["issue_model_status"] = "missing"
                result["issue_model_message"] = (
                    "Learned issue model is not trained yet; using rule-based issue_tags."
                )
                result["issue_tags"] = (
                    fallback_issue_tags
                    if fallback_issue_tags is not None
                    else result.get("issue_tags", [])
                )
                result["issue_labels"] = []
                result["issue_confidences"] = {}
                result["issue_thresholds"] = {}
        else:
            if fallback_issue_tags is not None:
                result["issue_tags"] = fallback_issue_tags
            result["issue_model_status"] = "not_applicable"
            result.setdefault("issue_labels", [])
            result.setdefault("issue_confidences", {})
            result.setdefault("issue_thresholds", {})
        return result

    # Special case for common single-word reviews (including abbreviations)
    SINGLE_WORD_POSITIVE = [
        # Standard words
        "ok", "okay", "Ð¾Ðº", "oke", "good", "great", "nice", "love", "excellent", 
        "amazing", "perfect", "awesome", "wonderful", "fantastic", "superb", "brilliant",
        # Abbreviations
        "gr8", "gud", "gd", "luv", "thx", "tks", "ty", "tyvm", "tysm", "noice", "lit", "fire",
        "10/10", "5/5", "a+", "perf",
    ]
    SINGLE_WORD_NEGATIVE = [
        # Standard words  
        "bad", "terrible", "awful", "horrible", "worst", "hate", "trash", "poor",
        "disappointed", "disappointing", "useless", "scam", "fraud", "fake",
        # Abbreviations
        "sux", "sucks", "meh", "nah", "ugh", "wtf", "0/10", "1/10", "f",
    ]
    
    text_clean = text.strip().lower()
    if text_clean in SINGLE_WORD_POSITIVE:
        return _attach_issue_outputs({
            "label": "POSITIVE",
            "confidence": "Medium",
            "probability": 0.75 if text_clean not in ["ok", "okay", "oke"] else 0.55,
            "preprocessed_text": text_clean,
            "fallback_reason": None,
            "thresholds": thresholds,
            "min_nnz": min_nnz,
        }, fallback_issue_tags=[])
    if text_clean in SINGLE_WORD_NEGATIVE:
        return _attach_issue_outputs({
            "label": "NEGATIVE",
            "confidence": "Medium",
            "probability": 0.15,
            "preprocessed_text": text_clean,
            "fallback_reason": None,
            "thresholds": thresholds,
            "min_nnz": min_nnz,
        }, fallback_issue_tags=[])
    
    # Rating pattern detection (e.g., "2/10", "3/5", "1 star")
    import re
    rating_match = re.match(r'^(\d+)\s*/\s*(\d+)$', text_clean)
    if rating_match:
        score, total = int(rating_match.group(1)), int(rating_match.group(2))
        ratio = score / total if total > 0 else 0.5
        if ratio >= 0.7:  # 7/10, 4/5, 5/5
            label = "POSITIVE"
        elif ratio <= 0.4:  # 0-4/10, 0-2/5
            label = "NEGATIVE"
        else:  # 5-6/10, 3/5
            label = "NEEDS_ATTENTION"
        return _attach_issue_outputs({
            "label": label,
            "confidence": "Medium",
            "probability": ratio,
            "preprocessed_text": text_clean,
            "fallback_reason": "rating_detected",
            "thresholds": thresholds,
            "min_nnz": min_nnz,
        }, fallback_issue_tags=[])

    cleaned = clean_text(
        text,
        enable_abbrev_norm=enable_abbrev,
        enable_negation=enable_neg,
        negation_window=neg_window,
    )
    tfidf_vec = vectorizer.transform([text])
    prob_positive = model.predict_proba(selector.transform(tfidf_vec))[0, 1]

    # === NEGATION-AWARE COMPLAINT DETECTION ===
    # Fix false positives: "not bad", "no problem", "without issues" should NOT be flagged
    
    CONTRAST_MARKERS = ["but", "however", "although", "though", "yet", "except"]
    
    # Grouped by issue type (for future categorization)
    COMPLAINT_KEYWORDS = {
        "shipping": ["slow", "late", "delayed", "wait", "waiting"],
        "quality": ["bad", "poor", "terrible", "awful", "horrible", "worst", "cheap", "flimsy", "defective"],
        "packaging": ["damaged", "broken", "missing"],
        "service": ["rude", "unhelpful", "disappointing", "disappointed", "frustrated", "annoying"],
        "usability": ["confusing", "difficult", "hard", "complicated", "error", "failed"],
        "value": ["expensive", "overpriced", "waste", "useless", "scam"],
        "general": ["issue", "problem", "trouble", "wrong", "never"],
    }
    ALL_COMPLAINT_WORDS = [w for words in COMPLAINT_KEYWORDS.values() for w in words]
    
    # Slang/multi-meaning words that should NOT trigger complaint (false positive prevention)
    SLANG_EXCLUSIONS = ["badass", "bad ass", "sick", "wicked", "killer", "insane"]
    
    # Negation patterns that neutralize complaint words
    NEGATION_PATTERNS = ["not ", "no ", "without ", "never had ", "zero ", "none ", "didn't have "]
    
    text_lower = text.lower()
    
    # Check for slang exclusions first
    has_slang_positive = any(slang in text_lower for slang in SLANG_EXCLUSIONS)
    
    def has_real_complaint(text_lower, keywords):
        """
        Check if text has complaint keywords that are NOT negated.
        Returns True only if keyword exists WITHOUT preceding negation.
        """
        for keyword in keywords:
            if keyword not in text_lower:
                continue
            
            # Find all occurrences of the keyword
            idx = 0
            while True:
                pos = text_lower.find(keyword, idx)
                if pos == -1:
                    break
                
                # Check if preceded by negation pattern
                prefix = text_lower[max(0, pos - 15):pos]  # Look back 15 chars
                is_negated = any(neg in prefix for neg in NEGATION_PATTERNS)
                
                if not is_negated:
                    return True  # Found un-negated complaint keyword
                
                idx = pos + 1
        
        return False
    
    def get_issue_tags(text_lower, complaint_keywords):
        """Get list of issue categories detected in text."""
        tags = []
        for category, keywords in complaint_keywords.items():
            if has_real_complaint(text_lower, keywords):
                tags.append(category)
        return tags
    
    has_contrast = any(marker in text_lower for marker in CONTRAST_MARKERS)
    has_complaint = has_real_complaint(text_lower, ALL_COMPLAINT_WORDS) and not has_slang_positive
    issue_tags = get_issue_tags(text_lower, COMPLAINT_KEYWORDS) if has_complaint else []
    mixed_signal = has_contrast and has_complaint  # positive text but has concern
    
    # Count complaint occurrences (for p>=0.95 exception)
    complaint_count = sum(1 for w in ALL_COMPLAINT_WORDS if w in text_lower)
    
    # === TRIAGE POLICY (Final Teacher-approved 4-state) ===
    # 1. NEGATIVE: p <= 0.40 (strong negative signal)
    # 2. POSITIVE: p >= 0.95 AND single complaint + no contrast (very confident + minor issue)
    #           OR p >= 0.60 AND no complaint keyword
    # 3. NEEDS_ATTENTION: has complaint keyword (intentionally high recall on complaints)
    # 4. UNCERTAIN: fallback (too_short, threshold_band, sparse)
    
    # Rule 1: Strong negative â†’ NEGATIVE
    if prob_positive <= 0.40:
        return _attach_issue_outputs({
            "label": "NEGATIVE",
            "confidence": "High",
            "probability": prob_positive,
            "preprocessed_text": cleaned,
            "fallback_reason": None,
            "issue_tags": issue_tags,
            "mixed_signal": mixed_signal,
            "thresholds": thresholds,
            "min_nnz": min_nnz,
        }, fallback_issue_tags=issue_tags)
    
    # Rule 2 exception: Very confident positive (p>=0.95) with single complaint + no contrast
    # â†’ Allow POSITIVE to reduce false alarm (business tunable)
    if prob_positive >= 0.95 and has_complaint and complaint_count == 1 and not has_contrast:
        return _attach_issue_outputs({
            "label": "POSITIVE",
            "confidence": "High",
            "probability": prob_positive,
            "preprocessed_text": cleaned,
            "fallback_reason": "very_confident_positive",
            "issue_tags": issue_tags,  # Still show tags for transparency
            "mixed_signal": False,
            "thresholds": thresholds,
            "min_nnz": min_nnz,
        }, fallback_issue_tags=issue_tags)
    
    # Rule 3: Has complaint keyword â†’ NEEDS_ATTENTION (high recall on complaints)
    if has_complaint:
        return _attach_issue_outputs({
            "label": "NEEDS_ATTENTION",
            "confidence": "Medium",
            "probability": prob_positive,
            "preprocessed_text": cleaned,
            "fallback_reason": "has_complaint_keyword",
            "issue_tags": issue_tags,
            "mixed_signal": mixed_signal,
            "thresholds": thresholds,
            "min_nnz": min_nnz,
        }, fallback_issue_tags=issue_tags)

    # Apply standard uncertainty rule for remaining cases
    decision_row = apply_uncertainty_rule(
        np.array([prob_positive]),
        [cleaned],
        tfidf_vec,
        thresholds=thresholds,
        min_nnz=min_nnz,
    ).iloc[0]

    label_map = {1: "POSITIVE", 0: "NEGATIVE", -1: "UNCERTAIN"}
    label = label_map.get(decision_row["decision"], "UNCERTAIN")
    confidence = "High" if decision_row["decision"] in [0, 1] else "Low"
    fallback_reason = decision_row["reason"]

    return _attach_issue_outputs({
        "label": label,
        "confidence": confidence,
        "probability": prob_positive if decision_row["decision"] != -1 else np.nan,
        "preprocessed_text": cleaned,
        "fallback_reason": fallback_reason,
        "issue_tags": [],
        "mixed_signal": False,
        "thresholds": thresholds,
        "min_nnz": min_nnz,
    }, fallback_issue_tags=[])


# ============== Display Result ==============
def display_result(result: dict, original_text: str = None):
    """Pretty print prediction result."""
    print("\n" + "=" * 60)
    print("             PREDICTION RESULT")
    print("=" * 60)
    
    label = result["label"]
    prob = result["probability"]
    confidence = result["confidence"]
    fallback_reason = result.get("fallback_reason")
    
    if original_text:
        display_text = original_text[:80] + "..." if len(original_text) > 80 else original_text
        print(f"\n  Input: \"{display_text}\"")
    
    # Symbol based on sentiment
    if "POSITIVE" in label and "Uncertain" not in label:
        symbol = "[+]"
    elif "NEGATIVE" in label and "Uncertain" not in label:
        symbol = "[-]"
    else:
        symbol = "[?]"
    
    print(f"\n  {symbol} Sentiment: {label}")
    
    # Handle NaN probability for fallback cases
    if np.isnan(prob):
        print(f"      P(Positive): N/A")
        print(f"      Confidence:  {confidence}")
        if fallback_reason:
            print(f"      Reason:      {fallback_reason}")
        print("\n  [Cannot compute probability bar for fallback cases]")
    else:
        print(f"      P(Positive): {prob:.4f}")
        print(f"      Confidence:  {confidence}")
        
        # Visual probability bar
        bar_length = 40
        filled = int(prob * bar_length)
        bar = "#" * filled + "-" * (bar_length - filled)
        print(f"\n  Negative |{bar}| Positive")
        print(f"         0% {' ' * (bar_length - 8)} 100%")
    
    # Threshold visualization
    low, high = result.get("thresholds", THRESHOLDS)
    print(f"\n  Thresholds: Negative <= {low:.2f} | Uncertain | {high:.2f} <= Positive")

    if label in {"NEGATIVE", "NEEDS_ATTENTION"}:
        issue_status = result.get("issue_model_status", "missing")
        if issue_status == "trained":
            issue_rows = result.get("issue_labels", [])
            if issue_rows:
                print("\n  Learned issue labels:")
                for row in issue_rows:
                    print(f"    - {row['label']}: {row['confidence']:.3f}")
            else:
                print("\n  Learned issue labels: none above threshold")
        else:
            print("\n  Learned issue model: not trained, using rule-based issue_tags")
            fallback_tags = result.get("issue_tags", [])
            if fallback_tags:
                print(f"    issue_tags: {', '.join(fallback_tags)}")
            else:
                print("    issue_tags: none")
    print("=" * 60 + "\n")


# ============== Interactive Mode ==============
def interactive_mode(vectorizer, selector, model, meta, issue_bundle=None):
    """Run interactive demo loop."""
    print("=" * 60)
    print("    NLP DEMO: Customer Review Sentiment Classification")
    print("=" * 60)
    print("\nType an English review to see the sentiment prediction.")
    print("Commands:")
    print("  'examples' - Show sample reviews")
    print("  'quit'     - Exit the demo\n")
    
    examples = [
        "This gift card is amazing! Perfect gift for my friend.",
        "Terrible experience. The card never worked and no refund was given.",
        "It's okay, nothing special. Works as expected.",
        "Absolutely love it! Fast delivery and easy to redeem.",
        "Waste of money. Customer service was unhelpful.",
        "The gift card arrived on time. My mom was happy with it.",
    ]
    
    while True:
        try:
            user_input = input(">> Enter review: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "examples":
            print("\nSample reviews:")
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            print("\nTip: Enter a number (1-6) to try that example.\n")
            continue
        
        # Check if user entered a number for example
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(examples):
                user_input = examples[idx]
                print(f"\nAnalyzing: \"{user_input}\"")
            else:
                print(f"Please enter a number from 1-{len(examples)}")
                continue
        
        # Predict and display
        result = predict_sentiment(
            user_input, vectorizer, selector, model, meta, issue_bundle=issue_bundle
        )
        display_result(result, user_input)


# ============== Batch Mode ==============
def batch_mode(reviews: list, vectorizer, selector, model, meta, issue_bundle=None):
    """Process multiple reviews at once."""
    print("\n" + "=" * 80)
    print("BATCH PROCESSING RESULTS")
    print("=" * 80)
    print(f"{'#':<4} {'Prob':>6} {'Sentiment':<22} {'Review':<45}")
    print("-" * 80)
    
    for i, review in enumerate(reviews, 1):
        result = predict_sentiment(
            review, vectorizer, selector, model, meta, issue_bundle=issue_bundle
        )
        label = result["label"]
        prob = result["probability"]
        
        # Truncate long reviews
        display_text = review[:42] + "..." if len(review) > 45 else review
        
        # Handle NaN probability
        prob_str = "N/A" if np.isnan(prob) else f"{prob:.3f}"
        print(f"{i:<4} {prob_str:>6} {label:<22} {display_text:<45}")
        if label in {"NEGATIVE", "NEEDS_ATTENTION"}:
            if result.get("issue_model_status") == "trained":
                issue_rows = result.get("issue_labels", [])
                if issue_rows:
                    issue_summary = ", ".join(
                        [f"{row['label']}:{row['confidence']:.2f}" for row in issue_rows]
                    )
                else:
                    issue_summary = "none_above_threshold"
                print(f"     issues: {issue_summary}")
            else:
                fallback_tags = result.get("issue_tags", [])
                fallback_text = ", ".join(fallback_tags) if fallback_tags else "none"
                print(f"     issues(fallback): {fallback_text}")
    
    print("-" * 80)
    print()


# ============== JSON Output Mode ==============
def json_output_mode(
    reviews: list, vectorizer, selector, model, meta, model_info: dict, issue_bundle=None
):
    """Output prediction results as single-line JSON (slide-friendly)."""
    for review in reviews:
        result = predict_sentiment(
            review, vectorizer, selector, model, meta, issue_bundle=issue_bundle
        )
        
        # Build clean JSON record
        record = {
            "text": review,
            "clean": result["preprocessed_text"],
            "p_pos": round(result["probability"], 4) if not np.isnan(result["probability"]) else None,
            "label": result["label"].replace(" (Uncertain)", "").replace("UNCERTAIN", "Uncertain"),
            "reason": result["fallback_reason"],
            "issue_tags": result.get("issue_tags", []),
            "issue_model_status": result.get("issue_model_status"),
            "issue_labels": [row["label"] for row in result.get("issue_labels", [])],
            "issue_confidences": result.get("issue_confidences", {}),
        }
        print(json.dumps(record, ensure_ascii=False))


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(
        description="NLP Demo: Customer Review Sentiment Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                           # Interactive mode
  python demo.py "great product!"          # Batch mode
  python demo.py --json "great product!"   # JSON output (slide-friendly)
  python demo.py --json "text1" "text2"    # Multiple JSON outputs
        """
    )
    parser.add_argument("reviews", nargs="*", help="Review texts to classify")
    parser.add_argument("--json", "-j", action="store_true", 
                        help="Output as single-line JSON (machine-readable)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress model info banner (for JSON mode)")
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    
    # Load models (quiet if JSON mode)
    verbose = not (args.json or args.quiet)
    vectorizer, selector, model, meta, model_info = load_models(base_dir, verbose=verbose)
    issue_bundle = load_issue_model(base_dir, verbose=verbose)
    
    if args.reviews:
        if args.json:
            # JSON output mode
            json_output_mode(
                args.reviews,
                vectorizer,
                selector,
                model,
                meta,
                model_info,
                issue_bundle=issue_bundle,
            )
        else:
            # Batch mode
            batch_mode(args.reviews, vectorizer, selector, model, meta, issue_bundle=issue_bundle)
    else:
        # Interactive mode
        interactive_mode(vectorizer, selector, model, meta, issue_bundle=issue_bundle)


if __name__ == "__main__":
    main()


