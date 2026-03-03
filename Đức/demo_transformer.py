# -*- coding: utf-8 -*-
"""
NLP Transformer Demo: Context-Aware Sentiment Classification
==============================================================
Uses fine-tuned DistilBERT to understand context and negation.

Usage:
    python demo_transformer.py                           # Interactive mode
    python demo_transformer.py "review text here"        # Batch mode
    python demo_transformer.py --json "review text"      # JSON output
"""

import argparse
import json
import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============== Configuration ==============
VERSION = "1.0.0"
THRESHOLDS = (0.40, 0.60)
MODEL_PATH = "models/transformer_model"  # Saved by nlp_ext training

# ============== Load Model ==============
def load_transformer_model(base_dir: Path, verbose: bool = True):
    """Load fine-tuned DistilBERT model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[ERROR] Missing optional dependencies: torch/transformers")
        print("        Install: pip install -r requirements-optional.txt")
        sys.exit(1)

    model_path = base_dir / MODEL_PATH
    
    if not model_path.exists():
        print(f"[ERROR] Transformer model not found at: {model_path}")
        print(f"        Run: python -m src.nlp_ext transformer_finetune --data_path data/Gift_Cards.jsonl")
        sys.exit(1)
    
    if verbose:
        print(f"[*] NLP Transformer Demo v{VERSION}")
        print(f"    Model: DistilBERT (fine-tuned)")
        print(f"    Thresholds: {THRESHOLDS[0]:.2f}/{THRESHOLDS[1]:.2f}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    if verbose:
        print(f"[OK] Model loaded!\n")
    
    return tokenizer, model


# ============== Prediction ==============
def predict_sentiment(text: str, tokenizer, model, thresholds=THRESHOLDS):
    """
    Predict sentiment using Transformer (context-aware).
    
    Returns:
        dict with keys: label, confidence, probability, fallback_reason
    """
    import torch

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Check for too short input
    token_count = inputs['input_ids'].shape[1] - 2  # Exclude [CLS] and [SEP]
    if token_count < 1 or text.strip() == "":
        return {
            "label": "UNCERTAIN (too short)",
            "confidence": "N/A",
            "probability": float('nan'),
            "fallback_reason": "too_short",
        }
    
    # OOV detection: only block gibberish (random characters)
    # Let Transformer understand context for real English words
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    content_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
    
    if content_tokens and len(content_tokens) <= 5:
        # Count subword tokens (##) - high ratio indicates gibberish
        subword_count = sum(1 for t in content_tokens if t.startswith('##'))
        whole_word_count = len(content_tokens) - subword_count
        
        # If mostly subword tokens with no real whole words â†’ gibberish
        # Examples: "bta" â†’ ['b', '##ta'], "xyz123" â†’ ['x', '##y', '##z', '##12', '##3']
        if whole_word_count <= 1 and subword_count >= 1 and len(content_tokens) <= 3:
            return {
                "label": "UNCERTAIN (unknown words)",
                "confidence": "N/A",
                "probability": float('nan'),
                "fallback_reason": "oov_detected",
            }
    
    # For short inputs: check if contains sentiment-indicating words
    # Prevents false positives on neutral greetings like "hello", "thanks"
    word_count = len(text.split())
    if word_count <= 3:
        text_lower = text.lower()
        sentiment_indicators = {
            'good', 'bad', 'great', 'terrible', 'excellent', 'awful', 'amazing',
            'horrible', 'wonderful', 'poor', 'best', 'worst', 'love', 'hate',
            'like', 'nice', 'fine', 'okay', 'ok', 'perfect', 'not', 
            'disappointed', 'happy', 'satisfied', 'recommend', 'broken',
            'fast', 'slow', 'cheap', 'expensive', 'worth', 'useless', 'useful',
            'waste', 'awesome', 'sucks', 'fantastic', 'mediocre', 'average',
        }
        has_sentiment = any(word in text_lower for word in sentiment_indicators)
        
        if not has_sentiment:
            return {
                "label": "UNCERTAIN (no clear sentiment)",
                "confidence": "N/A", 
                "probability": float('nan'),
                "fallback_reason": "no_sentiment",
            }
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prob_positive = probs[0][1].item()
    
    # Apply thresholds
    low, high = thresholds
    if prob_positive <= low:
        label = "NEGATIVE"
        confidence = "High"
    elif prob_positive >= high:
        label = "POSITIVE"
        confidence = "High"
    else:
        label = "UNCERTAIN"
        confidence = "Low"
    
    return {
        "label": label,
        "confidence": confidence,
        "probability": prob_positive,
        "fallback_reason": None,
    }


# ============== Display Result ==============
def display_result(result: dict, original_text: str = None):
    """Pretty print prediction result."""
    print("\n" + "=" * 60)
    print("        TRANSFORMER PREDICTION RESULT")
    print("=" * 60)
    
    if original_text:
        display_text = original_text[:80] + "..." if len(original_text) > 80 else original_text
        print(f"\n  Input: \"{display_text}\"")
    
    label = result["label"]
    prob = result["probability"]
    confidence = result["confidence"]
    fallback_reason = result.get("fallback_reason")
    
    # Symbol
    if "POSITIVE" in label:
        symbol = "[+]"
    elif "NEGATIVE" in label:
        symbol = "[-]"
    else:
        symbol = "[?]"
    
    print(f"\n  {symbol} Sentiment: {label}")
    
    # Handle NaN
    import math
    if math.isnan(prob):
        print(f"      P(Positive): N/A")
        print(f"      Confidence:  {confidence}")
        if fallback_reason:
            print(f"      Reason:      {fallback_reason}")
        print("\n  [Cannot compute probability bar for fallback cases]")
    else:
        print(f"      P(Positive): {prob:.4f}")
        print(f"      Confidence:  {confidence}")
        
        # Visual bar
        bar_length = 40
        filled = int(prob * bar_length)
        bar = "#" * filled + "-" * (bar_length - filled)
        print(f"\n  Negative |{bar}| Positive")
        print(f"         0% {' ' * (bar_length - 8)} 100%")
    
    low, high = THRESHOLDS
    print(f"\n  Thresholds: Negative <= {low:.2f} | Uncertain | {high:.2f} <= Positive")
    print("=" * 60 + "\n")


# ============== Interactive Mode ==============
def interactive_mode(tokenizer, model):
    """Interactive prediction mode."""
    print("=" * 60)
    print("  TRANSFORMER SENTIMENT ANALYSIS - Interactive Mode")
    print("=" * 60)
    print("\nType 'examples' to see test cases")
    print("Type 'quit' or 'exit' to stop\n")
    
    examples = [
        "This is amazing!",
        "Terrible experience",
        "not good, not bad",
        "not bad at all!",
        "good but late delivery",
    ]
    
    while True:
        user_input = input("Enter review text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'examples':
            print("\nExample reviews:")
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            print("\nType a number (1-5) to test an example, or enter your own text.\n")
            continue
        
        if not user_input:
            continue
        
        # Check if number
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(examples):
                user_input = examples[idx]
                print(f"\nAnalyzing: \"{user_input}\"")
            else:
                print(f"Please enter 1-{len(examples)}")
                continue
        
        result = predict_sentiment(user_input, tokenizer, model)
        display_result(result, user_input)


# ============== Batch Mode ==============
def batch_mode(reviews: list, tokenizer, model):
    """Process multiple reviews."""
    print("\n" + "=" * 80)
    print("TRANSFORMER BATCH PROCESSING")
    print("=" * 80)
    print(f"{'#':<4} {'Prob':>6} {'Sentiment':<22} {'Review':<45}")
    print("-" * 80)
    
    for i, review in enumerate(reviews, 1):
        result = predict_sentiment(review, tokenizer, model)
        label = result["label"]
        prob = result["probability"]
        
        display_text = review[:42] + "..." if len(review) > 45 else review
        
        import math
        prob_str = "N/A" if math.isnan(prob) else f"{prob:.3f}"
        print(f"{i:<4} {prob_str:>6} {label:<22} {display_text:<45}")
    
    print("-" * 80)
    print()


# ============== JSON Output ==============
def json_output_mode(reviews: list, tokenizer, model):
    """Output as JSON."""
    for review in reviews:
        result = predict_sentiment(review, tokenizer, model)
        
        import math
        record = {
            "text": review,
            "p_pos": round(result["probability"], 4) if not math.isnan(result["probability"]) else None,
            "label": result["label"],
            "reason": result["fallback_reason"],
            "model": "transformer",
        }
        print(json.dumps(record, ensure_ascii=False))


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(
        description="NLP Transformer Demo: Context-Aware Sentiment Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_transformer.py                           # Interactive mode
  python demo_transformer.py "not bad at all!"         # Batch mode
  python demo_transformer.py --json "not bad!"         # JSON output
        """
    )
    parser.add_argument("reviews", nargs="*", help="Review texts to classify")
    parser.add_argument("--json", "-j", action="store_true", help="JSON output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress banner")
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    
    verbose = not (args.json or args.quiet)
    tokenizer, model = load_transformer_model(base_dir, verbose=verbose)
    
    if args.reviews:
        if args.json:
            json_output_mode(args.reviews, tokenizer, model)
        else:
            batch_mode(args.reviews, tokenizer, model)
    else:
        interactive_mode(tokenizer, model)


if __name__ == "__main__":
    main()

