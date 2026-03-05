import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dm2_steps.common import (
    CS_THRESHOLD_PAIRS,
    DEFAULT_THRESHOLDS,
    DM2Config,
    MIN_NNZ_DEFAULT,
    clean_text,
    load_data,
    make_splits,
    metrics_from_probs,
    persist_core_artifacts,
    selective_metrics,
)
from src.dm2_steps.steps import _load_trained_artifacts, _train_best_lr
from src.text_features import DEFAULT_NEGATION_WINDOW
from .syllabus_upgrades import (
    build_course_fit_matrix,
    run_rnn_lstm_baseline,
    run_classic_syllabus_bench,
    run_ngram_language_model,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _decisions_from_probs(probs: np.ndarray, texts: list, thresholds) -> pd.DataFrame:
    low, high = thresholds
    records = []
    for prob, text in zip(probs, texts):
        token_count = len(text.split())
        decision = -1
        reason = None
        if token_count < 2 or text.strip() == "":
            reason = "too_short"
        else:
            if prob >= high:
                decision = 1
            elif prob <= low:
                decision = 0
            else:
                reason = "threshold_band"
        records.append({"prob": float(prob), "decision": decision, "reason": reason})
    return pd.DataFrame(records)


def _baseline_predict(texts: list, enable_abbrev_norm: bool, data_path: Path, output_dir: Path):
    vec, selector, model = _load_trained_artifacts()
    if vec is None or selector is None or model is None:
        config = DM2Config(
            data_path=data_path,
            output_dir=output_dir,
            enable_abbrev_norm=enable_abbrev_norm,
            min_nnz=MIN_NNZ_DEFAULT,
            thresholds=DEFAULT_THRESHOLDS,
        )
        splits, vec_bundle, selector, model, _, _ = _train_best_lr(config)
        vec = vec_bundle.vectorizer
        persist_core_artifacts(Path("models"), vec, selector, model)

    cleaned = [clean_text(t, enable_abbrev_norm) for t in texts]
    tfidf = vec.transform(cleaned)
    probs = model.predict_proba(selector.transform(tfidf))[:, 1]
    nnz = np.diff(tfidf.indptr)

    decisions = []
    for prob, ct, nz in zip(probs, cleaned, nnz):
        token_count = len(ct.split())
        if ct.strip() == "" or token_count < 2:
            decision = "Uncertain"
            reason = "too_short"
        elif nz < MIN_NNZ_DEFAULT:
            decision = "Uncertain"
            reason = "sparse_vector"
        elif prob >= DEFAULT_THRESHOLDS[1]:
            decision = "Positive"
            reason = None
        elif prob <= DEFAULT_THRESHOLDS[0]:
            decision = "Negative"
            reason = None
        else:
            decision = "Uncertain"
            reason = "threshold_band"
        decisions.append(
            {
                "text": ct,
                "prob": float(prob),
                "decision": decision,
                "reason": reason,
            }
        )
    return decisions


def transformer_finetune(args):
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print(
            "[NLP EXT] transformers/torch not installed. "
            "Install optional deps: pip install -r requirements-optional.txt"
        )
        return

    df = load_data(args.data_path)
    splits = make_splits(df, enable_abbrev_norm=args.enable_abbrev_norm)
    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    y_test = splits.test["label"].values

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    # Stratified-ish sampling for train size cap
    train_texts = splits.train["clean_text"].tolist()
    if len(train_texts) > args.max_train_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_texts), size=args.max_train_samples, replace=False)
        train_texts = [train_texts[i] for i in idx]
        y_train = y_train[idx]

    train_ds = ReviewDataset(train_texts, y_train)
    val_ds = ReviewDataset(splits.val["clean_text"].tolist(), y_val)
    test_ds = ReviewDataset(splits.test["clean_text"].tolist(), y_test)

    neg_count = max(1, int(np.sum(y_train == 0)))
    pos_count = max(1, int(np.sum(y_train == 1)))
    class_weights = torch.tensor(
        [len(y_train) / (2 * neg_count), len(y_train) / (2 * pos_count)],
        dtype=torch.float,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    train_args = TrainingArguments(
        output_dir="results/nlp_ext/hf_runs",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        seed=42,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    # Save model for demo usage
    model_save_path = Path("models/transformer_model")
    model_save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"[NLP EXT] Model saved to {model_save_path}")

    # Evaluation
    val_logits = trainer.predict(val_ds).predictions
    test_logits = trainer.predict(test_ds).predictions
    val_probs = _softmax(val_logits)[:, 1]
    test_probs = _softmax(test_logits)[:, 1]

    val_base = metrics_from_probs(y_val, val_probs, threshold=0.5)
    test_base = metrics_from_probs(y_test, test_probs, threshold=0.5)

    val_dec = _decisions_from_probs(val_probs, splits.val["clean_text"].tolist(), (args.threshold_low, args.threshold_high))
    test_dec = _decisions_from_probs(test_probs, splits.test["clean_text"].tolist(), (args.threshold_low, args.threshold_high))
    val_sel = selective_metrics(y_val, val_dec)
    test_sel = selective_metrics(y_test, test_dec)

    metrics_rows = [
        {"split": "val", **val_base, **val_sel},
        {"split": "test", **test_base, **test_sel},
    ]

    out_dir = Path("results/nlp_ext")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "nlp_metrics.csv", index=False)

    # Confusion matrix on test (0.5 decision)
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    y_pred_test = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Transformer Confusion Matrix (test)")
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_confusion_matrix.png", dpi=200)
    plt.close()

    # Probability histogram
    bins = np.linspace(0, 1, 31)
    plt.figure(figsize=(6, 4))
    plt.hist(test_probs[y_test == 0], bins=bins, alpha=0.6, label="Negative")
    plt.hist(test_probs[y_test == 1], bins=bins, alpha=0.6, label="Positive")
    plt.xlabel("P(Positive)")
    plt.ylabel("Count")
    plt.title("Transformer P(Positive) on test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_prob_hist_test.png", dpi=200)
    plt.close()

    # Threshold sweep
    sweep_rows = []
    for low, high in CS_THRESHOLD_PAIRS:
        dec = _decisions_from_probs(test_probs, splits.test["clean_text"].tolist(), (low, high))
        metrics = selective_metrics(y_test, dec)
        metrics.update({"threshold_low": low, "threshold_high": high})
        sweep_rows.append(metrics)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "nlp_threshold_sweep.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(sweep_df["coverage"], sweep_df["selective_recall_0"], marker="o")
    for _, row in sweep_df.iterrows():
        label = f"{row['threshold_low']:.2f}|{row['threshold_high']:.2f}"
        plt.text(row["coverage"], row["selective_recall_0"], label, fontsize=8, ha="right", va="bottom")
    plt.xlabel("Coverage")
    plt.ylabel("Selective recall_0")
    plt.title("Transformer threshold sweep (coverage vs recall_0)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_threshold_tradeoff.png", dpi=200)
    plt.close()

    # Hard cases comparison
    hard_cases = ["not bad", "not good", "good but late delivery", "gr8", "thx", "idk"]
    baseline_preds = _baseline_predict(
        hard_cases, args.enable_abbrev_norm, args.data_path, args.output_dir
    )
    hard_clean = [clean_text(h, args.enable_abbrev_norm) for h in hard_cases]
    hard_inputs = tokenizer(
        hard_clean,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        hard_logits = model(
            **{k: v.to(model.device) for k, v in hard_inputs.items()}
        ).logits
    hard_probs = torch.softmax(hard_logits, dim=1)[:, 1].cpu().numpy()
    transformer_decisions = []
    for prob, clean in zip(hard_probs, hard_clean):
        dec = _decisions_from_probs(
            np.array([prob]), [clean], (args.threshold_low, args.threshold_high)
        ).iloc[0]
        label = "Uncertain"
        if dec["decision"] == 1:
            label = "Positive"
        elif dec["decision"] == 0:
            label = "Negative"
        transformer_decisions.append(
            {"prob": float(prob), "decision": label, "reason": dec["reason"]}
        )

    comp_rows = []
    for hc, base, trans in zip(hard_cases, baseline_preds, transformer_decisions):
        comp_rows.append(
            {
                "text": hc,
                "baseline_decision": base["decision"],
                "baseline_prob": base["prob"],
                "baseline_reason": base["reason"],
                "transformer_decision": trans["decision"],
                "transformer_prob": trans["prob"],
                "transformer_reason": trans["reason"],
            }
        )
    pd.DataFrame(comp_rows).to_csv(out_dir / "hard_cases_comparison.csv", index=False)

    print(f"[NLP EXT] Metrics saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Optional NLP extensions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tf_parser = subparsers.add_parser("transformer_finetune", help="Fine-tune a transformer baseline")
    tf_parser.add_argument("--data_path", type=Path, default=Path("data/Gift_Cards.jsonl"))
    tf_parser.add_argument("--output_dir", type=Path, default=Path("results/dm2_steps"))
    tf_parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    tf_parser.add_argument("--epochs", type=float, default=1.0)
    tf_parser.add_argument("--batch_size", type=int, default=16)
    tf_parser.add_argument("--max_train_samples", type=int, default=8000)
    tf_parser.add_argument("--max_length", type=int, default=256)
    tf_parser.add_argument("--lr", type=float, default=2e-5)
    tf_parser.add_argument(
        "--enable_abbrev_norm", action="store_true", help="Apply abbreviation normalization"
    )
    tf_parser.add_argument("--threshold_low", type=float, default=DEFAULT_THRESHOLDS[0])
    tf_parser.add_argument("--threshold_high", type=float, default=DEFAULT_THRESHOLDS[1])

    bench_parser = subparsers.add_parser(
        "classic_syllabus_bench",
        help="Run strong classic NLP baselines (NB, Perceptron, SVM, FFNN over SVD).",
    )
    bench_parser.add_argument("--data_path", type=Path, default=Path("data/Gift_Cards.jsonl"))
    bench_parser.add_argument(
        "--output_dir", type=Path, default=Path("results/nlp_ext/syllabus")
    )
    bench_parser.add_argument("--variant", type=str, default="V6")
    bench_parser.add_argument("--max_train_samples", type=int, default=60000)
    bench_parser.add_argument("--svd_dim", type=int, default=256)
    bench_parser.add_argument("--mlp_max_iter", type=int, default=30)
    bench_parser.add_argument(
        "--enable_abbrev_norm", action="store_true", help="Apply abbreviation normalization"
    )
    bench_parser.add_argument(
        "--enable_negation_tagging", action="store_true", help="Enable negation tagging in cleaning"
    )
    bench_parser.add_argument(
        "--negation_window", type=int, default=DEFAULT_NEGATION_WINDOW
    )
    bench_parser.add_argument("--threshold_low", type=float, default=DEFAULT_THRESHOLDS[0])
    bench_parser.add_argument("--threshold_high", type=float, default=DEFAULT_THRESHOLDS[1])

    ngram_parser = subparsers.add_parser(
        "ngram_language_model",
        help="Train unigram/bigram language models and report perplexity.",
    )
    ngram_parser.add_argument("--data_path", type=Path, default=Path("data/Gift_Cards.jsonl"))
    ngram_parser.add_argument(
        "--output_dir", type=Path, default=Path("results/nlp_ext/syllabus")
    )
    ngram_parser.add_argument("--add_k", type=float, default=1.0)
    ngram_parser.add_argument("--gen_max_len", type=int, default=18)
    ngram_parser.add_argument(
        "--enable_abbrev_norm", action="store_true", help="Apply abbreviation normalization"
    )
    ngram_parser.add_argument(
        "--negation_window", type=int, default=DEFAULT_NEGATION_WINDOW
    )

    rnn_parser = subparsers.add_parser(
        "rnn_lstm_baseline",
        help="Train/evaluate an LSTM sentiment baseline.",
    )
    rnn_parser.add_argument("--data_path", type=Path, default=Path("data/Gift_Cards.jsonl"))
    rnn_parser.add_argument(
        "--output_dir", type=Path, default=Path("results/nlp_ext/syllabus_upgrade")
    )
    rnn_parser.add_argument("--max_train_samples", type=int, default=12000)
    rnn_parser.add_argument("--lstm_max_vocab", type=int, default=30000)
    rnn_parser.add_argument("--lstm_max_len", type=int, default=80)
    rnn_parser.add_argument("--lstm_emb_dim", type=int, default=128)
    rnn_parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    rnn_parser.add_argument("--lstm_num_layers", type=int, default=1)
    rnn_parser.add_argument("--lstm_dropout", type=float, default=0.2)
    rnn_parser.add_argument("--lstm_batch_size", type=int, default=128)
    rnn_parser.add_argument("--lstm_epochs", type=int, default=2)
    rnn_parser.add_argument("--lstm_lr", type=float, default=1e-3)
    rnn_parser.add_argument(
        "--enable_abbrev_norm", action="store_true", help="Apply abbreviation normalization"
    )
    rnn_parser.add_argument(
        "--enable_negation_tagging", action="store_true", help="Enable negation tagging in cleaning"
    )
    rnn_parser.add_argument(
        "--negation_window", type=int, default=DEFAULT_NEGATION_WINDOW
    )
    rnn_parser.add_argument("--variant", type=str, default="V6")
    rnn_parser.add_argument("--threshold_low", type=float, default=DEFAULT_THRESHOLDS[0])
    rnn_parser.add_argument("--threshold_high", type=float, default=DEFAULT_THRESHOLDS[1])

    fit_parser = subparsers.add_parser(
        "course_fit_matrix",
        help="Build a syllabus topic-coverage matrix from project artifacts.",
    )
    fit_parser.add_argument(
        "--output_dir", type=Path, default=Path("results/nlp_ext/syllabus")
    )

    full_parser = subparsers.add_parser(
        "full_syllabus_upgrade",
        help="Run classic bench + n-gram LM + course-fit matrix in one command.",
    )
    full_parser.add_argument("--data_path", type=Path, default=Path("data/Gift_Cards.jsonl"))
    full_parser.add_argument(
        "--output_dir", type=Path, default=Path("results/nlp_ext/syllabus")
    )
    full_parser.add_argument("--variant", type=str, default="V6")
    full_parser.add_argument("--max_train_samples", type=int, default=60000)
    full_parser.add_argument("--svd_dim", type=int, default=256)
    full_parser.add_argument("--mlp_max_iter", type=int, default=30)
    full_parser.add_argument("--add_k", type=float, default=1.0)
    full_parser.add_argument("--gen_max_len", type=int, default=18)
    full_parser.add_argument("--lstm_max_vocab", type=int, default=30000)
    full_parser.add_argument("--lstm_max_len", type=int, default=80)
    full_parser.add_argument("--lstm_emb_dim", type=int, default=128)
    full_parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    full_parser.add_argument("--lstm_num_layers", type=int, default=1)
    full_parser.add_argument("--lstm_dropout", type=float, default=0.2)
    full_parser.add_argument("--lstm_batch_size", type=int, default=128)
    full_parser.add_argument("--lstm_epochs", type=int, default=2)
    full_parser.add_argument("--lstm_lr", type=float, default=1e-3)
    full_parser.add_argument(
        "--enable_abbrev_norm", action="store_true", help="Apply abbreviation normalization"
    )
    full_parser.add_argument(
        "--enable_negation_tagging", action="store_true", help="Enable negation tagging in cleaning"
    )
    full_parser.add_argument(
        "--negation_window", type=int, default=DEFAULT_NEGATION_WINDOW
    )
    full_parser.add_argument("--threshold_low", type=float, default=DEFAULT_THRESHOLDS[0])
    full_parser.add_argument("--threshold_high", type=float, default=DEFAULT_THRESHOLDS[1])

    args = parser.parse_args()

    if args.command == "transformer_finetune":
        transformer_finetune(args)
    elif args.command == "classic_syllabus_bench":
        run_classic_syllabus_bench(args)
    elif args.command == "ngram_language_model":
        run_ngram_language_model(args)
    elif args.command == "rnn_lstm_baseline":
        run_rnn_lstm_baseline(args)
    elif args.command == "course_fit_matrix":
        build_course_fit_matrix(args)
    elif args.command == "full_syllabus_upgrade":
        run_classic_syllabus_bench(args)
        run_ngram_language_model(args)
        run_rnn_lstm_baseline(args)
        build_course_fit_matrix(args)


if __name__ == "__main__":
    main()
