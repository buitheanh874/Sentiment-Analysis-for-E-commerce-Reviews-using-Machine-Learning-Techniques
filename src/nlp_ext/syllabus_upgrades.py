import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from src.dm2_steps.common import (
    DEFAULT_THRESHOLDS,
    fit_vectorizer,
    load_data,
    make_splits,
    selective_metrics,
)
from src.text_features import CONTEXT_VARIANTS


SEED = 42


def _metrics_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1_vals, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_0": float(precision[0]),
        "recall_0": float(recall[0]),
        "f1_0": float(f1_vals[0]),
        "f2_0": float(fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0)),
        "precision_1": float(precision[1]),
        "recall_1": float(recall[1]),
        "f1_1": float(f1_vals[1]),
    }


def _subsample_train(X, y: np.ndarray, max_train_samples: int, seed: int):
    if max_train_samples <= 0 or len(y) <= max_train_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx_neg = np.where(y == 0)[0]
    idx_pos = np.where(y == 1)[0]
    n_neg = int(round(max_train_samples * (len(idx_neg) / len(y))))
    n_neg = min(len(idx_neg), max(1, n_neg))
    n_pos = max_train_samples - n_neg
    n_pos = min(len(idx_pos), max(1, n_pos))
    choose_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    choose_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    idx = np.concatenate([choose_neg, choose_pos])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _decision_from_probs(
    probs: np.ndarray, clean_texts: List[str], low: float, high: float
) -> pd.DataFrame:
    rows = []
    for prob, txt in zip(probs, clean_texts):
        token_count = len(str(txt).split())
        if token_count < 2 or str(txt).strip() == "":
            rows.append({"decision": -1, "reason": "too_short"})
            continue
        if prob >= high:
            rows.append({"decision": 1, "reason": None})
        elif prob <= low:
            rows.append({"decision": 0, "reason": None})
        else:
            rows.append({"decision": -1, "reason": "threshold_band"})
    return pd.DataFrame(rows)


def _pick_variant(name: str):
    for spec in CONTEXT_VARIANTS:
        if spec.name == name:
            return spec
    return next(v for v in CONTEXT_VARIANTS if v.name == "V6")


@dataclass
class SyllabusRunContext:
    splits: object
    X_train: object
    X_val: object
    X_test: object
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def _build_context(args) -> SyllabusRunContext:
    df = load_data(args.data_path)
    splits = make_splits(
        df,
        enable_abbrev_norm=args.enable_abbrev_norm,
        enable_negation=args.enable_negation_tagging,
        negation_window=args.negation_window,
    )
    spec = _pick_variant(args.variant)
    vec_bundle = fit_vectorizer(
        splits,
        variant=spec,
        enable_abbrev_norm=args.enable_abbrev_norm,
        negation_window=args.negation_window,
    )
    y_train = splits.train["label"].values
    y_val = splits.val["label"].values
    y_test = splits.test["label"].values
    return SyllabusRunContext(
        splits=splits,
        X_train=vec_bundle.X_train,
        X_val=vec_bundle.X_val,
        X_test=vec_bundle.X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def run_classic_syllabus_bench(args) -> None:
    np.random.seed(SEED)
    ctx = _build_context(args)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _subsample_train(
        ctx.X_train, ctx.y_train, args.max_train_samples, seed=SEED
    )
    X_val = ctx.X_val
    X_test = ctx.X_test
    y_val = ctx.y_val
    y_test = ctx.y_test

    rows = []
    test_comp = []

    models = {
        "logreg_l2": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=800,
            class_weight="balanced",
            random_state=SEED,
        ),
        "multinomial_nb": MultinomialNB(alpha=0.3),
        "complement_nb": ComplementNB(alpha=0.3),
        "perceptron": Perceptron(
            max_iter=25, class_weight="balanced", random_state=SEED, tol=1e-3
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=SEED),
        "sgd_log_loss": SGDClassifier(
            loss="log_loss",
            max_iter=35,
            class_weight="balanced",
            random_state=SEED,
            tol=1e-3,
        ),
    }

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - t0

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        val_metrics = _metrics_from_labels(y_val, y_val_pred)
        test_metrics = _metrics_from_labels(y_test, y_test_pred)

        rows.append(
            {
                "model": name,
                "split": "val",
                "train_seconds": train_seconds,
                **val_metrics,
            }
        )
        rows.append(
            {
                "model": name,
                "split": "test",
                "train_seconds": train_seconds,
                **test_metrics,
            }
        )
        test_comp.append(
            {
                "model": name,
                "recall_0": test_metrics["recall_0"],
                "precision_0": test_metrics["precision_0"],
                "f2_0": test_metrics["f2_0"],
            }
        )

        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]
            val_dec = _decision_from_probs(
                val_probs,
                ctx.splits.val["clean_text"].tolist(),
                args.threshold_low,
                args.threshold_high,
            )
            test_dec = _decision_from_probs(
                test_probs,
                ctx.splits.test["clean_text"].tolist(),
                args.threshold_low,
                args.threshold_high,
            )
            val_sel = selective_metrics(y_val, val_dec)
            test_sel = selective_metrics(y_test, test_dec)
            rows.append(
                {
                    "model": f"{name}_selective",
                    "split": "val",
                    "train_seconds": train_seconds,
                    **val_sel,
                }
            )
            rows.append(
                {
                    "model": f"{name}_selective",
                    "split": "test",
                    "train_seconds": train_seconds,
                    **test_sel,
                }
            )

    # Vector semantics + FFNN path (SVD embedding)
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=SEED)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)
    X_test_svd = svd.transform(X_test)

    sem_lr = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        random_state=SEED,
    )
    t0 = time.time()
    sem_lr.fit(X_train_svd, y_train)
    sem_lr_seconds = time.time() - t0
    sem_lr_val = _metrics_from_labels(y_val, sem_lr.predict(X_val_svd))
    sem_lr_test = _metrics_from_labels(y_test, sem_lr.predict(X_test_svd))
    rows.append(
        {"model": "svd_semantic_logreg", "split": "val", "train_seconds": sem_lr_seconds, **sem_lr_val}
    )
    rows.append(
        {"model": "svd_semantic_logreg", "split": "test", "train_seconds": sem_lr_seconds, **sem_lr_test}
    )
    test_comp.append(
        {
            "model": "svd_semantic_logreg",
            "recall_0": sem_lr_test["recall_0"],
            "precision_0": sem_lr_test["precision_0"],
            "f2_0": sem_lr_test["f2_0"],
        }
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=args.mlp_max_iter,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    t0 = time.time()
    mlp.fit(X_train_svd, y_train)
    mlp_seconds = time.time() - t0
    mlp_val = _metrics_from_labels(y_val, mlp.predict(X_val_svd))
    mlp_test = _metrics_from_labels(y_test, mlp.predict(X_test_svd))
    rows.append(
        {"model": "ffnn_mlp_svd", "split": "val", "train_seconds": mlp_seconds, **mlp_val}
    )
    rows.append(
        {"model": "ffnn_mlp_svd", "split": "test", "train_seconds": mlp_seconds, **mlp_test}
    )
    test_comp.append(
        {
            "model": "ffnn_mlp_svd",
            "recall_0": mlp_test["recall_0"],
            "precision_0": mlp_test["precision_0"],
            "f2_0": mlp_test["f2_0"],
        }
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "nlp_syllabus_bench_metrics.csv", index=False)
    pd.DataFrame(test_comp).to_csv(out_dir / "nlp_syllabus_bench_test_summary.csv", index=False)

    comp_df = pd.DataFrame(test_comp).sort_values(by=["recall_0", "f2_0"], ascending=False)

    plt.figure(figsize=(10, 4.8))
    plt.bar(comp_df["model"], comp_df["recall_0"], color="#2f6f9f")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Recall for negative class")
    plt.title("Test recall_0 across NLP syllabus baselines")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_syllabus_bench_recall0.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    plt.bar(comp_df["model"], comp_df["f2_0"], color="#4c9a2a")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("F2 for negative class")
    plt.title("Test F2_0 across NLP syllabus baselines")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_syllabus_bench_f2.png", dpi=220)
    plt.close()

    best = comp_df.iloc[0]
    summary_lines = [
        "# NLP Syllabus Bench Summary",
        "",
        f"Variant for sparse features: {args.variant}",
        f"Train subsample size: {len(y_train)} (max_train_samples={args.max_train_samples})",
        f"Threshold band for selective scoring: {args.threshold_low:.2f}/{args.threshold_high:.2f}",
        "",
        "Best test model by negative-first rule:",
        f"model={best['model']}, recall_0={best['recall_0']:.3f}, precision_0={best['precision_0']:.3f}, f2_0={best['f2_0']:.3f}",
        "",
        "Detailed metrics: nlp_syllabus_bench_metrics.csv",
        "Test summary: nlp_syllabus_bench_test_summary.csv",
    ]
    (out_dir / "nlp_syllabus_bench_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"[NLP EXT] Syllabus bench saved to {out_dir}")


def _tokenize_for_lm(text: str) -> List[str]:
    tokens = [tok for tok in str(text).split() if tok]
    return ["<s>"] + tokens + ["</s>"]


@dataclass
class NGramLM:
    order: int
    unigram: Counter
    bigram: Dict[str, Counter]
    vocab_size: int
    total_unigrams: int
    k: float

    def unigram_prob(self, w: str) -> float:
        return (self.unigram.get(w, 0) + self.k) / (
            self.total_unigrams + self.k * self.vocab_size
        )

    def bigram_prob(self, w_prev: str, w: str) -> float:
        c_prev = self.unigram.get(w_prev, 0)
        c_pair = self.bigram.get(w_prev, Counter()).get(w, 0)
        return (c_pair + self.k) / (c_prev + self.k * self.vocab_size)

    def sentence_log_prob(self, tokens: List[str]) -> float:
        if self.order == 1:
            return float(sum(math.log(self.unigram_prob(tok)) for tok in tokens[1:]))
        total = 0.0
        for i in range(1, len(tokens)):
            total += math.log(self.bigram_prob(tokens[i - 1], tokens[i]))
        return float(total)


def _fit_ngram_lm(texts: List[str], order: int = 2, k: float = 1.0) -> NGramLM:
    unigram = Counter()
    bigram = defaultdict(Counter)
    for text in texts:
        toks = _tokenize_for_lm(text)
        unigram.update(toks[1:])
        for i in range(1, len(toks)):
            bigram[toks[i - 1]][toks[i]] += 1
    vocab = set(unigram.keys())
    vocab.add("</s>")
    vocab_size = max(1, len(vocab))
    total_uni = int(sum(unigram.values()))
    return NGramLM(
        order=order,
        unigram=unigram,
        bigram=dict(bigram),
        vocab_size=vocab_size,
        total_unigrams=total_uni,
        k=k,
    )


def _perplexity(model: NGramLM, texts: List[str]) -> Tuple[float, float]:
    total_log_prob = 0.0
    total_tokens = 0
    for text in texts:
        toks = _tokenize_for_lm(text)
        total_log_prob += model.sentence_log_prob(toks)
        total_tokens += max(1, len(toks) - 1)
    avg_log_prob = total_log_prob / max(1, total_tokens)
    ppl = float(math.exp(-avg_log_prob))
    return ppl, float(avg_log_prob)


def _sample_next_from_counter(counter: Counter, rng: random.Random) -> str:
    if not counter:
        return "</s>"
    words, counts = zip(*counter.items())
    total = float(sum(counts))
    r = rng.random() * total
    cum = 0.0
    for w, c in zip(words, counts):
        cum += c
        if cum >= r:
            return w
    return words[-1]


def _generate_bigram(model: NGramLM, seed: str, max_len: int, rng: random.Random) -> str:
    prev = seed if seed else "<s>"
    if prev not in model.bigram:
        prev = "<s>"
    out = []
    for _ in range(max_len):
        nxt = _sample_next_from_counter(model.bigram.get(prev, Counter()), rng)
        if nxt == "</s>":
            break
        out.append(nxt)
        prev = nxt
    return " ".join(out)


def run_ngram_language_model(args) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    splits = make_splits(
        df,
        enable_abbrev_norm=args.enable_abbrev_norm,
        enable_negation=False,
        negation_window=args.negation_window,
    )

    train_texts = splits.train["clean_text"].tolist()
    val_texts = splits.val["clean_text"].tolist()
    test_texts = splits.test["clean_text"].tolist()

    unigram_lm = _fit_ngram_lm(train_texts, order=1, k=args.add_k)
    bigram_lm = _fit_ngram_lm(train_texts, order=2, k=args.add_k)

    rows = []
    for split_name, texts in [("val", val_texts), ("test", test_texts)]:
        ppl_u, alp_u = _perplexity(unigram_lm, texts)
        ppl_b, alp_b = _perplexity(bigram_lm, texts)
        rows.append(
            {
                "model": "unigram_add_k",
                "split": split_name,
                "perplexity": ppl_u,
                "avg_log_prob": alp_u,
                "add_k": args.add_k,
            }
        )
        rows.append(
            {
                "model": "bigram_add_k",
                "split": split_name,
                "perplexity": ppl_b,
                "avg_log_prob": alp_b,
                "add_k": args.add_k,
            }
        )
    lm_df = pd.DataFrame(rows)
    lm_df.to_csv(out_dir / "nlp_ngram_lm_metrics.csv", index=False)

    rng = random.Random(SEED)
    seeds = ["<s>", "great", "not", "delivery", "refund"]
    gen_rows = []
    for seed in seeds:
        for i in range(3):
            txt = _generate_bigram(bigram_lm, seed=seed, max_len=args.gen_max_len, rng=rng)
            gen_rows.append({"seed": seed, "sample_id": i + 1, "generated_text": txt})
    pd.DataFrame(gen_rows).to_csv(out_dir / "nlp_ngram_generated_samples.csv", index=False)

    test_u = lm_df[(lm_df["model"] == "unigram_add_k") & (lm_df["split"] == "test")].iloc[0]
    test_b = lm_df[(lm_df["model"] == "bigram_add_k") & (lm_df["split"] == "test")].iloc[0]
    summary_lines = [
        "# N-gram Language Model Summary",
        "",
        f"Add-k smoothing value: {args.add_k:.2f}",
        f"Unigram test perplexity: {test_u['perplexity']:.3f}",
        f"Bigram test perplexity: {test_b['perplexity']:.3f}",
        "",
        "Lower perplexity is better. Bigram should improve over unigram on this corpus.",
        "",
        "Files:",
        "nlp_ngram_lm_metrics.csv",
        "nlp_ngram_generated_samples.csv",
    ]
    (out_dir / "nlp_ngram_lm_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[NLP EXT] N-gram LM outputs saved to {out_dir}")


def build_course_fit_matrix(args) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_path = out_dir / "nlp_syllabus_bench_test_summary.csv"
    ngram_path = out_dir / "nlp_ngram_lm_metrics.csv"
    trans_path = Path("results/nlp_ext/nlp_metrics.csv")
    issue_path = Path("results/issue_steps_char_demo/02_metrics_overall.csv")

    has_bench = bench_path.exists()
    has_ngram = ngram_path.exists()
    has_trans = trans_path.exists()
    has_issue = issue_path.exists()

    # Simple coverage points by topic, max 1.0 each.
    topics = [
        ("Probability", 1.0 if has_ngram or has_bench else 0.5),
        ("Regular Expressions", 0.5),
        ("Evaluation Measures", 1.0),
        ("N-Gram Language Models", 1.0 if has_ngram else 0.2),
        ("Naive Bayes", 1.0 if has_bench else 0.2),
        ("Perceptron", 1.0 if has_bench else 0.2),
        ("Logistic Regression", 1.0),
        ("Feed-forward Neural Networks", 1.0 if has_bench else 0.3),
        ("Pytorch Basic", 0.6 if has_trans else 0.2),
        ("Vector Semantics and Embeddings", 0.8 if has_bench else 0.4),
        ("Recurrent Neural Networks", 0.2),
        ("Transformers and LLMs", 1.0 if has_trans else 0.3),
        ("Masked Language Models", 0.2),
        ("Applications of LLMs", 0.3),
    ]
    df = pd.DataFrame(topics, columns=["topic", "coverage_score"])
    df["coverage_percent"] = (df["coverage_score"] * 100).round(1)
    overall = float(df["coverage_score"].mean() * 100.0)
    df.to_csv(out_dir / "nlp_course_fit_matrix.csv", index=False)

    lines = [
        "# NLP Course Fit Matrix",
        "",
        f"Estimated overall syllabus coverage: {overall:.1f}%",
        "This matrix is score-based and uses available project artifacts only.",
        "",
        "Detailed rows are in nlp_course_fit_matrix.csv",
    ]
    (out_dir / "nlp_course_fit_matrix.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[NLP EXT] Course-fit matrix saved to {out_dir}")
