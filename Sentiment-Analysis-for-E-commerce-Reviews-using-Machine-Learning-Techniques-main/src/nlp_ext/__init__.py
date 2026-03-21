"""
Optional NLP extensions.

Entry points:
    python -m src.nlp_ext transformer_finetune --data_path data/Gift_Cards.jsonl
    python -m src.nlp_ext classic_syllabus_bench --data_path data/Gift_Cards.jsonl
    python -m src.nlp_ext classic_ablation --data_path data/Gift_Cards.jsonl
    python -m src.nlp_ext eval_rigor --data_path data/Gift_Cards.jsonl
    python -m src.nlp_ext issue_transformer_multilabel --labels_path data/issue_labels.csv --data_path data/Gift_Cards.jsonl
    python -m src.nlp_ext ngram_language_model --data_path data/Gift_Cards.jsonl
    python -m src.nlp_ext full_syllabus_upgrade --data_path data/Gift_Cards.jsonl
"""

__all__ = []
