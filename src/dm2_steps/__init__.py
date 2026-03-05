"""
NLP sentiment lab-style steps package (legacy module name: dm2_steps).

Use via: python -m src.dm2_steps <step> --data_path data/Gift_Cards.jsonl
"""

from .common import (
    DM2Config,
    DEFAULT_DATA_PATH,
    DEFAULT_THRESHOLDS,
    MIN_NNZ_DEFAULT,
    DEFAULT_NEGATION_WINDOW,
)
from .steps import (
    step01_data_overview,
    step02_cleaning_preview,
    step03_split_summary,
    step04_tfidf_stats,
    step05_baseline_lr,
    step06_feature_selection,
    step06b_context_feature_variants_sweep,
    step07_embedded_l1,
    step08_ensemble,
    step09_uncertainty_eval,
    step10_threshold_sweep,
    step11_demo_one_review,
)

__all__ = [
    "DM2Config",
    "DEFAULT_DATA_PATH",
    "DEFAULT_THRESHOLDS",
    "MIN_NNZ_DEFAULT",
    "DEFAULT_NEGATION_WINDOW",
    "step01_data_overview",
    "step02_cleaning_preview",
    "step03_split_summary",
    "step04_tfidf_stats",
    "step05_baseline_lr",
    "step06_feature_selection",
    "step06b_context_feature_variants_sweep",
    "step07_embedded_l1",
    "step08_ensemble",
    "step09_uncertainty_eval",
    "step10_threshold_sweep",
    "step11_demo_one_review",
]

