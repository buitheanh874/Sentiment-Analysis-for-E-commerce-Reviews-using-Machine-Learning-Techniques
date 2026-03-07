# Issue Transformer Multi-label Summary

model_name: distilbert-base-uncased
split_method: random
train_size: 3000
val_size: 1000
test_size: 2000
threshold_mode: fixed_0.5
hybrid_margin: 0.080
hybrid_max_route_rate: 0.250

Main test metrics (transformer_multilabel):
- micro_f1: 0.6464
- macro_f1: 0.1878
- subset_accuracy: 0.5510
- hamming_loss: 0.1013

Files:
- nlp_issue_transformer_metrics_overall.csv
- nlp_issue_transformer_metrics_per_label.csv
- nlp_issue_transformer_thresholds.csv
- nlp_issue_transformer_per_label_f1.png
- nlp_issue_hybrid_metrics.csv
- nlp_issue_hybrid_routing.csv
- hybrid_routed_rows: 527