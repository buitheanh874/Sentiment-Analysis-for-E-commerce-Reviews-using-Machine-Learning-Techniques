# Model Scoreboard

Generated from existing experiment artifacts.

| task                     | model                             | split   |   recall_0 |   precision_0 |       f2_0 |   micro_f1 |   macro_f1 |   coverage |   missed_negative_rate | source_file                                                          |
|:-------------------------|:----------------------------------|:--------|-----------:|--------------:|-----------:|-----------:|-----------:|-----------:|-----------------------:|:---------------------------------------------------------------------|
| issue_multilabel         | ovr_logreg                        | test    | nan        |    nan        | nan        |   0.764855 |   0.381154 | nan        |            nan         | results\issue_steps\02_metrics_overall.csv                           |
| rnn_lstm_sentiment       | lstm_text                         | test    |   0.910915 |      0.267142 |   0.614665 | nan        | nan        | nan        |              0.0890845 | results\nlp_ext\syllabus_upgrade\nlp_rnn_lstm_metrics.csv            |
| sentiment                | decision_tree|cw=none|k=10000     | test    |   0.629225 |      0.682061 |   0.639127 | nan        | nan        | nan        |              0.370775  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | decision_tree|cw=w5|k=10000       | test    |   0.643662 |      0.616318 |   0.638001 | nan        | nan        | nan        |              0.356338  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | decision_tree|cw=w2|k=10000       | test    |   0.629577 |      0.644789 |   0.632562 | nan        | nan        | nan        |              0.370423  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | decision_tree|cw=w10|k=10000      | test    |   0.607042 |      0.567665 |   0.598736 | nan        | nan        | nan        |              0.392958  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | decision_tree|cw=balanced|k=10000 | test    |   0.603521 |      0.576716 |   0.597963 | nan        | nan        | nan        |              0.396479  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | random_forest|cw=none|k=10000     | test    |   0.485915 |      0.902551 |   0.53534  | nan        | nan        | nan        |              0.514085  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | random_forest|cw=w2|k=10000       | test    |   0.455986 |      0.896194 |   0.505662 | nan        | nan        | nan        |              0.544014  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | random_forest|cw=w10|k=10000      | test    |   0.452113 |      0.872283 |   0.500312 | nan        | nan        | nan        |              0.547887  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | random_forest|cw=balanced|k=10000 | test    |   0.449296 |      0.871585 |   0.497505 | nan        | nan        | nan        |              0.550704  | results\dm2_steps\08_ensemble_metrics.csv                            |
| sentiment                | random_forest|cw=w5|k=10000       | test    |   0.435211 |      0.887294 |   0.484592 | nan        | nan        | nan        |              0.564789  | results\dm2_steps\08_ensemble_metrics.csv                            |
| syllabus_bench_sentiment | logreg_l2                         | test    |   0.91338  |      0.702981 |   0.861794 | nan        | nan        | nan        |              0.0866197 | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| syllabus_bench_sentiment | multinomial_nb                    | test    |   0.912676 |      0.633276 |   0.838672 | nan        | nan        | nan        |              0.0873239 | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| syllabus_bench_sentiment | sgd_log_loss                      | test    |   0.921479 |      0.616345 |   0.83846  | nan        | nan        | nan        |              0.0785211 | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| syllabus_bench_sentiment | linear_svm                        | test    |   0.847183 |      0.788594 |   0.834779 | nan        | nan        | nan        |              0.152817  | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| syllabus_bench_sentiment | complement_nb                     | test    |   0.952113 |      0.509516 |   0.811184 | nan        | nan        | nan        |              0.0478873 | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| syllabus_bench_sentiment | perceptron                        | test    |   0.828169 |      0.683721 |   0.794595 | nan        | nan        | nan        |              0.171831  | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| syllabus_bench_sentiment | svd_semantic_logreg               | test    |   0.919718 |      0.50144  |   0.788219 | nan        | nan        | nan        |              0.0802817 | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| syllabus_bench_sentiment | ffnn_mlp_svd                      | test    |   0.729577 |      0.798766 |   0.742439 | nan        | nan        | nan        |              0.270423  | results\nlp_ext\syllabus_upgrade\nlp_syllabus_bench_test_summary.csv |
| transformer_sentiment    | distilbert_finetune               | test    |   0.832746 |      0.821751 |   0.830524 | nan        | nan        |   0.910453 |              0.167254  | results\nlp_ext\nlp_metrics.csv                                      |

Notes:
- `missed_negative_rate = 1 - recall_0`.
- Multi-label rows use micro/macro F1 when class-0 metrics are not defined.