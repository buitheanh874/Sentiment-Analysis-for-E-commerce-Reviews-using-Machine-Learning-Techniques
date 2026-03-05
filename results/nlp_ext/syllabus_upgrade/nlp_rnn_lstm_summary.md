# RNN/LSTM Baseline Summary

Model: single-direction LSTM (1 layer(s), hidden=128)
Train samples: 60000
Vocab size: 16114
Max sequence length: 80
Epochs: 2

Test metrics (threshold=0.5):
recall_0=0.911, precision_0=0.267, f2_0=0.615

Selective test metrics (threshold band):
coverage=0.898, selective_recall_0=0.927, selective_f2_0=0.631

File:
nlp_rnn_lstm_metrics.csv