# Step 09 - Uncertainty Evaluation
- Best variant V6 (Chi2 K*=10000, class_weight=w10) trained on train+val (selector fit on train).
- Thresholds: 0.40/0.60; coverage=0.885, recall_0@covered=0.956.
- Fallback rate (empty/sparse) on test: 0.086.
- 3-star uncertainty fractions saved to 09_uncertainty_3star.csv.
- Hard cases comparison (baseline vs best variant) saved to hard_cases_comparison.csv.
- Artifacts saved to models/: tfidf_vectorizer.joblib, chi2_selector.joblib, best_lr_model.joblib.