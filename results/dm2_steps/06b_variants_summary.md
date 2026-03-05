# Step 06b - Context Feature Variants
- Variants V0-V6 swept with K in {1000,2000,5000,10000} and class_weight grid.
- Best variant (val negative-first): V6 with k=10000, class_weight=w10.
- Full validation grid saved to 06b_variants_val_table.csv.
- Test metrics for best per variant saved to 06b_variants_test_table.csv.
- Recall/precision bar plots saved to 06b_recall0_by_variant.png / 06b_precision0_by_variant.png.
- Best descriptor saved to 06b_best_variant.txt (Word 1-2 + char 3-5 + negation).