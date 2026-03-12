# Citation and Reference Update Report

## Scope
- Target source: `results/reports/NLP_project_report.tex`
- PDF target rebuilt from this source: `results/reports/NLP_project_report.pdf`

## Detailed Changes

### 1) Citation [14] removed from internal negative-first policy
- File: `results/reports/NLP_project_report.tex`
- Location: Task A paragraph (line 74)
- Before:
  - `Therefore model selection uses a negative-first rule with priority order on recall for class 0, then F2 for class 0, then precision for class 0 \cite{saito2015}.`
- After:
  - `Therefore model selection follows an internal negative-first heuristic: prioritize recall for class 0, then F2 for class 0, then precision for class 0.`
- Reason:
  - The `recall -> F2 -> precision` ordering is documented as team policy/heuristic, not claimed as directly supported by citation [14].

### 2) Citations [10] and [11] moved to justify multi-label formulation, not label-count claim
- File: `results/reports/NLP_project_report.tex`
- Location: Task B paragraph (line 77)
- Before:
  - `One review can contain more than one issue, so single-label tagging is not enough. The output label set has nine issue types and allows multi-label assignment per row \cite{zhang2014,tsoumakas2011}.`
- After:
  - `One review can contain more than one issue, so single-label tagging is not enough \cite{tsoumakas2011,zhang2014}. The output label set in this project has nine issue types and allows multi-label assignment per row.`
- Reason:
  - [10] and [11] now support the multi-label setting itself, while the project-specific fact (`nine issue types`) is left uncited.

### 3) Citation [14] reassigned to imbalanced-data evaluation context
- File: `results/reports/NLP_project_report.tex`
- Location: Dataset imbalance paragraph (line 105)
- Before:
  - `This imbalance is important for NLP modeling choices.`
- After:
  - `This imbalance is important for NLP modeling choices and for precision-recall focused evaluation \cite{saito2015}.`
- Reason:
  - [14] is now attached to imbalanced evaluation framing (precision-recall emphasis).

### 4) Citation [11] removed from one-vs-rest implementation sentence
- File: `results/reports/NLP_project_report.tex`
- Location: Issue extraction model paragraph (line 195)
- Before:
  - `A LinearSVM baseline is also available \cite{rifkin2004,zhang2014}.`
- After:
  - `A LinearSVM baseline is also available \cite{rifkin2004}.`
- Reason:
  - Keeps [11] focused on multi-label justification in Task B instead of implementation detail here.

### 5) Citation [14] constrained to PR-vs-ROC interpretation under imbalance
- File: `results/reports/NLP_project_report.tex`
- Location: Main metrics paragraph (line 211)
- Before:
  - `Selective metrics under uncertainty include coverage, uncertain rate, selective precision_0, selective recall_0, selective F2_0, and false-negative rate on covered samples \cite{saito2015}.`
- After:
  - `Because the strong-label set is imbalanced, model trade-offs are interpreted mainly through precision-recall behavior rather than ROC-only views \cite{saito2015}. Selective metrics under uncertainty include coverage, uncertain rate, selective precision_0, selective recall_0, selective F2_0, and false-negative rate on covered samples.`
- Reason:
  - [14] now supports only the intended message: evaluation under class imbalance and PR-vs-ROC interpretation.

### 6) Citation [15] limited to selective classification/abstention concept
- File: `results/reports/NLP_project_report.tex`
- Location: Uncertainty rule paragraph (line 216)
- Before:
  - `It can return uncertain in the middle range. It also returns uncertain for too short or sparse text cases in classic pipeline \cite{geifman2017}.`
- After:
  - `It follows a selective classification abstention setup and can return uncertain in the middle range \cite{geifman2017}. We apply an additional heuristic in the classic pipeline that returns uncertain for too short or sparse text cases.`
- Reason:
  - [15] is now used for general selective classification/abstention only; short/sparse handling is explicitly marked as an additional project heuristic without citation.

## References Cleanup Check
- File checked: `results/reports/NLP_project_report.tex`
- Checked for:
  - URLs containing inserted spaces
  - Broken hyphenated terms such as `Pre- training`, `Net- works`, `Pro- ceedings`
- Result:
  - No such explicit source-level errors were present, so no textual reference-entry normalization was required.
