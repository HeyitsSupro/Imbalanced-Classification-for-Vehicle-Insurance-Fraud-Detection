# Imbalanced Classification for Vehicle Insurance Fraud Detection

**Group F-CC:** Gurman Avtar (BSD-CC-2408), Ridhwan Choudhari (BSD-CC-2417), Supratik Biswas (BSD-CC-2423)

Only about 6% of vehicle insurance claims are fraudulent, so a model that predicts "legitimate" for every claim is 94% accurate but catches no fraud. This project compares three classifiers and four imbalance-handling techniques for detecting the fraudulent minority, with resampling applied to the training data only to avoid data leakage.

The full report is in [`Report_Group_F-CC.pdf`](Report_Group_F-CC.pdf).

## Dataset

[Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) (Kaggle, file `fraud_oracle.csv`): 15,420 claims from 1994–1996 with 33 columns, of which 923 (5.99%) are fraudulent. A copy is included as `code/vehicle_insurance.csv`.

## Approach

- **Exploratory analysis:** class imbalance, data-quality checks (e.g. `Age = 0` placeholders), feature associations with fraud (Cramér's V, Spearman) and a correlation heatmap of all features.
- **Classifiers:** Logistic Regression, Decision Tree, Random Forest.
- **Imbalance handling:** none (baseline), class weighting, random oversampling, SMOTE (implemented with SMOTENC for mixed numeric and categorical features) and random undersampling.
- **No data leakage:** preprocessing and resampling sit inside an imbalanced-learn `Pipeline`, so they only ever see the training part of each fold.
- **Evaluation:** 80/20 stratified split; 5-fold stratified cross-validation repeated 3 times on the development set; selection by cross-validated F1-score; final evaluation on the held-out test set with 95% bootstrap confidence intervals.
- **Metrics:** precision, recall, F1-score, false-positive rate, PR-AUC, ROC-AUC and confusion matrices.

## Key results

Without imbalance handling, all three models almost never predicted fraud at the 0.5 threshold. Every imbalance-handling technique improved recall and F1-score in all 15 cross-validation folds, at the cost of more false positives.

Selected model, **Random Forest with class weighting**, on the held-out test set (3,084 claims, 185 fraudulent):

| Metric | Random Forest + class weighting | Random Forest, no imbalance handling |
| --- | --- | --- |
| Recall | 0.595 (110 of 185 frauds caught) | 0.000 |
| Precision | 0.184 | 0.000 |
| F1-score | 0.281 | 0.000 |
| False-positive rate | 0.168 | 0.000 |
| PR-AUC | 0.193 | 0.215 |
| ROC-AUC | 0.829 | 0.824 |

- SMOTE improved on the baseline for every model but was the weakest of the four techniques on this dataset.
- Imbalance handling improved classification at the default threshold but did not clearly improve the threshold-free metrics (PR-AUC and ROC-AUC).

## Repository structure

```
├── Report_Group_F-CC.pdf      Project report
├── code/
│   ├── New_Insurance_C.py     Full analysis script (EDA, models, evaluation, figures)
│   ├── vehicle_insurance.csv  Dataset (copy of fraud_oracle.csv)
│   └── requirements.txt       Python packages used
└── outputs/
    ├── eda_outputs/           EDA tables, figures and eda_findings.txt
    └── model_outputs/         Cross-validation and test results, figures, model_selection_report.txt
```

## How to run

```bash
pip install -r code/requirements.txt
cd code
python New_Insurance_C.py
```

The script finds the dataset automatically if `vehicle_insurance.csv` or `fraud_oracle.csv` is next to it, in the current folder or in a `data/` folder. Otherwise, pass the path:

```bash
python New_Insurance_C.py --data-path path/to/fraud_oracle.csv
```

Results are written to `eda_outputs/` and `model_outputs/` next to the script. The full run takes about 12 minutes on a 2-core machine. All random steps use seed 2026, so the results in the report are reproduced exactly. The report's results were produced with Python 3.13, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2 and Matplotlib 3.11.2.
