Imbalanced Classification for Vehicle Insurance Fraud Detection
===============================================================

Contents
  Report_Group_F-CC.pdf          Project report
  code/New_Insurance_C.py        Full analysis script (EDA, models, evaluation, figures)
  code/vehicle_insurance.csv     Dataset (copy of the public Kaggle file fraud_oracle.csv)
  code/requirements.txt          Python packages used
  outputs/eda_outputs/           EDA tables, figures and eda_findings.txt
  outputs/model_outputs/         CV and test results, figures, model_selection_report.txt

Dataset
  Public Kaggle dataset "Vehicle Insurance Claim Fraud Detection":
  https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
  A copy is already included next to the script as code/vehicle_insurance.csv.

How to run
  1. pip install -r code/requirements.txt
  2. cd code
  3. python New_Insurance_C.py

  The script finds the dataset automatically if vehicle_insurance.csv or
  fraud_oracle.csv is next to the script, in the current folder, or in a data/
  folder. Otherwise give the path explicitly:
     python New_Insurance_C.py --data-path path/to/fraud_oracle.csv

  The full run takes about 12 minutes on a 2-core machine. All random steps use
  seed 2026, so the results in the report are reproduced exactly. Outputs are
  written to eda_outputs/ and model_outputs/ next to the script.

  Versions used for the report: Python 3.13, pandas 3.0.5, NumPy 2.5.3,
  SciPy 1.18.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2, Matplotlib 3.11.2.
