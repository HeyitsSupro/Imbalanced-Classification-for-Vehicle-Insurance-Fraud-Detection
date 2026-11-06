import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Limit a non-fatal scikit-learn parallel warning repeated by individual workers.
warnings.filterwarnings(
    "once",
    message=(
        r"`sklearn\.utils\.parallel\.delayed` should be used with "
        r"`sklearn\.utils\.parallel\.Parallel`.*"
    ),
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "vehicle_insurance.csv"
# Places the script looks for the data when --data-path is not given: the
# renamed file or the original Kaggle file, next to the script, in the current
# folder, or in a "data" folder beside the script or one level up.
DATA_FILE_NAMES = ("vehicle_insurance.csv", "fraud_oracle.csv")
DATA_SEARCH_DIRS = (SCRIPT_DIR, Path.cwd(), SCRIPT_DIR / "data", SCRIPT_DIR.parent,
                    SCRIPT_DIR.parent / "data")
KAGGLE_URL = "https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection"
TARGET = "FraudFound_P"
IDENTIFIER_COLUMNS = ["PolicyNumber"]
MODEL_OUTPUT_DIR = SCRIPT_DIR / "model_outputs"
EDA_OUTPUT_DIR = SCRIPT_DIR / "eda_outputs"
RANDOM_STATE = 2026
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 3
DEFAULT_THRESHOLD = 0.5
REDUNDANCY_CRAMERS_V = 0.7
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = RANDOM_STATE + 100
RARE_CATEGORY_MIN_COUNT = 30
AGE_BINS = [0, 25, 35, 45, 55, 65, 101]
AGE_BAND_LABELS = ["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"]
RATE_FEATURES = [
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "AccidentArea",
    "PastNumberOfClaims",
]

BASELINE = "No resampling"
SMOTE_NAME = "SMOTE (SMOTENC)"
STRATEGY_NAMES = [
    BASELINE,
    "Class weighting",
    "Random oversampling",
    SMOTE_NAME,
    "Random undersampling",
]
STRATEGY_COLORS = {
    BASELINE: "#9E9E9E",
    "Class weighting": "#4E7891",
    "Random oversampling": "#147D76",
    SMOTE_NAME: "#D97757",
    "Random undersampling": "#8E6BB0",
}
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]

# Selection criterion: the proposal asks for a balance between catching fraud
# (recall) and controlling false alarms (precision / FPR). Fraud-class F1 is the
# harmonic mean of precision and recall, so it rewards exactly that balance.
SELECTION_METRIC = "f1_score"
SELECTION_TIEBREAK = "pr_auc"

METRICS = (
    "precision",
    "recall",
    "f1_score",
    "pr_auc",
    "roc_auc",
    "false_positive_rate",
    "specificity",
)
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "false_positive_rate": "False-positive rate",
    "specificity": "Specificity",
}
# Metrics for which a lower value is better.
LOWER_IS_BETTER = {"false_positive_rate"}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def cramers_v(table):
    """Return bias-corrected Cramer's V for a contingency table."""
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    sample_size = table.to_numpy().sum()
    rows, columns = table.shape
    if sample_size <= 1 or min(rows, columns) <= 1:
        return 0.0
    phi_squared = chi2 / sample_size
    corrected_phi_squared = max(
        0.0,
        phi_squared - ((columns - 1) * (rows - 1)) / (sample_size - 1),
    )
    corrected_rows = rows - ((rows - 1) ** 2) / (sample_size - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (sample_size - 1)
    corrected_dimension = min(corrected_rows - 1, corrected_columns - 1)
    return (
        float(np.sqrt(corrected_phi_squared / corrected_dimension))
        if corrected_dimension > 0
        else 0.0
    )


def adjust_p_values(p_values):
    """Apply the Benjamini-Hochberg adjustment."""
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return values
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1)
    adjusted = values[order] * len(values) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def age_bands(age):
    """Assign the shared age bands used in tables and plots."""
    return pd.cut(age, bins=AGE_BINS, right=False, labels=AGE_BAND_LABELS)


def flatten_columns(frame):
    """Flatten a two-level column index produced by groupby().agg()."""
    frame.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in frame.columns
    ]
    return frame


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def clean_data(df):
    """Apply documented data corrections and return their counts."""
    cleaned = df.copy()
    cleaning_counts = {}
    if "Age" in cleaned.columns:
        age_zero_mask = cleaned["Age"].eq(0)
        cleaned["AgeWasZero"] = age_zero_mask.astype(int)
        cleaning_counts["Age equal to 0 (set to missing)"] = int(age_zero_mask.sum())
        cleaned.loc[age_zero_mask, "Age"] = np.nan
    for column in ("MonthClaimed", "DayOfWeekClaimed"):
        if column in cleaned.columns:
            invalid_mask = cleaned[column].astype("string").eq("0").fillna(False)
            cleaning_counts[f"{column} equal to '0' (set to missing)"] = int(
                invalid_mask.sum()
            )
            cleaned.loc[invalid_mask, column] = np.nan
    if "RepNumber" in cleaned.columns:
        cleaned["RepNumber"] = cleaned["RepNumber"].astype("object")
    if "Year" in cleaned.columns:
        cleaned["Year"] = cleaned["Year"].astype("object")
    return cleaned, cleaning_counts


def group_rare_categories(values, minimum_count=RARE_CATEGORY_MIN_COUNT):
    """Combine rare values for descriptive summaries only."""
    grouped = values.astype("string").fillna("Missing")
    counts = grouped.value_counts()
    rare_values = counts[counts < minimum_count].index
    return grouped.where(~grouped.isin(rare_values), f"Other (<{minimum_count})")


# ---------------------------------------------------------------------------
# Modelling building blocks
# ---------------------------------------------------------------------------


def create_model_pipeline(numeric_columns, categorical_columns, classifier, sampler):
    """Build preprocessing -> optional resampling -> one-hot -> classifier.

    The imbalanced-learn Pipeline only calls the sampler during ``fit``. At
    prediction time the sampler is skipped, so resampling touches training data
    only and can never leak into validation or test rows.
    """
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    # Categorical columns are ordinal-coded first so that SMOTENC can treat them
    # as categories; they are one-hot encoded after resampling for the model.
    encode_for_sampling = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", ordinal_categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    numeric_indices = list(range(len(numeric_columns)))
    categorical_indices = list(
        range(len(numeric_columns), len(numeric_columns) + len(categorical_columns))
    )
    encode_for_model = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_indices),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("encode_for_sampling", encode_for_sampling),
            ("sampler", sampler if sampler is not None else "passthrough"),
            ("encode_for_model", encode_for_model),
            ("classifier", classifier),
        ]
    )


def make_classifier_factories():
    """Return the three baseline classifiers named in the proposal."""
    return {
        "Logistic Regression": lambda class_weight: LogisticRegression(
            class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": lambda class_weight: DecisionTreeClassifier(
            class_weight=class_weight,
            max_depth=7,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": lambda class_weight: RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_strategies(categorical_indices):
    """Return (name, sampler factory, class_weight) for each imbalance strategy."""
    return [
        (BASELINE, lambda: None, None),
        ("Class weighting", lambda: None, "balanced"),
        (
            "Random oversampling",
            lambda: RandomOverSampler(random_state=RANDOM_STATE),
            None,
        ),
        (
            SMOTE_NAME,
            lambda: SMOTENC(
                categorical_features=categorical_indices,
                random_state=RANDOM_STATE,
            ),
            None,
        ),
        (
            "Random undersampling",
            lambda: RandomUnderSampler(random_state=RANDOM_STATE),
            None,
        ),
    ]


def evaluate_scores(target, scores, threshold=DEFAULT_THRESHOLD):
    """Calculate threshold-based and ranking metrics for the fraud class."""
    target_values = np.asarray(target, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    predictions = (score_values >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        target_values, predictions, labels=[0, 1]
    ).ravel()
    recall = recall_score(target_values, predictions, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "decision_threshold": float(threshold),
        "precision": precision_score(target_values, predictions, zero_division=0),
        "recall": recall,
        "f1_score": f1_score(target_values, predictions, zero_division=0),
        "pr_auc": average_precision_score(target_values, score_values),
        "roc_auc": roc_auc_score(target_values, score_values),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "specificity": specificity,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ---------------------------------------------------------------------------
# Step 4a: repeated stratified cross-validation on the development data
# ---------------------------------------------------------------------------


def run_cross_validation(
    x_development, y_development, numeric_columns, categorical_columns
):
    """Evaluate all 15 model/strategy pairs with repeated stratified CV."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    categorical_indices = list(
        range(len(numeric_columns), len(numeric_columns) + len(categorical_columns))
    )
    model_factories = make_classifier_factories()
    strategies = make_strategies(categorical_indices)
    fold_rows = []
    for split_number, (train_idx, test_idx) in enumerate(
        cv.split(x_development, y_development)
    ):
        repeat = split_number // CV_SPLITS + 1
        fold = split_number % CV_SPLITS + 1
        print(f"  CV repeat {repeat}/{CV_REPEATS}, fold {fold}/{CV_SPLITS}", flush=True)
        x_train = x_development.iloc[train_idx]
        y_train = y_development.iloc[train_idx]
        x_valid = x_development.iloc[test_idx]
        y_valid = y_development.iloc[test_idx]
        for model_name, model_factory in model_factories.items():
            for strategy_name, sampler_factory, class_weight in strategies:
                pipeline = create_model_pipeline(
                    numeric_columns,
                    categorical_columns,
                    model_factory(class_weight),
                    sampler_factory(),
                )
                pipeline.fit(x_train, y_train)  # resampling happens here only
                scores = pipeline.predict_proba(x_valid)[:, 1]
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": model_name,
                        "imbalance_strategy": strategy_name,
                        **evaluate_scores(y_valid, scores),
                    }
                )
    return pd.DataFrame(fold_rows)


def summarise_cv(fold_results):
    """Mean and standard deviation of every metric across the CV folds."""
    summary = flatten_columns(
        fold_results.groupby(["model", "imbalance_strategy"])[list(METRICS)]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["model"] = pd.Categorical(summary["model"], MODEL_NAMES, ordered=True)
    summary["imbalance_strategy"] = pd.Categorical(
        summary["imbalance_strategy"], STRATEGY_NAMES, ordered=True
    )
    summary = summary.sort_values(["model", "imbalance_strategy"]).reset_index(drop=True)
    summary["model"] = summary["model"].astype(str)
    summary["imbalance_strategy"] = summary["imbalance_strategy"].astype(str)
    return summary


def paired_change_vs_baseline(fold_results):
    """Per-fold paired change of each strategy versus the same model's baseline."""
    keys = ["repeat", "fold", "model"]
    baseline = fold_results[fold_results["imbalance_strategy"] == BASELINE][
        keys + list(METRICS)
    ].rename(columns={metric: f"{metric}_baseline" for metric in METRICS})
    paired = fold_results[fold_results["imbalance_strategy"] != BASELINE].merge(
        baseline, on=keys, validate="many_to_one"
    )
    rows = []
    for (model_name, strategy_name), group in paired.groupby(
        ["model", "imbalance_strategy"], sort=False
    ):
        row = {"model": model_name, "imbalance_strategy": strategy_name}
        for metric in METRICS:
            delta = group[metric] - group[f"{metric}_baseline"]
            better = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
            row[f"delta_{metric}_mean"] = delta.mean()
            row[f"delta_{metric}_std"] = delta.std()
            row[f"{metric}_folds_improved"] = f"{int(better.sum())}/{len(delta)}"
        rows.append(row)
    return pd.DataFrame(rows)


def select_candidates(cv_summary):
    """Pick the best strategy per model and the overall winner by CV F1."""
    ranked = cv_summary.sort_values(
        [f"{SELECTION_METRIC}_mean", f"{SELECTION_TIEBREAK}_mean"],
        ascending=[False, False],
    )
    best_per_model = (
        ranked.groupby("model", sort=False).head(1).set_index("model").loc[MODEL_NAMES]
    ).reset_index()
    overall = ranked.iloc[0]
    return best_per_model, overall


# ---------------------------------------------------------------------------
# Step 4b: held-out test evaluation
# ---------------------------------------------------------------------------


def fit_and_score_on_test(
    x_development, y_development, x_test, numeric_columns, categorical_columns
):
    """Refit every candidate on all development rows and score the test set."""
    categorical_indices = list(
        range(len(numeric_columns), len(numeric_columns) + len(categorical_columns))
    )
    test_scores = {}
    for model_name, model_factory in make_classifier_factories().items():
        for strategy_name, sampler_factory, class_weight in make_strategies(
            categorical_indices
        ):
            pipeline = create_model_pipeline(
                numeric_columns,
                categorical_columns,
                model_factory(class_weight),
                sampler_factory(),
            )
            pipeline.fit(x_development, y_development)
            test_scores[(model_name, strategy_name)] = pipeline.predict_proba(x_test)[:, 1]
    return test_scores


def bootstrap_intervals(target, scores, baseline_scores, threshold=DEFAULT_THRESHOLD):
    """Stratified bootstrap CIs for the winner and its gain over the baseline."""
    target_values = np.asarray(target, dtype=int)
    negatives = np.flatnonzero(target_values == 0)
    positives = np.flatnonzero(target_values == 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_samples = {metric: [] for metric in METRICS}
    delta_samples = {metric: [] for metric in METRICS}
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = np.concatenate(
            (
                rng.choice(negatives, size=len(negatives), replace=True),
                rng.choice(positives, size=len(positives), replace=True),
            )
        )
        selected = evaluate_scores(target_values[idx], scores[idx], threshold)
        baseline = evaluate_scores(target_values[idx], baseline_scores[idx], threshold)
        for metric in METRICS:
            metric_samples[metric].append(selected[metric])
            delta_samples[metric].append(selected[metric] - baseline[metric])
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "ci_lower": np.percentile(metric_samples[metric], 2.5),
                "ci_upper": np.percentile(metric_samples[metric], 97.5),
                "delta_vs_baseline_ci_lower": np.percentile(delta_samples[metric], 2.5),
                "delta_vs_baseline_ci_upper": np.percentile(delta_samples[metric], 97.5),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_cv_strategy_comparison(cv_summary):
    """Grouped bars: each metric by model, one bar per imbalance strategy."""
    metrics = ["recall", "precision", "f1_score", "false_positive_rate", "pr_auc", "roc_auc"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    width = 0.8 / len(STRATEGY_NAMES)
    positions = np.arange(len(MODEL_NAMES))
    for axis, metric in zip(axes.flat, metrics):
        for offset, strategy in enumerate(STRATEGY_NAMES):
            rows = cv_summary[cv_summary["imbalance_strategy"] == strategy].set_index(
                "model"
            ).loc[MODEL_NAMES]
            axis.bar(
                positions + (offset - (len(STRATEGY_NAMES) - 1) / 2) * width,
                rows[f"{metric}_mean"],
                width,
                yerr=rows[f"{metric}_std"],
                capsize=2,
                color=STRATEGY_COLORS[strategy],
                label=strategy,
                error_kw={"elinewidth": 0.8},
            )
        direction = " (lower is better)" if metric in LOWER_IS_BETTER else ""
        axis.set_title(f"{METRIC_LABELS[metric]}{direction}")
        axis.set_xticks(positions, MODEL_NAMES)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(STRATEGY_NAMES), fontsize=9)
    fig.suptitle(
        f"Repeated stratified {CV_SPLITS}x{CV_REPEATS} CV on development data "
        f"(threshold {DEFAULT_THRESHOLD:.2f}, mean ± std)"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(MODEL_OUTPUT_DIR / "cv_strategy_comparison.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrix_grid(y_test, test_scores, best_per_model):
    """Test confusion matrices: baseline vs best strategy for each model."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    image = None
    for column, model_name in enumerate(MODEL_NAMES):
        best_strategy = best_per_model.set_index("model").loc[
            model_name, "imbalance_strategy"
        ]
        for row, (strategy, label) in enumerate(
            ((BASELINE, "Baseline"), (best_strategy, "Best imbalance strategy"))
        ):
            axis = axes[row, column]
            predictions = (test_scores[(model_name, strategy)] >= DEFAULT_THRESHOLD).astype(int)
            matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
            image = axis.imshow(matrix, cmap="Blues")
            midpoint = matrix.max() / 2
            for i in range(2):
                for j in range(2):
                    share = matrix[i, j] / matrix[i].sum() if matrix[i].sum() else 0
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:,}\n{share:.1%}",
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] > midpoint else "black",
                    )
            axis.set(
                title=f"{model_name}\n{label}: {strategy}",
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"],
                xlabel="Predicted",
                ylabel="Actual",
            )
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Count (row % shown)")
    fig.suptitle(f"Held-out test confusion matrices (threshold {DEFAULT_THRESHOLD:.2f})")
    fig.savefig(MODEL_OUTPUT_DIR / "test_confusion_matrix_grid.png", dpi=160)
    plt.close(fig)


def plot_curves(y_test, test_scores, best_per_model):
    """Test PR and ROC curves for each model's baseline and best strategy."""
    fig, (pr_axis, roc_axis) = plt.subplots(1, 2, figsize=(15, 6))
    model_colors = dict(zip(MODEL_NAMES, ["#4E7891", "#147D76", "#D97757"]))
    for model_name in MODEL_NAMES:
        best_strategy = best_per_model.set_index("model").loc[
            model_name, "imbalance_strategy"
        ]
        for strategy, style in ((BASELINE, "--"), (best_strategy, "-")):
            scores = test_scores[(model_name, strategy)]
            precision, recall, _ = precision_recall_curve(y_test, scores)
            fpr, tpr, _ = roc_curve(y_test, scores)
            label = f"{model_name} – {strategy}"
            pr_axis.plot(
                recall, precision, style, color=model_colors[model_name],
                label=f"{label} (AP={average_precision_score(y_test, scores):.3f})",
            )
            roc_axis.plot(
                fpr, tpr, style, color=model_colors[model_name],
                label=f"{label} (AUC={roc_auc_score(y_test, scores):.3f})",
            )
    pr_axis.axhline(y_test.mean(), color="#777777", linestyle=":", label="No skill")
    roc_axis.plot([0, 1], [0, 1], ":", color="#777777", label="No skill")
    pr_axis.set(title="Precision-recall curves (test)", xlabel="Recall",
                ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    roc_axis.set(title="ROC curves (test)", xlabel="False-positive rate",
                 ylabel="True-positive rate (recall)", xlim=(0, 1), ylim=(0, 1))
    pr_axis.legend(fontsize=7, loc="upper right")
    roc_axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(MODEL_OUTPUT_DIR / "test_pr_roc_curves.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Steps 2-5 orchestrated
# ---------------------------------------------------------------------------


def run_model_comparison(df, eda_findings=()):
    """Split, cross-validate, select, refit, and evaluate on the test set."""
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = [
        column
        for column in df.columns
        if column != TARGET and column not in IDENTIFIER_COLUMNS
    ]
    numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df[feature_columns].select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    unhandled = set(feature_columns) - set(numeric_columns) - set(categorical_columns)
    if unhandled:
        raise ValueError(f"Features have unsupported data types: {sorted(unhandled)}")

    x_development, x_test, y_development, y_test = train_test_split(
        df[feature_columns],
        df[TARGET].astype(int),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    ratio = (y_development == 0).sum() / (y_development == 1).sum()
    print("\n=== Imbalanced classification: model and strategy comparison ===")
    print(
        f"Development rows: {len(x_development):,}; held-out test rows: {len(x_test):,}; "
        f"legitimate:fraud = {ratio:.1f}:1 (fraud {y_development.mean():.2%})."
    )
    print(
        "Resampling is applied inside each training fold only; validation and "
        "test rows keep their natural class balance.\n"
        f"Running {CV_SPLITS}-fold stratified CV repeated {CV_REPEATS} times "
        f"for {len(MODEL_NAMES)} models x {len(STRATEGY_NAMES)} strategies...",
        flush=True,
    )

    # ---- Cross-validation ------------------------------------------------
    fold_results = run_cross_validation(
        x_development, y_development, numeric_columns, categorical_columns
    )
    cv_summary = summarise_cv(fold_results)
    paired = paired_change_vs_baseline(fold_results)
    best_per_model, overall = select_candidates(cv_summary)
    selected_model = overall["model"]
    selected_strategy = overall["imbalance_strategy"]

    fold_results.to_csv(MODEL_OUTPUT_DIR / "cv_fold_results.csv", index=False)
    cv_summary.to_csv(MODEL_OUTPUT_DIR / "cv_summary.csv", index=False)
    paired.to_csv(MODEL_OUTPUT_DIR / "cv_change_vs_no_resampling.csv", index=False)
    best_per_model.to_csv(MODEL_OUTPUT_DIR / "cv_best_strategy_per_model.csv", index=False)

    shown = ["precision", "recall", "f1_score", "false_positive_rate", "pr_auc", "roc_auc"]
    printable = cv_summary[["model", "imbalance_strategy"]].copy()
    for metric in shown:
        printable[METRIC_LABELS[metric]] = [
            f"{m:.3f} ± {s:.3f}"
            for m, s in zip(cv_summary[f"{metric}_mean"], cv_summary[f"{metric}_std"])
        ]
    print(f"\n--- CV results at threshold {DEFAULT_THRESHOLD:.2f} (mean ± std) ---")
    print(printable.to_string(index=False))

    print("\n--- Paired change vs. same model without resampling (CV folds) ---")
    change_table = paired[["model", "imbalance_strategy"]].copy()
    for metric in ("recall", "f1_score", "precision", "false_positive_rate"):
        change_table[f"Δ {METRIC_LABELS[metric]}"] = [
            f"{m:+.3f} ({k} folds better)"
            for m, k in zip(
                paired[f"delta_{metric}_mean"], paired[f"{metric}_folds_improved"]
            )
        ]
    print(change_table.to_string(index=False))

    print("\n--- Best imbalance strategy per model (highest CV F1) ---")
    print(
        best_per_model[
            ["model", "imbalance_strategy", "f1_score_mean", "recall_mean",
             "precision_mean", "false_positive_rate_mean", "pr_auc_mean"]
        ].round(3).to_string(index=False)
    )
    print(
        f"\nSelected overall: {selected_model} + {selected_strategy} "
        f"(CV F1 {overall['f1_score_mean']:.3f} ± {overall['f1_score_std']:.3f})",
        flush=True,
    )

    # ---- Held-out test set (post-selection evaluation) -------------------------
    print(
        "\nSelection is complete. Refitting all 15 candidates on the full development "
        "set and scoring the held-out test set for reporting only...",
        flush=True,
    )
    test_scores = fit_and_score_on_test(
        x_development, y_development, x_test, numeric_columns, categorical_columns
    )
    test_results = pd.DataFrame(
        [
            {
                "model": model_name,
                "imbalance_strategy": strategy_name,
                "selected": (model_name, strategy_name)
                == (selected_model, selected_strategy),
                **evaluate_scores(y_test, scores),
            }
            for (model_name, strategy_name), scores in test_scores.items()
        ]
    )
    test_results.to_csv(MODEL_OUTPUT_DIR / "test_results_all_candidates.csv", index=False)

    print(f"\n--- Held-out test results at threshold {DEFAULT_THRESHOLD:.2f} ---")
    print(
        test_results[["model", "imbalance_strategy", "precision", "recall", "f1_score",
                      "false_positive_rate", "pr_auc", "roc_auc", "true_positive",
                      "false_positive"]]
        .round(3).to_string(index=False)
    )

    selected_scores = test_scores[(selected_model, selected_strategy)]
    baseline_scores = test_scores[(selected_model, BASELINE)]
    intervals = bootstrap_intervals(y_test, selected_scores, baseline_scores)
    selected_row = test_results[test_results["selected"]].iloc[0]
    baseline_row = test_results[
        (test_results["model"] == selected_model)
        & (test_results["imbalance_strategy"] == BASELINE)
    ].iloc[0]
    intervals.insert(1, "selected", [selected_row[m] for m in METRICS])
    intervals.insert(2, "baseline", [baseline_row[m] for m in METRICS])
    intervals.insert(3, "delta", intervals["selected"] - intervals["baseline"])
    intervals.to_csv(MODEL_OUTPUT_DIR / "selected_model_test_metrics.csv", index=False)

    print(
        f"\n--- Selected model on the test set: {selected_model} + {selected_strategy} "
        f"vs. {selected_model} without resampling (95% stratified bootstrap CI) ---"
    )
    for _, row in intervals.iterrows():
        print(
            f"  {METRIC_LABELS[row['metric']]:<20} {row['selected']:.3f} "
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]  baseline {row['baseline']:.3f}  "
            f"change {row['delta']:+.3f} [{row['delta_vs_baseline_ci_lower']:+.3f}, "
            f"{row['delta_vs_baseline_ci_upper']:+.3f}]",
            flush=True,
        )

    # ---- Plots and report -----------------------------------------------------
    plot_cv_strategy_comparison(cv_summary)
    plot_confusion_matrix_grid(y_test, test_scores, best_per_model)
    plot_curves(y_test, test_scores, best_per_model)
    write_report(
        eda_findings, y_development, y_test, cv_summary, paired, best_per_model,
        overall, intervals, test_results,
    )
    print(f"\nModel tables, plots and report saved to: {MODEL_OUTPUT_DIR}", flush=True)
    return {"model": selected_model, "strategy": selected_strategy, "test": intervals}


def describe_change(metric_row, label):
    """Describe a bootstrap change interval in words."""
    change = metric_row["delta"]
    lower = metric_row["delta_vs_baseline_ci_lower"]
    upper = metric_row["delta_vs_baseline_ci_upper"]
    if lower > 0:
        verdict = "a clear increase (interval excludes zero)"
    elif upper < 0:
        verdict = "a clear decrease (interval excludes zero)"
    else:
        verdict = "no clear change (interval includes zero)"
    return f"{label} {change:+.3f} (95% CI {lower:+.3f} to {upper:+.3f}): {verdict}"


def write_report(
    eda_findings, y_development, y_test, cv_summary, paired, best_per_model, overall,
    intervals, test_results,
):
    """Write a plain-text report that mirrors the proposal's structure."""
    selected_model = overall["model"]
    selected_strategy = overall["imbalance_strategy"]
    metric_row = intervals.set_index("metric")
    lines = [
        "VEHICLE INSURANCE FRAUD DETECTION - IMBALANCED CLASSIFICATION REPORT",
        "=" * 68,
        "",
        "1. Exploratory data analysis",
        *[f"   - {line}" for line in eda_findings],
        "",
        "2. Evaluation design",
        f"   - Stratified 80/20 split: development set {len(y_development):,} claims "
        f"({y_development.mean():.2%} fraud), held-out test set {len(y_test):,} claims "
        f"({y_test.mean():.2%} fraud).",
        "   - Candidates: Logistic Regression, Decision Tree and Random Forest, each "
        "with no resampling (baseline), class weighting (class_weight='balanced'), "
        "random oversampling, SMOTE and random undersampling: 15 candidates in total.",
        "   - SMOTE is implemented using SMOTENC because the data mix numeric and "
        "categorical features. Plain SMOTE interpolates every feature numerically, "
        "which would create invalid in-between category codes; SMOTENC interpolates "
        "the numeric features and assigns each categorical feature the most frequent "
        "value among the minority-class nearest neighbours.",
        f"   - Model comparison uses {CV_SPLITS}-fold stratified cross-validation "
        f"repeated {CV_REPEATS} times on the development set. Imputation, scaling, "
        "encoding and all resampling are fitted inside each training fold only, so "
        "no synthetic or duplicated rows reach a validation fold (no data leakage).",
        f"   - Selection criterion: highest mean CV fraud-class F1 at threshold "
        f"{DEFAULT_THRESHOLD:.2f}, PR-AUC as tie-break. The false-positive rate is "
        "reported for every candidate.",
        "   - The test set is reserved for post-selection evaluation. Once CV "
        "selection was complete, all 15 candidates were refit on the full "
        "development set and scored on the test set for reporting. Test results "
        "were not used to choose any model or strategy.",
        "",
        "3. Effect of imbalance handling (CV, paired change vs. same model without "
        f"resampling; folds improved out of {CV_SPLITS * CV_REPEATS})",
    ]
    for _, row in paired.iterrows():
        lines.append(
            f"   {row['model']:<20} {row['imbalance_strategy']:<22} "
            f"recall {row['delta_recall_mean']:+.3f} ({row['recall_folds_improved']}), "
            f"F1 {row['delta_f1_score_mean']:+.3f} ({row['f1_score_folds_improved']}), "
            f"FPR {row['delta_false_positive_rate_mean']:+.3f}, "
            f"PR-AUC {row['delta_pr_auc_mean']:+.3f} ({row['pr_auc_folds_improved']}), "
            f"ROC-AUC {row['delta_roc_auc_mean']:+.3f} ({row['roc_auc_folds_improved']})"
        )
    lines += ["", "4. Best imbalance strategy per model (CV mean at threshold 0.50)"]
    for _, row in best_per_model.iterrows():
        lines.append(
            f"   {row['model']:<20} {row['imbalance_strategy']:<22} "
            f"F1 {row['f1_score_mean']:.3f}, recall {row['recall_mean']:.3f}, "
            f"precision {row['precision_mean']:.3f}, FPR "
            f"{row['false_positive_rate_mean']:.3f}, PR-AUC {row['pr_auc_mean']:.3f}, "
            f"ROC-AUC {row['roc_auc_mean']:.3f}"
        )
    smote_rows = cv_summary[cv_summary["imbalance_strategy"] == SMOTE_NAME]
    base_rows = cv_summary[cv_summary["imbalance_strategy"] == BASELINE].set_index("model")
    lines += ["", "5. SMOTE (via SMOTENC) vs. no resampling (CV mean at threshold 0.50)"]
    for _, row in smote_rows.iterrows():
        base = base_rows.loc[row["model"]]
        lines.append(
            f"   {row['model']:<20} recall {base['recall_mean']:.3f} -> "
            f"{row['recall_mean']:.3f}, F1 {base['f1_score_mean']:.3f} -> "
            f"{row['f1_score_mean']:.3f}, FPR {base['false_positive_rate_mean']:.3f} -> "
            f"{row['false_positive_rate_mean']:.3f}, PR-AUC {base['pr_auc_mean']:.3f} -> "
            f"{row['pr_auc_mean']:.3f}"
        )
    sel = test_results[test_results["selected"]].iloc[0]
    lines += [
        "",
        f"6. Selected model: {selected_model} + {selected_strategy}",
        f"   CV F1 {overall['f1_score_mean']:.3f} ± {overall['f1_score_std']:.3f}, "
        f"recall {overall['recall_mean']:.3f}, precision {overall['precision_mean']:.3f}, "
        f"FPR {overall['false_positive_rate_mean']:.3f}.",
        f"   Held-out test (threshold {DEFAULT_THRESHOLD:.2f}), with 95% stratified "
        f"bootstrap CI ({BOOTSTRAP_REPLICATES:,} resamples) and change vs. "
        f"{selected_model} without resampling:",
    ]
    for metric in METRICS:
        row = metric_row.loc[metric]
        lines.append(
            f"     {METRIC_LABELS[metric]:<20} {row['selected']:.3f} "
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]   baseline "
            f"{row['baseline']:.3f}   change {row['delta']:+.3f} "
            f"[{row['delta_vs_baseline_ci_lower']:+.3f}, "
            f"{row['delta_vs_baseline_ci_upper']:+.3f}]"
        )
    lines += [
        f"   Test confusion matrix: TP {sel['true_positive']}, FN {sel['false_negative']}, "
        f"FP {sel['false_positive']}, TN {sel['true_negative']}.",
        "",
        "7. Interpretation",
        "   - Classification at the 0.50 threshold improves: "
        + "; ".join(
            describe_change(metric_row.loc[metric], METRIC_LABELS[metric])
            for metric in ("recall", "f1_score")
        )
        + ".",
        f"   - Ranking ability does not clearly improve: "
        f"{describe_change(metric_row.loc['pr_auc'], 'PR-AUC')}; "
        f"{describe_change(metric_row.loc['roc_auc'], 'ROC-AUC')}.",
    ]
    negative_pr = paired[paired["delta_pr_auc_mean"] < 0]
    lines += [
        f"   - Across CV, {len(negative_pr)} of {len(paired)} model/strategy pairs have a "
        "lower mean PR-AUC than the same model without resampling.",
        "   - Associations from the EDA are descriptive and do not imply causation.",
        "",
    ]
    (MODEL_OUTPUT_DIR / "model_selection_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Step 1: exploratory data analysis
# ---------------------------------------------------------------------------


def summarise_eda_findings(
    df, overview, cleaning_counts, duplicate_count, associations,
    numeric_associations, fraud_rates, cramer_matrix, age_result,
):
    """Turn the EDA tables into plain-language findings for the report."""
    n_rows = len(df)
    features = [c for c in df.columns if c != TARGET and c not in IDENTIFIER_COLUMNS]
    numeric = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    fraud = int(df[TARGET].sum())
    legitimate = n_rows - fraud
    findings = [
        f"Dataset: {n_rows:,} claims with {len(features)} model features "
        f"({len(numeric)} numeric, {len(categorical)} categorical, including the "
        "derived AgeWasZero flag); PolicyNumber is an identifier and is excluded."
    ]
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().astype(str).unique())
        findings.append(f"Claim years covered: {', '.join(years)}.")
    findings.append(
        f"Class imbalance: {legitimate:,} legitimate ({legitimate / n_rows:.2%}) vs "
        f"{fraud:,} fraudulent ({fraud / n_rows:.2%}) claims, a ratio of "
        f"{legitimate / fraud:.1f}:1. Always predicting 'legitimate' would be "
        f"{legitimate / n_rows:.1%} accurate while catching no fraud, so accuracy is "
        f"not used; the no-skill PR-AUC is {fraud / n_rows:.3f}."
    )
    quality = [
        f"{label.split(' (')[0]}: {count:,} claim{'s' if count != 1 else ''}"
        for label, count in cleaning_counts.items()
    ]
    findings.append(
        "Data quality: " + "; ".join(quality) + " - these placeholder values were set "
        f"to missing (Age=0 is also kept as the AgeWasZero flag). Exact duplicate "
        f"rows: {duplicate_count}."
    )
    missing = overview[(overview["missing"] > 0) & (overview.index != TARGET)]
    if not missing.empty:
        findings.append(
            "Missing values after cleaning: "
            + ", ".join(
                f"{name} {int(row['missing']):,} ({row['missing_pct']:.2f}%)"
                for name, row in missing.iterrows()
            )
            + ". They are imputed inside each training fold (median for numeric, most "
            "frequent level for categorical), so validation and test rows never "
            "inform the imputation."
        )
    top = associations.head(5)
    findings.append(
        "Strongest categorical associations with fraud (bias-corrected Cramer's V): "
        + "; ".join(
            f"{row['feature']} V={row['cramers_v']:.3f}"
            + (f", BH q={row['q_value_bh']:.1e}" if pd.notna(row["q_value_bh"])
               else ", chi-square not reported (sparse cells)")
            for _, row in top.iterrows()
        )
        + ". All associations are weak to moderate (V < 0.2), so no single feature "
        "separates fraud from legitimate claims."
    )
    if not numeric_associations.empty:
        findings.append(
            "Numeric features are only weakly related to fraud (Spearman rho): "
            + "; ".join(
                f"{row['feature']} {row['spearman_rho']:+.3f} (q={row['q_value_bh']:.3g})"
                for _, row in numeric_associations.iterrows()
            )
            + "."
        )
    if not fraud_rates.empty:
        comparisons = []
        for feature, group in fraud_rates[fraud_rates["feature"] != "Age band"].groupby(
            "feature", sort=False
        ):
            group = group[~group["level"].astype(str).str.startswith("Other (<")]
            if len(group) < 2:
                continue
            high = group.loc[group["fraud_rate"].idxmax()]
            low = group.loc[group["fraud_rate"].idxmin()]
            comparisons.append(
                f"{feature}: {high['level']} {high['fraud_rate']:.1f}% "
                f"(n={int(high['records']):,}) vs {low['level']} {low['fraud_rate']:.1f}% "
                f"(n={int(low['records']):,})"
            )
        findings.append(
            "Fraud-rate comparisons (highest vs lowest level, overall rate "
            f"{fraud / n_rows:.1%}): " + "; ".join(comparisons) + "."
        )
    if age_result:
        bands = fraud_rates[fraud_rates["feature"] == "Age band"]
        findings.append(
            f"Age: fraudulent claimants are slightly younger (median "
            f"{age_result['fraud_median']:.0f} vs {age_result['legit_median']:.0f}, "
            f"Mann-Whitney p={age_result['p_value']:.3g}); fraud rate by age band "
            f"ranges from {bands['fraud_rate'].min():.1f}% to "
            f"{bands['fraud_rate'].max():.1f}%, a small effect."
        )
    pairs = []
    columns = [c for c in cramer_matrix.columns if c != TARGET]
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            if {left, right} == {"Age", "AgeWasZero"}:
                continue  # AgeWasZero is derived from Age by construction
            value = cramer_matrix.loc[left, right]
            if value >= REDUNDANCY_CRAMERS_V:
                pairs.append((value, left, right))
    if pairs:
        findings.append(
            f"Redundant features (Cramer's V >= {REDUNDANCY_CRAMERS_V}): "
            + "; ".join(f"{l}-{r} {v:.2f}" for v, l, r in sorted(pairs, reverse=True))
            + ". PolicyType is effectively VehicleCategory combined with BasePolicy. "
            "All are kept: tree models are unaffected, but individual Logistic "
            "Regression coefficients for these features should not be interpreted."
        )
    findings.append(
        "These associations are descriptive and do not imply causation; the models "
        "use all features jointly."
    )
    return findings


def run_eda(df, data_path, cleaning_counts=None):
    """Run exploratory summaries and save tables and figures."""
    cleaning_counts = cleaning_counts or {}
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_features = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column != TARGET and column not in IDENTIFIER_COLUMNS
    ]
    overview = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100),
            "unique": df.nunique(dropna=False),
        }
    )
    overview.to_csv(EDA_OUTPUT_DIR / "column_overview.csv")

    duplicate_mask = df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore").duplicated()
    duplicate_count = int(duplicate_mask.sum())
    pd.DataFrame(
        [
            {
                "rows_before_duplicate_removal": len(df),
                "exact_duplicates_excluding_identifiers": duplicate_count,
            }
        ]
    ).to_csv(EDA_OUTPUT_DIR / "data_quality_summary.csv", index=False)
    if duplicate_count:
        df = df.loc[~duplicate_mask].copy()
        print(f"Removed {duplicate_count:,} duplicate feature rows before splitting.")
    else:
        print("Duplicate feature rows excluding identifiers: 0.")

    target_counts = df[TARGET].value_counts().sort_index()
    fraud_count = int(target_counts.get(1, 0))
    legitimate_count = int(target_counts.get(0, 0))
    imbalance_ratio = legitimate_count / fraud_count if fraud_count else float("inf")
    print(
        f"\n=== EDA: {len(df):,} rows, {len(df.columns)} columns ===\n"
        f"Class imbalance: {legitimate_count:,} legitimate vs {fraud_count:,} fraud "
        f"({fraud_count / len(df):.2%} fraud, {imbalance_ratio:.1f}:1)."
    )
    print(f"Input: {data_path}")
    missing = overview[overview["missing"] > 0][["missing", "missing_pct"]]
    print("Missing values after cleaning (imputed inside each training fold):")
    print(missing.round(2).to_string() if not missing.empty else "  none")

    distribution_rows = []
    for column in categorical_columns:
        counts = df[column].value_counts(dropna=False)
        distribution_rows.extend(
            {
                "feature": column,
                "level": str(level),
                "count": int(count),
                "fraction": count / len(df),
            }
            for level, count in counts.items()
        )
    pd.DataFrame(distribution_rows).to_csv(
        EDA_OUTPUT_DIR / "categorical_distributions.csv", index=False
    )

    numeric_rows = []
    for column in numeric_features:
        if df[column].nunique() > 1:
            paired = df[[column, TARGET]].dropna()
            rho, p_value = spearmanr(paired[column], paired[TARGET])
            numeric_rows.append({"feature": column, "spearman_rho": rho, "p_value": p_value})
    numeric_associations = pd.DataFrame(numeric_rows)
    if not numeric_associations.empty:
        numeric_associations["q_value_bh"] = adjust_p_values(numeric_associations["p_value"])
        numeric_associations = numeric_associations.sort_values(
            "spearman_rho", key=lambda values: values.abs(), ascending=False
        )
    numeric_associations.to_csv(EDA_OUTPUT_DIR / "numeric_fraud_associations.csv", index=False)

    association_rows = []
    for column in categorical_columns:
        contingency = pd.crosstab(df[column].astype("string").fillna("Missing"), df[TARGET])
        if min(contingency.shape) > 1:
            _, p_value, _, expected = chi2_contingency(contingency)
            status = "tested" if (expected >= 5).all() else "not tested: expected count < 5"
            association_rows.append(
                {
                    "feature": column,
                    "levels": contingency.shape[0],
                    "cramers_v": cramers_v(contingency),
                    "p_value": p_value if status == "tested" else np.nan,
                    "test_status": status,
                }
            )
    associations = pd.DataFrame(association_rows)
    if not associations.empty:
        associations["q_value_bh"] = np.nan
        tested = associations["p_value"].notna()
        associations.loc[tested, "q_value_bh"] = adjust_p_values(
            associations.loc[tested, "p_value"]
        )
        associations = associations.sort_values("cramers_v", ascending=False)
    associations.to_csv(EDA_OUTPUT_DIR / "categorical_fraud_associations.csv", index=False)
    print("\nStrongest categorical associations with fraud (bias-corrected Cramer's V):")
    print(associations.head(8)[["feature", "levels", "cramers_v"]].round(3).to_string(index=False))

    # Association ("correlation") matrix over ALL features and the target, using
    # bias-corrected Cramer's V. Numeric features are discrete here except Age,
    # which is grouped into the shared age bands for this matrix only.
    matrix_columns = [
        column for column in df.columns if column not in IDENTIFIER_COLUMNS
    ]
    as_levels = {}
    for column in matrix_columns:
        values = age_bands(df[column]) if column == "Age" else df[column]
        as_levels[column] = values.astype("string").fillna("Missing")
    # Order: target first, then features by their association with fraud.
    target_association = {
        column: cramers_v(pd.crosstab(as_levels[column], as_levels[TARGET]))
        for column in matrix_columns
        if column != TARGET
    }
    ordered = [TARGET] + sorted(
        target_association, key=target_association.get, reverse=True
    )
    cramer_matrix = pd.DataFrame(np.eye(len(ordered)), index=ordered, columns=ordered)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            value = cramers_v(pd.crosstab(as_levels[left], as_levels[right]))
            cramer_matrix.loc[left, right] = value
            cramer_matrix.loc[right, left] = value
    cramer_matrix.to_csv(EDA_OUTPUT_DIR / "feature_association_matrix.csv")
    labels = [f"{c} (target)" if c == TARGET else c for c in ordered]
    fig, axis = plt.subplots(figsize=(12, 10.5))
    image = axis.imshow(cramer_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    axis.set_xticks(range(len(ordered)), labels, rotation=90, fontsize=8)
    axis.set_yticks(range(len(ordered)), labels, fontsize=8)
    axis.set_title(
        "Correlation heatmap of all features and the target\n"
        "(bias-corrected Cramer's V; features ordered by association with fraud)"
    )
    fig.colorbar(image, ax=axis, label="Cramer's V", shrink=0.8)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "feature_correlation_heatmap.png", dpi=170)
    plt.close(fig)

    rate_summaries = []
    age_result = None
    for column in RATE_FEATURES:
        if column in df.columns:
            values = group_rare_categories(df[column])
            summary = (
                pd.DataFrame({"level": values, TARGET: df[TARGET]})
                .groupby("level", observed=True)[TARGET]
                .agg(records="size", fraud_count="sum", fraud_rate="mean")
                .reset_index()
            )
            summary.insert(0, "feature", column)
            rate_summaries.append(summary)
    if "Age" in df.columns:
        summary = (
            pd.DataFrame({"level": age_bands(df["Age"]), TARGET: df[TARGET]})
            .dropna(subset=["level"])
            .groupby("level", observed=True)[TARGET]
            .agg(records="size", fraud_count="sum", fraud_rate="mean")
            .reset_index()
        )
        summary.insert(0, "feature", "Age band")
        rate_summaries.append(summary)
        fraud_ages = df.loc[df[TARGET] == 1, "Age"].dropna()
        clean_ages = df.loc[df[TARGET] == 0, "Age"].dropna()
        _, p_value = mannwhitneyu(fraud_ages, clean_ages, alternative="two-sided")
        age_result = {
            "fraud_median": fraud_ages.median(),
            "legit_median": clean_ages.median(),
            "p_value": p_value,
        }
        print(
            f"\nAge comparison (Mann-Whitney U): fraud median={fraud_ages.median():.1f}, "
            f"non-fraud median={clean_ages.median():.1f}, p={p_value:.4g}."
        )
    fraud_rates = pd.DataFrame()
    if rate_summaries:
        fraud_rates = pd.concat(rate_summaries, ignore_index=True)
        fraud_rates["fraud_rate"] *= 100
        fraud_rates.to_csv(EDA_OUTPUT_DIR / "selected_fraud_rates.csv", index=False)

    # Three key EDA plots: class imbalance, which features relate to fraud, and
    # how the fraud rate differs across the levels of the strongest features.
    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(2, 2, width_ratios=[0.75, 1.25], height_ratios=[1, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    plt.rcParams.update({"font.size": 12})
    counts = target_counts.reindex([0, 1], fill_value=0)
    axes[0].bar(["Legitimate", "Fraud"], counts.values, color=["#4E7891", "#D97757"])
    for x, count in enumerate(counts.values):
        axes[0].text(x, count, f"{count:,}\n({count / len(df):.1%})", ha="center",
                     va="bottom", fontsize=12)
    axes[0].set(
        title=f"(a) Class imbalance ({imbalance_ratio:.1f}:1)",
        ylabel="Number of claims",
        ylim=(0, counts.max() * 1.15),
    )

    top = (
        pd.Series({c: cramer_matrix.loc[c, TARGET] for c in cramer_matrix.index if c != TARGET})
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )
    axes[1].barh(top.index, top.values, color="#4E7891")
    for y, value in enumerate(top.values):
        axes[1].text(value + 0.003, y, f"{value:.3f}", va="center", fontsize=11)
    axes[1].set(
        title="(b) Top 10 features associated with fraud",
        xlabel="Cramer's V with FraudFound_P",
        xlim=(0, top.max() * 1.2),
    )

    key_features = [f for f in ("Fault", "BasePolicy", "VehicleCategory") if f in df.columns]
    bar_labels, bar_rates, bar_colors = [], [], []
    palette = ["#4E7891", "#147D76", "#D97757"]
    for color, feature in zip(palette, key_features):
        rates = df.groupby(feature)[TARGET].mean().mul(100).sort_values(ascending=False)
        for level, rate in rates.items():
            bar_labels.append(f"{feature}: {level}")
            bar_rates.append(rate)
            bar_colors.append(color)
    positions = np.arange(len(bar_labels))[::-1]
    axes[2].barh(positions, bar_rates, color=bar_colors)
    axes[2].set_yticks(positions, bar_labels)
    for y, rate in zip(positions, bar_rates):
        axes[2].text(rate + 0.1, y, f"{rate:.1f}%", va="center", fontsize=11)
    overall_rate = df[TARGET].mean() * 100
    axes[2].axvline(overall_rate, color="#777777", linestyle="--", linewidth=1)
    axes[2].text(overall_rate + 0.1, positions.max() + 0.55, f"overall {overall_rate:.1f}%",
                 color="#555555", fontsize=11)
    axes[2].set(
        title="(c) Fraud rate by level of the strongest features",
        xlabel="Fraud rate (%)",
        xlim=(0, max(bar_rates) * 1.2),
        ylim=(-0.7, len(bar_labels) - 0.1),
    )
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "eda_key_plots.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})
    findings = summarise_eda_findings(
        df, overview, cleaning_counts, duplicate_count, associations,
        numeric_associations, fraud_rates, cramer_matrix, age_result,
    )
    (EDA_OUTPUT_DIR / "eda_findings.txt").write_text(
        "EDA FINDINGS\n============\n\n"
        + "\n\n".join(f"- {line}" for line in findings) + "\n",
        encoding="utf-8",
    )
    print("\n--- EDA findings ---")
    print("\n".join(f"- {line}" for line in findings))
    print(f"EDA tables and figures saved to: {EDA_OUTPUT_DIR}")
    return df, findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the vehicle insurance CSV. If omitted, the script looks for "
        "vehicle_insurance.csv or fraud_oracle.csv next to the script, in the "
        "current folder, or in a data/ folder.",
    )
    return parser.parse_args()


def find_dataset(explicit_path=None):
    """Return the dataset path, or stop with clear instructions if it is missing."""
    if explicit_path is not None:
        candidates = [Path(explicit_path).expanduser()]
    else:
        candidates = [folder / name for folder in DATA_SEARCH_DIRS for name in DATA_FILE_NAMES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(c) for c in dict.fromkeys(candidates))
    raise SystemExit(
        "Dataset not found. Looked for:\n  " + searched + "\n\n"
        f"Download fraud_oracle.csv from {KAGGLE_URL}\n"
        "and place it next to New_Insurance_C.py (it may keep its original name "
        "or be renamed to vehicle_insurance.csv),\n"
        "or run: python New_Insurance_C.py --data-path path/to/fraud_oracle.csv"
    )


def main():
    """Run data preparation, EDA, model comparison, and final evaluation."""
    args = parse_args()
    data_path = find_dataset(args.data_path)

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is missing from the dataset")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(f"Expected {TARGET!r} to contain exactly the labels 0 and 1")
    df, cleaning_counts = clean_data(df)
    print("=== Data-quality corrections ===")
    for correction, count in cleaning_counts.items():
        print(f"{correction}: {count:,}")
    df, eda_findings = run_eda(df, data_path, cleaning_counts)
    run_model_comparison(df, eda_findings)


if __name__ == "__main__":
    main()
