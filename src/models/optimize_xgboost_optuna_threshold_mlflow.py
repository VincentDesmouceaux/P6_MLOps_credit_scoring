from pathlib import Path
import gc
import re
from collections import Counter

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config import PROCESSED_DATA_DIR
from src.models.mlflow_utils import get_git_branch_name, get_git_commit_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_MODELING_FILE = "train_modeling.csv"
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

EXPERIMENT_NAME = "P6_credit_scoring_optimization"

RANDOM_STATE = 42
N_SPLITS = 3
N_TRIALS = 15

FN_COST = 10
FP_COST = 1

USE_SAMPLE = True
SAMPLE_SIZE = 80000

TRAIN_FINAL_MODEL_ON_FULL_DATA = True

THRESHOLDS = np.arange(0.10, 0.91, 0.05)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "optimization"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    file_path = PROCESSED_DATA_DIR / TRAIN_MODELING_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"Chargement : {file_path}")
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])

    return X, y


def clean_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    cleaned_columns = []
    counts = Counter()

    for column in X.columns:
        clean_column = re.sub(r"[^A-Za-z0-9_]+", "_", str(column))
        clean_column = re.sub(r"_+", "_", clean_column).strip("_")

        if not clean_column:
            clean_column = "feature"

        counts[clean_column] += 1

        if counts[clean_column] > 1:
            clean_column = f"{clean_column}_{counts[clean_column]}"

        cleaned_columns.append(clean_column)

    X.columns = cleaned_columns
    return X


def stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size >= len(X):
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def compute_business_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    business_cost = (FN_COST * fn) + (FP_COST * fp)
    business_cost_per_client = business_cost / len(y_true)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost": float(business_cost),
        "business_cost_per_client": float(business_cost_per_client),
    }


def build_xgboost_params_from_trial(
    trial: optuna.Trial,
    base_scale_pos_weight: float,
) -> dict:
    scale_pos_weight_multiplier = trial.suggest_float(
        "scale_pos_weight_multiplier",
        0.70,
        1.50,
    )

    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True,
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 20, 120),
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "scale_pos_weight": base_scale_pos_weight * scale_pos_weight_multiplier,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def build_xgboost_params_from_best_params(
    best_params: dict,
    base_scale_pos_weight: float,
) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": best_params["n_estimators"],
        "learning_rate": best_params["learning_rate"],
        "max_depth": best_params["max_depth"],
        "min_child_weight": best_params["min_child_weight"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "gamma": best_params["gamma"],
        "reg_alpha": best_params["reg_alpha"],
        "reg_lambda": best_params["reg_lambda"],
        "scale_pos_weight": (
            base_scale_pos_weight * best_params["scale_pos_weight_multiplier"]
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def evaluate_params_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict,
    verbose: bool = False,
) -> pd.DataFrame:
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_results = []

    for fold_index, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        if verbose:
            print(f"Fold {fold_index}/{N_SPLITS}")

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_valid)[:, 1]

        for threshold in THRESHOLDS:
            metrics = compute_business_metrics(
                y_true=y_valid,
                y_proba=y_proba,
                threshold=float(threshold),
            )

            metrics["fold"] = fold_index
            metrics["threshold"] = float(threshold)

            all_results.append(metrics)

        del model
        gc.collect()

    return pd.DataFrame(all_results)


def aggregate_threshold_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metrics_to_aggregate = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "business_cost",
        "business_cost_per_client",
    ]

    aggregation = {}

    for metric in metrics_to_aggregate:
        aggregation[f"mean_{metric}"] = (metric, "mean")
        aggregation[f"std_{metric}"] = (metric, "std")

    aggregated_df = (
        results_df
        .groupby("threshold")
        .agg(**aggregation)
        .reset_index()
        .sort_values(by="mean_business_cost", ascending=True)
    )

    return aggregated_df


def plot_cost_vs_threshold(
    aggregated_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = aggregated_df.sort_values(by="threshold")

    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_df["threshold"],
        plot_df["mean_business_cost"],
        marker="o",
    )
    plt.xlabel("Seuil de décision")
    plt.ylabel("Coût métier moyen")
    plt.title("Coût métier moyen selon le seuil de décision - XGBoost optimisé")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset()

    X_full, y_full = split_features_target(df)
    X_full = clean_feature_names(X_full)

    print(f"Dataset complet : {X_full.shape}")
    print(f"Taux de défaut complet : {y_full.mean():.4f}")

    if USE_SAMPLE:
        X_opt, y_opt = stratified_sample(
            X=X_full,
            y=y_full,
            sample_size=SAMPLE_SIZE,
        )
    else:
        X_opt, y_opt = X_full, y_full

    print(f"Dataset utilisé pour Optuna : {X_opt.shape}")
    print(f"Taux de défaut Optuna : {y_opt.mean():.4f}")

    negative_count = (y_opt == 0).sum()
    positive_count = (y_opt == 1).sum()
    base_scale_pos_weight = negative_count / positive_count

    print(f"base_scale_pos_weight : {base_scale_pos_weight:.4f}")

    def objective(trial: optuna.Trial) -> float:
        model_params = build_xgboost_params_from_trial(
            trial=trial,
            base_scale_pos_weight=base_scale_pos_weight,
        )

        results_df = evaluate_params_with_cv(
            X=X_opt,
            y=y_opt,
            model_params=model_params,
            verbose=False,
        )

        aggregated_df = aggregate_threshold_results(results_df)
        best_row = aggregated_df.iloc[0]

        trial.set_user_attr("best_threshold", float(best_row["threshold"]))
        trial.set_user_attr(
            "best_mean_business_cost",
            float(best_row["mean_business_cost"]),
        )
        trial.set_user_attr("best_mean_recall", float(best_row["mean_recall"]))
        trial.set_user_attr("best_mean_roc_auc", float(best_row["mean_roc_auc"]))
        trial.set_user_attr("best_mean_f1_score", float(best_row["mean_f1_score"]))

        print(
            f"Trial {trial.number} | "
            f"threshold={best_row['threshold']:.2f} | "
            f"cost={best_row['mean_business_cost']:.2f} | "
            f"recall={best_row['mean_recall']:.4f} | "
            f"auc={best_row['mean_roc_auc']:.4f}"
        )

        return float(best_row["mean_business_cost"])

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="xgboost_business_cost_optimization",
    )

    print("\nLancement de l'optimisation Optuna...")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nMeilleurs hyperparamètres Optuna :")
    print(study.best_params)

    best_model_params = build_xgboost_params_from_best_params(
        best_params=study.best_params,
        base_scale_pos_weight=base_scale_pos_weight,
    )

    print("\nRéévaluation du meilleur modèle avec validation croisée...")
    best_cv_results = evaluate_params_with_cv(
        X=X_opt,
        y=y_opt,
        model_params=best_model_params,
        verbose=True,
    )

    best_threshold_curve = aggregate_threshold_results(best_cv_results)
    best_row = best_threshold_curve.iloc[0]

    print("\nMeilleur seuil métier trouvé :")
    print(best_row)

    trials_path = OUTPUT_DIR / "optuna_xgboost_trials.csv"
    cv_results_path = OUTPUT_DIR / "xgboost_optimized_cv_results_by_fold.csv"
    threshold_curve_path = OUTPUT_DIR / "xgboost_optimized_threshold_curve.csv"
    figure_path = FIGURES_DIR / "xgboost_optimized_cost_vs_threshold.png"

    study.trials_dataframe().to_csv(trials_path, index=False)
    best_cv_results.to_csv(cv_results_path, index=False)
    best_threshold_curve.to_csv(threshold_curve_path, index=False)

    plot_cost_vs_threshold(
        aggregated_df=best_threshold_curve,
        output_path=figure_path,
    )

    if TRAIN_FINAL_MODEL_ON_FULL_DATA:
        X_final = X_full
        y_final = y_full
    else:
        X_final = X_opt
        y_final = y_opt

    print("\nEntraînement du modèle final XGBoost optimisé...")
    final_model = XGBClassifier(**best_model_params)
    final_model.fit(X_final, y_final)

    with mlflow.start_run(run_name="xgboost_optuna_threshold_optimized"):
        mlflow.set_tag("project", "P6_MLOps_credit_scoring")
        mlflow.set_tag("step", "hyperparameter_and_threshold_optimization")
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("description", "Optimisation Optuna des hyperparamètres XGBoost et du seuil métier")
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("git_branch", get_git_branch_name())

        mlflow.log_param("optimization_tool", "Optuna")
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("use_sample", USE_SAMPLE)
        mlflow.log_param("sample_size_for_optimization", len(X_opt))
        mlflow.log_param("final_training_size", len(X_final))
        mlflow.log_param("n_features", X_opt.shape[1])
        mlflow.log_param("target_rate", y_opt.mean())
        mlflow.log_param("fn_cost", FN_COST)
        mlflow.log_param("fp_cost", FP_COST)
        mlflow.log_param("base_scale_pos_weight", base_scale_pos_weight)
        mlflow.log_param("best_threshold", float(best_row["threshold"]))
        mlflow.log_param(
            "thresholds_tested",
            ",".join([str(round(t, 2)) for t in THRESHOLDS]),
        )

        for param_name, param_value in best_model_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        mlflow.log_metric(
            "best_mean_business_cost",
            float(best_row["mean_business_cost"]),
        )
        mlflow.log_metric(
            "best_std_business_cost",
            float(best_row["std_business_cost"]),
        )
        mlflow.log_metric(
            "best_mean_business_cost_per_client",
            float(best_row["mean_business_cost_per_client"]),
        )
        mlflow.log_metric("best_mean_accuracy", float(best_row["mean_accuracy"]))
        mlflow.log_metric("best_mean_precision", float(best_row["mean_precision"]))
        mlflow.log_metric("best_mean_recall", float(best_row["mean_recall"]))
        mlflow.log_metric("best_mean_f1_score", float(best_row["mean_f1_score"]))
        mlflow.log_metric("best_mean_roc_auc", float(best_row["mean_roc_auc"]))
        mlflow.log_metric("best_mean_tn", float(best_row["mean_tn"]))
        mlflow.log_metric("best_mean_fp", float(best_row["mean_fp"]))
        mlflow.log_metric("best_mean_fn", float(best_row["mean_fn"]))
        mlflow.log_metric("best_mean_tp", float(best_row["mean_tp"]))

        mlflow.log_artifact(str(trials_path))
        mlflow.log_artifact(str(cv_results_path))
        mlflow.log_artifact(str(threshold_curve_path))
        mlflow.log_artifact(str(figure_path))

        mlflow.xgboost.log_model(
            xgb_model=final_model,
            name="model",
        )

    print("\nOptimisation étape 4 terminée.")
    print(f"Meilleur seuil : {best_row['threshold']:.2f}")
    print(f"Meilleur coût métier moyen : {best_row['mean_business_cost']:.2f}")
    print(f"Recall moyen : {best_row['mean_recall']:.4f}")
    print(f"AUC moyenne : {best_row['mean_roc_auc']:.4f}")


if __name__ == "__main__":
    main()