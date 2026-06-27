import gc
import re
from collections import Counter

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DATA_DIR
from src.models.mlflow_utils import get_git_branch_name, get_git_commit_hash


TRAIN_MODELING_FILE = "train_modeling.csv"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

EXPERIMENT_NAME = "P6_credit_scoring_cross_validation"

RANDOM_STATE = 42
N_SPLITS = 3

FN_COST = 10
FP_COST = 1

# Pour une première validation croisée rapide et stable.
# On garde un échantillon stratifié, sinon RandomForest et LogisticRegression peuvent être longs.
USE_SAMPLE = True
SAMPLE_SIZE = 80000


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


def stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Crée un échantillon stratifié pour conserver la proportion de TARGET.
    """
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


def clean_lightgbm_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes pour LightGBM.
    """
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


def compute_business_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    business_cost = (FN_COST * fn) + (FP_COST * fp)
    business_cost_per_client = business_cost / len(y_true)

    metrics = {
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

    return metrics


def get_models(scale_pos_weight: float) -> dict:
    """
    Définit plusieurs familles de modèles.

    LogisticRegression : modèle linéaire simple.
    RandomForest : modèle ensembliste basé sur des arbres.
    LightGBM : modèle de boosting performant sur données tabulaires.
    XGBoost : autre modèle de gradient boosting.
    MLPClassifier : réseau de neurones simple.
    """
    models = {
        "logistic_regression_balanced_cv": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=500,
                        solver="saga",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "random_forest_balanced_cv": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_leaf=50,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "lightgbm_weighted_cv": lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),

        "xgboost_weighted_cv": XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),

        "mlp_balanced_cv": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.0001,
                        batch_size=512,
                        learning_rate_init=0.001,
                        max_iter=50,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def cross_validate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Entraîne un modèle avec StratifiedKFold et retourne les métriques par fold.
    """
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold_index, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nModèle : {model_name} | Fold {fold_index}/{N_SPLITS}")

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_valid)[:, 1]
        else:
            y_proba = y_pred

        metrics = compute_business_metrics(
            y_true=y_valid,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        metrics["fold"] = fold_index
        metrics["model_name"] = model_name

        fold_results.append(metrics)

        print(
            f"Fold {fold_index} | "
            f"recall={metrics['recall']:.4f} | "
            f"roc_auc={metrics['roc_auc']:.4f} | "
            f"business_cost={metrics['business_cost']:.0f}"
        )

    return pd.DataFrame(fold_results)


def log_cv_results_to_mlflow(
    model_name: str,
    model,
    cv_results: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float,
) -> None:
    """
    Logge les résultats moyens et l'écart-type dans MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("project", "P6_MLOps_credit_scoring")
        mlflow.set_tag("step", "cross_validation")
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("description", "Validation croisée avec StratifiedKFold")
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("git_branch", get_git_branch_name())

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("fn_cost", FN_COST)
        mlflow.log_param("fp_cost", FP_COST)
        mlflow.log_param("use_sample", USE_SAMPLE)
        mlflow.log_param("sample_size", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("target_rate", y.mean())
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        metrics_to_aggregate = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "business_cost",
            "business_cost_per_client",
            "fn",
            "fp",
            "tp",
            "tn",
        ]

        for metric in metrics_to_aggregate:
            mlflow.log_metric(f"mean_{metric}", cv_results[metric].mean())
            mlflow.log_metric(f"std_{metric}", cv_results[metric].std())

        for _, row in cv_results.iterrows():
            fold = int(row["fold"])

            for metric in metrics_to_aggregate:
                mlflow.log_metric(
                    f"fold_{fold}_{metric}",
                    row[metric],
                )

        output_path = f"cv_results_{model_name}.csv"
        cv_results.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)

        # Dans cette étape de validation croisée, on ne sauvegarde pas les modèles.
        # L'objectif est de comparer les performances moyennes et la robustesse
        # des algorithmes avec StratifiedKFold.
        #
        # Les modèles complets seront sauvegardés plus tard après sélection finale.
        # Cela évite aussi d'alourdir MLflow avec plusieurs artefacts de modèles.
        print (f"Modèle non sauvegardé en artefact pour le run CV : {model_name}")

def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_dataset()

    X, y = split_features_target(df)
    X = clean_lightgbm_feature_names(X)

    print(f"Dataset complet : {X.shape}")
    print(f"Taux de défaut complet : {y.mean():.4f}")

    if USE_SAMPLE:
        X, y = stratified_sample(
            X=X,
            y=y,
            sample_size=SAMPLE_SIZE,
        )

        print(f"Échantillon utilisé : {X.shape}")
        print(f"Taux de défaut échantillon : {y.mean():.4f}")

    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"scale_pos_weight : {scale_pos_weight:.4f}")

    models = get_models(scale_pos_weight=scale_pos_weight)

    for model_name, model in models.items():
        cv_results = cross_validate_model(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
        )

        log_cv_results_to_mlflow(
            model_name=model_name,
            model=model,
            cv_results=cv_results,
            X=X,
            y=y,
            scale_pos_weight=scale_pos_weight,
        )

        print(f"\nRésumé CV pour {model_name}")
        print(
            cv_results[
                [
                    "fold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "business_cost",
                    "business_cost_per_client",
                ]
            ]
        )

        del cv_results
        gc.collect()

    print("\nValidation croisée terminée.")


if __name__ == "__main__":
    main()