import gc

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR
from src.models.mlflow_utils import get_git_branch_name, get_git_commit_hash


TRAIN_MODELING_FILE = "train_modeling.csv"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

EXPERIMENT_NAME = "P6_credit_scoring_baseline"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

FN_COST = 10
FP_COST = 1

THRESHOLDS = np.arange(0.10, 0.91, 0.05)


def load_train_modeling_dataset() -> pd.DataFrame:
    """
    Charge le dataset de modélisation.
    """
    file_path = PROCESSED_DATA_DIR / TRAIN_MODELING_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"Chargement : {file_path}")
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare X et y.

    SK_ID_CURR est retiré car c'est un identifiant technique.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError("La colonne TARGET est absente du dataset.")

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])

    return X, y


def clean_lightgbm_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes pour LightGBM.

    LightGBM peut refuser certains caractères spéciaux dans les noms de features.
    On modifie seulement les noms de colonnes, pas les valeurs.
    """
    X = X.copy()

    X.columns = (
        X.columns
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return X


def build_lightgbm_model(scale_pos_weight: float) -> lgb.LGBMClassifier:
    """
    Construit un modèle LightGBM pondéré.
    """
    model = lgb.LGBMClassifier(
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
    )

    return model


def compute_metrics_for_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """
    Calcule les métriques pour un seuil de décision donné.

    Si y_proba >= threshold, on prédit TARGET = 1.
    Sinon, on prédit TARGET = 0.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    business_cost = (FN_COST * fn) + (FP_COST * fp)
    business_cost_per_client = business_cost / len(y_true)

    metrics = {
        "threshold": float(threshold),
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


def main() -> None:
    """
    Entraîne LightGBM puis teste plusieurs seuils de décision.

    Chaque seuil est enregistré comme un run MLflow enfant.
    Le meilleur seuil métier est enregistré dans le run parent.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_train_modeling_dataset()
    X, y = split_features_target(df)
    X = clean_lightgbm_feature_names(X)

    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Taux de défaut : {y.mean():.4f}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"X_train shape : {X_train.shape}")
    print(f"X_valid shape : {X_valid.shape}")
    print(f"scale_pos_weight : {scale_pos_weight:.4f}")

    model = build_lightgbm_model(scale_pos_weight=scale_pos_weight)

    with mlflow.start_run(run_name="lightgbm_threshold_tuning") as parent_run:
        mlflow.set_tag("project", "P6_MLOps_credit_scoring")
        mlflow.set_tag("step", "threshold_tuning")
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag(
            "description",
            "Optimisation du seuil de décision LightGBM selon le coût métier",
        )
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("git_branch", get_git_branch_name())

        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("objective", "binary")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("num_leaves", 31)
        mlflow.log_param("min_child_samples", 50)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("validation_size", VALIDATION_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("fn_cost", FN_COST)
        mlflow.log_param("fp_cost", FP_COST)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_rows", X.shape[0])
        mlflow.log_param("thresholds_tested", ",".join([str(round(t, 2)) for t in THRESHOLDS]))

        print("Entraînement du modèle LightGBM...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
        )

        print("Calcul des probabilités sur validation...")
        y_proba = model.predict_proba(X_valid)[:, 1]

        threshold_results = []

        print("Test des seuils de décision...")
        for threshold in THRESHOLDS:
            metrics = compute_metrics_for_threshold(
                y_true=y_valid,
                y_proba=y_proba,
                threshold=float(threshold),
            )

            threshold_results.append(metrics)

            run_name = f"lightgbm_threshold_{threshold:.2f}"

            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tag("project", "P6_MLOps_credit_scoring")
                mlflow.set_tag("step", "threshold_tuning_child")
                mlflow.set_tag("model_type", "LightGBM")
                mlflow.set_tag("parent_run_id", parent_run.info.run_id)
                mlflow.set_tag("git_commit", get_git_commit_hash())
                mlflow.set_tag("git_branch", get_git_branch_name())

                mlflow.log_param("model", "LightGBM")
                mlflow.log_param("threshold", float(threshold))
                mlflow.log_param("fn_cost", FN_COST)
                mlflow.log_param("fp_cost", FP_COST)
                mlflow.log_param("scale_pos_weight", scale_pos_weight)

                for metric_name, metric_value in metrics.items():
                    if metric_name != "threshold":
                        mlflow.log_metric(metric_name, metric_value)

        results_df = pd.DataFrame(threshold_results)

        best_row = results_df.sort_values(
            by="business_cost",
            ascending=True,
        ).iloc[0]

        best_threshold = float(best_row["threshold"])

        print("\nMeilleur seuil métier trouvé :")
        print(best_row)

        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_business_cost", float(best_row["business_cost"]))
        mlflow.log_metric(
            "best_business_cost_per_client",
            float(best_row["business_cost_per_client"]),
        )
        mlflow.log_metric("best_recall", float(best_row["recall"]))
        mlflow.log_metric("best_precision", float(best_row["precision"]))
        mlflow.log_metric("best_f1_score", float(best_row["f1_score"]))
        mlflow.log_metric("best_roc_auc", float(best_row["roc_auc"]))
        mlflow.log_metric("best_fn", float(best_row["fn"]))
        mlflow.log_metric("best_fp", float(best_row["fp"]))
        mlflow.log_metric("best_tp", float(best_row["tp"]))
        mlflow.log_metric("best_tn", float(best_row["tn"]))

        # Petit artefact CSV utile, pas volumineux.
        output_path = "threshold_tuning_results.csv"
        results_df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)

        # On sauvegarde le modèle une seule fois dans le run parent.
        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
        )

    del df, X, y, X_train, X_valid, y_train, y_valid, model
    gc.collect()

    print("\nOptimisation du seuil terminée.")


if __name__ == "__main__":
    main()