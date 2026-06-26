import gc

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
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


def load_train_modeling_dataset() -> pd.DataFrame:
    """
    Charge le dataset de modélisation généré à l'étape 1.
    """
    file_path = PROCESSED_DATA_DIR / TRAIN_MODELING_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"Chargement : {file_path}")
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les variables explicatives X et la cible y.

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

    LightGBM peut refuser certains caractères spéciaux dans les noms de
    variables car ils posent problème lors de la sérialisation du modèle.

    On garde les données inchangées, on modifie uniquement les noms de colonnes.
    """
    X = X.copy()

    X.columns = (
        X.columns
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return X


def compute_business_cost(y_true, y_pred) -> dict:
    """
    Calcule le coût métier.

    FN : client en défaut prédit comme bon client.
    FP : bon client prédit comme client risqué.

    Dans ce projet, FN coûte 10 fois plus que FP.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    business_cost = (FN_COST * fn) + (FP_COST * fp)
    business_cost_per_client = business_cost / len(y_true)

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost": float(business_cost),
        "business_cost_per_client": float(business_cost_per_client),
    }


def evaluate_model(model, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    """
    Calcule les métriques classiques et métier.
    """
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, zero_division=0),
        "recall": recall_score(y_valid, y_pred, zero_division=0),
        "f1_score": f1_score(y_valid, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, y_proba),
    }

    business_metrics = compute_business_cost(
        y_true=y_valid,
        y_pred=y_pred,
    )

    metrics.update(business_metrics)

    return metrics


def build_lightgbm_model(scale_pos_weight: float) -> lgb.LGBMClassifier:
    """
    Construit un modèle LightGBM.

    LightGBM est adapté aux données tabulaires et gère nativement les valeurs
    manquantes. Contrairement à la régression logistique, on n'a pas besoin
    de scaling.

    scale_pos_weight permet de prendre en compte le déséquilibre de classes.
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


def main() -> None:
    """
    Entraîne un modèle LightGBM et trace l'expérience avec MLflow.

    Ce run permet d'ajouter un modèle plus performant et compatible avec
    mlflow.lightgbm.autolog().
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_train_modeling_dataset()
    X, y = split_features_target(df)
    X = clean_lightgbm_feature_names (X)

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

    print(f"X_train shape : {X_train.shape}")
    print(f"X_valid shape : {X_valid.shape}")

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"scale_pos_weight : {scale_pos_weight:.4f}")

    model = build_lightgbm_model(scale_pos_weight=scale_pos_weight)

    with mlflow.start_run(run_name="lightgbm_baseline_weighted"):
        mlflow.set_tag("project", "P6_MLOps_credit_scoring")
        mlflow.set_tag("step", "modeling")
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag(
            "description",
            "LightGBM baseline avec gestion du déséquilibre via scale_pos_weight",
        )
        mlflow.set_tag("git_commit", get_git_commit_hash())
        mlflow.set_tag("git_branch", get_git_branch_name())

        # Autolog compatible LightGBM.
        # On garde aussi nos logs manuels pour les métriques métier.
        mlflow.lightgbm.autolog(log_models=False)

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

        print("Entraînement du modèle LightGBM...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
        )

        print("Évaluation du modèle LightGBM...")
        metrics = evaluate_model(model, X_valid, y_valid)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
        )

        print("\nMétriques du run MLflow :")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")

    del df, X, y, X_train, X_valid, y_train, y_valid, model
    gc.collect()

    print("\nRun MLflow LightGBM terminé.")


if __name__ == "__main__":
    main()