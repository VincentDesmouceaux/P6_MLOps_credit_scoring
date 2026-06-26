import gc

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import PROCESSED_DATA_DIR
from src.models.mlflow_utils import get_git_branch_name, get_git_commit_hash


TRAIN_MODELING_FILE = "train_modeling.csv"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

EXPERIMENT_NAME = "P6_credit_scoring_baseline"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

# Coût métier défini dans l'énoncé :
# un faux négatif coûte 10 fois plus cher qu'un faux positif.
FN_COST = 10
FP_COST = 1


def load_train_modeling_dataset() -> pd.DataFrame:
    """
    Charge le dataset de modélisation train.

    Ce fichier a été généré à l'étape 1.
    Il contient :
    - les features préparées ;
    - la variable cible TARGET ;
    - l'identifiant client SK_ID_CURR.
    """
    file_path = PROCESSED_DATA_DIR / TRAIN_MODELING_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"Chargement : {file_path}")
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les variables explicatives X et la cible y.

    On retire :
    - TARGET des variables explicatives ;
    - SK_ID_CURR car c'est un identifiant technique, pas une variable métier.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError("La colonne TARGET est absente du dataset.")

    y = df[TARGET_COLUMN].astype(int)

    X = df.drop(columns=[TARGET_COLUMN])

    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])

    return X, y


def compute_business_cost(y_true, y_pred) -> dict:
    """
    Calcule le coût métier basé sur la matrice de confusion.

    Dans le projet :
    - FN = client risqué prédit comme bon client ;
    - FP = bon client prédit comme risqué.

    Le faux négatif est plus grave, car cela signifie que l'entreprise accorde
    un crédit à un client qui risque de ne pas rembourser.
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


def evaluate_model(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    """
    Calcule les métriques du modèle sur le jeu de validation.

    On calcule :
    - accuracy ;
    - precision ;
    - recall ;
    - f1-score ;
    - AUC ;
    - matrice de confusion ;
    - score métier.
    """
    y_pred = model.predict(X_valid)

    # predict_proba permet de calculer l'AUC.
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


def build_dummy_pipeline() -> Pipeline:
    """
    Construit un pipeline baseline.

    Le DummyClassifier ne cherche pas de vraie relation dans les données.
    Il sert de modèle de référence minimal.

    L'imputation est placée dans le pipeline pour éviter les problèmes liés aux
    valeurs manquantes, sans apprendre quoi que ce soit sur le jeu de validation.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    return pipeline


def main() -> None:
    """
    Lance une première expérimentation MLflow avec un DummyClassifier.

    Objectif :
    vérifier que MLflow trace correctement les paramètres, les métriques
    et le modèle entraîné.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_train_modeling_dataset()
    X, y = split_features_target(df)

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

    model = build_dummy_pipeline()

    with mlflow.start_run(run_name="dummy_classifier_most_frequent"):
        # Tags pour mieux comprendre le run dans l'interface MLflow.
        mlflow.set_tag("project", "P6_MLOps_credit_scoring")
        mlflow.set_tag("step", "baseline")
        mlflow.set_tag("model_type", "DummyClassifier")
        mlflow.set_tag("description", "Baseline most_frequent pour vérifier MLflow")
        mlflow.set_tag ("git_commit", get_git_commit_hash ( ))
        mlflow.set_tag ("git_branch", get_git_branch_name ( ))

        # Paramètres principaux.
        mlflow.log_param("model", "DummyClassifier")
        mlflow.log_param("strategy", "most_frequent")
        mlflow.log_param("imputer_strategy", "median")
        mlflow.log_param("validation_size", VALIDATION_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("fn_cost", FN_COST)
        mlflow.log_param("fp_cost", FP_COST)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_rows", X.shape[0])

        print("Entraînement du modèle baseline...")
        model.fit(X_train, y_train)

        print("Évaluation du modèle baseline...")
        metrics = evaluate_model(model, X_valid, y_valid)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model (
            sk_model=model,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

        print("\nMétriques du run MLflow :")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")

    del df, X, y, X_train, X_valid, y_train, y_valid, model
    gc.collect()

    print("\nRun MLflow terminé.")


if __name__ == "__main__":
    main()