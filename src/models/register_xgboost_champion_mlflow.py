from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "P6_credit_scoring_optimization"
RUN_NAME = "xgboost_optuna_threshold_optimized"

NEW_REGISTERED_MODEL_NAME = "P6_credit_scoring_default_risk_model"
OLD_REGISTERED_MODEL_NAME = "P6_credit_scoring_lightgbm_champion"


REGISTERED_MODEL_DESCRIPTION = """
Modèle de scoring crédit du projet P6 MLOps.

Ce Registered Model centralise les versions successives du modèle de prédiction
du risque de défaut de paiement client. Le modèle actuellement promu avec
l'alias champion est un XGBoost optimisé avec Optuna.

Le choix du champion repose sur une métrique métier personnalisée :

business_cost = 10 × FN + 1 × FP

Cette fonction pénalise davantage les faux négatifs, car accepter à tort un
client risqué est plus coûteux que refuser à tort un bon client.
"""


MODEL_VERSION_DESCRIPTION = """
Version champion du modèle de scoring crédit.

Cette version correspond au modèle XGBoost optimisé avec Optuna lors de l'étape 4
du projet. Les hyperparamètres ont été optimisés en validation croisée stratifiée
et le seuil de décision métier a été ajusté à 0.45.

Le modèle a été sélectionné car il minimise le coût métier moyen par rapport aux
autres modèles testés : Logistic Regression, Random Forest, LightGBM, XGBoost
non optimisé et MLPClassifier.
"""


def get_metric(run_row, metric_name: str, default_value: str = "unknown") -> str:
    column_name = f"metrics.{metric_name}"
    if column_name in run_row and not mlflow.utils.validation._is_numeric(run_row[column_name]) is False:
        return str(run_row[column_name])
    if column_name in run_row:
        return str(run_row[column_name])
    return default_value


def get_param(run_row, param_name: str, default_value: str = "unknown") -> str:
    column_name = f"params.{param_name}"
    if column_name in run_row:
        return str(run_row[column_name])
    return default_value


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    client = MlflowClient()

    print("Tracking URI :", mlflow.get_tracking_uri())

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Expérience MLflow introuvable : {EXPERIMENT_NAME}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
        order_by=["start_time DESC"],
    )

    if runs.empty:
        raise ValueError(
            f"Aucun run trouvé avec le nom {RUN_NAME} "
            f"dans l'expérience {EXPERIMENT_NAME}"
        )

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]

    best_threshold = get_param(best_run, "best_threshold", "0.45")
    best_business_cost = get_metric(best_run, "best_mean_business_cost")
    best_recall = get_metric(best_run, "best_mean_recall")
    best_roc_auc = get_metric(best_run, "best_mean_roc_auc")

    print("Run XGBoost optimisé trouvé")
    print("Run ID :", run_id)
    print("Seuil métier :", best_threshold)
    print("Coût métier moyen :", best_business_cost)
    print("Recall moyen :", best_recall)
    print("AUC moyenne :", best_roc_auc)

    model_uri = f"runs:/{run_id}/model"

    print("Model URI :", model_uri)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=NEW_REGISTERED_MODEL_NAME,
    )

    print()
    print("Nouveau modèle enregistré")
    print("Nom :", NEW_REGISTERED_MODEL_NAME)
    print("Version :", model_version.version)

    # Description du Registered Model
    client.update_registered_model(
        name=NEW_REGISTERED_MODEL_NAME,
        description=REGISTERED_MODEL_DESCRIPTION.strip(),
    )

    # Tags globaux du Registered Model
    registered_model_tags = {
        "project": "P6_MLOps_credit_scoring",
        "dataset": "Home_Credit_Default_Risk",
        "problem_type": "binary_classification",
        "target": "TARGET",
        "business_metric": "10_FN_plus_1_FP",
        "current_champion_model_family": "XGBoost",
        "mlops_step": "model_registry",
        "owner": "Vincent_Desmouceaux",
    }

    for key, value in registered_model_tags.items():
        client.set_registered_model_tag(
            name=NEW_REGISTERED_MODEL_NAME,
            key=key,
            value=value,
        )

    # Description de la version précise du modèle
    client.update_model_version(
        name=NEW_REGISTERED_MODEL_NAME,
        version=model_version.version,
        description=MODEL_VERSION_DESCRIPTION.strip(),
    )

    # Tags de la version champion XGBoost
    model_version_tags = {
        "model_family": "XGBoost",
        "status": "champion",
        "selection_reason": "lowest_mean_business_cost",
        "optimized_with": "Optuna",
        "validation_strategy": "StratifiedKFold",
        "threshold": best_threshold,
        "mean_business_cost": best_business_cost,
        "mean_recall": best_recall,
        "mean_roc_auc": best_roc_auc,
        "source_experiment": EXPERIMENT_NAME,
        "source_run_name": RUN_NAME,
        "source_run_id": run_id,
    }

    for key, value in model_version_tags.items():
        client.set_model_version_tag(
            name=NEW_REGISTERED_MODEL_NAME,
            version=model_version.version,
            key=key,
            value=value,
        )

    # Alias champion sur la version XGBoost
    client.set_registered_model_alias(
        name=NEW_REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version.version,
    )

    print()
    print("Description, tags et alias ajoutés au modèle XGBoost champion.")
    print("Modèle :", NEW_REGISTERED_MODEL_NAME)
    print("Version champion :", model_version.version)

    # Suppression éventuelle de l'ancien alias champion sur LightGBM
    try:
        client.delete_registered_model_alias(
            name=OLD_REGISTERED_MODEL_NAME,
            alias="champion",
        )
        print()
        print("Ancien alias 'champion' supprimé du modèle LightGBM.")
    except Exception:
        print()
        print(
            "Ancien alias LightGBM absent ou impossible à supprimer. "
            "Ce n'est pas bloquant."
        )

    print()
    print("Changement du champion terminé.")


if __name__ == "__main__":
    main()