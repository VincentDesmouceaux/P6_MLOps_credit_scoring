from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# Racine du projet :
# fichier actuel = src/models/update_mlflow_experiments_metadata.py
# parents[2] = racine du projet P6_MLOps_credit_scoring
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

client = MlflowClient()


EXPERIMENTS_METADATA = {
    "P6_credit_scoring_baseline": {
        "description": (
            "Première série d'expérimentations du projet P6 credit scoring. "
            "Cette expérience compare un DummyClassifier, une Logistic Regression "
            "et un premier modèle LightGBM avec MLflow. Les métriques suivies sont "
            "l'accuracy, la precision, le recall, le F1-score, l'AUC-ROC et un coût "
            "métier personnalisé pénalisant davantage les faux négatifs."
        ),
        "tags": {
            "project": "P6_MLOps_credit_scoring",
            "step": "baseline_modeling",
            "target": "TARGET",
            "problem_type": "binary_classification",
            "business_metric": "10_FN_plus_1_FP",
            "class_imbalance": "TARGET_1_around_8_percent",
            "dataset": "Home_Credit_Default_Risk",
        },
    },
    "P6_credit_scoring_cross_validation": {
        "description": (
            "Comparaison de plusieurs familles de modèles avec validation croisée "
            "stratifiée pour le projet P6 credit scoring. Les modèles testés sont "
            "Logistic Regression, Random Forest, LightGBM, XGBoost et MLPClassifier. "
            "Les scores moyens et écarts-types sont suivis dans MLflow afin d'évaluer "
            "la performance et la robustesse des modèles."
        ),
        "tags": {
            "project": "P6_MLOps_credit_scoring",
            "step": "cross_validation_model_comparison",
            "target": "TARGET",
            "problem_type": "binary_classification",
            "validation_strategy": "StratifiedKFold",
            "business_metric": "10_FN_plus_1_FP",
            "class_imbalance": "TARGET_1_around_8_percent",
            "dataset": "Home_Credit_Default_Risk",
            "models_tested": "LogisticRegression_RandomForest_LightGBM_XGBoost_MLP",
        },
    },
}


def update_experiment_metadata(experiment_name: str, metadata: dict) -> None:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Expérience introuvable : {experiment_name}")
        return

    experiment_id = experiment.experiment_id

    client.set_experiment_tag(
        experiment_id,
        "mlflow.note.content",
        metadata["description"],
    )

    for tag_key, tag_value in metadata["tags"].items():
        client.set_experiment_tag(
            experiment_id,
            tag_key,
            tag_value,
        )

    print(f"Métadonnées mises à jour : {experiment_name}")


def main() -> None:
    print(f"Tracking URI MLflow : {mlflow.get_tracking_uri()}")

    for experiment_name, metadata in EXPERIMENTS_METADATA.items():
        update_experiment_metadata(
            experiment_name=experiment_name,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()