import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "P6_credit_scoring_baseline"


def main() -> None:
    """
    Configure les métadonnées de l'expérience MLflow.

    Cela permet d'avoir une description claire dans l'interface MLflow.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Expérience introuvable : {EXPERIMENT_NAME}")

    client = MlflowClient()

    client.set_experiment_tag(
        experiment.experiment_id,
        "project",
        "P6_MLOps_credit_scoring",
    )

    client.set_experiment_tag(
        experiment.experiment_id,
        "step",
        "model_tracking",
    )

    client.set_experiment_tag(
        experiment.experiment_id,
        "objective",
        "Comparer plusieurs modèles de scoring crédit avec MLflow",
    )

    client.set_experiment_tag(
        experiment.experiment_id,
        "business_context",
        "Le coût d'un faux négatif est fixé à 10 fois le coût d'un faux positif.",
    )

    client.set_experiment_tag(
        experiment.experiment_id,
        "mlflow.note.content",
        (
            "Cette expérience regroupe les premiers modèles de scoring crédit "
            "du projet P6. Les runs comparent un DummyClassifier baseline "
            "et une LogisticRegression équilibrée. Les métriques suivies sont "
            "l'accuracy, la precision, le recall, le F1-score, le ROC AUC, "
            "ainsi qu'un score métier pénalisant davantage les faux négatifs."
        ),
    )

    print(f"Expérience MLflow configurée : {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()