import mlflow

from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "P6_credit_scoring_baseline"
REGISTERED_MODEL_NAME = "P6_credit_scoring_lightgbm_champion"


def get_experiment_id(client: MlflowClient) -> str:
    """
    Récupère l'identifiant de l'expérience MLflow.

    L'expérience contient tous les runs de modélisation :
    DummyClassifier, LogisticRegression, LightGBM et tuning du seuil.
    """
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Expérience introuvable : {EXPERIMENT_NAME}")

    return experiment.experiment_id


def find_best_business_run(client: MlflowClient, experiment_id: str):
    """
    Recherche le meilleur run selon le coût métier.

    On privilégie d'abord les runs parents qui contiennent best_business_cost,
    car ce sont eux qui sauvegardent le modèle.

    Les runs enfants lightgbm_threshold_0.xx contiennent les métriques par seuil,
    mais pas le modèle enregistré.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=200,
    )

    if not runs:
        raise ValueError("Aucun run terminé trouvé dans l'expérience MLflow.")

    parent_candidates = []
    classic_candidates = []

    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags

        run_name = tags.get("mlflow.runName", "unknown")

        # Cas prioritaire : run parent de tuning avec meilleur seuil.
        if "best_business_cost" in metrics:
            parent_candidates.append(
                {
                    "run": run,
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "model": params.get("model", "unknown"),
                    "business_cost": metrics["best_business_cost"],
                    "threshold": metrics.get("best_threshold"),
                    "metric_source": "best_business_cost",
                }
            )

        # Cas secondaire : runs classiques avec business_cost.
        elif "business_cost" in metrics:
            classic_candidates.append(
                {
                    "run": run,
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "model": params.get("model", "unknown"),
                    "business_cost": metrics["business_cost"],
                    "threshold": params.get("threshold"),
                    "metric_source": "business_cost",
                }
            )

    # On choisit d'abord parmi les runs parents, car ils contiennent le modèle.
    if parent_candidates:
        best_candidate = sorted(
            parent_candidates,
            key=lambda item: item["business_cost"],
        )[0]

        return best_candidate

    if classic_candidates:
        best_candidate = sorted(
            classic_candidates,
            key=lambda item: item["business_cost"],
        )[0]

        return best_candidate

    raise ValueError("Aucun run avec business_cost ou best_business_cost trouvé.")


def register_model_from_run(client: MlflowClient, best_candidate: dict):
    """
    Enregistre le modèle du meilleur run dans le Model Registry MLflow.
    """
    run_id = best_candidate["run_id"]

    # Le modèle a été sauvegardé dans le run MLflow sous le nom "model".
    model_uri = f"runs:/{run_id}/model"

    print(f"Meilleur run trouvé : {best_candidate['run_name']}")
    print(f"Run ID : {run_id}")
    print(f"Modèle : {best_candidate['model']}")
    print(f"Coût métier : {best_candidate['business_cost']}")
    print(f"Seuil : {best_candidate['threshold']}")
    print(f"Model URI : {model_uri}")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = model_version.version

    print(f"\nModèle enregistré : {REGISTERED_MODEL_NAME}")
    print(f"Version créée : {version}")

    # Tags au niveau du modèle enregistré.
    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "project",
        "P6_MLOps_credit_scoring",
    )

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "business_objective",
        "Minimiser le coût métier du scoring crédit",
    )

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "error_cost_rule",
        "FN cost = 10 x FP cost",
    )

    # Tags au niveau de la version.
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "source_run_id",
        run_id,
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "business_cost",
        str(best_candidate["business_cost"]),
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "threshold",
        str(best_candidate["threshold"]),
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "model_type",
        str(best_candidate["model"]),
    )

    # Alias champion : indique que cette version est le meilleur modèle actuel.
    try:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "champion",
            version,
        )
        print("Alias 'champion' ajouté à cette version.")
    except Exception as error:
        print(f"Alias champion non ajouté : {error}")

    return model_version


def main() -> None:
    """
    Point d'entrée du script.

    Ce script permet de versionner le meilleur modèle dans MLflow.
    """
    client = MlflowClient()

    experiment_id = get_experiment_id(client)

    best_candidate = find_best_business_run(
        client=client,
        experiment_id=experiment_id,
    )

    register_model_from_run(
        client=client,
        best_candidate=best_candidate,
    )

    print("\nVersionnement du meilleur modèle terminé.")


if __name__ == "__main__":
    main()