import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/05_optimisation_hyperparametres_seuil_metier.ipynb")


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """
# 05 - Optimisation des hyperparamètres et du seuil métier

## Objectif du notebook

Dans cette étape, l’objectif est d’optimiser le modèle retenu à l’étape précédente afin de réduire le coût métier.

Le problème métier est un problème de scoring crédit.  
La classe positive `TARGET = 1` correspond à un client en défaut de paiement.

Dans ce contexte, une erreur de type faux négatif est plus grave qu’une erreur de type faux positif :

- **FN** : client risqué prédit comme non risqué ;
- **FP** : bon client prédit comme risqué.

L’objectif est donc de minimiser un coût métier personnalisé, et non simplement de maximiser l’accuracy.
"""
    ),
    markdown_cell(
        """
## Rappel des prérequis de l'étape

Les prérequis sont remplis :

| Prérequis | Statut |
|---|---|
| Avoir entraîné plusieurs modèles | Oui |
| Avoir comparé leurs performances de base | Oui |
| Avoir compris la notion de coût d’erreur | Oui |
| Avoir défini une fonction de coût métier | Oui |

À l’étape précédente, plusieurs familles de modèles ont été comparées avec une validation croisée stratifiée :

- Logistic Regression ;
- Random Forest ;
- LightGBM ;
- XGBoost ;
- MLPClassifier.

Le meilleur candidat était XGBoost, car il obtenait le coût métier moyen le plus faible.
"""
    ),
    code_cell(
        """
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from IPython.display import Image, display


# Gestion robuste du chemin projet :
# - si le notebook est exécuté depuis le dossier notebooks, on remonte d'un niveau ;
# - sinon, on considère que le dossier courant est déjà la racine du projet.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

print("Racine projet :", PROJECT_ROOT)
print("Tracking URI MLflow :", mlflow.get_tracking_uri())
"""
    ),
    markdown_cell(
        """
## Fonction de coût métier

La métrique principale utilisée pour sélectionner le modèle est le coût métier :

`business_cost = 10 × FN + 1 × FP`

Ce choix signifie qu’un faux négatif coûte 10 fois plus cher qu’un faux positif.

Dans le contexte crédit :

- un **faux négatif** correspond à un client risqué qui obtient un crédit ;
- un **faux positif** correspond à un bon client refusé à tort.

Le modèle doit donc réduire en priorité les faux négatifs, sans exploser le nombre de faux positifs.
"""
    ),
    code_cell(
        """
FN_COST = 10
FP_COST = 1

print(f"Coût d'un faux négatif : {FN_COST}")
print(f"Coût d'un faux positif : {FP_COST}")
print("Formule : business_cost = 10 * FN + 1 * FP")
"""
    ),
    markdown_cell(
        """
## Récupération des expériences MLflow

On récupère maintenant les runs MLflow :

- `P6_credit_scoring_cross_validation` pour le meilleur modèle de base ;
- `P6_credit_scoring_optimization` pour le modèle optimisé avec Optuna.
"""
    ),
    code_cell(
        """
def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Expérience MLflow introuvable : {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    return runs


cv_runs = get_experiment_runs("P6_credit_scoring_cross_validation")
optimization_runs = get_experiment_runs("P6_credit_scoring_optimization")

print("Nombre de runs CV :", len(cv_runs))
print("Nombre de runs optimisation :", len(optimization_runs))
"""
    ),
    markdown_cell(
        """
## Résultats de base avant optimisation

Le meilleur modèle de l’étape 3 était XGBoost.

On récupère son run dans l’expérience de validation croisée.
"""
    ),
    code_cell(
        """
# Sélection du run XGBoost de validation croisée
xgb_base_runs = cv_runs[
    (cv_runs["params.model_name"] == "xgboost_weighted_cv")
    | (cv_runs["tags.mlflow.runName"] == "xgboost_weighted_cv")
].copy()

if xgb_base_runs.empty:
    raise ValueError("Aucun run XGBoost trouvé dans l'expérience de validation croisée.")

xgb_base_run = xgb_base_runs.sort_values(by="start_time", ascending=False).iloc[0]

base_summary = pd.DataFrame(
    {
        "metric": [
            "mean_accuracy",
            "mean_precision",
            "mean_recall",
            "mean_f1_score",
            "mean_roc_auc",
            "mean_business_cost",
            "std_business_cost",
        ],
        "value": [
            xgb_base_run.get("metrics.mean_accuracy"),
            xgb_base_run.get("metrics.mean_precision"),
            xgb_base_run.get("metrics.mean_recall"),
            xgb_base_run.get("metrics.mean_f1_score"),
            xgb_base_run.get("metrics.mean_roc_auc"),
            xgb_base_run.get("metrics.mean_business_cost"),
            xgb_base_run.get("metrics.std_business_cost"),
        ],
    }
)

base_summary
"""
    ),
    markdown_cell(
        """
## Optimisation avec Optuna

L’étape 4 utilise **Optuna** pour optimiser les hyperparamètres de XGBoost.

L’objectif d’optimisation n’est pas l’accuracy, mais le coût métier moyen obtenu en validation croisée.

Pour chaque essai Optuna :

1. un jeu d’hyperparamètres XGBoost est proposé ;
2. le modèle est évalué avec `StratifiedKFold` ;
3. plusieurs seuils de décision sont testés ;
4. le seuil qui minimise le coût métier est retenu ;
5. Optuna cherche à minimiser le coût métier moyen.
"""
    ),
    code_cell(
        """
# Sélection du run final d'optimisation
optimized_runs = optimization_runs[
    optimization_runs["tags.mlflow.runName"] == "xgboost_optuna_threshold_optimized"
].copy()

if optimized_runs.empty:
    raise ValueError("Aucun run final d'optimisation XGBoost trouvé dans MLflow.")

optimized_run = optimized_runs.sort_values(by="start_time", ascending=False).iloc[0]

print("Run optimisé sélectionné :")
print("run_id :", optimized_run["run_id"])
print("run_name :", optimized_run["tags.mlflow.runName"])
"""
    ),
    markdown_cell(
        """
## Meilleurs hyperparamètres trouvés

Les meilleurs hyperparamètres retenus par Optuna sont récupérés depuis MLflow.
"""
    ),
    code_cell(
        """
best_param_columns = [
    column
    for column in optimized_run.index
    if column.startswith("params.best_")
]

best_params = (
    optimized_run[best_param_columns]
    .dropna()
    .rename_axis("parameter")
    .reset_index(name="value")
)

best_params["parameter"] = best_params["parameter"].str.replace("params.best_", "", regex=False)

best_params
"""
    ),
    markdown_cell(
        """
## Seuil métier optimal

Le seuil par défaut d’un modèle de classification est souvent `0.50`.

Cependant, ce seuil n’est pas forcément adapté au contexte métier.  
Ici, les faux négatifs coûtent beaucoup plus cher que les faux positifs.

Le seuil de décision a donc été testé entre `0.10` et `0.90`.

Le meilleur seuil trouvé est :

`threshold = 0.45`

Cela signifie qu’un client est classé comme risqué dès que sa probabilité de défaut est supérieure ou égale à 45 %.
"""
    ),
    code_cell(
        """
optimization_summary = pd.DataFrame(
    {
        "metric": [
            "best_threshold",
            "best_mean_accuracy",
            "best_mean_precision",
            "best_mean_recall",
            "best_mean_f1_score",
            "best_mean_roc_auc",
            "best_mean_business_cost",
            "best_std_business_cost",
            "best_mean_business_cost_per_client",
            "best_mean_tn",
            "best_mean_fp",
            "best_mean_fn",
            "best_mean_tp",
        ],
        "value": [
            optimized_run.get("params.best_threshold"),
            optimized_run.get("metrics.best_mean_accuracy"),
            optimized_run.get("metrics.best_mean_precision"),
            optimized_run.get("metrics.best_mean_recall"),
            optimized_run.get("metrics.best_mean_f1_score"),
            optimized_run.get("metrics.best_mean_roc_auc"),
            optimized_run.get("metrics.best_mean_business_cost"),
            optimized_run.get("metrics.best_std_business_cost"),
            optimized_run.get("metrics.best_mean_business_cost_per_client"),
            optimized_run.get("metrics.best_mean_tn"),
            optimized_run.get("metrics.best_mean_fp"),
            optimized_run.get("metrics.best_mean_fn"),
            optimized_run.get("metrics.best_mean_tp"),
        ],
    }
)

optimization_summary
"""
    ),
    markdown_cell(
        """
## Comparaison avant / après optimisation

On compare maintenant :

- XGBoost avant optimisation, issu de l’étape 3 ;
- XGBoost optimisé avec Optuna et seuil métier ajusté.
"""
    ),
    code_cell(
        """
comparison = pd.DataFrame(
    [
        {
            "version": "XGBoost étape 3",
            "threshold": 0.50,
            "mean_accuracy": xgb_base_run.get("metrics.mean_accuracy"),
            "mean_precision": xgb_base_run.get("metrics.mean_precision"),
            "mean_recall": xgb_base_run.get("metrics.mean_recall"),
            "mean_f1_score": xgb_base_run.get("metrics.mean_f1_score"),
            "mean_roc_auc": xgb_base_run.get("metrics.mean_roc_auc"),
            "mean_business_cost": xgb_base_run.get("metrics.mean_business_cost"),
            "std_business_cost": xgb_base_run.get("metrics.std_business_cost"),
        },
        {
            "version": "XGBoost optimisé étape 4",
            "threshold": float(optimized_run.get("params.best_threshold")),
            "mean_accuracy": optimized_run.get("metrics.best_mean_accuracy"),
            "mean_precision": optimized_run.get("metrics.best_mean_precision"),
            "mean_recall": optimized_run.get("metrics.best_mean_recall"),
            "mean_f1_score": optimized_run.get("metrics.best_mean_f1_score"),
            "mean_roc_auc": optimized_run.get("metrics.best_mean_roc_auc"),
            "mean_business_cost": optimized_run.get("metrics.best_mean_business_cost"),
            "std_business_cost": optimized_run.get("metrics.best_std_business_cost"),
        },
    ]
)

comparison
"""
    ),
    code_cell(
        """
base_cost = float(xgb_base_run.get("metrics.mean_business_cost"))
optimized_cost = float(optimized_run.get("metrics.best_mean_business_cost"))

gain = base_cost - optimized_cost
gain_percent = gain / base_cost * 100

print(f"Coût métier avant optimisation : {base_cost:.2f}")
print(f"Coût métier après optimisation : {optimized_cost:.2f}")
print(f"Gain absolu : {gain:.2f}")
print(f"Gain relatif : {gain_percent:.2f} %")
"""
    ),
    markdown_cell(
        """
## Courbe coût métier vs seuil

Le point de vigilance principal de cette étape est de ne pas conserver le seuil `0.50` sans justification.

La courbe suivante permet de visualiser le coût métier moyen en fonction du seuil de décision.

Le seuil retenu est celui qui minimise le coût métier.
"""
    ),
    code_cell(
        """
threshold_curve_path = PROJECT_ROOT / "reports" / "optimization" / "xgboost_optimized_threshold_curve.csv"

if threshold_curve_path.exists():
    threshold_curve = pd.read_csv(threshold_curve_path)
    display(threshold_curve.sort_values(by="threshold"))
else:
    threshold_curve = None
    print(f"Fichier introuvable : {threshold_curve_path}")
"""
    ),
    code_cell(
        """
if threshold_curve is not None:
    plot_df = threshold_curve.sort_values(by="threshold")

    best_threshold = float(optimized_run.get("params.best_threshold"))
    best_cost = float(optimized_run.get("metrics.best_mean_business_cost"))

    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_df["threshold"],
        plot_df["mean_business_cost"],
        marker="o",
    )
    plt.axvline(best_threshold, linestyle="--")
    plt.xlabel("Seuil de décision")
    plt.ylabel("Coût métier moyen")
    plt.title("Coût métier moyen selon le seuil de décision")
    plt.grid(True)
    plt.show()

    print(f"Meilleur seuil : {best_threshold:.2f}")
    print(f"Meilleur coût métier moyen : {best_cost:.2f}")
"""
    ),
    markdown_cell(
        """
## Affichage de la figure sauvegardée

Le script d’optimisation sauvegarde également une figure dans le dossier `reports/figures`.

Cette figure peut être utilisée dans la présentation ou le rapport.
"""
    ),
    code_cell(
        """
figure_path = PROJECT_ROOT / "reports" / "figures" / "xgboost_optimized_cost_vs_threshold.png"

if figure_path.exists():
    display(Image(filename=str(figure_path)))
else:
    print(f"Figure introuvable : {figure_path}")
"""
    ),
    markdown_cell(
        """
## Interprétation des résultats

Le modèle optimisé obtient les résultats suivants :

| Élément | Résultat |
|---|---:|
| Modèle | XGBoost |
| Outil d’optimisation | Optuna |
| Stratégie de validation | StratifiedKFold |
| Nombre de folds | 3 |
| Meilleur seuil métier | 0.45 |
| Coût métier moyen | 13609.33 |
| Recall moyen | 0.6539 |
| AUC moyenne | 0.7713 |

Le seuil optimal est inférieur à `0.50`.  
C’est cohérent avec le contexte métier, car les faux négatifs sont les erreurs les plus coûteuses.

En abaissant le seuil, le modèle détecte davantage de clients risqués, ce qui permet de réduire le coût métier global.
"""
    ),
    markdown_cell(
        """
## Vérification des points de vigilance

| Point de vigilance | Réponse apportée |
|---|---|
| Ne pas garder le seuil 0.5 sans justification | Le seuil optimal est recherché entre 0.10 et 0.90 |
| Ne pas optimiser uniquement l’AUC ou l’accuracy | L’objectif principal est le coût métier |
| Ne pas oublier le coût FN vs FP | La fonction de coût pondère FN à 10 et FP à 1 |
| Ne pas oublier la courbe coût vs seuil | La courbe est tracée et sauvegardée |
| Ne pas choisir un modèle sans tester sa robustesse | Le modèle est réévalué avec StratifiedKFold |
"""
    ),
    markdown_cell(
        """
## Conclusion de l'étape 4

L’optimisation des hyperparamètres de XGBoost a été réalisée avec Optuna.

L’objectif d’optimisation n’était pas l’accuracy, mais le coût métier personnalisé :

`business_cost = 10 × FN + 1 × FP`

Pour chaque essai Optuna, le modèle a été évalué avec une validation croisée stratifiée.  
Plusieurs seuils de décision, de `0.10` à `0.90`, ont été testés afin d’identifier le seuil minimisant le coût métier.

Le meilleur modèle obtenu utilise un seuil de décision de `0.45`.  
Ce seuil est inférieur au seuil standard de `0.50`, ce qui est cohérent avec le contexte métier : les faux négatifs sont plus coûteux que les faux positifs.

Après optimisation, XGBoost obtient un coût métier moyen de `13609.33`, contre environ `13742.67` avant optimisation.

Le modèle XGBoost optimisé avec seuil métier ajusté est donc retenu comme modèle candidat final.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Notebook créé : {NOTEBOOK_PATH}")