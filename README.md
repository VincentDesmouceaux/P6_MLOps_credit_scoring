# 🧠 P6 — Initiez-vous au MLOps 1/2 : Credit Scoring

Projet MLOps de scoring crédit basé sur le dataset **Home Credit Default Risk**.
L’objectif est de construire, comparer, tracker et optimiser des modèles de classification afin de prédire le risque de défaut de paiement d’un client.

Le projet intègre :

* préparation et enrichissement des données ;
* feature engineering à partir de plusieurs tables relationnelles ;
* entraînement de plusieurs modèles de classification ;
* suivi des expériences avec MLflow ;
* validation croisée stratifiée ;
* optimisation des hyperparamètres avec Optuna ;
* optimisation du seuil métier ;
* prise en compte du déséquilibre des classes ;
* analyse de feature importance ;
* versioning Git avec tags.

---

## 📌 Sommaire

* [À propos du projet](#-à-propos-du-projet)
* [Objectifs](#-objectifs)
* [Architecture globale](#-architecture-globale)
* [Technologies utilisées](#-technologies-utilisées)
* [Structure du projet](#-structure-du-projet)
* [Installation locale](#-installation-locale)
* [Données utilisées](#-données-utilisées)
* [Préparation et feature engineering](#-préparation-et-feature-engineering)
* [Modélisation](#-modélisation)
* [Tracking MLflow](#-tracking-mlflow)
* [Validation croisée](#-validation-croisée)
* [Optimisation Optuna et seuil métier](#-optimisation-optuna-et-seuil-métier)
* [Feature importance](#-feature-importance)
* [Notebooks principaux](#-notebooks-principaux)
* [Scripts principaux](#-scripts-principaux)
* [Commandes opérationnelles](#-commandes-opérationnelles)
* [Résultats principaux](#-résultats-principaux)
* [Versioning](#-versioning)
* [Limites connues](#-limites-connues)
* [Contact](#-contact)

---

## 🎯 À propos du projet

Ce projet s’inscrit dans le cadre du parcours **OpenClassrooms — Data Scientist Machine Learning**.

Il vise à mettre en place une première démarche MLOps autour d’un problème de scoring crédit.

Le modèle doit prédire si un client présente un risque de défaut de paiement.

La variable cible est :

| Valeur | Signification                   |
| -----: | ------------------------------- |
|    `0` | Client non risqué / sans défaut |
|    `1` | Client risqué / en défaut       |

Le problème est donc une **classification binaire déséquilibrée**.

Le dataset contient environ **8 % de clients en défaut**, ce qui rend l’accuracy insuffisante pour évaluer correctement les modèles.

---

## ✅ Objectifs

| Objectif                                          |    Statut |
| ------------------------------------------------- | --------: |
| Explorer les données brutes                       | ✅ Réalisé |
| Nettoyer et enrichir les données                  | ✅ Réalisé |
| Construire des features agrégées                  | ✅ Réalisé |
| Entraîner plusieurs modèles de classification     | ✅ Réalisé |
| Comparer les modèles avec validation croisée      | ✅ Réalisé |
| Prendre en compte le déséquilibre des classes     | ✅ Réalisé |
| Définir une fonction de coût métier               | ✅ Réalisé |
| Tracker les expériences avec MLflow               | ✅ Réalisé |
| Annoter les expériences avec descriptions et tags | ✅ Réalisé |
| Optimiser un modèle avec Optuna                   | ✅ Réalisé |
| Optimiser le seuil métier                         | ✅ Réalisé |
| Tracer la courbe coût métier vs seuil             | ✅ Réalisé |
| Analyser la feature importance                    | ✅ Réalisé |
| Versionner le projet avec Git et tags             | ✅ Réalisé |

---

## 🏗️ Architecture globale

Le flux global du projet est le suivant :

```text
Données brutes Home Credit
        ↓
Exploration des fichiers
        ↓
Nettoyage et préparation
        ↓
Feature engineering par agrégats
        ↓
Dataset final de modélisation
        ↓
Entraînement de plusieurs modèles
        ↓
Validation croisée stratifiée
        ↓
Tracking MLflow
        ↓
Sélection du meilleur candidat
        ↓
Optimisation Optuna
        ↓
Optimisation du seuil métier
        ↓
Modèle final XGBoost optimisé
```

---

## 🧱 Technologies utilisées

| Technologie      | Rôle                                          |
| ---------------- | --------------------------------------------- |
| Python 3.12      | Langage principal                             |
| uv               | Gestion de l’environnement et des dépendances |
| pandas           | Manipulation des données                      |
| NumPy            | Calcul numérique                              |
| scikit-learn     | Modèles, métriques, validation croisée        |
| LightGBM         | Modèle de gradient boosting                   |
| XGBoost          | Modèle final optimisé                         |
| Optuna           | Optimisation des hyperparamètres              |
| MLflow           | Tracking des expériences et artefacts         |
| matplotlib       | Visualisations                                |
| Jupyter Notebook | Analyse, documentation et restitution         |
| Git / GitHub     | Versioning du projet                          |

---

## 📁 Structure du projet

```text
P6_MLOps_credit_scoring/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_exploration_donnees.ipynb
│   ├── 02_preparation_features.ipynb
│   ├── 03_modelisation_mlflow.ipynb
│   ├── 04_modelisation_validation_croisee.ipynb
│   └── 05_optimisation_hyperparametres_seuil_metier.ipynb
├── reports/
│   ├── data_quality/
│   ├── figures/
│   └── optimization/
├── src/
│   ├── config.py
│   ├── data/
│   ├── models/
│   │   ├── mlflow_utils.py
│   │   ├── train_baseline_mlflow.py
│   │   ├── train_logistic_regression_mlflow.py
│   │   ├── train_lightgbm_mlflow.py
│   │   ├── cross_validate_models_mlflow.py
│   │   ├── optimize_xgboost_optuna_threshold_mlflow.py
│   │   ├── register_best_model_mlflow.py
│   │   └── update_mlflow_experiments_metadata.py
│   └── visualization/
├── tests/
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

---

## ⚙️ Installation locale

### Prérequis

Avant de lancer le projet, installer :

* Python 3.12 ;
* Git ;
* uv ;
* Jupyter ou PyCharm ;
* MLflow.

### Cloner le repository

```bash
git clone https://github.com/VincentDesmouceaux/P6_MLOps_credit_scoring.git
cd P6_MLOps_credit_scoring
```

### Installer les dépendances

Le projet utilise `uv`.

```bash
uv sync
```

Vérifier l’environnement :

```bash
python -V
uv pip list
```

---

## 📊 Données utilisées

Le projet utilise le dataset **Home Credit Default Risk**.

Les fichiers bruts principaux sont :

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
POS_CASH_balance.csv
installments_payments.csv
credit_card_balance.csv
HomeCredit_columns_description.csv
sample_submission.csv
```

Les données brutes ne sont pas nécessairement versionnées dans Git afin d’éviter d’alourdir le repository.

Elles doivent être placées dans :

```text
data/raw/
```

---

## 🧹 Préparation et feature engineering

La préparation des données comprend :

* analyse des fichiers bruts ;
* identification des clés de jointure ;
* analyse des valeurs manquantes ;
* suppression des colonnes trop incomplètes ;
* suppression des colonnes constantes ;
* agrégation des tables secondaires ;
* fusion autour de la clé `SK_ID_CURR` ;
* création du dataset final de modélisation.

Les tables secondaires sont agrégées afin de créer des variables synthétiques par client.

Exemples de tables utilisées :

| Table                   | Rôle                                  |
| ----------------------- | ------------------------------------- |
| `bureau`                | Historique des crédits externes       |
| `bureau_balance`        | Historique mensuel des crédits bureau |
| `previous_application`  | Anciennes demandes de crédit          |
| `POS_CASH_balance`      | Historique POS/Cash                   |
| `installments_payments` | Paiements et retards                  |
| `credit_card_balance`   | Informations cartes de crédit         |

Dataset final de modélisation :

```text
data/processed/train_modeling.csv
data/processed/test_modeling.csv
```

---

## 🤖 Modélisation

Plusieurs modèles de classification ont été testés :

```text
DummyClassifier
Logistic Regression
Random Forest
LightGBM
XGBoost
MLPClassifier
```

L’objectif n’est pas uniquement de maximiser l’accuracy, mais de réduire le coût métier lié aux erreurs de classification.

---

## 📉 Coût métier

Dans un contexte de scoring crédit, toutes les erreurs n’ont pas le même impact.

| Erreur | Signification métier                  | Gravité |
| ------ | ------------------------------------- | ------- |
| FP     | Bon client prédit comme risqué        | Moyenne |
| FN     | Client risqué prédit comme bon client | Forte   |

Un faux négatif est plus grave, car il correspond à un client risqué accepté à tort.

La fonction de coût métier définie est :

```text
business_cost = 10 × FN + 1 × FP
```

Cette métrique est utilisée pour sélectionner le meilleur modèle.

---

## 🧪 Tracking MLflow

MLflow est utilisé pour suivre les expérimentations.

Les informations trackées sont :

* paramètres ;
* métriques ;
* tags ;
* descriptions ;
* artefacts ;
* modèles.

Les principales expériences MLflow sont :

| Expérience                           | Rôle                                            |
| ------------------------------------ | ----------------------------------------------- |
| `P6_credit_scoring_baseline`         | Premiers modèles de référence                   |
| `P6_credit_scoring_cross_validation` | Comparaison des modèles avec validation croisée |
| `P6_credit_scoring_optimization`     | Optimisation Optuna et seuil métier             |

### Lancer MLflow UI

```bash
mlflow ui --host localhost --port 5001 --disable-security-middleware
```

Ouvrir ensuite :

```text
http://localhost:5001
```

### Accès au Model Registry

```text
http://localhost:5001/#/models
```

---

## 🔁 Validation croisée

Les modèles sont évalués avec une validation croisée stratifiée :

```text
StratifiedKFold, 3 folds
```

Cette stratégie permet de conserver la proportion de clients en défaut dans chaque fold.

Les métriques utilisées sont :

```text
accuracy
precision
recall
F1-score
ROC AUC
business_cost
business_cost_per_client
```

Résultats principaux de la validation croisée :

| Modèle              | Accuracy moyenne | Precision moyenne | Recall moyen | F1 moyen | AUC moyenne | Coût métier moyen |
| ------------------- | ---------------: | ----------------: | -----------: | -------: | ----------: | ----------------: |
| Logistic Regression |            0.706 |             0.169 |        0.676 |    0.271 |       0.755 |             14114 |
| Random Forest       |            0.727 |             0.169 |        0.609 |    0.265 |       0.737 |             14860 |
| LightGBM            |            0.799 |             0.211 |        0.543 |    0.304 |       0.765 |             14213 |
| XGBoost             |            0.748 |             0.188 |        0.637 |    0.290 |       0.769 |          13742.67 |
| MLPClassifier       |            0.919 |             0.317 |        0.008 |    0.015 |       0.722 |             21380 |

XGBoost est retenu comme meilleur candidat avant optimisation, car il obtient le coût métier moyen le plus faible.

---

## 🎯 Optimisation Optuna et seuil métier

L’étape finale consiste à optimiser XGBoost avec Optuna.

L’objectif d’optimisation est :

```text
minimiser le coût métier moyen en validation croisée
```

Les hyperparamètres optimisés incluent :

```text
n_estimators
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda
scale_pos_weight
```

Plusieurs seuils de décision sont testés :

```text
0.10 à 0.90
```

Le meilleur seuil trouvé est :

```text
threshold = 0.45
```

Résultats après optimisation :

| Version                    | Seuil | Recall moyen | AUC moyenne | Coût métier moyen |
| -------------------------- | ----: | -----------: | ----------: | ----------------: |
| XGBoost avant optimisation |  0.50 |        0.637 |       0.769 |          13742.67 |
| XGBoost optimisé           |  0.45 |        0.654 |       0.771 |          13609.33 |

Gain obtenu :

```text
13742.67 - 13609.33 = 133.34 points de coût métier
```

---

## 📌 Hyperparamètres du modèle final

Les meilleurs hyperparamètres trouvés avec Optuna sont :

| Hyperparamètre                |  Valeur |
| ----------------------------- | ------: |
| `scale_pos_weight_multiplier` |  0.8686 |
| `n_estimators`                |     400 |
| `learning_rate`               |  0.0461 |
| `max_depth`                   |       4 |
| `min_child_weight`            |      28 |
| `subsample`                   |  0.7732 |
| `colsample_bytree`            |  0.9510 |
| `gamma`                       |  1.1710 |
| `reg_alpha`                   | 0.00095 |
| `reg_lambda`                  | 12.2134 |

Le modèle final est un **XGBoost optimisé** avec un seuil métier ajusté à **0.45**.

---

## 📊 Feature importance

Une analyse de l’importance des variables a été ajoutée pour améliorer l’interprétabilité du modèle.

L’importance utilisée est de type `gain`.

Elle permet d’identifier les variables qui contribuent le plus aux décisions du modèle.

Attention : la feature importance indique une contribution prédictive, mais ne prouve pas une relation causale directe avec le défaut de paiement.

Fichiers générés :

```text
reports/figures/xgboost_feature_importance_top20.png
reports/optimization/xgboost_feature_importance_gain.csv
```

---

## 📓 Notebooks principaux

| Notebook                                             | Rôle                                                  |
| ---------------------------------------------------- | ----------------------------------------------------- |
| `01_exploration_donnees.ipynb`                       | Exploration des fichiers bruts                        |
| `02_preparation_features.ipynb`                      | Nettoyage, jointures, feature engineering             |
| `03_modelisation_mlflow.ipynb`                       | Premiers modèles et tracking MLflow                   |
| `04_modelisation_validation_croisee.ipynb`           | Comparaison des modèles avec validation croisée       |
| `05_optimisation_hyperparametres_seuil_metier.ipynb` | Optimisation Optuna, seuil métier, feature importance |

Les notebooks les plus importants pour la restitution sont :

```text
04_modelisation_validation_croisee.ipynb
05_optimisation_hyperparametres_seuil_metier.ipynb
```

---

## 🧾 Scripts principaux

| Script                                                   | Rôle                                              |
| -------------------------------------------------------- | ------------------------------------------------- |
| `src/models/train_baseline_mlflow.py`                    | Entraîne le DummyClassifier baseline              |
| `src/models/train_logistic_regression_mlflow.py`         | Entraîne la régression logistique                 |
| `src/models/train_lightgbm_mlflow.py`                    | Entraîne LightGBM                                 |
| `src/models/cross_validate_models_mlflow.py`             | Compare plusieurs modèles avec validation croisée |
| `src/models/optimize_xgboost_optuna_threshold_mlflow.py` | Optimise XGBoost avec Optuna et seuil métier      |
| `src/models/register_best_model_mlflow.py`               | Enregistre un modèle dans le Model Registry       |
| `src/models/update_mlflow_experiments_metadata.py`       | Ajoute descriptions et tags MLflow                |
| `src/models/mlflow_utils.py`                             | Ajoute des informations Git aux runs MLflow       |

---

## 🧰 Commandes opérationnelles

### Vérifier l’état Git

```bash
git status
git log --oneline --decorate -5
```

### Lancer la validation croisée

```bash
python -m src.models.cross_validate_models_mlflow
```

### Lancer l’optimisation Optuna

```bash
python -m src.models.optimize_xgboost_optuna_threshold_mlflow
```

### Ajouter les métadonnées MLflow

```bash
python -m src.models.update_mlflow_experiments_metadata
```

### Lancer MLflow UI

```bash
mlflow ui --host localhost --port 5001 --disable-security-middleware
```

### Ouvrir MLflow

```text
http://localhost:5001
```

### Ouvrir le Model Registry

```text
http://localhost:5001/#/models
```

---

## 📈 Résultats principaux

### Meilleur modèle avant optimisation

```text
XGBoost
Coût métier moyen : 13742.67
Recall moyen : 0.637
AUC moyenne : 0.769
Seuil : 0.50
```

### Modèle final optimisé

```text
XGBoost optimisé avec Optuna
Coût métier moyen : 13609.33
Recall moyen : 0.6539
AUC moyenne : 0.7713
Seuil métier : 0.45
```

### Interprétation

Le seuil final est inférieur à 0.50, ce qui est cohérent avec le contexte métier.

Comme les faux négatifs coûtent plus cher que les faux positifs, le modèle doit détecter davantage de clients risqués, même si cela augmente les faux positifs.

---

## 🚀 Versioning

Le projet utilise Git et des tags pour versionner les étapes importantes.

| Version  | Contenu                                                                       |
| -------- | ----------------------------------------------------------------------------- |
| `v0.1.0` | Préparation des données, premiers modèles, MLflow et validation croisée       |
| `v0.2.0` | Validation croisée complète, optimisation XGBoost avec Optuna et seuil métier |

Créer un tag :

```bash
git tag -a v0.2.0 -m "Version 0.2.0 - Pipeline d'entrainement, validation croisee et optimisation XGBoost"
git push origin v0.2.0
```

---

## ⚠️ Limites connues

* Le dataset est fortement déséquilibré.
* La precision reste faible, car le modèle détecte beaucoup de clients comme risqués.
* Le recall est privilégié car les faux négatifs coûtent plus cher.
* L’optimisation a été faite sur un échantillon stratifié de 80 000 lignes pour limiter le temps de calcul.
* L’augmentation du nombre de trials Optuna pourrait améliorer les résultats.
* La feature importance ne prouve pas une causalité.
* Une analyse SHAP pourrait être ajoutée pour améliorer l’interprétabilité.
* Le Model Registry contient un premier modèle champion LightGBM ; la suite logique serait d’enregistrer XGBoost optimisé comme nouveau champion.

---

## 🗺️ Roadmap

* [x] Créer la structure du projet
* [x] Explorer les données brutes
* [x] Nettoyer les données
* [x] Construire les features agrégées
* [x] Créer le dataset final de modélisation
* [x] Entraîner des modèles baseline
* [x] Tracker les expériences avec MLflow
* [x] Comparer plusieurs modèles avec validation croisée
* [x] Prendre en compte le déséquilibre des classes
* [x] Définir une fonction de coût métier
* [x] Optimiser XGBoost avec Optuna
* [x] Optimiser le seuil métier
* [x] Tracer la courbe coût vs seuil
* [x] Ajouter une analyse de feature importance
* [x] Versionner avec Git et tags
* [ ] Enregistrer XGBoost optimisé comme nouveau champion dans le Model Registry
* [ ] Ajouter une analyse SHAP
* [ ] Industrialiser davantage le pipeline d’entraînement

---

## 👤 Contact

Projet réalisé par **Vincent Desmouceaux**.

GitHub : [VincentDesmouceaux](https://github.com/VincentDesmouceaux)
Email : [desmontvincent@gmail.com](mailto:desmontvincent@gmail.com)

Lien du projet :

```text
https://github.com/VincentDesmouceaux/P6_MLOps_credit_scoring
```

---

## ✅ Résumé

Ce projet démontre une première chaîne MLOps appliquée au scoring crédit :

```text
Données Home Credit
→ Feature engineering
→ Modèles de classification
→ Validation croisée
→ Tracking MLflow
→ Optimisation Optuna
→ Seuil métier
→ Feature importance
→ Versioning Git
```

Le modèle final retenu est un **XGBoost optimisé avec Optuna**, utilisant un **seuil métier de 0.45**, choisi pour minimiser un coût métier pondérant fortement les faux négatifs.
