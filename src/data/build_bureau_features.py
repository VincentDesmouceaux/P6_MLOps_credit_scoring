import gc

import pandas as pd

from src.config import INTERIM_DATA_DIR
from src.data.load_data import load_raw_table


def one_hot_encode_dataframe(
    df: pd.DataFrame,
    nan_as_category: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode les variables catégorielles avec pandas.get_dummies.

    Objectif :
    - transformer les colonnes texte en colonnes numériques 0/1 ;
    - conserver la liste des nouvelles colonnes créées ;
    - permettre ensuite les agrégations numériques.
    """
    original_columns = list(df.columns)

    # Sélection des colonnes catégorielles.
    categorical_columns = list(
        df.select_dtypes(include=["object", "string"]).columns
    )

    # Encodage one-hot.
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dummy_na=nan_as_category,
    )

    # Colonnes créées par l'encodage.
    new_columns = [col for col in df.columns if col not in original_columns]

    return df, new_columns


def aggregate_bureau_balance() -> pd.DataFrame:
    """
    Agrège bureau_balance.csv au niveau SK_ID_BUREAU.

    bureau_balance contient l'historique mensuel des crédits bureau.
    Il y a plusieurs lignes par SK_ID_BUREAU.

    Objectif :
    - obtenir une seule ligne par SK_ID_BUREAU ;
    - créer des statistiques simples sur MONTHS_BALANCE ;
    - agréger les statuts mensuels encodés.
    """
    print("Chargement de bureau_balance...")
    bureau_balance = load_raw_table("bureau_balance")

    print(f"bureau_balance shape brute : {bureau_balance.shape}")

    # Encodage des statuts catégoriels, par exemple STATUS_0, STATUS_1, etc.
    bureau_balance, bureau_balance_cat_cols = one_hot_encode_dataframe(
        bureau_balance,
        nan_as_category=True,
    )

    # Agrégations numériques principales.
    aggregations = {
        "MONTHS_BALANCE": ["min", "max", "size"],
    }

    # Pour les colonnes catégorielles encodées, on calcule la moyenne.
    # Cela donne la proportion d'apparition de chaque statut.
    for col in bureau_balance_cat_cols:
        aggregations[col] = ["mean"]

    print("Agrégation de bureau_balance par SK_ID_BUREAU...")
    bureau_balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(aggregations)

    # Aplatir les noms de colonnes multi-index.
    bureau_balance_agg.columns = pd.Index(
        [
            f"BB_{col_name}_{agg_name}".upper()
            for col_name, agg_name in bureau_balance_agg.columns.tolist()
        ]
    )

    # SK_ID_BUREAU redevient une colonne normale.
    bureau_balance_agg = bureau_balance_agg.reset_index()

    print(f"bureau_balance agrégée shape : {bureau_balance_agg.shape}")

    del bureau_balance
    gc.collect()

    return bureau_balance_agg


def aggregate_bureau() -> pd.DataFrame:
    """
    Construit les features issues de bureau.csv et bureau_balance.csv.

    bureau.csv contient les crédits précédents du client dans d'autres
    institutions financières.

    Il peut y avoir plusieurs lignes bureau par client SK_ID_CURR.
    On doit donc agréger au niveau client avant fusion avec application_base.
    """
    print("Chargement de bureau...")
    bureau = load_raw_table("bureau")

    print(f"bureau shape brute : {bureau.shape}")

    # On agrège d'abord bureau_balance au niveau SK_ID_BUREAU.
    bureau_balance_agg = aggregate_bureau_balance()

    print("Jointure bureau + bureau_balance agrégée sur SK_ID_BUREAU...")
    bureau = bureau.merge(
        bureau_balance_agg,
        how="left",
        on="SK_ID_BUREAU",
    )

    print(f"bureau shape après jointure bureau_balance : {bureau.shape}")

    # Encodage des variables catégorielles de bureau.
    bureau, bureau_cat_cols = one_hot_encode_dataframe(
        bureau,
        nan_as_category=True,
    )

    # On supprime SK_ID_BUREAU avant l'agrégation client :
    # il sert à joindre bureau_balance, mais ce n'est pas une feature client.
    if "SK_ID_BUREAU" in bureau.columns:
        bureau = bureau.drop(columns=["SK_ID_BUREAU"])

    # Colonnes numériques à agréger si elles existent dans la table.
    numeric_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
    }

    # On garde uniquement les colonnes réellement présentes.
    numeric_aggregations = {
        column: aggregations
        for column, aggregations in numeric_aggregations.items()
        if column in bureau.columns
    }

    # Pour les colonnes catégorielles encodées, la moyenne donne une proportion.
    categorical_aggregations = {
        col: ["mean"]
        for col in bureau_cat_cols
        if col in bureau.columns
    }

    # On ajoute les colonnes de bureau_balance déjà agrégées au niveau bureau.
    # Elles commencent par BB_.
    bureau_balance_aggregations = {
        col: ["mean"]
        for col in bureau.columns
        if col.startswith("BB_")
    }

    all_aggregations = {
        **numeric_aggregations,
        **categorical_aggregations,
        **bureau_balance_aggregations,
    }

    print("Agrégation de bureau au niveau client SK_ID_CURR...")
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(all_aggregations)

    # Aplatir les noms de colonnes multi-index.
    bureau_agg.columns = pd.Index(
        [
            f"BURO_{col_name}_{agg_name}".upper()
            for col_name, agg_name in bureau_agg.columns.tolist()
        ]
    )

    bureau_agg = bureau_agg.reset_index()

    # Feature simple : nombre de crédits bureau par client.
    credit_count = (
        bureau.groupby("SK_ID_CURR")
        .size()
        .reset_index(name="BURO_CREDIT_COUNT")
    )

    bureau_agg = bureau_agg.merge(
        credit_count,
        how="left",
        on="SK_ID_CURR",
    )

    print(f"bureau_features shape finale : {bureau_agg.shape}")

    del bureau, bureau_balance_agg
    gc.collect()

    return bureau_agg


def main() -> None:
    """
    Point d'entrée du script.

    Il génère :
    data/interim/bureau_features.csv

    Ce fichier contient une seule ligne par client SK_ID_CURR.
    Il pourra ensuite être fusionné avec application_base.csv.
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    bureau_features = aggregate_bureau()

    output_path = INTERIM_DATA_DIR / "bureau_features.csv"
    bureau_features.to_csv(output_path, index=False)

    print(f"\nFichier sauvegardé : {output_path}")
    print("Préparation des features bureau terminée.")


if __name__ == "__main__":
    main()