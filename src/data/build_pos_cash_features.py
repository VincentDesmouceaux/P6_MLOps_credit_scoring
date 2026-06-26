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

    Dans POS_CASH_balance, la colonne NAME_CONTRACT_STATUS est catégorielle.
    On l'encode pour pouvoir calculer des proportions par client.
    """
    original_columns = list(df.columns)

    # Sélection des colonnes texte.
    categorical_columns = list(
        df.select_dtypes(include=["object", "string"]).columns
    )

    # Encodage one-hot des variables catégorielles.
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dummy_na=nan_as_category,
    )

    # Liste des nouvelles colonnes créées.
    new_columns = [col for col in df.columns if col not in original_columns]

    return df, new_columns


def aggregate_pos_cash() -> pd.DataFrame:
    """
    Agrège POS_CASH_balance au niveau client SK_ID_CURR.

    POS_CASH_balance contient plusieurs lignes par client et par ancien crédit.
    L'objectif est de produire une seule ligne par client avant la fusion
    avec application_base.
    """
    print("Chargement de POS_CASH_balance...")
    pos_cash = load_raw_table("pos_cash_balance")

    print(f"POS_CASH_balance shape brute : {pos_cash.shape}")

    # Encodage des statuts de contrat.
    pos_cash, pos_cash_cat_cols = one_hot_encode_dataframe(
        pos_cash,
        nan_as_category=True,
    )

    # Agrégations numériques principales.
    numeric_aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "CNT_INSTALMENT": ["mean", "sum"],
        "CNT_INSTALMENT_FUTURE": ["mean", "sum"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }

    # On conserve uniquement les colonnes réellement présentes.
    numeric_aggregations = {
        column: aggregations
        for column, aggregations in numeric_aggregations.items()
        if column in pos_cash.columns
    }

    # Les colonnes catégorielles encodées sont agrégées par moyenne.
    # La moyenne correspond à la proportion d'un statut pour un client.
    categorical_aggregations = {
        column: ["mean"]
        for column in pos_cash_cat_cols
        if column in pos_cash.columns
    }

    all_aggregations = {
        **numeric_aggregations,
        **categorical_aggregations,
    }

    print("Agrégation de POS_CASH_balance par client SK_ID_CURR...")
    pos_cash_agg = pos_cash.groupby("SK_ID_CURR").agg(all_aggregations)

    # Aplatir les noms de colonnes multi-index.
    pos_cash_agg.columns = pd.Index(
        [
            f"POS_{column}_{aggregation}".upper()
            for column, aggregation in pos_cash_agg.columns.tolist()
        ]
    )

    pos_cash_agg = pos_cash_agg.reset_index()

    # Nombre total de lignes POS/Cash par client.
    pos_cash_count = (
        pos_cash.groupby("SK_ID_CURR")
        .size()
        .reset_index(name="POS_CASH_COUNT")
    )

    pos_cash_agg = pos_cash_agg.merge(
        pos_cash_count,
        how="left",
        on="SK_ID_CURR",
    )

    print(f"pos_cash_features shape finale : {pos_cash_agg.shape}")

    del pos_cash
    gc.collect()

    return pos_cash_agg


def main() -> None:
    """
    Point d'entrée du script.

    Il génère :
    data/interim/pos_cash_features.csv

    Ce fichier contient une seule ligne par client SK_ID_CURR.
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pos_cash_features = aggregate_pos_cash()

    output_path = INTERIM_DATA_DIR / "pos_cash_features.csv"
    pos_cash_features.to_csv(output_path, index=False)

    print(f"\nFichier sauvegardé : {output_path}")
    print("Préparation des features POS_CASH_balance terminée.")


if __name__ == "__main__":
    main()