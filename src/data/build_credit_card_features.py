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

    Dans credit_card_balance, certaines colonnes comme NAME_CONTRACT_STATUS
    sont textuelles. On les transforme en colonnes numériques 0/1 afin de
    pouvoir les agréger au niveau client.
    """
    original_columns = list(df.columns)

    # Sélection des colonnes texte.
    categorical_columns = list(
        df.select_dtypes(include=["object", "string"]).columns
    )

    # Encodage one-hot.
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dummy_na=nan_as_category,
    )

    # Liste des nouvelles colonnes créées.
    new_columns = [col for col in df.columns if col not in original_columns]

    return df, new_columns


def aggregate_credit_card_balance() -> pd.DataFrame:
    """
    Agrège credit_card_balance au niveau client SK_ID_CURR.

    La table credit_card_balance contient l'historique mensuel des cartes de
    crédit précédentes. Il peut y avoir plusieurs lignes par client.

    L'objectif est de créer une seule ligne par client avant fusion avec la
    table principale application_base.
    """
    print("Chargement de credit_card_balance...")
    credit_card = load_raw_table("credit_card_balance")

    print(f"credit_card_balance shape brute : {credit_card.shape}")

    # Encodage des variables catégorielles.
    credit_card, credit_card_cat_cols = one_hot_encode_dataframe(
        credit_card,
        nan_as_category=True,
    )

    # SK_ID_PREV identifie un ancien crédit, mais pour notre dataset final
    # on veut des features au niveau client SK_ID_CURR.
    if "SK_ID_PREV" in credit_card.columns:
        credit_card = credit_card.drop(columns=["SK_ID_PREV"])

    # Toutes les colonnes numériques restantes peuvent être agrégées.
    # On exclut uniquement SK_ID_CURR, qui est la clé client.
    numeric_columns = [
        column
        for column in credit_card.select_dtypes(include=["number"]).columns
        if column != "SK_ID_CURR"
    ]

    # Agrégations générales : min, max, moyenne, somme, variance.
    aggregations = {
        column: ["min", "max", "mean", "sum", "var"]
        for column in numeric_columns
    }

    # Les colonnes catégorielles encodées sont déjà numériques.
    # On ajoute une moyenne pour obtenir une proportion par client.
    for column in credit_card_cat_cols:
        if column in credit_card.columns:
            aggregations[column] = ["mean"]

    print("Agrégation de credit_card_balance par client SK_ID_CURR...")
    credit_card_agg = credit_card.groupby("SK_ID_CURR").agg(aggregations)

    # Aplatir les noms de colonnes multi-index.
    credit_card_agg.columns = pd.Index(
        [
            f"CC_{column}_{aggregation}".upper()
            for column, aggregation in credit_card_agg.columns.tolist()
        ]
    )

    credit_card_agg = credit_card_agg.reset_index()

    # Nombre total de lignes carte bancaire par client.
    credit_card_count = (
        credit_card.groupby("SK_ID_CURR")
        .size()
        .reset_index(name="CC_COUNT")
    )

    credit_card_agg = credit_card_agg.merge(
        credit_card_count,
        how="left",
        on="SK_ID_CURR",
    )

    print(f"credit_card_features shape finale : {credit_card_agg.shape}")

    del credit_card
    gc.collect()

    return credit_card_agg


def main() -> None:
    """
    Point d'entrée du script.

    Il génère :
    data/interim/credit_card_features.csv

    Ce fichier contient une seule ligne par client SK_ID_CURR.
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    credit_card_features = aggregate_credit_card_balance()

    output_path = INTERIM_DATA_DIR / "credit_card_features.csv"
    credit_card_features.to_csv(output_path, index=False)

    print(f"\nFichier sauvegardé : {output_path}")
    print("Préparation des features credit_card_balance terminée.")


if __name__ == "__main__":
    main()