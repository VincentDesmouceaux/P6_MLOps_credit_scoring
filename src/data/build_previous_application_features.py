import gc

import numpy as np
import pandas as pd

from src.config import INTERIM_DATA_DIR
from src.data.load_data import load_raw_table


def one_hot_encode_dataframe(
    df: pd.DataFrame,
    nan_as_category: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode les variables catégorielles avec pandas.get_dummies.

    Dans previous_application, plusieurs colonnes sont textuelles :
    - type de contrat ;
    - statut de la demande ;
    - canal de vente ;
    - type de produit ;
    - etc.

    Pour pouvoir agréger ces informations au niveau client, on transforme
    chaque modalité en colonne binaire 0/1.
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

    # Liste des colonnes créées par l'encodage.
    new_columns = [col for col in df.columns if col not in original_columns]

    return df, new_columns


def clean_previous_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les anomalies connues dans previous_application.

    Dans Home Credit, certaines colonnes de dates utilisent 365243 comme
    valeur spéciale. Cette valeur ne correspond pas à une vraie date métier.
    On la remplace donc par NaN.
    """
    df = df.copy()

    # Colonnes de dates relatives où 365243 représente une anomalie.
    date_columns = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = df[column].replace(365243, np.nan)

    return df


def create_previous_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des ratios métier simples sur les anciennes demandes.

    Exemple important :
    - APP_CREDIT_PERC = montant demandé / montant accordé

    Ce ratio permet d'identifier si le montant réellement accordé était proche
    ou non du montant demandé lors des crédits précédents.
    """
    df = df.copy()

    # Ratio entre le montant demandé et le montant du crédit accordé.
    if {"AMT_APPLICATION", "AMT_CREDIT"}.issubset(df.columns):
        df["APP_CREDIT_PERC"] = df["AMT_APPLICATION"] / df["AMT_CREDIT"]

    return df


def aggregate_previous_application() -> pd.DataFrame:
    """
    Agrège previous_application au niveau client SK_ID_CURR.

    La table previous_application contient plusieurs anciennes demandes par
    client. Pour la fusion avec application_base, on doit obtenir une seule
    ligne par client.
    """
    print("Chargement de previous_application...")
    previous = load_raw_table("previous_application")

    print(f"previous_application shape brute : {previous.shape}")

    # Nettoyage des valeurs anormales.
    previous = clean_previous_application(previous)

    # Création de ratios métier.
    previous = create_previous_application_features(previous)

    # Encodage des variables catégorielles.
    previous, previous_cat_cols = one_hot_encode_dataframe(
        previous,
        nan_as_category=True,
    )

    # Colonnes numériques à agréger.
    numeric_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }

    # On garde uniquement les colonnes réellement présentes.
    numeric_aggregations = {
        column: aggregations
        for column, aggregations in numeric_aggregations.items()
        if column in previous.columns
    }

    # Les colonnes catégorielles encodées sont agrégées par moyenne.
    # Cela donne une proportion par client.
    categorical_aggregations = {
        column: ["mean"]
        for column in previous_cat_cols
        if column in previous.columns
    }

    all_aggregations = {
        **numeric_aggregations,
        **categorical_aggregations,
    }

    print("Agrégation globale par client SK_ID_CURR...")
    previous_agg = previous.groupby("SK_ID_CURR").agg(all_aggregations)

    # Aplatir les noms de colonnes multi-index.
    previous_agg.columns = pd.Index(
        [
            f"PREV_{column}_{aggregation}".upper()
            for column, aggregation in previous_agg.columns.tolist()
        ]
    )

    previous_agg = previous_agg.reset_index()

    # Nombre total d'anciennes demandes par client.
    previous_count = (
        previous.groupby("SK_ID_CURR")
        .size()
        .reset_index(name="PREV_APPLICATION_COUNT")
    )

    previous_agg = previous_agg.merge(
        previous_count,
        how="left",
        on="SK_ID_CURR",
    )

    # Agrégations spécifiques sur les anciennes demandes acceptées.
    previous_agg = add_status_specific_aggregations(
        previous=previous,
        previous_agg=previous_agg,
        status_column="NAME_CONTRACT_STATUS_Approved",
        prefix="APPROVED",
    )

    # Agrégations spécifiques sur les anciennes demandes refusées.
    previous_agg = add_status_specific_aggregations(
        previous=previous,
        previous_agg=previous_agg,
        status_column="NAME_CONTRACT_STATUS_Refused",
        prefix="REFUSED",
    )

    print(f"previous_application_features shape finale : {previous_agg.shape}")

    del previous
    gc.collect()

    return previous_agg


def add_status_specific_aggregations(
    previous: pd.DataFrame,
    previous_agg: pd.DataFrame,
    status_column: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Ajoute des agrégations spécifiques selon le statut de la demande.

    Exemple :
    - anciennes demandes acceptées ;
    - anciennes demandes refusées.

    Cela permet de distinguer le comportement passé du client selon le type
    de décision prise sur ses anciennes demandes.
    """
    if status_column not in previous.columns:
        print(f"Colonne absente, agrégation ignorée : {status_column}")
        return previous_agg

    status_df = previous[previous[status_column] == 1]

    if status_df.empty:
        print(f"Aucune ligne trouvée pour : {status_column}")
        return previous_agg

    status_numeric_aggregations = {
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"],
        "APP_CREDIT_PERC": ["mean"],
        "DAYS_DECISION": ["mean"],
        "CNT_PAYMENT": ["mean"],
    }

    # On garde uniquement les colonnes disponibles.
    status_numeric_aggregations = {
        column: aggregations
        for column, aggregations in status_numeric_aggregations.items()
        if column in status_df.columns
    }

    print(f"Agrégation spécifique : {prefix}...")
    status_agg = status_df.groupby("SK_ID_CURR").agg(status_numeric_aggregations)

    status_agg.columns = pd.Index(
        [
            f"{prefix}_{column}_{aggregation}".upper()
            for column, aggregation in status_agg.columns.tolist()
        ]
    )

    status_agg = status_agg.reset_index()

    # Nombre de demandes avec ce statut.
    status_count = (
        status_df.groupby("SK_ID_CURR")
        .size()
        .reset_index(name=f"{prefix}_APPLICATION_COUNT")
    )

    status_agg = status_agg.merge(
        status_count,
        how="left",
        on="SK_ID_CURR",
    )

    previous_agg = previous_agg.merge(
        status_agg,
        how="left",
        on="SK_ID_CURR",
    )

    return previous_agg


def main() -> None:
    """
    Point d'entrée du script.

    Il génère :
    data/interim/previous_application_features.csv

    Ce fichier contient une seule ligne par client SK_ID_CURR.
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    previous_features = aggregate_previous_application()

    output_path = INTERIM_DATA_DIR / "previous_application_features.csv"
    previous_features.to_csv(output_path, index=False)

    print(f"\nFichier sauvegardé : {output_path}")
    print("Préparation des features previous_application terminée.")


if __name__ == "__main__":
    main()