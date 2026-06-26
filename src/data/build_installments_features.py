import gc

import pandas as pd

from src.config import INTERIM_DATA_DIR
from src.data.load_data import load_raw_table


def clean_installments_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et prépare la table installments_payments.

    Cette table contient les paiements réalisés sur les anciens crédits.
    On crée ici des indicateurs métier importants :
    - paiement partiel ou complet ;
    - retard de paiement ;
    - paiement en avance ;
    - différence entre montant attendu et montant payé.
    """
    df = df.copy()

    # Ratio entre le montant payé et le montant attendu.
    # Si PAYMENT_PERC < 1 : le client a payé moins que prévu.
    # Si PAYMENT_PERC > 1 : le client a payé plus que prévu.
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(df.columns):
        df["PAYMENT_PERC"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"]

    # Différence entre le montant attendu et le montant payé.
    # Une valeur positive signifie qu'il manque une partie du paiement.
    if {"AMT_INSTALMENT", "AMT_PAYMENT"}.issubset(df.columns):
        df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

    # DPD = Days Past Due.
    # Nombre de jours de retard.
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(df.columns):
        df["DPD"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["DPD"] = df["DPD"].apply(lambda value: value if value > 0 else 0)

    # DBD = Days Before Due.
    # Nombre de jours de paiement en avance.
    if {"DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"}.issubset(df.columns):
        df["DBD"] = df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"]
        df["DBD"] = df["DBD"].apply(lambda value: value if value > 0 else 0)

    return df


def aggregate_installments_payments() -> pd.DataFrame:
    """
    Agrège installments_payments au niveau client SK_ID_CURR.

    installments_payments contient plusieurs lignes par client et par ancien
    crédit. L'objectif est de créer une seule ligne par client avant fusion
    avec application_base.
    """
    print("Chargement de installments_payments...")
    installments = load_raw_table("installments_payments")

    print(f"installments_payments shape brute : {installments.shape}")

    # Création des indicateurs de paiement.
    installments = clean_installments_payments(installments)

    # Agrégations numériques principales.
    numeric_aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "NUM_INSTALMENT_NUMBER": ["max", "mean"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
        "DAYS_INSTALMENT": ["max", "mean", "sum"],
    }

    # On garde uniquement les colonnes disponibles dans le fichier.
    numeric_aggregations = {
        column: aggregations
        for column, aggregations in numeric_aggregations.items()
        if column in installments.columns
    }

    print("Agrégation de installments_payments par client SK_ID_CURR...")
    installments_agg = installments.groupby("SK_ID_CURR").agg(numeric_aggregations)

    # Aplatir les noms de colonnes multi-index.
    installments_agg.columns = pd.Index(
        [
            f"INSTAL_{column}_{aggregation}".upper()
            for column, aggregation in installments_agg.columns.tolist()
        ]
    )

    installments_agg = installments_agg.reset_index()

    # Nombre total de paiements enregistrés par client.
    installments_count = (
        installments.groupby("SK_ID_CURR")
        .size()
        .reset_index(name="INSTAL_COUNT")
    )

    installments_agg = installments_agg.merge(
        installments_count,
        how="left",
        on="SK_ID_CURR",
    )

    print(f"installments_features shape finale : {installments_agg.shape}")

    del installments
    gc.collect()

    return installments_agg


def main() -> None:
    """
    Point d'entrée du script.

    Il génère :
    data/interim/installments_features.csv

    Ce fichier contient une seule ligne par client SK_ID_CURR.
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    installments_features = aggregate_installments_payments()

    output_path = INTERIM_DATA_DIR / "installments_features.csv"
    installments_features.to_csv(output_path, index=False)

    print(f"\nFichier sauvegardé : {output_path}")
    print("Préparation des features installments_payments terminée.")


if __name__ == "__main__":
    main()