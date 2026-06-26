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

    Objectif :
    - transformer les colonnes texte en colonnes numériques 0/1 ;
    - conserver une trace des nouvelles colonnes créées ;
    - rendre le dataset compatible avec les modèles scikit-learn / LightGBM.

    Exemple :
    colonne CODE_GENDER = "M" / "F"
    devient CODE_GENDER_M et CODE_GENDER_F.
    """
    original_columns = list(df.columns)

    # On sélectionne les colonnes de type texte.
    categorical_columns = list(
        df.select_dtypes(include=["object", "string"]).columns
    )

    # On applique le one-hot encoding.
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dummy_na=nan_as_category,
    )

    # On récupère uniquement les nouvelles colonnes créées.
    new_columns = [col for col in df.columns if col not in original_columns]

    return df, new_columns


def clean_application_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les anomalies simples de la table application.

    Dans le dataset Home Credit, la colonne DAYS_EMPLOYED contient parfois
    la valeur 365243, qui correspond à une anomalie métier connue.
    On la remplace par NaN pour permettre une imputation plus propre ensuite.
    """
    df = df.copy()

    # Remplacement de l'anomalie DAYS_EMPLOYED = 365243 par NaN.
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # On retire les rares lignes avec CODE_GENDER = XNA si la colonne existe.
    if "CODE_GENDER" in df.columns:
        df = df[df["CODE_GENDER"] != "XNA"]

    return df


def create_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features métier simples à partir des colonnes principales.

    Ces ratios sont utiles pour le scoring crédit :
    - rapport entre ancienneté professionnelle et âge ;
    - rapport entre revenu et montant du crédit ;
    - revenu par personne dans le foyer ;
    - poids de l'annuité dans le revenu ;
    - taux de paiement annuel par rapport au crédit.
    """
    df = df.copy()

    # Ancienneté professionnelle relative à l'âge.
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    # Ratio revenu / montant du crédit.
    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df.columns):
        df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]

    # Revenu moyen par membre du foyer.
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    # Poids de l'annuité dans le revenu.
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # Rapport entre annuité et montant total du crédit.
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    return df


def build_application_base() -> pd.DataFrame:
    """
    Construit la première table client propre à partir de :
    - application_train.csv ;
    - application_test.csv.

    Cette table servira de base pour les futures jointures avec les tables
    bureau, previous_application, installments, etc.
    """
    print("Chargement de application_train...")
    train_df = load_raw_table("application_train")

    print("Chargement de application_test...")
    test_df = load_raw_table("application_test")

    # On ajoute une colonne technique pour savoir d'où vient chaque ligne.
    train_df["dataset_type"] = "train"
    test_df["dataset_type"] = "test"

    # La table test n'a pas TARGET. On l'ajoute à NaN pour aligner les colonnes.
    if "TARGET" not in test_df.columns:
        test_df["TARGET"] = np.nan

    print(f"Train shape avant concat : {train_df.shape}")
    print(f"Test shape avant concat  : {test_df.shape}")

    # On concatène train et test pour appliquer les mêmes transformations.
    application_df = pd.concat(
        [train_df, test_df],
        axis=0,
        ignore_index=True,
    )

    print(f"Application shape après concat : {application_df.shape}")

    # Nettoyage des anomalies.
    application_df = clean_application_data(application_df)

    # Création des ratios métier.
    application_df = create_application_features(application_df)

    # Encodage des variables catégorielles.
    application_df, encoded_columns = one_hot_encode_dataframe(
        application_df,
        nan_as_category=True,
    )

    print(f"Nombre de colonnes encodées créées : {len(encoded_columns)}")
    print(f"Application shape finale : {application_df.shape}")

    # Nettoyage mémoire.
    del train_df, test_df
    gc.collect()

    return application_df


def main() -> None:
    """
    Point d'entrée du script.

    Il construit la table application enrichie et la sauvegarde dans :
    data/interim/application_base.csv
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    application_df = build_application_base()

    output_path = INTERIM_DATA_DIR / "application_base.csv"
    application_df.to_csv(output_path, index=False)

    print(f"\nTable application sauvegardée : {output_path}")
    print("Préparation application terminée.")


if __name__ == "__main__":
    main()