import gc

import numpy as np
import pandas as pd

from src.config import DATA_QUALITY_DIR, PROCESSED_DATA_DIR


TRAIN_INPUT_FILE = "train_processed.csv"
TEST_INPUT_FILE = "test_processed.csv"

TRAIN_OUTPUT_FILE = "train_modeling.csv"
TEST_OUTPUT_FILE = "test_modeling.csv"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

HIGH_MISSING_THRESHOLD = 0.80


def load_processed_dataset(file_name: str) -> pd.DataFrame:
    """
    Charge un fichier depuis data/processed.
    """
    file_path = PROCESSED_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"Chargement : {file_path.name}")
    return pd.read_csv(file_path)


def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs infinies par NaN.

    Certaines features créées par division peuvent produire :
    - inf ;
    - -inf.

    Les modèles ou imputers gèrent mieux NaN que inf.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes booléennes True/False en 1/0.

    Cela rend le dataset plus simple à utiliser avec les modèles.
    """
    df = df.copy()

    bool_columns = df.select_dtypes(include=["bool"]).columns

    for column in bool_columns:
        df[column] = df[column].astype("int8")

    return df


def find_high_missing_columns(train_df: pd.DataFrame) -> list[str]:
    """
    Identifie les colonnes avec trop de valeurs manquantes.

    Le calcul est fait uniquement sur le train pour éviter de prendre des
    décisions à partir du test.
    """
    missing_rate = train_df.isna().mean()

    high_missing_columns = missing_rate[
        missing_rate >= HIGH_MISSING_THRESHOLD
    ].index.tolist()

    # On ne supprime jamais TARGET ni l'identifiant client.
    protected_columns = [TARGET_COLUMN, ID_COLUMN]

    high_missing_columns = [
        column
        for column in high_missing_columns
        if column not in protected_columns
    ]

    return high_missing_columns


def find_constant_columns(train_df: pd.DataFrame) -> list[str]:
    """
    Identifie les colonnes constantes dans le train.

    Une colonne constante n'apporte aucune information au modèle.
    """
    constant_columns = []

    for column in train_df.columns:
        if column in [TARGET_COLUMN, ID_COLUMN]:
            continue

        unique_values = train_df[column].nunique(dropna=False)

        if unique_values <= 1:
            constant_columns.append(column)

    return constant_columns


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie légèrement les noms de colonnes.

    Certains modèles comme LightGBM peuvent être sensibles à certains
    caractères spéciaux dans les noms de features.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
        .str.replace(">", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(",", "_", regex=False)
    )

    return df


def prepare_modeling_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prépare les datasets finaux pour la modélisation.

    Attention :
    - pas de scaling ici ;
    - pas d'imputation globale ici ;
    - les décisions de suppression sont calculées sur le train uniquement ;
    - les mêmes colonnes sont supprimées dans le train et le test.
    """
    train_df = load_processed_dataset(TRAIN_INPUT_FILE)
    test_df = load_processed_dataset(TEST_INPUT_FILE)

    print(f"Train initial : {train_df.shape}")
    print(f"Test initial  : {test_df.shape}")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError("TARGET est absent du train.")

    if ID_COLUMN not in train_df.columns:
        raise ValueError("SK_ID_CURR est absent du train.")

    if ID_COLUMN not in test_df.columns:
        raise ValueError("SK_ID_CURR est absent du test.")

    # Remplacement des valeurs infinies.
    train_df = replace_infinite_values(train_df)
    test_df = replace_infinite_values(test_df)

    # Conversion des booléens.
    train_df = convert_boolean_columns(train_df)
    test_df = convert_boolean_columns(test_df)

    # Colonnes trop incomplètes, calculées uniquement sur train.
    high_missing_columns = find_high_missing_columns(train_df)

    # Colonnes constantes, calculées uniquement sur train.
    constant_columns = find_constant_columns(train_df)

    # Union des colonnes à supprimer.
    columns_to_drop = sorted(set(high_missing_columns + constant_columns))

    print(f"Colonnes avec trop de valeurs manquantes : {len(high_missing_columns)}")
    print(f"Colonnes constantes : {len(constant_columns)}")
    print(f"Total colonnes à supprimer : {len(columns_to_drop)}")

    # Suppression dans train et test.
    train_df = train_df.drop(columns=columns_to_drop, errors="ignore")
    test_df = test_df.drop(columns=columns_to_drop, errors="ignore")

    # Nettoyage des noms de colonnes.
    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    # Vérification : le train doit garder TARGET.
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError("TARGET a été supprimé par erreur.")

    # Vérification : le test ne doit pas avoir TARGET.
    if TARGET_COLUMN in test_df.columns:
        test_df = test_df.drop(columns=[TARGET_COLUMN])

    print(f"Train modeling final : {train_df.shape}")
    print(f"Test modeling final  : {test_df.shape}")

    # Sauvegarde du rapport des colonnes supprimées.
    dropped_columns_report = pd.DataFrame(
        {
            "column": columns_to_drop,
            "reason": [
                "high_missing_or_constant"
                for _ in columns_to_drop
            ],
        }
    )

    dropped_columns_path = DATA_QUALITY_DIR / "modeling_dropped_columns.csv"
    dropped_columns_report.to_csv(dropped_columns_path, index=False)

    print(f"Rapport colonnes supprimées : {dropped_columns_path}")

    return train_df, test_df


def main() -> None:
    """
    Point d'entrée du script.

    Génère :
    - data/processed/train_modeling.csv
    - data/processed/test_modeling.csv
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = prepare_modeling_dataset()

    train_output_path = PROCESSED_DATA_DIR / TRAIN_OUTPUT_FILE
    test_output_path = PROCESSED_DATA_DIR / TEST_OUTPUT_FILE

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"\nTrain modeling sauvegardé : {train_output_path}")
    print(f"Test modeling sauvegardé  : {test_output_path}")
    print("Préparation du dataset de modélisation terminée.")

    del train_df, test_df
    gc.collect()


if __name__ == "__main__":
    main()