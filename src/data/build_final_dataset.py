import gc

import pandas as pd

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


INTERIM_FEATURE_FILES = [
    "bureau_features.csv",
    "previous_application_features.csv",
    "pos_cash_features.csv",
    "installments_features.csv",
    "credit_card_features.csv",
]


def load_interim_table(file_name: str) -> pd.DataFrame:
    """
    Charge un fichier intermédiaire depuis data/interim.

    Exemple :
    - application_base.csv
    - bureau_features.csv
    - previous_application_features.csv
    """
    file_path = INTERIM_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"Chargement : {file_path.name}")
    return pd.read_csv(file_path)


def merge_feature_table(
    base_df: pd.DataFrame,
    feature_file_name: str,
) -> pd.DataFrame:
    """
    Fusionne une table de features avec la base principale.

    La fusion se fait en LEFT JOIN sur SK_ID_CURR :
    - on garde toutes les lignes de application_base ;
    - on ajoute les colonnes de la table de features ;
    - si un client n'a pas d'historique dans une table secondaire,
      les nouvelles colonnes seront à NaN.
    """
    features_df = load_interim_table(feature_file_name)

    print(
        f"Fusion avec {feature_file_name} | "
        f"base avant : {base_df.shape} | "
        f"features : {features_df.shape}"
    )

    # Vérification de sécurité : la clé doit être présente.
    if "SK_ID_CURR" not in features_df.columns:
        raise ValueError(f"SK_ID_CURR absent dans {feature_file_name}")

    # LEFT JOIN pour ne jamais perdre de clients de la table principale.
    merged_df = base_df.merge(
        features_df,
        how="left",
        on="SK_ID_CURR",
    )

    print(f"base après fusion : {merged_df.shape}")

    del features_df
    gc.collect()

    return merged_df


def split_train_test(final_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare le dataset final en train et test.

    Deux cas sont possibles :
    1. La colonne dataset_type existe encore sous forme texte.
    2. La colonne dataset_type a été encodée en dataset_type_train
       et dataset_type_test par le one-hot encoding.

    Cette fonction gère les deux cas pour éviter une erreur lors de la fusion.
    """
    final_df = final_df.copy()

    # Cas 1 : la colonne dataset_type existe encore.
    if "dataset_type" in final_df.columns:
        train_df = final_df[final_df["dataset_type"] == "train"].copy()
        test_df = final_df[final_df["dataset_type"] == "test"].copy()

        train_df = train_df.drop(columns=["dataset_type"])
        test_df = test_df.drop(columns=["dataset_type"])

    # Cas 2 : dataset_type a été transformée par le one-hot encoding.
    elif {"dataset_type_train", "dataset_type_test"}.issubset(final_df.columns):
        train_df = final_df[final_df["dataset_type_train"] == 1].copy()
        test_df = final_df[final_df["dataset_type_test"] == 1].copy()

        train_df = train_df.drop(
            columns=["dataset_type_train", "dataset_type_test"]
        )
        test_df = test_df.drop(
            columns=["dataset_type_train", "dataset_type_test"]
        )

    else:
        raise ValueError(
            "Impossible de séparer train/test : aucune colonne dataset_type, "
            "dataset_type_train ou dataset_type_test trouvée."
        )

    # Le fichier test Kaggle/OpenClassrooms n'a pas de vraie cible TARGET.
    # On retire donc TARGET du test final.
    if "TARGET" in test_df.columns:
        test_df = test_df.drop(columns=["TARGET"])

    return train_df, test_df

def build_final_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit le dataset final enrichi.

    Étapes :
    1. Charger application_base.csv.
    2. Fusionner toutes les tables de features agrégées.
    3. Séparer train et test.
    4. Retourner les deux datasets prêts pour l'étape modélisation.
    """
    print("Chargement de la base application...")
    final_df = load_interim_table("application_base.csv")

    print(f"application_base shape initiale : {final_df.shape}")

    # Fusion progressive des tables secondaires.
    for feature_file_name in INTERIM_FEATURE_FILES:
        final_df = merge_feature_table(
            base_df=final_df,
            feature_file_name=feature_file_name,
        )

    print(f"\nDataset final fusionné shape : {final_df.shape}")

    # Séparation train / test.
    train_df, test_df = split_train_test(final_df)

    print(f"Train final shape : {train_df.shape}")
    print(f"Test final shape  : {test_df.shape}")

    del final_df
    gc.collect()

    return train_df, test_df


def main() -> None:
    """
    Point d'entrée du script.

    Génère :
    - data/processed/train_processed.csv
    - data/processed/test_processed.csv
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = build_final_dataset()

    train_output_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_output_path = PROCESSED_DATA_DIR / "test_processed.csv"

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"\nTrain sauvegardé : {train_output_path}")
    print(f"Test sauvegardé  : {test_output_path}")
    print("Fusion finale terminée.")


if __name__ == "__main__":
    main()