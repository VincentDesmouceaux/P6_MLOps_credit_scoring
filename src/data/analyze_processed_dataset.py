import gc

import numpy as np
import pandas as pd

from src.config import DATA_QUALITY_DIR, PROCESSED_DATA_DIR


TRAIN_FILE_NAME = "train_processed.csv"
TEST_FILE_NAME = "test_processed.csv"

TARGET_COLUMN = "TARGET"

# Seuil au-delà duquel une colonne est considérée comme trop incomplète.
HIGH_MISSING_THRESHOLD = 0.80

# Seuil pour détecter deux variables très corrélées entre elles.
HIGH_CORRELATION_THRESHOLD = 0.90

# Pour éviter un calcul trop long sur Spearman et la matrice complète,
# on peut échantillonner les lignes.
CORRELATION_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def load_processed_dataset(file_name: str) -> pd.DataFrame:
    """
    Charge un fichier depuis data/processed.

    Exemple :
    - train_processed.csv
    - test_processed.csv
    """
    file_path = PROCESSED_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"Chargement : {file_path.name}")
    return pd.read_csv(file_path)


def build_dataset_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construit un résumé global des datasets train et test.

    On vérifie :
    - nombre de lignes ;
    - nombre de colonnes ;
    - nombre total de valeurs manquantes ;
    - taux global de valeurs manquantes ;
    - présence ou non de TARGET.
    """
    summary = pd.DataFrame(
        [
            {
                "dataset": "train",
                "rows": train_df.shape[0],
                "columns": train_df.shape[1],
                "total_missing_values": int(train_df.isna().sum().sum()),
                "missing_rate": round(
                    float(train_df.isna().sum().sum() / train_df.size),
                    4,
                ),
                "has_target": TARGET_COLUMN in train_df.columns,
            },
            {
                "dataset": "test",
                "rows": test_df.shape[0],
                "columns": test_df.shape[1],
                "total_missing_values": int(test_df.isna().sum().sum()),
                "missing_rate": round(
                    float(test_df.isna().sum().sum() / test_df.size),
                    4,
                ),
                "has_target": TARGET_COLUMN in test_df.columns,
            },
        ]
    )

    return summary


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les valeurs manquantes colonne par colonne.

    Le résultat permet d'identifier :
    - les colonnes très incomplètes ;
    - les colonnes à imputer ;
    - les colonnes potentiellement à supprimer.
    """
    missing_values = df.isna().sum()
    missing_rate = missing_values / len(df)

    missing_report = pd.DataFrame(
        {
            "column": missing_values.index,
            "missing_values": missing_values.values,
            "missing_rate": missing_rate.values,
        }
    )

    missing_report = missing_report.sort_values(
        by="missing_rate",
        ascending=False,
    )

    return missing_report


def analyze_target_distribution(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la distribution de la variable cible TARGET.

    Dans ce projet :
    - TARGET = 0 : client sans défaut ;
    - TARGET = 1 : client en défaut.

    Cette étape permet de confirmer le déséquilibre de classes.
    """
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError("La colonne TARGET est absente du train.")

    target_counts = train_df[TARGET_COLUMN].value_counts(dropna=False)
    target_rates = train_df[TARGET_COLUMN].value_counts(
        dropna=False,
        normalize=True,
    )

    target_distribution = pd.DataFrame(
        {
            "target_value": target_counts.index,
            "count": target_counts.values,
            "rate": target_rates.values,
        }
    )

    return target_distribution


def find_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifie les colonnes constantes.

    Une colonne constante n'apporte aucune information au modèle :
    elle a toujours la même valeur.
    """
    constant_columns = []

    for column in df.columns:
        unique_values = df[column].nunique(dropna=False)

        if unique_values <= 1:
            constant_columns.append(
                {
                    "column": column,
                    "unique_values": unique_values,
                }
            )

    return pd.DataFrame(constant_columns)


def prepare_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les colonnes numériques pour les analyses de corrélation.

    On inclut :
    - les colonnes numériques ;
    - les colonnes booléennes issues du one-hot encoding.

    Les valeurs infinies sont remplacées par NaN.
    """
    numeric_df = df.select_dtypes(include=["number", "bool"]).copy()

    # Les booléens True/False sont convertis en 1/0 pour les corrélations.
    bool_columns = numeric_df.select_dtypes(include=["bool"]).columns

    for column in bool_columns:
        numeric_df[column] = numeric_df[column].astype("int8")

    # Certaines divisions peuvent produire inf ou -inf.
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    return numeric_df


def sample_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Échantillonne le dataset pour accélérer les corrélations lourdes.

    Spearman peut être coûteux sur plus de 300 000 lignes.
    Un échantillon de 50 000 lignes suffit pour repérer les tendances fortes.
    """
    if len(df) > CORRELATION_SAMPLE_SIZE:
        return df.sample(
            n=CORRELATION_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    return df


def compute_target_correlations(
    train_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Calcule la corrélation des variables numériques avec TARGET.

    method peut valoir :
    - pearson : relation linéaire ;
    - spearman : relation monotone, plus robuste aux distributions asymétriques.
    """
    numeric_df = prepare_numeric_dataframe(train_df)

    if TARGET_COLUMN not in numeric_df.columns:
        raise ValueError("TARGET doit être numérique pour calculer la corrélation.")

    # On échantillonne pour éviter un calcul trop long.
    numeric_sample = sample_for_correlation(numeric_df)

    correlations = numeric_sample.corr(method=method)[TARGET_COLUMN]

    correlations_report = (
        correlations.drop(labels=[TARGET_COLUMN], errors="ignore")
        .dropna()
        .reset_index()
    )

    correlations_report.columns = ["column", f"{method}_correlation_with_target"]

    # Valeur absolue pour identifier les liens les plus forts,
    # qu'ils soient positifs ou négatifs.
    correlations_report[f"abs_{method}_correlation_with_target"] = (
        correlations_report[f"{method}_correlation_with_target"].abs()
    )

    correlations_report = correlations_report.sort_values(
        by=f"abs_{method}_correlation_with_target",
        ascending=False,
    )

    return correlations_report


def find_highly_correlated_pairs(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les paires de variables très corrélées entre elles.

    Une corrélation très élevée peut indiquer des variables redondantes.
    Cela peut être utile pour :
    - simplifier le modèle ;
    - limiter la multicolinéarité ;
    - réduire le nombre de features.
    """
    numeric_df = prepare_numeric_dataframe(train_df)

    # On retire TARGET : ici on analyse les corrélations entre features.
    numeric_df = numeric_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    # On échantillonne pour limiter le temps de calcul.
    numeric_sample = sample_for_correlation(numeric_df)

    print("Calcul de la matrice de corrélation Pearson entre features...")
    correlation_matrix = numeric_sample.corr(method="pearson").abs()

    # On garde uniquement le triangle supérieur de la matrice
    # pour éviter les doublons A-B et B-A.
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    highly_correlated_pairs = []

    for column in upper_triangle.columns:
        correlated_columns = upper_triangle.index[
            upper_triangle[column] > HIGH_CORRELATION_THRESHOLD
        ].tolist()

        for correlated_column in correlated_columns:
            highly_correlated_pairs.append(
                {
                    "feature_1": correlated_column,
                    "feature_2": column,
                    "pearson_correlation_abs": upper_triangle.loc[
                        correlated_column,
                        column,
                    ],
                }
            )

    pairs_report = pd.DataFrame(highly_correlated_pairs)

    if not pairs_report.empty:
        pairs_report = pairs_report.sort_values(
            by="pearson_correlation_abs",
            ascending=False,
        )

    return pairs_report


def main() -> None:
    """
    Point d'entrée du script.

    Génère plusieurs fichiers de contrôle qualité dans :
    reports/data_quality/
    """
    DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_processed_dataset(TRAIN_FILE_NAME)
    test_df = load_processed_dataset(TEST_FILE_NAME)

    print(f"Train shape : {train_df.shape}")
    print(f"Test shape  : {test_df.shape}")

    # Résumé global train/test.
    summary = build_dataset_summary(train_df, test_df)
    summary_path = DATA_QUALITY_DIR / "processed_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nRésumé global :")
    print(summary)

    # Valeurs manquantes.
    missing_report = analyze_missing_values(train_df)
    missing_path = DATA_QUALITY_DIR / "processed_missing_values.csv"
    missing_report.to_csv(missing_path, index=False)

    high_missing_columns = missing_report[
        missing_report["missing_rate"] >= HIGH_MISSING_THRESHOLD
    ]
    high_missing_path = DATA_QUALITY_DIR / "processed_high_missing_columns.csv"
    high_missing_columns.to_csv(high_missing_path, index=False)

    print("\nTop 20 colonnes avec le plus de valeurs manquantes :")
    print(missing_report.head(20))

    # Distribution de TARGET.
    target_distribution = analyze_target_distribution(train_df)
    target_path = DATA_QUALITY_DIR / "processed_target_distribution.csv"
    target_distribution.to_csv(target_path, index=False)

    print("\nDistribution de TARGET :")
    print(target_distribution)

    # Colonnes constantes.
    constant_columns = find_constant_columns(train_df)
    constant_path = DATA_QUALITY_DIR / "processed_constant_columns.csv"
    constant_columns.to_csv(constant_path, index=False)

    print(f"\nNombre de colonnes constantes : {len(constant_columns)}")

    # Corrélation Pearson avec TARGET.
    print("\nCalcul des corrélations Pearson avec TARGET...")
    pearson_correlations = compute_target_correlations(
        train_df=train_df,
        method="pearson",
    )
    pearson_path = DATA_QUALITY_DIR / "processed_target_correlations_pearson.csv"
    pearson_correlations.to_csv(pearson_path, index=False)

    print("\nTop 20 corrélations Pearson avec TARGET :")
    print(pearson_correlations.head(20))

    # Corrélation Spearman avec TARGET.
    print("\nCalcul des corrélations Spearman avec TARGET...")
    spearman_correlations = compute_target_correlations(
        train_df=train_df,
        method="spearman",
    )
    spearman_path = DATA_QUALITY_DIR / "processed_target_correlations_spearman.csv"
    spearman_correlations.to_csv(spearman_path, index=False)

    print("\nTop 20 corrélations Spearman avec TARGET :")
    print(spearman_correlations.head(20))

    # Paires de variables très corrélées entre elles.
    highly_correlated_pairs = find_highly_correlated_pairs(train_df)
    highly_correlated_pairs_path = (
        DATA_QUALITY_DIR / "processed_highly_correlated_pairs.csv"
    )
    highly_correlated_pairs.to_csv(highly_correlated_pairs_path, index=False)

    print(
        "\nNombre de paires de variables avec corrélation absolue "
        f">= {HIGH_CORRELATION_THRESHOLD} : {len(highly_correlated_pairs)}"
    )

    print("\nRapports sauvegardés dans :")
    print(DATA_QUALITY_DIR)

    del train_df, test_df
    gc.collect()

    print("\nAnalyse qualité du dataset final terminée.")


if __name__ == "__main__":
    main()