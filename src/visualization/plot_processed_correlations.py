import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_QUALITY_DIR, FIGURES_DIR, PROCESSED_DATA_DIR


TRAIN_MODELING_FILE = "train_modeling.csv"

PEARSON_FILE = "processed_target_correlations_pearson.csv"
SPEARMAN_FILE = "processed_target_correlations_spearman.csv"

TARGET_COLUMN = "TARGET"

TOP_N_BARPLOT = 20
TOP_N_MATRIX = 30

CORRELATION_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def load_csv(file_path):
    """
    Charge un fichier CSV avec vérification d'existence.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    return pd.read_csv(file_path)


def prepare_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les colonnes numériques pour les matrices de corrélation.

    On remplace inf et -inf par NaN.
    Les booléens sont convertis en 0/1 si besoin.
    """
    numeric_df = df.select_dtypes(include=["number", "bool"]).copy()

    bool_columns = numeric_df.select_dtypes(include=["bool"]).columns

    for column in bool_columns:
        numeric_df[column] = numeric_df[column].astype("int8")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    return numeric_df


def sample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Échantillonne le dataset pour accélérer les calculs de corrélation.

    50 000 lignes suffisent pour produire une visualisation robuste.
    """
    if len(df) > CORRELATION_SAMPLE_SIZE:
        return df.sample(
            n=CORRELATION_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    return df


def plot_target_correlations_barplot(
    correlation_df: pd.DataFrame,
    correlation_column: str,
    title: str,
    output_file_name: str,
) -> None:
    """
    Génère un graphique horizontal des variables les plus corrélées avec TARGET.
    """
    top_df = correlation_df.head(TOP_N_BARPLOT).copy()

    # On inverse l'ordre pour avoir la plus forte corrélation en haut du graphique.
    top_df = top_df.iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(
        top_df["column"],
        top_df[correlation_column],
    )
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Corrélation avec TARGET")
    plt.ylabel("Variable")
    plt.tight_layout()

    output_path = FIGURES_DIR / output_file_name
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def plot_correlation_matrix(
    train_df: pd.DataFrame,
    selected_columns: list[str],
    method: str,
    title: str,
    output_file_name: str,
) -> None:
    """
    Génère une matrice de corrélation sur un sous-ensemble de variables.

    On ne visualise pas toutes les colonnes car le dataset contient plusieurs
    centaines de variables. On sélectionne les variables les plus corrélées
    avec TARGET pour produire une matrice lisible.
    """
    numeric_df = prepare_numeric_dataframe(train_df)
    numeric_df = sample_dataframe(numeric_df)

    columns_to_use = [
        column
        for column in selected_columns
        if column in numeric_df.columns
    ]

    if TARGET_COLUMN in numeric_df.columns and TARGET_COLUMN not in columns_to_use:
        columns_to_use = [TARGET_COLUMN] + columns_to_use

    matrix_df = numeric_df[columns_to_use].copy()

    print(f"Calcul matrice {method} sur {len(columns_to_use)} colonnes...")
    correlation_matrix = matrix_df.corr(method=method)

    plt.figure(figsize=(16, 14))
    plt.imshow(correlation_matrix, aspect="auto")
    plt.colorbar(label=f"Corrélation {method}")

    plt.xticks(
        ticks=range(len(correlation_matrix.columns)),
        labels=correlation_matrix.columns,
        rotation=90,
        fontsize=7,
    )

    plt.yticks(
        ticks=range(len(correlation_matrix.index)),
        labels=correlation_matrix.index,
        fontsize=7,
    )

    plt.title(title)
    plt.tight_layout()

    output_path = FIGURES_DIR / output_file_name
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def main() -> None:
    """
    Point d'entrée du script.

    Génère les graphiques de corrélation dans reports/figures.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / TRAIN_MODELING_FILE
    pearson_path = DATA_QUALITY_DIR / PEARSON_FILE
    spearman_path = DATA_QUALITY_DIR / SPEARMAN_FILE

    print("Chargement du dataset de modélisation...")
    train_df = load_csv(train_path)

    print("Chargement des corrélations Pearson...")
    pearson_df = load_csv(pearson_path)

    print("Chargement des corrélations Spearman...")
    spearman_df = load_csv(spearman_path)

    print(f"Train modeling shape : {train_df.shape}")

    # Barplot Pearson.
    plot_target_correlations_barplot(
        correlation_df=pearson_df,
        correlation_column="pearson_correlation_with_target",
        title="Top corrélations Pearson avec TARGET",
        output_file_name="target_correlations_pearson_top20.png",
    )

    # Barplot Spearman.
    plot_target_correlations_barplot(
        correlation_df=spearman_df,
        correlation_column="spearman_correlation_with_target",
        title="Top corrélations Spearman avec TARGET",
        output_file_name="target_correlations_spearman_top20.png",
    )

    # Variables les plus corrélées avec TARGET selon Pearson.
    pearson_top_columns = pearson_df["column"].head(TOP_N_MATRIX).tolist()

    # Variables les plus corrélées avec TARGET selon Spearman.
    spearman_top_columns = spearman_df["column"].head(TOP_N_MATRIX).tolist()

    # Matrice Pearson sur les variables les plus importantes.
    plot_correlation_matrix(
        train_df=train_df,
        selected_columns=pearson_top_columns,
        method="pearson",
        title="Matrice de corrélation Pearson - Top variables liées à TARGET",
        output_file_name="correlation_matrix_pearson_top30.png",
    )

    # Matrice Spearman sur les variables les plus importantes.
    plot_correlation_matrix(
        train_df=train_df,
        selected_columns=spearman_top_columns,
        method="spearman",
        title="Matrice de corrélation Spearman - Top variables liées à TARGET",
        output_file_name="correlation_matrix_spearman_top30.png",
    )

    del train_df, pearson_df, spearman_df
    gc.collect()

    print("\nVisualisations de corrélation terminées.")


if __name__ == "__main__":
    main()