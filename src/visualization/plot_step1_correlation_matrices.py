import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_QUALITY_DIR, FIGURES_DIR, PROCESSED_DATA_DIR


TRAIN_MODELING_FILE = "train_modeling.csv"

PEARSON_FILE = "processed_target_correlations_pearson.csv"
SPEARMAN_FILE = "processed_target_correlations_spearman.csv"

TARGET_COLUMN = "TARGET"

TOP_N_FEATURES = 25
CORRELATION_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def load_csv(file_path):
    """
    Charge un fichier CSV avec vérification.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    return pd.read_csv(file_path)


def prepare_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garde uniquement les colonnes numériques et remplace les valeurs infinies.
    """
    numeric_df = df.select_dtypes(include=["number", "bool"]).copy()

    bool_columns = numeric_df.select_dtypes(include=["bool"]).columns

    for column in bool_columns:
        numeric_df[column] = numeric_df[column].astype("int8")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    return numeric_df


def sample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Échantillonne le dataset pour éviter un calcul trop lourd.
    """
    if len(df) > CORRELATION_SAMPLE_SIZE:
        return df.sample(
            n=CORRELATION_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    return df


def plot_correlation_heatmap(
    train_df: pd.DataFrame,
    selected_columns: list[str],
    method: str,
    title: str,
    output_file_name: str,
) -> None:
    """
    Génère une heatmap de corrélation.

    On sélectionne uniquement les variables les plus corrélées avec TARGET.
    Sinon, avec plus de 600 variables, la matrice serait illisible.
    """
    numeric_df = prepare_numeric_dataframe(train_df)
    numeric_df = sample_dataframe(numeric_df)

    columns_to_use = [
        column
        for column in selected_columns
        if column in numeric_df.columns
    ]

    if TARGET_COLUMN in numeric_df.columns:
        columns_to_use = [TARGET_COLUMN] + columns_to_use

    matrix_df = numeric_df[columns_to_use].copy()

    print(f"Calcul de la matrice {method} sur {len(columns_to_use)} colonnes...")

    correlation_matrix = matrix_df.corr(method=method)

    plt.figure(figsize=(18, 14))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        annot=False,
        square=False,
        linewidths=0.2,
        cbar_kws={"label": f"Corrélation {method}"},
    )

    plt.title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    output_path = FIGURES_DIR / output_file_name
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def main() -> None:
    """
    Génère les matrices de corrélation Pearson et Spearman.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / TRAIN_MODELING_FILE
    pearson_path = DATA_QUALITY_DIR / PEARSON_FILE
    spearman_path = DATA_QUALITY_DIR / SPEARMAN_FILE

    print("Chargement du train modeling...")
    train_df = load_csv(train_path)

    print("Chargement des corrélations Pearson...")
    pearson_df = load_csv(pearson_path)

    print("Chargement des corrélations Spearman...")
    spearman_df = load_csv(spearman_path)

    pearson_top_columns = pearson_df["column"].head(TOP_N_FEATURES).tolist()
    spearman_top_columns = spearman_df["column"].head(TOP_N_FEATURES).tolist()

    plot_correlation_heatmap(
        train_df=train_df,
        selected_columns=pearson_top_columns,
        method="pearson",
        title="Matrice de corrélation Pearson - Top variables liées à TARGET",
        output_file_name="step1_correlation_matrix_pearson_top25.png",
    )

    plot_correlation_heatmap(
        train_df=train_df,
        selected_columns=spearman_top_columns,
        method="spearman",
        title="Matrice de corrélation Spearman - Top variables liées à TARGET",
        output_file_name="step1_correlation_matrix_spearman_top25.png",
    )

    del train_df, pearson_df, spearman_df
    gc.collect()

    print("\nMatrices de corrélation générées.")


if __name__ == "__main__":
    main()