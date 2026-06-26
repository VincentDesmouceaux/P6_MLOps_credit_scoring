import gc

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns

from src.config import DATA_QUALITY_DIR, FIGURES_DIR, PROCESSED_DATA_DIR


TRAIN_MODELING_FILE = "train_modeling.csv"
MISSING_VALUES_FILE = "processed_missing_values.csv"
PEARSON_FILE = "processed_target_correlations_pearson.csv"
SPEARMAN_FILE = "processed_target_correlations_spearman.csv"
TARGET_DISTRIBUTION_FILE = "processed_target_distribution.csv"

TOP_N_MISSING = 30
TOP_N_CORRELATIONS = 20
MISSINGNO_SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def load_csv(file_path):
    """
    Charge un fichier CSV avec vérification d'existence.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    return pd.read_csv(file_path)


def plot_target_distribution() -> None:
    """
    Génère un graphique de distribution de la variable cible TARGET.

    Objectif :
    montrer visuellement le déséquilibre de classes.
    """
    target_path = DATA_QUALITY_DIR / TARGET_DISTRIBUTION_FILE
    target_distribution = load_csv(target_path)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=target_distribution,
        x="target_value",
        y="rate",
    )

    plt.title("Distribution de la variable cible TARGET")
    plt.xlabel("Classe TARGET")
    plt.ylabel("Proportion")
    plt.tight_layout()

    output_path = FIGURES_DIR / "step1_target_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def plot_missing_values_top30() -> None:
    """
    Génère un graphique des 30 colonnes avec le plus de valeurs manquantes.
    """
    missing_path = DATA_QUALITY_DIR / MISSING_VALUES_FILE
    missing_df = load_csv(missing_path)

    top_missing = missing_df.head(TOP_N_MISSING).copy()
    top_missing = top_missing.iloc[::-1]

    plt.figure(figsize=(12, 9))
    sns.barplot(
        data=top_missing,
        x="missing_rate",
        y="column",
    )

    plt.title("Top 30 des colonnes avec le plus de valeurs manquantes")
    plt.xlabel("Taux de valeurs manquantes")
    plt.ylabel("Variable")
    plt.tight_layout()

    output_path = FIGURES_DIR / "step1_missing_values_top30.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def plot_missingno_matrix() -> None:
    """
    Génère une matrice missingno sur un échantillon du train.

    On échantillonne pour éviter une image trop lourde.
    """
    train_path = PROCESSED_DATA_DIR / TRAIN_MODELING_FILE
    train_df = load_csv(train_path)

    if len(train_df) > MISSINGNO_SAMPLE_SIZE:
        train_sample = train_df.sample(
            n=MISSINGNO_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )
    else:
        train_sample = train_df.copy()

    plt.figure(figsize=(16, 8))
    msno.matrix(train_sample)
    plt.title("Visualisation des valeurs manquantes - échantillon train")
    plt.tight_layout()

    output_path = FIGURES_DIR / "step1_missingno_matrix_train_sample.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")

    del train_df, train_sample
    gc.collect()


def plot_target_correlations(
    file_name: str,
    correlation_column: str,
    title: str,
    output_file_name: str,
) -> None:
    """
    Génère un graphique des variables les plus corrélées avec TARGET.
    """
    correlation_path = DATA_QUALITY_DIR / file_name
    correlation_df = load_csv(correlation_path)

    top_corr = correlation_df.head(TOP_N_CORRELATIONS).copy()
    top_corr = top_corr.iloc[::-1]

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_corr,
        x=correlation_column,
        y="column",
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Corrélation avec TARGET")
    plt.ylabel("Variable")
    plt.tight_layout()

    output_path = FIGURES_DIR / output_file_name
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def main() -> None:
    """
    Génère les visualisations finales de l'étape 1.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_target_distribution()
    plot_missing_values_top30()
    plot_missingno_matrix()

    plot_target_correlations(
        file_name=PEARSON_FILE,
        correlation_column="pearson_correlation_with_target",
        title="Top corrélations Pearson avec TARGET",
        output_file_name="step1_target_correlations_pearson_top20.png",
    )

    plot_target_correlations(
        file_name=SPEARMAN_FILE,
        correlation_column="spearman_correlation_with_target",
        title="Top corrélations Spearman avec TARGET",
        output_file_name="step1_target_correlations_spearman_top20.png",
    )

    print("\nVisualisations de l'étape 1 terminées.")


if __name__ == "__main__":
    main()