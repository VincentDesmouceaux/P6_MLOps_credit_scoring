import pandas as pd

from src.config import DATA_QUALITY_DIR, FIGURES_DIR, RAW_FILES, TARGET_COLUMN
from src.data.load_data import load_raw_table


def summarize_table(file_key: str, nrows: int | None = None) -> dict:
    df = load_raw_table(file_key, nrows=nrows)

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_rate = total_missing / total_cells if total_cells > 0 else 0

    return {
        "table": file_key,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "total_missing_values": total_missing,
        "missing_rate": round(missing_rate, 4),
        "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": len(df.select_dtypes(include=["object", "string"]).columns),
    }


def build_raw_tables_summary(nrows: int | None = None) -> pd.DataFrame:
    summaries = []

    for file_key in RAW_FILES:
        summary = summarize_table(file_key, nrows=nrows)
        summaries.append(summary)

    return pd.DataFrame(summaries)


def save_missing_values_report(file_key: str, nrows: int | None = None) -> None:
    df = load_raw_table(file_key, nrows=nrows)

    missing_report = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_rate"})
    )

    output_path = DATA_QUALITY_DIR / f"{file_key}_missing_values.csv"
    missing_report.to_csv(output_path, index=False)


def save_target_distribution() -> None:
    df = load_raw_table("application_train")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La colonne {TARGET_COLUMN} est absente.")

    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    )

    target_distribution.columns = ["target", "percentage"]

    output_path = DATA_QUALITY_DIR / "target_distribution.csv"
    target_distribution.to_csv(output_path, index=False)

    print("Distribution de la target :")
    print(target_distribution)


def main() -> None:
    DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Création du rapport global des tables brutes...")

    summary = build_raw_tables_summary()
    summary_path = DATA_QUALITY_DIR / "raw_tables_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary)
    print(f"\nRapport sauvegardé : {summary_path}")

    print("\nCréation des rapports de valeurs manquantes...")
    for file_key in RAW_FILES:
        save_missing_values_report(file_key)

    print("\nCréation du rapport de distribution de la target...")
    save_target_distribution()

    print("\nExploration terminée.")


if __name__ == "__main__":
    main()