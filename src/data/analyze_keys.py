import pandas as pd

from src.config import DATA_QUALITY_DIR
from src.data.load_data import load_raw_table


TABLE_KEYS = {
    "application_train": ["SK_ID_CURR"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU"],
    "previous_application": ["SK_ID_CURR", "SK_ID_PREV"],
    "pos_cash_balance": ["SK_ID_CURR", "SK_ID_PREV"],
    "installments_payments": ["SK_ID_CURR", "SK_ID_PREV"],
    "credit_card_balance": ["SK_ID_CURR", "SK_ID_PREV"],
}


RELATIONS = [
    {
        "parent_table": "application_train",
        "child_table": "bureau",
        "key": "SK_ID_CURR",
    },
    {
        "parent_table": "bureau",
        "child_table": "bureau_balance",
        "key": "SK_ID_BUREAU",
    },
    {
        "parent_table": "application_train",
        "child_table": "previous_application",
        "key": "SK_ID_CURR",
    },
    {
        "parent_table": "application_train",
        "child_table": "pos_cash_balance",
        "key": "SK_ID_CURR",
    },
    {
        "parent_table": "application_train",
        "child_table": "installments_payments",
        "key": "SK_ID_CURR",
    },
    {
        "parent_table": "application_train",
        "child_table": "credit_card_balance",
        "key": "SK_ID_CURR",
    },
]


def analyze_table_keys() -> pd.DataFrame:
    results = []

    for table_name, keys in TABLE_KEYS.items():
        df = load_raw_table(table_name)

        for key in keys:
            if key not in df.columns:
                results.append(
                    {
                        "table": table_name,
                        "key": key,
                        "key_exists": False,
                        "rows": df.shape[0],
                        "unique_keys": None,
                        "missing_keys": None,
                        "duplicated_key_rows": None,
                        "max_rows_per_key": None,
                        "mean_rows_per_key": None,
                    }
                )
                continue

            key_counts = df[key].value_counts(dropna=False)

            results.append(
                {
                    "table": table_name,
                    "key": key,
                    "key_exists": True,
                    "rows": df.shape[0],
                    "unique_keys": df[key].nunique(dropna=True),
                    "missing_keys": int(df[key].isna().sum()),
                    "duplicated_key_rows": int(df.duplicated(subset=[key]).sum()),
                    "max_rows_per_key": int(key_counts.max()),
                    "mean_rows_per_key": round(float(key_counts.mean()), 2),
                }
            )

    return pd.DataFrame(results)


def analyze_relations() -> pd.DataFrame:
    results = []

    for relation in RELATIONS:
        parent_table = relation["parent_table"]
        child_table = relation["child_table"]
        key = relation["key"]

        parent_df = load_raw_table(parent_table)
        child_df = load_raw_table(child_table)

        parent_keys = set(parent_df[key].dropna().unique())
        child_keys = set(child_df[key].dropna().unique())

        common_keys = parent_keys.intersection(child_keys)
        child_only_keys = child_keys.difference(parent_keys)
        parent_only_keys = parent_keys.difference(child_keys)

        child_counts = child_df.groupby(key).size()

        results.append(
            {
                "parent_table": parent_table,
                "child_table": child_table,
                "key": key,
                "parent_unique_keys": len(parent_keys),
                "child_unique_keys": len(child_keys),
                "common_keys": len(common_keys),
                "parent_only_keys": len(parent_only_keys),
                "child_only_keys": len(child_only_keys),
                "mean_child_rows_per_key": round(float(child_counts.mean()), 2),
                "max_child_rows_per_key": int(child_counts.max()),
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    print("Analyse des clés par table...")
    keys_summary = analyze_table_keys()
    keys_output_path = DATA_QUALITY_DIR / "keys_summary.csv"
    keys_summary.to_csv(keys_output_path, index=False)

    print(keys_summary)
    print(f"\nRapport sauvegardé : {keys_output_path}")

    print("\nAnalyse des relations entre tables...")
    relations_summary = analyze_relations()
    relations_output_path = DATA_QUALITY_DIR / "relations_summary.csv"
    relations_summary.to_csv(relations_output_path, index=False)

    print(relations_summary)
    print(f"\nRapport sauvegardé : {relations_output_path}")

    print("\nAnalyse des clés terminée.")


if __name__ == "__main__":
    main()