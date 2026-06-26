import pandas as pd

from src.config import RAW_DATA_DIR, RAW_FILES


ENCODINGS_TO_TRY = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]


def get_raw_file_path(file_key: str):
    if file_key not in RAW_FILES:
        available_files = ", ".join(RAW_FILES.keys())
        raise ValueError(
            f"Fichier inconnu : {file_key}. "
            f"Fichiers disponibles : {available_files}"
        )

    return RAW_DATA_DIR / RAW_FILES[file_key]


def load_raw_table(file_key: str, nrows: int | None = None) -> pd.DataFrame:
    file_path = get_raw_file_path(file_key)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    last_error = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(
                file_path,
                nrows=nrows,
                encoding=encoding,
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Impossible de lire {file_path} avec les encodages testés. "
        f"Dernière erreur : {last_error}",
    )


def load_all_raw_tables(nrows: int | None = None) -> dict[str, pd.DataFrame]:
    tables = {}

    for file_key in RAW_FILES:
        tables[file_key] = load_raw_table(file_key, nrows=nrows)

    return tables