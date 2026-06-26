import subprocess


def get_git_commit_hash() -> str:
    """
    Récupère le hash du commit Git courant.

    Cela permet de relier un run MLflow à une version précise du code.
    Si le projet n'est pas encore versionné avec Git, on retourne une valeur explicite.
    """
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )

        return git_commit.decode("utf-8").strip()

    except Exception:
        return "no_git_commit_found"


def get_git_branch_name() -> str:
    """
    Récupère le nom de la branche Git courante.

    Cela aide à comprendre depuis quelle branche le run MLflow a été lancé.
    """
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        )

        return git_branch.decode("utf-8").strip()

    except Exception:
        return "no_git_branch_found"