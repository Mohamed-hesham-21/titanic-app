from pathlib import Path
import pandas as pd
import streamlit as st

# Dynamically resolve the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_PATHS = [
BASE_DIR / "data" / "titanic.csv",
BASE_DIR / "src" / "titanic_app" / "titanic.csv",
Path("data/titanic.csv"),
Path("titanic_app/src/titanic_app/titanic.csv"),
]


def _resolve_csv_path() -> Path:
    """Finds the first valid path to titanic.csv."""
    for path in CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find 'titanic.csv'. Checked paths: {[str(p) for p in CANDIDATE_PATHS]}"
    )


@st.cache_data(show_spinner="Loading Titanic dataset...")
def load_data() -> pd.DataFrame:
    """Loads, cleans, and engineers features for the Titanic dataset."""
    csv_path = _resolve_csv_path()
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Human-readable mappings
    df["class"] = df["pclass"].map({1: "First", 2: "Second", 3: "Third"})
    df["survival"] = df["survived"].map({0: "Died", 1: "Survived"})
    df["port"] = df["embarked"].map(
        {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    )

    # Feature engineering
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["child", "teen", "young adult", "adult", "senior"],
    )

    return df