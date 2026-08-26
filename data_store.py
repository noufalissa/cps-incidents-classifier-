from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
from ids import incident_id

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "incidents.csv"

COLUMNS = [
    "Year", "Incident Name", "Country/Region", "Sector", "Attack Type",
    "Attacker / Group", "Verified Impact Summary", "Source / Verification URL",
    "Verification Status", "URL1", "URL2", "URL3", "URL4",
]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMNS].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    for col in COLUMNS[1:]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["Incident Name"].str.strip() != ""].reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_repository_dataset() -> pd.DataFrame:
    return normalize_dataframe(pd.read_csv(DATA_PATH))


def initialize_dataset(force: bool = False) -> None:
    if force or "incidents_df" not in st.session_state:
        st.session_state["incidents_df"] = load_repository_dataset().copy()
        st.session_state["dataset_dirty"] = False


def get_dataset() -> pd.DataFrame:
    initialize_dataset()
    return st.session_state["incidents_df"]


def set_dataset(df: pd.DataFrame) -> None:
    st.session_state["incidents_df"] = normalize_dataframe(df)
    st.session_state["dataset_dirty"] = True


def reset_dataset() -> None:
    load_repository_dataset.clear()
    initialize_dataset(force=True)


def with_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_dataframe(df)
    out.insert(0, "Incident ID", [
        incident_id(r["Year"], r["Incident Name"], r["Country/Region"], r["Sector"])
        for _, r in out.iterrows()
    ])
    return out


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = normalize_dataframe(df)
    out["Year"] = out["Year"].astype("string").fillna("")
    return out.to_csv(index=False).encode("utf-8-sig")
