from __future__ import annotations

import pandas as pd
import streamlit as st


EVIDENCE_COLUMNS = [
    "Incident ID", "Incident Name", "Source Key", "Original URL", "Final URL",
    "Fetch Status", "Source Type", "Characters", "Extracted Text", "Error"
]

CLASS_COLUMNS = [
    "Incident ID", "Incident Name", "Year", "Sector", "Property", "Parent Category",
    "Status", "Status Score", "Semantic Similarity", "Incident Relevance", "Contradiction Score",
    "Neutral Score", "Evidence", "Evidence Source", "Evidence URL",
    "Decision Reason", "Model"
]


def _empty(cols):
    return pd.DataFrame(columns=cols)


def init_state():
    if "evidence_df" not in st.session_state:
        st.session_state["evidence_df"] = _empty(EVIDENCE_COLUMNS)
    if "classification_df" not in st.session_state:
        st.session_state["classification_df"] = _empty(CLASS_COLUMNS)


def get_evidence():
    init_state()
    return st.session_state["evidence_df"]


def set_evidence(df):
    out = df.copy()
    for c in EVIDENCE_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    st.session_state["evidence_df"] = out[EVIDENCE_COLUMNS].fillna("")


def append_evidence(rows):
    current = get_evidence()
    new = pd.DataFrame(rows)
    if new.empty:
        return

    for c in EVIDENCE_COLUMNS:
        if c not in new.columns:
            new[c] = ""
    new = new[EVIDENCE_COLUMNS].fillna("")

    # Replace same incident/source-key records so reruns update rather than duplicate.
    keys = set(zip(new["Incident ID"].astype(str), new["Source Key"].astype(str)))
    if not current.empty:
        keep = [
            (str(iid), str(skey)) not in keys
            for iid, skey in zip(current["Incident ID"], current["Source Key"])
        ]
        current = current.loc[keep]

    set_evidence(pd.concat([current, new], ignore_index=True))


def get_classifications():
    init_state()
    return st.session_state["classification_df"]


def set_classifications(df):
    out = df.copy()
    for c in CLASS_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    st.session_state["classification_df"] = out[CLASS_COLUMNS].fillna("")


def append_classifications(rows):
    current = get_classifications()
    new = pd.DataFrame(rows)
    if new.empty:
        return

    for c in CLASS_COLUMNS:
        if c not in new.columns:
            new[c] = ""
    new = new[CLASS_COLUMNS].fillna("")

    ids = set(new["Incident ID"].astype(str))
    if not current.empty:
        current = current[~current["Incident ID"].astype(str).isin(ids)]

    set_classifications(pd.concat([current, new], ignore_index=True))
