from __future__ import annotations
import pandas as pd
import streamlit as st
from data_store import initialize_dataset, get_dataset
from state_store import init_state, get_evidence, get_classifications

st.set_page_config(page_title="CPS Property Evidence Lab", page_icon="🧭", layout="wide")
initialize_dataset(); init_state()
df=get_dataset(); ev=get_evidence(); cl=get_classifications()

st.title("🧭 CPS Property Evidence Lab")
st.caption("Free, local Transformer/NLI classification of CPS incidents with URL evidence and correlation analysis.")

c1,c2,c3,c4=st.columns(4)
c1.metric("Incidents", f"{len(df):,}")
c2.metric("Sectors", df["Sector"].nunique())
c3.metric("Fetched/manual evidence records", f"{len(ev):,}")
c4.metric("Classified incident-property rows", f"{len(cl):,}")

st.markdown("### Fixed workflow")
st.markdown("**Incident Database → Evidence Fetcher → Local Transformer Classifier → Correlation Explorer**")
st.success("No paid API is used. No TF-IDF is used. Public Hugging Face models are downloaded and run locally in the Streamlit process.")
st.info("For reproducibility and cost control, this project uses a local Transformer semantic retriever plus a local Natural Language Inference (NLI) cross-encoder. The classifier only counts CONFIRMED properties in the scientific correlation analysis.")
