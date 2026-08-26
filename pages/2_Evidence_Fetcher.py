from __future__ import annotations

import pandas as pd
import streamlit as st

from data_store import get_dataset, with_ids
from fetch_engine import fetch_url
from state_store import append_evidence, get_evidence, set_evidence


st.set_page_config(page_title="Evidence Fetcher", page_icon="🌐", layout="wide")

df = with_ids(get_dataset())
evidence = get_evidence()

st.title("🌐 Evidence Fetcher")
st.caption(
    "Free fetch cascade: direct HTML/PDF → Jina Reader basic/free fallback → "
    "Wayback snapshot. No paid API key is required."
)

st.info(
    "The classifier page can now fetch URLs automatically too. "
    "This page remains useful for testing URLs and auditing failures before classification."
)

with st.expander("Import previous evidence CSV"):
    up = st.file_uploader("Upload evidence_state.csv", type=["csv"], key="evidence_upload")
    if up is not None and st.button("Load evidence CSV"):
        set_evidence(pd.read_csv(up).fillna(""))
        st.success("Evidence state loaded.")
        st.rerun()

st.subheader("Test one URL")
test_url = st.text_input("Public URL", placeholder="https://...")
if st.button("Test fetch", disabled=not test_url.strip()):
    with st.spinner("Fetching URL..."):
        res = fetch_url(test_url, use_jina=True, use_wayback=True)
    st.write({
        "Status": res.get("status"),
        "Final URL": res.get("final_url"),
        "Source type": res.get("source_type"),
        "Characters": len(res.get("text", "") or ""),
        "Error": res.get("error", ""),
    })
    if res.get("text"):
        st.text_area("Extracted text preview", res["text"][:8000], height=280)

st.divider()
st.subheader("Batch fetch dataset URLs")

f1, f2, f3 = st.columns(3)
sectors = sorted(df["Sector"].unique())

with f1:
    sector = st.selectbox("Sector", ["All"] + sectors)
with f2:
    max_urls = st.slider("Maximum URLs per incident", 1, 4, 2)
with f3:
    use_wayback = st.checkbox("Try Wayback after direct/Jina failure", value=True)

subset = df if sector == "All" else df[df["Sector"] == sector]

if len(subset) > 0:
    start, end = st.slider(
        "Rows in current filtered set",
        0,
        len(subset),
        (0, min(10, len(subset))),
    )
else:
    start, end = 0, 0

runset = subset.iloc[start:end]
st.write(f"Selected incidents: **{len(runset)}**")

if st.button("Fetch selected incidents", type="primary", disabled=runset.empty):
    rows = []
    progress = st.progress(0.0)
    msg = st.empty()

    for pos, (_, r) in enumerate(runset.iterrows(), start=1):
        msg.write(f"Fetching {pos}/{len(runset)} — {r['Incident Name']}")

        for n in range(1, max_urls + 1):
            key = f"URL{n}"
            url = str(r.get(key, "") or "").strip()
            if not url:
                continue

            res = fetch_url(url, use_jina=True, use_wayback=use_wayback)
            rows.append({
                "Incident ID": r["Incident ID"],
                "Incident Name": r["Incident Name"],
                "Source Key": key,
                "Original URL": url,
                "Final URL": res.get("final_url", ""),
                "Fetch Status": res.get("status", "FAILED"),
                "Source Type": res.get("source_type", ""),
                "Characters": len(res.get("text", "") or ""),
                "Extracted Text": res.get("text", "") or "",
                "Error": res.get("error", "") or "",
            })

        progress.progress(pos / max(len(runset), 1))

    append_evidence(rows)
    st.success(f"Saved {len(rows)} URL evidence records in this session.")
    st.rerun()

st.divider()
st.subheader("Manual fallback")

incident = st.selectbox(
    "Incident",
    df["Incident ID"].tolist(),
    format_func=lambda iid: df.loc[df["Incident ID"] == iid, "Incident Name"].iloc[0],
)
manual = st.text_area(
    "Paste relevant source text when URLs cannot be fetched",
    height=180,
)

if st.button("Save manual evidence", disabled=not manual.strip()):
    name = df.loc[df["Incident ID"] == incident, "Incident Name"].iloc[0]
    append_evidence([{
        "Incident ID": incident,
        "Incident Name": name,
        "Source Key": "MANUAL",
        "Original URL": "",
        "Final URL": "",
        "Fetch Status": "MANUAL",
        "Source Type": "TEXT",
        "Characters": len(manual),
        "Extracted Text": manual,
        "Error": "",
    }])
    st.success("Manual evidence saved.")
    st.rerun()

st.divider()
evidence = get_evidence()
st.subheader("Evidence audit")

if evidence.empty:
    st.info("No fetched evidence yet.")
else:
    audit = evidence.drop(columns=["Extracted Text"]).copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Evidence records", len(evidence))
    c2.metric(
        "Successful",
        int(evidence["Fetch Status"].astype(str).str.startswith("FETCHED").sum())
        + int((evidence["Fetch Status"] == "MANUAL").sum()),
    )
    c3.metric(
        "Failed",
        int(evidence["Fetch Status"].astype(str).str.contains("FAILED").sum()),
    )

    st.dataframe(audit, use_container_width=True, hide_index=True, height=420)
    st.download_button(
        "⬇ Download evidence_state.csv",
        evidence.to_csv(index=False).encode("utf-8-sig"),
        "evidence_state.csv",
        "text/csv",
    )
