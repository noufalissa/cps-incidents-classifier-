from __future__ import annotations

import pandas as pd
import streamlit as st

from data_store import get_dataset, with_ids
from fetch_engine import fetch_url
from model_engine import NLI_MODELS, classify_incident
from state_store import (
    append_classifications,
    append_evidence,
    get_classifications,
    get_evidence,
    set_classifications,
)


st.set_page_config(
    page_title="Local Transformer Classifier",
    page_icon="🧠",
    layout="wide",
)

df = with_ids(get_dataset())

st.title("🧠 Free Local Transformer Classifier")
st.caption(
    "Free local Transformer pipeline: incident-scoped URL fetching + "
    "Transformer evidence retrieval + NLI classification. No TF-IDF and no paid API."
)

st.success(
    "v3 fixes two problems: (1) the current run uses exactly the selected incidents "
    "and URL1…URLN only; (2) generic/background paragraphs from unrelated incidents "
    "on a fetched page are filtered by incident relevance."
)

model_label = st.selectbox("NLI model", list(NLI_MODELS.keys()), index=0)
model_id = NLI_MODELS[model_label]

c1, c2, c3 = st.columns(3)
with c1:
    topk = st.select_slider(
        "Evidence chunks tested per property",
        options=[2, 3, 4, 5],
        value=3,
    )
with c2:
    sensitivity = st.selectbox(
        "Decision sensitivity",
        ["Balanced", "Conservative", "Higher recall"],
        index=0,
    )
with c3:
    auto_fetch = st.checkbox(
        "Fetch URLs automatically before classification",
        value=True,
    )

max_urls = 2
if auto_fetch:
    max_urls = st.slider(
        "Maximum URLs for THIS run",
        1, 4, 2,
        help="If set to 2, this run will use only URL1 and URL2. Old URL3/URL4 evidence is ignored.",
    )

reuse_previous = st.checkbox(
    "Reuse already-fetched matching URL1…URLN from this session",
    value=True,
    help=(
        "Reuses only the same selected incident and same allowed URL key. "
        "It never imports URL3 when Maximum URLs = 2."
    ),
)

with st.expander("Import previous classification CSV"):
    up = st.file_uploader("Upload classification_results.csv", type=["csv"], key="class_upload")
    if up is not None and st.button("Load classification CSV"):
        set_classifications(pd.read_csv(up).fillna(""))
        st.success("Loaded.")
        st.rerun()

sectors = sorted(df["Sector"].unique())
sector = st.selectbox("Sector filter", ["All"] + sectors)
subset = df if sector == "All" else df[df["Sector"] == sector]

if len(subset):
    start, end = st.slider(
        "Rows in filtered set",
        0, len(subset),
        (0, min(3, len(subset))),
    )
else:
    start, end = 0, 0

runset = subset.iloc[start:end]

st.write(f"Selected incidents for THIS run: **{len(runset)}**")
if len(runset):
    url_cols = ["URL1", "URL2", "URL3", "URL4"][:max_urls]
    st.dataframe(
        runset[["Year", "Incident Name", "Sector"] + url_cols],
        use_container_width=True,
        hide_index=True,
    )


def _previous_matching(r, source_key, url):
    ev = get_evidence()
    if ev.empty:
        return None
    hit = ev[
        (ev["Incident ID"].astype(str) == str(r["Incident ID"]))
        & (ev["Source Key"].astype(str) == source_key)
        & (ev["Original URL"].astype(str) == url)
    ]
    if hit.empty:
        return None
    last = hit.iloc[-1]
    if (
        str(last["Fetch Status"]) in {"FETCHED_DIRECT", "FETCHED_JINA", "FETCHED_WAYBACK", "MANUAL"}
        and str(last["Extracted Text"]).strip()
    ):
        return last
    return None


def fetch_for_current_run(r):
    """
    Returns ONLY evidence allowed by max_urls for this selected incident.
    It may reuse matching saved URL1...URLN, but never stale URL3/URL4.
    """
    current_rows = []

    for n in range(1, max_urls + 1):
        key = f"URL{n}"
        url = str(r.get(key, "") or "").strip()
        if not url:
            continue

        old = _previous_matching(r, key, url) if reuse_previous else None
        if old is not None:
            current_rows.append({
                "Incident ID": r["Incident ID"],
                "Incident Name": r["Incident Name"],
                "Source Key": key,
                "Original URL": url,
                "Final URL": str(old["Final URL"]),
                "Fetch Status": "REUSED_" + str(old["Fetch Status"]),
                "Source Type": str(old["Source Type"]),
                "Characters": int(old["Characters"]) if str(old["Characters"]).strip() else len(str(old["Extracted Text"])),
                "Extracted Text": str(old["Extracted Text"]),
                "Error": "",
            })
            continue

        res = fetch_url(url, use_jina=True, use_wayback=True)
        row = {
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
        }
        current_rows.append(row)

        # Persist the real fetch result for later reuse/audit.
        append_evidence([row])

    return current_rows


def records_for_current_run(r, current_fetch_rows):
    records = []

    summary = str(r.get("Verified Impact Summary", "") or "").strip()
    if summary:
        records.append({
            "text": summary,
            "source": "Verified Impact Summary",
            "url": "",
        })

    for e in current_fetch_rows:
        text = str(e.get("Extracted Text", "") or "").strip()
        if text:
            records.append({
                "text": text,
                "source": str(e.get("Source Key", "")),
                "url": str(e.get("Final URL", "") or e.get("Original URL", "")),
            })

    return records


if st.button(
    "Fetch + classify selected incidents" if auto_fetch else "Classify selected incidents",
    type="primary",
    disabled=runset.empty,
):
    all_decisions = []
    current_fetch_audit = []
    progress = st.progress(0.0)
    msg = st.empty()

    for pos, (_, r) in enumerate(runset.iterrows(), start=1):
        msg.write(f"{pos}/{len(runset)} — {r['Incident Name']}")

        fetch_rows = fetch_for_current_run(r) if auto_fetch else []
        current_fetch_audit.extend(fetch_rows)

        records = records_for_current_run(r, fetch_rows)

        incident_summary = str(r.get("Verified Impact Summary", "") or "")
        incident_context = " | ".join([
            str(r.get("Incident Name", "") or ""),
            str(r.get("Country/Region", "") or ""),
            str(r.get("Attack Type", "") or ""),
            str(r.get("Attacker / Group", "") or ""),
            incident_summary,
        ])

        results = classify_incident(
            records,
            model_id=model_id,
            top_k_chunks=topk,
            sensitivity=sensitivity,
            incident_context=incident_context,
            incident_summary=incident_summary,
        )

        for x in results:
            all_decisions.append({
                "Incident ID": r["Incident ID"],
                "Incident Name": r["Incident Name"],
                "Year": r["Year"],
                "Sector": r["Sector"],
                **x,
                "Model": model_id,
            })

        progress.progress(pos / max(len(runset), 1))

    st.session_state["last_run_fetch_audit"] = pd.DataFrame(current_fetch_audit)
    st.session_state["last_run_incident_ids"] = runset["Incident ID"].astype(str).tolist()

    append_classifications(all_decisions)
    st.success(
        f"Finished THIS run: {len(runset)} incidents, "
        f"{len(current_fetch_audit)} URL records, {len(all_decisions)} property decisions."
    )
    st.rerun()

st.divider()

st.subheader("Fetch status — LAST CLASSIFIER RUN ONLY")
last_fetch = st.session_state.get("last_run_fetch_audit", pd.DataFrame())
if last_fetch.empty:
    st.info("Run the classifier to see the fetch audit for that exact run.")
else:
    audit_cols = [
        "Incident Name", "Source Key", "Original URL", "Final URL",
        "Fetch Status", "Source Type", "Characters", "Error"
    ]
    st.dataframe(
        last_fetch[[c for c in audit_cols if c in last_fetch.columns]],
        use_container_width=True,
        hide_index=True,
        height=300,
    )
    st.caption(
        f"This table contains {len(last_fetch)} URL records from the last classifier run only. "
        f"Maximum URLs for that run was {max_urls}."
    )

st.subheader("Classification results")
cl = get_classifications()
if cl.empty:
    st.info("No classifications yet.")
else:
    last_ids = set(st.session_state.get("last_run_incident_ids", []))
    if last_ids:
        show_base = cl[cl["Incident ID"].astype(str).isin(last_ids)].copy()
        st.caption("Showing the most recently selected incidents only.")
    else:
        show_base = cl.copy()

    counts = show_base["Status"].value_counts()
    m = st.columns(5)
    for box, label in zip(
        m, ["CONFIRMED", "POTENTIAL", "CLAIMED", "UNAFFECTED", "UNKNOWN"]
    ):
        box.metric(label, int(counts.get(label, 0)))

    status_filter = st.multiselect(
        "Status",
        ["CONFIRMED", "POTENTIAL", "CLAIMED", "UNAFFECTED", "UNKNOWN"],
        default=["CONFIRMED", "POTENTIAL", "UNAFFECTED"],
    )
    show = show_base if not status_filter else show_base[show_base["Status"].isin(status_filter)]

    preferred = [
        "Incident Name", "Property", "Parent Category", "Status",
        "Status Score", "Incident Relevance", "Semantic Similarity",
        "Contradiction Score", "Neutral Score", "Evidence Source",
        "Evidence", "Decision Reason"
    ]
    st.dataframe(
        show[[c for c in preferred if c in show.columns]],
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    st.download_button(
        "⬇ Download classification_results.csv",
        cl.to_csv(index=False).encode("utf-8-sig"),
        "classification_results.csv",
        "text/csv",
    )

    confirmed = cl[cl["Status"] == "CONFIRMED"]
    st.download_button(
        "⬇ Download confirmed_properties.csv",
        confirmed.to_csv(index=False).encode("utf-8-sig"),
        "confirmed_properties.csv",
        "text/csv",
    )
