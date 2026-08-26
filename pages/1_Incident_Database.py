from __future__ import annotations
import pandas as pd
import streamlit as st
from data_store import COLUMNS, get_dataset, set_dataset, reset_dataset, csv_bytes

st.set_page_config(page_title="Incident Database", page_icon="🗂️", layout="wide")
df=get_dataset().copy()
st.title("🗂️ Incident Database")
st.caption("Browse the 1,207-incident dataset. Add/edit/delete operations are session-local; download the CSV to make them permanent in GitHub.")

c1,c2,c3,c4=st.columns([1.3,1.2,1.4,2])
sectors=sorted([x for x in df["Sector"].unique() if str(x).strip()])
years=sorted(pd.to_numeric(df["Year"],errors="coerce").dropna().astype(int).unique().tolist())
countries=sorted([x for x in df["Country/Region"].unique() if str(x).strip()])
with c1: fs=st.multiselect("Sector",sectors)
with c2: fy=st.multiselect("Year",years)
with c3: fc=st.multiselect("Country/Region",countries)
with c4: q=st.text_input("Search")
view=df.copy()
if fs: view=view[view["Sector"].isin(fs)]
if fy: view=view[pd.to_numeric(view["Year"],errors="coerce").isin(fy)]
if fc: view=view[view["Country/Region"].isin(fc)]
if q:
    mask=pd.Series(False,index=view.index); ql=q.lower()
    for col in ["Incident Name","Attack Type","Attacker / Group","Verified Impact Summary","Verification Status"]:
        mask |= view[col].astype(str).str.lower().str.contains(ql,regex=False,na=False)
    view=view[mask]
st.metric("Matching incidents",len(view))
st.dataframe(view,hide_index=True,use_container_width=True,height=470)
st.download_button("⬇ Download filtered CSV",csv_bytes(view),"cps_incidents_filtered.csv","text/csv")

st.divider(); st.subheader("Manage records")
t1,t2,t3,t4=st.tabs(["➕ Add","✏️ Edit","🗑️ Delete","💾 Export / Reset"])
with t1:
    with st.form("add"):
        a,b,c=st.columns(3)
        with a: year=st.number_input("Year",1900,2100,2026); country=st.text_input("Country/Region")
        with b: name=st.text_input("Incident Name *"); sector=st.selectbox("Sector",sectors)
        with c: attack=st.text_input("Attack Type"); attacker=st.text_input("Attacker / Group")
        impact=st.text_area("Verified Impact Summary",height=120); source=st.text_input("Source / Verification URL"); status=st.text_input("Verification Status")
        u1,u2=st.columns(2)
        with u1: url1=st.text_input("URL1"); url2=st.text_input("URL2")
        with u2: url3=st.text_input("URL3"); url4=st.text_input("URL4")
        submit=st.form_submit_button("Add incident",type="primary")
    if submit:
        if not name.strip(): st.error("Incident Name is required.")
        else:
            row=pd.DataFrame([[year,name,country,sector,attack,attacker,impact,source,status,url1,url2,url3,url4]],columns=COLUMNS)
            set_dataset(pd.concat([df,row],ignore_index=True)); st.success("Added."); st.rerun()
with t2:
    opts=view.index.tolist()
    if not opts: st.info("No matching incident to edit.")
    else:
        idx=st.selectbox("Incident",opts,format_func=lambda i:f"#{i+1} — {df.loc[i,'Incident Name']} ({df.loc[i,'Year']})")
        r=df.loc[idx]
        with st.form("edit"):
            vals={}
            aa,bb,cc=st.columns(3)
            with aa:
                vals["Year"]=st.number_input("Year",1900,2100,int(r["Year"]) if pd.notna(r["Year"]) else 2026)
                vals["Country/Region"]=st.text_input("Country/Region",str(r["Country/Region"]))
            with bb:
                vals["Incident Name"]=st.text_input("Incident Name *",str(r["Incident Name"]))
                so=sectors if str(r["Sector"]) in sectors else sectors+[str(r["Sector"])]
                vals["Sector"]=st.selectbox("Sector",so,index=so.index(str(r["Sector"])))
            with cc:
                vals["Attack Type"]=st.text_input("Attack Type",str(r["Attack Type"])); vals["Attacker / Group"]=st.text_input("Attacker / Group",str(r["Attacker / Group"]))
            vals["Verified Impact Summary"]=st.text_area("Verified Impact Summary",str(r["Verified Impact Summary"]),height=130)
            vals["Source / Verification URL"]=st.text_input("Source / Verification URL",str(r["Source / Verification URL"]))
            vals["Verification Status"]=st.text_input("Verification Status",str(r["Verification Status"]))
            d,e=st.columns(2)
            with d: vals["URL1"]=st.text_input("URL1",str(r["URL1"])); vals["URL2"]=st.text_input("URL2",str(r["URL2"]))
            with e: vals["URL3"]=st.text_input("URL3",str(r["URL3"])); vals["URL4"]=st.text_input("URL4",str(r["URL4"]))
            save=st.form_submit_button("Save",type="primary")
        if save:
            if not vals["Incident Name"].strip(): st.error("Incident Name is required.")
            else:
                updated=df.copy()
                for col in COLUMNS: updated.loc[idx,col]=vals[col]
                set_dataset(updated); st.success("Updated."); st.rerun()
with t3:
    opts=view.index.tolist()
    if not opts: st.info("No matching incident to delete.")
    else:
        idx=st.selectbox("Incident to delete",opts,format_func=lambda i:f"#{i+1} — {df.loc[i,'Incident Name']}",key="del")
        confirm=st.checkbox("I confirm deletion")
        if st.button("Delete",type="primary",disabled=not confirm):
            set_dataset(df.drop(index=idx).reset_index(drop=True)); st.rerun()
with t4:
    st.download_button("⬇ Download current full incidents.csv",csv_bytes(get_dataset()),"incidents.csv","text/csv",use_container_width=True)
    if st.button("Reset to repository CSV"):
        reset_dataset(); st.rerun()
