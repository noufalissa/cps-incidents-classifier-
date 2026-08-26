from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from data_store import with_ids, get_dataset
from state_store import get_classifications
from taxonomy import TAXONOMY, PROPERTIES
from analytics import confirmed_matrix, phi_matrix, cooccurrence_matrix, dimension_matrix, within_box_prevalence, pairwise_table

st.set_page_config(page_title="Correlation Explorer",page_icon="📊",layout="wide")
inc=with_ids(get_dataset()); cl=get_classifications()
st.title("📊 Property Correlation Explorer")
st.caption("This page implements the analysis requested in the discussion: study each taxonomy box separately, then inspect pairwise correlations and the full 18×18 matrix. Only CONFIRMED labels are used.")

if cl.empty:
    st.warning("No classification results are loaded. Classify incidents first or import classification_results.csv on the classifier page.")
    st.stop()

f1,f2=st.columns(2)
sectors=sorted(inc["Sector"].unique())
years=sorted(pd.to_numeric(inc["Year"],errors="coerce").dropna().astype(int).unique())
with f1: sector=st.selectbox("Sector",["All"]+sectors)
with f2:
    if years:
        yr=st.slider("Year range",int(min(years)),int(max(years)),(int(min(years)),int(max(years))))
    else: yr=(0,9999)
sub=inc.copy()
if sector!="All": sub=sub[sub["Sector"]==sector]
yv=pd.to_numeric(sub["Year"],errors="coerce")
sub=sub[(yv>=yr[0])&(yv<=yr[1])]
mat=confirmed_matrix(cl,sub)
st.write(f"Incidents in filter: **{len(sub)}** · Incidents with ≥1 confirmed property: **{int((mat.sum(axis=1)>0).sum())}**")


def heat(df,title,zmid=0):
    fig=px.imshow(df.astype(float),text_auto=".2f" if df.to_numpy().dtype.kind=="f" else True,aspect="auto",color_continuous_scale="RdBu",zmin=-1,zmax=1,color_continuous_midpoint=zmid,title=title)
    fig.update_layout(height=max(420,30*len(df.index)+160))
    st.plotly_chart(fig,use_container_width=True)

def count_heat(df,title):
    fig=px.imshow(df.astype(float),text_auto=True,aspect="auto",color_continuous_scale="Blues",title=title)
    fig.update_layout(height=max(420,30*len(df.index)+160))
    st.plotly_chart(fig,use_container_width=True)

T1,T2,T3,T4,T5=st.tabs(["3 Dimensions","Within each Box","Cross-Box","Full 18×18","Strongest Correlations"])
with T1:
    d=dimension_matrix(mat)
    c1,c2=st.columns(2)
    with c1: heat(phi_matrix(d,list(d.columns)),"3×3 Phi correlation")
    with c2: count_heat(cooccurrence_matrix(d,list(d.columns)),"3×3 co-occurrence")
with T2:
    parent=st.selectbox("Taxonomy box",list(TAXONOMY.keys()))
    prev=within_box_prevalence(mat,parent)
    fig=px.bar(prev,x="Property",y="Percent",text=prev["Percent"].map(lambda x:f"{x:.1f}%"),title=f"Among incidents involving {parent}")
    fig.update_yaxes(title="Percent of box incidents")
    st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2); props=TAXONOMY[parent]
    with c1: heat(phi_matrix(mat,props),f"{len(props)}×{len(props)} Phi matrix")
    with c2: count_heat(cooccurrence_matrix(mat,props),f"{len(props)}×{len(props)} co-occurrence")
with T3:
    pairs=[("Functional Correctness","Information Protection"),("Functional Correctness","Operational Assurance"),("Information Protection","Operational Assurance")]
    a,b=st.selectbox("Boxes",pairs,format_func=lambda x:f"{x[0]} ↔ {x[1]}")
    rows,cols=TAXONOMY[a],TAXONOMY[b]
    cross=pd.DataFrame(index=rows,columns=cols,dtype=float); counts=pd.DataFrame(index=rows,columns=cols,dtype=int)
    from analytics import phi_pair
    for r in rows:
        for c in cols:
            cross.loc[r,c]=phi_pair(mat[r],mat[c])[0]; counts.loc[r,c]=int(((mat[r]==1)&(mat[c]==1)).sum())
    c1,c2=st.columns(2)
    with c1: heat(cross,f"{len(rows)}×{len(cols)} cross-box Phi")
    with c2: count_heat(counts,f"{len(rows)}×{len(cols)} cross-box co-occurrence")
with T4:
    mode=st.radio("Matrix",["Phi correlation","Co-occurrence counts"],horizontal=True)
    if mode=="Phi correlation": heat(phi_matrix(mat,PROPERTIES),"Full 18×18 Phi correlation matrix")
    else: count_heat(cooccurrence_matrix(mat,PROPERTIES),"Full 18×18 co-occurrence matrix")
with T5:
    tbl=pairwise_table(mat)
    minco=st.number_input("Minimum co-occurrence",0,10000,2)
    sig=st.checkbox("Require BH-FDR q < 0.05",value=False)
    show=tbl[tbl["Co-occurrence"]>=minco].copy()
    if sig: show=show[show["q-value (BH-FDR)"]<0.05]
    st.dataframe(show.drop(columns=["|Phi|"]),use_container_width=True,hide_index=True,height=520)
    st.download_button("⬇ Download pairwise_correlations.csv",tbl.to_csv(index=False).encode("utf-8-sig"),"pairwise_correlations.csv","text/csv")
