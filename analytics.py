from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from taxonomy import TAXONOMY, PROPERTIES, PARENT


def confirmed_matrix(class_df: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    ids = incidents["Incident ID"].astype(str).tolist()
    mat = pd.DataFrame(0, index=ids, columns=PROPERTIES, dtype=int)
    if class_df is None or class_df.empty:
        return mat
    c = class_df[class_df["Status"].astype(str).eq("CONFIRMED")]
    for _, r in c.iterrows():
        iid, prop = str(r["Incident ID"]), str(r["Property"])
        if iid in mat.index and prop in mat.columns:
            mat.loc[iid, prop] = 1
    return mat


def phi_pair(a, b):
    a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
    n11 = int(((a==1)&(b==1)).sum()); n10 = int(((a==1)&(b==0)).sum())
    n01 = int(((a==0)&(b==1)).sum()); n00 = int(((a==0)&(b==0)).sum())
    den = math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    phi = ((n11*n00)-(n10*n01))/den if den else np.nan
    return phi, n11, n10, n01, n00


def phi_matrix(mat: pd.DataFrame, cols=None):
    cols = cols or list(mat.columns)
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            out.loc[a,b] = 1.0 if a==b else phi_pair(mat[a], mat[b])[0]
    return out


def cooccurrence_matrix(mat: pd.DataFrame, cols=None):
    cols = cols or list(mat.columns)
    x = mat[cols].astype(int)
    return x.T.dot(x)


def dimension_matrix(mat: pd.DataFrame):
    d = pd.DataFrame(index=mat.index)
    for parent, props in TAXONOMY.items():
        d[parent] = (mat[props].sum(axis=1) > 0).astype(int)
    return d


def within_box_prevalence(mat: pd.DataFrame, parent: str):
    props = TAXONOMY[parent]
    subset_mask = mat[props].sum(axis=1) > 0
    n = int(subset_mask.sum())
    vals = []
    for p in props:
        pct = float(mat.loc[subset_mask, p].mean()*100) if n else 0.0
        vals.append({"Property":p, "Percent":pct, "Count":int(mat.loc[subset_mask,p].sum()), "Box Incidents":n})
    return pd.DataFrame(vals)


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def pairwise_table(mat: pd.DataFrame):
    rows=[]
    for i,a in enumerate(PROPERTIES):
        for b in PROPERTIES[i+1:]:
            phi,n11,n10,n01,n00 = phi_pair(mat[a],mat[b])
            table = [[n11,n10],[n01,n00]]
            try:
                odds,p = fisher_exact(table)
            except Exception:
                odds,p = np.nan,1.0
            rows.append({"Property A":a,"Property B":b,"Co-occurrence":n11,"Phi":phi,"Odds Ratio":odds,"p-value":p})
    out=pd.DataFrame(rows)
    if not out.empty:
        out["q-value (BH-FDR)"] = bh_fdr(out["p-value"].fillna(1.0).values)
        out["|Phi|"] = out["Phi"].abs()
        out = out.sort_values(["|Phi|","Co-occurrence"], ascending=[False,False]).reset_index(drop=True)
    return out
