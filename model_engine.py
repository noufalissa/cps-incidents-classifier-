from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
import streamlit as st
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from taxonomy import (
    HARD_NEGATIVE_PROTOTYPES,
    PAPER_DEFINITIONS,
    PARENT,
    POSITIVE_PROTOTYPES,
    PROPERTIES,
    PROPERTY_SPECS,
    STATUS_PROTOTYPES,
)


RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NLI_MODELS = {
    "Fast / smaller": "cross-encoder/nli-MiniLM2-L6-H768",
    "Stronger / larger": "cross-encoder/nli-deberta-v3-base",
}

# These patterns do NOT assign a property.  They only protect the evidence status
# after the Transformer has selected an incident-relevant property candidate.
NEGATION_PATTERNS = [
    r"\bno\b.{0,110}\b(impact|disruption|outage|damage|harm|compromise|breach|access|exposure|theft|loss|effect|interruption|modification|change)\b",
    r"\bnot\b.{0,110}\b(affected|impacted|disrupted|compromised|breached|accessed|exposed|stolen|damaged|interrupted|modified|altered)\b",
    r"\b(remained|was|were|continued)\s+(unaffected|available|operational|intact|secure|normal)\b",
    r"\bwithout\b.{0,90}\b(impact|disruption|damage|compromise|exposure|outage|loss|modification)\b",
    r"\bno confirmed\b",
    r"\bno evidence of\b",
]

POTENTIAL_PATTERNS = [
    r"\bcould\b", r"\bmay\b", r"\bmight\b", r"\bpotential(?:ly)?\b",
    r"\bpossible\b", r"\bcapable of\b", r"\bdesigned to\b",
    r"\bintended to\b", r"\brisk of\b", r"\bwould allow\b",
    r"\bdemonstrat(?:e|ed|ion)\b", r"\bproof[- ]of[- ]concept\b",
    r"\btheoretical(?:ly)?\b", r"\bcan be used to\b",
]

CLAIM_PATTERNS = [
    r"\bclaim(?:ed|s)?\b", r"\balleg(?:ed|edly|ation)\b",
    r"\breportedly\b", r"\bunverified\b", r"\bnot independently confirmed\b",
    r"\baccording to (?:the )?(?:attacker|attackers|hacker|hackers|group)\b",
    r"\bsaid it (?:had|has) stolen\b", r"\btook responsibility\b",
]

QUESTION_PATTERNS = [
    r"\?$", r"\bwas .+ attacked\?", r"\bcould .+\?", r"\bmay .+\?",
]


@dataclass
class Chunk:
    text: str
    source: str
    url: str


def _has_pattern(text: str, patterns) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t, flags=re.I | re.S) for p in patterns)


def split_chunks(records, max_chars: int = 900) -> List[Chunk]:
    """Sentence + two-sentence windows with deduplication."""
    chunks = []
    seen = set()

    for rec in records:
        text = re.sub(r"\s+", " ", str(rec.get("text", "") or "")).strip()
        if not text:
            continue

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]
        if not sentences:
            sentences = [text]

        windows = []
        for i, sentence in enumerate(sentences):
            windows.append(sentence[:max_chars])
            if i + 1 < len(sentences):
                pair = sentence + " " + sentences[i + 1]
                if len(pair) <= max_chars:
                    windows.append(pair)

        for w in windows:
            key = re.sub(r"\W+", " ", w.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                chunks.append(
                    Chunk(
                        text=w,
                        source=str(rec.get("source", "")),
                        url=str(rec.get("url", "")),
                    )
                )

    return chunks


@st.cache_resource(show_spinner="Loading free local retrieval Transformer...")
def load_retriever():
    tok = AutoTokenizer.from_pretrained(RETRIEVER_MODEL)
    model = AutoModel.from_pretrained(RETRIEVER_MODEL)
    model.eval()
    return tok, model


@st.cache_resource(show_spinner="Loading free local NLI Transformer...")
def load_nli(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tok, model


def _mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts, batch_size=24):
    tok, model = load_retriever()
    out = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            features = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            hidden = model(**features).last_hidden_state
            emb = _mean_pool(hidden, features["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())

    return np.vstack(out) if out else np.empty((0, 384))


def _label_indexes(model):
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    def find(name, fallback):
        for idx, label in labels.items():
            if name in label:
                return idx
        return fallback

    return {
        "contradiction": find("contrad", 0),
        "entailment": find("entail", 1),
        "neutral": find("neutral", 2),
    }


def nli_probabilities(pairs, model_id: str, batch_size=16):
    tok, model = load_nli(model_id)
    idx = _label_indexes(model)
    rows = []

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            b = pairs[i:i + batch_size]
            premises = [x[0] for x in b]
            hypotheses = [x[1] for x in b]

            features = tok(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            logits = model(**features).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for p in probs:
                rows.append({
                    "contradiction": float(p[idx["contradiction"]]),
                    "entailment": float(p[idx["entailment"]]),
                    "neutral": float(p[idx["neutral"]]),
                })

    return rows


def _operating_point(model_id: str, sensitivity: str):
    """
    Conservative is the recommended research setting until a manual gold subset
    is available.  These are operating points, not calibrated probabilities.
    """
    stronger = "deberta" in model_id.lower()
    if sensitivity == "Conservative":
        return {
            "confirmed": 0.62 if not stronger else 0.58,
            "status_margin": 0.10,
            "contrast_margin": 0.035,
            "prototype_floor": 0.27,
        }
    if sensitivity == "Higher recall":
        return {
            "confirmed": 0.43 if not stronger else 0.40,
            "status_margin": 0.035,
            "contrast_margin": 0.005,
            "prototype_floor": 0.20,
        }
    return {
        "confirmed": 0.52 if not stronger else 0.48,
        "status_margin": 0.065,
        "contrast_margin": 0.020,
        "prototype_floor": 0.235,
    }


def _incident_relevance_scores(chunks, incident_context: str):
    if not chunks:
        return np.array([])
    if not incident_context.strip():
        return np.ones(len(chunks), dtype=float)
    chunk_emb = encode_texts([c.text for c in chunks])
    ctx_emb = encode_texts([incident_context])
    return (ctx_emb @ chunk_emb.T).reshape(-1)


def _prototype_matrices(chunks):
    """
    Contrast each evidence chunk against positive and hard-negative semantic
    prototypes for every property.  This is the custom taxonomy layer.
    """
    chunk_emb = encode_texts([c.text for c in chunks])

    positive_texts = []
    positive_owner = []
    negative_texts = []
    negative_owner = []

    for prop in PROPERTIES:
        for text in POSITIVE_PROTOTYPES[prop]:
            positive_texts.append(text)
            positive_owner.append(prop)
        for text in HARD_NEGATIVE_PROTOTYPES[prop]:
            negative_texts.append(text)
            negative_owner.append(prop)

    pos_emb = encode_texts(positive_texts)
    neg_emb = encode_texts(negative_texts)

    pos_all = chunk_emb @ pos_emb.T
    neg_all = chunk_emb @ neg_emb.T

    pos_scores = np.zeros((len(PROPERTIES), len(chunks)), dtype=float)
    neg_scores = np.zeros((len(PROPERTIES), len(chunks)), dtype=float)

    for pi, prop in enumerate(PROPERTIES):
        pidx = [i for i, owner in enumerate(positive_owner) if owner == prop]
        nidx = [i for i, owner in enumerate(negative_owner) if owner == prop]
        pos_scores[pi, :] = np.max(pos_all[:, pidx], axis=1)
        neg_scores[pi, :] = np.max(neg_all[:, nidx], axis=1)

    return chunk_emb, pos_scores, neg_scores


def _prototype_nli_for_candidates(candidates, model_id: str):
    """
    Contrastive entailment:
      evidence -> positive property prototypes
      evidence -> hard-negative/confusable prototypes

    This avoids the unstable four-way status competition from v4.
    """
    pairs = []
    meta = []

    for item in candidates:
        prop = item["Property"]
        for idx, hypothesis in enumerate(POSITIVE_PROTOTYPES[prop]):
            pairs.append((item["Evidence"], hypothesis))
            meta.append((prop, item["Chunk Index"], "POS", idx))
        for idx, hypothesis in enumerate(HARD_NEGATIVE_PROTOTYPES[prop]):
            pairs.append((item["Evidence"], hypothesis))
            meta.append((prop, item["Chunk Index"], "NEG", idx))

    probs = nli_probabilities(pairs, model_id=model_id)

    by_candidate = {}
    for (prop, ci, side, idx), prob in zip(meta, probs):
        key = (prop, ci)
        rec = by_candidate.setdefault(key, {"POS": [], "NEG": []})
        rec[side].append({
            "index": idx,
            "entailment": float(prob["entailment"]),
            "contradiction": float(prob["contradiction"]),
            "neutral": float(prob["neutral"]),
        })

    return by_candidate


def _decision_params(model_id: str, sensitivity: str):
    """
    Operating points for the custom contrastive-entailment classifier.
    They are NOT calibrated probabilities; validation on a manually labelled
    subset is still required before the final 1,207-incident run.
    """
    stronger = "deberta" in model_id.lower()

    if sensitivity == "Conservative":
        return {
            "positive_nli": 0.60 if not stronger else 0.56,
            "nli_margin": 0.15,
            "semantic_floor": 0.30,
            "semantic_margin": 0.035,
            "diagnostic_floor": 0.26,
        }

    if sensitivity == "Higher recall":
        return {
            "positive_nli": 0.34 if not stronger else 0.31,
            "nli_margin": 0.055,
            "semantic_floor": 0.21,
            "semantic_margin": 0.000,
            "diagnostic_floor": 0.18,
        }

    # Balanced
    return {
        "positive_nli": 0.46 if not stronger else 0.42,
        "nli_margin": 0.095,
        "semantic_floor": 0.25,
        "semantic_margin": 0.015,
        "diagnostic_floor": 0.22,
    }


def classify_incident(
    records,
    model_id: str,
    top_k_chunks: int = 3,
    sensitivity: str = "Balanced",
    incident_context: str = "",
    incident_summary: str = "",
    fetched_relevance_floor: float = 0.24,
):
    """
    CE-18 v4.1: paper-grounded Contrastive Entailment classifier.

    Pipeline
    --------
    1. Split description + fetched evidence into passages.
    2. Keep passages relevant to the selected incident.
    3. Retrieve property candidates with paper-derived positive and hard-negative
       Transformer prototypes.
    4. Run LOCAL NLI directly against the positive prototypes AND the
       hard-negative/confusable prototypes.
    5. A property is CONFIRMED only when:
         - positive semantic support is sufficient,
         - positive NLI entailment is sufficient,
         - positive evidence beats the property's hard-negative evidence,
         - the evidence is factual (not hypothetical / merely claimed / negated).
    6. Keep all confirmed labels. No top-3 truncation.

    No TF-IDF and no paid API are used.
    """
    chunks = split_chunks(records)
    if not chunks:
        return []

    params = _decision_params(model_id, sensitivity)
    _, positive_scores, negative_scores = _prototype_matrices(chunks)
    incident_rel = _incident_relevance_scores(chunks, incident_context)

    candidates = []

    for pi, prop in enumerate(PROPERTIES):
        ranked = []

        for ci, ch in enumerate(chunks):
            is_primary = ch.source in {"Verified Impact Summary", "MANUAL"}
            rel = float(incident_rel[ci])

            if not is_primary and rel < fetched_relevance_floor:
                continue

            pos_sem = float(positive_scores[pi, ci])
            neg_sem = float(negative_scores[pi, ci])
            sem_margin = pos_sem - neg_sem

            # Retrieval only. This does not assign the property.
            joint = (
                0.52 * pos_sem
                + 0.23 * max(sem_margin, -0.20)
                + 0.25 * (1.0 if is_primary else rel)
            )
            ranked.append((joint, ci, pos_sem, neg_sem, sem_margin, rel))

        ranked.sort(reverse=True)

        for joint, ci, pos_sem, neg_sem, sem_margin, rel in ranked[:max(1, top_k_chunks)]:
            candidates.append({
                "Property": prop,
                "Chunk Index": ci,
                "Evidence": chunks[ci].text,
                "Evidence Source": chunks[ci].source,
                "Evidence URL": chunks[ci].url,
                "Positive Semantic": pos_sem,
                "Negative Semantic": neg_sem,
                "Semantic Margin": sem_margin,
                "Incident Relevance": rel,
                "Retrieval Joint": joint,
            })

    if not candidates:
        return []

    contrast_nli = _prototype_nli_for_candidates(candidates, model_id=model_id)
    results = []

    for prop in PROPERTIES:
        evaluated = []

        for item in [c for c in candidates if c["Property"] == prop]:
            scores = contrast_nli.get((prop, item["Chunk Index"]))
            if not scores:
                continue

            pos_rows = scores["POS"]
            neg_rows = scores["NEG"]

            best_pos = max(pos_rows, key=lambda x: x["entailment"])
            best_neg = max(neg_rows, key=lambda x: x["entailment"])

            pos_nli = float(best_pos["entailment"])
            neg_nli = float(best_neg["entailment"])
            nli_margin = pos_nli - neg_nli

            evidence = item["Evidence"].strip()

            has_negation = _has_pattern(evidence, NEGATION_PATTERNS)
            has_potential = _has_pattern(evidence, POTENTIAL_PATTERNS)
            has_claim = _has_pattern(evidence, CLAIM_PATTERNS)
            question_like = _has_pattern(evidence, QUESTION_PATTERNS)

            semantic_supported = (
                item["Positive Semantic"] >= params["semantic_floor"]
                and item["Semantic Margin"] >= params["semantic_margin"]
            )
            nli_supported = (
                pos_nli >= params["positive_nli"]
                and nli_margin >= params["nli_margin"]
            )

            diagnostic_supported = (
                item["Positive Semantic"] >= params["diagnostic_floor"]
                and (
                    item["Semantic Margin"] >= -0.02
                    or pos_nli >= max(0.30, params["positive_nli"] - 0.12)
                )
            )

            # IMPORTANT:
            # status cues are applied only AFTER property-specific support.
            # Therefore "No physical service disruption" can affect Safety or
            # Availability when their hard-negative evidence fits, but it no
            # longer marks all 18 properties UNAFFECTED.
            final_status = "UNKNOWN"
            reason = "Property-specific positive evidence did not beat its hard-negative/confusable evidence."

            if question_like:
                final_status = "UNKNOWN"
                reason = "Question/non-factual wording cannot establish a property violation."

            elif semantic_supported and nli_supported:
                if has_potential:
                    final_status = "POTENTIAL"
                    reason = "The property is supported, but the evidence explicitly describes a capability, demonstration, possibility, or hypothetical consequence."
                elif has_claim:
                    final_status = "CLAIMED"
                    reason = "The property is supported, but the evidence is explicitly framed as an allegation/claim rather than established fact."
                elif has_negation and neg_nli >= pos_nli - 0.02:
                    final_status = "UNAFFECTED"
                    reason = "Property-relevant evidence contains explicit no-impact/negation wording and the hard-negative entailment is competitive."
                else:
                    final_status = "CONFIRMED"
                    reason = "Actual incident evidence supports the paper-defined property and beats property-specific hard-negative counterexamples."

            elif diagnostic_supported:
                if has_potential and item["Semantic Margin"] >= 0.0:
                    final_status = "POTENTIAL"
                    reason = "The evidence is property-relevant but explicitly hypothetical/capability-oriented; it is not eligible for CONFIRMED."
                elif has_claim and item["Semantic Margin"] >= 0.0:
                    final_status = "CLAIMED"
                    reason = "The evidence is property-relevant but explicitly claim/allegation-oriented; it is not eligible for CONFIRMED."
                elif has_negation and neg_nli > pos_nli and item["Negative Semantic"] >= item["Positive Semantic"] - 0.03:
                    final_status = "UNAFFECTED"
                    reason = "A property-specific hard-negative case is better supported than an actual violation."

            # Ranking chooses the best evidence passage for THIS property.
            status_rank = {
                "CONFIRMED": 5,
                "POTENTIAL": 4,
                "CLAIMED": 3,
                "UNAFFECTED": 2,
                "UNKNOWN": 1,
            }[final_status]

            evidence_strength = (
                0.42 * pos_nli
                + 0.22 * max(nli_margin, -0.5)
                + 0.20 * item["Positive Semantic"]
                + 0.10 * max(item["Semantic Margin"], -0.5)
                + 0.06 * item["Incident Relevance"]
            )

            evaluated.append({
                **item,
                "Final Status": final_status,
                "Positive NLI": pos_nli,
                "Negative NLI": neg_nli,
                "NLI Margin": nli_margin,
                "Best Positive Prototype": POSITIVE_PROTOTYPES[prop][best_pos["index"]],
                "Best Negative Prototype": HARD_NEGATIVE_PROTOTYPES[prop][best_neg["index"]],
                "Reason": reason,
                "Rank": status_rank,
                "Evidence Strength": evidence_strength,
            })

        if not evaluated:
            results.append({
                "Property": prop,
                "Parent Category": PARENT[prop],
                "Status": "UNKNOWN",
                "Status Score": 0.0,
                "Semantic Similarity": 0.0,
                "Incident Relevance": 0.0,
                "Contradiction Score": 0.0,
                "Neutral Score": 0.0,
                "Evidence": "",
                "Evidence Source": "",
                "Evidence URL": "",
                "Decision Reason": "CE-18 v4.1: no incident-relevant evidence candidate.",
            })
            continue

        best = max(evaluated, key=lambda x: (x["Rank"], x["Evidence Strength"]))

        # Preserve the existing Streamlit/CSV schema.
        # Status Score            -> positive NLI entailment
        # Semantic Similarity     -> positive-prototype semantic similarity
        # Contradiction Score     -> hard-negative NLI entailment
        # Neutral Score           -> positive-minus-negative NLI margin
        results.append({
            "Property": prop,
            "Parent Category": PARENT[prop],
            "Status": best["Final Status"],
            "Status Score": round(best["Positive NLI"], 4),
            "Semantic Similarity": round(best["Positive Semantic"], 4),
            "Incident Relevance": round(best["Incident Relevance"], 4),
            "Contradiction Score": round(best["Negative NLI"], 4),
            "Neutral Score": round(best["NLI Margin"], 4),
            "Evidence": best["Evidence"],
            "Evidence Source": best["Evidence Source"],
            "Evidence URL": best["Evidence URL"],
            "Decision Reason": (
                f"CE-18 v4.1 | {best['Reason']} "
                f"| positive_sem={best['Positive Semantic']:.3f}; "
                f"hardneg_sem={best['Negative Semantic']:.3f}; "
                f"sem_margin={best['Semantic Margin']:.3f}; "
                f"positive_nli={best['Positive NLI']:.3f}; "
                f"hardneg_nli={best['Negative NLI']:.3f}; "
                f"nli_margin={best['NLI Margin']:.3f}; "
                f"positive_prototype=\"{best['Best Positive Prototype']}\"; "
                f"hard_negative=\"{best['Best Negative Prototype']}\"."
            ),
        })

    return results
