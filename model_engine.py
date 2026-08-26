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


def _status_nli_for_candidates(candidates, model_id: str):
    """
    Run a four-way status competition by NLI entailment:
      CONFIRMED / POTENTIAL / UNAFFECTED / CLAIMED.
    """
    pairs = []
    meta = []
    for item in candidates:
        prop = item["Property"]
        for status in ["CONFIRMED", "POTENTIAL", "UNAFFECTED", "CLAIMED"]:
            pairs.append((item["Evidence"], STATUS_PROTOTYPES[prop][status]))
            meta.append((item["Property"], item["Chunk Index"], status))

    probs = nli_probabilities(pairs, model_id=model_id)

    by_candidate = {}
    for (prop, ci, status), prob in zip(meta, probs):
        by_candidate.setdefault((prop, ci), {})[status] = prob
    return by_candidate


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
    Paper-grounded contrastive multi-label classifier.

    Pipeline:
      1. split incident description + fetched evidence into local passages;
      2. score passage relevance to the specific incident;
      3. contrast each passage against property-specific positive AND hard-negative
         Transformer prototypes derived from the paper taxonomy;
      4. run NLI competition among CONFIRMED/POTENTIAL/UNAFFECTED/CLAIMED;
      5. apply the paper's Stage-1 rule: only explicitly documented ACTUAL
         violations can become CONFIRMED; hypothetical consequences are excluded;
      6. retain all confirmed properties (no top-3 truncation).

    The classifier remains free/local.  TF-IDF and paid APIs are not used.
    """
    chunks = split_chunks(records)
    if not chunks:
        return []

    params = _operating_point(model_id, sensitivity)
    _, positive_scores, negative_scores = _prototype_matrices(chunks)
    incident_rel = _incident_relevance_scores(chunks, incident_context)

    # Build the top incident-specific contrastive candidates for every property.
    candidates = []
    candidate_map = {}

    for pi, prop in enumerate(PROPERTIES):
        ranked = []
        for ci, ch in enumerate(chunks):
            is_primary = ch.source in {"Verified Impact Summary", "MANUAL"}
            rel = float(incident_rel[ci])
            if not is_primary and rel < fetched_relevance_floor:
                continue

            pos = float(positive_scores[pi, ci])
            neg = float(negative_scores[pi, ci])
            contrast = pos - neg

            # Primary incident summary is trusted for incident identity, but it
            # still has to pass the property prototype contrast.
            joint = 0.55 * pos + 0.25 * max(contrast, -0.20) + 0.20 * (1.0 if is_primary else rel)
            ranked.append((joint, ci, pos, neg, contrast, rel))

        ranked.sort(reverse=True)
        for joint, ci, pos, neg, contrast, rel in ranked[:max(1, top_k_chunks)]:
            item = {
                "Property": prop,
                "Chunk Index": ci,
                "Evidence": chunks[ci].text,
                "Evidence Source": chunks[ci].source,
                "Evidence URL": chunks[ci].url,
                "Positive Prototype": pos,
                "Negative Prototype": neg,
                "Prototype Contrast": contrast,
                "Incident Relevance": rel,
                "Retrieval Joint": joint,
            }
            candidates.append(item)
            candidate_map[(prop, ci)] = item

    if not candidates:
        return []

    nli = _status_nli_for_candidates(candidates, model_id=model_id)
    results = []

    for prop in PROPERTIES:
        prop_candidates = [c for c in candidates if c["Property"] == prop]
        evaluated = []

        for item in prop_candidates:
            scores = nli.get((prop, item["Chunk Index"]), {})
            if not scores:
                continue

            ent = {s: float(scores[s]["entailment"]) for s in scores}
            confirmed = ent.get("CONFIRMED", 0.0)
            potential = ent.get("POTENTIAL", 0.0)
            unaffected = ent.get("UNAFFECTED", 0.0)
            claimed = ent.get("CLAIMED", 0.0)

            # Model status competition.
            status_order = sorted(ent.items(), key=lambda x: x[1], reverse=True)
            model_status, model_status_score = status_order[0]
            runner_up = status_order[1][1] if len(status_order) > 1 else 0.0
            status_margin = model_status_score - runner_up

            text = item["Evidence"].strip()
            has_negation = _has_pattern(text, NEGATION_PATTERNS)
            has_potential = _has_pattern(text, POTENTIAL_PATTERNS)
            has_claim = _has_pattern(text, CLAIM_PATTERNS)
            question_like = _has_pattern(text, QUESTION_PATTERNS)

            # External actuality evidence overrides generic NLI ambiguity.
            if question_like:
                final_status = "UNKNOWN"
                guard_reason = "Question/non-factual wording cannot establish an actual violation."
            elif has_negation:
                final_status = "UNAFFECTED"
                guard_reason = "Explicit negation/no-impact wording overrides a positive property inference."
            elif has_claim:
                final_status = "CLAIMED"
                guard_reason = "The evidence is explicitly framed as a claim/allegation/unverified report."
            elif has_potential:
                final_status = "POTENTIAL"
                guard_reason = "The evidence describes capability, possibility, demonstration, or hypothetical consequence."
            else:
                final_status = model_status
                guard_reason = "Status selected by four-way local NLI competition."

            # CONFIRMED has additional paper-grounded gates.
            if final_status == "CONFIRMED":
                if confirmed < params["confirmed"]:
                    final_status = "UNKNOWN"
                    guard_reason = "Confirmed NLI support is below the selected operating point."
                elif status_margin < params["status_margin"]:
                    final_status = "UNKNOWN"
                    guard_reason = "Confirmed did not beat competing evidence statuses by a sufficient margin."
                elif item["Positive Prototype"] < params["prototype_floor"]:
                    final_status = "UNKNOWN"
                    guard_reason = "Evidence is too weakly aligned with the paper-grounded positive property prototypes."
                elif item["Prototype Contrast"] < params["contrast_margin"]:
                    final_status = "UNKNOWN"
                    guard_reason = "Evidence is at least as compatible with a known hard-negative/confusable case as with a true violation."

            # Composite ranking score chooses the best evidence for THIS property.
            # Status score is diagnostic; it is not claimed to be calibrated probability.
            composite = (
                0.48 * confirmed
                + 0.24 * item["Positive Prototype"]
                + 0.16 * max(item["Prototype Contrast"], -0.25)
                + 0.12 * item["Incident Relevance"]
            )

            evaluated.append({
                **item,
                "Final Status": final_status,
                "Confirmed NLI": confirmed,
                "Potential NLI": potential,
                "Unaffected NLI": unaffected,
                "Claimed NLI": claimed,
                "Model Status": model_status,
                "Status Margin": status_margin,
                "Composite": composite,
                "Guard Reason": guard_reason,
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
                "Neutral Score": 1.0,
                "Evidence": "",
                "Evidence Source": "",
                "Evidence URL": "",
                "Decision Reason": "No incident-relevant candidate evidence was available.",
            })
            continue

        # Prefer explicit non-UNKNOWN evidence statuses; within a status use the
        # composite evidence score. This avoids a high generic NLI score on a weak
        # background sentence eclipsing a stronger incident-specific candidate.
        status_priority = {
            "CONFIRMED": 5,
            "UNAFFECTED": 4,
            "POTENTIAL": 3,
            "CLAIMED": 2,
            "UNKNOWN": 1,
        }
        best = max(
            evaluated,
            key=lambda x: (status_priority[x["Final Status"]], x["Composite"]),
        )

        # For the legacy UI columns, Status Score is the relevant status evidence;
        # Semantic Similarity is the positive property-prototype similarity.
        if best["Final Status"] == "CONFIRMED":
            status_score = best["Confirmed NLI"]
        elif best["Final Status"] == "POTENTIAL":
            status_score = best["Potential NLI"]
        elif best["Final Status"] == "UNAFFECTED":
            status_score = best["Unaffected NLI"]
        elif best["Final Status"] == "CLAIMED":
            status_score = best["Claimed NLI"]
        else:
            status_score = max(
                best["Confirmed NLI"], best["Potential NLI"],
                best["Unaffected NLI"], best["Claimed NLI"]
            )

        # Keep the existing output schema so the current Streamlit pages and
        # correlation analytics continue working without replacement.
        results.append({
            "Property": prop,
            "Parent Category": PARENT[prop],
            "Status": best["Final Status"],
            "Status Score": round(float(status_score), 4),
            "Semantic Similarity": round(float(best["Positive Prototype"]), 4),
            "Incident Relevance": round(float(best["Incident Relevance"]), 4),
            "Contradiction Score": round(float(best["Negative Prototype"]), 4),
            "Neutral Score": round(float(best["Status Margin"]), 4),
            "Evidence": best["Evidence"],
            "Evidence Source": best["Evidence Source"],
            "Evidence URL": best["Evidence URL"],
            "Decision Reason": (
                f"{best['Guard Reason']} | Positive prototype={best['Positive Prototype']:.3f}; "
                f"hard-negative={best['Negative Prototype']:.3f}; contrast={best['Prototype Contrast']:.3f}; "
                f"NLI[C={best['Confirmed NLI']:.3f}, P={best['Potential NLI']:.3f}, "
                f"U={best['Unaffected NLI']:.3f}, Cl={best['Claimed NLI']:.3f}]."
            ),
        })

    return results
