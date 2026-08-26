from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
import streamlit as st
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from taxonomy import PROPERTIES, PARENT, PROPERTY_SPECS


RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NLI_MODELS = {
    "Fast / smaller": "cross-encoder/nli-MiniLM2-L6-H768",
    "Stronger / larger": "cross-encoder/nli-deberta-v3-base",
}

# Shorter, direct factual hypotheses work better with generic NLI models than
# long policy-style definitions. Multiple hypotheses are used where useful.
FACT_HYPOTHESES = {
    "Safety": [
        "A cyber action caused physical damage or an unsafe physical condition.",
        "The physical system entered a hazardous or unsafe state because of the incident.",
    ],
    "Liveness": [
        "A required system action, task, or process failed to complete.",
        "Normal system progress was blocked or indefinitely prevented.",
    ],
    "Reachability": [
        "An attacker or malware reached a protected system, controller, privilege, or network zone.",
        "Unauthorized access to a protected system or component occurred.",
    ],
    "Timing Constraints": [
        "A required deadline, timing bound, synchronization requirement, or time-critical response was violated.",
    ],
    "Hybrid Dynamics": [
        "A cyber action changed or destabilized the continuous physical process or control dynamics.",
        "The incident changed physical variables such as trajectory, pressure, flow, frequency, speed, or process evolution.",
    ],
    "Confidentiality": [
        "Sensitive information was stolen, leaked, exposed, disclosed, or exfiltrated to an unauthorized party.",
    ],
    "Integrity": [
        "Data, commands, software, firmware, configuration, or control logic were modified or corrupted without authorization.",
    ],
    "Availability": [
        "A required system, service, controller, application, or communication resource became unavailable or inaccessible.",
        "Authorized users or processes lost access to a required system or service.",
    ],
    "Authenticity": [
        "A false or malicious identity, device, signal, credential, message, or update was accepted as authentic.",
        "Successful spoofing or impersonation occurred.",
    ],
    "Authorization": [
        "An entity performed an action or operation that it was not authorized to perform.",
        "Unauthorized commands, privilege use, or configuration changes occurred.",
    ],
    "Accountability": [
        "Important actions could not be reliably attributed because logs or audit records were missing, destroyed, or untrustworthy.",
    ],
    "Non-repudiation": [
        "Evidence needed to prove who performed an action was invalid, missing, or undermined.",
    ],
    "Privacy": [
        "Personal or sensitive personal information was exposed, misused, processed, or disseminated without authorization.",
    ],
    "Reliability": [
        "The system failed to perform its intended function consistently or dependably.",
    ],
    "Resilience": [
        "The system failed to maintain essential functionality during the attack or disruption.",
    ],
    "Recoverability": [
        "Restoration of normal trusted operation was substantially delayed, impaired, or failed.",
    ],
    "Compliance": [
        "The incident caused a documented violation of a legal, regulatory, certification, or mandatory compliance requirement.",
    ],
    "Explainability": [
        "An important system decision or behavior could not be adequately explained or traced to understandable reasons.",
    ],
}


NEGATION_PATTERNS = [
    r"\bno\b.{0,90}\b(impact|disruption|outage|damage|harm|compromise|breach|access|exposure|theft|loss|effect|interruption)\b",
    r"\bnot\b.{0,90}\b(affected|impacted|disrupted|compromised|breached|accessed|exposed|stolen|damaged|interrupted)\b",
    r"\b(remained|was|were)\s+(unaffected|available|operational|intact|secure)\b",
    r"\bwithout\b.{0,70}\b(impact|disruption|damage|compromise|exposure|outage|loss)\b",
    r"\bno confirmed\b",
    r"\bno evidence of\b",
]

POTENTIAL_PATTERNS = [
    r"\bcould\b", r"\bmay\b", r"\bmight\b", r"\bpotential(?:ly)?\b",
    r"\bpossible\b", r"\bcapable of\b", r"\bdesigned to\b",
    r"\bintended to\b", r"\brisk of\b", r"\bwould allow\b",
    r"\bdemonstrat(?:e|ed|ion)\b", r"\bproof[- ]of[- ]concept\b",
]

CLAIM_PATTERNS = [
    r"\bclaim(?:ed|s)?\b", r"\balleg(?:ed|edly|ation)\b",
    r"\breportedly\b", r"\bunverified\b", r"\bnot independently confirmed\b",
    r"\baccording to (?:the )?(?:attacker|hackers?|group)\b",
]


@dataclass
class Chunk:
    text: str
    source: str
    url: str


def _has_pattern(text: str, patterns) -> bool:
    t = text.lower()
    return any(re.search(p, t, flags=re.I | re.S) for p in patterns)


def split_chunks(records, max_chars: int = 850) -> List[Chunk]:
    chunks = []
    seen = set()

    for rec in records:
        text = re.sub(r"\s+", " ", str(rec.get("text", "") or "")).strip()
        if not text:
            continue

        # Conservative sentence splitting. Fetched pages can contain imperfect punctuation.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]
        if not sentences:
            sentences = [text]

        windows = []
        for i, sentence in enumerate(sentences):
            windows.append(sentence[:max_chars])

            # Also test a two-sentence context window where possible.
            if i + 1 < len(sentences):
                pair = sentence + " " + sentences[i + 1]
                if len(pair) <= max_chars:
                    windows.append(pair)

        for w in windows:
            key = w.lower()
            if key not in seen:
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


def _confirmed_threshold(model_id: str, sensitivity: str) -> float:
    # These are operating points, not claims of calibrated probabilities.
    if sensitivity == "Conservative":
        return 0.62
    if sensitivity == "Higher recall":
        return 0.38
    # Balanced
    return 0.46 if "MiniLM" in model_id else 0.42


def _summary_sentences(summary: str) -> list[str]:
    summary = re.sub(r"\s+", " ", summary or "").strip()
    if not summary:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", summary)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def _incident_relevance_scores(chunks, incident_context: str):
    """
    Transformer-semantic relevance to the specific incident.
    This is not a label decision; it prevents generic/background paragraphs
    from unrelated incidents on the same source page becoming evidence.
    """
    if not incident_context.strip():
        return np.ones(len(chunks), dtype=float)
    chunk_emb = encode_texts([c.text for c in chunks])
    ctx_emb = encode_texts([incident_context])
    return (ctx_emb @ chunk_emb.T).reshape(-1)


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
    Incident-aware local Transformer classifier.

    Decision design:
      1) Transformer embeddings retrieve evidence for each property.
      2) Fetched passages must also be relevant to THIS incident.
      3) NLI tests direct factual property hypotheses.
      4) Explicit negation / potential / claim wording overrides CONFIRMED.
      5) Questions/non-factual snippets cannot be CONFIRMED.

    Important:
      - no TF-IDF
      - semantic similarity is retrieval/relevance only
      - all final labels come from NLI + evidence-status safeguards
    """
    chunks = split_chunks(records)
    if not chunks:
        return []

    chunk_emb = encode_texts([c.text for c in chunks])
    retrieval_queries = [PROPERTY_SPECS[p]["retrieval"] for p in PROPERTIES]
    prop_emb = encode_texts(retrieval_queries)
    prop_sims = prop_emb @ chunk_emb.T

    incident_rel = _incident_relevance_scores(chunks, incident_context)

    # Prefer description/manual evidence and fetched passages relevant to the incident.
    candidate_indices_by_property = {}
    for pi, prop in enumerate(PROPERTIES):
        scored = []
        for ci, ch in enumerate(chunks):
            is_primary = ch.source in {"Verified Impact Summary", "MANUAL"}
            rel_ok = is_primary or incident_rel[ci] >= fetched_relevance_floor
            if not rel_ok:
                continue

            # Joint retrieval score: property relevance + incident relevance.
            joint = float(prop_sims[pi, ci])
            if not is_primary:
                joint = 0.68 * joint + 0.32 * float(incident_rel[ci])
            else:
                joint = 0.78 * joint + 0.22

            scored.append((joint, ci))

        scored.sort(reverse=True)
        k = min(max(1, top_k_chunks), len(scored))
        candidate_indices_by_property[prop] = [ci for _, ci in scored[:k]]

    pairs = []
    meta = []
    for pi, prop in enumerate(PROPERTIES):
        for ci in candidate_indices_by_property[prop]:
            for hypothesis in FACT_HYPOTHESES[prop]:
                pairs.append((chunks[ci].text, hypothesis))
                meta.append({
                    "Property": prop,
                    "Chunk Index": ci,
                    "Semantic": float(prop_sims[pi, ci]),
                    "Incident Relevance": float(incident_rel[ci]),
                    "Hypothesis": hypothesis,
                })

    if not pairs:
        return []

    probs = nli_probabilities(pairs, model_id=model_id)

    best = {}
    for m, p in zip(meta, probs):
        prop = m["Property"]
        item = {**m, **p}
        if prop not in best or item["entailment"] > best[prop]["entailment"]:
            best[prop] = item

    threshold = _confirmed_threshold(model_id, sensitivity)
    results = []

    for prop in PROPERTIES:
        if prop not in best:
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
                "Decision Reason": "No incident-relevant evidence candidate passed the relevance gate.",
            })
            continue

        item = best[prop]
        chunk = chunks[item["Chunk Index"]]
        evidence = chunk.text.strip()

        has_negation = _has_pattern(evidence, NEGATION_PATTERNS)
        has_potential = _has_pattern(evidence, POTENTIAL_PATTERNS)
        has_claim = _has_pattern(evidence, CLAIM_PATTERNS)
        question_like = evidence.endswith("?") or evidence.count("?") >= 1

        ent = item["entailment"]
        con = item["contradiction"]
        neu = item["neutral"]
        semantic = item["Semantic"]
        inc_rel = item["Incident Relevance"]

        status = "UNKNOWN"
        reason = "No sufficiently strong factual support."

        # Status language is an override, not a weak secondary hint.
        if has_negation:
            status = "UNAFFECTED"
            reason = "Explicit negation/no-impact wording prevents CONFIRMED."

        elif has_claim:
            status = "CLAIMED"
            reason = "The selected evidence is framed as a claim/allegation/unverified report."

        elif has_potential:
            status = "POTENTIAL"
            reason = "The selected evidence describes capability, possibility, demonstration, or hypothetical consequence."

        elif question_like:
            status = "UNKNOWN"
            reason = "A question/non-factual snippet cannot establish a confirmed violation."

        elif ent >= threshold and ent >= neu and ent >= con:
            status = "CONFIRMED"
            reason = "Incident-relevant factual evidence entails the direct property-violation hypothesis."

        results.append({
            "Property": prop,
            "Parent Category": PARENT[prop],
            "Status": status,
            "Status Score": round(ent, 4),
            "Semantic Similarity": round(semantic, 4),
            "Incident Relevance": round(inc_rel, 4),
            "Contradiction Score": round(con, 4),
            "Neutral Score": round(neu, 4),
            "Evidence": evidence,
            "Evidence Source": chunk.source,
            "Evidence URL": chunk.url,
            "Decision Reason": reason,
        })

    return results
