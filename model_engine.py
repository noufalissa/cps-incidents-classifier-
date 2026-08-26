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


# ---------------------------------------------------------------------------
# CE-18 v4.2: paper-specific necessary evidence gates
# ---------------------------------------------------------------------------
#
# These patterns DO NOT replace the Transformer. They encode necessary,
# paper-derived evidence conditions for the 18 properties. The Transformer
# still retrieves and contrasts positive vs hard-negative prototypes.
#
# A label cannot become CONFIRMED merely because a generic NLI model thinks a
# sentence "sounds like" Resilience, Compliance, Authenticity, etc.
# ---------------------------------------------------------------------------

PROPERTY_EVIDENCE_GATES = {
    "Safety": [
        r"\bphysical damage\b",
        r"\b(?:unsafe|hazardous|dangerous)\s+(?:state|condition|operation|behavio(?:u)?r|switching|process)\b",
        r"\b(?:damage|damaged|destroyed)\b.{0,70}\b(?:centrifuge|equipment|machinery|plant|device|motor|pump|valve|turbine|physical)\b",
        r"\b(?:injur(?:y|ies)|fatalit(?:y|ies)|death|deaths|collision|explosion|fire)\b",
    ],
    "Liveness": [
        r"\b(?:failed|unable|could not|did not)\b.{0,70}\b(?:complete|finish|progress|execute|terminate)\b",
        r"\bindefinit(?:e|ely)\b.{0,70}\b(?:blocked|prevented|delayed|stalled)\b",
        r"\b(?:required|critical)\b.{0,60}\b(?:task|command|process|shutdown|operation)\b.{0,80}\b(?:never|failed|unable|could not)\b",
    ],
    "Reachability": [
        r"\b(?:gained|obtained|achieved|established)\b.{0,30}\b(?:access|privilege|administrator|root)\b",
        r"\bunauthorized access\b.{0,90}\b(?:controller|plc|scada|control|operational|industrial|protected|network)\b",
        r"\b(?:propagat(?:ed|ion)|spread|moved)\b.{0,90}\b(?:into|across|through)\b.{0,90}\b(?:control environment|operational network|industrial network|segmented network|plc|controller|scada)\b",
        r"\b(?:command|commands|malware)\b.{0,80}\b(?:reached|entered|accessed)\b.{0,80}\b(?:controller|plc|control system|operational environment)\b",
    ],
    "Timing Constraints": [
        r"\bmissed\b.{0,30}\bdeadline\b",
        r"\b(?:timing|temporal)\b.{0,50}\b(?:violation|constraint|deadline|bound)\b",
        r"\b(?:excessive|increased)\b.{0,30}\blatency\b",
        r"\bsynchroni[sz]ation\b.{0,40}\b(?:failure|failed|loss|violation)\b",
        r"\bstale\b.{0,30}\b(?:sensor|measurement|data)\b",
        r"\bdelayed\b.{0,40}\b(?:emergency|braking|shutdown|control response|time-critical)\b",
    ],
    "Hybrid Dynamics": [
        r"\b(?:altered|changed|manipulated|destabili[sz]ed|modified)\b.{0,80}\b(?:frequency|speed|pressure|flow|trajectory|rotation|rpm|temperature|voltage|physical process|control loop|process dynamics)\b",
        r"\b(?:frequency|speed|pressure|flow|trajectory|rotation|rpm|temperature|voltage)\b.{0,80}\b(?:altered|changed|manipulated|destabili[sz]ed|oscillat|varied)\b",
        r"\bcontinuous\b.{0,60}\b(?:physical|process|dynamics|evolution)\b.{0,60}\b(?:altered|changed|destabili[sz]ed|manipulated)\b",
    ],
    "Confidentiality": [
        r"\b(?:stole|stolen|theft|exfiltrat(?:ed|ion)|leak(?:ed|age)?|expos(?:ed|ure)|disclos(?:ed|ure))\b.{0,100}\b(?:data|files|information|records|credentials|documents|secrets|configuration)\b",
        r"\b(?:data|files|information|records|credentials|documents|secrets)\b.{0,100}\b(?:stole|stolen|exfiltrat(?:ed|ion)|leak(?:ed|age)?|expos(?:ed|ure)|disclos(?:ed|ure))\b",
    ],
    "Integrity": [
        r"\b(?:manipulat(?:ed|ion)|modif(?:ied|ication)|alter(?:ed|ation)|falsif(?:ied|ication)|corrupt(?:ed|ion)|tamper(?:ed|ing))\b.{0,100}\b(?:logic|plc|firmware|software|configuration|parameter|command|sensor|measurement|data|value|code|control)\b",
        r"\b(?:logic|plc|firmware|software|configuration|parameter|command|sensor|measurement|data|value|code|control)\b.{0,100}\b(?:manipulat(?:ed|ion)|modif(?:ied|ication)|alter(?:ed|ation)|falsif(?:ied|ication)|corrupt(?:ed|ion)|tamper(?:ed|ing))\b",
        r"\bfalse[- ]data injection\b",
    ],
    "Availability": [
        r"\b(?:outage|unavailable|inaccessible|offline|lockout|service disruption|service interruption|loss of access|lost access)\b",
        r"\b(?:shut down|shutdown|disabled|disrupted|interrupted)\b.{0,80}\b(?:service|system|network|controller|operations|production|power|communication)\b",
        r"\b(?:service|system|network|controller|operations|production|power|communication)\b.{0,80}\b(?:unavailable|offline|inaccessible|disrupted|interrupted|shut down)\b",
    ],
    "Authenticity": [
        r"\b(?:spoof(?:ed|ing)|impersonat(?:ed|ion)|masquerad(?:ed|ing))\b",
        r"\bforged\b.{0,50}\b(?:credential|identity|signal|message|update|certificate|signature)\b",
        r"\b(?:fake|malicious|counterfeit)\b.{0,50}\b(?:device|component|signal|message|update)\b.{0,60}\b(?:accepted|trusted|recognized as legitimate)\b",
    ],
    "Authorization": [
        r"\bunauthorized\b.{0,70}\b(?:command|operation|action|change|modification|configuration|execution|maintenance|control)\b",
        r"\bprivilege escalation\b",
        r"\b(?:performed|executed|issued)\b.{0,70}\b(?:without permission|without authorization|outside .* privileges)\b",
    ],
    "Accountability": [
        r"\b(?:logs?|audit (?:trail|records?)|event records?)\b.{0,80}\b(?:missing|destroyed|deleted|corrupted|tampered|untrustworthy|unavailable)\b",
        r"\b(?:could not|unable to)\b.{0,80}\b(?:attribute|reconstruct|determine responsibility|identify who)\b",
    ],
    "Non-repudiation": [
        r"\b(?:signature|signed record|authenticated record|cryptographic proof|audit evidence)\b.{0,80}\b(?:invalid|missing|destroyed|unavailable|lost|forged)\b",
        r"\b(?:deny|repudiate)\b.{0,70}\b(?:action|command|transaction)\b.{0,70}\b(?:proof|evidence|signature|record)\b",
    ],
    "Privacy": [
        r"\b(?:pii|personally identifiable|personal data|personal information|medical records?|patient data|location data|location traces|behavioral data|identity data)\b.{0,100}\b(?:expos(?:ed|ure)|leak(?:ed|age)?|stolen|disclos(?:ed|ure)|misus(?:ed|e)|shared|processed)\b",
        r"\b(?:expos(?:ed|ure)|leak(?:ed|age)?|stolen|disclos(?:ed|ure)|misus(?:ed|e))\b.{0,100}\b(?:pii|personal data|personal information|medical records?|patient data|location data|behavioral data)\b",
    ],
    "Reliability": [
        r"\b(?:unstable|unreliable|unpredictable|intermittent|repeated malfunction|repeated failure|inconsistent)\b.{0,80}\b(?:operation|behavior|behaviour|performance|system|sensor|communication|production|navigation)\b",
        r"\b(?:system|sensor|communication|production|navigation)\b.{0,80}\b(?:unstable|unreliable|unpredictable|intermittent|repeatedly failed|malfunctioned)\b",
    ],
    "Resilience": [
        r"\b(?:essential|core|critical)\b.{0,60}\b(?:functionality|service|operation|functions)\b.{0,80}\b(?:failed|lost|could not be maintained|not maintained|stopped)\b",
        r"\b(?:failed|unable|could not)\b.{0,80}\bmaintain\b.{0,60}\b(?:essential|core|critical)\b.{0,50}\b(?:functionality|service|operation)\b",
        r"\bgraceful degradation\b.{0,60}\b(?:failed|not possible|could not)\b",
    ],
    "Recoverability": [
        r"\b(?:recovery|restoration|restore|restored|rebuild|reconstruction)\b.{0,80}\b(?:failed|delayed|prolonged|difficult|took .* days|took .* weeks|unable|could not)\b",
        r"\b(?:days|weeks|months|hours)\b.{0,30}\b(?:to restore|to recover|before .* resumed)\b",
        r"\b(?:could not|unable to)\b.{0,60}\b(?:restore|recover|rebuild|resume normal operation)\b",
    ],
    "Compliance": [
        r"\b(?:violat(?:ed|ion)|non[- ]compliance|breach)\b.{0,100}\b(?:gdpr|nis2|iec\s*62443|iso\s*26262|iec\s*61508|iec\s*62304|regulation|regulatory|law|legislation|standard|certification|mandatory requirement)\b",
        r"\b(?:gdpr|nis2|iec\s*62443|iso\s*26262|iec\s*61508|iec\s*62304|regulation|regulatory|law|legislation|standard|certification)\b.{0,100}\b(?:violat(?:ed|ion)|non[- ]compliance|breach|fine|sanction)\b",
        r"\bregulator\b.{0,100}\b(?:found|determined|confirmed|fined|sanctioned)\b",
    ],
    "Explainability": [
        r"\b(?:could not|unable to|failed to)\b.{0,80}\b(?:explain|justify|reconstruct)\b.{0,80}\b(?:decision|reasoning|rationale|automated|ai|system)\b",
        r"\b(?:opaque|unexplainable|uninterpretable)\b.{0,80}\b(?:decision|model|reasoning|system)\b",
    ],
}


def _property_gate(prop: str, text: str) -> bool:
    t = text or ""
    return any(re.search(p, t, flags=re.I | re.S) for p in PROPERTY_EVIDENCE_GATES[prop])


def _prototype_nli_for_candidates(candidates, model_id: str):
    """
    Contrastive entailment:
      evidence -> positive property prototypes
      evidence -> hard-negative/confusable prototypes
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
    stronger = "deberta" in model_id.lower()

    if sensitivity == "Conservative":
        return {
            "positive_nli": 0.58 if not stronger else 0.54,
            "nli_margin": 0.14,
            "semantic_floor": 0.28,
            "semantic_margin": 0.025,
            "fetched_status_relevance": 0.56,
        }

    if sensitivity == "Higher recall":
        return {
            "positive_nli": 0.32 if not stronger else 0.30,
            "nli_margin": 0.05,
            "semantic_floor": 0.20,
            "semantic_margin": -0.005,
            "fetched_status_relevance": 0.44,
        }

    # Balanced
    return {
        "positive_nli": 0.43 if not stronger else 0.40,
        "nli_margin": 0.08,
        "semantic_floor": 0.23,
        "semantic_margin": 0.010,
        "fetched_status_relevance": 0.50,
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
    CE-18 v4.2: Paper-Constrained Contrastive Entailment.

    Key scientific constraint:
      a generic Transformer score alone can NEVER confirm a property.
      The evidence must satisfy a necessary, paper-derived evidence condition
      for that exact property.

    This prevents generic phrases such as "architectural CPS security failure"
    from becoming Resilience or Compliance, while direct evidence such as
    "manipulated ICS logic" can establish Integrity.

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

            # Loose retrieval floor. Final status has a stricter incident gate.
            if not is_primary and rel < fetched_relevance_floor:
                continue

            pos_sem = float(positive_scores[pi, ci])
            neg_sem = float(negative_scores[pi, ci])
            sem_margin = pos_sem - neg_sem

            direct_gate = _property_gate(prop, ch.text)

            joint = (
                0.44 * pos_sem
                + 0.18 * max(sem_margin, -0.20)
                + 0.24 * (1.0 if is_primary else rel)
                + 0.14 * (1.0 if direct_gate else 0.0)
            )
            ranked.append((joint, ci, pos_sem, neg_sem, sem_margin, rel, direct_gate))

        ranked.sort(reverse=True)

        # Keep a few candidates, but always include any direct-gate evidence so
        # explicit wording is not accidentally pushed out by generic semantic text.
        selected = ranked[:max(1, top_k_chunks)]
        direct_extra = [x for x in ranked if x[6] and x not in selected]
        selected += direct_extra[:2]

        for joint, ci, pos_sem, neg_sem, sem_margin, rel, direct_gate in selected:
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
                "Direct Gate": direct_gate,
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
            is_primary = item["Evidence Source"] in {"Verified Impact Summary", "MANUAL"}

            has_negation = _has_pattern(evidence, NEGATION_PATTERNS)
            has_potential = _has_pattern(evidence, POTENTIAL_PATTERNS)
            has_claim = _has_pattern(evidence, CLAIM_PATTERNS)
            question_like = _has_pattern(evidence, QUESTION_PATTERNS)

            direct_gate = bool(item["Direct Gate"])

            incident_ok = (
                is_primary
                or item["Incident Relevance"] >= params["fetched_status_relevance"]
            )

            semantic_supported = (
                item["Positive Semantic"] >= params["semantic_floor"]
                and item["Semantic Margin"] >= params["semantic_margin"]
            )
            nli_supported = (
                pos_nli >= params["positive_nli"]
                and nli_margin >= params["nli_margin"]
            )

            # Direct paper evidence can rescue known domain-language failures of
            # generic NLI (e.g., "Manipulated ICS logic").
            direct_actual_support = (
                direct_gate
                and incident_ok
                and item["Positive Semantic"] >= max(0.20, params["semantic_floor"] - 0.04)
                and nli_margin >= -0.02
            )

            transformer_support = (
                direct_gate
                and incident_ok
                and semantic_supported
                and nli_supported
            )

            final_status = "UNKNOWN"
            reason = "No paper-specific necessary evidence condition was satisfied for this property."

            if question_like:
                final_status = "UNKNOWN"
                reason = "Question/non-factual wording cannot establish a property violation."

            elif not direct_gate:
                final_status = "UNKNOWN"
                reason = "Transformer similarity alone is insufficient: the passage lacks the paper-required evidence condition for this property."

            elif not incident_ok:
                final_status = "UNKNOWN"
                reason = "The fetched passage is property-relevant but not sufficiently tied to the selected incident; background incidents are excluded."

            elif has_potential:
                # Potential is allowed only when the property itself has direct evidence.
                final_status = "POTENTIAL"
                reason = "The passage directly concerns this property but explicitly describes capability, possibility, demonstration, or hypothetical consequence."

            elif has_claim:
                final_status = "CLAIMED"
                reason = "The passage directly concerns this property but explicitly frames the violation as an allegation/claim rather than established fact."

            elif has_negation:
                final_status = "UNAFFECTED"
                reason = "The passage directly concerns this property and explicitly states that the corresponding consequence did not occur."

            elif transformer_support or direct_actual_support:
                final_status = "CONFIRMED"
                if transformer_support:
                    reason = "Paper-specific direct evidence plus positive-vs-hard-negative Transformer support establish an actual violation."
                else:
                    reason = "Explicit paper-specific incident evidence establishes the violation; the direct-evidence gate rescues a generic NLI under-score."

            status_rank = {
                "CONFIRMED": 5,
                "POTENTIAL": 4,
                "CLAIMED": 3,
                "UNAFFECTED": 2,
                "UNKNOWN": 1,
            }[final_status]

            evidence_strength = (
                0.33 * pos_nli
                + 0.18 * max(nli_margin, -0.5)
                + 0.20 * item["Positive Semantic"]
                + 0.10 * max(item["Semantic Margin"], -0.5)
                + 0.09 * item["Incident Relevance"]
                + 0.10 * (1.0 if direct_gate else 0.0)
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
                "Decision Reason": "CE-18 v4.2 | No incident-relevant evidence candidate.",
            })
            continue

        best = max(evaluated, key=lambda x: (x["Rank"], x["Evidence Strength"]))

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
                f"CE-18 v4.2 | {best['Reason']} "
                f"| direct_gate={best['Direct Gate']}; "
                f"positive_sem={best['Positive Semantic']:.3f}; "
                f"hardneg_sem={best['Negative Semantic']:.3f}; "
                f"sem_margin={best['Semantic Margin']:.3f}; "
                f"positive_nli={best['Positive NLI']:.3f}; "
                f"hardneg_nli={best['Negative NLI']:.3f}; "
                f"nli_margin={best['NLI Margin']:.3f}; "
                f"incident_rel={best['Incident Relevance']:.3f}; "
                f"positive_prototype=\"{best['Best Positive Prototype']}\"; "
                f"hard_negative=\"{best['Best Negative Prototype']}\"."
            ),
        })

    return results
