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

# These patterns do NOT assign a property. They interpret the status of
# a PROPERTY-SPECIFIC local evidence span only.
NEGATION_PATTERNS = [
    r"\bno\b.{0,90}\b(impact|disruption|outage|damage|harm|compromise|breach|access|exposure|theft|loss|effect|interruption|modification|change)\b",
    r"\bnot\b.{0,90}\b(affected|impacted|disrupted|compromised|breached|accessed|exposed|stolen|damaged|interrupted|modified|altered)\b",
    r"\b(remained|was|were|continued)\s+(unaffected|available|operational|intact|secure|normal)\b",
    r"\bwithout\b.{0,80}\b(impact|disruption|damage|compromise|exposure|outage|loss|modification)\b",
    r"\bno confirmed\b",
    r"\bno evidence of\b",
    r"\bmade no attempts? to\b",
]

POTENTIAL_PATTERNS = [
    r"\bcould\b", r"\bmay\b", r"\bmight\b", r"\bpotential(?:ly)?\b",
    r"\bpossible\b", r"\bcapable of\b", r"\brisk of\b",
    r"\bwould allow\b", r"\bdemonstrat(?:e|ed|ion)\b",
    r"\bproof[- ]of[- ]concept\b", r"\btheoretical(?:ly)?\b",
    r"\bcan be used to\b",
]

CLAIM_PATTERNS = [
    r"\bclaim(?:ed|s)?\b", r"\balleg(?:ed|edly|ation)\b",
    r"\breportedly\b", r"\bunverified\b", r"\bnot independently confirmed\b",
    r"\baccording to (?:the )?(?:attacker|attackers|hacker|hackers|group)\b",
    r"\bsaid it (?:had|has) stolen\b", r"\btook responsibility\b",
    r"\bappears? to\b", r"\bbelieved to\b", r"\bsuspected\b",
]

QUESTION_PATTERNS = [
    r"\?$", r"\bwas .+ attacked\?", r"\bcould .+\?", r"\bmay .+\?",
]


@dataclass
class Chunk:
    text: str
    source: str
    url: str
    anchor_context: str = ""


def _has_pattern(text: str, patterns) -> bool:
    t = (text or "").lower().strip()
    return any(re.search(p, t, flags=re.I | re.S) for p in patterns)


def _atomic_units(text: str, max_chars: int = 760) -> list[str]:
    """
    Precision-first evidence units.

    v4.3 used sentence + two-sentence windows. That allowed a word such as
    'could' in one sentence to downgrade an actual violation in another, and
    allowed background incidents on the same page to leak into the label.

    v4.4 keeps sentences and useful clauses as independent evidence units.
    """
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []

    # Normal sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])",
                         clean)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 18]
    if not sentences:
        sentences = [clean]

    units = []
    seen = set()

    for sentence in sentences:
        sentence = sentence[:max_chars]

        # Always retain the full sentence.
        candidates = [sentence]

        # Also create smaller clauses. This is critical for sentences such as:
        # "After incident A, company B was also knocked offline."
        clause_parts = re.split(
            r"\s*(?:;|—|–|\|\s+|\s+-\s+)\s*"
            r"|,\s+(?=(?:but|however|while|whereas|although|though|the|a|an|this|these|those|based|according)\b)"
            r"|\s+(?=(?:however|whereas|although|though)\b)",
            sentence,
            flags=re.I,
        )

        for part in clause_parts:
            part = part.strip(" ,;:-")
            if len(part) >= 28:
                candidates.append(part[:max_chars])

        # Prefer smaller units later, but keep deduplicated.
        for c in sorted(candidates, key=len):
            key = re.sub(r"\W+", " ", c.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                units.append(c)

    return units


def split_chunks(records, max_chars: int = 760) -> List[Chunk]:
    """
    CE-18 v4.5 evidence segmentation.

    `text` is the exact sentence/clause used for property and status decisions.
    `anchor_context` contains only the immediately adjacent sentence context,
    used for incident/coreference anchoring and never for the property status.
    """
    chunks = []
    seen = set()

    for rec in records:
        source = str(rec.get("source", ""))
        url = str(rec.get("url", ""))
        clean = re.sub(r"\s+", " ", str(rec.get("text", "") or "")).strip()
        if not clean:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])",
            clean,
        )
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 18]
        if not sentences:
            sentences = [clean]

        for i, sentence in enumerate(sentences):
            sentence = sentence[:max_chars]

            prev_s = sentences[i - 1][-max_chars:] if i > 0 else ""
            next_s = sentences[i + 1][:max_chars] if i + 1 < len(sentences) else ""
            context = " ".join(x for x in [prev_s, sentence, next_s] if x).strip()

            units = [sentence]
            clause_parts = re.split(
                r"\s*(?:;|—|–|\|\s+|\s+-\s+)\s*"
                r"|,\s+(?=(?:but|however|while|whereas|although|though|the|a|an|this|these|those|based|according)\b)"
                r"|\s+(?=(?:however|whereas|although|though)\b)",
                sentence,
                flags=re.I,
            )

            for part in clause_parts:
                part = part.strip(" ,;:-")
                if len(part) >= 28:
                    units.append(part[:max_chars])

            for unit in sorted(units, key=len):
                key = (source, re.sub(r"\W+", " ", unit.lower()).strip())
                if not key[1] or key in seen:
                    continue
                seen.add(key)
                chunks.append(
                    Chunk(
                        text=unit,
                        source=source,
                        url=url,
                        anchor_context=context,
                    )
                )

    return chunks


def _incident_name_from_context(incident_context: str) -> str:
    return (incident_context or "").split(" | ", 1)[0].strip()


ANCHOR_STOPWORDS = {
    "attack", "attacks", "incident", "campaign", "breach", "intrusion",
    "malware", "virus", "worm", "targeting", "targeted", "zero", "day",
    "system", "systems", "control", "cyber", "security", "energy", "power",
    "lab", "laboratory", "company", "group", "project", "operation",
    "operations", "here", "have", "there", "this", "that", "with",
    "from", "into", "against", "related", "reported", "alleged",
    "hack", "hacking",
}

GENERIC_ANCHOR_PHRASES = {
    "here you have",
    "smart grid worm",
    "energy attack",
    "cyber attack",
    "security incident",
}


def _context_parts(incident_context: str) -> list[str]:
    return [p.strip() for p in (incident_context or "").split(" | ")]


def _anchor_terms(incident_context: str) -> list[str]:
    """
    Precision-first anchors from incident name and attacker/group only.
    Generic impact/sector words are intentionally excluded.
    """
    parts = _context_parts(incident_context)
    name = parts[0] if parts else ""
    attacker = parts[3] if len(parts) > 3 else ""

    terms = []
    for raw in [name, attacker]:
        raw = re.sub(r"[\(\)\[\]]", " ", raw)
        for piece in [p.strip() for p in re.split(r"[/|,:]", raw) if p.strip()]:
            normalized = re.sub(r"[^A-Za-z0-9]+", " ", piece).strip()
            low_phrase = normalized.lower()

            useful_tokens = [
                tok.lower()
                for tok in normalized.split()
                if len(tok) >= 4 and tok.lower() not in ANCHOR_STOPWORDS
            ]

            if (
                len(normalized) >= 5
                and low_phrase not in GENERIC_ANCHOR_PHRASES
                and useful_tokens
            ):
                terms.append(low_phrase)

            terms.extend(useful_tokens)

    return sorted(set(terms), key=lambda x: (-len(x), x))


def _anchor_hit(text: str, incident_context: str) -> bool:
    t = (text or "").lower()
    return any(term in t for term in _anchor_terms(incident_context))


def _adjacent_anchor_hit(chunk: Chunk, incident_context: str) -> bool:
    return _anchor_hit(chunk.anchor_context, incident_context)


def _background_incident_marker(text: str) -> bool:
    """
    Generic cross-incident transition markers. This does not reject evidence by
    itself; it raises the incident-anchoring requirement.
    """
    t = (text or "").lower()
    patterns = [
        r"\banother\b.{0,80}\b(?:attack|incident|company|operator|facility)\b",
        r"\bseparately\b",
        r"\bmeanwhile\b",
        r"\belsewhere\b",
        r"\balso\b.{0,80}\b(?:attacked|breached|infected|compromised|hit|knocked offline|shut down)\b",
        r"\bless than .{0,60}\bafter\b.{0,80}\bincident\b",
        r"\bfollowing\b.{0,80}\bincident\b",
        r"\bin a different incident\b",
    ]
    return any(re.search(p, t, flags=re.I | re.S) for p in patterns)


def _local_property_span(prop: str, text: str) -> str:
    """
    Return the smallest clause/sentence that actually satisfies the property's
    direct evidence gate. Status words are evaluated on THIS span only.
    """
    units = _atomic_units(text)
    matching = [u for u in units if _property_gate(prop, u)]
    if not matching:
        return text.strip()
    return min(matching, key=len).strip()


def _summary_has_positive_cps_impact(summary: str) -> bool:
    s = (summary or "").lower()

    negative = [
        r"\bno\b.{0,70}\b(?:ot|operational|production|physical|control|scada|ics)\b.{0,50}\b(?:impact|disruption|effect)\b",
        r"\b(?:ot|operational|production|physical|control|scada|ics)\b.{0,70}\b(?:impact|disruption)\b.{0,40}\b(?:not|unverified|unclear|not clearly verified|not confirmed)\b",
        r"\bno direct\b.{0,70}\b(?:ot|production|operational)\b.{0,50}\bimpact\b",
        r"\bnot clearly verified\b",
        r"\bnot confirmed\b",
    ]
    if any(re.search(p, s, flags=re.I | re.S) for p in negative):
        return False

    positive = [
        r"\b(?:production|plant|grid|power delivery|electricity|scada|ics|control system|controller|plc|pipeline|process|operations?)\b.{0,100}\b(?:disrupted|interrupted|unavailable|offline|shut down|stopped|delayed|affected)\b",
        r"\b(?:disrupted|interrupted|unavailable|offline|shut down|stopped|delayed|affected)\b.{0,100}\b(?:production|plant|grid|power delivery|electricity|scada|ics|control system|controller|plc|pipeline|process|operations?)\b",
    ]
    return any(re.search(p, s, flags=re.I | re.S) for p in positive)


def _availability_operational_support(text: str, incident_summary: str) -> bool:
    """
    Availability is CPS-operational availability, not website/email/office-IT
    downtime by itself.
    """
    e = (text or "").lower()

    cps_terms = [
        "scada", "ics", "industrial control", "control system", "controller",
        "plc", "turbine", "plant", "production", "power delivery", "electricity",
        "grid", "substation", "pipeline", "process", "restart", "operational",
        "manufacturing", "factory", "water treatment", "pump", "valve",
        "rail", "train", "traffic", "vehicle", "flight", "airport operations",
        "clinical", "patient care", "medical device", "hospital operations",
        "emergency service",
    ]
    it_only_terms = [
        "website", "web site", "email", "sharepoint", "voicemail",
        "wireless lan", "office computers", "public-facing web",
        "public web server", "internet access", "corporate website",
        "internal network services", "office network", "business website",
    ]

    evidence_has_cps = any(term in e for term in cps_terms)
    evidence_is_it_only = any(term in e for term in it_only_terms) and not evidence_has_cps

    if evidence_is_it_only:
        return False

    return evidence_has_cps or _summary_has_positive_cps_impact(incident_summary)


def _availability_causality_support(text: str) -> bool:
    """
    Reject scheduled/planned/pre-existing outages unless the local evidence
    explicitly attributes additional downtime or delayed restart to the attack.
    """
    t = (text or "").lower()

    scheduled = re.search(
        r"\b(?:scheduled|planned|maintenance|equipment upgrade|pre[- ]existing)\b"
        r".{0,80}\b(?:outage|shutdown|downtime)\b",
        t,
        flags=re.I | re.S,
    )

    causal = (
        re.search(
            r"\b(?:infection|malware|attack|incident|ransomware|intrusion|worm|virus)\b"
            r".{0,100}\b(?:caused|resulted in|led to|forced|triggered|extended|delayed)\b"
            r".{0,100}\b(?:downtime|outage|restart|shutdown|unavailability|interruption|lockout)\b",
            t,
            flags=re.I | re.S,
        )
        or re.search(
            r"\b(?:downtime|outage|restart|shutdown|unavailability|interruption|lockout)\b"
            r".{0,100}\b(?:caused by|resulting from|due to|because of)\b"
            r".{0,80}\b(?:infection|malware|attack|incident|ransomware|intrusion|worm|virus)\b",
            t,
            flags=re.I | re.S,
        )
    )

    if scheduled and not causal:
        return False
    return True


def _incident_is_disputed(incident_context: str, incident_summary: str) -> bool:
    t = f"{incident_context} {incident_summary or ''}".lower()
    return any(re.search(p, t, flags=re.I | re.S) for p in [
        r"\balleg(?:ed|edly|ation)\b",
        r"\bunverified\b",
        r"\bdisputed\b",
        r"\bnot independently confirmed\b",
        r"\bnot confirmed\b",
        r"\bapparent(?:ly)?\b",
        r"\bsuspected\b",
        r"\bclaim(?:ed|s)?\b",
    ])


def _independent_confirmation_signal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, flags=re.I | re.S) for p in [
        r"\b(?:forensic|technical) analysis\b.{0,80}\b(?:found|confirmed|showed|identified|demonstrated)\b",
        r"\b(?:investigation|investigators)\b.{0,80}\b(?:found|confirmed|determined|identified)\b",
        r"\b(?:company|operator|agency|government|cisa|fbi|cert)\b.{0,80}\bconfirmed\b",
        r"\bconfirmed by\b",
        r"\bverified by\b",
    ])


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
# CE-18 v4.5: paper-specific necessary evidence gates
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
        r"\b(?:gained|obtained|achieved|established)\b.{0,35}\b(?:access|privilege|administrator|root)\b",
        r"\bunauthorized access\b.{0,100}\b(?:controller|plc|scada|control|operational|industrial|protected|network)\b",
        r"\b(?:propagat(?:ed|ion)|spread|moved)\b.{0,100}\b(?:into|across|through)\b.{0,100}\b(?:control environment|operational network|industrial network|segmented network|plc|controller|scada)\b",
        r"\b(?:command|commands|malware)\b.{0,80}\b(?:reached|entered|accessed)\b.{0,80}\b(?:controller|plc|control system|operational environment)\b",
        r"\bcompromised\b.{0,80}\bremote access\b.{0,120}\b(?:ics|scada|control|operational|pipeline)\b",
        r"\b(?:credentials?|accounts?)\b.{0,80}\b(?:used|employed)\b.{0,60}\bto reach\b.{0,80}\b(?:protected|moderate impact|control|operational|industrial)\b",
        r"\b(?:seiz(?:e|es|ed)|took|take|taking|gain(?:ed)?)\b.{0,40}\bcontrol\b.{0,80}\b(?:controller|plc|cpu|control system|industrial system|engineering station|workstation)\b",
        r"\b(?:controller|plc|cpu|control system|industrial system|engineering station|workstation)\b.{0,80}\b(?:seiz(?:ed|ure)|taken over|compromised|under attacker control)\b",
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
        r"\b(?:stole|stolen|steal|theft|exfiltrat(?:ed|ion)|leak(?:ed|age)?|expos(?:ed|ure)|disclos(?:ed|ure))\b.{0,120}\b(?:data|files|information|records|credentials|passwords?|documents|secrets|configuration|intellectual property|designs?|formulas?|manuals?|process details?)\b",
        r"\b(?:data|files|information|records|credentials|passwords?|documents|secrets|intellectual property|designs?|formulas?|manufacturing process(?:es)?|process details?)\b.{0,120}\b(?:stole|stolen|steal|exfiltrat(?:ed|ion)|leak(?:ed|age)?|expos(?:ed|ure)|disclos(?:ed|ure)|collected|obtained)\b",
        r"\bdata breach\b",
        r"\bindustrial espionage\b.{0,120}\b(?:collecting|collected|steal|stolen|obtained)\b.{0,80}\b(?:intellectual property|proprietary|design|formula|process)\b",
        r"\bcollect(?:ed|ing)\b.{0,100}\b(?:intellectual property|proprietary information|ics-related information|credentials|passwords?|system manuals?|design documents?)\b",
        r"\btransfer(?:red|ring)?\b.{0,80}\bstolen data\b",
        r"\bstolen data\b.{0,80}\b(?:command-and-control|c2|server)\b",
        r"\bgrabbed\b.{0,50}\b(?:credentials|passwords?|files|documents)\b",
    ],
    "Integrity": [
        r"\b(?:manipulat(?:ed|ion)|modif(?:ied|ication)|alter(?:ed|ation)|falsif(?:ied|ication)|corrupt(?:ed|ion)|tamper(?:ed|ing))\b.{0,110}\b(?:logic|plc|firmware|software|configuration|parameter|command|sensor|measurement|data|value|code|control)\b",
        r"\b(?:logic|plc|firmware|software|configuration|parameter|command|sensor|measurement|data|value|code|control)\b.{0,110}\b(?:manipulat(?:ed|ion)|modif(?:ied|ication)|alter(?:ed|ation)|falsif(?:ied|ication)|corrupt(?:ed|ion)|tamper(?:ed|ing))\b",
        r"\bfalse[- ]data injection\b",
        r"\b(?:wiped|deleted|destroyed|erased)\b.{0,80}\b(?:data|files|records|disks?|computers?)\b",
        r"\b(?:data|files|records)\b.{0,80}\b(?:wiped|deleted|destroyed|erased)\b",
    ],
    "Availability": [
        r"\b(?:outage|unavailable|inaccessible|offline|lockout|service disruption|service interruption|loss of access|lost access)\b",
        r"\b(?:shut down|shutdown|disabled|interrupted)\b.{0,90}\b(?:service|system|network|controller|production|power delivery|communication)\b",
        r"\b(?:service|system|network|controller|production|power delivery|communication)\b.{0,90}\b(?:unavailable|offline|inaccessible|interrupted|shut down)\b",
        r"\b(?:users?|operators?|controllers?|applications?)\b.{0,80}\b(?:could not access|lost access|were locked out|unable to access)\b",
        r"\b(?:delay|delayed)\b.{0,60}\b(?:restart|resumption|production restart)\b",
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
        r"\b(?:pii|personally identifiable|personal data|personal information|medical records?|patient data|location data|location traces|behavioral data|identity data|names? and phone numbers?)\b.{0,110}\b(?:expos(?:ed|ure)|leak(?:ed|age)?|stolen|disclos(?:ed|ure)|misus(?:ed|e)|shared|processed|siphon(?:ed|ing))\b",
        r"\b(?:expos(?:ed|ure)|leak(?:ed|age)?|stolen|disclos(?:ed|ure)|misus(?:ed|e)|siphon(?:ed|ing))\b.{0,110}\b(?:pii|personal data|personal information|medical records?|patient data|location data|behavioral data|names? and phone numbers?)\b",
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
        r"\b(?:recovery|restoration|restore|restored|rebuild|reconstruction|restart)\b.{0,90}\b(?:failed|delayed|prolonged|difficult|took .* days|took .* weeks|unable|could not)\b",
        r"\b(?:delay|delayed)\b.{0,60}\b(?:restart|restoration|recovery|resumption)\b",
        r"\b(?:days|weeks|months|hours)\b.{0,35}\b(?:to restore|to recover|before .* resumed|delay in restart)\b",
        r"\b(?:could not|unable to)\b.{0,60}\b(?:restore|recover|rebuild|resume normal operation|restart)\b",
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
    Contrastive entailment over the LOCAL property evidence span.
    """
    pairs = []
    meta = []

    for idx, item in enumerate(candidates):
        prop = item["Property"]
        for pidx, hypothesis in enumerate(POSITIVE_PROTOTYPES[prop]):
            pairs.append((item["Evidence"], hypothesis))
            meta.append((idx, "POS", pidx))
        for nidx, hypothesis in enumerate(HARD_NEGATIVE_PROTOTYPES[prop]):
            pairs.append((item["Evidence"], hypothesis))
            meta.append((idx, "NEG", nidx))

    probs = nli_probabilities(pairs, model_id=model_id)

    out = {i: {"POS": [], "NEG": []} for i in range(len(candidates))}
    for (idx, side, proto_idx), prob in zip(meta, probs):
        out[idx][side].append({
            "index": proto_idx,
            "entailment": float(prob["entailment"]),
            "contradiction": float(prob["contradiction"]),
            "neutral": float(prob["neutral"]),
        })
    return out


def _decision_params(model_id: str, sensitivity: str):
    stronger = "deberta" in model_id.lower()

    if sensitivity == "Conservative":
        return {
            "positive_nli": 0.58 if not stronger else 0.54,
            "nli_margin": 0.14,
            "semantic_floor": 0.28,
            "semantic_margin": 0.025,
            "fetched_status_relevance": 0.58,
        }

    if sensitivity == "Higher recall":
        return {
            "positive_nli": 0.32 if not stronger else 0.30,
            "nli_margin": 0.05,
            "semantic_floor": 0.20,
            "semantic_margin": -0.005,
            "fetched_status_relevance": 0.46,
        }

    # Balanced
    return {
        "positive_nli": 0.43 if not stronger else 0.40,
        "nli_margin": 0.08,
        "semantic_floor": 0.23,
        "semantic_margin": 0.010,
        "fetched_status_relevance": 0.53,
    }


def _localized_candidate_metrics(candidates, incident_context: str):
    """
    Recompute all semantic metrics AFTER localizing the property evidence.
    This is important: a full article sentence can be incident-relevant while
    the exact sub-clause proving a property is actually about another incident.
    """
    if not candidates:
        return

    local_chunks = [
        Chunk(
            text=c["Evidence"],
            source=c["Evidence Source"],
            url=c["Evidence URL"],
            anchor_context=c.get("Anchor Context", c["Evidence"]),
        )
        for c in candidates
    ]

    _, pos_scores, neg_scores = _prototype_matrices(local_chunks)
    incident_rel = _incident_relevance_scores(local_chunks, incident_context)

    prop_index = {p: i for i, p in enumerate(PROPERTIES)}
    for ci, item in enumerate(candidates):
        pi = prop_index[item["Property"]]
        item["Positive Semantic"] = float(pos_scores[pi, ci])
        item["Negative Semantic"] = float(neg_scores[pi, ci])
        item["Semantic Margin"] = item["Positive Semantic"] - item["Negative Semantic"]
        item["Incident Relevance"] = float(incident_rel[ci])
        item["Anchor Hit"] = _anchor_hit(item["Evidence"], incident_context)
        item["Adjacent Anchor Hit"] = _anchor_hit(
            item.get("Anchor Context", item["Evidence"]),
            incident_context,
        )
        item["Background Marker"] = _background_incident_marker(item["Evidence"])


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
    CE-18 v4.5: Incident-Anchored, Clause-Scoped, Paper-Constrained
    Contrastive Entailment.

    v4.5 extends the 20-incident pilot fixes with:
      1) stronger entity anchoring and adjacent-sentence coreference;
      2) rejection of background incidents on the same source page;
      3) CPS-operational + attack-causality guards for Availability;
      4) narrower potential cues;
      5) Reachability support for taking control of PLC/controller/CPU;
      6) incident-level alleged/disputed downgrade without independent confirmation.

    No TF-IDF and no paid API are used.
    """
    chunks = split_chunks(records)
    if not chunks:
        return []

    params = _decision_params(model_id, sensitivity)

    # Coarse retrieval over sentence/clause chunks.
    _, coarse_pos, coarse_neg = _prototype_matrices(chunks)
    coarse_rel = _incident_relevance_scores(chunks, incident_context)

    candidates = []

    for pi, prop in enumerate(PROPERTIES):
        ranked = []

        for ci, ch in enumerate(chunks):
            is_primary = ch.source in {"Verified Impact Summary", "MANUAL"}
            rel = float(coarse_rel[ci])

            if not is_primary and rel < fetched_relevance_floor and not _anchor_hit(ch.text, incident_context):
                continue

            direct_gate = _property_gate(prop, ch.text)
            pos_sem = float(coarse_pos[pi, ci])
            neg_sem = float(coarse_neg[pi, ci])
            sem_margin = pos_sem - neg_sem

            # Direct evidence is favored, but generic semantic similarity alone
            # remains retrieval-only.
            joint = (
                0.40 * pos_sem
                + 0.16 * max(sem_margin, -0.20)
                + 0.22 * (1.0 if is_primary else rel)
                + 0.16 * (1.0 if direct_gate else 0.0)
                + 0.06 * (1.0 if _anchor_hit(ch.text, incident_context) else 0.0)
            )

            ranked.append((joint, ci, direct_gate))

        ranked.sort(reverse=True)

        selected = ranked[:max(1, top_k_chunks)]
        # Never lose an explicit paper-gate sentence simply because generic
        # embeddings ranked a background sentence above it.
        direct_extra = [x for x in ranked if x[2] and x not in selected]
        selected += direct_extra[:3]

        for _, ci, direct_gate in selected:
            evidence = chunks[ci].text
            if direct_gate:
                evidence = _local_property_span(prop, evidence)

            candidates.append({
                "Property": prop,
                "Evidence": evidence,
                "Evidence Source": chunks[ci].source,
                "Evidence URL": chunks[ci].url,
                "Anchor Context": chunks[ci].anchor_context,
                "Direct Gate": bool(_property_gate(prop, evidence)),
            })

    if not candidates:
        return []

    # Deduplicate same property/local-span candidates.
    deduped = []
    seen = set()
    for item in candidates:
        key = (
            item["Property"],
            item["Evidence Source"],
            re.sub(r"\W+", " ", item["Evidence"].lower()).strip(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    candidates = deduped

    # Recompute metrics on the exact evidence span actually used for the label.
    _localized_candidate_metrics(candidates, incident_context)
    contrast_nli = _prototype_nli_for_candidates(candidates, model_id=model_id)

    results = []

    for prop in PROPERTIES:
        evaluated = []

        for idx, item in enumerate(candidates):
            if item["Property"] != prop:
                continue

            scores = contrast_nli[idx]
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
            anchor_hit = bool(item["Anchor Hit"])
            adjacent_anchor_hit = bool(item["Adjacent Anchor Hit"])
            background_marker = bool(item["Background Marker"])

            # Primary summaries are already incident-scoped.
            # Fetched evidence must either name/anchor the incident or be
            # strongly semantically tied to it. Background-transition language
            # requires the lexical anchor.
            high_semantic_anchor = item["Incident Relevance"] >= 0.72

            incident_ok = (
                is_primary
                or anchor_hit
                or (adjacent_anchor_hit and not background_marker)
                or (high_semantic_anchor and not background_marker)
            )

            operational_ok = True
            causality_ok = True
            if prop == "Availability":
                operational_ok = _availability_operational_support(evidence, incident_summary)
                causality_ok = _availability_causality_support(evidence)

            semantic_supported = (
                item["Positive Semantic"] >= params["semantic_floor"]
                and item["Semantic Margin"] >= params["semantic_margin"]
            )

            nli_supported = (
                pos_nli >= params["positive_nli"]
                and nli_margin >= params["nli_margin"]
            )

            # A direct paper gate can rescue a modest generic-NLI under-score,
            # but not an essentially unsupported hypothesis.
            direct_actual_support = (
                direct_gate
                and incident_ok
                and operational_ok
                and causality_ok
                and item["Positive Semantic"] >= max(0.20, params["semantic_floor"] - 0.04)
                and pos_nli >= 0.25
                and nli_margin >= 0.05
            )

            transformer_support = (
                direct_gate
                and incident_ok
                and operational_ok
                and causality_ok
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
                reason = "Transformer similarity alone is insufficient: the local passage lacks the paper-required evidence condition for this property."

            elif not incident_ok:
                final_status = "UNKNOWN"
                reason = "The exact property evidence span is not sufficiently anchored to the selected incident; likely background/cross-incident text."

            elif prop == "Availability" and not operational_ok:
                final_status = "UNKNOWN"
                reason = "Availability evidence is IT-only (e.g. website/email/office systems) without documented CPS operational impact."

            elif prop == "Availability" and not causality_ok:
                final_status = "UNKNOWN"
                reason = "The local outage/downtime evidence is scheduled, planned, or pre-existing and is not causally attributed to the cyber incident."

            elif has_potential:
                final_status = "POTENTIAL"
                reason = "This exact property evidence span explicitly describes capability, possibility, demonstration, or intended/hypothetical consequence."

            elif has_claim:
                final_status = "CLAIMED"
                reason = "This exact property evidence span explicitly frames the violation as uncertain, alleged, suspected, or inferred rather than established fact."

            elif has_negation:
                final_status = "UNAFFECTED"
                reason = "This exact property evidence span explicitly states that the corresponding consequence did not occur."

            elif transformer_support or direct_actual_support:
                final_status = "CONFIRMED"
                if transformer_support:
                    reason = "Incident-anchored local evidence satisfies the paper gate and beats the property-specific hard negative."
                else:
                    reason = "Direct paper-specific incident evidence establishes the violation; local contrastive NLI provides minimum support."

                if (
                    _incident_is_disputed(incident_context, incident_summary)
                    and not _independent_confirmation_signal(evidence)
                ):
                    final_status = "CLAIMED"
                    reason = "The property evidence is technically compatible, but the incident context is explicitly alleged/disputed and this local evidence lacks an independent confirmation signal."

            status_rank = {
                "CONFIRMED": 5,
                "POTENTIAL": 4,
                "CLAIMED": 3,
                "UNAFFECTED": 2,
                "UNKNOWN": 1,
            }[final_status]

            evidence_strength = (
                0.31 * pos_nli
                + 0.18 * max(nli_margin, -0.5)
                + 0.19 * item["Positive Semantic"]
                + 0.09 * max(item["Semantic Margin"], -0.5)
                + 0.10 * item["Incident Relevance"]
                + 0.08 * (1.0 if direct_gate else 0.0)
                + 0.05 * (1.0 if anchor_hit or is_primary else 0.0)
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
                "Operational OK": operational_ok,
                "Causality OK": causality_ok,
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
                "Decision Reason": "CE-18 v4.5 | No incident-relevant evidence candidate.",
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
                f"CE-18 v4.5 | {best['Reason']} "
                f"| direct_gate={best['Direct Gate']}; "
                f"anchor_hit={best['Anchor Hit']}; "
                f"adjacent_anchor_hit={best['Adjacent Anchor Hit']}; "
                f"background_marker={best['Background Marker']}; "
                f"operational_ok={best['Operational OK']}; "
                f"causality_ok={best['Causality OK']}; "
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
