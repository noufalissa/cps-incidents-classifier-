from __future__ import annotations
import hashlib


def incident_id(year, name, country, sector) -> str:
    raw = f"{year}|{name}|{country}|{sector}".strip().lower()
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
