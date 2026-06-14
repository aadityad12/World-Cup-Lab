from __future__ import annotations

import re
import unicodedata


ALIASES = {
    "u s a": "united states",
    "u s": "united states",
    "usa": "united states",
    "us": "united states",
    "usmnt": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "czechia": "czech republic",
    "turkiye": "turkey",
    "tuerkiye": "turkey",
    "ivory coast": "cote d ivoire",
    "cote d ivoire": "cote d ivoire",
    "cote divoire": "cote d ivoire",
    "côte d ivoire": "cote d ivoire",
    "côte d’ivoire": "cote d ivoire",
    "bosnia herzegovina": "bosnia and herzegovina",
    "bosnia herzergovina": "bosnia and herzegovina",
    "bosnia and herzergovina": "bosnia and herzegovina",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "dr congo": "dr congo",
    "d r congo": "dr congo",
    "congo dr": "dr congo",
    "democratic republic of congo": "dr congo",
    "curaçao": "curacao",
    "curacao": "curacao",
}


def strip_accents(value: str) -> str:
    """Return ASCII-ish text while preserving ordinary letters/numbers."""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_team_name(value: str) -> str:
    """Canonical team-name key shared by ingestion, training, and prediction.

    The previous code had three slightly different normalizers, which caused silent
    prediction fallbacks for teams with accents or alternate names, e.g. Curaçao.
    """
    value = str(value).lower().strip().replace("&", " and ")
    value = strip_accents(value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return ALIASES.get(value, value)


def candidate_team_keys(value: str) -> list[str]:
    """Keys to try when reading older artifacts generated before normalization fixes."""
    canonical = normalize_team_name(value)
    raw_lower = str(value).lower().strip()
    legacy_unicode = re.sub(r"[^\w]+", " ", raw_lower, flags=re.UNICODE)
    legacy_unicode = re.sub(r"\s+", " ", legacy_unicode).strip()
    candidates = [canonical, raw_lower, legacy_unicode]
    # Compatibility for existing checked-in artifacts.
    compatibility = {
        "curacao": ["curaçao", "cura ao"],
        "cote d ivoire": ["côte d’ivoire", "cote d ivoire"],
    }
    candidates.extend(compatibility.get(canonical, []))
    deduped: list[str] = []
    for key in candidates:
        if key and key not in deduped:
            deduped.append(key)
    return deduped
