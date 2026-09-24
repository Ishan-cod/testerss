from collections import defaultdict
import re
from . import config


# Tokens we don't want to block on — too common
STOPWORDS = {
    "the", "and", "of", "for", "a", "an", "in", "on", "at", "to",
    "road", "street", "avenue", "lane", "drive", "floor", "plot",
    "near", "opposite", "india", "united", "states", "usa", "us",
    "corporation", "incorporated", "limited", "private", "company",
    "enterprise", "enterprises", "llp", "llc",
}


def _rare_tokens(name: str, min_len: int = 5):
    for tok in set(name.split()):
        if len(tok) >= min_len and tok not in STOPWORDS:
            yield tok


def _name_prefix(name: str, n: int = 4):
    name = name.replace(" ", "")
    return name[:n] if len(name) >= n else ""


def build_index(df):
    """
    Build inverted index over S2+S3 records.
    Returns: dict[(key_type, value)] -> set(entity_ids)
    """
    idx = defaultdict(set)
    for _, row in df.iterrows():
        eid = row["entity_id"]
        country = row["country_clean"]

        # 1) Postal code
        if row["postal"]:
            idx[("postal", row["postal"])].add(eid)

        # 2) Country + name prefix
        pref = _name_prefix(row["name_clean"])
        if pref:
            idx[("prefix", country, pref)].add(eid)

        # 3) Country + rare tokens
        for tok in _rare_tokens(row["name_clean"]):
            idx[("token", country, tok)].add(eid)

        # 4) Country + address rare tokens (helps when name is broken)
        for tok in _rare_tokens(row["addr_clean"], min_len=6):
            idx[("addr", country, tok)].add(eid)

        # 5) Country-only (last resort — bounded by cap)
        idx[("country", country)].add(eid)

    return idx


def get_candidates(s1_row, idx, max_candidates=config.MAX_CANDIDATES_PER_S1):
    """
    Union of candidate IDs from multiple blocking keys.
    Country-only bucket is used only if we're still short on candidates.
    """
    cands = set()
    country = s1_row["country_clean"]

    # Strong keys first
    if s1_row["postal"]:
        cands.update(idx.get(("postal", s1_row["postal"]), set()))

    pref = _name_prefix(s1_row["name_clean"])
    if pref:
        cands.update(idx.get(("prefix", country, pref), set()))

    for tok in _rare_tokens(s1_row["name_clean"]):
        cands.update(idx.get(("token", country, tok), set()))

    for tok in _rare_tokens(s1_row["addr_clean"], min_len=6):
        cands.update(idx.get(("addr", country, tok), set()))

    # Fallback: if nothing found, use country bucket (still bounded)
    if not cands:
        cands.update(idx.get(("country", country), set()))

    # Cap
    if len(cands) > max_candidates:
        cands = set(list(cands)[:max_candidates])

    return cands