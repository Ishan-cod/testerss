import re
import numpy as np
from rapidfuzz import fuzz

_NAME_STOP_TOKENS = {
    "corporation", "incorporated", "limited", "private", "company",
    "enterprise", "enterprises", "llp", "llc", "the", "and", "of",
}


def _tokens(x):
    return set(t for t in x.split() if t and t not in _NAME_STOP_TOKENS)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dice(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _acronym_match(a: str, b: str) -> int:
    def acro(x):
        return "".join(w[0] for w in x.split() if w)
    return int(acro(a) == acro(b) and len(acro(a)) >= 2)


def pair_features(s1: dict, s2: dict) -> dict:
    nc1, nc2 = s1["name_clean"], s2["name_clean"]
    ac1, ac2 = s1["addr_clean"], s2["addr_clean"]

    f = {}

    # ---------- Name similarity ----------
    f["name_ratio"]         = fuzz.ratio(nc1, nc2) / 100.0
    f["name_partial"]       = fuzz.partial_ratio(nc1, nc2) / 100.0
    f["name_token_set"]     = fuzz.token_set_ratio(nc1, nc2) / 100.0
    f["name_token_sort"]    = fuzz.token_sort_ratio(nc1, nc2) / 100.0
    f["name_wratio"]        = fuzz.WRatio(nc1, nc2) / 100.0

    # ---------- Address similarity ----------
    f["addr_ratio"]         = fuzz.ratio(ac1, ac2) / 100.0
    f["addr_partial"]       = fuzz.partial_ratio(ac1, ac2) / 100.0
    f["addr_token_set"]     = fuzz.token_set_ratio(ac1, ac2) / 100.0
    f["addr_token_sort"]    = fuzz.token_sort_ratio(ac1, ac2) / 100.0

    # ---------- Token overlap ----------
    n1_toks, n2_toks = _tokens(nc1), _tokens(nc2)
    a1_toks, a2_toks = _tokens(ac1), _tokens(ac2)
    f["name_jaccard"]       = _jaccard(n1_toks, n2_toks)
    f["name_dice"]          = _dice(n1_toks, n2_toks)
    f["addr_jaccard"]       = _jaccard(a1_toks, a2_toks)
    f["addr_dice"]          = _dice(a1_toks, a2_toks)
    f["name_token_overlap"] = len(n1_toks & n2_toks)
    f["addr_token_overlap"] = len(a1_toks & a2_toks)

    # ---------- Structure / length ----------
    f["len_name_diff"]      = abs(len(nc1) - len(nc2))
    f["len_addr_diff"]      = abs(len(ac1) - len(ac2))
    f["name_len_min"]       = min(len(nc1), len(nc2))
    f["addr_len_min"]       = min(len(ac1), len(ac2))

    # ---------- Exact-match style ----------
    f["same_country"]       = int(s1["country_clean"] == s2["country_clean"])
    f["same_postal"]        = int(
        s1["postal"] != "" and s1["postal"] == s2["postal"]
    )
    f["same_house_no"]      = int(
        s1["house_no"] != "" and s1["house_no"] == s2["house_no"]
    )
    f["first_token_eq"]     = int(
        nc1.split()[0] == nc2.split()[0] if nc1 and nc2 else 0
    )
    f["acronym_match"]      = _acronym_match(nc1, nc2)

    # ---------- Substring containment ----------
    f["name_in_name"]       = int(nc1 in nc2 or nc2 in nc1) if nc1 and nc2 else 0
    f["addr_in_addr"]       = int(ac1 in ac2 or ac2 in ac1) if ac1 and ac2 else 0

    # Guard against NaN/inf
    for k, v in f.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            f[k] = 0.0
    return f


FEATURE_NAMES = None  # set on first call


def feature_names():
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        # Deterministic: call once with dummy data
        dummy = dict(name_clean="a b", addr_clean="c d",
                     country_clean="us", postal="", house_no="")
        FEATURE_NAMES = sorted(pair_features(dummy, dummy).keys())
    return FEATURE_NAMES