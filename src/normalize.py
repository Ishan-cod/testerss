import re
import unicodedata
import pandas as pd

# Optional Devanagari transliteration — gracefully degrade on Python 3.14
try:
    from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
    _HAS_TRANSLIT = True
except Exception:
    _HAS_TRANSLIT = False

# ---------- Abbreviation maps ----------
NAME_ABBREV = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bco\b": "company",
    r"\bintl\b": "international",
    r"\bent\b": "enterprises",
    r"\benterprises\b": "enterprise",
    r"\bsvcs\b": "services",
    r"\bsvc\b": "service",
    r"\btech\b": "technology",
    r"\bmgmt\b": "management",
    r"\bassoc\b": "associates",
    r"\bbros\b": "brothers",
}

ADDR_ABBREV = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bhwy\b": "highway",
    r"\bno\b": "number",
    r"\bnr\b": "near",
    r"\bnear\b": "near",
    r"\bopp\b": "opposite",
    r"\bapt\b": "apartment",
    r"\bflr\b": "floor",
    r"\bpin\b": "pincode",
    r"\bzip\b": "zipcode",
}

LEGAL_SUFFIXES = [
    "corporation", "incorporated", "limited", "private", "company",
    "enterprise", "enterprises", "llp", "llc", "plc",
]


def _maybe_transliterate(text: str) -> str:
    """If text has Devanagari and library available, transliterate to Latin."""
    if not text:
        return ""
    if _HAS_TRANSLIT and re.search(r"[\u0900-\u097F]", text):
        try:
            return transliterate(text, DEVANAGARI, ITRANS)
        except Exception:
            return text
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _maybe_transliterate(str(text))
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    # Common replacements
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_name(text: str) -> str:
    t = clean_text(text)
    for k, v in NAME_ABBREV.items():
        t = re.sub(k, v, t)
    # Also remove pure legal suffixes to compare "core" name
    tokens = [tok for tok in t.split() if tok not in LEGAL_SUFFIXES]
    t = " ".join(tokens)
    return re.sub(r"\s+", " ", t).strip()


def clean_addr(text: str) -> str:
    t = clean_text(text)
    for k, v in ADDR_ABBREV.items():
        t = re.sub(k, v, t)
    return re.sub(r"\s+", " ", t).strip()


def extract_postal(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b(\d{5,6})\b", text)
    return m.group(1) if m else ""


def extract_house_number(text: str) -> str:
    if not text:
        return ""
    m = re.match(r"\s*(\d+[a-zA-Z]?)", text)
    return m.group(1) if m else ""


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_clean"] = df["business_name"].apply(clean_name)
    df["addr_clean"] = df["business_address"].apply(clean_addr)
    df["country_clean"] = df["country"].astype(str).str.lower().str.strip()
    df["postal"] = df["business_address"].apply(extract_postal)
    df["house_no"] = df["business_address"].apply(extract_house_number)
    return df