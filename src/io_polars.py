import polars as pl
from . import config

NAME_ABBREV = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "co": "company", "intl": "international",
    "ent": "enterprises", "svc": "service", "svcs": "services",
    "tech": "technology", "mgmt": "management", "assoc": "associates",
}
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "hwy": "highway", "apt": "apartment",
    "flr": "floor", "nr": "near", "opp": "opposite",
}
LEGAL_SUFFIXES = ["corporation","incorporated","limited","private",
                  "company","enterprise","enterprises","llp","llc","plc"]


def _clean_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
          .fill_null("")
          .str.to_lowercase()
          .str.replace_all(r"[^a-z0-9\s]", " ")
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
    )


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        _clean_expr("business_name").alias("name_clean"),
        _clean_expr("business_address").alias("addr_clean"),
        pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("country_clean"),
    ])
    # Apply abbreviation maps via nested replace_all
    name_expr = pl.col("name_clean")
    for k, v in NAME_ABBREV.items():
        name_expr = name_expr.str.replace_all(rf"\b{k}\b", v)
    df = df.with_columns(name_expr.alias("name_clean"))

    addr_expr = pl.col("addr_clean")
    for k, v in ADDR_ABBREV.items():
        addr_expr = addr_expr.str.replace_all(rf"\b{k}\b", v)
    df = df.with_columns(addr_expr.alias("addr_clean"))

    # Remove legal suffixes for a "core name"
    core = pl.col("name_clean")
    df = df.with_columns(core.alias("name_core"))
    # Remove suffix tokens
    df = df.with_columns(
        pl.col("name_core")
          .str.split(" ")
          .list.eval(pl.element().filter(~pl.element().is_in(LEGAL_SUFFIXES)))
          .list.join(" ")
          .alias("name_core")
    )
    # Extract postal (5-6 digits)
    df = df.with_columns(
        pl.col("addr_clean").str.extract(r"(\d{5,6})", 1).fill_null("").alias("postal")
    )
    return df


def load_source(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
    return normalize(df)