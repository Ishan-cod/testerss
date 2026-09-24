import numpy as np
import polars as pl
from rapidfuzz import fuzz, process


def compute_features(pairs: pl.DataFrame,
                     s1_feats: pl.DataFrame,
                     s23_feats: pl.DataFrame) -> pl.DataFrame:
    """
    pairs: DataFrame with columns [source1_entity_id, candidate_entity_id]
    s1_feats, s23_feats: normalized dataframes with required columns
    """
    left = s1_feats.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("name_clean").alias("s1_name"),
        pl.col("name_core").alias("s1_name_core"),
        pl.col("addr_clean").alias("s1_addr"),
        pl.col("country_clean").alias("s1_country"),
        pl.col("postal").alias("s1_postal"),
    ])
    right = s23_feats.select([
        pl.col("entity_id").alias("candidate_entity_id"),
        pl.col("name_clean").alias("c_name"),
        pl.col("name_core").alias("c_name_core"),
        pl.col("addr_clean").alias("c_addr"),
        pl.col("country_clean").alias("c_country"),
        pl.col("postal").alias("c_postal"),
    ])

    df = pairs.join(left, on="source1_entity_id", how="left") \
              .join(right, on="candidate_entity_id", how="left")

    # Vectorized exact features
    df = df.with_columns([
        (pl.col("s1_country") == pl.col("c_country")).cast(pl.Int8).alias("same_country"),
        ((pl.col("s1_postal") != "") &
         (pl.col("s1_postal") == pl.col("c_postal"))).cast(pl.Int8).alias("same_postal"),
        (pl.col("s1_name_core").str.len_chars()).alias("s1_name_len"),
        (pl.col("c_name_core").str.len_chars()).alias("c_name_len"),
        (pl.col("s1_addr").str.len_chars()).alias("s1_addr_len"),
        (pl.col("c_addr").str.len_chars()).alias("c_addr_len"),
    ])
    df = df.with_columns([
        (pl.col("s1_name_len") - pl.col("c_name_len")).abs().alias("len_name_diff"),
        (pl.col("s1_addr_len") - pl.col("c_addr_len")).abs().alias("len_addr_diff"),
    ])

    # Fuzzy features in batches using rapidfuzz.process.cpdist
    fuzzy_cols = ["name_ratio","name_partial","name_token_set",
                  "name_token_sort","name_wratio",
                  "addr_ratio","addr_partial","addr_token_set",
                  "addr_token_sort","core_ratio","core_token_set"]

    n = df.height
    name_ratio = np.empty(n, dtype=np.float32)
    name_partial = np.empty(n, dtype=np.float32)
    name_token_set = np.empty(n, dtype=np.float32)
    name_token_sort = np.empty(n, dtype=np.float32)
    name_wratio = np.empty(n, dtype=np.float32)
    addr_ratio = np.empty(n, dtype=np.float32)
    addr_partial = np.empty(n, dtype=np.float32)
    addr_token_set = np.empty(n, dtype=np.float32)
    addr_token_sort = np.empty(n, dtype=np.float32)
    core_ratio = np.empty(n, dtype=np.float32)
    core_token_set = np.empty(n, dtype=np.float32)

    s1_names = df["s1_name"].to_numpy()
    c_names  = df["c_name"].to_numpy()
    s1_addrs = df["s1_addr"].to_numpy()
    c_addrs  = df["c_addr"].to_numpy()
    s1_core  = df["s1_name_core"].to_numpy()
    c_core   = df["c_name_core"].to_numpy()

    B = 200_000
    for start in range(0, n, B):
        end = min(start + B, n)
        name_ratio[start:end]       = process.cpdist(s1_names[start:end], c_names[start:end], scorer=fuzz.ratio, workers=-1) / 100.0
        name_partial[start:end]     = process.cpdist(s1_names[start:end], c_names[start:end], scorer=fuzz.partial_ratio, workers=-1) / 100.0
        name_token_set[start:end]   = process.cpdist(s1_names[start:end], c_names[start:end], scorer=fuzz.token_set_ratio, workers=-1) / 100.0
        name_token_sort[start:end]  = process.cpdist(s1_names[start:end], c_names[start:end], scorer=fuzz.token_sort_ratio, workers=-1) / 100.0
        name_wratio[start:end]      = process.cpdist(s1_names[start:end], c_names[start:end], scorer=fuzz.WRatio, workers=-1) / 100.0
        addr_ratio[start:end]       = process.cpdist(s1_addrs[start:end], c_addrs[start:end], scorer=fuzz.ratio, workers=-1) / 100.0
        addr_partial[start:end]     = process.cpdist(s1_addrs[start:end], c_addrs[start:end], scorer=fuzz.partial_ratio, workers=-1) / 100.0
        addr_token_set[start:end]   = process.cpdist(s1_addrs[start:end], c_addrs[start:end], scorer=fuzz.token_set_ratio, workers=-1) / 100.0
        addr_token_sort[start:end]  = process.cpdist(s1_addrs[start:end], c_addrs[start:end], scorer=fuzz.token_sort_ratio, workers=-1) / 100.0
        core_ratio[start:end]       = process.cpdist(s1_core[start:end], c_core[start:end], scorer=fuzz.ratio, workers=-1) / 100.0
        core_token_set[start:end]   = process.cpdist(s1_core[start:end], c_core[start:end], scorer=fuzz.token_set_ratio, workers=-1) / 100.0

    df = df.with_columns([
        pl.Series("name_ratio", name_ratio),
        pl.Series("name_partial", name_partial),
        pl.Series("name_token_set", name_token_set),
        pl.Series("name_token_sort", name_token_sort),
        pl.Series("name_wratio", name_wratio),
        pl.Series("addr_ratio", addr_ratio),
        pl.Series("addr_partial", addr_partial),
        pl.Series("addr_token_set", addr_token_set),
        pl.Series("addr_token_sort", addr_token_sort),
        pl.Series("core_ratio", core_ratio),
        pl.Series("core_token_set", core_token_set),
    ])
    df = df.drop(["s1_name","c_name","s1_addr","c_addr","s1_name_core","c_name_core"])
    return df


FEATURE_COLS = [
    "same_country","same_postal","len_name_diff","len_addr_diff",
    "name_ratio","name_partial","name_token_set","name_token_sort","name_wratio",
    "addr_ratio","addr_partial","addr_token_set","addr_token_sort",
    "core_ratio","core_token_set",
]