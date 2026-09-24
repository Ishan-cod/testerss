import os
import polars as pl
from tqdm import tqdm
from . import config
from .blocking import add_block_keys, generate_candidate_pairs
from .features import compute_features


def build_features_for_split(s1: pl.DataFrame,
                             s23: pl.DataFrame,
                             split_name: str) -> str:
    """
    Generates all (S1, candidate) pairs and features, chunked by S1,
    writes to parquet. Returns the parquet path.
    """
    cache = os.path.join(config.CACHE_DIR, f"{split_name}_features.parquet")
    if os.path.exists(cache):
        return cache

    s1 = add_block_keys(s1)
    s23 = add_block_keys(s23)

    writers = []
    n_chunks = (s1.height + config.S1_CHUNK - 1) // config.S1_CHUNK

    for i in tqdm(range(n_chunks), desc=f"features[{split_name}]"):
        start = i * config.S1_CHUNK
        end = min(start + config.S1_CHUNK, s1.height)
        s1_chunk = s1.slice(start, end - start)

        pairs = generate_candidate_pairs(s1_chunk, s23)
        if pairs.height == 0:
            continue

        feats = compute_features(pairs, s1_chunk, s23)
        writers.append(feats)

    if not writers:
        empty = pl.DataFrame({c: [] for c in ["source1_entity_id","candidate_entity_id"] + __import__("src.features", fromlist=["FEATURE_COLS"]).FEATURE_COLS})
        empty.write_parquet(cache)
        return cache

    # Stream to parquet in append mode
    first = writers[0]
    first.write_parquet(cache)
    for w in writers[1:]:
        # polars has no direct append; use pyarrow
        import pyarrow.parquet as pq
        tbl = w.to_arrow()
        pq.write_table(tbl, cache, append=True)
    return cache