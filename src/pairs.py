import polars as pl
from tqdm import tqdm
from . import config
from .blocking import add_block_keys, build_keys_long, prune_common_keys
from .features import compute_features


def _candidate_pairs_for_chunk(s1_chunk: pl.DataFrame,
                               s2_keys: pl.DataFrame,
                               s3_keys: pl.DataFrame) -> pl.DataFrame:
    """
    For a small S1 chunk, compute candidate pairs against S2 and S3 keys.
    Returns DataFrame(source1_entity_id, candidate_entity_id).
    """
    # <-- ADD THIS LINE
    s1_chunk = add_block_keys(s1_chunk)

    s1_keys = build_keys_long(s1_chunk).select(
        pl.col("entity_id").alias("source1_entity_id"),
        "block_keys",
    )

    def _join_one(keys_df):
        return (
            s1_keys.join(keys_df, on="block_keys", how="inner")
                   .select(["source1_entity_id", "entity_id"])
                   .rename({"entity_id": "candidate_entity_id"})
        )

    pairs = pl.concat([_join_one(s2_keys), _join_one(s3_keys)], how="vertical")
    pairs = pairs.unique()

    # Cap per S1
    pairs = (
        pairs.with_columns(
            pl.int_range(pl.len()).over("source1_entity_id").alias("rank")
        )
        .filter(pl.col("rank") < config.MAX_CANDIDATES_PER_S1)
        .drop("rank")
    )
    return pairs


def build_features_for_split(s1: pl.DataFrame,
                             s2: pl.DataFrame,
                             s3: pl.DataFrame,
                             split_name: str) -> str:
    """
    Chunked feature generation. Returns path to feature parquet(s) directory.
    Features are written as cache/<split>_features/part_*.parquet.
    """
    out_dir = config.CACHE_DIR + f"/{split_name}_features"
    import os, glob, shutil
    if os.path.isdir(out_dir) and glob.glob(out_dir + "/part_*.parquet"):
        return out_dir
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Build keys once
    s2 = add_block_keys(s2)
    s3 = add_block_keys(s3)
    s2_keys = prune_common_keys(build_keys_long(s2))
    s3_keys = prune_common_keys(build_keys_long(s3))

    # Lookup tables keyed by entity_id (in memory — S2/S3 sampled small in SAMPLE_MODE)
    s2_lookup = s2.select(["entity_id","name_clean","name_core","addr_clean",
                           "country_clean","postal"])
    s3_lookup = s3.select(["entity_id","name_clean","name_core","addr_clean",
                           "country_clean","postal"])
    s23_lookup = pl.concat([s2_lookup, s3_lookup], how="vertical_relaxed")

    n_chunks = (s1.height + config.S1_CHUNK - 1) // config.S1_CHUNK
    part_idx = 0
    for i in tqdm(range(n_chunks), desc=f"features[{split_name}]"):
        start = i * config.S1_CHUNK
        end = min(start + config.S1_CHUNK, s1.height)
        s1_chunk = s1.slice(start, end - start)

        pairs = _candidate_pairs_for_chunk(s1_chunk, s2_keys, s3_keys)
        if pairs.height == 0:
            continue

        feats = compute_features(pairs, s1_chunk, s23_lookup)
        feats.write_parquet(f"{out_dir}/part_{part_idx:05d}.parquet")
        part_idx += 1

        # Free
        del pairs, feats

    return out_dir