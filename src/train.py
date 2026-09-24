import numpy as np
import polars as pl
import lightgbm as lgb
from . import config
from .features import FEATURE_COLS


def attach_labels(features_parquet: str, gt_map: dict) -> pl.DataFrame:
    df = pl.read_parquet(features_parquet)
    # Build a set of (s1, cand) positives
    pos = set()
    for s1, cands in gt_map.items():
        for c in cands:
            pos.add((s1, c))

    df = df.with_columns([
        pl.struct(["source1_entity_id", "candidate_entity_id"])
          .map_elements(lambda s: 1 if (s["source1_entity_id"], s["candidate_entity_id"]) in pos else 0,
                        return_dtype=pl.Int8)
          .alias("label")
    ])
    return df


def subsample(df: pl.DataFrame, neg_ratio: int = 5, seed: int = 0) -> pl.DataFrame:
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)
    n_keep = min(neg.height, pos.height * neg_ratio)
    neg = neg.sample(n=n_keep, seed=seed)
    return pl.concat([pos, neg])


def train_lgb(df: pl.DataFrame) -> lgb.LGBMClassifier:
    X = df.select(FEATURE_COLS).to_pandas()
    y = df["label"].to_numpy()
    model = lgb.LGBMClassifier(**config.LGB_PARAMS)
    model.fit(X, y)
    return model