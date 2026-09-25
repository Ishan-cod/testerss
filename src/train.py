import glob
import numpy as np
import polars as pl
import lightgbm as lgb
from . import config
from .features import FEATURE_COLS


def attach_labels(features_dir: str, gt_map: dict) -> pl.DataFrame:
    """Reads all part_*.parquet, labels each pair, returns a single DataFrame."""
    files = sorted(glob.glob(f"{features_dir}/part_*.parquet"))
    if not files:
        raise RuntimeError(f"No feature files in {features_dir}")

    pos_set = set()
    for s1, cands in gt_map.items():
        for c in cands:
            pos_set.add((s1, c))

    parts = []
    for f in files:
        df = pl.read_parquet(f)
        labels = [
            1 if (s1, c) in pos_set else 0
            for s1, c in zip(df["source1_entity_id"].to_list(),
                             df["candidate_entity_id"].to_list())
        ]
        df = df.with_columns(pl.Series("label", labels, dtype=pl.Int8))
        parts.append(df)
    return pl.concat(parts, how="vertical_relaxed")


def subsample(df: pl.DataFrame, neg_ratio: int = 4, seed: int = 0) -> pl.DataFrame:
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)
    n_keep = min(neg.height, max(pos.height, 1) * neg_ratio)
    neg = neg.sample(n=n_keep, seed=seed)
    return pl.concat([pos, neg], how="vertical_relaxed")


def train_lgb(df: pl.DataFrame) -> lgb.LGBMClassifier:
    X = df.select(FEATURE_COLS).to_numpy()
    y = df["label"].to_numpy()
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    params = dict(config.LGB_PARAMS)
    params["scale_pos_weight"] = neg / pos
    print(f"  scale_pos_weight = {params['scale_pos_weight']:.2f}")
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model