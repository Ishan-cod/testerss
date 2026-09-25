import os, random, time
import numpy as np
import polars as pl

from . import config
from .io_polars import load_source
from .pairs import build_features_for_split
from .train import attach_labels, subsample, train_lgb
from .metrics import macro_f05
from .features import FEATURE_COLS
from .predict import predict_and_write


def parse_gt(path):
    gt = pl.read_csv(path, separator="\t", infer_schema_length=0,
                     quote_char=None).fill_null("")
    out = {}
    for s1, ids in zip(gt["source1_entity_id"].to_list(),
                       gt["matched_entity_ids"].to_list()):
        out[s1] = {x.strip() for x in ids.split(",") if x.strip()}
    return out


def sample_train(s1, s2, s3, gt_map):
    """Sample S1; force-include every true-match S2/S3 record."""
    s1_small = s1.sample(n=config.TRAIN_S1_SAMPLE, seed=config.RANDOM_STATE)

    required = set()
    for s1_id in s1_small["entity_id"].to_list():
        required |= gt_map.get(s1_id, set())
    s2_req_ids = [x for x in required if x.startswith("S2-")]
    s3_req_ids = [x for x in required if x.startswith("S3-")]

    s2_req = s2.filter(pl.col("entity_id").is_in(s2_req_ids))
    s3_req = s3.filter(pl.col("entity_id").is_in(s3_req_ids))

    extra2 = max(0, config.S23_SAMPLE - s2_req.height)
    extra3 = max(0, config.S23_SAMPLE - s3_req.height)
    s2_extra = s2.sample(n=min(extra2, s2.height), seed=config.RANDOM_STATE + 1)
    s3_extra = s3.sample(n=min(extra3, s3.height), seed=config.RANDOM_STATE + 2)

    s2_small = pl.concat([s2_req, s2_extra], how="vertical_relaxed").unique("entity_id")
    s3_small = pl.concat([s3_req, s3_extra], how="vertical_relaxed").unique("entity_id")

    print(f"  S1 sample: {s1_small.height:,}")
    print(f"  S2 sample: {s2_small.height:,} (required {s2_req.height:,})")
    print(f"  S3 sample: {s3_small.height:,} (required {s3_req.height:,})")
    return s1_small, s2_small, s3_small


def sample_test(s1, s2, s3):
    s1_small = s1.sample(n=min(config.TEST_S1_SAMPLE, s1.height), seed=config.RANDOM_STATE)
    s2_small = s2.sample(n=min(config.S23_SAMPLE, s2.height), seed=config.RANDOM_STATE + 1)
    s3_small = s3.sample(n=min(config.S23_SAMPLE, s3.height), seed=config.RANDOM_STATE + 2)
    print(f"  Test S1 sample: {s1_small.height:,}")
    print(f"  Test S2 sample: {s2_small.height:,}")
    print(f"  Test S3 sample: {s3_small.height:,}")
    return s1_small, s2_small, s3_small


def tune_threshold(feats_df: pl.DataFrame, gt_map, s1_ids, model):
    X = feats_df.select(FEATURE_COLS).to_numpy()
    probs = model.predict_proba(X)[:, 1]
    feats_df = feats_df.with_columns(pl.Series("prob", probs))

    best_thr, best = 0.5, -1.0
    for thr in np.arange(0.10, 0.96, 0.025):
        preds = (
            feats_df.filter(pl.col("prob") >= thr)
                    .group_by("source1_entity_id")
                    .agg(pl.col("candidate_entity_id").alias("m"))
        )
        pred_map = {s: set(l) for s, l in zip(preds["source1_entity_id"],
                                              preds["m"])}
        score = macro_f05(gt_map, pred_map, s1_ids)
        if score > best:
            best, best_thr = score, float(thr)
    return best_thr, best


def main():
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    t0 = time.time()

    print(f"SAMPLE_MODE = {config.SAMPLE_MODE}")
    print("Loading + normalizing train ...")
    tr_s1 = load_source(config.TRAIN_S1)
    tr_s2 = load_source(config.TRAIN_S2)
    tr_s3 = load_source(config.TRAIN_S3)
    gt_map = parse_gt(config.TRAIN_GT)

    if config.SAMPLE_MODE:
        print("Sampling train ...")
        tr_s1, tr_s2, tr_s3 = sample_train(tr_s1, tr_s2, tr_s3, gt_map)

    print("Building train features ...")
    tr_feat_dir = build_features_for_split(tr_s1, tr_s2, tr_s3, "train")

    print("Labeling + subsampling ...")
    df = attach_labels(tr_feat_dir, gt_map)
    print(f"  pairs: {df.height:,}, positives: {int(df['label'].sum()):,}")

    # Split by S1
    all_s1 = df["source1_entity_id"].unique().to_list()
    random.shuffle(all_s1)
    n_val = int(len(all_s1) * config.VAL_FRACTION)
    val_ids = set(all_s1[:n_val])
    trn_ids = set(all_s1[n_val:])

    df_tr = df.filter(pl.col("source1_entity_id").is_in(list(trn_ids)))
    df_va = df.filter(pl.col("source1_entity_id").is_in(list(val_ids)))
    del df

    df_tr_ss = subsample(df_tr, neg_ratio=4, seed=config.RANDOM_STATE)
    print(f"  training rows after subsample: {df_tr_ss.height:,}")

    print("Training LightGBM ...")
    model = train_lgb(df_tr_ss)

    print("Tuning threshold ...")
    thr, val_f05 = tune_threshold(df_va, gt_map, list(val_ids), model)
    print(f"  Val macro F0.5 = {val_f05:.4f} @ thr = {thr:.3f}")

    # ---- test ----
    print("Loading + normalizing test ...")
    te_s1 = load_source(config.TEST_S1)
    te_s2 = load_source(config.TEST_S2)
    te_s3 = load_source(config.TEST_S3)

    if config.SAMPLE_MODE:
        print("Sampling test ...")
        te_s1, te_s2, te_s3 = sample_test(te_s1, te_s2, te_s3)

    print("Building test features ...")
    te_feat_dir = build_features_for_split(te_s1, te_s2, te_s3, "test")

    print("Writing outputs ...")
    predict_and_write(model, thr, te_feat_dir,
                      te_s1["entity_id"].to_list(),
                      config.OUT_MATCH, config.OUT_CAND)

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()