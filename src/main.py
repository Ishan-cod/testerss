import time, random
import numpy as np
import polars as pl
from . import config
from .io_polars import load_source
from .pairs import build_features_for_split
from .train import attach_labels, subsample, train_lgb
from .metrics import macro_f05, f05_single
from .predict import predict_and_write


def parse_gt(path):
    gt = pl.read_csv(path, separator="\t", infer_schema_length=0, quote_char=None).fill_null("")
    out = {}
    for s1, ids in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        out[s1] = {x.strip() for x in ids.split(",") if x.strip()}
    return out


def tune_threshold(df_pairs: pl.DataFrame, gt_map, s1_ids, model):
    X = df_pairs.select(__import__("src.features", fromlist=["FEATURE_COLS"]).FEATURE_COLS).to_pandas()
    prob = model.predict_proba(X)[:, 1]
    df_pairs = df_pairs.with_columns(pl.Series("prob", prob))

    best_thr, best = 0.5, -1.0
    for thr in np.arange(0.10, 0.96, 0.025):
        preds = (
            df_pairs.filter(pl.col("prob") >= thr)
                    .group_by("source1_entity_id")
                    .agg(pl.col("candidate_entity_id").alias("m"))
        )
        pred_map = {s1: set(l) for s1, l in zip(preds["source1_entity_id"], preds["m"])}
        score = macro_f05(gt_map, pred_map, s1_ids)
        if score > best:
            best, best_thr = score, float(thr)
    return best_thr, best


def main():
    random.seed(config.RANDOM_STATE); np.random.seed(config.RANDOM_STATE)

    print("Loading and normalizing ...")
    tr_s1  = load_source(config.TRAIN_S1)
    tr_s2  = load_source(config.TRAIN_S2)
    tr_s3  = load_source(config.TRAIN_S3)
    te_s1  = load_source(config.TEST_S1)
    te_s2  = load_source(config.TEST_S2)
    te_s3  = load_source(config.TEST_S3)

    tr_s23 = pl.concat([tr_s2, tr_s3], how="vertical_relaxed")
    te_s23 = pl.concat([te_s2, te_s3], how="vertical_relaxed")

    gt_map = parse_gt(config.TRAIN_GT)

    # ---------- TRAIN feature generation ----------
    print("Building training features (chunked) ...")
    tr_feat_path = build_features_for_split(tr_s1, tr_s23, "train")

    print("Attaching labels and subsampling ...")
    df = attach_labels(tr_feat_path, gt_map)

    # Split by S1 for validation
    all_s1 = df["source1_entity_id"].unique().to_list()
    random.shuffle(all_s1)
    n_val = int(len(all_s1) * config.VAL_FRACTION)
    val_ids = set(all_s1[:n_val]); trn_ids = set(all_s1[n_val:])

    df_tr = df.filter(pl.col("source1_entity_id").is_in(trn_ids))
    df_va = df.filter(pl.col("source1_entity_id").is_in(val_ids))

    df_tr_ss = subsample(df_tr, neg_ratio=5, seed=config.RANDOM_STATE)

    print("Training LightGBM ...")
    model = train_lgb(df_tr_ss)

    print("Tuning threshold on validation ...")
    thr, val_f05 = tune_threshold(df_va, gt_map, list(val_ids), model)
    print(f"Val macro F0.5 = {val_f05:.4f} @ thr = {thr:.3f}")

    # ---------- TEST inference ----------
    print("Building test features (chunked) ...")
    te_feat_path = build_features_for_split(te_s1, te_s23, "test")

    print("Writing outputs ...")
    predict_and_write(model, thr, te_feat_path,
                      te_s1["entity_id"].to_list(),
                      config.OUT_MATCH, config.OUT_CAND)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Total time: {time.time() - t:.1f}s")