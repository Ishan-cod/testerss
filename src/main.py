import time
import random
import numpy as np
import pandas as pd

from . import config
from .load_data import load_train, load_test, parse_ground_truth
from .normalize import add_normalized_columns
from .blocking import build_index
from .train import build_training_frame, train_model, tune_threshold
from .predict import predict_test, write_outputs
from .metrics import macro_f05


def _seed():
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)


def main():
    t0 = time.time()
    _seed()

    print("== Loading data ==")
    tr_s1, tr_s2, tr_s3, gt = load_train()
    te_s1, te_s2, te_s3 = load_test()

    print(f"Train S1={len(tr_s1)}, S2={len(tr_s2)}, S3={len(tr_s3)}")
    print(f"Test  S1={len(te_s1)}, S2={len(te_s2)}, S3={len(te_s3)}")

    print("== Normalizing ==")
    tr_s1 = add_normalized_columns(tr_s1)
    tr_s2 = add_normalized_columns(tr_s2)
    tr_s3 = add_normalized_columns(tr_s3)
    te_s1 = add_normalized_columns(te_s1)
    te_s2 = add_normalized_columns(te_s2)
    te_s3 = add_normalized_columns(te_s3)

    tr_s23 = pd.concat([tr_s2, tr_s3], ignore_index=True)
    te_s23 = pd.concat([te_s2, te_s3], ignore_index=True)

    gt_map = parse_ground_truth(gt)

    print("== Splitting train/val by S1 entity ==")
    s1_ids = list(tr_s1["entity_id"])
    random.shuffle(s1_ids)
    n_val = int(len(s1_ids) * config.VAL_FRACTION)
    val_ids = set(s1_ids[:n_val])
    trn_ids = set(s1_ids[n_val:])

    trn_s1 = tr_s1[tr_s1["entity_id"].isin(trn_ids)].reset_index(drop=True)
    val_s1 = tr_s1[tr_s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    # For training only, index over S2+S3 (this leaks S1? No — S2/S3 have no S1 IDs)
    print("== Building blocking index (train) ==")
    train_idx = build_index(tr_s23)

    print("== Building training frame ==")
    X_tr, y_tr, meta_tr = build_training_frame(trn_s1, tr_s23, gt_map, train_idx)
    X_val, y_val, meta_val = build_training_frame(val_s1, tr_s23, gt_map, train_idx)

    print(f"Train pairs: {len(X_tr)} (pos={int(y_tr.sum())})")
    print(f"Val   pairs: {len(X_val)} (pos={int(y_val.sum())})")

    print("== Training model ==")
    model = train_model(X_tr, y_tr)

    print("== Tuning threshold for F0.5 ==")
    best_thr, best_f05 = tune_threshold(model, X_val, meta_val, gt_map, val_ids)
    print(f"Best threshold={best_thr:.3f} | Val macro F0.5={best_f05:.4f}")

    # Retrain on full training data (train + val) for the final model
    print("== Retraining on full training set ==")
    X_full, y_full, _ = build_training_frame(tr_s1, tr_s23, gt_map, train_idx)
    final_model = train_model(X_full, y_full)

    # Rebuild threshold using the *validation* split but the *final* model
    # (fast, since we already have X_val / meta_val)
    final_thr, final_f05 = tune_threshold(
        final_model, X_val, meta_val, gt_map, val_ids
    )
    print(f"Final model threshold={final_thr:.3f} | Val macro F0.5={final_f05:.4f}")

    # ---------- Test inference ----------
    print("== Blocking on test ==")
    test_idx = build_index(te_s23)
    s23_lookup = te_s23.set_index("entity_id").to_dict("index")

    print("== Predicting ==")
    cand_map, match_map = predict_test(
        final_model, final_thr, te_s1, te_s23, test_idx, s23_lookup
    )

    print("== Writing outputs ==")
    write_outputs(list(te_s1["entity_id"]), cand_map, match_map)

    # Sanity stats
    n_singletons_pred = sum(1 for s1 in te_s1["entity_id"] if not match_map[s1])
    avg_cands = sum(len(v) for v in cand_map.values()) / max(len(cand_map), 1)
    avg_matches = sum(len(v) for v in match_map.values()) / max(len(match_map), 1)
    print(f"Predicted singletons: {n_singletons_pred}/{len(te_s1)}")
    print(f"Avg candidates per S1: {avg_cands:.2f}")
    print(f"Avg matches per S1:   {avg_matches:.2f}")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()