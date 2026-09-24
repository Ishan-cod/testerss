import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config
from .features import pair_features, feature_names
from .metrics import macro_f05
from .blocking import get_candidates


def build_training_frame(s1_df, s23_df, gt_map, idx):
    """
    For every S1 in s1_df, get candidates from s23_df, extract features, label.
    Positives are guaranteed to be added even if blocking missed them (for training only).
    """
    s23_lookup = s23_df.set_index("entity_id").to_dict("index")
    feats_list = []
    labels = []
    s1_ids = []
    cand_ids = []

    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        true_matches = gt_map.get(s1_id, set())

        cands = get_candidates(s1_row, idx)

        # Force-add positives (train-only recall fix)
        for tm in true_matches:
            if tm in s23_lookup:
                cands.add(tm)

        for cid in cands:
            if cid not in s23_lookup:
                continue
            s2_row = s23_lookup[cid]
            feats = pair_features(s1_row.to_dict(), s2_row)
            feats_list.append(feats)
            labels.append(1 if cid in true_matches else 0)
            s1_ids.append(s1_id)
            cand_ids.append(cid)

    X = pd.DataFrame(feats_list)[feature_names()]
    y = np.array(labels)
    meta = pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids})
    return X, y, meta


def train_model(X_train, y_train):
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    scale = neg / pos
    params = dict(config.LGB_PARAMS)
    params["scale_pos_weight"] = scale

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def tune_threshold(model, X_val, meta_val, val_truth_map, val_s1_ids):
    """Pick threshold that maximizes macro F0.5 on validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    val = meta_val.copy()
    val["prob"] = probs

    best_thr, best_score = 0.5, -1.0
    for thr in config.THRESHOLD_GRID:
        preds = {}
        for s1_id, grp in val.groupby("s1_id"):
            preds[s1_id] = set(grp.loc[grp["prob"] >= thr, "cand_id"])
        score = macro_f05(val_truth_map, preds, val_s1_ids)
        if score > best_score:
            best_score, best_thr = score, thr

    return best_thr, best_score