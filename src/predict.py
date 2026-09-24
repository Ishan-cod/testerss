import pandas as pd
from . import config
from .blocking import get_candidates
from .features import pair_features, feature_names


def predict_test(model, threshold, test_s1, test_s23, idx, s23_lookup):
    feature_cols = feature_names()

    # Build all (S1, candidate) rows to score
    rows = []
    cand_map = {}   # s1_id -> list of candidate ids (pre-threshold)

    for _, s1_row in test_s1.iterrows():
        s1_id = s1_row["entity_id"]
        cands = get_candidates(s1_row, idx)
        # Only keep S2/S3 IDs that exist in the test set
        cands = {c for c in cands if c in s23_lookup}
        cand_map[s1_id] = sorted(cands)

        s1d = s1_row.to_dict()
        for cid in cand_map[s1_id]:
            feats = pair_features(s1d, s23_lookup[cid])
            feats["_s1"] = s1_id
            feats["_cid"] = cid
            rows.append(feats)

    if rows:
        X = pd.DataFrame(rows)
        meta = X[["_s1", "_cid"]].rename(columns={"_s1": "s1_id", "_cid": "cand_id"})
        X = X[feature_cols]
        probs = model.predict_proba(X)[:, 1]
    else:
        meta = pd.DataFrame(columns=["s1_id", "cand_id"])
        probs = []

    scored = meta.copy()
    scored["prob"] = probs

    # Aggregate matches by S1
    match_map = {s1: set() for s1 in test_s1["entity_id"]}
    for s1_id, grp in scored.groupby("s1_id"):
        match_map[s1_id] = set(grp.loc[grp["prob"] >= threshold, "cand_id"])

    return cand_map, match_map


def write_outputs(all_s1_ids, cand_map, match_map):
    # candidate_pairs.tsv
    cand_rows = []
    for s1_id in all_s1_ids:
        cids = cand_map.get(s1_id, [])
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cids),
        })
    pd.DataFrame(cand_rows).to_csv(config.OUT_CAND, sep="\t", index=False)

    # matching_results.tsv
    match_rows = []
    for s1_id in all_s1_ids:
        mids = sorted(match_map.get(s1_id, set()))
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(mids),
        })
    pd.DataFrame(match_rows).to_csv(config.OUT_MATCH, sep="\t", index=False)