import os
import sys
import pandas as pd


def _read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def validate(matching_path, candidate_path, test_dir):
    issues = []

    match = _read_tsv(matching_path)
    cand = _read_tsv(candidate_path)

    # Column names
    if list(match.columns) != ["source1_entity_id", "matched_entity_ids"]:
        issues.append(f"matching_results.tsv wrong columns: {list(match.columns)}")
    if list(cand.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        issues.append(f"candidate_pairs.tsv wrong columns: {list(cand.columns)}")

    test_s1 = _read_tsv(os.path.join(test_dir, "test_source1.tsv"))
    test_s2 = _read_tsv(os.path.join(test_dir, "test_source2.tsv"))
    test_s3 = _read_tsv(os.path.join(test_dir, "test_source3.tsv"))

    s1_ids = set(test_s1["entity_id"])
    valid_targets = set(test_s2["entity_id"]) | set(test_s3["entity_id"])

    # Every S1 present exactly once
    if set(match["source1_entity_id"]) != s1_ids:
        issues.append("matching_results.tsv missing or extra S1 entities")
    if match["source1_entity_id"].duplicated().any():
        issues.append("duplicate S1 rows in matching_results.tsv")

    if set(cand["source1_entity_id"]) != s1_ids:
        issues.append("candidate_pairs.tsv missing or extra S1 entities")
    if cand["source1_entity_id"].duplicated().any():
        issues.append("duplicate S1 rows in candidate_pairs.tsv")

    cand_map = {}
    for _, row in cand.iterrows():
        ids = [x.strip() for x in row["candidate_entity_ids"].split(",") if x.strip()]
        cand_map[row["source1_entity_id"]] = set(ids)
        if len(ids) != len(set(ids)):
            issues.append(f"duplicate IDs in candidates for {row['source1_entity_id']}")
        if any(i not in valid_targets for i in ids):
            issues.append(f"invalid S2/S3 ID in candidates for {row['source1_entity_id']}")

    for _, row in match.iterrows():
        ids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            issues.append(f"duplicate IDs in matches for {row['source1_entity_id']}")
        if any(i not in valid_targets for i in ids):
            issues.append(f"invalid S2/S3 ID in matches for {row['source1_entity_id']}")
        missing_from_cand = set(ids) - cand_map.get(row["source1_entity_id"], set())
        if missing_from_cand:
            issues.append(
                f"matched ID not in candidates for {row['source1_entity_id']}: {missing_from_cand}"
            )

    if issues:
        print("FAIL:")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()
    validate(args.matching, args.candidate, args.test_dir)