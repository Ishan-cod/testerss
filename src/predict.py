import glob
import polars as pl
from .features import FEATURE_COLS


def predict_and_write(model, threshold, features_dir, test_s1_ids,
                      out_match, out_cand):
    files = sorted(glob.glob(f"{features_dir}/part_*.parquet"))
    all_parts = []
    for f in files:
        df = pl.read_parquet(f)
        X = df.select(FEATURE_COLS).to_numpy()
        prob = model.predict_proba(X)[:, 1]
        df = df.select(["source1_entity_id", "candidate_entity_id"]) \
               .with_columns(pl.Series("prob", prob))
        all_parts.append(df)
    if all_parts:
        df = pl.concat(all_parts, how="vertical_relaxed")
    else:
        df = pl.DataFrame({"source1_entity_id": [], "candidate_entity_id": [],
                           "prob": []})

    # Candidates
    cand = (
        df.group_by("source1_entity_id")
          .agg(pl.col("candidate_entity_id").sort().alias("lst"))
    )
    # Matches
    match = (
        df.filter(pl.col("prob") >= threshold)
          .group_by("source1_entity_id")
          .agg(pl.col("candidate_entity_id").sort().alias("lst"))
    )

    all_s1 = pl.DataFrame({"source1_entity_id": test_s1_ids})

    cand = (
        all_s1.join(cand, on="source1_entity_id", how="left")
              .with_columns(
                  pl.col("lst").fill_null([]).list.join(",").alias("candidate_entity_ids")
              )
              .select(["source1_entity_id", "candidate_entity_ids"])
    )
    match = (
        all_s1.join(match, on="source1_entity_id", how="left")
              .with_columns(
                  pl.col("lst").fill_null([]).list.join(",").alias("matched_entity_ids")
              )
              .select(["source1_entity_id", "matched_entity_ids"])
    )

    cand.write_csv(out_cand, separator="\t", quote_style="never")
    match.write_csv(out_match, separator="\t", quote_style="never")