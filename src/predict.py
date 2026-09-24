import polars as pl
from .features import FEATURE_COLS


def predict_and_write(model, threshold, features_parquet, test_s1_ids, out_match, out_cand):
    df = pl.read_parquet(features_parquet)

    X = df.select(FEATURE_COLS).to_pandas()
    prob = model.predict_proba(X)[:, 1]
    df = df.with_columns(pl.Series("prob", prob))

    # Candidates: everything we scored
    cand = (
        df.group_by("source1_entity_id")
          .agg(pl.col("candidate_entity_id").sort().alias("cand_list"))
    )
    # Matches: above threshold
    match = (
        df.filter(pl.col("prob") >= threshold)
          .group_by("source1_entity_id")
          .agg(pl.col("candidate_entity_id").sort().alias("match_list"))
    )

    # Ensure every S1 appears
    all_s1 = pl.DataFrame({"source1_entity_id": test_s1_ids})
    cand = all_s1.join(cand, on="source1_entity_id", how="left").with_columns(
        pl.col("cand_list").fill_null([]).list.join(",").alias("candidate_entity_ids")
    ).select(["source1_entity_id","candidate_entity_ids"])

    match = all_s1.join(match, on="source1_entity_id", how="left").with_columns(
        pl.col("match_list").fill_null([]).list.join(",").alias("matched_entity_ids")
    ).select(["source1_entity_id","matched_entity_ids"])

    cand.write_csv(out_cand, separator="\t", quote_style="never")
    match.write_csv(out_match, separator="\t", quote_style="never")