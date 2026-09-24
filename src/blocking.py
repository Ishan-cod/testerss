import polars as pl
from . import config

STOPWORDS = {
    "the","and","of","for","a","an","in","on","at","to",
    "road","street","avenue","lane","drive","floor","plot",
    "near","opposite","india","united","states","usa","us","france",
    "corporation","incorporated","limited","private","company",
    "enterprise","enterprises","llp","llc",
}


def _keys_expr() -> pl.Expr:
    """
    For each row produce a list[str] of blocking keys.
    Keys are typed with prefixes to avoid collisions.
    """
    name_tokens = (
        pl.col("name_core").str.split(" ")
          .list.eval(pl.element().filter(
              (pl.element().str.len_chars() >= 5) &
              (~pl.element().is_in(list(STOPWORDS)))
          ))
    )
    name_prefix = pl.col("name_core").str.replace_all(" ", "").str.slice(0, 5)
    addr_tokens = (
        pl.col("addr_clean").str.split(" ")
          .list.eval(pl.element().filter(
              (pl.element().str.len_chars() >= 6) &
              (~pl.element().is_in(list(STOPWORDS)))
          ))
    )
    keys = (
        pl.concat_list([
            (pl.lit("p:") + pl.col("postal")).filter(pl.col("postal") != ""),
            (pl.lit("n:") + pl.col("country_clean") + pl.lit(":") + name_prefix)
                .filter(name_prefix.str.len_chars() >= 3),
            (pl.lit("t:") + pl.col("country_clean") + pl.lit(":") + name_tokens),
            (pl.lit("a:") + pl.col("country_clean") + pl.lit(":") + addr_tokens),
        ]).list.flatten().list.unique()
    )
    return keys.alias("block_keys")


def add_block_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_keys_expr())


def build_keys_table(df: pl.DataFrame) -> pl.DataFrame:
    """Explode to (entity_id, block_keys) long form."""
    return (
        df.select(["entity_id", "block_keys"])
          .explode("block_keys")
          .drop_nulls("block_keys")
    )


def prune_common_keys(keys_df: pl.DataFrame) -> pl.DataFrame:
    counts = keys_df.group_by("block_keys").len()
    common = counts.filter(pl.col("len") > config.MAX_COMMON_KEY_FREQ)["block_keys"]
    return keys_df.filter(~pl.col("block_keys").is_in(common))


def generate_candidate_pairs(s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    """
    Returns DataFrame(entity_id, cand_id) — one row per (S1, candidate).
    """
    s1k = prune_common_keys(build_keys_table(s1))
    s23k = prune_common_keys(build_keys_table(s23))

    pairs = (
        s1k.join(s23k, on="block_keys", how="inner")
            .select([
                pl.col("entity_id").alias("source1_entity_id"),
                pl.col("entity_id_right").alias("candidate_entity_id"),
            ])
            .unique()
    )

    # Cap per S1
    pairs = (
        pairs.with_columns(pl.int_range(pl.len()).over("source1_entity_id").alias("rank"))
             .filter(pl.col("rank") < config.MAX_CANDIDATES_PER_S1)
             .drop("rank")
    )
    return pairs