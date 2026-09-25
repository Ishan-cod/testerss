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
    empty_list = pl.lit([], dtype=pl.List(pl.String))

    name_prefix = pl.col("name_core").str.replace_all(" ", "").str.slice(0, 5)

    # --- Scalar branches: wrap in a 1-element list, or empty list when N/A ---
    postal_key = (
        pl.when(pl.col("postal") != "")
          .then(pl.concat_list(pl.lit("p:") + pl.col("postal")))
          .otherwise(empty_list)
    )
    prefix_key = (
        pl.when(name_prefix.str.len_chars() >= 3)
          .then(pl.concat_list(
              pl.lit("n:") + pl.col("country_clean") + pl.lit(":") + name_prefix
          ))
          .otherwise(empty_list)
    )

    # --- List branches: split → filter → prefix each element ---
    name_tokens = (
        pl.col("name_core").str.split(" ")
          .list.eval(pl.element().filter(
              (pl.element().str.len_chars() >= 5) &
              (~pl.element().is_in(list(STOPWORDS)))
          ))
          .list.eval(pl.lit("t:") + pl.element())
    )
    addr_tokens = (
        pl.col("addr_clean").str.split(" ")
          .list.eval(pl.element().filter(
              (pl.element().str.len_chars() >= 6) &
              (~pl.element().is_in(list(STOPWORDS)))
          ))
          .list.eval(pl.lit("a:") + pl.element())
    )

    # --- All four branches are now List(String) with the same row count ---
    keys = (
        pl.concat_list([postal_key, prefix_key, name_tokens, addr_tokens])
          .list.eval(pl.element().explode())     # flatten List(List(String)) → List(String)
          .list.unique()
    )
    return keys.alias("block_keys")


def add_block_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_keys_expr())


def build_keys_long(df: pl.DataFrame) -> pl.DataFrame:
    """Explode block_keys into long form: (entity_id, block_key)."""
    return (
        df.select(["entity_id", "block_keys"])
          .explode("block_keys")
          .drop_nulls("block_keys")
    )


def prune_common_keys(keys_df: pl.DataFrame) -> pl.DataFrame:
    counts = keys_df.group_by("block_keys").len()
    common = counts.filter(pl.col("len") > config.MAX_COMMON_KEY_FREQ)["block_keys"]
    return keys_df.filter(~pl.col("block_keys").is_in(common))