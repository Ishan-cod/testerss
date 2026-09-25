import polars as pl
from src.blocking import add_block_keys
from src.io_polars import load_source
from src import config

df = load_source(config.TEST_S2).head(1000)
out = add_block_keys(df)
print(out.select("entity_id", "block_keys").head(5))
print("rows:", out.height)