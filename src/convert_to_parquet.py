# %%
import polars as pl


# %%
data_file = "data/wowah_parsed.csv"

schema = {
    "timestamp": pl.Utf8,
    "avatar_id": pl.Int32,
    "guild": pl.Int32,
    "level": pl.Int32,
    "race": pl.Utf8,
    "charclass": pl.Utf8,
    "zone": pl.Utf8,
}

df = pl.scan_csv(data_file, has_header=True, schema=schema)

# %%
df.with_columns(
    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%y %H:%M:%S")
).sort(["timestamp", "avatar_id"]).collect().write_parquet("data/wowah_data.parquet")
# %%
