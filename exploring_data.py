# %%
import pandas as pd
import polars as pl
import skrub
from skrub import TableReport

# %%
path = "data/wowah_data_raw.parquet"


 %%
def with_polars(path):
    df = pl.read_parquet(path)

    df_sorted = df.sort("char", "timestamp")
    session_duration = 20 * 60 * 1000  # 20 minutes in milliseconds

    df_sessionized = (
        df_sorted.with_columns(
            (pl.col("char").diff().fill_null(0) > 0).alias("char_diff"),
            (
                pl.col("timestamp").dt.epoch("ms").diff().fill_null(0)
                > session_duration
            ).alias("time_diff"),
        )
        .with_columns(
            (pl.col("char_diff") | pl.col("time_diff")).alias("is_new_session")
        )
        .with_columns(pl.col("is_new_session").cum_sum().alias("session_id"))
        .drop(["char_diff", "time_diff", "is_new_session"])
    )
    df_sessionized = df_sessionized.with_columns(
        pl.col("char").count().over("session_id").alias("session_duration"),
        pl.col("session_id").count().over("char").alias("sessions_per_char"),
        (pl.col("session_id").count().over("char") * session_duration).alias(
            "total_session_time"
        ),
    )
    return df_sessionized

# %%
def with_pandas(path):
    df = pd.read_parquet(path)

    df_sorted = df.sort_values(["char", "timestamp"])
    session_duration = 20 * 60 * 1000  # 20 minutes in milliseconds

    df_sorted["char_diff"] = df_sorted["char"].diff().fillna(0) > 0
    df_sorted["time_diff"] = (
        df_sorted["timestamp"].diff().fillna(0) > session_duration
    )
    df_sorted["is_new_session"] = df_sorted["char_diff"] | df_sorted["time_diff"]
    df_sorted["session_id"] = df_sorted["is_new_session"].cumsum()

    df_sessionized = df_sorted.drop(columns=["char_diff", "time_diff", "is_new_session"])

    # Calculate session duration and sessions per character
    session_stats = (
        df_sessionized.groupby("session_id")
        .agg(session_duration=("timestamp", "count"))
        .reset_index()
    )
    char_stats = (
        df_sessionized.groupby("char")
        .agg(sessions_per_char=("session_id", "nunique"))
        .reset_index()
    )

    # Merge stats back to the original DataFrame
    df_sessionized = df_sessionized.merge(session_stats, on="session_id", how="left")
    df_sessionized = df_sessionized.merge(char_stats, on="char", how="left")
    df_sessionized["total_session_time"] = (
        df_sessionized["sessions_per_char"] * session_duration
    )

    return df_sessionized
# %%
with_polars(path)
# %%
with_pandas(path)
# %%

df = pd.read_parquet(path)

df_sorted = df.sort_values(["char", "timestamp"]).head(100_000)
session_duration = 20 * 60 * 1000  # 20 minutes in milliseconds
# %%
df_sorted["char_diff"] = df_sorted["char"].diff().fillna(0) > 0
#%%

df_sorted["time_diff"] = (
    df_sorted["timestamp"].astype(int).diff().fillna(0) // 10**9 > session_duration
)
df_sorted["is_new_session"] = df_sorted["char_diff"] | df_sorted["time_diff"]
df_sorted["session_id"] = df_sorted["is_new_session"].cumsum()
# %%
df_sorted
# %%
3