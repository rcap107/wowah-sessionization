# %%
# In this script I am testing the features that I can add to the historical data

import polars as pl
import skrub
from src.utils import (
    add_session_features,
    add_char_features,
    add_aggregated_features,
    sample_by_user,
)


df = pl.read_parquet("data/wowah_data_raw.parquet")
df_user_month = pl.read_parquet("data/wowah_churn_data.parquet").select(
    "char", "month", "first_month"
)
# %%
df_user_month = sample_by_user(df_user_month, fraction=0.1)
df = df.with_columns(guild=pl.col("guild").replace(-1, None))
# %%
df
# %%
target_month = pl.datetime(2008, 2, 1)
historical_data = df.filter(pl.col("timestamp") < target_month)
historical_data = historical_data.with_columns(
    month=pl.col("timestamp").dt.truncate("1mo")
)
query_data = df_user_month.filter(
    (pl.col("month") == target_month) & (pl.col("first_month") < target_month)
)
# %%
from skrub._session_encoder import SessionEncoder

session_encoder = SessionEncoder(
    group_by="char", timestamp_col="timestamp", session_gap=30
)
historical_data_with_sessions = session_encoder.fit_transform(historical_data)

# %%
# Fixed features
# - Character race
# - Character class
# - First month seen
#
# Features for the current month
# - Max level reached in the month
# - Number of unique zones visited in the month
# - Most frequent location
# - Number of guilds joined in the month
# - Last guild in month
#
# Session features for the current month
# - Number of sessions in the month
# - Total session duration in the month
# - Average session duration in the month
#
# Playerbase features for the current month
# - Average level overall
# - Average level by class
# - Most frequent location
# - Number of players overall
# - Number of players by class
# - Overall time played by all players
#
# Playerbase features up to the current month


# %%
def add_fixed_features(df):
    return df.select("char", "race", "charclass")


query_data = query_data.join(add_fixed_features(historical_data), on="char", how="left")
query_data
# %%
_ = historical_data_with_sessions.with_columns(
    pl.col("timestamp").first().over("timestamp_session_id").alias("session_start"),
    pl.col("timestamp").last().over("timestamp_session_id").alias("session_end"),
).with_columns(
    session_duration=pl.col("session_end") - pl.col("session_start"),
)
session_bounds = _.select(
    "char", "session_start", "session_end", "timestamp_session_id"
).unique("timestamp_session_id")
# %%
_.unique("timestamp_session_id").group_by("char").agg(
    pl.col("session_duration").sum().alias("total_session_duration"),
    pl.col("session_duration").mean().alias("avg_session_duration"),
).sort("total_session_duration", descending=True)