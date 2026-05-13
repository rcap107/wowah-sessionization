# %%
# In this script I am testing the features that I can add to the historical data

import polars as pl
import datetime
import skrub
from src.utils import (
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
# Here I define the target month, i.e., the month for which I want to predict churn.
# The cutoff month is the month before the target month: the idea is that during
# the "cutoff month" we can take some action to prevent churn in the "target month".
# I then filter the historical data to only include data before the cutoff month.
target_month = datetime.datetime(2008, 6, 1)
cutoff_month = pl.Series([target_month]).dt.offset_by("-1mo").first()
historical_data = df.filter(pl.col("timestamp") < target_month)
historical_data = historical_data.with_columns(
    month=pl.col("timestamp").dt.truncate("1mo")
)
query_data = df_user_month.filter(
    (pl.col("month") == target_month) & (pl.col("first_month") < target_month)
)
query_data = query_data.with_columns(
    previous_month=pl.col("month").dt.offset_by("-1mo")
)
# %%
from skrub._session_encoder import SessionEncoder

session_encoder = SessionEncoder(
    group_by="char", timestamp_col="timestamp", session_gap=30
)
historical_data_with_sessions = session_encoder.fit_transform(historical_data)

# %% [markdown]
# Fixed features
# - [x] Character race
# - [x]Character class
# - [x] First month seen
#
# Features for the current month
# - [ ] Max level reached in the month
# - [ ] Number of unique zones visited in the month
# - [ ] Most frequent location
# - [ ] Number of guilds joined in the month
# - [ ] Last guild in month
#
# Session features for the current month
# - [x] Number of sessions in the month
# - [x] Total session duration in the month
# - [x] Average session duration in the month
#
# Playerbase features for the current month
# - [ ] Average level overall
# - [ ] Average level by class
# - [ ] Most frequent location
# - [ ] Number of players overall
# - [ ] Number of players by class
# - [ ] Overall time played by all players
#
# Playerbase features up to the current month


# %%
def add_fixed_features(df):
    return df.select("char", "race", "charclass").unique("char")


query_data = query_data.join(add_fixed_features(historical_data), on="char", how="left")
# %%
# Adding the session start and end to find the session duration
# Sessions that end within a single heartbeat have the same start and end, thus
# duration = 0. I will replace those with a duration of 1 minute so that the
# total logged time over a month is not 0. This is useful to distinguish between
# players that never logged in and players that logged in but had very short sessions.


def get_session_duration():
    _ = (
        historical_data_with_sessions.with_columns(
            pl.col("timestamp")
            .first()
            .over("timestamp_session_id")
            .alias("session_start"),
            pl.col("timestamp")
            .last()
            .over("timestamp_session_id")
            .alias("session_end"),
        )
        .with_columns(
            (pl.col("session_end") - pl.col("session_start")).alias("session_duration"),
        )
        .with_columns(
            pl.when(pl.col("session_duration") == pl.duration(microseconds=0))
            .then(pl.duration(minutes=1))
            .otherwise(pl.col("session_duration"))
            .alias("session_duration")
        )
    )
    return _


hist_session_duration = get_session_duration()
hist_session_duration

# %%
historical_duration = (
    hist_session_duration.unique("timestamp_session_id")
    .group_by("char")
    .agg(
        pl.col("session_duration").sum().alias("hist_total_session_duration"),
        pl.col("session_duration").mean().alias("hist_avg_session_duration"),
    )
    .sort("hist_total_session_duration", descending=True)
)


# %%
def add_session_features(df):
    session_duration = (
        hist_session_duration.unique("timestamp_session_id")
        .group_by("char")
        .agg(
            pl.col("session_duration").sum().alias("hist_total_session_duration"),
            pl.col("session_duration").mean().alias("hist_avg_session_duration"),
            pl.col("session_duration").count().alias("hist_num_sessions"),
        )
    )
    monthly_duration = (
        hist_session_duration.unique("timestamp_session_id")
        .group_by("char", "month")
        .agg(
            pl.col("session_duration").sum().alias("monthly_total_session_duration"),
            pl.col("session_duration").mean().alias("monthly_avg_session_duration"),
            pl.col("session_duration").count().alias("monthly_num_sessions"),
        )
    )
    df = df.join(session_duration, on="char", how="left")
    df = (
        df
        .join(
            monthly_duration,
            left_on=["char", "previous_month"],
            right_on=["char", "month"],
            how="left",
        )
    )
    return df


query_data_with_monthly_features = add_session_features(query_data)
query_data_with_monthly_features


# %%
def add_monthly_player_features(df):
    _ = hist_session_duration.group_by("char", "month").agg(
        pl.col("level").max().alias("max_level_month"),
        pl.col("zone").n_unique().alias("num_zones_month"),
        pl.col("zone").mode().first().alias("most_freq_zone_month"),
        pl.col("guild").n_unique().alias("num_guilds_month"),
        pl.col("guild").last().alias("last_guild_month"),
        pl.col("guild").first().alias("first_guild_month"),
    )
    return (
        df
        .join(
            _,
            left_on=["char", "previous_month"],
            right_on=["char", "month"],
            how="left",
        )
    )
query_data_with_player_features = add_monthly_player_features(query_data_with_monthly_features)

# %%
def add_class_features(df):
    _ = (
        hist_session_duration.with_columns(
            pl.col("level")
            .mean()
            .over(["charclass", "month"])
            .alias("class_avg_level_month"),
            pl.col("char")
            .n_unique()
            .over(["charclass", "month"])
            .alias("class_num_players_month"),
        )
        .select(
            "charclass",
            "month",
            "class_avg_level_month",
            "class_num_players_month",
        )
        .unique(["charclass", "month"])
    )
    return df.join(_, left_on=["charclass", "previous_month"], right_on=["charclass", "month"], how="left")

query_data_with_class_features = add_class_features(query_data_with_player_features)
query_data_with_class_features

# %%
