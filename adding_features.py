# %%
# In this script I am testing the features that I can add to the historical data

import polars as pl
import datetime
import skrub
from src.utils import (
    sample_by_user,
    get_session_duration
)

from skrub._session_encoder import SessionEncoder


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

    


def add_fixed_features(df, historical_data):
    return df.join(
        historical_data.select("char", "race", "charclass").unique("char"),
        on="char",
        how="left",
    )


def add_class_features(df, hist_session_duration):
    # Add monthly class-based features
    _ = (
        hist_session_duration.with_columns(
            pl.col("level")
            .mean()
            .over(["charclass", "month"])
            .alias("monthly_class_avg_level"),
            pl.col("char")
            .n_unique()
            .over(["charclass", "month"])
            .alias("monthly_class_num_players"),
        )
        .select(
            "charclass",
            "month",
            "monthly_class_avg_level",
            "monthly_class_num_players",
        )
        .unique(["charclass", "month"])
    )
    return df.join(
        _,
        left_on=["charclass", "month"],
        right_on=["charclass", "month"],
        how="left",
    )


def add_session_features(df, hist_session_duration):
    monthly_duration = (
        hist_session_duration.unique("timestamp_session_id")
        .group_by("char", "month")
        .agg(
            pl.col("session_duration").sum().alias("monthly_total_session_duration"),
            pl.col("session_duration").mean().alias("monthly_avg_session_duration"),
            pl.col("session_duration").count().alias("monthly_num_sessions"),
        )
    )
    df = df.join(
        monthly_duration,
        left_on=["char", "month"],
        right_on=["char", "month"],
        how="left",
    )
    return df


def add_monthly_player_features(df, hist_session_duration):
    _ = hist_session_duration.group_by("char", "month").agg(
        pl.col("level").max().alias("monthly_max_level_month"),
        pl.col("zone").n_unique().alias("monthly_num_zones_month"),
        pl.col("zone").mode().first().alias("monthly_most_freq_zone_month"),
        pl.col("guild").n_unique().alias("monthly_num_guilds_month"),
        pl.col("guild").last().alias("monthly_last_guild_month"),
        pl.col("guild").first().alias("monthly_first_guild_month"),
    )
    return df.join(
        _,
        left_on=["char", "month"],
        right_on=["char", "month"],
        how="left",
    )


# %%
def adding_other_features(df, historical_data):
    hist_session_duration = get_session_duration(historical_data)
    df = add_session_features(df, hist_session_duration)
    df = add_monthly_player_features(df, hist_session_duration)
    df = add_class_features(df, hist_session_duration)
    return df

# %%
if __name__ == "main":
    df = pl.read_parquet("data/wowah_data_raw.parquet")
    df_user_month = pl.read_parquet("data/wowah_churn_data.parquet").select(
        "char", "month", "first_month"
    )
    # %%
    df_user_month = sample_by_user(df_user_month, fraction=0.1)
    df = df.with_columns(guild=pl.col("guild").replace(-1, None))
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

    session_encoder = SessionEncoder(
        group_by="char", timestamp_col="timestamp", session_gap=30
    )
    historical_data_with_sessions = session_encoder.fit_transform(historical_data)
    adding_other_features(query_data, historical_data_with_sessions)
    # %%
    get_session_duration(historical_data_with_sessions)