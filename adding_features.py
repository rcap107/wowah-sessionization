# %%
# In this script I am testing the features that I can add to the historical data

import polars as pl
import datetime
import skrub
from src.utils import sample_by_user, get_session_duration

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
        maintain_order="left",
    )
    return df


# TODO: check joins for maintain_order


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
        maintain_order="left"
    )


def get_zone_rarity(df):
    """
    This function prepares a dataframe that contains the relative rarity of each zone.

    The rarity is computed as log(N/n_v) where N is the number of unique players
    and n_v is the number of unique visitors. This gives high rarity to places that
    have been visited by fewer players.

    Then, columns are marked as "hub" or not in the "is_hub" column. If the rarity
    of a column is lower than the 10th quantile, then it's marked as "hub", i.e.,
    a lot of players go to this location.

    Then, the average player level of each zone is added.
    """

    n_unique_characters = df.n_unique("char")
    df_rarity = (
        (
            df.group_by("zone")
            .agg(unique_visitors=pl.col("char").n_unique())
            .with_columns(
                rarity=(n_unique_characters / pl.col("unique_visitors")).log()
            )
            .sort("rarity", descending=False)
            .with_columns(
                is_hub=pl.when(pl.col("rarity") < pl.col("rarity").quantile(0.1))
                .then(True)
                .otherwise(False)
            )
        )
        .join(
            df.select(pl.col("zone"), pl.col("level").mean().over("zone")).unique(
                "zone"
            ),
            on="zone",
        )
        .rename({"level": "zone_avg_level"})
    )

    return df_rarity


def add_player_rarity(df, df_rarity):
    """
    This function finds the average and max rarity of the locations a user visits,
    based on the overall rarity computed across the playerbase.

    High max rarity means that a user goes to rare (usually high-level) locations,
    high mean rarity means they tend to spend more time out of hubs.

    The "in_hub" column tracks the fraction of time a player spends in a zone that
    is marked as "hub".
    """
    df_users_rarity = (
        df.join(df_rarity, on="zone", how="left", maintain_order="left").select(
            pl.col("char"),
            pl.col("rarity")
            .max()
            .over(
                "char",
            )
            .alias("max_rarity"),
            pl.col("rarity")
            .mean()
            .over(
                "char",
            )
            .alias("mean_rarity"),
            (
                pl.col("is_hub")
                .sum()
                .over(
                    "char",
                )
                / pl.col("is_hub")
                .count()
                .over(
                    "char",
                )
            ).alias("in_hub"),
        )
    ).unique("char")
    return df_users_rarity


# Measuring the Gini coefficient of the time spent by location. This metric shows
# the distribution of time spent by a user across different locations.
# The idea is that if a user visits a lot of different for a (somewhat) equal length
# of time, they are more likely to be a "casual explorer", while if a player spends
# a very large fraction of their time in a small number of locations they are more
# likely to be "grinding" specific locations.
#
# This is interesting to compare with the average rarity of the locations that
# each player visits.
#
# The coefficient is measured by finding the amount of time a user spends in each
# location in a month.
#
# Some factors I might want to consider:
# - Calculate the Gini coefficient only for non-hub locations
# - Filter out low-playtime players


def gini(group: pl.DataFrame):
    n = len(group)
    sorted = (
        group.sort("session_duration")
        .with_columns(
            cumulative=pl.col("session_duration").dt.total_minutes().cum_sum()
        )
        .with_columns(
            gini=(n + 1 - 2 * pl.col("cumulative").sum() / pl.col("cumulative").last())
            / n
        )
    )
    return sorted.select("char", "gini").unique()


def get_location_gini(df, df_rarity, with_hub=False):
    """
    with_hub allows to choose whether we want to compute gini with hubs (locations
    where everyone goes)

    players with low gini tend to stick to low-level/hub areas
    """

    if df_rarity.is_empty():
        df_with_gini = df.select(pl.col("char"), pl.col("char").alias("gini").cast(pl.Float64))
        return df_with_gini
    if not with_hub:
        groups = (
            df.join(df_rarity, on="zone")
            .filter(~pl.col("is_hub"))
            .group_by("char", "zone")
        )
    else:
        groups = df.join(df_rarity, on="zone").group_by("char", "zone")
    df_with_gini = (
        groups.agg(pl.sum("session_duration")).group_by("char").map_groups(gini)
    )

    return df_with_gini


def add_gini_features(df, historical_data_zones, df_rarity):
    df_with_gini = get_location_gini(historical_data_zones, df_rarity, with_hub=False)
    return df.join(
        df_with_gini,
        on=[
            "char",
        ],
        how="left",
    )


def add_rarity_features(df, historical_data_zones, location_rarity):
    df = df.join(
        add_player_rarity(historical_data_zones, location_rarity),
        on="char",
        how="left",
        maintain_order="left",
    )
    return df


# %%
def add_location_features(df, historical_data_zones):
    """
    historical_data_zones contains user-zone sessions, historical_data_sessions
    contains the full sessions
    """
    # zone rarity is a useful indicator for various features
    location_rarity = get_zone_rarity(historical_data_zones)
    df = add_rarity_features(df, historical_data_zones, location_rarity)
    df = add_gini_features(df, historical_data_zones, location_rarity)
    return df


# %%
def add_general_features(df, historical_data):
    df = add_session_features(df, historical_data)
    df = add_monthly_player_features(df, historical_data)
    df = add_class_features(df, historical_data)
    return df


# %%
if __name__ == "main":
    df = pl.read_parquet("data/wowah_data_raw.parquet")
    df_user_month = pl.read_parquet("data/wowah_churn_data.parquet").select(
        "char", "month", "first_month"
    )
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

    session_encoder = SessionEncoder(
        group_by="char", timestamp_col="timestamp", session_gap=30
    )
    historical_data_with_sessions = session_encoder.fit_transform(historical_data)
    add_general_features(query_data, historical_data_with_sessions)
    get_session_duration(historical_data_with_sessions)
