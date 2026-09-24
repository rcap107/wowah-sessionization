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
        maintain_order="left",
    )


def add_session_features(df, hist_session_duration):
    s_duration = pl.col("session_duration")

    monthly_duration = (
        hist_session_duration.unique("timestamp_session_id")
        .group_by("char", "month")
        .agg(
            s_duration.sum().alias("monthly_total_session_duration"),
            s_duration.mean().alias("monthly_avg_session_duration"),
            s_duration.std()
            .fill_null(pl.duration(seconds=0))
            .alias("monthly_std_session_duration"),
            s_duration.count().alias("monthly_num_sessions"),
            pl.col("session_start").mean().alias("monthly_avg_session_start"),
            pl.col("session_end").mean().alias("monthly_avg_session_end"),
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
        maintain_order="left",
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


def add_player_rarity(df):
    """
    This function finds the average and max rarity of the locations a user visits,
    based on the overall rarity computed across the playerbase.

    High max rarity means that a user goes to rare (usually high-level) locations,
    high mean rarity means they tend to spend more time out of hubs.

    The "in_hub" column tracks the fraction of time a player spends in a zone that
    is marked as "hub".
    """
    rarity = pl.col("rarity")
    is_hub = pl.col("is_hub")
    df_users_rarity = (
        df.lazy()
        .select(
            pl.col("char"),
            rarity.max().over("char").alias("max_rarity"),
            rarity.mean().over("char").alias("mean_rarity"),
            (
                is_hub.sum().over(
                    "char",
                )
                / is_hub.count().over("char")
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


def get_location_gini(df, with_hub=False):
    """
    with_hub allows to choose whether we want to compute gini with hubs (locations
    where everyone goes)

    players with low gini tend to stick to low-level/hub areas
    """
    df = df.collect()
    if df.is_empty():
        df_with_gini = df.select(
            pl.col("char"), pl.col("char").alias("gini").cast(pl.Float64)
        )
        return df_with_gini
    if not with_hub:
        groups = (
            df
            .filter(~pl.col("is_hub"))
            .group_by("char", "zone")
        )
    else:
        groups = df.group_by("char", "zone")
    df_with_gini = (
        groups.agg(pl.sum("session_duration")).group_by("char").map_groups(gini)
    )

    return df_with_gini.lazy()


def add_gini_features(df, historical_data_zones):
    df_with_gini = get_location_gini(historical_data_zones, with_hub=False)
    return df.join(
        df_with_gini.lazy(),
        on=[
            "char",
        ],
        how="left",
        maintain_order="left",
    )


def add_rarity_features(df, historical_data_zones):
    return df.join(
        add_player_rarity(historical_data_zones),
        on="char",
        how="left",
        maintain_order="left",
    )


# %%
def add_location_features(df, historical_data_zones, add_gini=False):
    """
    historical_data_zones contains user-zone sessions, historical_data_sessions
    contains the full sessions
    """
    # zone rarity is a useful indicator for various features
    location_rarity = get_zone_rarity(historical_data_zones.collect())
    historical_data_zones = historical_data_zones.join(
        location_rarity.lazy(), on="zone", how="left", maintain_order="left"
    )
    df = add_rarity_features(df.lazy(), historical_data_zones)
    if add_gini:
        df = add_gini_features(df.lazy(), historical_data_zones)
    return df.collect()


# %%
def add_general_features(df, historical_data):
    df = add_session_features(df, historical_data)
    df = add_monthly_player_features(df, historical_data)
    df = add_class_features(df, historical_data)
    return df


def add_lagged_features(
    df,
    lags=(1, 2),
    diff_lags=(1,),
    exclude_cols=(
        "char",
        "month",
        "race",
        "charclass",
        "index",
        "first_month",
        "has_played",
    ),
):
    """
    Add lagged versions of the per-character monthly features so that a model
    can learn how a character's behavior changes from month to month.

    For every feature column not in `exclude_cols`, this adds a `<col>_lag{k}`
    column for every k in `lags`, obtained by shifting the column by k periods
    within each character's timeline (ordered by month). For numeric columns,
    it also adds `<col>_diff{k}` columns holding the difference between the
    current value and the value k periods before, to capture trends directly.

    Rows corresponding to a character's first months have no history to lag
    from, so the resulting lag/diff columns are null for those rows; this is
    expected and should be handled downstream (e.g. by the vectorizer/imputer).
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = [c for c in feature_cols if df.schema[c].is_numeric()]

    df = df.sort(["char", "month"])

    lag_exprs = [
        pl.col(c).shift(lag).over("char").alias(f"{c}_lag{lag}")
        for c in feature_cols
        for lag in lags
    ]

    # Unsigned integer columns (the various monthly_num_* counters) wrap around
    # to huge positive values when a diff would be negative, so those need to
    # be cast to a signed type first. Float columns are left uncast to avoid
    # truncating fractional values (e.g. monthly_class_avg_level).
    diff_exprs = []
    for c in numeric_cols:
        col_expr = (
            pl.col(c).cast(pl.Int32)
            if df.schema[c].is_unsigned_integer()
            else pl.col(c)
        )
        for lag in diff_lags:
            diff_exprs.append(
                (col_expr - col_expr.shift(lag).over("char")).alias(f"{c}_diff{lag}")
            )

    return df.with_columns(lag_exprs + diff_exprs)


if __name__ == "__main__":
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    historical_data = pl.read_parquet("data/wowah_data_raw.parquet")
    fixed_attr = historical_data.select(pl.col("char", "charclass", "race")).unique()

    df = df.join(fixed_attr, on="char", how="left", maintain_order="left")

    session_encoder = SessionEncoder(
        split_by="char", timestamp_col="timestamp", session_gap=60 * 30
    )
    historical_data = historical_data.with_columns(
        month=pl.col("timestamp").dt.truncate("1mo")
    )
    historical_data = session_encoder.fit_transform(historical_data)
    historical_data = get_session_duration(historical_data)
    df = add_general_features(df, historical_data)

    location_rarity = get_zone_rarity(historical_data)
    # Grouping by character and zone so that I can get the time spent in each zone
    # Even if users leave the zone, this lets me find how much time a user spends in
    # a given zone
    session_encoder_zone = SessionEncoder(
        split_by=["char", "zone"],
        timestamp_col="timestamp",
        session_gap=60 * 30,
        suffix="zone",
    )
    # Zone-session features: a session lasts from the first time a character
    # enters a zone to the moment it leaves it
    # This is useful to get zone-specific features
    historical_data_zone_sessions = session_encoder_zone.fit_transform(historical_data)
    historical_data_zone_sessions = get_session_duration(historical_data_zone_sessions)
    df_with_features = add_location_features(
        df,
        historical_data_zone_sessions,
        add_gini=True,
    )
    skrub.TableReport(df_with_features.sort("char", "month")).open()
