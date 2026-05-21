# %%
import datetime

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import skrub
from skrub._session_encoder import SessionEncoder

from src.utils import get_session_duration, sample_by_user

df = pl.read_parquet("data/wowah_data_raw.parquet")

df = sample_by_user(df)

# Filtering so that I'm taking only a single month, April

df = df.filter(pl.col("timestamp").dt.month() == 4)


# %%
# First I generate the regular session IDs for the user based only on the char ID
session_encoder = SessionEncoder(
    group_by="char", timestamp_col="timestamp", session_gap=30
)
df_with_sessions = session_encoder.fit_transform(df)

# I get the session duration and the total logged time by character
df_with_sessions = get_session_duration(df_with_sessions)
# Renaming the column because the SessionEncoder is overwriting columns
df_with_sessions = df_with_sessions.rename({"timestamp_session_id": "session_id"}).drop(
    "session_start", "session_end"
)
# Adding total logged time by user
total_logged_time = (
    df_with_sessions.unique("session_id")
    .select(
        pl.col("char"),
        pl.col("session_duration").sum().over("char").alias("total_logged_time"),
    )
    .unique("char")
)
total_logged_time
# %%
# Grouping by character and zone so that I can get the time spent in each zone
# Even if users leave the zone, this lets me find how much time a user spends in
# a given zone
session_encoder_zone = SessionEncoder(
    group_by=["char", "zone"], timestamp_col="timestamp", session_gap=30
)
df_zone_session = session_encoder_zone.fit_transform(df_with_sessions)
df_zone_session = get_session_duration(df_zone_session)
df_zone_session = df_zone_session.rename(
    {
        "session_duration": "zone_session_duration",
        "timestamp_session_id": "zone_session_id",
    }
).drop("session_start", "session_end")
df_zone_session


# %%
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
            .with_row_index(offset=1)
            .with_columns(
                is_hub=pl.when(pl.col("rarity") < pl.col("rarity").quantile(0.1))
                .then(True)
                .otherwise(False)
            )
            .drop("index")
        )
        .join(
            df.select(pl.col("zone"), pl.col("level").mean().over("zone")).unique(
                "zone"
            ),
            on="zone",
        )
        .rename({"level": "avg_level"})
    )

    return df_rarity


df_rarity = get_zone_rarity(df)
df_rarity.sort("rarity")


# %%
def add_player_rarity(df_with_sessions, df_rarity):
    """
    This function finds the average and max rarity of the locations a user visits,
    based on the overall rarity computed across the playerbase.

    High max rarity means that a user goes to rare (usually high-level) locations,
    high mean rarirty means they tend to spend more time out of hubs.

    The "in_hub" column tracks the fraction of time a player spends in a zone that
    is marked as "hub".
    """
    df_users_rarity = (
        df_with_sessions.join(df_rarity, on="zone")
        .select(
            pl.col("char"),
            pl.col("rarity").max().over("char").alias("max_rarity"),
            pl.col("rarity").mean().over("char").alias("mean_rarity"),
            (
                pl.col("is_hub").sum().over("char")
                / pl.col("is_hub").count().over("char")
            ).alias("in_hub"),
        )
        .unique("char")
    )
    return df_users_rarity


df_users_rarity = add_player_rarity(df_with_sessions, df_rarity)
df_users_rarity
# %%
# Weighted rarity rescales the time spent in each location by the location's rarity.
# If a location is rare, the time that is spent in the location is therefore increased,
# if a location is a hub, then the time is reduced.
#
# Then, I add the difference between the "real" time and the "adjusted" time, and
# the relative difference.
# I am not convinced this is a useful metric

df_weighted_rarity = (
    df_zone_session.join(df_rarity, on="zone")
    .with_columns(weighted_time=pl.col("rarity") * pl.col("zone_session_duration"))
    .unique("zone_session_id")
    .with_columns(sum_weighted_time=pl.col("weighted_time").sum().over("char"))
    .unique("char")
)
_ = (
    df_weighted_rarity.join(total_logged_time, on="char")
    .select("char", "total_logged_time", "sum_weighted_time")
    .with_columns(diff=pl.col("total_logged_time") - pl.col("sum_weighted_time"))
    .with_columns(diff_perc=pl.col("diff") / pl.col("total_logged_time"))
)

# %%
# Level-adjusted offset tracks situation in which a player is going to a location
# whose average level is different from their level. If the value is < 1, then
# the player is in a location whose average level is higher than the player level,
# if it is > 1, then the player is over-leveled compared to the zone.
#
# Applying the sum and aggregating shows that there are some players that spend
# the vast majority of their time in low-level zone.

below_70 = (
    df_zone_session.join(df_rarity, on="zone")
    .filter(pl.col("level") < 70)
    .with_columns(level_adj_offset=(pl.col("level") / pl.col("avg_level")).log())
    .select("char", pl.col("level_adj_offset").sum().over("char"))
)
above_70 = (
    df_zone_session.join(df_rarity, on="zone")
    .filter(pl.col("level") >= 70)
    .with_columns(level_adj_offset=(pl.col("level") / pl.col("avg_level")).log())
    .select("char", pl.col("level_adj_offset").sum().over("char"))
)


fig, ax = plt.subplots()
sns.histplot(data=below_70.to_pandas(), x="level_adj_offset", ax=ax)
sns.histplot(data=above_70.to_pandas(), x="level_adj_offset", ax=ax)


# %%
# Measuring the Gini coefficient of the time spent by location.


def gini(group: pl.Series):
    n = len(group)
    sorted = (
        group.sort("zone_session_duration")
        .with_columns(
            cumulative=pl.col("zone_session_duration").dt.total_minutes().cum_sum()
        )
        .with_columns(
            gini=(n + 1 - 2 * pl.col("cumulative").sum() / pl.col("cumulative").last())
            / n
        )
    )
    # r = n + 1 - 2 * cumulative.sum() / cumulative.last() / n
    return sorted.select("char", "gini").unique()


# %%
df_zone_session
# %%
df_with_gini = (
    df_zone_session.group_by("char", "zone")
    .agg(pl.sum("zone_session_duration"))
    .group_by("char")
    .map_groups(gini)
)
df_with_gini
# %%
for gidx, g in _.group_by("char"):
    d = g.pipe(gini)
    break
d
# %%
from skrub import TableReport

TableReport(_)
# %%
