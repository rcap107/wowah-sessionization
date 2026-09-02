# %%
import numpy as np
import datetime

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import skrub
from skrub._session_encoder import SessionEncoder

from src.utils import get_session_duration, sample_by_user

df = pl.read_parquet("data/wowah_data_raw.parquet")

# df = sample_by_user(df)

# Filtering so that I'm taking only a single month, April

df = df.filter(pl.col("timestamp").dt.month() == 4)


# %%
# First I generate the regular session IDs for the user based only on the char ID
session_encoder = SessionEncoder(
    split_by="char", timestamp_col="timestamp", session_gap=30
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
    high mean rarity means they tend to spend more time out of hubs.

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
        group.sort("zone_session_duration")
        .with_columns(
            cumulative=pl.col("zone_session_duration").dt.total_minutes().cum_sum()
        )
        .with_columns(
            gini=(n + 1 - 2 * pl.col("cumulative").sum() / pl.col("cumulative").last())
            / n
        )
    )
    return sorted.select("char", "gini").unique()


def get_location_gini(df):
    df_with_gini = (
        df.group_by("char", "zone")
        .agg(pl.sum("zone_session_duration"))
        .group_by("char")
        .map_groups(gini)
    )
    return df_with_gini


# %%
# Comparing the distribution of Gini scores with and without hubs
df_gini_hub = get_location_gini(df_zone_session.join(df_rarity, on="zone"))
df_gini_nohub = get_location_gini(
    df_zone_session.join(df_rarity, on="zone").filter(~pl.col("is_hub"))
)
fig, ax = plt.subplots()
sns.histplot(data=df_gini_hub.to_pandas(), x="gini", ax=ax, label="with hubs")
sns.histplot(data=df_gini_nohub.to_pandas(), x="gini", ax=ax, label="no hubs")
ax.legend()

# %%
# Comparing the distribution of Gini scores for players that have at least
# 10 total hours over the month, against all players
players_10h = df_zone_session.join(total_logged_time, on="char").filter(
    pl.col("total_logged_time").dt.total_hours() > 10
)
df_gini_10h = get_location_gini(players_10h.join(df_rarity, on="zone"))
df_gini_all = get_location_gini(df_zone_session.join(df_rarity, on="zone"))

fig, ax = plt.subplots()
sns.histplot(
    data=df_gini_all.to_pandas(), x="gini", ax=ax, label="all players", bins=20
)
sns.histplot(
    data=df_gini_10h.to_pandas(), x="gini", ax=ax, label=">10h played", bins=20
)
ax.legend()


# %%
# Getting the fraction of time that is spent in the top 3 locations, and the
# fraction that is spent in top-3 non-hub locations.

df_time_in_hub = (
    df_zone_session.group_by("char", "zone")
    .agg(pl.sum("zone_session_duration"))
    .join(df_rarity, on="zone")
    .with_columns(
        top_3_time=pl.col("zone_session_duration").top_k(3).sum().over("char")
        / pl.col("zone_session_duration").sum().over("char"),
        top_3_time_nohub=pl.col("zone_session_duration")
        .filter(~pl.col("is_hub"))
        .top_k(3)
        .sum()
        .over("char")
        / pl.col("zone_session_duration").sum().over("char"),
    )
    .select("char", "top_3_time_nohub", "top_3_time")
    .unique("char")
)

# %%
# Getting the variability in sessions.
# I'm getting the standard deviation of the start and end hours after converting
# them to radians (so that 23 is close to 0), and the start and end day.
# High variance in the hour of day/day of week means that the user connects at
# random times, low variance means that the player tends to play at the same time
# every day (more dedicated).
# High variance in the session duration may mean that the player has periods where
# they play for a long time (grinding or events), low variance players have a
# consistent schedule.
#
# Entropy can be used to measure how "unpredictable" a player is depending on their
# behavior. It can be temporal, spatial, or behavioral.
#
# Temporal entropy: hour of day, day of week
# - the player logs in at 8pm every day -> low entropy, the player has a habit,
# the player is likely to be dedicated -> likely high retention
# - the player logs in at random times every day -> high entropy, the player connects
# whenever they have some time -> likely to not be very dedicated -> low retention
#
# Spatial entropy: locations
# - the player spends most of their time in few locations -> low entropy, likely
# the player is grinding a specific location and is optimizing their gameplay ->
# dedicated player -> high retention
# - the player spends 10% of their time in 10 different locations -> high entropy
# the player is likely exploring or hasn't found an efficient place to grind

def get_temporal_variance(df):
    df_std = (
        df.with_columns(
            start_hour=pl.col("timestamp").min().over("session_id").dt.hour(),
            end_hour=pl.col("timestamp").max().over("session_id").dt.hour(),
            session_start=pl.col("timestamp").min().over("session_id").dt.hour()
            * 2
            * np.pi
            / 24,
            session_end=pl.col("timestamp").max().over("session_id").dt.hour()
            * 2
            * np.pi
            / 24,
            session_start_day=pl.col("timestamp").min().over("session_id").dt.day(),
            session_end_day=pl.col("timestamp").max().over("session_id").dt.day(),
        )
        .group_by("char")
        .agg(
            pl.col("session_start").mean().alias("avg_hour_start"),
            pl.col("session_start").std().alias("std_hour_start"),
            pl.col("session_end").mean().alias("avg_hour_end"),
            pl.col("session_end").std().alias("std_hour_end"),
            pl.col("start_hour").entropy().alias("entropy_hour_start"),
            pl.col("end_hour").entropy().alias("entropy_hour_end"),
            pl.col("session_start_day").std().alias("std_day_start"),
            pl.col("session_end_day").std().alias("std_day_end"),
            pl.col("session_duration").dt.total_minutes().mean().alias("avg_duration"),
            pl.col("session_duration").dt.total_minutes().std().alias("std_duration"),
            pl.col("session_duration")
            .dt.total_minutes()
            .entropy()
            .alias("entropy_duration"),
        )
        .sort("char")
    )
    return df_std
    
df_std = get_temporal_variance(df_with_sessions)
df_std

# %%
# Spatial entropy
# Measuring the spatial entropy of each user. This is done by taking the time spent
# in each location by each user, then measuring the entropy of the distribution. 
# I'm measuring the entropy of the distribution rather than the entropy of the 
# number of visits, which is liable to being skewed by very short visits. 
#
# I am filtering out hubs and visits with duration = 1min (i.e., single heartbeats).
#
# I am also normalizing the entropy by the number of zones a user has visited to 
# have a measure of how much of the time has been spent in a single location (low
# normalized entropy -> focusing on a single location).  

def get_spatial_entropy(df, df_rarity):

    df_entropy = df.group_by("char", "zone").agg(pl.sum("zone_session_duration")).filter(
        pl.col("zone_session_duration").dt.total_minutes() > 1,
        ~pl.col("zone").is_in(df_rarity.filter(is_hub=True)["zone"].implode())
    ).with_columns(
        p_zone=pl.col("zone_session_duration").dt.total_minutes()
        / pl.col("zone_session_duration").sum().over("char").dt.total_minutes()
    ).with_columns(
        entropy=pl.col("p_zone").entropy(base=2, normalize=False).over("char")
    ).with_columns(
        normalized_entropy=pl.col("entropy")
        / pl.when(pl.col("zone").n_unique() > 1)
        .then(pl.col("zone").n_unique().log(2))
        .otherwise(1),
        count_zones=pl.col("zone").count().over("char"),
    ).drop("zone", "zone_session_duration", "p_zone").unique("char").sort(
        "normalized_entropy", descending=True
    )
    return df_entropy   

df_entropy = get_spatial_entropy(df_zone_session, df_rarity)
df_entropy
# %%
data = (
    df_users_rarity.join(df_gini_10h, on="char")
    .join(df_time_in_hub, on="char")
    .join(df_std, on="char")
    .join(df_entropy, on="char")
)
# %%
from sklearn.cluster import HDBSCAN, KMeans

# c = KMeans(n_clusters=8)
c = HDBSCAN(min_cluster_size=5)
c.fit(
    data[
        "gini",
        "mean_rarity",
        # "std_hour_start",
        # "std_hour_end",
        # "std_duration",
        # "std_day_start",
        "normalized_entropy",
        "entropy_duration"
    ]
)
labels = c.labels_

data = data.with_columns(labels=pl.Series(labels))

fig, ax = plt.subplots()
g = sns.scatterplot(
    data=data.to_pandas(),
    x="normalized_entropy",
    y="gini",
    hue="labels",
    palette="tab10",
    ax=ax,
)

# ax.set_xscale("log")
# %%
fig, ax = plt.subplots()
g = sns.scatterplot(
    data=data.filter(~pl.col("labels").is_in([-1, 6])).to_pandas(),
    x="entropy_duration",
    y="gini",
    hue="labels",
    palette="tab10",
    ax=ax,
)


# %%
