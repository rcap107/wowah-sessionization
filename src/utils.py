import polars as pl
from skrub import deferred

@deferred
def add_session_features(df):
    return df.with_columns(
        pl.col("timestamp").first().over("timestamp_session_id").alias("session_start"),
        pl.col("timestamp").last().over("timestamp_session_id").alias("session_end"),
        pl.col("level")
        .max()
        .over("timestamp_session_id")
        .alias("session_max_level"),
        pl.col("level")
        .n_unique()
        .over("timestamp_session_id")
        .alias("session_levels_count"),
        pl.col("zone")
        .n_unique()
        .over("timestamp_session_id")
        .alias("session_zones_count"),
    )


@deferred
def add_char_features(df):
    return df.with_columns(
        pl.col("timestamp").first().over("char").alias("char_first_seen"),
        pl.col("timestamp").last().over("char").alias("char_last_seen"),
        pl.col("level").max().over("char").alias("max_level"),
        pl.col("zone").n_unique().over("char").alias("unique_zones_visited"),
        pl.col("timestamp_session_id").count().over("char").alias("sessions_per_char"),
        pl.col("timestamp_session_id")
        .count()
        .over("timestamp_session_id")
        .alias("session_duration"),
        pl.col("guild").n_unique().over("char").alias("guilds_joined"),
    )


@deferred
def add_aggregated_features(df):
    return df.with_columns(
        pl.col("level").mean().over("race").alias("agg_avg_level_by_race"),
        pl.col("level").mean().over("charclass").alias("agg_avg_level_by_class"),
        pl.col("level").mean().over("guild").alias("agg_avg_level_by_guild"),
        pl.col("level").mean().over("zone").alias("agg_avg_level_by_zone"),
        pl.col("charclass").count().over("charclass").alias("agg_count_by_charclass"),
        pl.col("race").count().over("race").alias("agg_count_by_race"),
        pl.col("char")
        .count()
        .over("race", "charclass")
        .alias("agg_count_by_race_charclass"),
    )


def sample_by_user(df, fraction=0.1):
    '''
    Sample a fraction of users (chars) and return all their data.
    '''
    data = df.filter(
        pl.col("char").is_in(
            df.select(pl.col("char").unique())
            .sample(fraction=fraction, seed=42)["char"]
            .implode()
        )
    )
    return data