import polars as pl

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

    
    
def get_session_duration(df, session_column="timestamp_session_id"):
    # Adding the session start and end to find the session duration
    # Sessions that end within a single heartbeat have the same start and end, thus
    # duration = 0. I will replace those with a duration of 1 minute so that the
    # total logged time over a month is not 0. This is useful to distinguish between
    # players that never logged in and players that logged in but had very short sessions.
    _ = (
        df.with_columns(
            # Get the start of the session
            pl.col("timestamp")
            .first()
            .over(session_column)
            .alias("session_start"),
            # Get the end of the session
            pl.col("timestamp")
            .last()
            .over(session_column)
            .alias("session_end"),
        )
        .with_columns(
            # This is the "session_duration"
            (pl.col("session_end") - pl.col("session_start")).alias("session_duration"),
        )
        .with_columns(
            # If the duration of the session is 0, set it to 1 minute (arbitrary value)
            pl.when(pl.col("session_duration") == pl.duration(microseconds=0))
            .then(pl.duration(minutes=1))
            .otherwise(pl.col("session_duration"))
            .alias("session_duration")
        )
    )
    return _
