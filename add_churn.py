"""
This script is used to add a "target" column to the dataset.

The idea is that, given a month, we want to use all the data up to the last day
of the previous month to predict whether a user will churn in the next month.

So, if we are in the month of April, we want to use all the data up to March 31st
to predict whether a user will churn in May. We want to predict May because we
might want to take action in April to prevent the user from churning in May.

To do this, we need to create a dataset that has one row per user per month,
and a column that indicates whether the user has played in that month or not.

We need then to create a dataset with all the months in the range we have data for,
then we need to do the cross product with the unique characters.
Now we want to prepare the data we have so that the "has played" colum is True
if the user has played in that month, and False otherwise.

We can do this by finding all the unique combinations of user and month in the
original dataset, and then doing a left join with the "user month" dataset.
Any null values in the "has played" column will be filled with False, indicating
that the user did not play in that month.
Unique combinations of user and month include months prior to the first month
a player has appeared in the dataset, so they need to be filtered out before
creating the final dataset.

This is the "churn" dataset, which we can then use to train a model.
"""

# %%
import polars as pl


def make_user_month(df):
    """
    Create a DataFrame with all unique combinations of users and months.
    This will be used to ensure that we have a row for each user for each month
    in the range of the dataset, even if the user did not play in that month.

    This is done with a cross-product between the unique users and the range of months.

    Note that since the months are generated from the range of the dataset, they
    may include months prior to a user's first activity. These will need to be
    filtered out later.
    """

    months = pl.datetime_range(
        start=df.select(pl.col("timestamp").dt.truncate("1mo").min()).collect().item(),
        end=df.select(pl.col("timestamp").dt.truncate("1mo").max()).collect().item(),
        interval="1mo",
        closed="both",
        eager=True,
    )

    char_month = (
        df.with_columns(month_left=pl.col("timestamp").dt.truncate("1mo"))
        .select("char")
        .unique()
        .join(months.to_frame(name="month").lazy(), how="cross")
    )
    return char_month


def make_data(df):
    """
    Prepare a dataframe that contains the months in which each player has
    actually played. The column "has_played" is then set to True for those months.
    Any month in which the player did not play is missing from this dataset,
    because the dataset is built starting from player activity, so any "empty
    month" simply doesn't exist in this dataset.
    """

    data = (
        df.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
        .unique(subset=["char", "month"])
        .with_columns(pl.lit(True).alias("has_played"))
    )
    return data


def add_churn(user_month, data):
    """
    Add churn information to the user-month DataFrame by merging it with the actual
    play data. The resulting DataFrame will have a "has_played" column indicating
    whether the user played in that month. Any missing values in "has_played" are
    filled with False: missing values mean that the player did not play in that
    particular month.
    """

    df_with_user_month = (
        user_month.join(
            data.select(
                "char",
                "month",
                "has_played",
                "first_month",
            ),
            on=["char", "month"],
            how="left",
        )
        .with_columns(pl.col("has_played").fill_null(False))
        .select(
            "char",
            "month",
            "has_played",
            "first_month",
        )
    )
    return df_with_user_month


def remove_unrealistic_entries(churn_data, data):
    """
    Remove entries from the churn data where the month is before the first month
    seen for the character. This ensures that we do not have rows for months that
    are unrealistic given the player's activity history.
    """
    churn_data = (
        churn_data.join(
            data.select("char", "first_month").unique(), on="char", how="left"
        )
        .filter(pl.col("month") >= pl.col("first_month_right"))
        .drop("first_month")
        .select(
            pl.col("char"),
            pl.col("month"),
            pl.col("has_played"),
            pl.col("first_month_right").alias("first_month"),
        )
    )
    return churn_data


def build_churn_dataset():
    # Load the dataset
    df = pl.scan_parquet("data/wowah_data_raw.parquet")
    df = df.with_columns(
        first_month=pl.col("timestamp").dt.truncate("1mo").min().over("char")
    )
    #
    data = make_data(df)
    # Add the combinations of user months
    user_month = make_user_month(df)
    churn_data = add_churn(user_month, data)
    churn_data = remove_unrealistic_entries(churn_data, data)
    return churn_data.collect()


if __name__ == "__main__":
    churn_data = build_churn_dataset()
    churn_data.write_parquet("data/wowah_churn_data.parquet")
