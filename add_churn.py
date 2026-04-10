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

This is the "churn" dataset, which we can then use to train a model. 
"""

# %%
import polars as pl

# %%
# Load the dataset
df = pl.scan_parquet("data/wowah_data_raw.parquet")
# %%
def make_user_month(df):
    months = pl.datetime_range(
        start=df.select(pl.col("timestamp").dt.truncate("1mo").min()).collect().item(),
        end=df.select(pl.col("timestamp").dt.truncate("1mo").max()).collect().item(),
        interval="1mo",
        closed="left",
        eager=True,
    )

    char_month = (
        df.with_columns(month_left=pl.col("timestamp").dt.truncate("1mo"))
        .select("char")
        .unique()
        .join(months.to_frame(name="month").lazy(), how="cross")
    )
    return char_month


user_month = make_user_month(df)


# %%
def make_data(df):
    data = (
        df.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
        .unique(subset=["char", "month"])
        .with_columns(pl.lit(True).alias("has_played"))
    )
    return data


data = make_data(df)
# %%


def add_churn(user_month, data):
    df_with_user_month = (
        user_month.join(
            data.select("char", "month", "has_played"), on=["char", "month"], how="left"
        )
        .with_columns(pl.col("has_played").fill_null(False))
        .select("char", "month", "has_played")
    )
    return df_with_user_month


# %%
churn_data = add_churn(user_month, data)
# %%
churn_data.collect().write_parquet("data/wowah_churn_data.parquet")
# %%
