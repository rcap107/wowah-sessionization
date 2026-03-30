"""
This script is used to add a "target" column to the dataset. The target is a
binary variable that checks whether a user has churned or not. A user is considered to have churned if
their last month of activity is followed by a month of inactivity. The resulting dataset is saved as a parquet file.
"""

# %%
import polars as pl

# %%
# Load the dataset
df = pl.read_parquet("data/wowah_data_raw.parquet")
# %%
from skrub import TableReport

TableReport(df)
# %%
months = pl.datetime_range(
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    interval="1mo",
    closed="left",
)
# %%
df_with_user_month = df.with_columns(month=pl.col("timestamp").dt.truncate("1mo"))
# %%

_ = df_with_user_month.join(
    df_with_user_month.group_by("char").agg(last_month=pl.max("month")), on="char"
)

# %%
# A user is considered to have churned if their last month of activity is followed by a
# month of inactivity.
# So we check if the difference between the last month and the current month is 0 months:
# That means that the last month is the same as the current month, meaning
# it's the final month the user has been active.
# Since the dataset is for a single year, if the last month is december then we
# don't consider the user to have churned. In other cases, if the difference between
# the last month and the current month is 0, then we consider the user to have churned.


_.select("char", "month", "last_month").with_columns(
    diff=pl.col("month").dt.month() - pl.col("last_month").dt.month()
).with_columns(
    pl.when((pl.col("diff") == 0) & (pl.col("month").dt.month() != 12))
    .then(1)
    .otherwise(0)
    .alias("churned")
).drop("diff", "last_month").write_parquet("data/wowah_user_month_with_churn.parquet")

# %%
