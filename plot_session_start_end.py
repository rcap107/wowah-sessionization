'''
Plot the distribution of session start and end
'''
# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from skrub import SessionEncoder

from src.utils import (
    sample_by_user,
)

df = pl.read_parquet("data/wowah_data_raw.parquet")
df_user_month = pl.read_parquet("data/wowah_churn_data.parquet").select(
    "char", "month", "first_month"
)
# %%
df_user_month = sample_by_user(df_user_month, fraction=0.1)
df = df.with_columns(guild=pl.col("guild").replace(-1, None))
# %%
target_month = pl.datetime(2008, 2, 1)
historical_data = df.with_columns(
    month=pl.col("timestamp").dt.truncate("1mo")
)
query_data = df_user_month.filter(
    (pl.col("month") == target_month) & (pl.col("first_month") < target_month)
)
# %%

session_encoder = SessionEncoder(
    group_by="char", timestamp_col="timestamp", session_gap=30
)
historical_data_with_sessions = session_encoder.fit_transform(historical_data)

_ = historical_data_with_sessions.with_columns(
    pl.col("timestamp").first().over("timestamp_session_id").alias("session_start"),
    pl.col("timestamp").last().over("timestamp_session_id").alias("session_end"),
).with_columns(
    session_duration=pl.col("session_end") - pl.col("session_start"),
)
session_bounds = _.select(
    "char", "session_start", "session_end", "timestamp_session_id"
).unique("timestamp_session_id")
# %%
_.unique("timestamp_session_id").group_by("char").agg(
    pl.col("session_duration").sum().alias("total_session_duration"),
    pl.col("session_duration").mean().alias("avg_session_duration"),
).sort("total_session_duration", descending=True)
# %%

BIN_MINUTES = 15  # change this to adjust bin size (e.g. 60 for hourly, 15 for 15-min)
n_bins = 24 * 60 // BIN_MINUTES
bins = np.arange(n_bins)
theta = bins / n_bins * 2 * np.pi
width = 2 * np.pi / n_bins


def compute_bin_counts(col):
    return (
        session_bounds.with_columns(
            bin=(
                pl.col(col).dt.hour().cast(pl.Int16) * 60
                + pl.col(col).dt.minute().cast(pl.Int16)
            )
            // BIN_MINUTES
        )
        .group_by("bin")
        .len()
        .sort("bin")
    )


def counts_for_bins(counts_df):
    bin_to_count = dict(zip(counts_df["bin"].to_list(), counts_df["len"].to_list()))
    return np.array([bin_to_count.get(b, 0) for b in bins])


start_vals = counts_for_bins(compute_bin_counts("session_start"))
end_vals = counts_for_bins(compute_bin_counts("session_end"))

# Show a tick label every full hour, regardless of bin size
tick_step = 60 // BIN_MINUTES  # bins per hour
tick_bins = bins[::tick_step]
tick_theta = theta[::tick_step]
tick_labels = [f"{h:02d}h" for h in range(24)]

fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 6))

for ax, vals, title in zip(
    axes, [start_vals, end_vals], ["Session Starts", "Session Ends"]
):
    ax.bar(theta, vals, width=width, align="center", alpha=0.8)
    ax.set_theta_zero_location("N")  # hour 0 at the top
    ax.set_theta_direction(-1)  # clockwise, like a clock
    ax.set_xticks(tick_theta)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_title(title, pad=15)

fig.suptitle("Distribution of Session Start/End Times", fontsize=13)
plt.tight_layout()
plt.show()

# %%
