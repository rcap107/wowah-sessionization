# %%
import polars as pl
# %%
df = pl.read_parquet("data/wowah_data_raw.parquet")
# %%
from skrub import TableReport

TableReport(df)
# %%
df.group_by("charclass").agg(pl.count()).sort("count", descending=True)
# %%
df = df.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
# %% 

 
stats_by_month = df.group_by("month").agg(pl.count(), pl.col("level").mean().alias("avg_level"), pl.col("level").median().alias("median_level")).sort("month")
# %%
import matplotlib.pyplot as plt

# Prepare data
stats_df = stats_by_month.to_pandas()
months = stats_df['month'].astype(str)
avg_levels = stats_df['avg_level']
counts = stats_df['count']

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot average level on primary axis
color = 'tab:blue'
ax1.set_xlabel('Month')
ax1.set_ylabel('Average Level', color=color)
ax1.bar(months, avg_levels, alpha=0.7, label='Average Level', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(months, rotation=45, ha='right')

# Create secondary axis for count
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Character Count', color=color)
ax2.plot(months, counts, color=color, marker='o', linewidth=2, markersize=8, label='Character Count')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and adjust layout
plt.title('Average Level and Character Count by Month', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()
# %%
# Plot median and average of "char" column by month
char_stats_by_month = df.group_by("month").agg(
    pl.col("char").mean().alias("avg_char"),
    pl.col("char").median().alias("median_char")
).sort("month")

# %%
char_stats_df = char_stats_by_month.to_pandas()
months = char_stats_df['month'].astype(str)
avg_char = char_stats_df['avg_char']
median_char = char_stats_df['median_char']

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

# Plot both lines
ax.plot(months, avg_char, marker='o', linewidth=2, markersize=8, label='Average', color='tab:blue')
ax.plot(months, median_char, marker='s', linewidth=2, markersize=8, label='Median', color='tab:orange')

# Customize plot
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Character Value', fontsize=12)
ax.set_title('Average and Median Character Value by Month', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
fig.tight_layout()
plt.show()
# %%
