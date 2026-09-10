# %%
import polars as pl
import matplotlib.pyplot as plt

# sns.set_context("talk")
plt.style.use("seaborn-v0_8-talk")
# plt.rc("font", family="serif")

df = pl.read_parquet("../data/wowah_churn_data.parquet")
# %%
df.group_by("month").agg(pl.len()).sort("month")
# %%
df_grouped = df.group_by("month").agg(pl.len()).sort("month")

fig, ax = plt.subplots()
ax.barh(df_grouped["month"].cast(str), df_grouped["len"])
plt.show()
# %%
df_grouped = df_grouped.with_columns(cumsum=pl.col("len").cum_sum())
# %%
fig, ax = plt.subplots()
ax.barh(df_grouped["month"].cast(str), df_grouped["cumsum"])
plt.show()
# %%
months = df_grouped["month"].dt.strftime("%Y-%m")
lens = df_grouped["len"].to_list()
lefts = [0] + df_grouped["cumsum"].to_list()#[:-1]  # left offset for each month's segment
n = len(months)

split_labels = [f"Split {i+1}" for i in range(n)]
cmap = plt.cm.viridis
colors = [cmap(j / max(n - 1, 1)) for j in range(n)]

fig, ax = plt.subplots(figsize=(10, 6))

for j in range(n):
    ax.barh(
        months[j:],
        [lens[j]] * (n - j),
        left=lefts[j],
        color=colors[j],
        label=split_labels[j],
    )

cumsums = df_grouped["cumsum"].to_list()
for i, total in enumerate(cumsums):
    diff = lens[i] if i == 0 else lefts[i+1] - lefts[i]
    sign = "+" if diff >= 0 else ""
    ax.text(total, months[i], f"  {total:,}\n ({sign}{diff:,})", va="center", fontsize=8)

ax.invert_yaxis()  # Split 1 at top for natural reading order
ax.set_xlabel("Number of characters in the training set (users in the current month)")
ax.set_ylabel("CV Split (Month)")
ax.set_title("Number of unique characters in each CV split")
handles, labels = ax.get_legend_handles_labels()
plt.legend(
    handles, labels,
    title="CV Split", loc="upper right",
    fontsize=8,
)

desc = """
The number of unique characters in the training set increases with each CV split, 
and the increase is not constant. Later splits are much larger (up to 3 times the
size of earlier splits).
"""


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# text box at top-centre; one arrow to first (smallest) split, one to last (largest)
ax.annotate(
    desc.strip(),
    xy=(0.1, 0.92),                 # arrow tip: near the short first-split bar
    xytext=(0.55, 0.80),             # text box: top-middle of the axes
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center", va="top", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", fc="lightgrey", ec="grey", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color="grey"),
)
ax.annotate(
    "",
    xy=(0.85, 0.2),                 # arrow tip: toward the long last-split bar
    xytext=(0.60, 0.71),             # bottom edge of the text box
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="grey"),
)

plt.tight_layout()
plt.show()
# %%
fig.savefig("user_months_plot.png")
# %%
