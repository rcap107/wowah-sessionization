# %%
import polars as pl
import skrub
from skrub import SessionEncoder, TableReport
from main import build_feature_table, add_features
from datetime import datetime
MIN_DATE = datetime.strptime("2008-01-01", "%Y-%m-%d")
MAX_DATE = datetime.strptime("2008-06-30", "%Y-%m-%d")
#%% 
df = pl.read_parquet("data/wowah_churn_data.parquet")
historical_data = pl.read_parquet("data/wowah_data_raw.parquet")
# %%
feature_table = build_feature_table(historical_data)
r = add_features(df["char", "month"], feature_table)
# %%
TableReport(r.sort("char", "month"))
# %%
import skrub.selectors as s

sel = s.filter_names(lambda x: "total_session" in x) | s.cols("char", "month")
# %%
s.select(r, sel)
# %%
c = pl.col("monthly_num_sessions")
r.select((c).over("char"))
# %%
r.select((c))
# %%
df.filter(pl.col("month") >= MIN_DATE).filter(pl.col("month") <= MAX_DATE).select(pl.col("has_played").mean())
# %%