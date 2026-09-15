# %%
import polars as pl
import skrub
from skrub import SessionEncoder

#%% 
df = pl.read_parquet("data/wowah_churn_data.parquet")
historical_data = pl.read_parquet("data/wowah_data_raw.parquet")
# %%
session_encoder = SessionEncoder(
    split_by="char", timestamp_col="timestamp", session_gap=30*60
)
historical_data = historical_data.with_columns(
    month=pl.col("timestamp").dt.truncate("1mo")
)
last_month = df["month"].max()
# %% 
historical_data_with_sessions = session_encoder.fit_transform(
    historical_data.filter(pl.col("month") <= last_month)
)
# %%