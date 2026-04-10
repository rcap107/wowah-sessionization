"""
This script is used to build the predictive pipeline that is used to predict
user churn. The objective is to predict, for each user and month, if the
user will churn in the next month or not.

We need to be careful with splitting the data and avoid having leakage in the
data and the target. We need to define a splitter that iterates by month, and
we need to make sure that, when we build the features for a given month,
we only use data from previous months.

"""

# %%
import polars as pl
from add_churn import make_user_month
import skrub
from skrub import SessionEncoder
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.dummy import DummyClassifier
from datetime import datetime, timedelta
from skrub import TableVectorizer, DatetimeEncoder, ApplyToCols
import skrub.selectors as s

# This needs to start in february to have at least one month of
MIN_DATE = datetime.strptime("2008-02-01", "%Y-%m-%d")
# I had to set the max date to november because otherwise I was getting
# ValueError: No valid specification of the columns.
MAX_DATE = datetime.strptime("2008-11-30", "%Y-%m-%d")


# MIN_DATE = datetime.strptime("2005-12-31", "%Y-%m-%d")
# MAX_DATE = datetime.strptime("2009-01-10", "%Y-%m-%d")
class Splitter:
    def split(self, user_month, has_played):
        time_range = pl.date_range(MIN_DATE, MAX_DATE, "1mo", eager=True)
        for split_point in time_range:
            from dateutil.relativedelta import relativedelta
            
            test_month = split_point + relativedelta(months=1)
            train_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") == split_point)["idx"]
                .to_list()
            )
            test_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") == test_month)["idx"]
                .to_list()
            )

            yield train_idx, test_idx


@skrub.deferred
def filter_past(X, historical_data):
    assert X["month"].n_unique() == 1
    assert X["char"].n_unique() == X.shape[0]
    historical_data = historical_data.with_columns(
        month=pl.col("timestamp").dt.truncate("1mo")
    )
    
    return historical_data.filter(
        pl.col("month") < (X["month"][0] - timedelta(days=30))
    )


@skrub.deferred
def load(file):
    return pl.read_parquet(file)


@skrub.deferred
def add_session_features(df):
    return df.with_columns(
        pl.col("timestamp").first().over("timestamp_session_id").alias("session_start"),
        pl.col("timestamp").last().over("timestamp_session_id").alias("session_end"),
        pl.col("level")
        .max()
        .over("timestamp_session_id")
        .alias("max_level_in_session"),
        pl.col("level")
        .n_unique()
        .over("timestamp_session_id")
        .alias("levels_in_session"),
        pl.col("zone")
        .n_unique()
        .over("timestamp_session_id")
        .alias("zones_in_session"),
    )


@skrub.deferred
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


@skrub.deferred
def add_aggregated_features(df):
    return df.with_columns(
        pl.col("level").mean().over("race").alias("avg_level_by_race"),
        pl.col("level").mean().over("charclass").alias("avg_level_by_class"),
        pl.col("level").mean().over("guild").alias("avg_level_by_guild"),
        pl.col("level").mean().over("zone").alias("avg_level_by_zone"),
        pl.col("charclass").count().over("charclass").alias("count_by_charclass"),
        pl.col("race").count().over("race").alias("count_by_race"),
        pl.col("char")
        .count()
        .over("race", "charclass")
        .alias("count_by_race_charclass"),
    )



def add_features(X, historical_data):
    timestamp_encoder = ApplyToCols(
        DatetimeEncoder(periodic_encoding="circular"), keep_original=True, cols="timestamp"
    )
    data_vectorized = (
        historical_data.skb.apply(
            TableVectorizer(n_jobs=-1), exclude_cols=["char", "timestamp"]
        )
        .skb.apply(timestamp_encoder)
        .skb.set_name("data_vectorized")
    )
    # After vectorization, I'm aggregating the data by month since I have the
    # churn labels by month. I'm taking the mean of the features for each month
    # Here we could go with a choose_from and different agg functions
    data_monthly = (
        data_vectorized.with_columns(month=pl.col("timestamp").dt.truncate("1mo"))
        .group_by("char", "month")
        .agg(pl.all().mean())
    )

    data_monthly_with_churn = data_monthly.join(X, on=["char", "month"], how="left")
    return data_monthly_with_churn

def apply_session_encoder(data_op):
    # Create a session encoder with a 30 minute timeout
    encoder = SessionEncoder(group_by="char", timestamp_col="timestamp", session_gap=30)

    data_with_sessions = data_op.skb.apply(encoder)
    return data_with_sessions

def make_data_op():
    user_month_has_played = skrub.var("query")
    X = user_month_has_played["char", "month"].skb.mark_as_X(cv=Splitter())
    y = user_month_has_played["has_played"].skb.mark_as_y()
    historical_data_file = skrub.var("historical_data_file")
    historical_data = load(historical_data_file)
    kept_historical_data = filter_past(X, historical_data)
    historical_data_with_sessions = apply_session_encoder(kept_historical_data)
    features = add_features(X, historical_data_with_sessions)
    # data_op = features.skb.apply(HGB(), y=y)
    data_op = features.skb.apply(DummyClassifier(), y=y)
    return data_op


def cross_validate():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    historical_data_file = "data/wowah_data_raw.parquet"
    results = make_data_op().skb.cross_validate(
        {"query": df, "historical_data_file": historical_data_file}
    )
    return results

def evaluate():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    historical_data_file = "data/wowah_data_raw.parquet"
    results = make_data_op().skb.eval(
        {"query": df, "historical_data_file": historical_data_file}
    )
    return results

# %%

df = pl.read_parquet("data/wowah_churn_data.parquet")
data_op = make_data_op()
# %%
results = cross_validate()
# %%
evaluation_results = evaluate() 
# %%
