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
from datetime import datetime, timedelta

import polars as pl
import skrub
import skrub.selectors as s
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from skrub import ApplyToCols, DatetimeEncoder, SessionEncoder, TableVectorizer

from add_churn import make_user_month
from src.utils import (
    add_aggregated_features,
    add_char_features,
    add_session_features,
    sample_by_user,
)

# This needs to start in february to have at least one month of historical data
# before I can build features.
# I had to set the max date to november because otherwise I was getting
# ValueError: No valid specification of the columns.
# This is again because we are predicting on month N for month N+1
MIN_DATE = datetime.strptime("2008-02-01", "%Y-%m-%d")
MAX_DATE = datetime.strptime("2008-11-30", "%Y-%m-%d")
# Actual ranges for the full dataset
# MIN_DATE = datetime.strptime("2005-12-31", "%Y-%m-%d")
# MAX_DATE = datetime.strptime("2009-01-10", "%Y-%m-%d")


# The splitter iterates over the months and selects all the months up to the
# split point, which is the month in the current iteration.
class Splitter:
    def split(self, user_month, has_played=None):
        # Not needed in this splitter since we are only splitting based on the month
        del has_played
        time_range = pl.date_range(MIN_DATE, MAX_DATE, "1mo", eager=True)
        for split_point in time_range:
            # I can either use dateutils.relative delta
            # test_month = split_point + relativedelta(months=1)
            # Or do this with polars which is more consistent with the rest of the code
            test_month = pl.Series([split_point]).dt.offset_by("1mo").first()
            train_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") <= split_point)["idx"]
                .to_list()
            )
            test_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") == test_month)["idx"]
                .to_list()
            )
            if train_idx and test_idx:
                yield train_idx, test_idx


# This function is needed to make sure that we are only ever using historical data
# up to the given month - 1 month. This is to avoid any leakage in the data.
@skrub.deferred
def add_features(X, historical_data):
    features_by_month = []

    # Create a session encoder with a 30 minute timeout
    encoder = SessionEncoder(group_by="char", timestamp_col="timestamp", session_gap=30)

    last_month = X["month"].max().strftime("%Y-%m-%d")
    print("New split, filtering historical data up to month", last_month)
    for month in X["month"].unique():
        # I need to truncate the historical timestamp to month to be able to
        # compare it with the month in the target
        kept_historical_data = historical_data.with_columns(
            month=pl.col("timestamp").dt.truncate("1mo")
        ).filter(pl.col("month") < month)
        historical_data_with_sessions = encoder.fit_transform(kept_historical_data)
        # add_features
        features = pl.DataFrame({"month": [month]})
        features_by_month.append(features)

    # all_features = X
    # for features in features_by_month:
    #     all_features = all_features.join(features, on=["char", "month"], how="left")
    return X
    # return all_features


@skrub.deferred
def load(file):
    return pl.read_parquet(file)

def make_data_op():
    user_month_has_played = skrub.var("query")
    X = user_month_has_played["char", "month"].skb.mark_as_X(cv=Splitter())
    y = user_month_has_played["has_played"].skb.mark_as_y()
    historical_data_file = skrub.var("historical_data_file")
    historical_data = load(historical_data_file)
    all_features = add_features(X, historical_data)
    # historical_data_with_sessions = apply_session_encoder(kept_historical_data)
    # data_op = features.skb.apply(HGB(), y=y)
    data_op = all_features.skb.apply(DummyClassifier(), y=y)
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
df = sample_by_user(df, fraction=0.1)
data_op = make_data_op()
# %%
results = cross_validate()
# %%
evaluation_results = evaluate()
# %%
print(results)

# %%
