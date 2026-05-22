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
import polars.selectors as cs
import skrub
import skrub.selectors as s
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from skrub import ApplyToCols, DatetimeEncoder, SessionEncoder, TableVectorizer
from sklearn.impute import SimpleImputer

from add_churn import make_user_month
from src.utils import (
    add_aggregated_features,
    add_char_features,
    add_session_features,
    sample_by_user,
)
from adding_features import adding_other_features

# This needs to start in February to have one month of historical data and one month
# of break before I can build features.
# I had to set the max date to november because otherwise I was getting
# ValueError: No valid specification of the columns.
# This is again because we are predicting on month N for month N+1
MIN_DATE = datetime.strptime("2008-02-01", "%Y-%m-%d")
MAX_DATE = datetime.strptime("2008-11-30", "%Y-%m-%d")
# Actual ranges for the full dataset
# MIN_DATE = datetime.strptime("2005-12-31", "%Y-%m-%d")
# MAX_DATE = datetime.strptime("2009-01-10", "%Y-%m-%d")


# The splitter iterates over the months and selects all the months up to the
# split point, which is the month during which we want to perform some operation
# on users that are marked as "churn risks".
class Splitter:
    def split(self, user_month, has_played=None):
        # has_played is not needed in this splitter since we are only splitting
        # based on the month
        del has_played
        time_range = pl.date_range(MIN_DATE, MAX_DATE, "1mo", eager=True)
        for split_point in time_range:
            # I can either use dateutils.relative delta
            # test_month = split_point + relativedelta(months=1)
            # Or do this with polars which is more consistent with the rest of the code
            test_month = pl.Series([split_point]).dt.offset_by("1mo").first()
            # Train indices are up to split_point excluded
            train_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") <= split_point)["idx"]
                .to_list()
            )
            # Test indices are for the month after split_point
            test_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") == test_month)["idx"]
                .to_list()
            )
            if train_idx and test_idx:
                print("split_point ", split_point)
                yield train_idx, test_idx


def filter_df_by_month(df, month):
    return df.filter(pl.col("month") == month)


# This function is needed to make sure that we are only ever using historical data
# up to the given month - 1 month. This is to avoid any leakage in the data.
@skrub.deferred
def add_features(X, historical_data):
    features_by_month = []

    # Create a session encoder with a 30 minute timeout
    # This encoder is used as a stateless transformer so it is refitted for every
    # month
    encoder = SessionEncoder(group_by="char", timestamp_col="timestamp", session_gap=30)
    historical_data = historical_data.with_columns(
        month=pl.col("timestamp").dt.truncate("1mo")
    )
    last_month = X["month"].max()

    # Adding fixed features: these features are fixed by character so they don't
    # change over time.
    # historical_data is selected up until the last month because if I select only
    # a single month then any character that did not play in that month will be
    # missing those features
    X = X.join(
        historical_data.filter(pl.col("month") <= last_month)
        .select("char", "race", "charclass")
        .unique("char"),
        on="char",
        how="left",
    ).with_row_index()
    
    X_last_month = filter_df_by_month(X, last_month)
    months = historical_data.filter(pl.col("month").dt.month() < X["month"].dt.month().max())[
        "month"
    ].unique()
    # This is used to add the historical data up to the given month
    for month in months:
        # I need to truncate the historical timestamp to month to be able to
        # compare it with the month in the target
        kept_historical_data = filter_df_by_month(historical_data, month)
        this_month_X = filter_df_by_month(X, month)
        historical_data_with_sessions = encoder.fit_transform(kept_historical_data)
        df_with_features = adding_other_features(
            this_month_X, historical_data_with_sessions
        )
        df_with_features = df_with_features.with_columns(
            cs.duration().dt.total_minutes()
        )
        features_by_month.append(df_with_features)

    X_res = pl.concat(features_by_month)
    X_res = pl.concat([X_res, X_last_month], how="diagonal")
    X_res = X_res.sort("index").drop("index")

    return X_res
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
    encoded = all_features.skb.apply(skrub.TableVectorizer())
    # data_op = encoded.skb.apply(SimpleImputer()).skb.apply(LogisticRegression(), y=y)
    # data_op = encoded.skb.apply(HGB(), y=y)
    data_op = encoded.skb.apply(DummyClassifier(), y=y)
    return data_op


def cross_validate():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    df = sample_by_user(df, fraction=0.1)
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
# evaluation_results = evaluate()
# %%
# print(results)

# %%
