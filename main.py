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

from src.utils import sample_by_user

from adding_features import (
    add_general_features,
    get_session_duration,
    add_location_features,
)

MIN_DATE = datetime.strptime("2008-01-01", "%Y-%m-%d")
MAX_DATE = datetime.strptime("2008-12-30", "%Y-%m-%d")
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
                print(f"Working on month {split_point}")
                yield train_idx, test_idx

    def get_n_splits(self, X, y):
        return sum(1 for _ in self.split(X, y))


def filter_df_by_month(df, month):
    return df.filter(pl.col("month") == month)


# %%
# This function is needed to make sure that we are only ever using historical data
# up to the given month - 1 month. This is to avoid any leakage in the data.
def add_features(X, historical_data, session_gap=30, use_location=True):
    features_by_month = []

    # Create a session encoder with a 30 minute timeout
    # This encoder is used as a stateless transformer so it is refitted for every
    # month
    session_encoder = SessionEncoder(
        split_by="char", timestamp_col="timestamp", session_gap=session_gap
    )
    historical_data = historical_data.with_columns(
        month=pl.col("timestamp").dt.truncate("1mo")
    )
    last_month = X["month"].max()

    # Grouping by character and zone so that I can get the time spent in each zone
    # Even if users leave the zone, this lets me find how much time a user spends in
    # a given zone
    session_encoder_zone = SessionEncoder(
        split_by=["char", "zone"], timestamp_col="timestamp", session_gap=session_gap
    )
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
        maintain_order="left",
    ).with_row_index()  # adding row index so that I can reorder at the end after
    # concatenating
    # kinda defeats the point of using data ops but I think it simplifies the code

    # This is used to add the historical data up to the given month
    # Sorting months is not needed, but forces a consistent order (better for debugging)
    for month in X["month"].unique().sort():
        this_month_X = filter_df_by_month(X, month)

        # Selecting only the entries in the historical data whose month + 2 is equal
        # to the month I am trying to predict on.
        # This means that if the "target month" is April, then the historical data
        # should be filtered to keep only the rows where the current month + 2
        # is equal to April, that is, the month is February. This is equivalent
        # to saying "I want the rows for the current month - 2 months", but it's
        # easier to implement
        kept_historical_data = historical_data.with_columns(
            pl.col("month").dt.offset_by("2mo")
        ).filter(pl.col("month") == month)

        # Session features: a session starts from a heartbeat, then it ends when
        # no more heartbeats are detected for session_gap minutes
        historical_data_with_sessions = session_encoder.fit_transform(
            kept_historical_data
        )
        historical_data_with_sessions = get_session_duration(
            historical_data_with_sessions
        )

        # Zone-session features: a session lasts from the first time a character
        # enters a zone to the moment it leaves it
        # This is useful to get zone-specific features
        historical_data_zone_sessions = session_encoder_zone.fit_transform(
            kept_historical_data
        )
        historical_data_zone_sessions = get_session_duration(
            historical_data_zone_sessions
        )

        # General features: add session based and playerbase features
        df_with_features = add_general_features(
            this_month_X, historical_data_with_sessions
        )

        if use_location:
            # Location features can be useful but take longer to generate
            df_with_features = add_location_features(
                df_with_features,
                historical_data_zone_sessions,
            )

        # this step won't be needed once #2069 gets merged
        df_with_features = df_with_features.with_columns(
            cs.duration().dt.total_minutes()
        )
        features_by_month.append(df_with_features)
        assert len(df_with_features) == len(this_month_X)

    X_res = pl.concat(features_by_month, how="vertical")
    X_res = X_res.sort("index").drop("index")

    return X_res


def load(file):
    return pl.read_parquet(file)


# %%
def make_data_op():
    user_month_has_played = skrub.var("query")
    X = user_month_has_played["char", "month"].skb.mark_as_X(cv=Splitter())
    y = user_month_has_played["has_played"].skb.mark_as_y()
    historical_data_file = skrub.var("historical_data_file")
    historical_data = historical_data_file.skb.apply_func(load)
    # In the original data, "guild == -1" means "no guild", so I'm replacing -1
    # with nulls.
    historical_data = historical_data.with_columns(
        pl.when(pl.col("guild") == -1)
        .then(None)
        .otherwise(pl.col("guild"))
        .alias("guild")
    )

    # Hyperparameters
    session_gap = skrub.choose_from([60], name="session_gap")
    use_location = skrub.choose_bool(name="location_features")

    all_features = X.skb.apply_func(
        add_features,
        historical_data,
        session_gap=session_gap,
        use_location=use_location,
    )
    encoded = all_features.skb.apply(skrub.TableVectorizer())
    # data_op = encoded.skb.apply(SimpleImputer()).skb.apply(LogisticRegression(), y=y)
    data_op = encoded.skb.apply(HGB(learning_rate=skrub.choose_float(0.01, 1.0, log=True)), y=y)
    # data_op = encoded.skb.apply(DummyClassifier(), y=y)
    return data_op


# %%


def get_env():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    df = sample_by_user(df, fraction=0.1)
    historical_data_file = "data/wowah_data_raw.parquet"
    return {"query": df, "historical_data_file": historical_data_file}


def cross_validate():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    df = sample_by_user(df, fraction=0.1)
    historical_data_file = "data/wowah_data_raw.parquet"
    results = make_data_op().skb.cross_validate(
        {"query": df, "historical_data_file": historical_data_file}
    )
    return results

def random_search():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    df = sample_by_user(df, fraction=0.05)
    historical_data_file = "data/wowah_data_raw.parquet"
    search = make_data_op().skb.make_randomized_search(
        backend="optuna", n_jobs=-1, n_iter=5
    )
    env = {"query": df, "historical_data_file": historical_data_file}
    
    return search, env
    

def evaluate():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    historical_data_file = "data/wowah_data_raw.parquet"
    results = make_data_op().skb.eval(
        {"query": df, "historical_data_file": historical_data_file}
    )
    return results

search, env = random_search()

# %%
results = cross_validate()
print(results)

# %%
