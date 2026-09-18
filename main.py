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

from add_churn import build_churn_dataset
from src.utils import sample_by_user

from adding_features import (
    add_general_features,
    get_session_duration,
    add_location_features,
    add_lagged_features,
)

# MIN_DATE = datetime.strptime("2008-01-01", "%Y-%m-%d")
# MAX_DATE = datetime.strptime("2008-06-30", "%Y-%m-%d")
# Actual ranges for the full dataset
MIN_DATE = datetime.strptime("2006-01-01", "%Y-%m-%d")
MAX_DATE = datetime.strptime("2009-01-10", "%Y-%m-%d")


# The splitter iterates over the months and selects all the months up to the
# split point, which is the month during which we want to perform some operation
# on users that are marked as "churn risks".
class Splitter:
    def split(self, user_month, has_played=None, interval="1mo"):
        # has_played is not needed in this splitter since we are only splitting
        # based on the month
        del has_played
        time_range = pl.date_range(MIN_DATE, MAX_DATE, interval, eager=True)
        for split_point in time_range:
            # I can either use dateutils.relative delta
            # test_month = split_point + relativedelta(months=1)
            # Or do this with polars which is more consistent with the rest of the code
            test_month = pl.Series([split_point]).dt.offset_by(interval).first()
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
#
# It builds the feature table for every (char, month) pair a character could
# appear in (not just the rows asked for by a given CV fold), because lagged
# features need each character's full monthly timeline: a CV fold's X may
# contain a single month (e.g. the test fold), which would otherwise leave
# every lag/diff column null.
def build_feature_table(
    historical_data,
    session_gap=30,
    use_location=True,
    add_gini=False,
    interval="1mo",
    lags=(1, 2),
):
    features_by_month = []

    # Create a session encoder with a 30 minute timeout
    # This encoder is used as a stateless transformer so it is refitted for every
    # month
    session_encoder = SessionEncoder(
        split_by="char", timestamp_col="timestamp", session_gap=session_gap
    )
    historical_data = historical_data.with_columns(
        month=pl.col("timestamp").dt.truncate(interval)
    )

    # Grouping by character and zone so that I can get the time spent in each zone
    # Even if users leave the zone, this lets me find how much time a user spends in
    # a given zone
    session_encoder_zone = SessionEncoder(
        split_by=["char", "zone"], timestamp_col="timestamp", session_gap=session_gap
    )

    # Every (char, month) pair a character could appear in, whether or not they
    # played that month: the full grid lag/diff features are built over.
    user_month = build_churn_dataset(historical_data.lazy()).select("char", "month")
    # Adding fixed features: these features are fixed by character so they don't
    # change over time.
    user_month = user_month.join(
        historical_data.select("char", "race", "charclass").unique("char"),
        on="char",
        how="left",
        maintain_order="left",
    )

    # This is used to add the historical data up to the given month
    # Sorting months is not needed, but forces a consistent order (better for debugging)
    for month in user_month["month"].unique().sort():
        this_month_X = filter_df_by_month(user_month, month)

        # Selecting only the entries in the historical data whose month + 2 is equal
        # to the month I am trying to predict on.
        # This means that if the "target month" is April, then the historical data
        # should be filtered to keep only the rows where the current month + 2
        # is equal to April, that is, the month is February. This is equivalent
        # to saying "I want the rows for the current month - 2 months", but it's
        # easier to implement
        kept_historical_data = historical_data.with_columns(
            pl.col("month").dt.offset_by(interval)
        ).filter(pl.col("month") == month)

        # Session features: a session starts from a heartbeat, then it ends when
        # no more heartbeats are detected for session_gap minutes
        historical_data_with_sessions = session_encoder.fit_transform(
            kept_historical_data
        )
        historical_data_with_sessions = get_session_duration(
            historical_data_with_sessions
        )

        # General features: add session based and playerbase features
        df_with_features = add_general_features(
            this_month_X, historical_data_with_sessions
        )

        # Location features can be useful but take much longer to generate
        if use_location:
            # Zone-session features: a session lasts from the first time a character
            # enters a zone to the moment it leaves it
            # This is useful to get zone-specific features
            historical_data_zone_sessions = session_encoder_zone.fit_transform(
                kept_historical_data
            )
            historical_data_zone_sessions = get_session_duration(
                historical_data_zone_sessions
            )
            df_with_features = add_location_features(
                df_with_features,
                historical_data_zone_sessions,
                add_gini=add_gini,
            )

        features_by_month.append(df_with_features)
        assert len(df_with_features) == len(this_month_X)

    feature_table = pl.concat(features_by_month, how="vertical")
    to_fill = pl.col("monthly_total_session_duration", "monthly_avg_session_duration")
    feature_table = feature_table.with_columns(
        to_fill.fill_null(pl.duration(seconds=0))
    )
    if lags is not None:
        feature_table = add_lagged_features(feature_table, lags=lags)
    return feature_table


def add_features(X, feature_table):
    """Attach this CV fold's (char, month) rows to the precomputed features."""
    return X.join(
        feature_table, on=["char", "month"], how="left", maintain_order="left"
    )


def load(file, fraction=0.1):
    if fraction == 1:
        return pl.scan_parquet(file)

    df = pl.scan_parquet(file)
    df = sample_by_user(df.collect(), fraction=fraction)
    return df.lazy()


# %%
def make_data_op():
    historical_data_file = skrub.var("historical_data_file")
    historical_data = historical_data_file.skb.apply_func(load, fraction=0.01)
    # In the original data, "guild == -1" means "no guild", so I'm replacing -1
    # with nulls.
    historical_data = historical_data.with_columns(
        pl.when(pl.col("guild") == -1)
        .then(None)
        .otherwise(pl.col("guild"))
        .alias("guild")
    )

    user_month_has_played = historical_data.skb.apply_func(build_churn_dataset)
    X = user_month_has_played["char", "month"].skb.mark_as_X(cv=Splitter())
    y = user_month_has_played["has_played"].skb.mark_as_y()

    # Hyperparameters
    session_gap = skrub.choose_from([30, 60, 15], name="session_gap")
    use_location = skrub.choose_bool(name="location_features")
    add_gini = skrub.choose_bool(name="add_gini")
    lags = skrub.choose_from(
        {"no": None, "1": (1,), "1_2": (1, 2), "1_2_3": (1, 2, 3)}, name="lags"
    )

    # Built from the full historical data (a plain, non-X node), independent of
    # the CV fold split of X, so lagged features always see each character's
    # complete monthly timeline regardless of which fold is being evaluated.
    feature_table = historical_data.collect().skb.apply_func(
        build_feature_table,
        session_gap=session_gap,
        use_location=use_location,
        add_gini=add_gini,
        lags=lags,
    )
    all_features = X.skb.apply_func(add_features, feature_table)
    encoded = all_features.skb.apply(skrub.TableVectorizer())
    data_op = encoded.skb.apply(
        HGB(learning_rate=skrub.choose_float(0.1, 0.5, log=True)), y=y
    ).skb.with_scoring("roc_auc")
    # data_op = encoded.skb.apply(DummyClassifier(), y=y) #.skb.with_scoring("roc_auc")
    return data_op


# %%


def get_env():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    df = sample_by_user(df, fraction=0.1)
    historical_data_file = "data/wowah_data_raw.parquet"
    return {"query": df, "historical_data_file": historical_data_file}


def cross_validate():
    historical_data_file = "data/wowah_data_all.parquet"
    results = make_data_op().skb.cross_validate(
        {"historical_data_file": historical_data_file}
    )
    return results


def random_search():
    df = pl.read_parquet("data/wowah_churn_data.parquet")
    df = sample_by_user(df, fraction=0.15)
    historical_data_file = "data/wowah_data_raw.parquet"
    search = make_data_op().skb.make_randomized_search(
        backend="optuna",
        n_jobs=-1,
        n_iter=16,
        # study_name="wowah_churn_study",
        storage="sqlite:///wowah_churn_study.db",
    )
    env = {"historical_data_file": historical_data_file}

    return search, env


def evaluate():
    historical_data_file = "data/wowah_data_raw.parquet"
    data_op = make_data_op()
    results = data_op.skb.eval({"historical_data_file": historical_data_file})
    return results


if __name__ == "__main__":
    # results = cross_validate()
    # print(results)

    search, env = random_search()

    search.fit(env)
