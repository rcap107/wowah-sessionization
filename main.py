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
MIN_DATE = datetime.strptime("2006-02-01", "%Y-%m-%d")
# The last split point is October 2008, so the last test month is November: its
# target depends on December, while the January 2009 data in the full dataset is
# incomplete (it stops on January 10th).
MAX_DATE = datetime.strptime("2008-10-01", "%Y-%m-%d")

# Input files that main.py can run on: the year 2008 only (fast, good for
# development) or the full 2006-2009 dataset.
DATASETS = {
    "2008": "data/wowah_data_2008.parquet",
    "all": "data/wowah_data_all.parquet",
}
DEFAULT_DATASET = "2008"


# The splitter iterates over the months. For each split point, the test month is
# the following month: it is the month during which we would score users and act
# on the ones marked as "churn risks".
#
# The target of a row (churn_next) is the churn status of the month after it, so
# it is only known one month later. The training set therefore stops one month
# before the split point: a row from the split point month would have a target
# computed from the test month.
class Splitter:
    def split(self, user_month, has_played=None, interval="1mo"):
        # the target is not needed in this splitter since we are only splitting
        # based on the month
        del has_played
        time_range = pl.date_range(MIN_DATE, MAX_DATE, interval, eager=True)
        for split_point in time_range:
            # I can either use dateutils.relative delta
            # test_month = split_point + relativedelta(months=1)
            # Or do this with polars which is more consistent with the rest of the code
            test_month = pl.Series([split_point]).dt.offset_by(interval).first()
            last_train_month = pl.Series([split_point]).dt.offset_by("-" + interval).first()
            # Train indices are up to last_train_month, included
            train_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") <= last_train_month)["idx"]
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

    # Create a session encoder. The gap is a hyperparameter that can be tuned.
    # This encoder is used as a stateless transformer so it is refitted for every
    # month
    session_encoder = SessionEncoder(
        split_by="char", timestamp_col="timestamp", session_gap=session_gap * 60
    )
    historical_data = historical_data.with_columns(
        month=pl.col("timestamp").dt.truncate(interval)
    ).lazy()

    # Every (char, month) pair a character could appear in, whether or not they
    # played that month: the full grid lag/diff features are built over.
    user_month = build_churn_dataset(historical_data).select("char", "month").lazy()
    # Adding fixed features: these features are fixed by character so they don't
    # change over time.
    user_month = user_month.join(
        historical_data.select("char", "race", "charclass").unique("char"),
        on="char",
        how="left",
        maintain_order="left",
    ).collect()

    # This is used to add the historical data up to the given month.
    # Sorting months is not needed, but forces a consistent order (better for debugging)
    for month in user_month["month"].unique().sort():
        this_month_X = filter_df_by_month(user_month, month)

        # Selecting only the entries in the historical data from the month before
        # the one I am building features for.
        # This means that if the row's month is April, then the historical data
        # is filtered to keep only the rows from March. Shifting the data forward
        # by one interval and filtering on the current month is equivalent to
        # saying "I want the rows for the current month - 1 month", but it's
        # easier to implement
        kept_historical_data = historical_data.with_columns(
            pl.col("month").dt.offset_by(interval)
        ).filter(pl.col("month") == month)

        # Session features: a session starts from a heartbeat, then it ends when
        # no more heartbeats are detected for session_gap minutes
        historical_data_sessions = session_encoder.fit_transform(
            kept_historical_data.collect()
        ).lazy()
        historical_data_sessions = get_session_duration(historical_data_sessions)

        # General features: add session based and playerbase features
        df_with_features = add_general_features(
            this_month_X.lazy(), historical_data_sessions
        )

        # Location features can be useful but take much longer to generate
        if use_location:
            # Grouping by character and zone so that I can get the time spent in each zone
            # Even if users leave the zone, this lets me find how much time a user spends in
            # a given zone
            session_encoder_zone = SessionEncoder(
                split_by=["char", "zone"],
                timestamp_col="timestamp",
                session_gap=session_gap * 60,
                suffix="session_id_zone",
            )
            # Zone-session features: a session lasts from the first time a character
            # enters a zone to the moment it leaves it
            # This is useful to get zone-specific features
            historical_data_zone_sessions = session_encoder_zone.fit_transform(
                kept_historical_data.collect()
            ).lazy()
            historical_data_zone_sessions = get_session_duration(
                historical_data_zone_sessions,
                session_column="timestamp_session_id_zone",
            )
            df_with_features = add_location_features(
                df_with_features,
                historical_data_zone_sessions,
                add_gini=add_gini,
            )
        else:
            df_with_features = df_with_features.collect()
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
def make_data_op(load_fraction=0.1):
    historical = skrub.var("historical_data_file")
    historical_data = historical.skb.apply_func(load, fraction=load_fraction)
    # In the original data, "guild == -1" means "no guild", so I'm replacing -1
    # with nulls.
    historical_data = historical_data.with_columns(
        pl.when(pl.col("guild") == -1)
        .then(None)
        .otherwise(pl.col("guild"))
        .alias("guild")
    )

    user_month_has_played = historical_data.skb.apply_func(
        build_churn_dataset
    ).skb.set_name("query")
    X = user_month_has_played["char", "month"].skb.mark_as_X(cv=Splitter())
    y = user_month_has_played["churn_next"].skb.mark_as_y()

    # Hyperparameters
    session_gap = skrub.choose_from([30, 60, 15], name="session_gap")
    use_location = skrub.choose_bool(name="location_features")
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
        add_gini=True,
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


def get_env(dataset=DEFAULT_DATASET):
    # The characters to use are sampled inside the plan (see `load`), so the
    # features and the (char, month) rows always cover the same characters.
    return {"historical_data_file": DATASETS[dataset]}


def cross_validate(dataset=DEFAULT_DATASET, load_fraction=0.1):
    data_op = make_data_op(load_fraction=load_fraction)
    return data_op.skb.cross_validate(environment=get_env(dataset))


def random_search(dataset=DEFAULT_DATASET, load_fraction=0.1):
    # df = pl.read_parquet("data/wowah_churn_data.parquet")
    # df = sample_by_user(df, fraction=0.15)
    search = make_data_op(load_fraction=load_fraction).skb.make_randomized_search(
        backend="optuna",
        n_jobs=-1,
        n_iter=16,
        # study_name="wowah_churn_study",
        storage="sqlite:///wowah_churn_study.db",
    )
    return search, get_env(dataset)


def evaluate(dataset=DEFAULT_DATASET, load_fraction=0.1):
    data_op = make_data_op(load_fraction=load_fraction)
    results = data_op.skb.eval(get_env(dataset))
    return results, data_op


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the churn prediction DataOps plan.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default=DEFAULT_DATASET,
        help="'2008' for the year 2008 only, 'all' for the full dataset",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="fraction of characters to sample (1 keeps all of them)",
    )
    args = parser.parse_args()

    results, data_op = evaluate(args.dataset, load_fraction=args.fraction)
    report = data_op.skb.report(environment=get_env(args.dataset))
    # results = cross_validate(args.dataset, load_fraction=args.fraction)
    # print(results)

    # search, env = random_search(args.dataset, load_fraction=args.fraction)
    # search.fit(env)
