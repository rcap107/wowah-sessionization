# Repository Report: WoWAH Player Churn Prediction Pipeline

## What this repository is about

This repository predicts **player churn** in *World of Warcraft* from server
snapshots. For every `(character, month)` pair, a model estimates whether the
character will have churned by the end of the following month, using only
behaviour observed in the previous month. The code is also the running example of
two talks about [skrub](https://skrub-data.org/) DataOps (see
[Documents](#documents)).

The scripts were last run end to end with Python 3.15, polars 2.1.0, skrub 0.11.0
and scikit-learn 1.9.1 (`pixi.toml` describes the environment).

---

## The dataset: World of Warcraft Avatar History (WoWAH)

The raw data is the public **WoWAH** dataset (Lee et al., ACM Multimedia Systems 2011):
91,065 avatars of the Horde faction observed over 1,107 days between January 2006
and January 2009. The game server was polled about every 10 minutes and the state of
every online character was logged (a "heartbeat"). Avatar and guild names are
anonymised to integers.

| Column | Description |
|---|---|
| `timestamp` | Date and time of the observation |
| `char` | Anonymised character ID (called `avatar_id` in the parsed CSV) |
| `guild` | Guild ID, `-1` when the character has no guild (replaced with null in the pipeline) |
| `level` | Character level |
| `race`, `charclass` | Character race and class |
| `zone` | In-game zone |

### Files in `data/`

`data/` is git-ignored. Two older parquet files are still tracked by git.

| File | Size | Description |
|---|---|---|
| `wowah.rar` | 551 MB | Original archive of `.txt` logs |
| `wowah_parsed_all.csv` / `.parquet` | 1.9 GB / 163 MB | Output of `src/parse_to_single_table.py` and `src/convert_to_parquet.py` |
| `wowah_data_all.parquet` | 138 MB | Full dataset: 35.6M rows, 90,456 characters, 2005-12-31 to 2009-01-10 |
| `wowah_data_2008.parquet` | 75 MB | **Year 2008 only**: 10.8M rows, 37,354 characters. Default input of the pipeline |
| `wowah_churn_data.parquet` | 383 KB | Legacy `(char, month, has_played, first_month)` table |
| `wowah_user_month_with_churn.parquet` | 14 MB | Legacy `(char, month, churned)` table |
| `no_matches_files.log`, `index.html`, `img/` | | Parser log and the original dataset page |

No script in the repository creates the `2008` and `all` parquet files from the
parsed files (the renaming of `avatar_id` to `char` and the selection of 2008
happened elsewhere). The churn table is no longer saved to disk: the pipeline builds
it in memory, so `main.py` does not need either legacy table. `lagged_features.py`,
`main-unpacked.py`, `adding_features.py` (its `__main__` block) and two plotting
scripts still read `wowah_churn_data.parquet`.

---

## Problem definition

Churn is defined at monthly resolution in `add_churn.py`:

1. Build the grid of all `(char, month)` pairs, from the first month in the data to
   3 months after the last one, and drop rows before the character's first month.
2. `has_played` is `True` if the character has at least one heartbeat in the month.
3. `prev_played` is `has_played` shifted by one month within each character.
4. `churn = ~has_played & ~prev_played`: the character has been absent for **two
   consecutive months**, the month itself and the one before. A character that
   skips one month and then comes back is not marked as churned.
5. **`churn_next`** is `churn` shifted one month back: the churn status in the
   *following* month. This is the target `y`. It is null for the last observed
   month (and for the padding months), and `build_churn_dataset()` drops those rows.

`build_churn_dataset()` returns `char, month, has_played, prev_played, churn,
churn_next, first_month`. `X` is `(char, month)`.

**Timeline.** For a row at month `M` (the month in which one could act on at-risk
players):

- features come from the heartbeats of month `M - 1` only (`build_feature_table` shifts
  the month column by one interval and filters on `M`);
- the target is whether the character is absent in both `M` and `M + 1`.

The target used to be `churn` itself (absent in `M` and `M - 1`). Because the features
describe `M - 1`, any character with features was automatically "not churned", which gave
cross-validation AUCs of about 0.95. Shifting the target removed this shortcut.

**Validation.** The custom `Splitter` in `main.py` iterates over split points from
`MIN_DATE` (2006-02-01) to `MAX_DATE` (2008-10-01). For a split point `S`:

- the test set is the rows of month `S + 1`;
- the training set is all rows up to `S - 1`. The month `S` is skipped, because its
  target depends on the test month.

Splits with an empty train or test set are skipped, so the 2008 file gives 9 folds
(test months March to November 2008) and the full dataset 33 (March 2006 to November
2008). The last test month is November: its target depends on December, and the January
2009 data is incomplete. A custom splitter is needed because the number of rows per month
grows over time, which breaks `TimeSeriesSplit`'s assumption of equal-size folds.

---

## Code walkthrough

### `main.py`: pipeline

- `DATASETS` maps `"2008"` and `"all"` to the parquet files. The command line is
  `python main.py [--dataset {2008,all}] [--fraction F]` and the same arguments exist on
  `evaluate()`, `cross_validate()` and `random_search()`.
- `build_feature_table(historical_data, session_gap, use_location, add_gini, interval, lags)`
  builds a feature table for **every** `(char, month)` pair, independently of the CV
  fold. This is needed because lag features need each character's full monthly
  timeline. For each month it:
  1. keeps the previous month's heartbeats;
  2. runs `skrub.SessionEncoder` (split by `char`, gap `session_gap` minutes) and
     computes session durations;
  3. adds the general features (below);
  4. optionally runs a second `SessionEncoder` split by `["char", "zone"]` and adds
     location features.

  Afterwards it fills null session durations with 0 and adds lag/diff features.
- `add_features(X, feature_table)` joins a fold's `(char, month)` rows to that table.
- `make_data_op(load_fraction=0.1)` builds the skrub DataOps plan:
  `load` (sample of characters) → guild `-1` to null → `build_churn_dataset` →
  `X` / `y = churn_next` marked with `Splitter` → `build_feature_table` → `add_features` →
  `TableVectorizer` → `HistGradientBoostingClassifier` scored with ROC AUC.
  Searched choices: `session_gap` ∈ {15, 30, 60}, `location_features` (bool),
  `lags` ∈ {none, (1), (1, 2), (1, 2, 3)}, learning rate ∈ [0.1, 0.5] (log scale).
  `add_gini` is fixed to `True`.
- `get_env(dataset)` only passes the parquet path. The characters are sampled inside the plan
  (`load` with `sample_by_user`), so the target rows and the features always cover the same
  characters.
- `random_search()` uses the Optuna backend and stores results in `wowah_churn_study.db`.

### `add_churn.py`: target table

`make_data`, `make_user_month`, `add_churn`, `remove_unrealistic_entries` and
`build_churn_dataset` implement the definition above. Running the file prints the head of
the table.

### `adding_features.py`: features

Fixed per character: `race`, `charclass`.

Per character and month, from the previous month's data:

- **Sessions** (`add_session_features`): total, average and standard deviation of the
  session duration, number of sessions, average session start and end time. The
  standard deviation is computed on microseconds, because polars 2.1 cannot compute `std` on a
  Duration column.
- **Player** (`add_monthly_player_features`): max level, number of distinct zones, most
  frequent zone, number of distinct guilds, first and last guild.
- **Playerbase by class** (`add_class_features`): average level and number of active players of
  the character's class.
- **Location** (`add_location_features`, optional):
  - zone rarity `log(N / unique_visitors)`, zones under the 10th percentile of rarity flagged as hubs;
  - per player max and mean rarity, and the fraction of zone sessions in hubs;
  - optionally the Gini coefficient of time spent per non-hub zone (`add_gini`).

  Location features are computed from the single previous month, not the full history.
- **Lags** (`add_lagged_features`): for every feature, `<col>_lag{k}` for each requested
  lag, plus `<col>_diff1` for numeric columns (unsigned counters are cast to signed
  to avoid wrap-around).

### `src/`

- `utils.sample_by_user`: samples a fraction of characters and keeps all their rows.
- `utils.get_session_duration`: adds session start, end and duration; a session with a
  single heartbeat gets a duration of 1 minute, so that "logged in briefly" differs from "never logged in".
- `parse_to_single_table.py`: multiprocessing regex parser from the raw `.txt` logs to a CSV.
- `convert_to_parquet.py`: CSV to parquet. Both scripts have hard-coded paths from an earlier layout
  (for example `~/work/sessionization/WoWAH`, where `~` is not expanded by `Path`).

### Exploration and plotting scripts

Cell-style scripts (`# %%`), run from the repository root:

- `location_features.py`: exploration of zone rarity, hubs, Gini, spatial entropy,
  time-of-day and day-of-week variability and HDBSCAN clustering. Only the rarity and Gini ideas
  made it into `adding_features.py`.
- `lagged_features.py`: scratch script that builds the feature table and shows it in a `TableReport`.
- `main-unpacked.py`: an older, more granular copy of the plan (reads
  `wowah_churn_data.parquet`, uses `has_played` as target). It has not been updated for
  `churn_next`.
- `plotting/plot_user_months.py`, `plot_stats_by_month.py`, `plot_session_start_end.py`:
  rows per month, monthly stats, polar plots of session start and end times.
  `plot_user_months.py` uses `../data/...` paths and so runs from `plotting/`.

---

## Documents

- `draft.qmd`: blog post about the pipeline. It describes an **earlier version**: the target
  is `has_played` and features come from `M - 2`.
- `skrub-dataops.qmd`, `pain_points.qmd`: reveal.js slides on the DataOps pain points
  (leakage, irregular CV, tuning wrangling choices, one deployable learner, debugging, subsampling,
  staying in the dataframe API).
- `_quarto.yml` renders to `docs/`; `_freeze/` and `.jupyter_cache/` hold execution caches.
  `report.pdf` is an older rendered report.
- `report.dot/.mmd/.svg` and `build_feature_table.mmd/.svg` are diagrams of the plan.

---

## Results of a run

`cross_validate()` with a 10% sample of the characters and default hyperparameters, ROC AUC
per test month.

2008 file (9 folds):

| Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.49 | 0.74 | 0.78 | 0.82 | 0.82 | 0.80 | 0.85 | 0.85 |

Full dataset (33 folds, March 2006 to November 2008): the first two folds are 0.50 and 0.45,
then scores are mostly 0.8 to 0.9, with a range of 0.65 to 0.92 (the lowest are two consecutive
folds, around the 11th and 12th test months).

The first folds have very little training data (a few months of a growing playerbase), which
explains the near-chance scores. These numbers are a sanity check, not a tuned benchmark.
`wowah_churn_study.db` contains Optuna runs from earlier searches; they predate the
`churn_next` target, so their scores are not comparable with the table above.

## Caveats

- The evaluation covers all characters, including those already absent in `M - 1`. For
  them, "stays absent" is easy to predict, so the AUC is higher than it would be on
  the characters that are still active and could actually be retained. Evaluating only on
  characters that played in `M - 1` is not implemented yet.
- With the 2008-only file, every character seen in January 2008 looks like a new
  player (`first_month` is cut off at the start of the data).
- Several scripts still depend on legacy data files (see [Files in `data/`](#files-in-data)).

## Technology stack

| Library | Role |
|---|---|
| polars | Dataframes, lazy queries, parquet I/O |
| skrub | `SessionEncoder`, `TableVectorizer`, DataOps (`skb`), `TableReport` |
| scikit-learn | `HistGradientBoostingClassifier`, metrics |
| optuna | Randomized hyperparameter search backend |
| matplotlib / seaborn | Plots |
| graphviz / pydot | DataOps report graph |
| quarto / jupyter | Rendering the `.qmd` documents |
