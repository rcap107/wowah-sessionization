# Repository Report: WoWAH Player Churn Prediction Pipeline

## What this repository is about

This repository predicts **player churn** in *World of Warcraft* from server
snapshots. For every `(character, month)` pair, a model estimates whether the
character has churned, using only behaviour observed in the previous month. The
code is also the running example of two talks about [skrub](https://skrub-data.org/)
DataOps (see [Documents](#documents)).

The scripts were last run end to end with Python 3.15, polars 2.1.0, skrub 0.11.0
and scikit-learn 1.9.1 (`pixi.toml` describes the intended environment).

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

`data/` is git-ignored except for two parquet files.

| File | Size | Description |
|---|---|---|
| `wowah.rar` | 551 MB | Original archive of `.txt` logs |
| `wowah_parsed_all.csv` / `.parquet` | 1.9 GB / 163 MB | Output of `src/parse_to_single_table.py` and `src/convert_to_parquet.py` |
| `wowah_data_all.parquet` | 138 MB | Full dataset: 35.6M rows, 90,456 characters, 2005-12-31 to 2009-01-10 |
| `wowah_data_raw.parquet` | 75 MB | **Year 2008 only**: 10.8M rows, 37,354 characters. Default input of the pipeline |
| `wowah_churn_data.parquet` | 383 KB | Legacy `(char, month, has_played, first_month)` table |
| `wowah_user_month_with_churn.parquet` | 14 MB | Legacy `(char, month, churned)` table |
| `no_matches_files.log`, `index.html`, `img/` | | Parser log and the original dataset page |

No script in the repository creates `wowah_data_raw.parquet` or
`wowah_data_all.parquet` from the parsed files (the renaming of `avatar_id` to `char`
and the selection of 2008 happened elsewhere). The churn table used to be
written by `add_churn.py`, but that line is now commented out; the pipeline builds
the table in memory instead. `main.py`'s `get_env()`, the older scripts and two
plotting scripts still read `wowah_churn_data.parquet`.

---

## Problem definition

Churn is defined at monthly resolution in `add_churn.py`:

1. Build the grid of all `(char, month)` pairs, from the first month in the data to
   **3 months after the last one**. The extra months allow lagged features to be
   evaluated for months with no data yet.
2. `has_played` is `True` if the character has at least one heartbeat in the month.
3. Drop rows before the character's first month.
4. `prev_played` is `has_played` shifted by one month within each character.
5. `churn = ~has_played & ~prev_played`, i.e. the character was absent for **two
   consecutive months**, the month itself and the one before. A character that
   skips one month and then comes back is not marked as churned.

`build_churn_dataset()` returns char, month, has_played, prev_played, churn,
first_month. The model's target `y` is `churn`; `X` is `(char, month)`.

**Feature timing.** For a row at month `M`, `build_feature_table` keeps only the
heartbeats from month `M - 1` (it shifts the month column by one interval and
filters on `M`). This is what prevents leakage. A comment in `main.py` still mentions
`M - 2`; the code uses one month.

**Validation.** The custom `Splitter` in `main.py` iterates over months from
`MIN_DATE` (2006-02-01) to `MAX_DATE` (2008-11-01). At each `split_point` it trains on
all rows with `month <= split_point` and tests on the rows of the next month. Splits
with an empty train or test set are skipped, so the 2008 data gives 11 folds. A
custom splitter is needed because the number of rows per month grows over time, which
breaks `TimeSeriesSplit`'s assumption of equal-size folds.

---

## Code walkthrough

### `main.py`: pipeline

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
  `load` (10% sample of characters) → guild `-1` to null → `build_churn_dataset` →
  `X`/`y` marked with `Splitter` → `build_feature_table` → `add_features` →
  `TableVectorizer` → `HistGradientBoostingClassifier` scored with ROC AUC.
  Searched choices: `session_gap` ∈ {15, 30, 60}, `location_features` (bool),
  `lags` ∈ {none, (1), (1, 2), (1, 2, 3)}, learning rate ∈ [0.1, 0.5] (log scale).
  `add_gini` is fixed to `True`.
- Entry points: `evaluate()` (run by `python main.py`, then `data_op.skb.report()`),
  `cross_validate()`, `random_search()` (Optuna, results in `wowah_churn_study.db`).
  `get_env()` supplies the inputs: the raw parquet path and a query restricted to
  10,000 random characters taken from `wowah_churn_data.parquet`.

### `add_churn.py`: churn table

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
- `convert_to_parquet.py`: CSV to parquet. Both scripts have hard-coded paths from an earlier layout.

### Exploration and plotting scripts

Cell-style scripts (`# %%`), run from the repository root:

- `location_features.py`: exploration of zone rarity, hubs, Gini, spatial entropy,
  time-of-day and day-of-week variability and HDBSCAN clustering. Only the rarity and Gini ideas
  made it into `adding_features.py`.
- `lagged_features.py`: scratch script that builds the feature table and shows it in a `TableReport`.
- `main-unpacked.py`: an older, more granular copy of the plan (reads
  `wowah_churn_data.parquet`, uses `has_played` as target).
- `plotting/plot_user_months.py`, `plot_stats_by_month.py`, `plot_session_start_end.py`:
  rows per month, monthly stats, polar plots of session start and end times.
  `plot_user_months.py` uses `../data/...` paths and so runs from `plotting/`.

---

## Documents

- `draft.qmd`: blog post about the pipeline. It describes an **earlier version**: the target
  is `has_played` and features come from `M - 2`.
- `skrub-dataops.qmd`, `pain_points.qmd`: reveal.js slides on the DataOps pain points
  (leakage, irregular CV, tuning wrangling choices, one deployable learner, debugging, subsampling,
  staying in the dataframe API). They describe churn as two months of inactivity, as in the current code.
- `_quarto.yml` renders to `docs/`; `_freeze/` and `.jupyter_cache/` hold execution caches.
  `report.pdf` is an older rendered report.
- `report.dot/.mmd/.svg` and `build_feature_table.mmd/.svg` are diagrams of the plan.

---

## Results of a run

`cross_validate()` on `data/wowah_data_raw.parquet` (10% of characters, default
hyperparameters, 36 s):

| Fold (test month) | Feb 2008 | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ROC AUC | n/a (one class) | 0.50 | 0.71 | 0.74 | 0.76 | 0.81 | 0.81 | 0.79 | 0.85 | 0.87 | 0.74 |

The first fold has only one class in the test set, so AUC is undefined. These numbers are a
smoke test, not a tuned benchmark. `wowah_churn_study.db` contains the Optuna runs of
earlier searches.

## Caveats

- Because `churn` requires absence in month `M - 1` and features come from that same month,
  a character with no activity in `M - 1` has all-null current-month features, and
  that is a necessary condition for `churn = True`. Check how much of the AUC comes from this
  structure before comparing it with other definitions.
- With the 2008-only file, every character seen in January 2008 looks like a new
  player (`first_month` is cut off at the start of the data). The 3 padding months after
  the data end are all absences by construction, so they are never evaluated: `MAX_DATE` limits the
  last test month to December 2008.
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
