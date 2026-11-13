# WoWAH Player Churn Prediction

A pipeline that predicts, for each World of Warcraft character and month,
whether the character will have churned, i.e. stopped playing, by the end of the
following month. It uses the public
[World of Warcraft Avatar History (WoWAH)](https://qualinet.github.io/databases/world_of_warcraft_avatar_history_dataset/)
dataset: roughly one snapshot every 10 minutes of every online Horde character
between 2006 and 2009.

The pipeline is built with [polars](https://pola.rs/) and
[skrub](https://skrub-data.org/) DataOps, with a scikit-learn model on top. A
custom month-by-month splitter makes sure features and targets never use data
from the future. See [REPORT.md](REPORT.md) for a detailed walkthrough.

## Task definition

Let `M` be the month in which we score a character (and could act on it).

- **Unit of analysis**: a `(char, month)` pair, from the character's first month
  on.
- **Churned as of a month** (`churn`): the character was **not seen in that
  month and not seen in the previous month**.
- **Target** (`churn_next`): the character has churned as of `M + 1`, i.e. it is
  absent in both `M` and `M + 1`.
- **Features** for a row at month `M` are computed from the heartbeats of month
  `M - 1` only (plus fixed attributes such as race and class).
- **Validation**: `Splitter` tests on a month `T` and trains on the rows up to
  `T - 2`. The target of a row at `T - 1` depends on `T`, so it is left out. The
  first test month is March 2006 and the last is November 2008. Metric: ROC AUC.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | Builds the skrub DataOps plan (target, feature table, `TableVectorizer`, `HistGradientBoostingClassifier`), the `Splitter`, and the `evaluate`, `cross_validate` and `random_search` entry points |
| `add_churn.py` | Builds the `(char, month)` table with `has_played`, `churn` and the target `churn_next` (`build_churn_dataset`) |
| `explore_data.py` | Exploratory profiling of the heartbeat parquet: attributes, per-character activity, heartbeat gaps, sessions, monthly activity, and a `skrub` report on a character sample (`pixi run explore`) |
| `adding_features.py` | Session, monthly player, class, location (zone rarity, hubs, Gini) and lag/diff features |
| `src/utils.py` | `sample_by_user` (sample whole character histories) and `get_session_duration` |
| `src/parse_to_single_table.py`, `src/convert_to_parquet.py` | Parse the raw WoWAH `.txt` logs to CSV, then to parquet |
| `location_features.py`, `lagged_features.py`, `main-unpacked.py` | Exploration scripts and an older variant of `main.py` split into more DataOps nodes |
| `plotting/` | Plots: users per month, monthly stats, polar plots of session start/end times |
| `draft.qmd`, `pain_points.qmd`, `skrub-dataops.qmd` | Blog post draft and two slide decks; rendered output goes to `docs/` |
| `wowah_churn_study.db` | SQLite storage with the Optuna studies from `random_search()` |
| `report.dot`, `report.mmd`, `report.svg`, `build_feature_table.*` | Diagrams of the DataOps plan |

## Setup

```bash
pixi install            # default environment
pixi install -e docs    # also installs quarto and jupyter to render the .qmd files
```

`graphviz` and `pydot` are included because `data_op.skb.report()`, called at
the end of `main.py`, needs the `dot` command. Use `pixi run` or `pixi shell` so
that it is on the `PATH`.

## Data

Everything lives in `data/`, which is git-ignored. Two old parquet files
(`wowah_data_raw.parquet`, `wowah_user_month_with_churn.parquet`) are still
tracked by git from before that.

| File | Content |
|---|---|
| `wowah_data_2008.parquet` | Heartbeats for 2008 only (about 10.8M rows, 37k characters). Default input of the pipeline |
| `wowah_data_all.parquet` | Full dataset, 2005-12 to 2009-01 (about 35.6M rows, 90k characters) |
| `wowah_parsed_all.csv` / `.parquet` | Output of the parsing scripts (column `avatar_id` instead of `char`) |
| `wowah_churn_data.parquet`, `wowah_user_month_with_churn.parquet` | Legacy churn tables, no longer produced or used by `main.py`. Some older scripts still read the first one |
| `wowah.rar`, `no_matches_files.log`, `index.html`, `img/` | Original archive, parser log and the dataset's documentation page |

The 2008 and "all" parquet files have columns `timestamp, char, guild, level,
race, charclass, zone`. No script in this repository creates them from
`wowah_parsed_all.*`. The parsing scripts also contain hard-coded paths from an
earlier layout and need adjusting before use.

## Usage

```bash
pixi run evaluate                       # 2008 data, 10% of the characters
pixi run evaluate-all                   # full dataset, 10% of the characters
pixi run python main.py --dataset all --fraction 1   # everything
```

`--dataset` is `2008` (default) or `all`, and `--fraction` is the share of
characters to sample. The same options exist as arguments of `evaluate()`,
`cross_validate()` and `random_search()` (Optuna backend, results in
`wowah_churn_study.db`). Uncomment the calls at the bottom of `main.py` to use
them.

Searched hyperparameters: `session_gap` (15, 30 or 60 minutes), whether to use
location features, the lags (none, 1, 1-2 or 1-2-3 months) and the learning
rate of the gradient boosting model.

Other tasks: `pixi run churn` prints the churn table, `pixi run explore` profiles
the full heartbeat dataset (HTML report and plots in `reports/exploration/`),
and `pixi run -e docs render` renders the documents.
