# WoWAH Player Churn Prediction

A pipeline that predicts, for each World of Warcraft character and month,
whether the character has churned, i.e. stopped playing. It uses the public
[World of Warcraft Avatar History (WoWAH)](https://qualinet.github.io/databases/world_of_warcraft_avatar_history_dataset/)
dataset: roughly one snapshot every 10 minutes of every online Horde character
between 2006 and 2009.

The pipeline is built with [polars](https://pola.rs/) and
[skrub](https://skrub-data.org/) DataOps, with a scikit-learn model on top. A
custom month-by-month splitter makes sure features never use data from the
future. See [REPORT.md](REPORT.md) for a detailed walkthrough.

## Task definition

- **Unit of analysis**: a `(char, month)` pair. The grid contains every month
  from a character's first appearance up to 3 months past the end of the data.
- **Target** (`churn`): `True` when the character was **not seen in the month and
  not seen in the previous month**.
- **Features** for a row at month `M` are computed from the heartbeats of month
  `M - 1` only, which leaves one month to act on at-risk players.
- **Validation**: `Splitter` trains on all months up to `M` and tests on month
  `M + 1`, for every month. Metric: ROC AUC.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | Builds the skrub DataOps plan (churn target, feature table, `TableVectorizer`, `HistGradientBoostingClassifier`), the `Splitter`, and the `evaluate`, `cross_validate` and `random_search` entry points |
| `add_churn.py` | Builds the `(char, month)` churn table from the raw heartbeats (`build_churn_dataset`) |
| `adding_features.py` | Session, monthly player, class, location (zone rarity, hubs, Gini) and lag/diff features |
| `src/utils.py` | `sample_by_user` (sample whole character histories) and `get_session_duration` |
| `src/parse_to_single_table.py`, `src/convert_to_parquet.py` | Parse the raw WoWAH `.txt` logs to CSV, then to parquet |
| `location_features.py`, `lagged_features.py`, `main-unpacked.py` | Exploration scripts and a variant of `main.py` split into more DataOps nodes |
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
the end of `main.py`, needs the `dot` command.

## Data

Everything lives in `data/`, which is git-ignored apart from two parquet files.

| File | Content |
|---|---|
| `wowah_data_raw.parquet` | Heartbeats for 2008 only (about 10.8M rows, 37k characters). Default input of the pipeline |
| `wowah_data_all.parquet` | Full dataset, 2005-12 to 2009-01 (about 35.6M rows, 90k characters) |
| `wowah_parsed_all.csv` / `.parquet` | Output of the parsing scripts (column `avatar_id` instead of `char`) |
| `wowah_churn_data.parquet` | Legacy `(char, month, has_played, first_month)` table. `main.py`'s `get_env()` still reads it to sample characters, and the older scripts also load it |
| `wowah_user_month_with_churn.parquet` | Legacy `(char, month, churned)` table |
| `wowah.rar`, `no_matches_files.log`, `index.html`, `img/` | Original archive, parser log and the dataset's documentation page |

The raw and "all" parquet files have columns `timestamp, char, guild, level,
race, charclass, zone`. No script in this repository creates them from
`wowah_parsed_all.*`. The parsing scripts also contain hard-coded paths from an
earlier layout and need adjusting before use.

## Usage

```bash
pixi run churn      # build and preview the churn table
pixi run evaluate   # run the DataOps plan on a 10% sample of characters and generate the DataOps report
```

`main.py` also defines `cross_validate()` and `random_search()` (Optuna backend,
results in `wowah_churn_study.db`). Uncomment the calls at the bottom of the file
to use them. Cross-validation on the 2008 data takes under a minute on a
laptop.

Searched hyperparameters: `session_gap` (15, 30 or 60 minutes), whether to use
location features, the lags (none, 1, 1-2 or 1-2-3 months) and the learning
rate of the gradient boosting model.

To render the documents: `pixi run -e docs render`.
