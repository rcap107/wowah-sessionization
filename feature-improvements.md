# Possible feature improvements for the churn pipeline

These notes collect changes worth considering for `main.py`, `adding_features.py`,
`add_churn.py` and `src/utils.py`, based on the exploration in
`explore_data.py` and the follow-up checks. **Nothing here is implemented yet**:
this is a design note, not a patch.

## What the exploration showed

- The data is 35.6M heartbeats for 90,456 characters, sampled roughly every
  10 minutes. Per-character gaps have a median of 600s; 53.8% are ≤10 min and
  91.1% are ≤30 min. A single missed poll produces a 1,200s gap, and the 90th
  percentile gap is 1,201s.
- There are 17,166 rows that duplicate another `(char, timestamp)` pair, 8,932
  zero-second gaps and 95,116 gaps of exactly 5s. The 5s pairs almost always
  have the same zone and level (99.6% and 99.5%) and are concentrated in
  Apr–May 2007 and Oct 2007–Jan 2008: they are duplicate/near-duplicate polls,
  not real short sessions.
- October 2008 contains a large anomaly: 12,534 distinct characters versus
  7,056 in September. 6,325 of them are first seen in October, 5,701 are never
  seen again, 5,986 are below level 11, 5,389 are in Durotar, and 1,676 were
  first seen in a single guild (guild 460). The number of characters with at least 2
  active days and 5 heartbeats is only 4,673 in October, close to September's
  4,231 and November's 4,716. This is a mass low-level character event, most
  likely bots or a mass-recruiting guild, not expansion-driven player growth.
- The dataset is the Taiwanese realm *Light's Hope*. The Burning Crusade
  launched there on 2007-04-03 and Wrath of the Lich King on 2008-11-18 (the
  first Death Knights and Northrend zones appear on 2008-11-19; the first level
  80 on 2008-11-21). The level cap is 60 until March 2007, 70 from April 2007,
  and 80 from November 2008.
- The collector's `who` query returns at most 50 avatars per call (paper), so
  crowded level bands — especially the level 1–10 starting zones — are the
  least reliable part of the data. The October event should be treated as
  unreliable low-level activity whether it came from mass character creation or
  from a change in query coverage.
- The WoWAH paper reports 21,324 missing samples out of 159,408, partly because
  of the weekly Thursday maintenance window (04:00–12:00) and client issues.
  Count-based features are therefore not comparable across months unless they
  are normalized by the available coverage.

## Priority overview

| Priority | Change | Main area |
|---|---|---|
| P0 | Deduplicate and debounce heartbeats | `main.py::load` / pre-feature cleaning |
| P0 | Normalize counts by monthly coverage | `adding_features.py` |
| P0 | Separate regular characters from one-off observations | features, labels, evaluation |
| P1 | Make the split expansion-aware | `main.py::Splitter`, `MAX_DATE` |
| P1 | Add tenure/recency and level-trajectory features | `adding_features.py` |
| P1 | Fix the session-gap search space and censoring treatment | `main.py`, `src/utils.py` |
| P2 | Improve guild, zone and playerbase aggregates | `adding_features.py` |
| P2 | Harden encoders against unseen classes/zones | `main.py` (vectorizer) |

## 1. Clean the heartbeat stream

### 1.1 Remove exact duplicates

**Evidence.** 17,166 rows share a `(char, timestamp)` with another row; 8,932
per-character gaps are exactly 0s, and those pairs have identical zone, level
and guild.

**Suggestion.** Drop exact duplicates as soon as the raw logs are parsed:

```python
heartbeats = heartbeats.unique(subset=["char", "timestamp"], keep="first")
```

Do it before `build_feature_table`, in `main.py::load` or in a dedicated
cleaning node, so every downstream consumer (sessions, location features,
playerbase aggregates) sees the same stream. Add an assertion or a small test
that the cleaned table has no duplicate `(char, timestamp)` pairs.

**Where.** `main.py::load`, before `historical_data` is passed to
`build_feature_table`; optionally a new `clean_heartbeats` function in
`src/utils.py`.

### 1.2 Collapse near-duplicate polls (optional but recommended)

**Evidence.** 95,116 gaps of exactly 5s and 11,031 of 6s, concentrated in a few
months. They do not create extra sessions at a 30-minute gap, but they inflate
any heartbeat/session-beat counter and can create zero-duration sessions at the
edges of the data.

**Suggestion.** Either ignore them (exact dedup is already a big improvement)
or collapse consecutive beats whose gap is ≤60s, keeping the first and last
state. A 60s threshold is well below the 10-minute sampling interval, so it
cannot merge genuine visits. If session duration is computed from first/last
beats, keeping the last beat of a duplicate group is important.

**Where.** Same cleaning node as 1.1.

### 1.3 Use a coverage measure per month

**Evidence.** The paper reports 21,324 missing samples, the Thursday
maintenance window removes half a day per week, and October 2008 has fewer
heartbeats than September despite more characters.

**Suggestion.** Compute, for each month, the number of distinct snapshot
timestamps and the number of covered days. Join that table into the feature
table and use it in two ways:

- normalize count features: `heartbeats / n_snapshots`,
  `monthly_num_sessions / n_snapshots`, `active_days / covered_days`;
- expose `monthly_n_snapshots` and `monthly_covered_days` as features so the
  model can learn about sparse months.

Only the previous month's coverage is used for month `M` features, so this does
not introduce leakage. Avoid using the current month's coverage for the target.

**Where.** `adding_features.py`, probably in `add_general_features` or a new
`add_coverage_features`; needs the raw heartbeat table for the previous month,
which `build_feature_table` already filters.

## 2. Separate regular characters from one-off observations

### 2.1 Define a regular-character criterion

**Evidence.** October 2008's raw distinct count is 12,534, but only 4,673
characters have at least 2 active days and 5 heartbeats in the month. Across
the dataset, 22,382 characters (25%) are observed only once and 42,244 (47%)
on a single day; the per-character median is 7 heartbeats and 2 active days.

**Suggestion.** Introduce a reusable definition, for example:

```python
regular = (pl.col("active_days") >= 2) & (pl.col("heartbeats") >= 5)
```

computed per `(char, month)` from the heartbeat table. Use it consistently for:

- playerbase and class aggregates (see 5.1),
- reporting metrics on regular vs all characters (see 7.3),
- optional sample weighting or filtering for training (see 2.3).

**Where.** `adding_features.py` (helper), `main.py` (evaluation), possibly
`add_churn.py` if the label itself should be coverage-adjusted.

### 2.2 Add activity-history features

**Evidence.** Character histories vary enormously: `n_events` has p25=2, p50=7,
p90=505 and p99=8,430; active days have p50=2, p90=44 and p99=394. The current
features describe the previous month but not the character's tenure.

**Suggestion.** Add, using only data up to month `M-1`:

- `tenure_months`: months since first seen;
- `tenure_days`: days since first seen (if a finer resolution is wanted);
- `active_months`: number of months with any heartbeat;
- `months_since_last_seen` (1 for a character active last month);
- `is_new_character`: first month in the data;
- `is_one_off`: fewer than 2 active days / 5 heartbeats in the whole history;
- `active_days_last_30` and `heartbeats_last_30`;
- `regular_last_month`: the 2.1 criterion for `M-1`.

These are likely stronger churn predictors than raw monthly counts and make
the model's behavior on new characters explicit.

**Where.** `adding_features.py::add_general_features` or a new history-feature
function that joins a per-character table; `add_lagged_features` already
provides lags/diffs for the monthly columns.

### 2.3 Do not drop one-off characters silently

One-off characters are valid churn cases, and dropping them changes the target
population. Prefer:

- reporting metrics on all rows and on regular rows;
- adding `is_one_off` / `is_new_character` features;
- using sample weights (e.g. lower weight for characters with very little
  history);
- filtering only for diagnostic evaluation, not for the main label definition.

**Where.** `main.py::evaluate` / `cross_validate` reporting,
`src/utils.py::sample_by_user` (stratified sampling).

## 3. Sessionization

### 3.1 Remove or adapt the 15-minute session gap

**Evidence.** The sampling interval is ~10 minutes; a single missed poll gives
a 1,200s gap. A 15-minute threshold therefore splits a session whenever one
poll is missed, and only 53.8% of gaps are ≤10 min. At 30 minutes, 91.1% of
gaps stay inside a session; at 60 minutes, 92.0%.

**Suggestion.** Change

```python
session_gap = skrub.choose_from([30, 60, 15], name="session_gap")
```

to `[30, 60]` (or `[20, 30, 60]` if a finer search is wanted), and document
that 30 minutes tolerates up to two missed polls. Alternatively, derive the
threshold from the observed cadence, e.g. `max(30 min, 2.5 × median gap)`.

**Where.** `main.py::make_data_op`, `main.py::build_feature_table`
(`session_gap` argument), `adding_features.py::get_session_duration`.

### 3.2 Treat session durations as censored

**Evidence.** Durations are measured from the first to the last heartbeat.
Sessions with a single heartbeat have duration 0; `get_session_duration`
currently replaces those with 1 minute. The median session duration is 10
minutes, i.e. one sampling interval, so short sessions are systematically
underestimated.

**Suggestion.** Be explicit about the censoring:

- add the sampling interval (or half of it) to the measured duration, or keep
  the raw duration and add `session_duration_censored` / `is_single_beat`;
- add `session_n_beats` and `single_beat_session_share`;
- use the median as well as the mean duration.

This makes the 1-minute replacement unnecessary or at least better justified.

**Where.** `src/utils.py::get_session_duration`,
`adding_features.py::add_session_features`.

### 3.3 Add session-rhythm features

The current session features are total, average and standard deviation of
duration, number of sessions, and average start/end time. Add:

- `median_session_duration`, `max_session_duration`;
- `sessions_per_active_day`;
- `beats_per_session`, `single_beat_session_share`;
- `weekend_share` and `night_share` (e.g. 22:00–06:00);
- circular encoding of the average start hour (`sin`/`cos`), plus a
  start-hour entropy measure;
- `days_since_last_session`.

**Where.** `adding_features.py::add_session_features`.

## 4. Level and expansion features

### 4.1 Normalize level by the current cap

**Evidence.** The cap changes from 60 to 70 in April 2007 and to 80 in
November 2008. The raw `level` column therefore has a different meaning before
and after each expansion, and lag/diff features that straddle a cap change mix
two regimes.

**Suggestion.** Add `level_cap` for the month (80 from Nov 2008, 70 from Apr
2007, 60 before), and derive `level_ratio = level / level_cap`,
`is_max_level`, and `level_gain` in cap-relative terms. Keep the raw level too,
but let the model see the normalized version.

**Where.** `adding_features.py`; expansion dates as constants in `main.py`.

### 4.2 Add level-trajectory features

The current monthly feature is only `monthly_max_level_month`. Add:

- `level_gain_1m` / `level_gain_3m` (the generic diff features partially cover
  this, but a dedicated feature is clearer);
- `days_since_level_up`;
- `level_at_month_start` / `level_at_month_end`;
- `level_std` within the month (a character that is power-levelling behaves
  differently from one that is parked at the cap).

**Where.** `adding_features.py::add_monthly_player_features`.

### 4.3 Expansion-phase features

**Evidence.** TBC: 2007-04-03; WotLK: 2008-11-18 (first content 2008-11-19).
The level plot's markers now use these dates.

**Suggestion.** Add `expansion` (a categorical: base/TBC/WotLK) and
`days_since_expansion` (signed), plus an `is_launch_month` flag. These are known
at prediction time and do not leak future information. Interaction terms with
level/activity can capture returning-player behavior. Avoid computing
`days_since_expansion` from the label month in a way that uses future data; the
release date is fixed and known.

**Where.** `adding_features.py`; constants in `main.py` next to `MIN_DATE` and
`MAX_DATE`.

## 5. Playerbase and class features

### 5.1 Use regular characters and robust statistics

**Evidence.** `monthly_class_num_players` and `monthly_class_avg_level` are
computed over all heartbeats. In October 2008, thousands of level-1 Orc/Troll
characters inflate the Warrior/Shaman counts and pull the average level down;
the October anomaly then contaminates the November features.

**Suggestion.**

- Compute class/playerbase counts over regular characters (2.1), or at least
  report both.
- Replace `monthly_class_avg_level` with median and p90 level.
- Add `monthly_class_share` (class count / regular playerbase),
  `monthly_class_new_share`, and `monthly_class_returning_share`.
- Add realm-wide `monthly_regular_players`, `monthly_new_players` and
  `monthly_returning_players`, decomposed from first-seen history.

**Where.** `adding_features.py::add_class_features` and
`add_monthly_player_features`.

### 5.2 Add previous-month churn/retention rates

A prior-month churn rate for the character's class and for the realm is
computable without leakage. It gives the model context about the current regime
and is more robust than raw counts.

**Where.** `adding_features.py`, using the existing `(char, month)` table.

## 6. Guild and zone features

### 6.1 Guild features

**Evidence.** Guild is missing for 17.2% of heartbeats. Guild 460 contains
1,798 members, of which 1,676 were first seen in October 2008 and 1,771 are
below level 10, so raw guild counts/sizes can be dominated by anomalous mass
guilds.

**Suggestion.**

- `is_guilded`;
- `guild_size` measured as regular distinct members in the month, winsorized
  or capped;
- `guild_level_median` / `guild_level_p90`;
- `guild_changes_1m`, `days_since_guild_change`;
- keep first/last guild, but treat guild IDs as high-cardinality categoricals
  and avoid letting a single mass guild dominate.

**Where.** `adding_features.py::add_monthly_player_features`, plus a guild-level
aggregate.

### 6.2 Zone features

The current location features are rarity, hub share (`in_hub`) and Gini. Add:

- `starting_zone_share` (Durotar, Eversong Woods, etc.) — flags low-level alts
  and bots;
- `capital_share` (Orgrimmar, Silvermoon, Undercity, Thunder Bluff);
- `expansion_zone_share` (Outland/Northrend);
- `n_zones_per_active_day` (exploration rate);
- `n_new_zones` (zones seen for the first time by this character) and
  `n_zone_changes` within a session;
- a zone-entropy measure if the Gini coefficient is not enough.

**Where.** `adding_features.py::add_location_features`.

## 7. Labels, splits and evaluation

### 7.1 Make the validation expansion-aware

**Evidence.** `MAX_DATE = 2008-10-01` means the last split point is October
2008 and the last test month is **November 2008**, which contains the WotLK
launch on 18 November. Features for that fold come from October (level cap 70,
pre-launch), while the target comes from December (cap 80, post-launch). The
November fold is a different regime.

**Suggestion.** Pick one of:

- end evaluation before the launch (`MAX_DATE = 2008-09-01`, last test month
  October 2008);
- keep November but report it separately as an out-of-distribution fold and
  add expansion-phase features;
- add an `is_post_expansion` flag and interaction terms.

Also consider a separate "launch month" evaluation rather than mixing it into
the regular monthly splits.

**Where.** `main.py::MIN_DATE`, `main.py::MAX_DATE`, `main.py::Splitter`.

### 7.2 Coverage-adjusted churn labels

**Evidence.** Missing snapshots and Thursday maintenance can make a character
look absent when the collector was simply down. January 2009 is also partial
(10 days), and many characters are observed only once.

**Suggestion.**

- If a month has too few snapshots/days (threshold to choose), set the churn
  label to null or attach a low reliability weight instead of treating absence
  as `has_played=False`.
- Exclude partial months (December 2005 and January 2009) explicitly.
- For characters with less than a minimum history, downweight the label or
  report metrics separately.

**Where.** `add_churn.py::add_churn`, `build_churn_dataset`; `main.py`
(sample weights or masks).

### 7.3 Report metrics on regular characters

Compute ROC AUC / calibration on:

- all rows,
- regular characters only (2.1),
- new characters (first month) vs established characters,
- pre-expansion vs post-expansion months.

This shows whether the model is mostly predicting that one-off characters
disappear, which is easy and not actionable.

**Where.** `main.py::evaluate` and `cross_validate`.

## 8. Encoding, sampling and leakage details

- **Unseen categories.** Death Knights first appear on 2008-11-19; if the
  November 2008 test fold is kept, the training data has no Death Knights.
  Ensure the categorical encoder handles unseen categories (or add an
  `is_new_class`/`is_unknown_category` flag) and report the DK rows separately.
- **Expansion-boundary lags.** `add_lagged_features` computes diffs between
  consecutive months. A diff from March to April 2007 or October to November
  2008 mixes two level-cap regimes. Consider resetting diffs at expansion
  boundaries or adding expansion interactions.
- `sample_by_user` samples whole characters, which is the right unit, but
  one-off characters are a large share of the 90k characters and can dominate
  the training signal. Stratify by tenure/activity, or sample regular
  characters with a higher probability and keep the rest for evaluation.
- **Maintenance windows.** If coverage normalization is not enough, add an
  `is_maintenance_week` flag or exclude Thursday 04:00–12:00 heartbeats from
  count-based features (but be careful not to change the definition of
  `has_played`).

## 9. Suggested implementation order

1. Deduplicate `(char, timestamp)` and add the coverage table (P0).
2. Add `regular` / `is_one_off` / tenure features and regular-only playerbase
   aggregates (P0).
3. Add the regular-character line to the exploration plot (done) and use the
   same criterion in evaluation (P0).
4. Fix the session-gap search space and add censoring-aware session features
   (P1).
5. Add level normalization and expansion-phase features (P1).
6. Decide how to handle the November 2008 launch fold (P1).
7. Add guild/zone refinements and unseen-category handling (P2).

## Things to avoid

- Do not use raw monthly distinct-character counts as a playerbase feature
  without filtering or normalizing: October 2008 is dominated by one-off
  low-level characters.
- Do not use a 15-minute session gap with 10-minute sampling; a single missed
  poll already exceeds it.
- Do not compare raw mean level across expansions; normalize by the level cap.
- Do not interpret absence during a low-coverage or maintenance period as
  churn.
- Do not silently drop one-off characters from the label definition; evaluate
  them separately instead.