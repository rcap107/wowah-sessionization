# WoWAH User Churn Prediction Pipeline

A predictive pipeline for forecasting World of Warcraft (WoWAH) player churn.
This project predicts whether a player will churn (stop playing) in the next
month based on historical gaming behavior and session data.

## Overview

The pipeline constructs a machine learning model that predicts monthly user churn
while carefully avoiding data leakage. It uses a temporal cross-validation strategy where:

- **Training data**: All historical data up to a given month
- **Test data**: Data from the following month only
- **Target**: Whether a user will play in the month after the test month

## Key Components

### `add_churn.py` - Preparation of the churn feature

### `main.py` - Core Pipeline

- **Splitter class**: Custom temporal cross-validator that splits data by month to prevent leakage
- **add_features()**: Constructs features from historical data using session encoding and aggregations
- **make_data_op()**: Builds the complete ML pipeline using HistGradientBoostingClassifier
- Functions to cross-validate and evaluate the model

### `utils.py` - Utility Functions

Provides helper functions for feature engineering:

- `get_session_duration()`: Calculates session durations with minimum 1-minute floor
- `sample_by_user()`: Samples a fraction of users for analysis

### `adding_features.py` - Feature Engineering on user sessions and playerbase

Constructs sophisticated features:

- **Session Features**: Monthly session count, total/average duration
- **Player Features**: Max level, unique zones, guilds per month
- **Class Features**: Monthly class-level statistics

### `location_features.py` - Feature Engineering on user locations

- **location features**:
  - zone rarity (log ratio of unique players to visitors)
  - hub identification (frequently visited zones)
  - player location diversity (gini coefficient)
  - average zone levels

## Data Requirements

- `data/wowah_churn_data.parquet`: User-month churn labels with columns: `char`, `month`, `has_played`
- `data/wowah_data_raw.parquet`: Raw game logs with columns: `char`, `timestamp`, `zone`, `level`, `race`, `charclass`, `guild`

## Usage

Run `main.py` to execute execute a crossvalidation run with default parameters and
print the results.

## Dependencies

- `polars`: Data processing and manipulation
- `skrub`: ML pipeline construction and transformers
- `scikit-learn`: Machine learning models and utilities
- `datetime`: Time-based operations

## Model Details

- **Algorithm**: HistGradientBoostingClassifier
- **Session Gap**: 30 or 60 minutes (configurable)
- **Location Features**: Optional (can be disabled for performance)
- **Temporal Validation**: Month-based splitting to prevent leakage
