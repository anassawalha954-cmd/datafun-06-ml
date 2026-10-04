"""src/datafun/app.py - Project script.

Author: Anas
Date: 2026-10

HOW TO RUN THIS FILE:

From the VS Code menu (with only this project open in VS Code),
click "Terminal" / New Terminal to
open an integrated Terminal in the root project folder.
Paste the following command and press ENTER or RETURN
to run this file as a script:

uv run python -m datafun.app

DOMAIN:

A dataset of penguins.
See docs/data-card.md for more information about the dataset.

EXPLORE:

Earlier analysis showed relationships among
numeric penguin measurements.

In this project, we use one numeric feature
to predict one numeric target
with a simple linear regression model.

CUSTOM MODIFICATION:

I changed the main feature from bill length to flipper length,
and added a comparison of several candidate features
to see which single measurement predicts body mass best.

A standard predictive modeling process is:

1. OBSERVE the data and prior findings.
2. DECLARE the target and feature.
3. PREPARE the modeling data.
4. SPLIT into training and test data.
5. BASELINE with a simple reference model.
6. TRAIN a LinearRegression model.
7. PREDICT on X_test.
8. EVALUATE baseline vs model on y_test.
9. COMPARE candidate features (custom addition).
10. VISUALIZE predictions and residuals.
11. ASSESS the results.
"""

# === DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
from ml_vizkit import save_chart
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# === CONFIGURE LOGGER ONCE FOR THE APPLICATION ===

LOG: logging.Logger = get_logger("P06", level="DEBUG")

# === DECLARE GLOBAL CONSTANTS ===

DATA_FILE_PATH: Final[Path] = Path("data") / "raw" / "penguins.csv"

CHART_DIR: Final[Path] = Path("docs") / "images"

PREDICTION_CHART_PATH: Final[Path] = CHART_DIR / "regression-predictions.png"
RESIDUAL_CHART_PATH: Final[Path] = CHART_DIR / "regression-residuals.png"
COMPARISON_CHART_PATH: Final[Path] = CHART_DIR / "feature-comparison.png"

GRAIN: Final[str] = "one penguin"

# === DECLARE THE TARGET ===

TARGET_COLUMN: Final[str] = "body_mass_g"

# === DECLARE THE FEATURE (CUSTOM CHANGE) ===

FEATURE_COLUMN: Final[str] = "flipper_length_mm"

# Candidate features for the comparison step.

CANDIDATE_FEATURES: Final[list[str]] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
]

FEATURE_DECISION: Final[str] = r"""
I want to predict body mass.

I selected flipper length as the main feature.

Flipper length reflects overall body size.
A penguin with longer flippers is likely larger
and may therefore weigh more.

In an earlier version I used bill length.
I want to test whether flipper length is a better predictor.

I do not know yet how well it will work.
The modeling process will provide evidence.
"""

# === DECLARE THE TRAIN / TEST SPLIT ===

TEST_FRACTION: Final[float] = 0.20
RANDOM_SEED: Final[int] = 42

SPLIT_DECISION: Final[str] = r"""
I will use 80% of the modeling rows for training
and hold back 20% for testing.

Most of the data is available for learning,
while a separate set of unseen observations
is kept for evaluating the model.

I use a fixed random seed of 42.
The value itself is not analytically important.
It makes the split reproducible,
so results are easy to repeat and compare.
"""

# === DECLARE THE BASELINE ===

BASELINE_STRATEGY: Final[str] = "mean"

BASELINE_DECISION: Final[str] = r"""
Before evaluating the LinearRegression model,
I need a simple baseline for comparison.

The baseline ignores the feature
and predicts the average body mass
from the training data for every test observation.

A useful model should improve on this reference.
"""

MODEL_DECISION: Final[str] = r"""
I will use LinearRegression.

It fits a straight-line relationship
between the selected feature and the target.
This gives a simple, interpretable model
that can be compared with the baseline.

Fitting a line does not prove a straight line
is a good description of the relationship.
The metrics and the residual plot
will help assess whether the model is useful.
"""


# === HELPER FUNCTION (CUSTOM ADDITION) ===


def compare_features(
    df: pd.DataFrame,
    target: str,
    features: list[str],
) -> pd.DataFrame:
    """Train one simple linear model per feature and compare them.

    All features use the same rows and the same split,
    so the comparison is fair.

    Arguments:
        df: The full DataFrame.
        target: The numeric target column.
        features: The candidate numeric feature columns.

    Returns:
        A DataFrame with RMSE and R-squared per feature,
        sorted from best to worst R-squared.
    """
    df_common: pd.DataFrame = df.dropna(subset=[target, *features]).copy()

    train_idx, test_idx = train_test_split(
        df_common.index,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
    )

    y_train: pd.Series = df_common.loc[train_idx, target]
    y_test: pd.Series = df_common.loc[test_idx, target]

    rows: list[dict[str, float | str]] = []

    for feature in features:
        X_train: pd.DataFrame = df_common.loc[train_idx, [feature]]
        X_test: pd.DataFrame = df_common.loc[test_idx, [feature]]

        candidate_model = LinearRegression()
        candidate_model.fit(X_train, y_train)
        predictions: np.ndarray = candidate_model.predict(X_test)

        rows.append(
            {
                "feature": feature,
                "rmse": float(root_mean_squared_error(y_test, predictions)),
                "r_squared": float(r2_score(y_test, predictions)),
            }
        )

    return pd.DataFrame(rows).sort_values("r_squared", ascending=False)


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Entry point when running this file as a Python script.

    Arguments: None.
    Returns: None.
    """
    log_header(LOG, "P06 - LINEAR REGRESSION")

    LOG.info("===================================")
    LOG.info("START main()")
    LOG.info("===================================")

    LOG.info("-------------------------------")
    LOG.info("01. OBSERVE the data and prior findings.")
    LOG.info("-------------------------------")

    log_path(LOG, "data file", path=DATA_FILE_PATH)

    df: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)

    LOG.info("Data loaded successfully.")
    LOG.info(f"Grain: {GRAIN}")
    LOG.info(f"Rows: {df.shape[0]}")
    LOG.info(f"Columns: {df.shape[1]}")
    LOG.info(f"Column names: {df.columns.tolist()}")

    LOG.info("-------------------------------")
    LOG.info("02. DECLARE the target and feature.")
    LOG.info("-------------------------------")

    LOG.info(f"Target:  {TARGET_COLUMN}")
    LOG.info(f"Feature: {FEATURE_COLUMN}")
    LOG.info(FEATURE_DECISION)

    LOG.info("-------------------------------")
    LOG.info("03. PREPARE the modeling data.")
    LOG.info("-------------------------------")

    required_columns: list[str] = [FEATURE_COLUMN, TARGET_COLUMN]

    df_model: pd.DataFrame = df.dropna(subset=required_columns).copy()

    count_original: int = df.shape[0]
    count_model: int = df_model.shape[0]
    count_dropped: int = count_original - count_model

    LOG.info(f"Original rows: {count_original}")
    LOG.info(f"Modeling rows: {count_model}")
    LOG.info(f"Rows dropped: {count_dropped}")

    X: pd.DataFrame = df_model[[FEATURE_COLUMN]]
    y: pd.Series = df_model[TARGET_COLUMN]

    LOG.info(f"X shape: {X.shape}")
    LOG.info(f"y shape: {y.shape}")

    LOG.info("-------------------------------")
    LOG.info("04. SPLIT into training and test data.")
    LOG.info("-------------------------------")

    LOG.info(SPLIT_DECISION)

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
    )

    LOG.info(f"Training rows: {X_train.shape[0]}")
    LOG.info(f"Test rows: {X_test.shape[0]}")

    LOG.info("-------------------------------")
    LOG.info("05. BASELINE with a simple reference model.")
    LOG.info("-------------------------------")

    LOG.info(BASELINE_DECISION)

    baseline_model = DummyRegressor(strategy=BASELINE_STRATEGY)
    baseline_model.fit(X_train, y_train)

    baseline_predictions: np.ndarray = baseline_model.predict(X_test)

    baseline_rmse: float = float(root_mean_squared_error(y_test, baseline_predictions))
    baseline_r_squared: float = float(r2_score(y_test, baseline_predictions))

    LOG.info(f"Baseline strategy: {BASELINE_STRATEGY}")
    LOG.info(f"Baseline RMSE: {baseline_rmse:.2f}")
    LOG.info(f"Baseline R-squared: {baseline_r_squared:.3f}")

    LOG.info("-------------------------------")
    LOG.info("06. TRAIN a LinearRegression model.")
    LOG.info("-------------------------------")

    LOG.info(MODEL_DECISION)

    model = LinearRegression()
    model.fit(X_train, y_train)

    slope: float = float(model.coef_[0])
    intercept: float = float(model.intercept_)

    LOG.info("The model learned this line:")
    LOG.info(f"{TARGET_COLUMN} = {slope:.3f} * {FEATURE_COLUMN} + {intercept:.3f}")

    LOG.info("-------------------------------")
    LOG.info("07. PREDICT on X_test.")
    LOG.info("-------------------------------")

    model_predictions: np.ndarray = model.predict(X_test)

    LOG.info(f"Predictions created: {len(model_predictions)}")

    LOG.info("-------------------------------")
    LOG.info("08. EVALUATE baseline vs model on y_test.")
    LOG.info("-------------------------------")

    model_rmse: float = float(root_mean_squared_error(y_test, model_predictions))
    model_r_squared: float = float(r2_score(y_test, model_predictions))

    LOG.info("BASELINE RESULTS")
    LOG.info(f"RMSE:      {baseline_rmse:.2f}")
    LOG.info(f"R-squared: {baseline_r_squared:.3f}")

    LOG.info("LINEAR REGRESSION RESULTS")
    LOG.info(f"RMSE:      {model_rmse:.2f}")
    LOG.info(f"R-squared: {model_r_squared:.3f}")

    LOG.info("-------------------------------")
    LOG.info("09. COMPARE candidate features (custom addition).")
    LOG.info("-------------------------------")

    LOG.info(f"Candidates: {CANDIDATE_FEATURES}")

    comparison: pd.DataFrame = compare_features(
        df,
        TARGET_COLUMN,
        CANDIDATE_FEATURES,
    )

    for _, row in comparison.iterrows():
        LOG.info(
            f"{row['feature']:<20} "
            f"RMSE: {row['rmse']:.2f}  "
            f"R-squared: {row['r_squared']:.3f}"
        )

    best_feature: str = str(comparison.iloc[0]["feature"])
    LOG.info(f"Best single feature by R-squared: {best_feature}")

    LOG.info("-------------------------------")
    LOG.info("10. VISUALIZE predictions, residuals, and comparison.")
    LOG.info("-------------------------------")

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # === PREDICTIONS CHART ===

    _prediction_figure, prediction_ax = plt.subplots()

    x_test_values: np.ndarray = X_test[FEATURE_COLUMN].to_numpy()
    y_test_values: np.ndarray = y_test.to_numpy()

    prediction_ax.scatter(x_test_values, y_test_values, label="Actual")

    prediction_order: np.ndarray = np.argsort(x_test_values)

    prediction_ax.plot(
        x_test_values[prediction_order],
        model_predictions[prediction_order],
        label="Predicted",
    )

    prediction_ax.set_title("Flipper Length vs. Body Mass")
    prediction_ax.set_xlabel("Flipper Length (mm)")
    prediction_ax.set_ylabel("Body Mass (g)")
    prediction_ax.legend()

    save_chart(prediction_ax, PREDICTION_CHART_PATH)

    LOG.info(f"Chart saved successfully at {PREDICTION_CHART_PATH}.")

    # === RESIDUAL CHART ===

    residuals: np.ndarray = y_test_values - model_predictions

    _residual_figure, residual_ax = plt.subplots()

    residual_ax.scatter(x_test_values, residuals)
    residual_ax.axhline(0)

    residual_ax.set_title("Residuals for Flipper Length Model")
    residual_ax.set_xlabel("Flipper Length (mm)")
    residual_ax.set_ylabel("Residual (Actual - Predicted Body Mass)")

    save_chart(residual_ax, RESIDUAL_CHART_PATH)

    LOG.info(f"Chart saved successfully at {RESIDUAL_CHART_PATH}.")

    # === FEATURE COMPARISON CHART (CUSTOM) ===

    _comparison_figure, comparison_ax = plt.subplots()

    comparison_ax.bar(
        comparison["feature"].tolist(),
        comparison["r_squared"].tolist(),
    )

    comparison_ax.set_title("Which Single Feature Predicts Body Mass Best?")
    comparison_ax.set_xlabel("Feature")
    comparison_ax.set_ylabel("R-squared on Test Data")

    save_chart(comparison_ax, COMPARISON_CHART_PATH)

    LOG.info(f"Chart saved successfully at {COMPARISON_CHART_PATH}.")

    LOG.info("-------------------------------")
    LOG.info("11. ASSESS the results.")
    LOG.info("-------------------------------")

    # Run the app, then replace each ... with your real numbers
    # and observations before submitting.

    LOG.info(r"""CUSTOM OBSERVATIONS:
    I used flipper length to predict body mass.

    The baseline RMSE was ...
    The LinearRegression RMSE was ...

    Compared with the baseline,
    the LinearRegression model ...

    The model R-squared was ...

    Comparing features, the best predictor was ...
    and the weakest was ...

    In the residual plot, I observed ...

    Based on this evidence,
    I conclude ...

    Next, I would like to try ...
    (for example: training a separate model per species)
    """)

    LOG.info("In a script, call plt.show() at the end to display all charts.")
    LOG.info("Close all chart windows (with the close button) to continue.")

    plt.show()

    LOG.info("===================================")
    LOG.info("END main() - Executed successfully!")
    LOG.info("===================================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()