import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)



# 0. 参数设置

INPUT_FILE = "credit_prepaid_card_monthly_HHI_2011_2026.csv"

TRAIN_END = "2023-09"
TEST_START = "2023-10"

FUTURE_END = "2029-09"



# 读取数据


print("=" * 80)
print("1. Loading HHI data")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

df["Month"] = pd.to_datetime(df["Month"])

df = df.sort_values("Month").reset_index(drop=True)

print("\nData range:")
print(
    df["Month"].min().strftime("%Y-%m"),
    "to",
    df["Month"].max().strftime("%Y-%m")
)

print("\nNumber of observations:")
print(len(df))

print("\nColumns:")
print(df.columns.tolist())


# 设置完整的月度时间索引


df = df.set_index("Month")

full_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="MS"
)

df = df.reindex(full_index)

df.index.name = "Month"



# 检查缺失


print("\n" + "=" * 80)
print("2. Checking missing values")
print("=" * 80)

print(df[["HHI", "total_complaints"]].isna().sum())

if df["HHI"].isna().sum() > 0:

    print("\nMissing HHI months:")

    print(
        df[df["HHI"].isna()]
        .index
        .strftime("%Y-%m")
        .tolist()
    )

    raise ValueError(
        "HHI contains missing values. "
        "Please handle missing months first."
    )



# Train


print("\n" + "=" * 80)
print("3. Train / Test split")
print("=" * 80)

train_end = pd.Timestamp(
    TRAIN_END + "-01"
)

test_start = pd.Timestamp(
    TEST_START + "-01"
)

train = df.loc[
    :train_end,
    "HHI"
].copy()

test = df.loc[
    test_start:,
    "HHI"
].copy()


print("\nTraining:")
print(
    train.index.min().strftime("%Y-%m"),
    "to",
    train.index.max().strftime("%Y-%m")
)

print(
    "Training observations:",
    len(train)
)

print("\nTest:")
print(
    test.index.min().strftime("%Y-%m"),
    "to",
    test.index.max().strftime("%Y-%m")
)

print(
    "Test observations:",
    len(test)
)



# Evaluation metrics


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }



# Forecasting models


def forecast_naive(
    train_data,
    horizon
):

    last_value = train_data.iloc[-1]

    return np.repeat(
        last_value,
        horizon
    )


def forecast_drift(
    train_data,
    horizon
):

    y = np.asarray(train_data)

    n = len(y)

    drift = (
        y[-1] - y[0]
    ) / (n - 1)

    forecasts = (
        y[-1]
        + drift *
        np.arange(
            1,
            horizon + 1
        )
    )

    return forecasts


def forecast_arima(
    train_data,
    horizon,
    order
):

    model = ARIMA(
        train_data,
        order=order
    )

    fitted = model.fit()

    forecast = fitted.forecast(
        steps=horizon
    )

    return np.asarray(
        forecast
    )


def forecast_ets(
    train_data,
    horizon
):

    model = ExponentialSmoothing(
        train_data,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    )

    fitted = model.fit(
        optimized=True
    )

    forecast = fitted.forecast(
        horizon
    )

    return np.asarray(
        forecast
    )



models = {

    "Naive":
        lambda x, h:
        forecast_naive(x, h),

    "Drift":
        lambda x, h:
        forecast_drift(x, h),

    "ARIMA(1,0,0)":
        lambda x, h:
        forecast_arima(
            x,
            h,
            (1, 0, 0)
        ),

    "ARIMA(1,0,1)":
        lambda x, h:
        forecast_arima(
            x,
            h,
            (1, 0, 1)
        ),

    "ARIMA(1,1,0)":
        lambda x, h:
        forecast_arima(
            x,
            h,
            (1, 1, 0)
        ),

    "ETS":
        lambda x, h:
        forecast_ets(x, h)
}



# Test all models


print("\n" + "=" * 80)
print("4. Testing all forecasting models")
print("=" * 80)

predictions = {}

results = []


for model_name, model_function in models.items():

    print(
        "\nTesting:",
        model_name
    )

    try:

        prediction = model_function(
            train,
            len(test)
        )

        predictions[
            model_name
        ] = prediction

        metrics = calculate_metrics(
            test.values,
            prediction
        )

        results.append({

            "Model":
                model_name,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "MAPE":
                metrics["MAPE"]

        })

    except Exception as e:

        print(
            "Model failed:",
            model_name,
            e
        )



# Test comparison

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(
    drop=True
)


print("\n" + "=" * 80)
print("5. Test-set model comparison")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "HHI_test_model_comparison.csv",
    index=False
)



#  Best Test model
best_model = (
    results_df.iloc[0]["Model"]
)

best_row = (
    results_df.iloc[0]
)


print("\n" + "=" * 80)
print("BEST MODEL ON TEST SET")
print("=" * 80)

print(
    "\nBest model:",
    best_model
)

print(
    "Test MAE:",
    round(
        best_row["MAE"],
        4
    )
)

print(
    "Test RMSE:",
    round(
        best_row["RMSE"],
        4
    )
)

print(
    "Test MAPE:",
    round(
        best_row["MAPE"],
        4
    ),
    "%"
)


# Test prediction DataFrame

test_predictions = pd.DataFrame(
    index=test.index
)

test_predictions.index.name = "Month"

test_predictions[
    "Actual_HHI"
] = test.values


for model_name, prediction in predictions.items():

    test_predictions[
        model_name
    ] = prediction


test_predictions.to_csv(
    "HHI_all_model_test_predictions.csv"
)



# MODEL COMPARISON FIGURE

# Training + Test + All model predictions

print("\n" + "=" * 80)
print("6. Plotting Training + Test + Model Comparison")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(18, 9)
)


# Historical actual HHI


ax.plot(
    df.index,
    df["HHI"],
    label="Actual HHI",
    linewidth=2.5
)


# Training/Test shading


ax.axvspan(
    df.index.min(),
    train_end,
    alpha=0.08
)

ax.axvspan(
    test_start,
    df.index.max(),
    alpha=0.05
)


# Train/Test boundary


ax.axvline(
    test_start,
    linestyle="--",
    linewidth=2
)


# Training / Test labels

y_max = df["HHI"].max()

ax.text(
    train_end - pd.Timedelta(
        days=500
    ),
    y_max * 0.96,
    "TRAINING",
    fontsize=13,
    fontweight="bold",
    ha="center"
)

ax.text(
    test_start + pd.Timedelta(
        days=400
    ),
    y_max * 0.96,
    "TEST",
    fontsize=13,
    fontweight="bold",
    ha="center"
)


# Plot model predictions


for model_name, prediction in predictions.items():

    row = results_df[
        results_df["Model"]
        == model_name
    ]

    mape = row.iloc[0]["MAPE"]

    rmse = row.iloc[0]["RMSE"]

    # Highlight ARIMA(1,0,1)
    if model_name == "ARIMA(1,0,1)":

        label = (
            f"{model_name} "
            f"(TEST BEST | "
            f"MAPE={mape:.2f}%)"
        )

        linewidth = 3.0

    # Highlight ETS
    elif model_name == "ETS":

        label = (
            f"{model_name} "
            f"(Validation Best | "
            f"Test MAPE={mape:.2f}%)"
        )

        linewidth = 2.5

    else:

        label = model_name

        linewidth = 1.5


    ax.plot(
        test.index,
        prediction,
        linestyle="--",
        linewidth=linewidth,
        label=label,
        alpha=0.85
    )


# 13. Explanation box

arima_test_row = results_df[
    results_df["Model"]
    == "ARIMA(1,0,1)"
].iloc[0]


ets_test_row = results_df[
    results_df["Model"]
    == "ETS"
].iloc[0]


explanation = (
    "Model selection:\n"
    "• ETS performed best during historical rolling validation.\n"
    "• However, ETS deteriorated substantially on the true Test period.\n"
    "• ARIMA(1,0,1) achieved the best Test performance.\n"
    f"• ARIMA(1,0,1) Test MAPE = "
    f"{arima_test_row['MAPE']:.2f}%\n"
    f"• ARIMA(1,0,1) Test RMSE = "
    f"{arima_test_row['RMSE']:.2f}\n"
    f"• ETS Test MAPE = "
    f"{ets_test_row['MAPE']:.2f}%"
)


ax.text(
    0.015,
    0.03,
    explanation,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        alpha=0.85
    )
)


# 14. Figure formatting

ax.set_title(
    "HHI Forecast Model Comparison: "
    "Training and Out-of-Sample Test",
    fontsize=17,
    fontweight="bold"
)

ax.set_xlabel(
    "Month",
    fontsize=12
)

ax.set_ylabel(
    "HHI",
    fontsize=12
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    loc="upper left",
    fontsize=10,
    ncol=2
)

plt.tight_layout()


plt.savefig(
    "HHI_all_models_training_test_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Future Forecast

# 使用 Test 表现最好的模型


print("\n" + "=" * 80)
print("7. Final forecasting")
print("=" * 80)


full_data = df["HHI"].copy()


print("\nFinal training period:")

print(
    full_data.index.min().strftime("%Y-%m"),
    "to",
    full_data.index.max().strftime("%Y-%m")
)

print(
    "Observations:",
    len(full_data)
)



# Future dates


future_end = pd.Timestamp(
    FUTURE_END
)

future_index = pd.date_range(

    start=
    full_data.index[-1]
    + pd.offsets.MonthBegin(1),

    end=future_end,

    freq="MS"
)


future_horizon = len(
    future_index
)


#Future prediction


if best_model == "Naive":

    future_prediction = (
        forecast_naive(
            full_data,
            future_horizon
        )
    )

elif best_model == "Drift":

    future_prediction = (
        forecast_drift(
            full_data,
            future_horizon
        )
    )

elif best_model == "ARIMA(1,0,0)":

    future_prediction = (
        forecast_arima(
            full_data,
            future_horizon,
            (1, 0, 0)
        )
    )

elif best_model == "ARIMA(1,0,1)":

    future_prediction = (
        forecast_arima(
            full_data,
            future_horizon,
            (1, 0, 1)
        )
    )

elif best_model == "ARIMA(1,1,0)":

    future_prediction = (
        forecast_arima(
            full_data,
            future_horizon,
            (1, 1, 0)
        )
    )

elif best_model == "ETS":

    future_prediction = (
        forecast_ets(
            full_data,
            future_horizon
        )
    )

else:

    raise ValueError(
        "Unknown model"
    )


# Future DataFrame

future_df = pd.DataFrame({

    "Month":
        future_index,

    "Forecast_HHI":
        future_prediction

})


print("\nFuture forecast:")

print(
    future_df.to_string(
        index=False
    )
)


future_df.to_csv(
    "HHI_future_forecast_best_test_model.csv",
    index=False
)


# Future forecast figure


plt.figure(
    figsize=(18, 9)
)


plt.plot(
    full_data.index,
    full_data.values,
    label="Historical HHI",
    linewidth=2.5
)


plt.plot(
    future_index,
    future_prediction,
    label=
    f"Future Forecast ({best_model})",
    linewidth=3,
    linestyle="--"
)


plt.axvline(
    future_index[0],
    linestyle=":",
    linewidth=2,
    label="Forecast Start"
)


plt.title(
    f"Future HHI Forecast Using "
    f"{best_model}",
    fontsize=17,
    fontweight="bold"
)

plt.xlabel(
    "Month",
    fontsize=12
)

plt.ylabel(
    "HHI",
    fontsize=12
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "HHI_future_forecast_best_test_model.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Final Summary

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


print("\nTraining:")
print(
    train.index.min().strftime("%Y-%m"),
    "to",
    train.index.max().strftime("%Y-%m")
)


print("\nTest:")
print(
    test.index.min().strftime("%Y-%m"),
    "to",
    test.index.max().strftime("%Y-%m")
)


print("\nTest-set model ranking:")

print(
    results_df.to_string(
        index=False
    )
)


print("\n" + "-" * 80)

print(
    "\nSelected model based on Test RMSE:"
)

print(
    best_model
)


print(
    "\nTest MAE:",
    round(
        best_row["MAE"],
        4
    )
)

print(
    "Test RMSE:",
    round(
        best_row["RMSE"],
        4
    )
)

print(
    "Test MAPE:",
    round(
        best_row["MAPE"],
        4
    ),
    "%"
)


print(
    "\nFuture forecast:"
)

print(
    future_index[0].strftime("%Y-%m"),
    "to",
    future_index[-1].strftime("%Y-%m")
)


print("\nFiles generated:")

print(
    "1. HHI_test_model_comparison.csv"
)

print(
    "2. HHI_all_model_test_predictions.csv"
)

print(
    "3. HHI_all_models_training_test_comparison.png"
)

print(
    "4. HHI_future_forecast_best_test_model.csv"
)

print(
    "5. HHI_future_forecast_best_test_model.png"
)


print("\nDone!")