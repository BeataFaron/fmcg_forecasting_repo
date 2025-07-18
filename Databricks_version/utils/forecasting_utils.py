import mlflow
import mlflow.spark

from pyspark.sql.functions import col, lag, avg, stddev, when, expr, weekofyear, year, sum as _sum
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

# Feature engineering on daily level (lags, rolling, momentum)
def engineer_features_daily(df):
    w = Window.orderBy("date")
    
    df = df.withColumn("lag_1", lag("units_sold", 1).over(w))
    df = df.withColumn("lag_2", lag("units_sold", 2).over(w))
    df = df.withColumn("rolling_mean_4", avg("units_sold").over(w.rowsBetween(-3, 0)))
    df = df.withColumn("rolling_std_4", stddev("units_sold").over(w.rowsBetween(-3, 0)))
    df = df.withColumn("momentum", (col("lag_1") - col("lag_2")) / (col("lag_2") + 1e-5))
    df = df.withColumn("avg_by_channel_region", avg("units_sold").over(Window.partitionBy("channel", "region")))

    return df

# Time-aware split for time series
def time_split(df, date_col="date", split_ratio=0.8):
    """
    Perform a time-aware train-test split on a PySpark DataFrame.
    Uses distinct dates to find the cutoff point.
    """
    # Get distinct, sorted dates from the DataFrame
    dates = df.select(date_col).distinct().orderBy(date_col).toPandas().reset_index(drop=True)
    
    # Make sure the split index doesn't go out of bounds
    split_idx = int(len(dates) * split_ratio)
    if split_idx >= len(dates):
        split_idx = len(dates) - 1

    split_point = dates.loc[split_idx, date_col]

    # Split the DataFrame based on the cutoff date
    train = df.filter(col(date_col) <= F.lit(split_point))
    test = df.filter(col(date_col) > F.lit(split_point))

    print(f"📆 Time-aware split at: {split_point} → Train: {train.count()} rows, Test: {test.count()} rows")
    return train, test

# Train model and evaluate
def train_model(df_train, df_test, features, label="target_next_week"):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    model = RandomForestRegressor(featuresCol="features", labelCol=label)
    pipeline = Pipeline(stages=[assembler, model])
    fitted = pipeline.fit(df_train)
    preds = fitted.transform(df_test)

    rmse = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse").evaluate(preds)
    mae = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="mae").evaluate(preds)
    print(f"✅ RMSE: {rmse:.2f}  |  MAE: {mae:.2f}")
    return fitted, preds, rmse, mae

# Aggregate predictions to weekly level
def aggregate_to_week(df, date_col="date", pred_col="prediction", true_col="units_sold"):
    df = df.withColumn("week", weekofyear(col(date_col)))
    df = df.withColumn("year", col(date_col).substr(1, 4).cast("int"))
    df_weekly = df.groupBy("week", "year").agg(
        _sum(pred_col).alias("predicted"),
        _sum(true_col).alias("actual")
    )
    return df_weekly

# Log model, params and metrics to MLflow
def log_model_with_metrics(model, rmse, mae, features, sku_id="MI-006", data_type="daily"):
    with mlflow.start_run():
        mlflow.log_param("sku", sku_id)
        mlflow.log_param("data_type", data_type)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.set_tag("pipeline", "random_forest")

        mlflow.spark.log_model(model, "model")
        print("✅ Model and metrics logged to MLflow.")
