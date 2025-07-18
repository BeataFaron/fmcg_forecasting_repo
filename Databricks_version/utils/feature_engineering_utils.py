from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, weekofyear, dayofweek, month, year,
    when, avg, lag, lead, stddev, mean, count,
    lit, to_date, date_trunc, date_format)
from pyspark.sql.window import Window

def create_weekly_features(df_daily: DataFrame) -> DataFrame:
    """
    Transforms daily data into engineered weekly features for modeling.

    Parameters:
        df_daily: Spark DataFrame with daily-level data

    Returns:
        Spark DataFrame with weekly-level features
    """
    
    # 🎯 Step 1: Extract calendar features
    df = df_daily.withColumn("day_of_week", dayofweek("date") - 1) \
                 .withColumn("week_number", weekofyear("date")) \
                 .withColumn("month", month("date")) \
                 .withColumn("year", year("date")) \
                 .withColumn("is_weekend", when(col("day_of_week").isin([5,6]), 1).otherwise(0)) \
                 .withColumn("is_holiday_peak", when(col("month").isin([11,12]), 1).otherwise(0)) \
                 .withColumn("is_summer", when(col("month").isin([6,7,8]), 1).otherwise(0)) \
                 .withColumn("is_winter", when(col("month").isin([12,1,2]), 1).otherwise(0)) \
                 .withColumn("week", date_trunc("week", col("date")))

    # 🎯 Step 2: Aggregate to weekly level
    from pyspark.sql import functions as F

    weekly_df = df.groupBy("sku","channel", "region", "week", "week_number", "month", "year").agg(
        F.sum("units_sold").alias("units_sold"),
        F.avg("stock_available").alias("stock_available"),
        F.avg("price_unit").alias("price_unit"),
        F.avg("promotion_flag").alias("promotion_flag"),
        F.avg("delivery_days").alias("delivery_days"),
        F.max("is_holiday_peak").alias("is_holiday_peak"),
        F.max("is_summer").alias("is_summer"),
        F.max("is_winter").alias("is_winter")
    )
    # Rename aggregated columns
    weekly_df = weekly_df.withColumnRenamed("units_sold", "units_sold") \
                         .withColumnRenamed("stock_available", "stock_available") \
                         .withColumnRenamed("price_unit", "price_unit") \
                         .withColumnRenamed("promotion_flag", "promotion_flag") \
                         .withColumnRenamed("delivery_days", "delivery_days")

    # 🎯 Step 3: Create lag & rolling features
    window_spec = Window.partitionBy("channel", "region").orderBy("week")

    weekly_df = weekly_df \
        .withColumn("lag_1", lag("units_sold", 1).over(window_spec)) \
        .withColumn("lag_2", lag("units_sold", 2).over(window_spec)) \
        .withColumn("rolling_mean_4", mean("units_sold").over(window_spec.rowsBetween(-3, 0))) \
        .withColumn("rolling_std_4", stddev("units_sold").over(window_spec.rowsBetween(-3, 0))) \
        .withColumn("momentum", (col("units_sold") - col("lag_1")) / (col("lag_1") + lit(1e-5)))  # avoid division by zero

    # 🎯 Step 4: Create proxy feature (avg by channel-region)
    proxy_avg = weekly_df.select("channel", "region", "units_sold") \
        .groupBy("channel", "region") \
        .agg(F.avg("units_sold").alias("avg_by_channel_region")) \
        .dropDuplicates(["channel", "region"])

    weekly_df = weekly_df.join(proxy_avg, on=["channel", "region"], how="left")

    # 🎯 Step 5: Create target variable (next week's units_sold)
    weekly_df = weekly_df.withColumn("target_next_week", lead("units_sold", 1).over(window_spec))
    # 📆 Add readable week string (optional, for plotting/export)
    weekly_df = weekly_df.withColumn("week_str", date_format("week", "yyyy-MM-dd"))

    return weekly_df
