from pyspark.sql import SparkSession

def load_data(file_path: str, file_format: str = "parquet"):
    """
    Loads data from a CSV or Parquet file into a Spark DataFrame.
    """
    spark = SparkSession.builder.getOrCreate()

    if file_format == "csv":
        df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)
    elif file_format == "parquet":
        df = spark.read.parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'parquet'.")

    return df
