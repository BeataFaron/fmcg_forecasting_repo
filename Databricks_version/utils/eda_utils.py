from pyspark.sql.functions import col, count, when, isnan

def describe_columns(df):
    """
    Returns descriptive statistics for numeric columns.
    """
    return df.describe().toPandas()

def get_missing_summary(df):
    """
    Returns the number of missing (null) values per column.
    Handles all data types.
    """
    total_rows = df.count()
    
    missing_df = df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    
    pdf = missing_df.toPandas().T.rename(columns={0: "missing_count"})
    pdf["missing_percent"] = (pdf["missing_count"] / total_rows * 100).round(2)
    
    return pdf.sort_values(by="missing_count", ascending=False)

def get_unique_counts(df):
    """
    Returns the number of unique values per column.
    """
    return {col_name: df.select(col_name).distinct().count() for col_name in df.columns}
def groupby_summary(df, group_col, target_col, agg_func="sum"):
    """
    Groups the DataFrame by `group_col` and applies aggregation to `target_col`.

    Parameters:
        df (DataFrame): input Spark DataFrame
        group_col (str): column to group by (e.g. 'brand')
        target_col (str): numeric column to aggregate (e.g. 'units_sold')
        agg_func (str): aggregation function: 'sum', 'avg', 'count'

    Returns:
        Pandas DataFrame sorted descending by aggregated value
    """
    from pyspark.sql import functions as F

    agg_map = {
        "sum": F.sum,
        "avg": F.avg,
        "count": F.count
    }

    if agg_func not in agg_map:
        raise ValueError("Unsupported aggregation. Use: 'sum', 'avg', or 'count'")

    result = df.groupBy(group_col).agg(agg_map[agg_func](target_col).alias(f"{agg_func}_{target_col}"))
    return result.orderBy(F.desc(f"{agg_func}_{target_col}")).toPandas()
def value_counts(df, col, normalize=False):
    """
    Returns counts (and optionally percentages) of unique values in a column.

    Parameters:
        df (DataFrame): Spark DataFrame
        col (str): column name
        normalize (bool): if True, returns percentage instead of count

    Returns:
        Pandas DataFrame sorted by count
    """
    total = df.count()
    counts = df.groupBy(col).count().orderBy("count", ascending=False)
    pdf = counts.toPandas().rename(columns={"count": "value_count"})
    
    if normalize:
        pdf["percent"] = (pdf["value_count"] / total * 100).round(2)
    
    return pdf
import matplotlib.pyplot as plt

def plot_grouped(df, group_col, target_col, agg_func="sum", top_n=10):
    """
    Creates a bar plot of aggregated values by category.

    Parameters:
        df (DataFrame): Spark DataFrame
        group_col (str): column to group by
        target_col (str): numeric column to aggregate
        agg_func (str): 'sum', 'avg', or 'count'
        top_n (int): how many top values to plot

    Returns:
        matplotlib figure
    """
    pdf = groupby_summary(df, group_col, target_col, agg_func).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(pdf[group_col], pdf[f"{agg_func}_{target_col}"])
    plt.xlabel(f"{agg_func} of {target_col}")
    plt.title(f"{agg_func.upper()} of {target_col} by {group_col}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.types import NumericType

def correlation_matrix(df):
    """
    Computes correlation matrix for numeric columns and plots a heatmap.
    Skips rows with partial nulls (pairwise complete).
    """
    from pyspark.sql.types import NumericType
    import seaborn as sns
    import matplotlib.pyplot as plt

    # ✅ FIX: find all numeric columns by checking instance of NumericType
    num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

    if not num_cols:
        print("❌ No numeric columns to compute correlation.")
        return None

    pandas_df = df.select(num_cols).toPandas()

    if pandas_df.empty:
        print("❌ DataFrame is empty. Cannot compute correlation.")
        return None

    corr = pandas_df.corr(numeric_only=True)

    if corr.empty:
        print("❌ Correlation matrix is empty.")
        return None

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    return corr
def run_eda(df, target_col="units_sold"):
    """
    Runs full EDA pipeline and prints summaries.
    """

    print(" Descriptive statistics:")
    display(describe_columns(df))

    print(" Missing values:")
    display(get_missing_summary(df))

    print(" Unique values:")
    unique = get_unique_counts(df)
    for col, count in unique.items():
        print(f"{col}: {count} unique")

    print(f"\n Top segments by {target_col}:")
    display(groupby_summary(df, "segment", target_col, agg_func="sum"))

    print(f"\n Top channels by {target_col}:")
    display(groupby_summary(df, "channel", target_col, agg_func="sum"))

    print(f"\n Top brands by {target_col}:")
    display(groupby_summary(df, "brand", target_col, agg_func="sum"))

    print("\n Correlation matrix:")
    correlation_matrix(df)
