from kedro.extras.datasets.spark.spark_dataset import SparkDataSet, _strip_dbfs_prefix
import pyspark.pandas


class SparkDataFrame(SparkDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self) -> pyspark.pandas.frame.DataFrame:
        spark_sql_dataframe = super()._load()
        return spark_sql_dataframe.to_pandas_on_spark()

    def _save(self, data: pyspark.pandas.frame.DataFrame):
        data = data.to_spark()
        return super()._save(data)
