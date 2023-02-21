from kedro.extras.datasets.spark.spark_dataset import SparkDataSet, _strip_dbfs_prefix
import pyspark.pandas


class SparkDataFrame(SparkDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self) -> pyspark.pandas.frame.DataFrame:
        load_path = _strip_dbfs_prefix(self._fs_prefix + str(self._get_load_path()))
        read_obj = self._get_spark().read

        # Pass schema if defined
        if self._schema:
            read_obj = read_obj.schema(self._schema)

        spark_sql_dataframe = read_obj.load(load_path, self._file_format, **self._load_args)
        spark_pandas_dataframe = spark_sql_dataframe.to_pandas_on_spark()
        return spark_pandas_dataframe
