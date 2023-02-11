from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # # Load the spark configuration in spark.yaml using the config loader
        # parameters = context.config_loader.get("parameters*", "parameters/**")["spark"]
        spark_conf = SparkConf()
        spark_conf.set('spark.driver.maxResultSize', '1g')
        spark_conf.set('spark.driver.memory', '8g')
        spark_conf.set('spark.executor.memory', '4g')
        spark_conf.set('spark.scheduler.mode', 'FAIR')
        spark_conf.set('spark.cores.max', '4')

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(context._package_name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("OFF")