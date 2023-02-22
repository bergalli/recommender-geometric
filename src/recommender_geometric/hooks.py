from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
# import ray
# import raydp
import os


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        backend = context.params.get("backend", None)

        if backend:
            os.environ["RECOMMENDER_GEOMETRIC_BACKEND"] = backend
        else:
            raise ValueError(" Set backend:[spark, ray, local] in cong/base/parameters.yaml")

        if backend == "spark":
            # # Load the spark configuration in spark.yaml using the config loader
            # parameters = context.config_loader.get("parameters*", "parameters/**")["spark"]
            spark_conf = SparkConf()
            spark_conf.set("spark.driver.maxResultSize", "4g")
            spark_conf.set("spark.driver.memory", "8g")
            # spark_conf.set('spark.executor.memory', '8g')
            spark_conf.set("spark.scheduler.mode", "FAIR")
            # spark_conf.set('spark.cores.max', '4')
            spark_conf.set("petastorm.spark.converter.parentCacheDirUrl", "file://data")

            # Initialise the spark session
            spark_session_conf = (
                SparkSession.builder.appName(context._package_name)
                .enableHiveSupport()
                .config(conf=spark_conf)
            )
            _spark_session = spark_session_conf.getOrCreate()
            _spark_session.sparkContext.setLogLevel("OFF")
        elif backend == "ray":
            ray.init()
            spark = raydp.init_spark(
                app_name="example", num_executors=3, executor_cores=2, executor_memory="2GB"
            )
        elif backend == "local":
            pass
