from multiprocessing import cpu_count
from pyspark.sql import SparkSession

import warnings

# Suppress PySpark warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_spark_session(name: str = "AntiMoneyLaundering", verbose: bool = False):
    num_cores: int = cpu_count()
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(f"{name}") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.enabled", "true") \
        .config("spark.executor.instances", num_cores) \
        .config("spark.executor.cores", 1) \
        .config("spark.sql.shuffle.partitions", num_cores) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    
    if verbose:
        print()
        print(f"Spark version: {spark.version}")
        print(f"Spark master: {spark.sparkContext.master}")
        print(f"Spark app name: {spark.sparkContext.appName}")
        print(f"Spark executor instances: {spark.sparkContext._conf.get('spark.executor.instances')}")
        print(f"Spark executor cores: {spark.sparkContext._conf.get('spark.executor.cores')}")
        print(f"Spark shuffle partitions: {spark.sparkContext._conf.get('spark.sql.shuffle.partitions')}")
        print(f"Spark Arrow enabled: {spark.sparkContext._conf.get('spark.sql.execution.arrow.enabled')}")
        print(f"Spark Arrow pyspark enabled: {spark.sparkContext._conf.get('spark.sql.execution.arrow.pyspark.enabled')}")
        print()

    return spark
