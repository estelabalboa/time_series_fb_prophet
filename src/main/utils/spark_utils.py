from pyspark.sql import SparkSession
import logging

logging.getLogger('py4j').setLevel(logging.ERROR)


def init_spark(app_name: str):
    """
    Initializes or gets a spark session.
    :param app_name: name used to identify the spark session.
    :return: a Spark session
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark
