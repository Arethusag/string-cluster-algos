

def get_spark_session():

    import findspark
    findspark.init()
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("deduplication") \
        .master("local[*]") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
        .getOrCreate()
    

    spark.sparkContext.setCheckpointDir("/tmp/")

    return spark