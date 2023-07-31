if __name__ == "__main__":

    from modules.spark_module import get_spark_session
    from modules.data_module import create_synthetic_records
    from modules.algorithm_module import identify_duplicate_pairs, identify_duplicate_clusters

    spark = get_spark_session()

    num_records = 1000 # 100,000 inital records
    num_duplicates0 = 50 # duplicate first 5000 records
    num_duplicates1 = 30 # duplicate first 3000 records
    num_duplicates2 = 20 # duplicate first 2000 records
    num_duplicates3 = 10 # duplicate first record 1000 times


    df = spark.createDataFrame(create_synthetic_records(num_records, num_duplicates0, num_duplicates1, num_duplicates2, num_duplicates3))

    client_strings = ['client_name', 'address', 'email', 'phone_number']


    duplicates_df = identify_duplicate_pairs(df,client_strings,0.1)
    duplicates_df.show()

    clusters_df = identify_duplicate_clusters(duplicates_df)
    clusters_df.show()
