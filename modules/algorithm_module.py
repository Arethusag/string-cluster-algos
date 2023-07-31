    
def identify_duplicate_pairs(df,string_list,distance_threshold=0.1):
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import Tokenizer, SQLTransformer, RegexTokenizer, NGram, HashingTF
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml import Pipeline

    from pyspark.ml.feature import MinHashLSH

    from graphframes import GraphFrame


    #p
    transform0  =    SQLTransformer(statement=f"""SELECT *,LOWER(REGEXP_REPLACE(CONCAT({','.join(string_list)}), 
            '[\\s\\W]', '')) AS record_strings FROM __THIS__ """)
    token0      =    Tokenizer(inputCol="record_strings", outputCol="token" )
    transform1  =    SQLTransformer(statement="SELECT *, concat_ws(' ', token) concat FROM __THIS__")
    token1      =    RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=1 )
    ngram       =    NGram(n=2, inputCol="char", outputCol="ngram")
    hash      =    HashingTF(inputCol="ngram", outputCol="vector")

    # feat        =    VectorAssembler(inputCols=["vector"], outputCol="features")
    # kmeans      =    KMeans(k = 2, seed = 1, predictionCol="kmeans")

    stages = [transform0,token0,transform1,token1,ngram,hash]
    #pre-processing
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    model_df = model.transform(df)

    
    #knn then jaccard distance
    lsh_model = MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3).fit(model_df)
    similarity_df = lsh_model.approxSimilarityJoin(model_df, model_df, distance_threshold, distCol="text_distance").filter("datasetA.client_id != datasetB.client_id")
    return similarity_df

def identify_duplicate_clusters(similarity_df):
    from pyspark.sql.window import Window
    from pyspark.sql.functions import col, concat_ws, split, array_sort, min

    #graphx
    edges = (similarity_df.selectExpr("datasetA.client_id as src","datasetB.client_id as dst")
            .withColumn('set_col', concat_ws(',', col('src'), col('dst')))
            .withColumn('sorted_set', array_sort(split(col('set_col'), ',')))
            .dropDuplicates(['sorted_set']).select(col("src"), col("dst")))

    vertices = (similarity_df.selectExpr("datasetA.client_id as id").union(similarity_df.selectExpr("datasetB.client_id as id"))).distinct()


    #connections graph
    graph_frame = GraphFrame(vertices, edges)

    #slow
    components_df = graph_frame.connectedComponents().withColumn("min_id", min(col("id")).over(Window.partitionBy("component")))
    return components_df