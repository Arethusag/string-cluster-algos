# %%
import findspark
import time
findspark.init()



start_time = time.time()


from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("deduplication") \
    .master("local[*]") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
    .getOrCreate()

spark.sparkContext.setCheckpointDir("/tmp/")


# %%
def create_synthetic_records(n_records,n_duplicates0 = 0,n_duplicates1 = 0,n_duplicates2 = 0,n_duplicates3 = 0 ):

    from faker import Faker
    import pandas as pd
    import numpy as np
    import random

    # Instantiate Faker
    fake = Faker() 

    # Create a new DataFrame for storing the original records and their duplicates
    df_new = pd.DataFrame({
        'client_id': range(1, n_records+1),
        'client_name': [fake.name() for _ in range(n_records)],
        'address': [fake.address().replace('\n', ', ') for _ in range(n_records)],
        'email': [fake.email() for _ in range(n_records)],
        'phone_number': [fake.phone_number() for _ in range(n_records)],
        'duplicate_id': range(1, n_records+1),  # Add a new column for the duplicate ID
        'num_modifications': [0] * n_records  # Add a new column for the number of modifications
    })

    # Define modification functions
    def modify_name(name):
        mods = ['add_middle_initial'
                ,'add_prefix'
                ,'typo'
                ,'switch_name_order'
                ,'change_case']
        mod = random.choice(mods)
        print(f"Chosen modification: {mod}")
        
        if mod == 'add_middle_initial':
            if len(name.split()) == 2:
                first, last = name.split(' ')
                middle_initial = fake.random_uppercase_letter()
                modified_name = f"{first} {middle_initial}. {last}"
            else:
                modified_name = f"{name} {fake.random_uppercase_letter}."
                
        elif mod == 'add_prefix':
            prefix = fake.prefix()
            modified_name = f"{prefix} {name}"
            
        elif mod == 'typo':
            index = random.randint(0, len(name)-1)
            modified_name = name[:index] + fake.random_lowercase_letter() + name[index+1:]
            
        elif mod == 'switch_name_order':
            name_parts = name.split()
            modified_name = ', '.join(name_parts[::-1])
            
        elif mod == 'change_case':
            modified_name = name.upper() if random.choice([True, False]) else name.lower()
        
        print(f"Modified name: {modified_name}")
        return modified_name


    def modify_email(email):
        user, domain = email.split('@')
        new_domain = fake.free_email_domain()
        return f"{user}@{new_domain}"

    def modify_address(address):
        new_address = fake.street_address()
        city_state_zip = address.split(', ')[1:]
        return f"{new_address}, {', '.join(city_state_zip)}"

    def modify_phone_number(phone_number): 
        digits = [char for char in phone_number if char.isdigit()]
        digit_to_replace = random.choice(digits)
        return phone_number.replace(digit_to_replace, str(fake.random_digit()), 1)

    def modification(field, value):
        if field == 'client_name':
            return modify_name(value)
        elif field == 'address':
            return modify_address(value)
        elif field == 'email':
            return modify_email(value)
        elif field == 'phone_number':
            return modify_phone_number(value)

    # Make slight modifications to create duplicates
    for idx in range(n_duplicates0):

        #print("duplicate #: ", idx)
        # Create the first duplicate
        duplicate = df_new.loc[idx].copy()
        #print(duplicate)
        new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
        duplicate['client_id'] = new_client_id
        duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

        # Randomly select zero or more fields to modify (excluding client_id)
        num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
        duplicate['num_modifications'] = num_modifications
        if num_modifications > 0:
            fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

            for field in fields_to_modify:
                # Apply the modification
                value = duplicate[field]
                duplicate[field] = modification(field,value)

        # Append the duplicate to the DataFrame
        df_new = pd.concat([df_new, duplicate.to_frame().transpose()], ignore_index=True)
        #print(idx,duplicate)

        # Create additional duplicates for a subset of the original records
        if idx < n_duplicates1:
            # Create one more duplicate for the first 50 original records
            duplicate = df_new.loc[idx].copy()
            new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
            duplicate['client_id'] = new_client_id
            duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

            num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
            duplicate['num_modifications'] = num_modifications
            if num_modifications > 0:
                fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

                for field in fields_to_modify:
                    # Apply the modification
                    value = duplicate[field]
                    duplicate[field] = modification(field,value)

            # Append the duplicate to the DataFrame
            df_new = pd.concat([df_new, duplicate.to_frame().transpose()], ignore_index=True)
            #print(idx,duplicate)

        if idx < n_duplicates2:
            # Create more duplicates
            duplicate = df_new.loc[idx].copy()
            new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
            duplicate['client_id'] = new_client_id
            duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

            num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
            duplicate['num_modifications'] = num_modifications
            if num_modifications > 0:
                fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

                for field in fields_to_modify:
                    # Apply the modification
                    value = duplicate[field]
                    duplicate[field] = modification(field,value)

            # Append the duplicate to the DataFrame
            df_new = pd.concat([df_new, duplicate.to_frame().transpose()], ignore_index=True)
            #print(idx,duplicate)

        if idx == 0:
            # Create more duplicates for the first original record
            for _ in range(n_duplicates3):
                duplicate = df_new.loc[idx].copy()
                new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
                duplicate['client_id'] = new_client_id
                duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

                num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
                duplicate['num_modifications'] = num_modifications
                if num_modifications > 0:
                    fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

                    for field in fields_to_modify:
                        # Apply the modification
                        value = duplicate[field]
                        duplicate[field] = modification(field,value)

                # Append the duplicate to the DataFrame
                df_new = pd.concat([df_new, duplicate.to_frame().transpose()], ignore_index=True)
                #print(idx,duplicate)
        
        #print(len(df_new))

    # Shuffle the DataFrame
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    return df_new


num_records = 100000
num_duplicates0 = 5000
num_duplicates1 = 3000
num_duplicates2 = 2000
num_duplicates3 = 1000

df_pandas = create_synthetic_records(num_records, num_duplicates0, num_duplicates1, num_duplicates2, num_duplicates3)


df_pandas = df_pandas.astype({
    'client_id': 'int64',  # or 'int32' if your numbers are not too large
    'client_name': 'string',
    'address': 'string',
    'email': 'string',
    'phone_number': 'string',
    'duplicate_id': 'int64',  # or 'int32' if your numbers are not too large
    'num_modifications': 'int64'  # or 'int32' if your numbers are not too large
})



df = spark.createDataFrame(df_pandas)

# %%
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Tokenizer, SQLTransformer, RegexTokenizer, NGram, HashingTF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

from pyspark.sql.functions import col, concat_ws, split, array_sort, min
from pyspark.sql.window import Window
from pyspark.ml.feature import MinHashLSH

from graphframes import *
#parameters
distance_threshold = 0.1

#p
transform0  =    SQLTransformer(statement="""SELECT *,LOWER(REGEXP_REPLACE(CONCAT(client_name, address, email, phone_number), 
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


end_time = time.time()
runtime = end_time - start_time
print(f"The runtime of the script is {runtime} seconds.")