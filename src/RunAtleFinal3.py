from pyspark.sql import SparkSession

from pyspark.sql.types import (
StructType, StructField, StringType, IntegerType,
DoubleType, LongType, DateType, TimestampType,
)

from pyspark.sql.functions import (col, lit, count, split, size, coalesce, expr, col, lit, sum as spark_sum)

import csv

import time


#List of swearwods, based on the one in the report, smaller however.
swear_words = [
    "arsehole", "ass", "asshole", "bitch", "bastard", "beaner", "bellend",
    "bimbo", "bloody", "bollocks", "boner", "bonk", "boobs", "bugger",
    "bullshit", "butt", "damn", "darkie", "dike", "dildo", "dork", "eff",
    "fanny", "fart", "frig", "fuck", "gash", "gook", "idiot", "jackass",
    "jap", "jerk", "jiss", "jug", "shit", "ktfo", "stfu", "gtfo", "ngaf",
    "dgaf", "ffs", "fml", "omfg", "af", "tf", "wtf", "lmao", "lmfao",
    "rofl", "chink", "coon", "crap", "cum", "cock", "cunt", "dick",
    "faggot", "hoe", "hore", "kike", "nigger", "knob", "lesbo", "minger",
    "moron", "muff", "nonce", "nympho", "pecker", "pedo", "pikey", "pimp",
    "piss", "poofter", "prick", "puke", "pussy", "queef", "shag", "skank",
    "slag", "slut", "shit", "spastic", "retard", "tits", "tosser",
    "tranny", "turd", "twat", "wank", "whore"
]


hdfs = "hdfs://192.168.2.47:9000"


#Function setting up spark session, taking input parameters 
def spark_setup(workers, vCPU_per_worker, RAM):
    print("running setup")
    spark = SparkSession.builder \
    .appName("AtleSolBenmaker") \
    .master("spark://192.168.2.47:7077") \
    .config("spark.executor.instances",workers) \
    .config("spark.executor.cores",vCPU_per_worker) \
    .config("spark.executor.memory", RAM) \
    .config("spark.cores.max", workers * vCPU_per_worker) \
    .config("spark.dynamicAllocation.enabled", "false") \
    .getOrCreate()

    # Defining the explicit schema
    reddit_schema = StructType([
    StructField("author", StringType(), True),
    StructField("body", StringType(), True),
    StructField("normalizedBody", StringType(), True),
    StructField("content", StringType(), True),
    StructField("content_len", LongType(), True),               
    StructField("summary", StringType(), True),
    StructField("summary_len", LongType(), True),
    StructField("id", StringType(), True),
    StructField("subreddit", StringType(), True),
    StructField("subreddit_id", StringType(), True),
    StructField("title", StringType(), True),
    ])

    print("Setup Done")
    
    return spark, reddit_schema



#function to actually how the spark cluster is configured    
def get_cluster_resources(spark):
    #sc = sparkcluster
    sc = spark.sparkContext

    #counting executors, using using a java map. removing the driver since we only care about workers.
    total_executors = sc._jsc.sc().getExecutorMemoryStatus().size() - 1  # remove driver

    #getting cores per executor from config, note that this is not the same thing is the amount of cores actually used.
    cores_per_executor = int(sc.getConf().get("spark.executor.cores", "1"))

    #counting the amount of total cores
    total_cores = total_executors * cores_per_executor

    #assumption of how the core distribution looks, (works in our case cuase we try to split the evenly)
    cores_list = [cores_per_executor] * total_executors

    return total_executors, total_cores, cores_list



#the benchmaker function does a full calculation with the different shit.
def the_benchmaker(spec_list, warmup):
    #running setup
    workers = spec_list[0] 
    vCPU_per_worker = spec_list[1] 
    RAM = spec_list[2]

    #calling the setup function
    spark, reddit_schema = spark_setup(workers, vCPU_per_worker, RAM)

    #reading the input file. not that the only "content and subreddit is read", this is the data we care about
    df = spark.read \
    .format("json") \
    .schema(reddit_schema) \
    .load(f"{hdfs}/dataset/corpus-webis-tldr-17.json") \
    .select("subreddit", "content" ) \
    .repartition(workers * vCPU_per_worker * 4) 
    #repartition, so we get similar split, depending on workers and active CPU:s

    #initial data cleaning and prepping
    df = (
    df
    .withColumn("content", coalesce(col("content"), lit(""))) #if content is null replace with empty string
    .withColumn("words", split(expr("regexp_replace(lower(content), '[^a-zA-Z ]', '')"), r"\s+")) #makes everything lowercase, removes nonletters (we care about words) and split into a list of words.
    .withColumn("words", expr("filter(words, x -> x != '')"))  #remove empty elements from the string
    .withColumn("word_count", size(col("words")))              #count number of words
    )
    
    #adds single counts around the words in the swearword list since it is spark requirement, and then join into a string.
    swear_list = ",".join([f"'{w}'" for w in swear_words])

    #takes the list of words, removes everything that is not in the swearlist, and the count what remains
    df = df.withColumn(
    "swear_count",
    expr(f"size(filter(words, x -> x IN ({swear_list})))")
    )

    #starting to build the result groupby is a heavy function since it involves shuffeling data between workers. We also count the total amout of swearwords and wordcount in the whole dataset
    result = (
        df
        .groupBy("subreddit")
        .agg(
        spark_sum("word_count").alias("total_words"),
        spark_sum("swear_count").alias("total_swear_words")
        )
    )
    
    #aggregates the total words and swearwords into a global one
    totals = result.agg(
        spark_sum("total_words").alias("all_words"),
        spark_sum("total_swear_words").alias("all_swear_words")
    )

    #calculates the percentage of swearwords.
    result = result.withColumn(
        "swear_percentage",
        col("total_swear_words") / col("total_words") * 100
    )

    #calculates the global percentage of swearwords over all the subreddits.
    totals = totals.withColumn(
        "total_swear_percentage",
        col("all_swear_words") / col("all_words") * 100
    )

    #adds new column subreddit, with value all to lable the totals row.
    totals = totals.withColumn("subreddit", lit("ALL"))

    #restructures so it looks the same
    final_df = result.select(
        "subreddit", "total_words", "total_swear_words", "swear_percentage"
    ).unionByName(
        totals.selectExpr(
            "subreddit",
            "all_words as total_words",
            "all_swear_words as total_swear_words",
            "total_swear_percentage as swear_percentage"
        )
    )
    
    #starts timer
    start = time.time()
    
    #the action!
    rows = final_df.collect()

    #writes data to a csv file that can be downlaoded and opened with ex excel, used python, was buggy when i tried spark write (?) don't know why
    with open("/home/ubuntu/subreddit_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(final_df.columns)   #writes column names
        writer.writerows(rows)              #write the rows

    #stop the count!
    runtime = time.time() - start
    
    #get the data from the cluster
    actual_workers, total_cores, cores_per_executor  = get_cluster_resources(spark)

    #print if (used for checking data)
    print(f"Actual executors: {actual_workers}")
    print(f"Total cores: {total_cores}")
    print(f"Cores per executor: {cores_per_executor }")

    # stops sparks instance (new one will be lanhced)
    spark.stop()
    
    #if it is a warmup round, don,t log the data in benchmark result file the file
    if warmup == False:
        with open("benchmark_results_testing_final3.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([actual_workers, total_cores, cores_per_executor , RAM, runtime])
    
    #sleep, to make sure stuff "cools down" unsure if actually needed but felt safe
    time.sleep(10)
    
    
    return [workers, vCPU_per_worker, RAM, runtime]

result_list = []
print("testrun")

#ram is the same for every run
RAM = "1g"

#changing the amount of workers and of cpu per worker to test horizontal and vertical scaling. Run three times per config, to get better data.

worker= 3
vCPU_per_worker = 2
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)

worker= 2
vCPU_per_worker = 2

the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)

worker= 1
vCPU_per_worker = 2

the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)

worker= 3
vCPU_per_worker = 1
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)

worker= 2
vCPU_per_worker = 1
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)

worker= 1
vCPU_per_worker = 1
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)
the_benchmaker([worker,vCPU_per_worker,RAM], False)
