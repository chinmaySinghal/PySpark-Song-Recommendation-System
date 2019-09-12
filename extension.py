import sys
from pyspark.sql import SparkSession
import pyspark
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
import math

def compute_MAP(model, users, df):
        predictions = model.recommendForUserSubset(users,500)
        print("Generated predictions")
        userRec = (predictions.select("userIndex",F.explode("recommendations").alias("recommendation")).select("userIndex","recommendation.*"))
        rankings = userRec.groupby('userIndex').agg(F.collect_list('trackIndex').alias('ranked_tracks'))
        print("Generated rankings")
        truth = df.groupby('userIndex').agg(F.collect_set('trackIndex').alias('ground_truth'))
        print("Generated ground truth")
        final = rankings.join(truth, rankings.userIndex == truth.userIndex).select('ranked_tracks','ground_truth')
        metrics = RankingMetrics(final.rdd)
        print("Generated metrics")
        map_val = metrics.meanAveragePrecision
        return map_val


def main(spark, train_file, val_file, test_file, ext_type):
        train_df = spark.read.parquet(train_file)
        print("Loaded train file")
        val_df = spark.read.parquet(val_file)
        print("Loaded val file")
        test_df = spark.read.parquet(test_file)
        print("Loaded test file")

        val_df.createOrReplaceTempView("val_df")
        users_val = spark.sql("SELECT DISTINCT userIndex FROM val_df")
        print("Created val users list")

        test_df.createOrReplaceTempView("test_df")
        users_test = spark.sql("SELECT DISTINCT userIndex FROM test_df")
        print("Created test users list")

        if ext_type == "log":
                train_df = train_df.withColumn("count",F.log(1+train_df["count"]))
        elif ext_type == "square":
                train_df = train_df.withColumn("count",train_df["count"]*train_df["count"])
        elif ext_type == "cube":
                train_df = train_df.withColumn("count",train_df["count"]*train_df["count"]*train_df["count"])
        elif ext_type == "log2":
                train_df = train_df.withColumn("count",F.log(1+train_df["count"])/math.log(2))
        print("Transformed counts")

        params = {"regParam":10,"rank":100,"alpha":40.0}
        reg = params["regParam"]
        rank = params["rank"]
        alpha = params["alpha"]

        als = ALS(maxIter=10, regParam=reg, rank = rank, alpha = alpha, implicitPrefs=True, userCol="userIndex", itemCol="trackIndex", ratingCol="count")
        model = als.fit(train_df)
        print("Fitted ALS model")

        map_val = compute_MAP(model,users_val,val_df)
        print('Validation: RegParam:{} | Rank:{} | Alpha:{} | MAP:{}'.format(reg, rank, alpha, map_val))
        map_test = compute_MAP(model,users_test,test_df)        
        print('Test: RegParam:{} | Rank:{} | Alpha:{} | MAP:{}'.format(reg, rank, alpha, map_test))


if __name__ == "__main__":
        spark = SparkSession.builder.appName('recommendation_system').getOrCreate()

        train_file = sys.argv[1]
        val_file = sys.argv[2]
        test_file = sys.argv[3]
        ext_type = sys.argv[4]

        main(spark, train_file, val_file, test_file, ext_type)
