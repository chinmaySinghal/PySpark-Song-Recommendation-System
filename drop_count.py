import sys
from pyspark.sql import SparkSession
import pyspark
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.functions import *

def compute_MAP(model,users,df):
        predictions = model.recommendForUserSubset(users,500)
        print("Generated predictions")
        userRec = (predictions.select("userIndex",F.explode("recommendations").alias("recommendation")).select("userIndex","recommendation.*"))
        rankings = userRec.groupby('userIndex').agg(F.collect_list('trackIndex').alias('ranked_tracks'))
        print("Generated rankings")
        truth = df.groupby('userIndex').agg(F.collect_list('trackIndex').alias('ground_truth'))
        print("Generated ground truth")
        final = rankings.join(truth, rankings.userIndex == truth.userIndex).select('ranked_tracks','ground_truth')
        metrics = RankingMetrics(final.rdd)
        map_val = metrics.meanAveragePrecision
        return map_val


def fit_model(df):
        bestParams = {"regParam":10,"rank":100,"alpha":40.0}
        als = ALS(maxIter=10, regParam=bestParams["regParam"], rank = bestParams["rank"], alpha = bestParams["alpha"], implicitPrefs=True, userCol="userIndex", itemCol="trackIndex", ratingCol="count")
        model = als.fit(df)
        return model


def main(spark, train_file, val_file, test_file, ext_type):
        train_df = spark.read.parquet(train_file)
        train_df.createOrReplaceTempView("train_df")
        print("Loaded train file")        
        val_df = spark.read.parquet(val_file)
        val_df.createOrReplaceTempView("val_df")
        print("Loaded val file")
        test_df= spark.read.parquet(test_file)
        test_df.createOrReplaceTempView("test_df")
        print("Loaded test file")

        users_val = spark.sql('select distinct userIndex from val_df')
        users_test = spark.sql('select distinct userIndex from test_df')
        print("Created users list")

        if ext_type == "drop":
            counts_to_drop = [1,2,3,4,5]
            for cnt in counts_to_drop:
                count_df = spark.sql("select * from train_df where count > {}".format(cnt))
                print('Generated a dataset with dropped counts upto:{}\n'.format(cnt))
                model = fit_model(count_df)
                print("Fitted ALS model")
                map_val = compute_MAP(model,users_val,val_df)
                print('Count Dropped:{} | MAP:{}'.format(cnt, map_val))
                map_test = compute_MAP(model,users_test,test_df)
                print('Count Dropped:{} | MAP:{}'.format(cnt, map_test))


        elif ext_type == "binning":
            count_df = train_df.withColumn('count_New',when(train_df['count'] > 5,train_df['count']).otherwise(1)).drop(train_df['count']).select(col('count_New').alias('count'),col('userIndex'), col('trackIndex'))
            model = fit_model(count_df)
            print("Fitted ALS model")
            map_val = compute_MAP(model,users_val,val_df)
            print('MAP:{}'.format(map_val))      
            map_test = compute_MAP(model,users_test,test_df)
            print('MAP:{}'.format(map_test))
        

if __name__ == "__main__":

    spark = SparkSession.builder.appName('recommendation_system').getOrCreate()

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]
    ext_type = sys.argv[4]

    main(spark, train_file, val_file, test_file, ext_type)
