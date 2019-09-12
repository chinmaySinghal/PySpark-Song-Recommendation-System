import sys
from pyspark.sql import SparkSession
import pyspark
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F

def compute_MAP(model,users,df):
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


def main(spark, train_file, val_file, test_file):
        train_df = spark.read.parquet(train_file)
        print("Loaded train file")
        val_df = spark.read.parquet(val_file)
        print("Loaded val file")
        val_df.createOrReplaceTempView("val_df")

        test_df = spark.read.parquet(test_file)
        print("Loaded test file")
        test_df.createOrReplaceTempView("test_df")

        users_val = spark.sql('select distinct userIndex from val_df')
        print("Created users val list")
        users_test = spark.sql('select distinct userIndex from test_df')
        print("Created users test list")

        bestParams = {"regParam":0,"rank":0,"alpha":0}
        regParam = [0.01, 0.1,1,10,100]
        ranks = [20,50,100]
        alphas = [0.01,20.0,40.0,100.0]
        bestMAP = 0.0
        for reg in regParam:
                for rank in ranks:
                        for alpha in alphas:
                                als = ALS(maxIter=10, regParam=reg, rank = rank, alpha = alpha, implicitPrefs=True, userCol="userIndex", itemCol="trackIndex", ratingCol="count")
                                model = als.fit(train_df)
                                print("Fitted ALS model")
                               
                                map_val = compute_MAP(model,users_val,val_df)
                                print('RegParam:{} | Rank:{} | Alpha:{} | MAP:{}'.format(reg, rank, alpha, map_val))
                                if map_val>bestMAP:
                                        bestMAP = map_val
                                        bestParams["regParam"] = reg
                                        bestParams["rank"] = rank
                                        bestParams["alpha"] = alpha

        print("Best params are : {}".format(bestParams))
        
        print("Fitting best model for test...")
        als = ALS(maxIter=10, regParam=bestParams["regParam"], rank = bestParams["rank"], alpha = bestParams["alpha"], implicitPrefs=True, userCol="userIndex", itemCol="trackIndex", ratingCol="count")
        model = als.fit(train_df)
        print("Fitted best ALS model")
        map_test = compute_MAP(model,users_test,test_df)
        print('RegParam:{} | Rank:{} | Alpha:{} | MAP:{}'.format(bestParams["regParam"], bestParams["rank"], bestParams["alpha"], map_test))


if __name__ == "__main__":
        spark = SparkSession.builder.appName('recommendation_system').getOrCreate()

        train_file = sys.argv[1]
        val_file = sys.argv[2]
        test_file = sys.argv[3]

        main(spark, train_file, val_file, test_file)
