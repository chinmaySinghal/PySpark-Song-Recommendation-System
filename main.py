import sys
from pyspark.sql import SparkSession
import pyspark
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def main(spark):
	train_df = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
	train_df.createOrReplaceTempView("train")

	val_df = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_validation.parquet')
	val_df.createOrReplaceTempView("val")    
	
	test_df = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_test.parquet')
	test_df.createOrReplaceTempView("test")
	
	meta_df = spark.read.parquet('hdfs:/user/bm106/pub/project/metadata.parquet')

	sampled_train = spark.sql('select t.* from train t inner join val v on t.user_id = v.user_id union select t.* from train t inner join test tst on t.user_id = tst.user_id')
	indexer_user = StringIndexer(inputCol="user_id", outputCol="userIndex").fit(sampled_train)
	print('User Fitted')
	indexer_track = StringIndexer(inputCol="track_id", outputCol="trackIndex").fit(meta_df)
	print('Track Fitted')

	out_train_df_temp = indexer_user.transform(sampled_train)
	out_train_df = indexer_track.transform(out_train_df_temp)
	print('Indexed Train file')

	#out_train_df = out_train_df.repartition(2000, 'userIndex')
	out_train_df.select('userIndex', 'count', 'trackIndex').write.parquet('hdfs:/user/sa5154/pub/train_full.parquet')
	print("Stored training file")
	del train_df
	del out_train_df
	del out_train_df_temp

	out_val_df_temp = indexer_user.transform(val_df)
	out_val_df = indexer_track.transform(out_val_df_temp)
	print('Indexed Val file')
	#out_val_df = out_val_df.repartition(2000, 'userIndex')
	out_val_df.select('userIndex', 'count', 'trackIndex').write.parquet('hdfs:/user/sa5154/pub/val_full.parquet')
	print("Stored validation file")
	del val_df
	del out_val_df
	del out_val_df_temp

	out_test_df_temp = indexer_user.transform(test_df)
	out_test_df = indexer_track.transform(out_test_df_temp)
	print('Indexed Test file')
	#out_test_df = out_test_df.repartition(2000, 'userIndex')
	out_test_df.select('userIndex', 'count', 'trackIndex').write.parquet('hdfs:/user/sa5154/pub/test_full.parquet')
	print("Stored test file")
	del test_df
	del out_test_df
	del out_test_df_temp


if __name__ == "__main__":
	spark = SparkSession.builder.appName('supervised_train').getOrCreate()
	main(spark)
