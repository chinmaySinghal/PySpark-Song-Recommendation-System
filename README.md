# PySpark-Song-Recommendation-System

## Overview
Building recommendation systems (RS) using implicit feedback is a challenging task. Most of the literature on recommendation systems is mainly focused on modelling explicit feedback because of its convenient interpretation and the availability of well-defined evaluation metrics, such as RMSE, to gauge test-set performance. Nevertheless, certain ways are indeed available to extract confidence from implicit data and here we present and implement some of these methods to develop a RS using collaborative filtering for recommending songs to users based on the number of times (counts) a user listened to a particular song. so here I build and evaluate such a recommender system using PySpark and Hadoop.

The distribution of songs and their respective counts in the training dataset is as follows - 
![Counts ](Counts.png)

## The data set

On Dumbo's HDFS, you will find the following files in `hdfs:/user/bm106/pub/project`:

  - `cf_train.parquet`
  - `cf_validation.parquet`
  - `cf_test.parquet`
  
  - `metadata.parquet`
  - `features.parquet`
  - `tags.parquet`
  - `lyrics.parquet`
  
  
The first three files contain training, validation, and testing data for the collaborative filter.  Specifically, each file contains a table of triples `(user_id, count, track_id)` which measure implicit feedback derived from listening behavior.  The first file `cf_train` contains full histories for approximately 1M users, and partial histories for 110,000 users, located at the end of the table.

`cf_validation` contains the remainder of histories for 10K users, and should be used as validation data to tune your model.

`cf_test` contains the remaining history for 100K users, which should be used for your final evaluation.

The four additional files consist of supplementary data for each track (item) in the dataset.  You are not required to use any of these, but they may be helpful when implementing extensions to the baseline model.

## Basic recommender system [80% of grade]

Your recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.  This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors,
  - the *regularization* parameter, and
  - *alpha*, the scaling parameter for handling implicit feedback (count) data.

The choice of evaluation criteria for hyper-parameter tuning is entirely up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.

Once your model is trained, evaluate it on the test set using the ranking metrics provided by spark.  Evaluations should be based on predictions of the top 500 items for each user.


## Extensions [20% of grade]

For full credit, implement an extension on top of the baseline collaborative filter model.  (Again, if you're working in a group of 3 students, you must implement two extensions for full credit here.)

The choice of extension is up to you, but here are some ideas:

  - *Alternative model formualtions*: the `AlternatingLeastSquares` model in Spark implements a particular form of implicit-feedback modeling, but you could change its behavior by modifying the count data.  Conduct a thorough evaluation of different modification strategies (e.g., log compression, or dropping low count values) and their impact on overall accuracy.
  - *Fast search*: use a spatial data structure (e.g., LSH or partition trees) to implement accelerated search at query time.  For this, it is best to use an existing library such as `annoy` or `nmslib`, and you will need to export the model parameters from Spark to work in your chosen environment.  For full credit, you should provide a thorough evaluation of the efficiency gains provided by your spatial data structure over a brute-force search method.
  - *Cold-start*: using the supplementary data, build a model that can map observable feature data to the learned latent factor representation for items.  To evaluate its accuracy, simulate a cold-start scenario by holding out a subset of items during training (of the recommender model), and compare its performance to a full collaborative filter model.
  - *Error analysis*: after training the model, analyze the errors that it makes.  Are certain types of item over- or under-represented?  Make use of the supplementary metadata and tag information to inform your analysis.
  - *Exploration*: use the learned representation to develop a visualization of the items and users, e.g., using T-SNE or UMAP.  The visualization should somehow integrate additional information (features, metadata, or tags) to illustrate how items are distributed in the learned space.

You are welcome to propose your own extension ideas, but they must be submitted in writing and approved by the course staff (Brian, Mayank, or Saumya) by 2019-05-06 at the latest.  If you want to propose an extension, please get in contact as soon as possible so that we have sufficient time to consider and approve the idea.


# What to turn in

In addition to all of your code, produce a final report (not to exceed 4 pages), describing your implementation, evaluation results, and extensions.  Your report should clearly identify the contributions of each member of your group.  If any additional software components were required in your project, your choices should be described and well motivated here.  

Include a PDF copy of your report in the github repository along with your code submission.

Any additional software components should be documented with installation instructions.


# Checklist

It will be helpful to commit your work in progress to the repository.  Toward this end, we recommend the following timeline:

- [ ] 2019/05/08: working baseline implementation 
- [ ] 2019/05/10: select extension(s)
- [ ] 2019/05/18: final project submission (NO EXTENSIONS PAST THIS DATE)
