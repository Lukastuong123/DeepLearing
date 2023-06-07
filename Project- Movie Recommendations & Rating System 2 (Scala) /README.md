# Movie recommendation Engine using Spark MLlib & Scala
Use MovieLens dataset to build a movie recommender engine using collaborative filtering with Spark's Alternating Least Saqures implementation.

ALS (Alternating Least Squares) is a collaborative filtering algorithm used for recommendation systems. It is widely used to make personalized recommendations based on user-item interactions, such as user ratings or purchase history.

The ALS algorithm is based on matrix factorization, where it decomposes the user-item interaction matrix into two lower-rank matrices: one representing users' preferences and the other representing item attributes. The idea is that by reducing the dimensionality of the original matrix, we can capture the underlying patterns and relationships between users and items.
Here's an overview of how ALS works:

1. Input: The input to ALS is a user-item interaction matrix, where each entry represents the level of interaction (e.g., rating) between a user and an item. This matrix is often sparse since not all users have interacted with all items.
2. Initialization: ALS starts by randomly initializing the user and item matrices with small values. These matrices are iteratively updated to improve the accuracy of the recommendations.
3. 	Alternating Optimization: ALS employs an alternating optimization approach. It alternates between fixing one set of matrices (either user or item) and optimizing the other set. This process continues until convergence.
4.	Optimization Step: Given the fixed set of matrices, ALS uses least squares optimization to update the other set of matrices. It solves a least squares problem to minimize the difference between the observed ratings and the predicted ratings based on the current matrices.
5.	Convergence: ALS repeats the alternating optimization step until the user and item matrices converge to a stable solution. Convergence is achieved when the difference between predicted and observed ratings reaches a satisfactory level.
6.	Prediction: Once the user and item matrices have converged, ALS can make recommendations by estimating the missing entries in the user-item interaction matrix. It predicts the ratings for items that users have not yet interacted with and recommends items with the highest predicted ratings.

ALS has proven to be effective in generating high-quality recommendations and is widely used in various recommendation systems, including movie recommendations, music recommendations, and personalized product recommendations in e-commerce. It is implemented in Spark's MLlib library and provides a scalable and distributed solution for large-scale recommendation tasks.



## Prerequisites
1. Download MovieLens DataSet
2. Move ratings.csv and movies.csv to src/main/resources/
```
scalaVersion := "2.13.11"
```

## build.sbt
name := "MovieRecommendation"

```
version := "0.1.0-SNAPSHOT"

scalaVersion := "2.13.11"

libraryDependencies ++= {
  val sparkVer = "2.1.0"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer % "provided" withSources(),
    "org.apache.spark" %% "spark-sql" % sparkVer,
    "org.apache.spark" %% "spark-mllib" % "2.1.0"
  )
}
```
