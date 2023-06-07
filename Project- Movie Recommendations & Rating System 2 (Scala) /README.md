# Movie recommendation Engine using Spark MLlib & Scala
Use MovieLens dataset to build a movie recommender engine using collaborative filtering with Spark's Alternating Least Saqures implementation.


## Prerequisites
Download MovieLens DataSet
Move ratings.csv and movies.csv to src/main/resources/
scalaVersion := "2.13.11"
build.sbt
name := "RecommendationSystem"

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
