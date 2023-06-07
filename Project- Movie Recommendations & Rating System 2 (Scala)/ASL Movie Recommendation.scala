import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions.{col, _}

//---------------------------------------------------------------------------------
// Context subject is needed for creating and managing the SparkSession
object Context {

  def getCtx() : SparkSession = {
    SparkSession.builder.
      master("local[*]")
      .appName("spark session example")
      .getOrCreate()

  }
}

//---------------------------------------------------------------------------------
// Create an Function to pass the information in for the training later on
object MagicML {

  //Variables for the movies and dataframe
  var  userId: Int = 0
  var  movies: DataFrame = null
  var  ratings: DataFrame = null
  var  personalNormalizedRatings: DataFrame = null


  //Define the method train() for MatrixFactorizationModel
  def train(): MatrixFactorizationModel ={

    val sc = Context.getCtx()
    import sc.implicits._

    //Splitting training data 90% for training & 10% for testing
    val set = ratings.randomSplit(Array(0.9, 0.1), seed = 12345)
    val training = personalNormalizedRatings.union(set(0)).cache()     //.union combines the DataFrame and the training set into a single datafrme
                                                                       //.cache the resulting is cached in memory
    val test = set(1).cache()
    println(s"Training: ${training.count()}, test: ${test.count()}")

    //Parameters to put into the training
    val trainRDD = training.as[Rating].rdd   //Converting training dataframe into the Rrd format of type 'Rating'
    val rank = 10
    val numIterations = 100

    //Training the recommendation model using ALS from the mllib
    ALS.train(trainRDD, rank, numIterations, 0.01)

  }


  //Predict a user recommendations
  //Method to predict ratings for a given user using a pre-trained recommendation model (model) and the user-movie combinations (usersProducts) extracted from the movies DataFram
  def predict(userId: Int, model : MatrixFactorizationModel): Dataset[Rating] = {
    val sc = Context.getCtx()   //Context.getCtx() is a method call to the getCtx() method defined in the Context object. This method is responsible for creating and returning an instance of SparkSession.
    import sc.implicits._       // importing the implicit conversions provided by the implicits object of the sc variable (which is an instance of SparkSession) into the current scope.

    val usersProducts = movies.select(lit(userId), col("movieId")).map{
      row => (row.getInt(0), row.getInt(1))          //Selects the userId and movieId columns from the movies DataFrame. Then, it applies a mapping operation using the map function to transform each row into a tuple of (Int, Int), representing the userId and movieId values.
    }

    model.predict(usersProducts.toJavaRDD).toDS()    // Invokes the predict method on the model object, passing in the usersProducts data as a JavaRDD. This predicts the ratings for the user-movie combinations in usersProducts. The result is then converted to a Dataset[Rating] using the toDS() method.

  }
}


//---------------------------------------------------------------------------------
//Create the stand-alone application ApplicationRecommendation to run the app
object ApplicationRecommendation extends App {

  //Sets the log level of the logger named "akka","org" to OFF. This disables logging for the "akka","org" category
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val sc = Context.getCtx()
  import sc.implicits._

  //Load the ratings file
  var ratings  = sc.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("src/main/scala/ratings.csv")
    .drop(col("timestamp"))


  // Load the movies file
  var movies  = sc.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("src/main/resources/movies.csv")
    .withColumn("genres", split(col("genres"), "\\|"))


  //Try to show a movie that is avaliable
  //movies.show(5)


  //Create a dataframe containing a sequence of tuples. Each tuple represents a movie title and its corresponding rating.
  // The structure of each tuple is (String, Double),
  val personalRatings = Seq(
    ("Toy Story (1995)", 5.0),
    ("Saving Private Ryan (1998)", 1.0),
    ("Sixth Sense, The (1999)", 1.0),
    ("Ace Ventura: When Nature Calls (1995)", 1.0),
    ("Aladdin (1992)", 5.0),
    ("Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)", 1.0),
    ("Jumanji (1995)", 4.0),
    ("Mortal Kombat (1995)", 1.0),
    ("Inferno (1980)", 1.0),
    ("Pocahontas (1995)", 5.0),
    ("Wrong Trousers, The (1993)", 1.0),
    ("Balto (1995)", 5.0),
    ("Godfather, The (1972)", 1.0),
    ("Silence of the Lambs, The (1991)", 2.0),
    ("Indiana Jones and the Last Crusade (1989)", 1.0),
    ("Heat (1995)", 1.0),
    ("Fugitive, The (1993)", 1.0),
    ("Man of the Year (1995)", 5.0)
  ).toDF("title", "rating")




  // Convert ratings above into model-friendly form
  // User ID will have special value of "0"
  var userId = 0

  var normlizedPersonalRatings = personalRatings.
    join(movies, "title").
    select(lit(userId).as("user"), col("movieId").as("product") , col("rating"))

  // normlizedPersonalRatings.show(5)


  MagicML.userId = userId
  MagicML.movies = movies
  MagicML.ratings = ratings
  MagicML.personalNormalizedRatings = normlizedPersonalRatings

  val model = MagicML.train()

  val usersProducts = movies.select(lit(userId), col("movieId")).map{
    row => (row.getInt(0), row.getInt(1))
  }

  model.predict(usersProducts.rdd).toDS()

  val result = MagicML.predict(userId, model)
  val df = result.filter(r => r.user == userId)

  val recommendationList = df.toDF().sort(col("rating").desc).join(movies, movies("movieId") === df("product"), "inner")
  recommendationList.select("movieId", "title", "genres").show()

}