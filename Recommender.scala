
import org.apache.spark.ml.recommendation.{ALS,ALSModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder,CrossValidator,CrossValidatorModel}
import org.apache.spark.ml.feature.{StringIndexer,SQLTransformer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.evaluation.RegressionEvaluator

def getSQLTransformer(): SQLTransformer = {
  val sql = new SQLTransformer()
  sql.setStatement("select cast(userId as int) as `userId`, cast(itemId as int) as `itemId`, rating from __THIS__")
  sql
}

val sqlC = new SQLContext(sc)
import sqlC.implicits._

//val hc = new HiveContext(sc)
//import hc.implicits._

val data = sc.parallelize(Seq(
("x1","a",1.toFloat),("x1","b",1.toFloat),("x1","c",1.toFloat),("x1","d",1.toFloat),
("x2","e",1.toFloat),("x2","v",1.toFloat),("x2","w",1.toFloat),("x2","h",1.toFloat),
("x3","f",1.toFloat),("x3","b",1.toFloat),("x3","c",1.toFloat),("x3","i",1.toFloat),
("x4","g",1.toFloat),("x4","u",1.toFloat),("x4","x",1.toFloat),("x4","j",1.toFloat),
("x5","h",1.toFloat),("x5","b",1.toFloat),("x5","c",1.toFloat),("x5","k",1.toFloat),
("x6","i",1.toFloat),("x6","t",1.toFloat),("x6","y",1.toFloat),("x6","l",1.toFloat),
("x7","j",1.toFloat),("x7","b",1.toFloat),("x7","c",1.toFloat),("x7","m",1.toFloat),
("x8","k",1.toFloat),("x8","s",1.toFloat),("x8","z",1.toFloat),("x8","n",1.toFloat),
("x9","l",1.toFloat),("x9","b",1.toFloat),("x9","c",1.toFloat),("x9","o",1.toFloat),
("x0","m",1.toFloat),("x0","r",1.toFloat),("x0","q",1.toFloat),("x0","p",1.toFloat))).toDF("user_string","item_string","rating").cache()

val user_indexer = new StringIndexer().setInputCol("user_string").setOutputCol("userId").setHandleInvalid("skip") 
val item_indexer = new StringIndexer().setInputCol("item_string").setOutputCol("itemId").setHandleInvalid("skip") 

val sql = getSQLTransformer()
//val sql = new SQLTransformer()
//sql.setStatement("select user_string, item_string, cast(userId as int) as `userId`, cast(itemId as int) as `itemId`, rating from __THIS__")


//val data2 = user_indexer.fit(data).transform(data)
//val data3 = item_indexer.fit(data2).transform(data2)
//sql.transform(data3).show

val als = new ALS().setImplicitPrefs(true).setNonnegative(true).setUserCol("userId").setItemCol("itemId")

val pipeline = new Pipeline()
pipeline.setStages(Array(user_indexer, item_indexer, sql, als))

val evaluator = new RegressionEvaluator()
evaluator.setLabelCol("rating")

val param_grid = new ParamGridBuilder().addGrid(als.alpha, Array(0.01D, 0.1D, 0.5D)).addGrid(als.maxIter, Array(1.toInt, 5.toInt, 10.toInt)).build()

val cv = new CrossValidator()
cv.setEstimator(pipeline)
cv.setEstimatorParamMaps(param_grid)
cv.setEvaluator(evaluator)

val cvModel = cv.fit(data)

val data4 = cvModel.transform(data).cache()
cvModel.avgMetrics
//data4.show(50)


sc.parallelize(param_grid.map(_.toString) zip cvModel.avgMetrics).toDF("params","score")
