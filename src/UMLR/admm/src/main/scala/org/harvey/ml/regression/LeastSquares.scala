package org.harvey.ml.regression

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import breeze.linalg.{Axis, DenseMatrix => BDM, DenseVector => BDV}
import org.harvey.ml.util.Utils.loadFiles

object LeastSquares {
  org.apache.log4j.Logger.getLogger("org.apache.spark").setLevel(org.apache.log4j.Level.WARN)

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder.master("local[4]").appName("Test").getOrCreate
    val (training: DataFrame, test: DataFrame) = loadFiles(spark, "D:\\Documents\\Research\\DataSets\\lpsa.csv")
    val ls = new LeastSquares(1e-3, 400)
    ls.fit(training)
    ls.transform(test)
  }
}

class LeastSquares(rho: Double = 1e-3, epochs: Int = 100, stop: Double = 1e-4) extends Serializable {
  var W: BDV[Double] = _
  var features = 0
  def fit(dataset: Dataset[_]): Unit = {
    implicit val mapEncoder: Encoder[Map[String, Any]] = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]
    features = dataset.select("features").first.getAs[Vector](0).size
    W = BDV.zeros[Double](features)
    train(dataset)
  }

  private def train(dataset: Dataset[_]): Unit ={
    val instances = dataset.rdd.map{ case Row(label: Double, features: Vector) =>
      Instance(label, features)
    }
    optimize(instances: RDD[Instance], epochs)
  }

  private def optimize(instances: RDD[Instance], epochs: Int = 100): BDV[Double] ={
    val seqOp = (c: BDV[Double], instance: Instance) => c + gradient(BDV[Double](instance.features.toArray), instance.label)
    val combOp = (c1: BDV[Double], c2: BDV[Double]) => c1 + c2
    var prevCoefficient = BDV.zeros[Double](features)
    for (epoch <- 0 until epochs) {
      val dW = instances.treeAggregate(BDV.zeros[Double](features))(seqOp, combOp)
      W -= rho * dW

      val diff = breeze.linalg.sum(breeze.numerics.abs(prevCoefficient - W)) / W.size
      prevCoefficient = W.copy
      println("Epoch " + epoch + ": " + diff)
    }
    W
  }

  private def gradient(X: BDV[Double], y: Double): BDV[Double] = {
    X * (X dot W - y)
  }

  def transform(dataset: Dataset[_]): DataFrame = {
    val predict = udf((features: Vector) => {
      val y_pred = BDV[Double](features.toArray) dot W
      y_pred
    })
    val labels = dataset.select("label", "features")
      .withColumn("y_pred", predict(col("features")))
      .select("label", "y_pred")
    labels.show()
    val mae = labels.select("label", "y_pred")
      .withColumn("error", abs(col("label")-col("y_pred")))
      .agg(("error","avg"))
      .withColumnRenamed("avg(error)", "MAE")
    mae.show()
    labels
  }
  case class Instance(label: Double, features: Vector)
}
