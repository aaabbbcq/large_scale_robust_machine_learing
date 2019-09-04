package org.harvey.ml.classification

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{Vector, Vectors}
import breeze.linalg.{*, Axis, DenseMatrix => BDM, DenseVector => BDV}
import org.harvey.ml.util.Utils.loadFiles

object MLR_Splitting_Samples {
  org.apache.log4j.Logger.getLogger("org.apache.spark").setLevel(org.apache.log4j.Level.WARN)

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder.master("local[4]").appName("MLR").getOrCreate
    val (training: DataFrame, test: DataFrame) = loadFiles(spark, "D:\\Documents\\Research\\DataSets\\ArtificialDataSet\\Normal_datasets\\S2_0.89.csv")
    val lr = new MLR_Splitting_Samples(rho = 1e-3, lambda = 1e-5, eta = 1*1e-5, epochs = 400)
    lr.fit(training)
    lr.transform(test)
  }
}

class MLR_Splitting_Samples(rho: Double = 1e-2, lambda: Double = 1e-5, eta: Double = 1e-5, epochs: Int = 100,
                            stop: Double = 1e-4) extends Serializable {
  var W: BDM[Double] = _
  var classes = 0
  var samples = 0
  var features = 0

  def fit(dataset: Dataset[_]): Unit = {
    implicit val mapEncoder: Encoder[Map[String, Any]] = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]
    classes = dataset.select("label").distinct().count().toInt
    samples = dataset.count().toInt
    features = dataset.select("features").first.getAs[Vector](0).size
    W = BDM.zeros[Double](features, classes)
    train(dataset)
  }

  private def train(dataset: Dataset[_]): Unit ={
    val instances = dataset.rdd.map{ case Row(label: Double, features: Vector) =>
      Instance(label, features)
    }
    adaptiveMomentEstimation(instances: RDD[Instance], epochs)
  }

  private def adaptiveMomentEstimation(instances: RDD[Instance], epochs: Int = 100,
                       beta_1: Double = 0.9, beta_2: Double = 0.9999, epsilon: Double = 1e-8): BDM[Double] ={
    val seqOp = (c: BDM[Double], instance: Instance) => c + gradient(BDV[Double](instance.features.toArray), instance.label)
    val combOp = (c1: BDM[Double], c2: BDM[Double]) => c1 + c2

    var t = 0
    var m = BDM.zeros[Double](features, classes)
    var v = BDM.zeros[Double](features, classes)

    var prevCoefficient = BDM.zeros[Double](features, classes)

    for (epoch <- 0 until epochs) {
      val dW = instances.treeAggregate(BDM.zeros[Double](features, classes))(seqOp, combOp)

      t += 1
      m = beta_1 * m + (1 - beta_1) * dW
      v = beta_2 * v + (1 - beta_2) * breeze.numerics.pow(dW, 2)
      val m_t = m / (1 - scala.math.pow(beta_1, t))
      val v_t = v / (1 - scala.math.pow(beta_2, t))

      W -= rho * m_t / (breeze.numerics.sqrt(v_t) + epsilon)

      val diff = breeze.linalg.sum(breeze.numerics.abs(prevCoefficient - W)) / W.size
      prevCoefficient = W.copy
      println("Epoch " + epoch + ": " + diff)
    }
    W
  }

  private def gradient(X: BDV[Double], y: Double): BDM[Double] = {
    var predictions = W.t * X
    predictions -= breeze.linalg.max(predictions)
    var softmax = breeze.numerics.exp(predictions)
    softmax /= breeze.linalg.sum(softmax) // + 0.000001
    var loss = -breeze.numerics.log(softmax(y.toInt))
    loss += 0.5 * lambda * breeze.linalg.sum(breeze.numerics.pow(W, 2))
    val diffVector = BDV[Double](Range(0, classes).map(elem => if (elem == y.toInt) 1.0 else 0.0).toArray)
    softmax -= diffVector
    val normed = breeze.numerics.pow(breeze.linalg.norm(W.t(*, ::)), 2)
    val uncorrelated = BDV.ones[Double](classes) * breeze.linalg.sum(normed) - normed
    val dW = X * softmax.t + eta * (W(*, ::) * uncorrelated)
    dW
  }

  def transform(dataset: Dataset[_]): DataFrame = {
    val predict = udf((features: Vector) => {
      val y_pred = W.t * BDV[Double](features.toArray)
      breeze.linalg.argmax(y_pred)
    })
    val labels = dataset.select("label", "features")
      .withColumn("y_pred", predict(col("features")))
      .select(col("label").cast(IntegerType), col("y_pred"))
    labels.show()
    val compare = udf((y_real: Int, y_pred: Int) => {
      if (y_real == y_pred) 1 else 0
    })
    val accuracy = labels.select("label", "y_pred")
      .withColumn("error", compare(col("label"), col("y_pred")))
      .agg(("error","avg"))
      .withColumnRenamed("avg(error)", "ACC")
    accuracy.show()
    labels
  }

  case class Instance(label: Double, features: Vector)
}
