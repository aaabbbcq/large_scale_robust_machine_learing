package org.harvey.ml.classification

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.linalg.{Vector, Vectors}
import breeze.linalg.{*, Axis, DenseMatrix => BDM, DenseVector => BDV}
import org.harvey.ml.util.Utils.{getMatrix, loadFiles}

import scala.collection.mutable.ArrayBuffer

object MLR_Consensus {
  org.apache.log4j.Logger.getLogger("org.apache.spark").setLevel(org.apache.log4j.Level.WARN)

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().appName("LR").master("local[2]").getOrCreate()
    spark.sparkContext.setCheckpointDir("D:/Temp/CheckPoint")
    val (training: DataFrame, test: DataFrame) = loadFiles(spark, "D:\\Documents\\Research\\DataSets\\iris.csv")
    val lr = new MLR_Consensus(rho = 1*1e-2, lambda = 1*1e-5, eta = 0*1e-5, epochs = 400)
    lr.fit(training)
    lr.transform(test)
  }
}

class MLR_Consensus(epsilon: Double = 1e-2, rho: Double = 1e-2, lambda: Double = 1e-5,
                eta: Double = 1e-5, epochs: Int = 100, stop: Double = 1e-4) extends Serializable {
  var W: BDM[Double] = BDM.zeros[Double](0, 0)
  var features: Int = _
  var classes: Int = _
  var samples: Int = _

  def fit(dataset: Dataset[_]): Unit = {
    classes = dataset.select("label").distinct().count().toInt
    samples = dataset.count().toInt
    features = dataset.select("features").first.getAs[Vector](0).size
    W = BDM.zeros[Double](features, classes)
    val partitions = dataset.rdd.getNumPartitions

    var admmStruct = dataset.rdd.mapPartitionsWithIndex{ case (index: Int, row: Iterator[Row]) =>
      val featuresArray = ArrayBuffer[Double]()
      val label = ArrayBuffer[Double]()
      while (row.hasNext) {
        val data = row.next()
        label += data.asInstanceOf[Row].getAs[Double](0)
        featuresArray ++= data.asInstanceOf[Row].getAs[Vector](1).toArray
      }
      val labelVector = BDV[Double](label.toArray)
      val featuresMatrix = new BDM(rows = features, cols = label.length, featuresArray.toArray).t
      Iterator(ADMMStruct(labelVector, featuresMatrix, BDM.ones[Double](features, classes), BDM.ones[Double](features, classes)))
    }.cache()

    var z = BDM.ones[Double](features, classes)
    var xBar = BDM.ones[Double](features, classes)
    var uBar = BDM.ones[Double](features, classes)

    var prevCoefficient = BDM.zeros[Double](features, classes)

    for(epoch <- 0 until epochs) {
      admmStruct = admmStruct.map { admm =>
        var margin = admm.X * admm.x
        margin = margin(::, *) - breeze.linalg.max(margin, Axis._1)
        var softmax = breeze.numerics.exp(margin)
        softmax = softmax(::, *) / breeze.linalg.sum(softmax, Axis._1)
        val diff_matrix = getMatrix(Range(0, admm.X.rows).toArray.map(row => new Array[Double](admm.y(row).toInt) ++:
          Array(1.0) ++: new Array[Double](classes - admm.y(row).toInt - 1)))
        softmax -= diff_matrix
        val dx = partitions / samples.toDouble * admm.X.t * softmax + rho * (admm.x - z + admm.u)
        admm.u += admm.x - z
        admm.x -= epsilon * dx
        admm
      }.cache()

      if(epoch % 50 == 0) admmStruct.checkpoint()

      uBar = admmStruct.map(_.u).reduce(_ + _) / partitions.toDouble
      xBar = admmStruct.map(_.x).reduce(_ + _) / partitions.toDouble

      val normed = breeze.numerics.pow(breeze.linalg.norm(z.t(*, ::)), 2)
      val uncorrelated = BDV.ones[Double](classes) * breeze.linalg.sum(normed) - normed

      val matrix = breeze.linalg.pinv(eta * breeze.linalg.diag(uncorrelated) +
        partitions * rho * breeze.linalg.diag(BDV.ones[Double](z.cols)))

      z = rho * partitions * (xBar + uBar) * matrix

      val diff = breeze.linalg.sum(breeze.numerics.abs(prevCoefficient - z)) / z.size
      prevCoefficient = z.copy
      println("Epoch " + epoch + ": " + diff)
    }
    W = z
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
      .agg(("error", "avg"))
      .withColumnRenamed("avg(error)", "ACC")
    accuracy.show()
    labels
  }

  case class ADMMStruct(y: BDV[Double], X: BDM[Double], var x: BDM[Double], var u: BDM[Double])
}
