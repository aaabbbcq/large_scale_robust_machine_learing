package org.harvey.ml.classification

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import breeze.linalg.{*, Axis, DenseMatrix => BDM, DenseVector => BDV}
import org.harvey.ml.util.KeyMapPartitioner
import org.harvey.ml.util.Utils.{getMatrix, loadFiles}

import scala.collection.mutable.ArrayBuffer

object MLR_Sharing {
  org.apache.log4j.Logger.getLogger("org.apache.spark").setLevel(org.apache.log4j.Level.WARN)

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder.master("local[4]").appName("MLR").getOrCreate
    spark.sparkContext.setCheckpointDir("D:/Temp/CheckPoint")
    val (training: DataFrame, test: DataFrame) = loadFiles(spark, "D:\\Documents\\Research\\DataSets\\segment-challenge-all.csv")
    val lr = new MLR_Sharing(rho = 1e-3, lambda = 1e-5, epochs = 300)
    lr.fit(training)
    lr.transform(test)
  }
}

class MLR_Sharing(partitions: Int = 4, epsilon: Double = 1e-2, rho: Double = 1e-2, lambda: Double = 1e-5,
                  eta: Double = 1e-5, epochs: Int = 100, stop: Double = 1e-4) extends Serializable {
  var features: Int = _
  var samples: Int = _
  var classes: Int = _
  var x: BDM[Double] = _
  var z: BDM[Double] = _
  var u: BDM[Double] = _

  def fit(dataset: Dataset[_]): Unit = {
    implicit val mapEncoder: Encoder[Map[String, Any]] = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]
    classes = dataset.select("label").distinct().count().toInt
    samples = dataset.count().toInt
    features = dataset.select("features").first.getAs[Vector](0).size
    z = BDM.ones[Double](samples, classes)
    u = z.copy
    var A_x_bar = z.copy
    val y = BDV[Double](dataset.select("label").rdd.map(_.getAs[Double](0)).collect())
    var admmUpdater = getTransformed(dataset)

    var prevCoefficient = BDM.zeros[Double](samples, classes)

    for (epoch <- 0.until(epochs)) {
      admmUpdater = admmUpdater.map { admm =>
        val normed = breeze.numerics.pow(breeze.linalg.norm(admm.x.t(*, ::)), 2)
        val uncorrelated = BDV.ones[Double](classes) * breeze.linalg.sum(normed) - normed
        val xArray = ArrayBuffer[Array[Double]]()
        for(i <- 0 until admm.x.cols){
          val xCol = rho * breeze.linalg.pinv(eta * uncorrelated(i) * BDM.eye[Double](admm.X.cols) +
            rho * admm.X.t * admm.X) * (admm.X.t * admm.X * admm.x(::, i) + admm.X.t * z(::, i) -
            admm.X.t * A_x_bar(::, i) - admm.X.t * u(::, i))
          xArray += xCol.toArray
        }
        admm.x = new BDM(rows = admm.x.rows, cols = admm.x.cols, xArray.toArray.flatten)
//        admm.x += breeze.linalg.pinv(admm.X.t * admm.X) * admm.X.t * (z - A_x_bar - u)
        admm
      }.cache

      if (epoch % 10 == 0) admmUpdater.checkpoint

      A_x_bar = admmUpdater.map { admm =>
        admm.X * admm.x
      }.treeAggregate(BDM.zeros[Double](samples, classes))((c: BDM[Double], instance: BDM[Double]) => c + instance,
        (c1: BDM[Double], c2: BDM[Double]) => c1 + c2
      ) / partitions.toDouble

      z = zUpdate(u, z, A_x_bar, y)

      u += A_x_bar - z

      val diff = breeze.linalg.sum(breeze.numerics.abs(prevCoefficient - u)) / u.size
      prevCoefficient = u.copy
      println("Epoch " + epoch + ": " + diff)
    }

    x = BDM.vertcat[Double](admmUpdater.map(_.x).collect(): _*)
  }

  private def zUpdate(u: BDM[Double], z: BDM[Double], A_x_bar: BDM[Double], y: BDV[Double]): BDM[Double] = {
    var margin = partitions.toDouble * z
    margin = margin(::, *) - breeze.linalg.max(margin, Axis._1)
    var softmax = breeze.numerics.exp(margin)
    softmax = softmax(::, *) / breeze.linalg.sum(softmax, Axis._1)
    val diffMatrix = getMatrix(Range(0, samples).toArray.map(row => new Array[Double](y(row).toInt) ++:
      Array(1.0) ++: new Array[Double](classes - y(row).toInt - 1)))
    softmax -= diffMatrix
    val dz = partitions.toDouble * (softmax + rho * (z - A_x_bar - u))
    z -= epsilon * dz
    z
  }

  private def getTransformed(dataset: Dataset[_]): RDD[ADMMStruct] = {
    val transformedDateset = dataset.rdd.zipWithIndex().flatMap { case (Row(label: Double, feature: Vector), rowIndex: Long) =>
      val baseSize = features / partitions
      val mod = features % partitions
      var indexSum = 0
      var interval = Array[(Int, Int)]()
      for (i <- Range(0, partitions - mod).map(_ => baseSize).++(Range(0, mod).map(_ => baseSize + 1))) {
        interval ++= Array((indexSum, indexSum + i))
        indexSum += i
      }
      interval.zipWithIndex.map { case ((low: Int, high: Int), colIndex: Int) =>
        (colIndex, (colIndex, rowIndex, label, new DenseVector(feature.toArray.slice(low, high))))
      }
    }.partitionBy(new KeyMapPartitioner(partitions))
      .map(_._2)
      .mapPartitionsWithIndex { case (index: Int, row: Iterator[(Int, Long, Double, Vector)]) =>
        val features = ArrayBuffer[Double]()
        val label = ArrayBuffer[Double]()
        while (row.hasNext) {
          val data = row.next
          label += data._3
          features ++= data._4.toArray
        }
        val labelVector = BDV[Double](label.toArray)
        val featuresMatrix = new BDM(rows = features.length / label.length, cols = label.length, features.toArray).t
        Iterator(ADMMStruct(index, labelVector, featuresMatrix, BDM.ones[Double](featuresMatrix.cols, classes)))
      }
    transformedDateset
  }

  def transform(dataset: Dataset[_]): DataFrame = {
    val predict = udf((features: Vector) => {
      val y_pred = x.t * BDV[Double](features.toArray)
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

  case class ADMMStruct(index: Int, label: BDV[Double], X: BDM[Double], var x: BDM[Double])
}
