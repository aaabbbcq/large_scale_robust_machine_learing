package org.harvey.ml.util

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import breeze.linalg.{DenseMatrix => BDM}

object Utils {
  def loadFiles(spark: SparkSession, path: String, testPath: String = null, testSize: Double = 0.2, numPartitions: Int = 4): (DataFrame, DataFrame) = {
    import org.apache.spark.ml.feature.VectorAssembler
    val dataSetRaw = spark.read
      .format("csv")
      .option("sep", ",")
      .option("header", "false")
      .option("inferSchema", "true")
      .option("numPartitions", numPartitions)
      .load(path)
      .repartition(numPartitions)
      .toDF()
    if (testPath != null) {
      val testDataSetRaw = spark.read
        .format("csv")
        .option("sep", ",")
        .option("header", "false")
        .option("inferSchema", "true")
        .load(testPath)
        .repartition(numPartitions)
        .toDF()
      (new VectorAssembler()
        .setInputCols(dataSetRaw.columns.slice(0, dataSetRaw.columns.length - 1))
        .setOutputCol("features")
        .transform(dataSetRaw)
        .select(col(dataSetRaw.columns(dataSetRaw.columns.length - 1)).cast(DoubleType), col("features"))
        .toDF("label", "features"), new VectorAssembler()
        .setInputCols(testDataSetRaw.columns.slice(0, testDataSetRaw.columns.length - 1))
        .setOutputCol("features")
        .transform(testDataSetRaw)
        .select(col(testDataSetRaw.columns(testDataSetRaw.columns.length - 1)).cast(DoubleType), col("features"))
        .toDF("label", "features"))
    }
    else
      new VectorAssembler()
        .setInputCols(dataSetRaw.columns.slice(0, dataSetRaw.columns.length - 1))
        .setOutputCol("features")
        .transform(dataSetRaw)
        .select(col(dataSetRaw.columns(dataSetRaw.columns.length - 1)).cast(DoubleType), col("features"))
        .toDF("label", "features")
        .randomSplit(Array(1 - testSize, testSize), seed = 1L)
        .map(_.toDF("label", "features")) match {
        case Array(trainDataset, testDataset) =>
          (trainDataset, testDataset)
      }
  }

  def getMatrix(array: Array[Array[Double]]): BDM[Double] = {
    new BDM(array(0).length, array.length, array.flatten).t
  }
}
