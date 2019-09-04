package regression

import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.rdd.RDD
import admm.model.regression.SharingLasso

/**
  * Created by Evan on 2018/8/6.
  */
object SharingLassoTest {
	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("Lasso").setMaster("local[2]")
		val sc = new SparkContext(conf)
		sc.setCheckpointDir("src/tempCheckPointDir/")
		sc.setLogLevel("ERROR")

		val data: RDD[(Double, Array[Double])] = sc.textFile("src/test/resource/lpsa1.data")
		  .map(line => line.split(","))
		  .map(line => (line(0).toDouble, line(1).split(" ").map(_.toDouble)))

		val parameter: Map[String, String] = Map("maxItertimes" -> "200",
			"conThreshold" -> "0.01",
			"lambda" -> "0.5",
			"rho" -> "0.2",
			"kind" -> "sharing",
			"numPartition" -> "2",
			"minIterForCheck" -> "5",
			"minIterVerConverge" -> "5")
		val lasso = new SharingLasso(parameter)
		val w:BDM[Double] = lasso.fit(data)
		println(w)
	}
}