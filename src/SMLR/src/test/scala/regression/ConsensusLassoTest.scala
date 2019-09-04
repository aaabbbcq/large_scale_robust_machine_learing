package regression

import admm.model.regression.ConsensusLasso
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Evan on 2018/8/6.
  */
object ConsensusLassoTest {
	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("Lasso").setMaster("local")
		val sc = new SparkContext(conf)
		sc.setCheckpointDir("src/tempCheckPointDir/")
		sc.setLogLevel("ERROR")

		val data = sc.textFile("src/test/resource/lpsa1.data")
		  .map(line => line.split(","))
		  .map(line => (line(0).toDouble, line(1).split(" ").map(_.toDouble)))

		val parameter: Map[String, String] = Map("maxItertimes" -> "300",
			"conThreshold" -> "0.01",
			"lambda" -> "0.5",
			"rho" -> "0.2",
			"kind" -> "consensus",
			"numPartition" -> "2",
			"minIterForCheck" -> "5",
			"minIterVerConverge" -> "5")

		val lasso = new ConsensusLasso(parameter)
		val w = lasso.fit(data)
		print("模型参数： ")
		print(w.t(0,::))
	}
}

