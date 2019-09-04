package classification

import java.io.File

import admm.model.classification.SharingSMLR
import breeze.linalg.csvwrite
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Evan on 2018/8/19.
  */
object SharingSMLRTest {

	def nFoldFit(sc: SparkContext, dataPath: String, parameter: Map[String, String]) = {
		println("k-fold fit...")
		val data = sc.textFile("src/test/resource/COIL20.csv")
		  .map(line => line.split(","))
		  .map(line=> (line.last.toDouble, line.init.map(_.toDouble)))

		val seed = 1
		val ni = 196
		val nFolds = 5
		val eva = new Array[Double](nFolds)
		val splits = MLUtils.kFold(data, nFolds, seed)
		val smlr = new SharingSMLR(parameter)

		val start = System.currentTimeMillis();   	//start time
		splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
			val train = training
			val test = validation

			smlr.fit(train)
			val preds = smlr.predict(test.map(_._2)).toArray
			val trues = test.map(lp => lp._1.toInt).collect()

			val labelAndPreds = preds.zip(trues)
			val correctRate = labelAndPreds.count(res => res._1 == res._2).toDouble / labelAndPreds.length
			println("correct rate_" + splitIndex + ": " + correctRate)
			eva(splitIndex) = correctRate
		}
		println("average correct rate: " + (eva.sum / eva.length.toDouble))
		val end = System.currentTimeMillis(); 		//end time
		System.err.println("running time: "+(end-start)/(1000.0 * 60)+" min = " + ((end-start)/1000.0) + " s")
	}

	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("Lasso").setMaster("local[*]")
		val sc = new SparkContext(conf)
		sc.setCheckpointDir("src/tempCheckPointDir/")
		sc.setLogLevel("ERROR")

//		val data = sc.textFile("src/test/resource/COIL20.csv")
//		  .map(line => line.split(","))
//		  .map(line=> (line.last.toDouble, line.init.map(_.toDouble)))

		val parameter: Map[String, String] = Map("maxItertimes" -> "50",
			"conThreshold" -> "0.01",
			//"lambda" -> "0.0001",
			"lambda" -> "0.01",
			"rho" -> "0.01",
			"alpha" -> "0.01",
			"kind" -> "sharing",
			"numPartition" -> "6",
			"minIterForCheck" -> "5",
			"minIterVerConverge" -> "5")

//		val smlr = new SharingSMLR(parameter)
//		val w = smlr.fit(data)
//		//csvwrite(new File("src/source/m.csv"),w)
//		val pre = smlr.predict(data.map(_._2))
//		val true_pre = data.map(_._1).collect().zip(pre.toArray)
//		def istrue(item:(Double, Int)):Int=if (item._1==item._2) 1 else 0
//		//println(pre)
//		print("Train_Set_Accuracy: ")
//		println(true_pre.map(istrue).sum.toDouble / true_pre.length)

		nFoldFit(sc,"src/test/resource/COIL20.csv",parameter)
	}
}
