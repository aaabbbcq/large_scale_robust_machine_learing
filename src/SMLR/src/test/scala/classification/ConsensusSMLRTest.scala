package classification

import java.io.File

import admm.model.classification.{ConsensusSMLR, SharingSMLR}
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.csvwrite
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics

/**
  * Created by Evan on 2018/8/13.
  */
object ConsensusSMLRTest {

	def trainTestFit(sc: SparkContext, parameter: Map[String, String]): Unit ={
		println("train test split fit...")
		val train = sc.textFile("src/test/resource/COIL20_train.csv")
		  .map(line => line.split(","))
		  .map(line=> (line.last.toDouble, line.init.map(_.toDouble)))
		val test = sc.textFile("src/test/resource/COIL20_test.csv")
		  .map(line => line.split(","))
		  .map(line=> (line.last.toDouble, line.init.map(_.toDouble)))

		val start = System.currentTimeMillis();   	//start time
		val smlr = new ConsensusSMLR(parameter)

		smlr.fit(train)

		val preds = smlr.predict(test.map(_._2)).toArray
		val trues = test.map(lp => lp._1.toInt).collect()

		val labelAndPreds = preds.zip(trues)
		val correctRate = labelAndPreds.count(res => res._1 == res._2).toDouble / labelAndPreds.length
		val errorRate = labelAndPreds.count(res => res._1 != res._2).toDouble / labelAndPreds.length

		println("correct rate: " + correctRate)
		println("error rate: " + errorRate)

		val res = sc.parallelize(labelAndPreds,2).map(term=>(term._1.toDouble,term._2.toDouble))
		val metrics = new MulticlassMetrics(res)
		val labels = metrics.labels
		var macro_avg = 0.0
		labels.foreach(l=>macro_avg += metrics.precision(l))
		println("macro_avg: "+ macro_avg/labels.length)

		val end = System.currentTimeMillis(); 		//end time
		System.err.println("running time: "+(end-start)/(1000.0 * 60)+" min = " + ((end-start)/1000.0) + " s")
	}

	def nFoldFit(sc: SparkContext, dataPath: String, parameter: Map[String, String]) = {
		println("k-fold fit...")
		val data = sc.textFile(dataPath)
		  .map(line => line.split(","))
		  .map(line => (line.last.toDouble, line.init.map(_.toDouble)))

		val seed = 1
		val ni = 196
		val nFolds = 5
		val eva = new Array[Double](nFolds)
		val splits = MLUtils.kFold(data, nFolds, seed)
		val smlr = new ConsensusSMLR(parameter)

		val start = System.currentTimeMillis();   	//start time
		splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
			val train = training
			val test = validation

			smlr.fit(train)
			val preds = smlr.predict(test.map(_._2)).toArray
			val trues = test.map(lp => lp._1.toInt).collect()

			val labelAndPreds = preds.zip(trues)
//			val correctRate = labelAndPreds.count(res => res._1 == res._2).toDouble / labelAndPreds.length
//			println("correct rate_" + splitIndex + ": " + correctRate)

			val res = sc.parallelize(labelAndPreds,2).map(term=>(term._1.toDouble,term._2.toDouble))
			val metrics = new MulticlassMetrics(res)
			val labels = metrics.labels
			var macro_avg = 0.0
			labels.foreach(l=>macro_avg += metrics.precision(l))
			println("macro avg: "+ macro_avg/labels.length)
			println("micro avg: "+ metrics.accuracy)

			eva(splitIndex) = metrics.accuracy
		}
		println("average correct rate: " + (eva.sum / eva.length.toDouble))
		val end = System.currentTimeMillis(); 		//end time
		System.err.println("running time: "+(end-start)/(1000.0 * 60)+" min = " + ((end-start)/1000.0) + " s")
	}

	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("Lasso").setMaster("local[6]")
		val sc = new SparkContext(conf)
		sc.setCheckpointDir("src/tempCheckPointDir/")
		sc.setLogLevel("ERROR")

		val parameter: Map[String, String] = Map("maxItertimes" -> "50",
			"conThreshold" -> "0.01",
			"lambda" -> "0.0001",
//			"rho" -> "0.0001",
//			"alpha" -> "0.0001",
			"rho" -> "0.001",
			"alpha" -> "0.001",
			"kind" -> "Consensus",
			"numPartition" -> "6",
			"minIterForCheck" -> "5",
			"minIterVerConverge" -> "5")

//		val data = sc.textFile("src/test/resource/COIL20_gauss_0_1.csv")
//		  .map(line => line.split(","))
//		  .map(line=> (line.last.toDouble, line.init.map(_.toDouble)))

//		val smlr = new ConsensusSMLR(parameter)
//		val w = smlr.fit(data)
//		//println(w)
//		//csvwrite(new File("src/source/m_csmlr.csv"),w)
//		val pre = smlr.predict(data.map(_._2)).toArray
//		val true_pre = data.map(_._1).collect().zip(pre)
//		def istrue(item:(Double, Int)):Int=if (item._1==item._2) 1 else 0
//		print("Train_Set_Accuracy: ")
//		println(true_pre.map(istrue).sum.toDouble / true_pre.length)
		//val path = "/Users/ch_cmpter/Desktop/实验数据集/COIL20/分块污染数据集/block_noise_COIL20_2.csv"
		//val path = "/Users/ch_cmpter/Desktop/data4spark/shuffled_MNIST.csv"
		val path ="src/test/resource/COIL20.csv"
		nFoldFit(sc,path,parameter)
		//trainTestFit(sc,parameter)
	}
}

/**
  *所有样本加入噪声：
  *  /Users/ch_cmpter/Desktop/实验数据集/所有样本加入噪声/COIL20_gauss_0_0.1.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/所有样本加入噪声/COIL20_gauss_0_0.5.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/所有样本加入噪声/COIL20_gauss_0_1.csv
  */

/**
  *按样本加入噪声：
  *  30%：
  *  /Users/ch_cmpter/Desktop/实验数据集/按样本添加噪声数据集/30%添加噪声/COIL20_gauss_0_0.1.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按样本添加噪声数据集/30%添加噪声/COIL20_gauss_0_0.5.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按样本添加噪声数据集/30%添加噪声/COIL20_gauss_0_1.csv
  */

/**
  *按特征加入噪声：
  *  60%：
  *  /Users/ch_cmpter/Desktop/实验数据集/按样本添加噪声数据集/60%添加噪声/COIL20_gauss_0_0.1.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按样本添加噪声数据集/60%添加噪声/COIL20_gauss_0_0.5.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按样本添加噪声数据集/60%添加噪声/COIL20_gauss_0_1.csv
  */

/**
  *按特征加入噪声：
  *  30%：
  *  /Users/ch_cmpter/Desktop/实验数据集/按特征添加噪声数据集/30%添加噪声/COIL20_gauss_0_0.1.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按特征添加噪声数据集/30%添加噪声/COIL20_gauss_0_0.5.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按特征添加噪声数据集/30%添加噪声/COIL20_gauss_0_1.csv
  */

/**
  *按特征加入噪声：
  *  60%：
  *  /Users/ch_cmpter/Desktop/实验数据集/按特征添加噪声数据集/60%添加噪声/COIL20_gauss_0_0.1.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按特征添加噪声数据集/60%添加噪声/COIL20_gauss_0_0.5.csv
  *  /Users/ch_cmpter/Desktop/实验数据集/按特征添加噪声数据集/60%添加噪声/COIL20_gauss_0_1.csv
  */
