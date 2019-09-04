package admm.core

import org.apache.spark.rdd.RDD
import breeze.linalg.{norm, DenseMatrix => BDM}
import math.{pow,ceil}

/**
  * Created by Evan on 2018/8/21.
  */
abstract class AdmmOptimizer(paraMap: Map[String, String]) extends Serializable {
	private val admmParameter: Map[String, String] = Map("rho" -> "0.2",
		"numPartition" -> "2",
		"conThreshold" -> "0.01",
		"maxItertimes" -> "100",
		"minIterForCheck" -> "5",
		"minIterVerConverge" -> "5") ++ paraMap

	var kind: String = "serial"
	var gZ: BDM[Double] = _
	var gU: BDM[Double] = _
	var gAiXi: BDM[Double] = _
	var targetMatrix: BDM[Double] = _

	val rho: Double = admmParameter("rho").toDouble
	val threshold: Double = admmParameter("conThreshold").toDouble
	val numPartition: Int = admmParameter("numPartition").toInt

	def runRound(states: RDD[AdmmState]): RDD[AdmmState]

	def initialize(data: RDD[(Double, Array[Double])]): RDD[AdmmState]

	def isConverged(newStates: RDD[AdmmState], preX: Array[BDM[Double]]): Boolean = {
		if (kind == "sharing") {
			val l = newStates.count()
			val primalResidual: Double = norm((this.gAiXi - this.gZ).toDenseVector, 2) * l
			val dualResidual: Double = newStates.map(_.x).collect().zip(preX).map(m => norm((m._1 - m._2).toDenseVector, 2)).sum
//			println(getCurrentWeights(newStates))
//			println("------------------------------------------")
			println("primalResidual: ", primalResidual, "dualResidual: ", dualResidual)
			if (primalResidual <= threshold && dualResidual <= threshold) true else false
		}else{
			val x_bar: BDM[Double] = newStates.map(_.x).collect().reduce((m1, m2) => m1 + m2) / numPartition.toDouble
			var old_x_bar: BDM[Double] = preX.reduce((m1, m2) => m1 + m2) / numPartition.toDouble
			val primalResidual: Double = newStates.map(s => norm((s.x - x_bar).toDenseVector, 2)).collect().sum
			val dualResidual: Double = numPartition * pow(rho, 2) * norm((x_bar - old_x_bar).toDenseVector, 2)
			println("primalResidual: ", primalResidual, "dualResidual: ", dualResidual)
			if (primalResidual <= threshold && dualResidual <= threshold) true else false
		}
	}

	def getCurrentWeights(states: RDD[AdmmState]): BDM[Double] = {
		if (kind == "sharing") {
			states.map(_.x).collect().reduce((m1, m2) => BDM.vertcat(m1, m2))
		} else {
			val localMatrixs: Array[BDM[Double]] = states.map(_.x).collect()
			localMatrixs.reduce((m1, m2) => m1 + m2) / localMatrixs.length.toDouble
		}
	}

	def optimize(data: RDD[(Double, Array[Double])]): BDM[Double] = {
		println(this.admmParameter)
		val maxItertimes = admmParameter("maxItertimes").toInt
		val minIterForCheck = admmParameter("minIterForCheck").toInt
		val minIterVerConverge = admmParameter("minIterVerConverge").toInt

		var iterCount = 0
		var isEnd = false
		var admmStates = initialize(data)
		var preX = admmStates.map(_.x).collect()
		val start = System.currentTimeMillis()
		while (iterCount < maxItertimes && !isEnd) {
			admmStates = runRound(admmStates)
			//println("------------------------------------------")
			//println(getCurrentWeights(admmStates))
			if (iterCount % minIterForCheck == 0) {
				admmStates.checkpoint()
			}
			if (iterCount % minIterVerConverge == 0) {
				println("iterCount:" + iterCount.toString + " ")
				isEnd = isConverged(admmStates, preX)
				preX = admmStates.map(_.x).collect()
			}
			iterCount += 1
		}
		val end = System.currentTimeMillis()
		System.err.println("running time: " + (end - start) / (1000.0 * 60) + " min = " + ((end - start) / 1000.0) + " s")

		getCurrentWeights(admmStates)
	}
}

object AdmmOptimizer {
	def splitDataForConsensus(data: RDD[(Double, Array[Double])], numPartition: Int): RDD[(Int, Iterable[(Double, Array[Double])])] = {
		data.groupBy((point: (Double, Array[Double])) => Math.abs(point.hashCode()) % numPartition, numPartition)
	}

	def experiment_split(data: RDD[(Double, Array[Double])], numPartition: Int): RDD[(Int, Iterable[(Double, Array[Double])])] = {
		val data1 = data.map(line => (line._1, line._2.last, line._2.init)
		).groupBy((point: (Double, Double, Array[Double])) => point._1.toInt, 5
		).map(p => (p._1, p._2.map(x => (x._2, x._3))))
		print("p:")
		println(data1.getNumPartitions)
		data1
	}

//	def splitDataForSharing(data: RDD[(Double, Array[Double])], numPartition: Int): RDD[Iterable[(Int, BDM[Double])]] = {
//		var (count, partition) = (0, 0)
//		var increment: Double = data.count() / numPartition.toDouble
//		var stage = increment
//
//		val d1: RDD[(Int, Iterable[(Double, Array[(Double)])])] = data.groupBy((point: (Double, Array[Double])) => {
//			if (count < stage) {
//				count += 1
//				partition
//			} else {
//				count += 1
//				stage += increment
//				partition += 1
//				partition
//			}
//		}, numPartition).sortBy(_._1)
//		println(d1.collect().length)
//		val d2: RDD[BDM[Double]] = d1.map(d => BDM(d._2.toArray.map(_._2): _*).t)
//
//		increment = data.first()._2.length / numPartition.toDouble
//		stage = increment
//		val d3: RDD[Array[(Int, BDM[Double])]] = d2.map { matrix =>
//			var start = 0
//			var block: List[BDM[Double]] = List()
//			for (r <- 0 until matrix.rows) {
//				if ((r + 1) >= stage) {
//					block = matrix(start to r, ::) :: block
//					stage += increment
//					start = r + 1
//				}
//			}
//			(numPartition until 0 by -1).zip(block).toArray
//		}
//		val d4 = d3.flatMap(x => x).groupBy((block: (Int, BDM[Double])) => block._1, numPartition).sortBy(_._1).map(_._2)
//		println(d4.collect().length)
//		for (d <- d4.map(_.toArray).collect()){
//			for (data <- d){
//				println(data)
//			}
//		}
//		d4
//	}

	def splitDataForSharing(data: RDD[(Double, Array[Double])], numPartition: Int): RDD[Iterable[(Int, BDM[Double])]] = {
		var increment: Double = ceil(data.count() / numPartition.toDouble)
		val d1: RDD[(Int, Iterable[(Double, Array[(Double)])])] = data.zipWithIndex().groupBy((point: ((Double, Array[Double]),Long)) =>{
			(point._2 / increment).toInt
		},numPartition).map(x => (x._1,x._2.map(_._1))).sortBy(_._1)

		val d2: RDD[BDM[Double]] = d1.map(d => BDM(d._2.toArray.map(_._2): _*).t)

		increment = ceil(data.first()._2.length / numPartition.toDouble)
		val d3: RDD[Array[(Int, BDM[Double])]] = d2.map { matrix =>
			var start: Int = 0
			var block: List[BDM[Double]] = List()
			for (multiple <- (1 until numPartition)) {
				block = matrix(start to (multiple * increment - 1).toInt, ::) :: block
				start = (multiple * increment - 1).toInt + 1
			}
			block = matrix(start until matrix.rows, ::) :: block
			(numPartition until 0 by -1).zip(block).toArray
		}

		d3.flatMap(x => x).groupBy((block: (Int, BDM[Double])) => block._1, numPartition).sortBy(_._1).map(_._2)
	}

	def splitDataForSerial(data: RDD[(Double, Array[Double])]): RDD[(Int, Iterable[(Double, Array[Double])])] = {
		data.groupBy((point: (Double, Array[Double])) => Math.abs(point.hashCode()) % 1, 1)
	}
}
