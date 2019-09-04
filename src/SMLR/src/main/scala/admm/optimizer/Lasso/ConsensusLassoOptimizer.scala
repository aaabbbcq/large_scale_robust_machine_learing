package admm.optimizer.Lasso

import admm.core.{AdmmOptimizer, AdmmState}
import breeze.linalg.{inv, DenseMatrix => BDM}
import org.apache.spark.rdd.RDD

import math.max

/**
  * Created by Evan on 2018/8/3.
  */
private[admm] class ConsensusLassoOptimizer(paraMap: Map[String, String]) extends AdmmOptimizer(paraMap) {
	private val lassoParameter: Map[String, String] = Map(
		"lambda" -> "0.5"
		) ++ paraMap

	kind = "consensus"

	private val lamb: Double = lassoParameter("lambda").toDouble
	private val k: Double = lamb / (numPartition * rho)

	private def shrink(a: Double): Double = max(0, a - k) - max(0, -1 * (a + k))

	def initialize(data: RDD[(Double, Array[Double])]): RDD[AdmmState] = {
		val n_features = data.first()._2.length
		val states:RDD[AdmmState] = AdmmOptimizer.splitDataForConsensus(data, numPartition).map(d => ConsensusLassoState(d._2.toArray, rho, n_features))
		this.gZ = BDM.zeros[Double](n_features, 1)
		states
	}

	def runRound(states: RDD[AdmmState]): RDD[AdmmState] = {
		var admmStates = states
		admmStates = states.map(_.xUpdate(this.gZ)).cache()
		val x_bar: BDM[Double] = admmStates.map(_.x).collect().reduce((m1, m2) => m1 + m2) / numPartition.toDouble
		val u_bar: BDM[Double] = admmStates.map(_.u).collect().reduce((m1, m2) => m1 + m2) / numPartition.toDouble
		this.gZ = (x_bar + u_bar).map(shrink)
		admmStates.map(_.uUpdate(this.gZ))
	}

	override def getCurrentWeights(states: RDD[AdmmState]): BDM[Double] = this.gZ
}


private[admm] class ConsensusLassoState(initX: BDM[Double], initU: BDM[Double], val inver: BDM[Double], val AtDotb: BDM[Double], val rho: Double)
  extends AdmmState(initX, initU) {

	override def xUpdate(para: BDM[Double]*): AdmmState = {
		val newZ = para(0); this.x = inver * (AtDotb + rho * (newZ - this.u)); this
	}

	override def uUpdate(para: BDM[Double]*): AdmmState = {
		val newZ = para(0); this.u += this.x - newZ; this
	}
}

private[admm] object ConsensusLassoState {
	def apply(data: Array[(Double, Array[Double])], rho: Double, n_features: Int): ConsensusLassoState = {
		val initX = BDM.zeros[Double](n_features, 1)
		val initU = BDM.zeros[Double](n_features, 1)
		val A = BDM(data.map(_._2): _*)
		val I = BDM.eye[Double](n_features)
		val b = BDM(Array(data.map(_._1)): _*).t
		val AtDotb = A.t * b
		val inver = inv(A.t * A + rho * I)
		new ConsensusLassoState(initX = initX, initU = initU, inver = inver, AtDotb = AtDotb, rho = rho)
	}
}


